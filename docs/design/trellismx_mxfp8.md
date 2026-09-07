# RFC: native TrellisMX-MXFP8 support on Jovian Judgement

Status: **owner-review proposal, not implemented serving support**.
Target inspected: `dev/jovian-judgement` at
`9a6b4fb3a6f5598fd2fb68cf0de92bfe145294c1` in
`local-inference-lab/vllm`. This RFC changes documentation only. It does not
register a quantization method, load a checkpoint, or qualify a new backend.

## Decision requested

Review the checkpoint/backend boundary and implementation gates below before
porting the working TrellisMX P8 runtime. Keep the generic codec contract
separate from GLM-specific model integration. Do not make an unmodified
ModelOpt or EXL3 loader silently interpret TrellisMX data.

The first target is **GLM-5.3-Flash-TrellisMX-MXFP8**, a coupled checkpoint
with 17 K5 and 25 K4 routed layers, served with native E4M3 block-scaled MMA.
The model upload is in progress at
[brandonmusic/GLM-5.3-Flash-TrellisMX-MXFP8](https://huggingface.co/brandonmusic/GLM-5.3-Flash-TrellisMX-MXFP8).
The HF repository remains private during upload verification.

## Existing evidence and source boundary

The working image is
`verdictai/trellismx@sha256:609a5fc1cd7d994ba32d9c03626c414d315947eb9f13fab474a15bc8dfbe0129`.
It contains vLLM commit `6dc2f516688fe6f84c6994dcd20fddf296853a6c` plus image
patches and B12X commit `36bce2c1552ba2d47dc09f20a6f64fbfc8ec4ff8` plus P8
kernels. Its checked runtime sources, Compose and source-rebuild receipt
are published in the
[release directory](https://github.com/brandonmmusic-max/glm53-hadamard-shapleymcg-kld/tree/03527b1f092cfbf274ae3ca78cabb23077c37057/deploy/trellismx).
Required design-identity files were added in
`3da5c57b6ac3cccc6474e8b14b5a45d48ecc7146` of that repository.

That image is **not** current-Jovian qualification. Existing RC5 patches use
`sitecustomize.py` hooks around `ModelOptNvFp4FusedMoE`,
`UnquantizedFusedMoEMethod`, `FusedMoEMethodBase` and `P8NativeTPMoE`.
The current target exposes custom quantization registration in
`vllm/model_executor/layers/quantization/__init__.py` and
`RoutedExperts`-based ModelOpt methods. Presence of similarly named classes
does not establish ABI compatibility.

### Observed results, not predictions for the port

Shared model: coupled 17-K5/25-K4, 42 routed layers, **4.6587417643 stored
routed bpw including metadata**. Runtime conditions are kept with each row.

| Measurement | Result | Conditions and limitation |
| --- | --- | --- |
| Historical true-decode KLD | 0.0341811459; BCa 95% [0.0291483518, 0.0409784257] | 32 opened CF development windows; B12X_MLA_SPARSE, NVFP4 MLA KV, native P8 E4M3; historical runtime, not RC5 or new Jovian KLD |
| Uniform coupled K4 comparison | 0.0369674524; K4/K5 is 7.54% lower | Larger bit budget, runtime image revisions differ; not an equal-size or matched-stock claim |
| Estonia | 30/30 PASS | RC5, TP4/DCP1, MTP3, B12X_MLA_SPARSE, NVFP4 MLA KV, native P8 E4M3, 4.65874 bpw; C10, reasoning max |
| LAVD | 22 exact + 6 near + 2 token-cap truncations | Same RC5 regime; official 28/30, not 30/30 exact |
| Hotel Lights | 28 exact + 2 FAIL | Same RC5 regime; neither failure hit the token cap |
| Sustained decode | C1 178.336464; C2 235.605573; C4 300.874345 aggregate tokens/s | TP4/DCP1, MTP3, B12X attention/PCIe collectives, NVFP4 KV, native P8 E4M3, 4.65874 bpw; graphs; separate cache-disabled launch |
| Client prefill | 8k 8063; 32k 7998 tokens/s | Same checkpoint/compute family; separate prefix-cache-enabled prefill run |

Reasoning tests used 30 requests/profile, max output 100,000 and zero request
errors. All 28 finished LAVD answers were credited by the scorer, including
six NEAR answers; this does not prove the two unfinished answers correct.
Speed is from final `llm_decode_bench` JSON plus matching logs, not vLLM
rolling throughput. Speed reflects one configuration run, four RTX PRO 6000
96GB GPUs, PCIe/no NVLink, 300W each and +6000 memory offset. No independent
replicate uncertainty claim, 200-token/s claim or new-Jovian speed claim is
made. The 1M launch default is not 1M retrieval qualification.

[Full results, protocol and raw receipts](https://github.com/brandonmmusic-max/glm53-hadamard-shapleymcg-kld/blob/a0c3407e79228ac06a1abf6a79484f38f38bd90a/results/RC5_RELEASE_RESULTS_20260907.md).

## Proposed generic checkpoint contract

Use an explicit, versioned `trellismx` method registration. The spelling and
final public JSON schema require maintainer review; they are not existing
CLI features. A manifest must define:

- format/schema version; procedural law and reconstruction alphabet;
- per-layer or per-projection stored rate, packed layout and logical shape;
- block-scale dtype, group size and stored metadata size;
- TP rank/slice geometry and exact sidecar content hashes;
- coupled transform/scale boundary, cast order, signs and activation contract;
- architecture adapter identity and target versus MTP ownership;
- pinned carrier dependencies for transitional overlay checkpoints.

Reject unsupported values or missing files before CUDA graph capture.
No fallback from an explicitly claimed TrellisMX tensor to a stock tensor.
Separate stored bitrate from reconstruction precision:

| Property | First supported P8 target |
| --- | --- |
| Trellis rate | K4 and K5 in this checkpoint; K3 only when independently tested |
| Reconstruction | E4M3 with UE8M0 scales per 32 elements |
| Law | `procedural-mcg-alpha2`; exact current bitstream and decode law |
| Coupled boundary | `coupled-h512-h128-suh-svh-v1`, sign draw 0, SiLU cap 10 |
| Native compute | `mxf8f6f4`, not NVFP4-rate arithmetic |
| Routed layer set | GLM layers 3–44; layer 45 MTP remains separately handled |

The generic manifest validator must not hard-code GLM's 4096 hidden width or
288 experts as format-wide constraints. The GLM adapter must validate those
values for this checkpoint. Unknown architectures require an adapter and
validation; a generic schema is not universal serving support.

## Proposed loader and execution lifecycle

1. Inspect explicit metadata and select the TrellisMX method without changing
   unrelated ModelOpt/EXL3 selection behavior.
2. Validate complete rank/layer inventories, shapes, rates and dependencies.
   Preserve the working overlay layout initially. Standalone repacking is a
   later loader change with its own weight/forward closure, not a file rename.
3. Load rank-local packed streams/scales; verify design and transform identities.
   Own/free carrier tensors at an explicit lifecycle point. Do not accidentally
   retain both complete carrier experts and their P8 replacements on device.
4. Bind decoder/kernel descriptors and workspaces once. Register warmup with
   Jovian's B12X warmup interface; isolate target/MTP and graph-shape caches.
5. Dispatch small decode/verification shapes to their measured small-M path
   and real prefill rows to grouped kernels. Do not route MTP3 verification
   through a padded M64 kernel just because the prefill kernel accepts it.
6. Apply coupled transforms to routed inputs only, with the required input/
   output channel scales and FC1/FC2 cast order. Preserve router and shared
   expert semantics. Avoid applying the same transform twice.
7. Decode trellis elements in parallel in the MMA prologue and feed native
   E4M3 operands. A BF16 weight materialization fallback does not satisfy
   the intended P8 path and must be reported or rejected explicitly.

P8 needs twice the NVFP4 MMA issue count for equal dimensions. The storage
codec can reduce bytes without promising NVFP4 arithmetic rate. P4 is a
different future backend and is excluded from this proposal's first port.

## Overlap and attribution

- [EXL3 Jovian PR 562](https://github.com/local-inference-lab/vllm/pull/562),
  head `97ba04f40bb48273762491749b3f0834ec32ad93`: compose loader/warmup ideas
  where compatible; do not duplicate it or claim it already serves P8.
- [Coupled QSRT PR 566](https://github.com/local-inference-lab/vllm/pull/566),
  head `118a8523b0945cb626439058583ec691ef6500a0`, targets **Infernal**, not
  Jovian. Review coupled W4A8 prefill semantics; do not assume merged status.
- [Wide trellis prefill PR 563](https://github.com/local-inference-lab/vllm/pull/563)
  also targets Infernal and concerns W4A16, not this P8 compute contract.

Preserve attribution to Brandon M. Music, Z.ai, Local Inference Lab, vLLM,
B12X, ExLlamaV3, KQuant, QSRT and `w4a8_trellis` as applicable. TrellisMX's
SHAPLEYMCG source-available license is not Apache-2.0. Do not paste its
restricted implementation into Apache-only source files or invent a
relicense. Agree with maintainers on an optional external runtime/plugin
boundary or an explicit licensing arrangement before vendoring code.

## Predeclared implementation gates

No new GPU tests run for this documentation proposal. Before promoting a port:

1. CPU metadata tests: malformed schema, missing/extra ranks, inconsistent
   rates, wrong TP geometry, hash mismatches, incompatible scale/transform,
   incorrect architecture, and target/MTP isolation must fail closed.
2. CPU bitstream/reference tests: all supported rates and boundary crossing
   offsets, including non-aligned tails; compare exact decoded E4M3 codes
   and scale bytes. Preserve original fixtures and contrary cases.
3. Device closure: rank/layer inventory, bit-exact decoder and five replay
   determinism checks; test small-M, MTP verification and grouped prefill.
   GEMM rounding tolerance must follow the documented stagewise contract,
   not an ad-hoc relaxed comparison.
4. Serving closure: coherent target-only and MTP3 outputs, then matched
   full-model teacher-to-student KLD on the authorized opened CF32 role.
   Leave the 28 protected confirmation logits unopened.
5. Performance: final client JSON/log sustained C1/C2/C4 and prefill with
   matched graph/cache/attention/KV conditions. Prefix reuse must not inflate
   decode. Keep cold rebuild, warm runtime and speculative acceptance separate.
6. Clean HF-download-to-serving test before calling the new integration ready.

Before each GPU test, record its comparison rule, changed variables and
failure behavior. Current RC5 receipts are historical context, not passing
results for these gates. Human review and relevant test execution are
required before upstream submission under the target contribution policy.
