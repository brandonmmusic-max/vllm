# Implementation-draft validation, September 7, 2026

Target: Jovian `9a6b4fb3a6f5598fd2fb68cf0de92bfe145294c1` plus this PR.
No GPUs, production services or clocks were used or changed.

## Completed

| Check | Outcome | Boundary |
| --- | --- | --- |
| CPU manifest and actual Jovian method tests | 29 passed | CUDA launches replaced by recorder in method tests; manifest fixtures use tiny headers |
| Real checkpoint inventory | 168/168 passed | Actual file sizes, K4/K5 header metadata and source-design allowlist; no new full payload hash pass |
| Companion B12X P8 and Jovian imports | Passed after dependency split | CPU-only dependency container, current source at pinned B12X fork revision; no current vLLM native extension build |
| B12X rate/law, scratch and ownership contracts | 14 passed | CPU constructors and geometry only, not decoder/MMA device closure |

The 29 loader tests were rerun after switching to the companion B12X fork:
29 passed with 15 warnings (missing source-tree version metadata and Torch
deprecations). Earlier 65-file hash checks and isolated runtime wheel builds
applied to the superseded bundled draft, not this dependency-split version.
The earlier real-checkpoint inventory check remains header-only evidence.

Reproduce CPU loader and source checks from the PR checkout:

```bash
B12X_SOURCE=/absolute/path/to/companion-b12x bash examples/trellismx/check_cpu.sh
bash -n examples/trellismx/{build,serve,check_cpu}.sh
MODEL_ROOT=/model P8_CHECKPOINT_ROOT=/checkpoint \
  docker compose -f examples/trellismx/compose.yaml config --quiet
```

The CPU container does not execute its legacy `sitecustomize` bootstrap:
Python starts with `-S` and explicit dependency directories. It emits expected
warnings about missing current-source `vllm._version` / `vllm._C`, inactive
Triton driver and `triton_kernels.matmul_ogs`. These are not suppressed or
represented as a successfully built Jovian engine. Torch is 2.13.0.

The first carrier-weight fixture failed because its CPU process had no TP
group. The corrected fixture substitutes rank/world-size accessors only,
then executes the real inherited ModelOpt allocation implementation. A
mechanical namespace pass initially duplicated a torch-op namespace prefix.
The split eliminates the isolated namespace; the regression test now checks
that P8 uses the existing `torch.ops.b12x` registrations.

## Not tested — promotion blockers

- Full new-Jovian Docker/native-extension build.
- Compatibility of latest B12X attention/PCIe with this new assembled image.
- P8 CUDA compilation, all-rate device closure, CUDA graph replay and
  five-run determinism on this port.
- Full-model output, all-42-layer serving receipts, MTP acceptance and KLD.
- New-Jovian throughput and the default one-million-token launch/capacity.

Historical campaign KLD and RC5 benchmarks in README are not substituted for
any missing test above. This draft is code for review, not a production image.
