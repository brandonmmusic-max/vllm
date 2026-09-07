# Implementation-draft validation, September 7, 2026

Target: Jovian `9a6b4fb3a6f5598fd2fb68cf0de92bfe145294c1` plus this PR.
No GPUs, production services or clocks were used or changed.

## Completed

| Check | Outcome | Boundary |
| --- | --- | --- |
| CPU manifest and actual Jovian method tests | 29 passed | CUDA launches replaced by recorder in method tests; manifest fixtures use tiny headers |
| Real checkpoint inventory | 168/168 passed | Actual file sizes, K4/K5 header metadata and source-design allowlist; no new full payload hash pass |
| P8 source provenance | 65 file hashes verified | Reviewable source dependency closure, including dynamically discovered tuning profile |
| Imported-source equivalence | 65/65 passed | Compared to original sources after only namespace/sibling-import and EOF normalization; no numerical or scheduling edits |
| Native P8 and Jovian adapter imports | Passed | CPU-only RC5 dependency container, mounted current Jovian Python source; no current extension binaries |
| CUTLASS DSL 4.6.2 imports | Passed | Explicit isolated installation; both distribution version and imported module location asserted |
| Runtime Python wheel | Built successfully | `trellismx_runtime-0.1.0.dev1-py3-none-any.whl`; not GPU compilation |
| Ruff, shell syntax, Compose schema | Passed | Integration source and serving scripts; imported kernel source preserved rather than reformatted |
| Applicable repository pre-commit checks | Passed | Includes Ruff, mypy, typos, markdown, shellcheck, SPDX and forbidden-import checks; unrelated hooks skipped by file selection |

Reproduce CPU loader and source checks from the PR checkout:

```bash
bash examples/trellismx/check_cpu.sh
bash -n examples/trellismx/{build,serve,check_cpu}.sh
MODEL_ROOT=/model P8_CHECKPOINT_ROOT=/checkpoint \
  docker compose -f examples/trellismx/compose.yaml config --quiet
uv build --wheel --out-dir /tmp/trellismx-runtime-wheels third_party/trellismx
```

The CPU container does not execute its legacy `sitecustomize` bootstrap:
Python starts with `-S` and explicit dependency directories. It emits expected
warnings about missing current-source `vllm._version` / `vllm._C`, inactive
Triton driver and `triton_kernels.matmul_ogs`. These are not suppressed or
represented as a successfully built Jovian engine. Torch is 2.13.0.

The first carrier-weight fixture failed because its CPU process had no TP
group. The corrected fixture substitutes rank/world-size accessors only,
then executes the real inherited ModelOpt allocation implementation. A
mechanical namespace pass initially duplicated a torch-op namespace prefix;
that was fixed and has an explicit operator-registration regression test.

## Not tested — promotion blockers

- Full new-Jovian Docker/native-extension build.
- Compatibility of latest B12X attention/PCIe with this new assembled image.
- P8 CUDA compilation, all-rate device closure, CUDA graph replay and
  five-run determinism on this port.
- Full-model output, all-42-layer serving receipts, MTP acceptance and KLD.
- New-Jovian throughput and the default one-million-token launch/capacity.

Historical campaign KLD and RC5 benchmarks in README are not substituted for
any missing test above. This draft is code for review, not a production image.
