#!/usr/bin/env bash
# CPU-only source/loader tests using pinned dependencies; no GPU devices exposed.
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
image=verdictai/trellismx@sha256:609a5fc1cd7d994ba32d9c03626c414d315947eb9f13fab474a15bc8dfbe0129
docker run --rm --network none --entrypoint /opt/venv/bin/python \
  -e PYTHONPATH=/jj:/review \
  -v "$root:/jj:ro" -v "$root/third_party/trellismx:/review:ro" -w /jj \
  "$image" -S -c '
import site
for path in ["/usr/local/lib/python3.12/dist-packages", "/usr/lib/python3/dist-packages", "/opt/venv/lib/python3.12/site-packages"]:
    site.addsitedir(path)
import runpy
runpy.run_path("examples/trellismx/verify_runtime.py")
import pytest
raise SystemExit(pytest.main([
    "--confcutdir=tests/quantization", "-p", "no:cacheprovider",
    "tests/quantization/test_trellismx_manifest.py",
    "tests/quantization/test_trellismx_method.py", "-q",
]))'
