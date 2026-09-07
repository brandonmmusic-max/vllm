# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the reviewed runtime source bundle without loading CUDA."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2] / "third_party/trellismx"
manifest = json.loads((root / "provenance.json").read_text())
for record in manifest["files"]:
    path = root / record["path"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != record["sha256"]:
        raise SystemExit(f"Runtime source mismatch: {record['path']}")
print(f"Verified {len(manifest['files'])} reviewed runtime source hashes")
