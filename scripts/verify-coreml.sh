#!/usr/bin/env bash
set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
  echo "[verify-coreml] python3 not available; skipping validation." >&2
  exit 0
fi

python3 <<'PYCODE'
import importlib
import sys

print("[verify-coreml] Running baseline diagnostics...")
coremltools_spec = importlib.util.find_spec("coremltools")
if coremltools_spec is None:
    print("[verify-coreml] coremltools not installed; install before model integration.")
    sys.exit(0)

import coremltools as ct  # noqa: E402

print(f"[verify-coreml] coremltools {ct.__version__} detected.")
print("[verify-coreml] No models to validate yet; placeholder check complete.")
PYCODE
