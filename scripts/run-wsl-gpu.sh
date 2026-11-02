#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv-wsl-gpu/bin/python"
if [[ ! -x "$PYTHON" ]]; then echo "Run scripts/setup-wsl-gpu.sh first." >&2; exit 1; fi
export PATH="$ROOT/.venv-wsl-gpu/bin:/opt/rocm/bin:/snap/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1
# Device selection belongs to the Python command; this environment supports CPU and GPU.
exec "$PYTHON" "$@"
