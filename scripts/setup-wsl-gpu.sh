#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$ROOT/.venv-wsl-gpu"
UV="$(command -v uv || true)"
if [[ -z "$UV" && -x "$HOME/.local/bin/uv" ]]; then UV="$HOME/.local/bin/uv"; fi
if [[ -z "$UV" ]]; then echo "Install uv before running this setup." >&2; exit 1; fi
if [[ ! -f /opt/rocm/lib/libhsa-runtime64.so.1 ]]; then
  echo "The ROCm 6.4.2 WSL runtime must be installed first." >&2; exit 1
fi
if [[ ! -x "$ENV_DIR/bin/python" ]]; then "$UV" venv --python python3.12 "$ENV_DIR"; fi
"$UV" pip install --python "$ENV_DIR/bin/python" -r "$ROOT/requirements-wsl-gpu.txt"
# AMD instructs WSL users to remove the wheel's bundled Linux HSA runtime.
# Preserve it outside torch/lib instead, so the installed WSL runtime is used.
"$ENV_DIR/bin/python" - <<'PY'
from pathlib import Path
import sysconfig
import shutil
import sys
root = Path(sys.prefix).resolve()
library = Path(sysconfig.get_paths()['purelib']) / 'torch' / 'lib'
backup = root / 'bundled-hsa-backup'
backup.mkdir(exist_ok=True)
for source in library.glob('libhsa-runtime64.so*'):
    if not source.parent.resolve().is_relative_to(root):
        raise RuntimeError('Refusing to move a library outside the GPU environment')
    target = backup / source.name
    if target.exists() or target.is_symlink():
        raise RuntimeError(f'Existing library backup needs review: {target}')
    shutil.move(str(source), str(target))
PY
"$ROOT/scripts/run-wsl-gpu.sh" -c 'import torch; print("PyTorch:", torch.__version__, "HIP:", torch.version.hip); print("GPU:", torch.cuda.get_device_name(0))'
