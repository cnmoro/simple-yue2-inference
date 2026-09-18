#!/usr/bin/env bash
# Create the conda env and install YuE2 + webUI dependencies.
#
# Usage:
#   bash setup_env.sh
#
# Overrides:
#   ENV_NAME=yue2        conda env name
#   TORCH_INDEX=cu128    PyTorch wheel index tag
set -euo pipefail

ENV_NAME="${ENV_NAME:-yue2}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_INDEX="${TORCH_INDEX:-cu128}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">> Creating conda env '$ENV_NAME' (python $PYTHON_VERSION)"
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
else
  echo "   env already exists, reusing"
fi

run() { conda run --no-capture-output -n "$ENV_NAME" "$@"; }

echo ">> Installing PyTorch (index: $TORCH_INDEX)"
run pip install --upgrade pip
run pip install torch==2.10.0 --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"

echo ">> Downloading official YuE2 inference wheel"
WHEEL="$ROOT/yue2_infer-0.1.5-py3-none-any.whl"
if [ ! -f "$WHEEL" ]; then
  run pip install "huggingface-hub==0.36.2"
  run huggingface-cli download m-a-p/YuE2-3B yue2_infer-0.1.5-py3-none-any.whl \
      --local-dir "$ROOT"
fi

echo ">> Installing YuE2 inference package"
run pip install "$WHEEL"

echo ">> Installing webUI dependencies"
run pip install -r "$ROOT/requirements.txt"

echo ">> Verifying environment"
run python "$ROOT/doctor.py"

echo
echo "Done. Activate with:  conda activate $ENV_NAME"
echo "Then run:            python app.py"
