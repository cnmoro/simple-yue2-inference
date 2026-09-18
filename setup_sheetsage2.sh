#!/usr/bin/env bash
# Separate environment for SheetSage2 (audio -> melody ABC), used by covers.
#
# SheetSage2 pins torch 2.8 / transformers 4.45 / numpy 1.24, which conflict with
# the YuE2 generation env. Keep them apart and exchange ABC/audio files.
set -euo pipefail

ENV_NAME="${ENV_NAME:-yue2-sheetsage2}"
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

echo ">> Installing torch/torchaudio 2.8.0 (index: $TORCH_INDEX)"
run pip install --upgrade pip
run pip install torch==2.8.0 torchaudio==2.8.0 --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"

echo ">> Downloading SheetSage2 and its MERT-v2-FullSong encoder"
run pip install "huggingface-hub==0.36.0"
run huggingface-cli download m-a-p/SheetSage2 --local-dir "$ROOT/models/SheetSage2"
run huggingface-cli download m-a-p/MERT-v2-FullSong --local-dir "$ROOT/models/MERT-v2-FullSong"

echo ">> Installing SheetSage2 requirements"
run pip install -r "$ROOT/models/SheetSage2/requirements.txt"

echo
echo "Done. Covers are enabled; start the app with the YuE2 env as usual."
