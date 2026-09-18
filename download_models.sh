#!/usr/bin/env bash
# Download YuE2-3B and the audio decoder into ./models (local, so the app
# never needs to resolve anything from the Hub at startup).
set -euo pipefail

ENV_NAME="${ENV_NAME:-yue2}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAE="${VAE:-YuE2-Vae}"   # or YuE2-Vae-legacy for paper benchmark reproduction

run() { conda run --no-capture-output -n "$ENV_NAME" "$@"; }

run huggingface-cli download m-a-p/YuE2-3B --local-dir "$ROOT/models/YuE2-3B"
run huggingface-cli download "m-a-p/$VAE" --local-dir "$ROOT/models/$VAE"

# Only needed for covers (audio -> melody ABC). Skip with SKIP_COVER=1.
if [ "${SKIP_COVER:-0}" != "1" ]; then
  run huggingface-cli download m-a-p/SheetSage2 --local-dir "$ROOT/models/SheetSage2"
  run huggingface-cli download m-a-p/MERT-v2-FullSong --local-dir "$ROOT/models/MERT-v2-FullSong"
fi

echo "Models ready under $ROOT/models"
