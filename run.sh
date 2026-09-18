#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-yue2}"
exec conda run --no-capture-output -n "$ENV_NAME" python "$ROOT/app.py"
