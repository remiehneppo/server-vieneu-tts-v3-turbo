#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export VIENEU_MODEL_DIR="${VIENEU_MODEL_DIR:-./models}"
export VIENEU_DEFAULT_VOICE="${VIENEU_DEFAULT_VOICE:-Thục Đoan (Nữ - Miền Nam)}"
export VIENEU_DEVICE="${VIENEU_DEVICE:-cpu}"
export VIENEU_N_GPU_LAYERS="${VIENEU_N_GPU_LAYERS:-0}"

python3 -m uvicorn app:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8004}"
