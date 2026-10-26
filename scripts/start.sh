#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VENV_DIR="${VENV_DIR:-.venv}"
if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Missing venv. Run ./scripts/install.sh first." >&2
  exit 1
fi

export VIENEU_MODEL_ID="${VIENEU_MODEL_ID:-pnnbao-ump/VieNeu-TTS-v3-Turbo}"
export VIENEU_MODE="${VIENEU_MODE:-v3turbo}"
export VIENEU_MODEL_DIR="${VIENEU_MODEL_DIR:-./models}"
export VIENEU_DEFAULT_VOICE="${VIENEU_DEFAULT_VOICE:-Thục Đoan (Nữ - Miền Nam)}"

if [ -z "${VIENEU_DEVICE:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 || [ -e /dev/kfd ]; then
    export VIENEU_DEVICE="cuda"
  else
    export VIENEU_DEVICE="cpu"
  fi
fi

"$VENV_DIR/bin/python" -m uvicorn app:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8004}"
