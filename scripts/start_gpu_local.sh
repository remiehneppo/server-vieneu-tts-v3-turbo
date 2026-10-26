#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VENV_DIR="${VENV_DIR:-.venv}"
if [ ! -d "$VENV_DIR" ] && [ -d ".venv-gpu" ]; then
  VENV_DIR=".venv-gpu"
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Missing GPU venv ($VENV_DIR). Run ./scripts/install_gpu_local.sh first." >&2
  exit 1
fi

export VIENEU_DEVICE="${VIENEU_DEVICE:-cuda}"
export VIENEU_MODEL_ID="${VIENEU_MODEL_ID:-pnnbao-ump/VieNeu-TTS-v3-Turbo}"
export VIENEU_MODE="${VIENEU_MODE:-v3turbo}"
export VIENEU_MODEL_DIR="${VIENEU_MODEL_DIR:-./models}"
export HF_HOME="${HF_HOME:-./models}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-./models/hub}"
export VIENEU_BENCHMARK_DIR="${VIENEU_BENCHMARK_DIR:-/tmp/vieneu-gpu-benchmarks}"

mkdir -p "$VIENEU_BENCHMARK_DIR"

echo "Starting VieNeu-TTS v3 Turbo server on GPU (${VIENEU_DEVICE})..."
"$VENV_DIR/bin/python" -m uvicorn app:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8004}"
