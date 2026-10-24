#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VENV_DIR="${VENV_DIR:-.venv-gpu}"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Missing GPU venv. Run ./scripts/install_gpu_local.sh first." >&2
  exit 1
fi

export VIENEU_DEVICE="${VIENEU_DEVICE:-cuda}"
export VIENEU_N_GPU_LAYERS="${VIENEU_N_GPU_LAYERS:--1}"
export VIENEU_MODEL_DIR="${VIENEU_MODEL_DIR:-./models}"
export HF_HOME="${HF_HOME:-./models}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-./models/hub}"
export VIENEU_BENCHMARK_DIR="${VIENEU_BENCHMARK_DIR:-/tmp/vieneu-gpu-benchmarks}"

mkdir -p "$VIENEU_BENCHMARK_DIR"

"$VENV_DIR/bin/python" -m uvicorn app:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8010}"
