#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  for p in python3.12 python3.11 python3.10 python3; do
    if command -v "$p" >/dev/null 2>&1; then
      PYTHON_BIN="$p"
      break
    fi
  done
fi

if [ -z "$PYTHON_BIN" ] || ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing Python 3.10+ interpreter." >&2
  exit 1
fi

VENV_DIR="${VENV_DIR:-.venv}"

IS_ROCM=false
IS_CUDA=false

if command -v rocm-smi >/dev/null 2>&1 || [ -e /dev/kfd ] || [ -d /opt/rocm ]; then
  IS_ROCM=true
  echo "Detected AMD ROCm GPU."
elif command -v nvidia-smi >/dev/null 2>&1 || [ -d /usr/local/cuda ]; then
  IS_CUDA=true
  echo "Detected NVIDIA CUDA GPU."
else
  echo "Warning: No GPU detected, will install PyTorch default."
fi

if command -v uv >/dev/null 2>&1; then
  echo "Using uv for fast installation..."
  uv venv "$VENV_DIR" --python "$PYTHON_BIN"
  PIP_CMD=(uv pip install --python "$VENV_DIR/bin/python")
else
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
  PIP_CMD=("$VENV_DIR/bin/pip" install)
fi

if [ "$IS_ROCM" = true ]; then
  echo "Installing PyTorch for ROCm 6.0..."
  "${PIP_CMD[@]}" torch==2.4.1+rocm6.0 torchaudio==2.4.1+rocm6.0 --index-url https://download.pytorch.org/whl/rocm6.0
elif [ "$IS_CUDA" = true ]; then
  echo "Installing PyTorch for CUDA..."
  "${PIP_CMD[@]}" torch torchaudio --index-url https://download.pytorch.org/whl/cu124
fi

echo "Installing server requirements..."
"${PIP_CMD[@]}" -r requirements.txt

echo "Setup completed successfully in $VENV_DIR."
echo "Run ./scripts/start_gpu_local.sh to start the TTS server."
