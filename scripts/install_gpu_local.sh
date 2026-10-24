#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"
VENV_DIR="${VENV_DIR:-.venv-gpu}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing Python: $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -x "$CUDACXX" ]; then
  echo "Missing CUDA compiler: $CUDACXX" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel cmake ninja
"$VENV_DIR/bin/pip" install fastapi "uvicorn[standard]" pydantic psutil numpy requests vieneu

CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 CUDACXX="$CUDACXX" CUDA_HOME="$CUDA_HOME" \
  "$VENV_DIR/bin/pip" install --no-cache-dir --force-reinstall --no-binary llama-cpp-python llama-cpp-python
