#!/usr/bin/env bash
set -euo pipefail

CUDA_TEST_IMAGE="${CUDA_TEST_IMAGE:-nvidia/cuda:12.4.1-base-ubuntu22.04}"

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

warn() {
  printf 'WARN: %s\n' "$*" >&2
}

pass() {
  printf 'PASS: %s\n' "$*"
}

command -v docker >/dev/null 2>&1 || fail "docker is not installed or not in PATH"
docker info >/dev/null 2>&1 || fail "docker daemon is not reachable"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail "nvidia-smi is not available on the host; install/repair the NVIDIA driver first"
fi

host_gpu="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits | head -n 1 || true)"
if [ -z "$host_gpu" ]; then
  fail "nvidia-smi did not report a host GPU"
fi
pass "host GPU visible: $host_gpu"

if ! command -v nvidia-container-runtime >/dev/null 2>&1 && ! command -v nvidia-container-cli >/dev/null 2>&1; then
  warn "nvidia-container-runtime/nvidia-container-cli not found in PATH"
  warn "install nvidia-container-toolkit and restart Docker before using docker compose GPU mode"
fi

if docker run --rm --gpus all "$CUDA_TEST_IMAGE" nvidia-smi >/tmp/vieneu-docker-gpu-check.log 2>&1; then
  pass "Docker GPU runtime works with $CUDA_TEST_IMAGE"
else
  cat /tmp/vieneu-docker-gpu-check.log >&2 || true
  fail "Docker GPU runtime is not working; install/configure NVIDIA Container Toolkit"
fi
