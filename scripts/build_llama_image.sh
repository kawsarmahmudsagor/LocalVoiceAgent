#!/usr/bin/env bash
# Builds a llama.cpp server image with sm_120 (RTX 5090) kernels, inside Docker (the host nvcc is never used).
set -euo pipefail
cd "$(dirname "$0")/.."
src=tools/llama.cpp-src
if [ ! -d "$src/.git" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$src"
fi
cd "$src"
docker build -f .devops/cuda.Dockerfile --target server \
  --build-arg CUDA_VERSION=12.8.0 --build-arg CUDA_DOCKER_ARCH=120 \
  -t voicebot/llama-server:cuda12.8 .
echo "set LLAMA_IMAGE=voicebot/llama-server:cuda12.8 in .env"
