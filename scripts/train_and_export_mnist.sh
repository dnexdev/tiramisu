#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT}/data"
BUILD_DIR="${ROOT}/build"
CHECKPOINT="${ROOT}/checkpoints/mnist_mlp.bin"

mkdir -p "${DATA_DIR}" "${ROOT}/checkpoints"

download() {
  local url="$1"
  local out="$2"
  if [[ -f "${out}" ]]; then
    return
  fi
  echo "Downloading $(basename "${out}")..."
  curl -fsSL "${url}" -o "${out}"
}

download "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte" \
  "${DATA_DIR}/train-images.idx3-ubyte"
download "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte" \
  "${DATA_DIR}/train-labels.idx1-ubyte"
download "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte" \
  "${DATA_DIR}/t10k-images.idx3-ubyte"
download "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte" \
  "${DATA_DIR}/t10k-labels.idx1-ubyte"

cmake -S "${ROOT}" -B "${BUILD_DIR}" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target mnist

"${BUILD_DIR}/examples/mnist" \
  --train-images "${DATA_DIR}/train-images.idx3-ubyte" \
  --train-labels "${DATA_DIR}/train-labels.idx1-ubyte" \
  --test-images "${DATA_DIR}/t10k-images.idx3-ubyte" \
  --test-labels "${DATA_DIR}/t10k-labels.idx1-ubyte" \
  --export "${CHECKPOINT}"

ls -lh "${CHECKPOINT}"
