#!/usr/bin/env bash
# Train ~10M-parameter char-level GPT on Tiny Shakespeare.
# On a machine with CUDA: cmake -DTIRAMISU_ENABLE_CUDA=ON .. && build, then rent a GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ROOT}/build"
BIN="${BUILD}/examples/train_shakespeare"

cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" --target train_shakespeare -j

"$BIN" \
  --preset 10m \
  --data "${ROOT}/data/tiny_shakespeare.txt" \
  --epochs 5 \
  --batch-size 16 \
  --seq-len 256 \
  --checkpoint "${BUILD}/gpt_10m.ckpt"

echo "Done. Checkpoint: ${BUILD}/gpt_10m.ckpt"
