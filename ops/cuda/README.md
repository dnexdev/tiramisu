# ops/cuda

CUDA kernels for tiramisu ops. Enable with `-DTIRAMISU_ENABLE_CUDA=ON` at configure time (requires NVIDIA CUDA toolkit).

Implemented kernels (forward):
- `matmul` (2D contiguous)
- `add`, `mul`, `gelu`, `relu`
- `softmax`, `layernorm`

CPU fallback remains the default. Training examples run on CPU; CUDA is for GPU experimentation and future scaling.
