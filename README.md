# tiramisu

A machine learning framework built from scratch in C++: tensors, autograd,
NN modules, optimizers, transformers, and more.

Python bindings will be added one day.

## Build

Requires CMake 3.20+ and a C++20 compiler.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Debug builds with ASan + UBSan by default (`TIRAMISU_ENABLE_SANITIZER`).
Use `DCMAKE_BUILD_TYPE=Release` for an optimized, sanitizer-free build.

## Layout

```
core/      Storage, Tensor, dtype, device
ops/cpu/   elementwise, reduction, matmul, conv kernels (CPU)
ops/cuda/  same interface
autograd/  Function/tape, backward(), gradient checker
nn/        Module, Parameter, Linear, Conv2d, Embedding, transformer block
optim/     SGD, Adam, AdamW, LR scheduling
serialize/ weight save/load, training checkpoints
quant/     fp32 -> int8 export
python/    pybind11 bindings (I'm not going to work on this for a while)
tests/     GoogleTest suite
bench/     microbenchmarks, not wired into the default build
examples/  small runnable demos (train_mnist, eventually train a tiny GPT)
```