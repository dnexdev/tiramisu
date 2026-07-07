![tiramisu](docs/assets/banner.png)

A from-scratch ML framework in **C++20** with a PyTorch-like Python API.

## Features

- **Automatic differentiation** -- reverse-mode autograd with gradient checking on every op
- **Strided tensor engine** -- zero-copy views, NumPy interop
- **Hand-written AVX2 SIMD matmul** -- ~8x faster than naive on x86
- **Full transformer stack** -- multi-head attention, LayerNorm, GELU, GPT
- **CUDA backend** -- optional GPU kernels (`--device cuda` in Python, `-DTIRAMISU_ENABLE_CUDA=ON` in CMake)
- **PyTorch-familiar API** -- `Tensor`, `requires_grad`, `backward()`, `nn.Linear`, `optim.Adam`

[![PyPI](https://img.shields.io/pypi/v/tiramisu-ml.svg)](https://pypi.org/project/tiramisu-ml/)
[![Python](https://img.shields.io/pypi/pyversions/tiramisu-ml.svg)](https://pypi.org/project/tiramisu-ml/)
[![CMake on multiple platforms](https://github.com/dnexdev/tiramisu/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/dnexdev/tiramisu/actions/workflows/cmake-multi-platform.yml)
[![License: MIT](https://img.shields.io/pypi/l/tiramisu-ml.svg)](LICENSE)

## Installation

```bash
pip install tiramisu-ml
```

Requires Python 3.10+. Wheels ship for Linux (x86_64, aarch64), macOS (x86_64, arm64), and Windows (amd64); other platforms build from sdist and need CMake + a C++20 compiler.

## Quick Start

```python
import numpy as np
import tiramisu as tr

# Random batch of 32 samples, 10 features, 3 classes
x = tr.from_numpy(np.random.randn(32, 10).astype(np.float32))
targets = tr.from_numpy(np.random.randint(0, 3, size=(32,)).astype(np.float32))

layer1 = tr.nn.Linear(10, 64)
layer2 = tr.nn.Linear(64, 3)

# Forward pass
h = tr.relu(layer1.forward(x))
logits = layer2.forward(h)
loss = tr.nn.cross_entropy_loss(logits, targets)

# Backward pass
loss.backward()

# Optimize
params = layer1.parameters() + layer2.parameters()
optimizer = tr.optim.Adam(params, lr=1e-3)
optimizer.step()
optimizer.zero_grad()
```

## API Reference

### Tensor

```python
import numpy as np
import tiramisu as tr

# Creation
tr.Tensor([2, 3])                        # zero-filled float32
tr.from_numpy(np.zeros((2, 3), np.float32))

# Arithmetic (with autograd)
tr.add(a, b)       tr.add(a, 2.0)      # addition
tr.sub(a, b)                             # subtraction
tr.mul(a, b)       tr.mul(a, 2.0)      # multiplication
tr.div(a, b)                             # division
tr.neg(a)                                # negation

# Activations
tr.relu(a)       tr.gelu(a)            # ReLU, GELU
tr.softmax(a)                          # softmax

# Reductions
tr.sum(a)        tr.mean(a)            # sum / mean over all elements

# Layout
a.reshape([6])                           # reshape
tr.transpose(a)                        # swap last two dims
a.contiguous()                         # ensure contiguous memory

# Linear algebra
tr.matmul(a, b)                        # matrix multiplication

# Autograd
x = tr.from_numpy(np.array([2.0], np.float32))
x.requires_grad = True
y = tr.mul(x, x)
y.backward()
x.grad                                 # gradient tensor

# NumPy interop
arr = np.asarray(a)                    # zero-copy when contiguous
a.numpy()                              # same, via method
```

### Neural Network Modules

```python
import tiramisu as tr

linear = tr.nn.Linear(in_features=784, out_features=10)
layernorm = tr.nn.LayerNorm(features=128)
gpt = tr.nn.GPT(
    vocab_size=65,
    d_model=128,
    num_heads=4,
    num_layers=2,
    max_seq_len=256,
)

out = linear.forward(x)
params = gpt.parameters()
cfg = gpt.config()
```

### Functional Operations

```python
import tiramisu as tr

loss = tr.nn.cross_entropy_loss(logits, targets)
sm = tr.softmax(logits)
g = tr.gelu(x)
```

### Optimizers

```python
import tiramisu as tr

optimizer = tr.optim.Adam(parameters, lr=1e-3)
optimizer = tr.optim.AdamW(parameters, lr=3e-4, weight_decay=0.1)
optimizer = tr.optim.SGD(parameters, lr=0.01)

optimizer.step()
optimizer.zero_grad()

tr.optim.clip_grad_norm_(parameters, max_norm=1.0)

scheduler = tr.optim.CosineAnnealingLR(base_lr=3e-4, total_steps=1000)
lr = scheduler.step()
```

### Serialize

```python
import tiramisu as tr

tr.serialize.save_gpt("model.ckpt", model, step=100, epoch=1)
step, epoch = tr.serialize.load_gpt("model.ckpt", model)
```

### Device

```python
import tiramisu as tr

tr.cuda_available()                    # True if built with CUDA
x = tr.from_numpy(arr, device="cuda")  # place tensor on GPU
loss.cpu().numpy()                     # move back to CPU for NumPy
```

Token indices are stored as `float32` tensors today (the C++ embedding path reads float indices).

## Architecture

```
Python API (tiramisu)
    │
    └─→ pybind11 (_C)
            │
            ├─→ core/      Tensor, Storage, dtype, device
            ├─→ ops/       CPU kernels (AVX2); optional CUDA
            ├─→ autograd/  backward tape, gradcheck
            ├─→ nn/        Linear, GPT, LayerNorm, loss
            └─→ optim/     Adam, AdamW, SGD, schedulers
```

## Examples

Runnable scripts live in [`examples/python/`](examples/python/) and [`examples/`](examples/) (C++).

| Script | Description |
|--------|-------------|
| [`examples/python/autograd_demo.py`](examples/python/autograd_demo.py) | Minimal autograd: y = x² + 3x |
| [`examples/python/linear_forward.py`](examples/python/linear_forward.py) | Forward pass + NumPy interop |
| [`examples/python/train_mnist.py`](examples/python/train_mnist.py) | 2-layer MLP on MNIST |
| [`examples/python/gpt_step.py`](examples/python/gpt_step.py) | Single GPT training step |
| [`examples/python/train_shakespeare.py`](examples/python/train_shakespeare.py) | Char-level GPT on Tiny Shakespeare |

```bash
# Python (after pip install)
python examples/python/train_mnist.py --data-dir data --epochs 5
python examples/python/train_shakespeare.py --preset tiny --epochs 3
```

C++ examples require a local build (see below):

```bash
cmake --build build --target mnist && ./build/examples/mnist
cmake --build build --target train_shakespeare
./build/examples/train_shakespeare --preset tiny --epochs 3
```

## Building from source

Only needed for C++ development, CUDA, or contributing. Requires CMake 3.20+ and a C++20 compiler.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

CUDA: add `-DTIRAMISU_ENABLE_CUDA=ON`. Debug builds enable ASan + UBSan by default (`TIRAMISU_ENABLE_SANITIZERS=ON`).

Python editable install from the repo root:

```bash
pip install -e ".[dev]"
pytest tests/python -v
```

See [`python/README.md`](python/README.md) for CMake Python extension build notes.

## License

MIT
