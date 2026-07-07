![tiramisu](docs/assets/banner.png)

A from-scratch ML framework in **C++20**.

## Features

- A strided tensor engine with zero-copy views
- Reverse-mode autograd with gradient checking on every op
- Hand-written AVX2 SIMD matmul (8x faster than naive)
- Full transformer stack (multi-head attention, LayerNorm, GELU)
- CUDA backend with custom kernels

[![PyPI](https://img.shields.io/pypi/v/tiramisu-ml.svg)](https://pypi.org/project/tiramisu-ml/)
[![Python](https://img.shields.io/pypi/pyversions/tiramisu-ml.svg)](https://pypi.org/project/tiramisu-ml/)
[![CMake on multiple platforms](https://github.com/dnexdev/tiramisu/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/dnexdev/tiramisu/actions/workflows/cmake-multi-platform.yml)
[![License: MIT](https://img.shields.io/pypi/l/tiramisu-ml.svg)](LICENSE)

## Install

```bash
pip install tiramisu-ml
```

Requires Python 3.10+. Wheels ship for Linux (x86_64, aarch64) and macOS (x86_64, arm64); other platforms build from sdist and need CMake + a C++20 compiler.

## Quick start

```python
import numpy as np
import tiramisu as tr

x = tr.from_numpy(np.random.randn(2, 784).astype(np.float32))
layer = tr.nn.Linear(784, 10)
print(layer.forward(x).shape())  # [2, 10]
```

## Example: train an MLP on MNIST

Grab the [MNIST IDX files](http://yann.lecun.com/exdb/mnist/) into `data/`, then:

```python
import numpy as np
import tiramisu as tr

def load_images(path):
    with open(path, "rb") as f:
        _, n, r, c = np.frombuffer(f.read(16), ">u4")
        return np.frombuffer(f.read(), np.uint8).reshape(n, r * c).astype(np.float32) / 255

def load_labels(path):
    with open(path, "rb") as f:
        _, n = np.frombuffer(f.read(8), ">u4")
        return np.frombuffer(f.read(), np.uint8).astype(np.float32)

X = load_images("data/train-images-idx3-ubyte")
y = load_labels("data/train-labels-idx1-ubyte")

fc1 = tr.nn.Linear(784, 128)
fc2 = tr.nn.Linear(128, 10)
opt = tr.optim.Adam(fc1.parameters() + fc2.parameters(), lr=1e-3)

for epoch in range(5):
    idx = np.random.permutation(len(X))
    for i in range(0, len(idx), 64):
        b = idx[i:i + 64]
        bx, by = tr.from_numpy(X[b]), tr.from_numpy(y[b])
        logits = fc2.forward(tr.relu(fc1.forward(bx)))
        loss = tr.nn.cross_entropy_loss(logits, by)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"epoch {epoch}: loss={float(np.asarray(loss)[0]):.4f}")
```

Expected: loss decreases to ~0.1 after 5 epochs, ~95%+ test accuracy.

## Example: one GPT training step

```python
import numpy as np
import tiramisu as tr

vocab, seq = 65, 8
model = tr.nn.GPT(vocab_size=vocab, d_model=32, num_heads=2, num_layers=1, max_seq_len=seq)
opt = tr.optim.Adam(model.parameters(), lr=1e-3)

tokens = tr.from_numpy((np.arange(seq) % vocab).reshape(1, seq).astype(np.float32))
logits = model.forward(tokens)  # (batch, seq, vocab)

flat_logits = tr.from_numpy(np.asarray(logits)[:, :-1, :].reshape(-1, vocab))
flat_targets = tr.from_numpy(np.asarray(tokens)[:, 1:].reshape(-1))
loss = tr.nn.cross_entropy_loss(flat_logits, flat_targets)

opt.zero_grad(); loss.backward(); opt.step()
print(f"loss={float(np.asarray(loss)[0]):.4f}")
```

Full training loops for both live in [`examples/`](examples/) (C++) and [`examples/python/`](examples/python/).

## API

**Tensor ops**: `add`, `sub`, `mul`, `div`, `neg`, `matmul`, `sum`, `mean`, `reshape`, `transpose`, `contiguous`, `relu`, `gelu`, `softmax`, `from_numpy`, `backward`

**Modules**:: `nn.Linear`, `nn.LayerNorm`, `nn.GPT`, `nn.cross_entropy_loss`

**Optimizers**:: `optim.Adam`

Full binding reference in [`python/README.md`](python/README.md).

## Build from source

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

CUDA: `-DTIRAMISU_ENABLE_CUDA=ON`. Debug builds enable ASan+UBSan by default.

Run the C++ MNIST example:

```bash
cmake --build build --target mnist && ./build/examples/mnist
```

Char-level GPT on Tiny Shakespeare (presets `tiny`, `2m`, `10m`; add `--cuda` for GPU):

```bash
cmake --build build --target train_shakespeare
./build/examples/train_shakespeare --preset tiny --epochs 3
```

## Layout

```
core/       Storage, Tensor, dtype, device
ops/cpu/    Forward kernels (elementwise, reduce, matmul, normalization)
ops/cuda/   Optional CUDA kernels
autograd/   Differentiable wrappers, backward(), gradcheck
nn/         Module, Linear, GPT, LayerNorm, loss
optim/      SGD, Adam, AdamW, grad clipping, cosine LR
python/     pybind11 bindings
serialize/  GPT checkpoint save/load
examples/   hello_tiramisu, mnist, train_shakespeare
```

License: MIT.
