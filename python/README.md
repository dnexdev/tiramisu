# Python bindings

pybind11 wrappers over the C++ framework. Install from the repo root:

```bash
pip install .
# or editable dev install:
pip install -e ".[dev]"
pytest tests/python -v
```

## Quick examples

### Autograd

```python
import numpy as np
import tiramisu as tr

x = tr.from_numpy(np.array([2.0], dtype=np.float32))
x.requires_grad = True
y = tr.add(tr.mul(x, x), tr.mul(x, 3.0))
y.backward()
print(np.asarray(x.grad))  # [7.]
```

### Linear forward + NumPy

```python
import numpy as np
import tiramisu as tr

layer = tr.nn.Linear(784, 10)
x = tr.from_numpy(np.random.randn(2, 784).astype(np.float32))
out = layer.forward(x)
arr = np.asarray(out)  # zero-copy view when contiguous
```

### GPT — explicit causal loss

`forward()` returns logits. Apply the next-token shift yourself:

```python
import numpy as np
import tiramisu as tr

vocab_size = 65
seq_len = 8
model = tr.nn.GPT(vocab_size=vocab_size, d_model=32, num_layers=1, max_seq_len=seq_len)

input_ids = tr.from_numpy(np.arange(seq_len, dtype=np.float32).reshape(1, seq_len))
logits = model.forward(input_ids)  # (batch, seq, vocab)

logits_np = np.asarray(logits)
targets_np = np.asarray(input_ids)
flat_logits = tr.from_numpy(logits_np[:, :-1, :].reshape(-1, vocab_size))
flat_targets = tr.from_numpy(targets_np[:, 1:].reshape(-1))

loss = tr.nn.cross_entropy_loss(flat_logits, flat_targets)
loss.backward()
```

Token indices are stored as `float32` tensors today (the C++ embedding path reads float indices).

## API surface (v1)

| Module | Symbols |
|--------|---------|
| `tiramisu` | `Tensor`, `from_numpy`, `add`, `mul`, `sub`, `div`, `relu`, `gelu`, `softmax`, `matmul`, `sum`, `mean`, `reshape`, `transpose`, `backward` |
| `tiramisu.nn` | `Linear`, `LayerNorm`, `GPT`, `cross_entropy_loss` |
| `tiramisu.optim` | `Adam` |

### `Tensor`

- `Tensor(shape)` — float32 CPU tensor
- `from_numpy(arr)` — copy from NumPy (`float32` or castable)
- `numpy()` / `np.asarray(t)` — zero-copy export when contiguous
- `requires_grad`, `grad`, `backward()`
- `shape()`, `numel()`, `reshape()`, `slice()`, `contiguous()`

### `GPT`

Keyword args: `vocab_size`, `d_model`, `num_heads=2`, `num_layers=2`, `max_seq_len=256`.

Methods: `forward(token_ids)`, `parameters()`, `config()`.

## Build notes

CMake option `TIRAMISU_BUILD_PYTHON=ON` builds the `tiramisu._C` extension. The pip build sets this automatically and disables sanitizers/tests for a clean wheel.

```bash
cmake -S . -B build -DTIRAMISU_BUILD_PYTHON=ON -DTIRAMISU_BUILD_TESTS=OFF \
  -DTIRAMISU_ENABLE_SANITIZERS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --target _C
```
