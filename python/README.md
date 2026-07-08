# Python bindings

pybind11 wrappers over the C++ framework.

**API reference and runnable examples** live in the root [README](../README.md) and [`examples/python/`](../examples/python/).

## Install

From the repo root:

```bash
pip install .
# or editable dev install:
pip install -e ".[dev]"
pytest tests/python -v
```

## Examples

| Script | Description |
|--------|-------------|
| [`autograd_demo.py`](../examples/python/autograd_demo.py) | Minimal autograd |
| [`linear_forward.py`](../examples/python/linear_forward.py) | Linear forward + NumPy |
| [`train_mnist.py`](../examples/python/train_mnist.py) | MNIST MLP training |
| [`gpt_step.py`](../examples/python/gpt_step.py) | Single GPT training step |
| [`train_shakespeare.py`](../examples/python/train_shakespeare.py) | Char-level GPT training |

## Build notes

CMake option `TIRAMISU_BUILD_PYTHON=ON` builds the `tiramisu._C` extension. The pip build sets this automatically and disables sanitizers/tests for a clean wheel.

```bash
cmake -S . -B build -DTIRAMISU_BUILD_PYTHON=ON -DTIRAMISU_BUILD_TESTS=OFF \
  -DTIRAMISU_ENABLE_SANITIZERS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --target _C
```
