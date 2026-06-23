#!/usr/bin/env python3
"""Forward pass through a Linear layer — headline Python demo."""

import numpy as np

import tiramisu as tr

x = tr.Tensor([2, 784])
layer = tr.nn.Linear(784, 10)
out = layer.forward(x)
print("shape:", out.shape())

x = tr.from_numpy(np.random.randn(2, 784).astype(np.float32))
out = layer.forward(x)
arr = np.asarray(out)
print("numpy view shape:", arr.shape, "zero-copy:", arr.base is not None)
