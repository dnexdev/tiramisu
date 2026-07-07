#!/usr/bin/env python3
"""Minimal autograd example: y = x^2 + 3x, dy/dx = 2x + 3."""

import numpy as np
import tiramisu as tr

x = tr.from_numpy(np.array([2.0], dtype=np.float32))
x.requires_grad = True
y = tr.add(tr.mul(x, x), tr.mul(x, 3.0))
y.backward()
print(np.asarray(x.grad))  # [7.]
