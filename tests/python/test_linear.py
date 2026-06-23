import numpy as np

import tiramisu as tr


def test_linear_backward_smoke():
    layer = tr.nn.Linear(3, 2)
    x = tr.from_numpy(np.random.randn(1, 3).astype(np.float32))
    x.requires_grad = True
    out = layer.forward(x)
    loss = tr.sum(out)
    loss.backward()
    assert x.grad is not None
