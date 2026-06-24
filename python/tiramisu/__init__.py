"""tiramisu — a readable from-scratch ML framework."""

from tiramisu._C import (  # noqa: F401
    Tensor,
    add,
    backward,
    contiguous,
    div,
    from_numpy,
    gelu,
    matmul,
    mean,
    mul,
    neg,
    relu,
    reshape,
    softmax,
    sub,
    sum,
    transpose,
)

__all__ = [
    "Tensor",
    "add",
    "backward",
    "contiguous",
    "div",
    "from_numpy",
    "gelu",
    "matmul",
    "mean",
    "mul",
    "neg",
    "relu",
    "reshape",
    "softmax",
    "sub",
    "sum",
    "transpose",
    "nn",
    "optim",
]

from tiramisu import nn, optim  # noqa: E402
