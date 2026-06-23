"""Optimizers."""

from tiramisu._C import optim as _optim

Adam = _optim.Adam

__all__ = ["Adam"]
