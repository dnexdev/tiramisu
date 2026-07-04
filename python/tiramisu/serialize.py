"""Checkpoint save/load."""

from tiramisu._C import serialize as _serialize

save_gpt = _serialize.save_gpt
load_gpt = _serialize.load_gpt

__all__ = ["save_gpt", "load_gpt"]
