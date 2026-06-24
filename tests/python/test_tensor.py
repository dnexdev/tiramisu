import numpy as np
import pytest

import tiramisu as tr


def test_tensor_shape_and_numpy_roundtrip():
    arr = np.random.randn(2, 3).astype(np.float32)
    t = tr.from_numpy(arr)
    assert t.shape() == [2, 3]
    assert t.numel() == 6
    back = np.asarray(t)
    np.testing.assert_allclose(back, arr)


def test_autograd_quadratic():
    x = tr.from_numpy(np.array([2.0], dtype=np.float32))
    x.requires_grad = True
    y = tr.add(tr.mul(x, x), tr.mul(x, 3.0))
    y.backward()
    assert x.grad is not None
    np.testing.assert_allclose(np.asarray(x.grad), [7.0])


def test_linear_forward_shape():
    layer = tr.nn.Linear(4, 2)
    x = tr.from_numpy(np.random.randn(3, 4).astype(np.float32))
    out = layer.forward(x)
    assert out.shape() == [3, 2]


def test_gpt_forward_and_backward():
    vocab = 16
    seq = 4
    model = tr.nn.GPT(
        vocab_size=vocab,
        d_model=8,
        num_heads=2,
        num_layers=1,
        max_seq_len=seq,
    )
    ids = tr.from_numpy(np.arange(seq, dtype=np.float32).reshape(1, seq))
    logits = model.forward(ids)
    assert logits.shape() == [1, seq, vocab]

    logits_np = np.asarray(logits)
    targets_np = np.asarray(ids)
    flat_logits = tr.from_numpy(logits_np[:, :-1, :].reshape(-1, vocab))
    flat_targets = tr.from_numpy(targets_np[:, 1:].reshape(-1))
    loss = tr.nn.cross_entropy_loss(flat_logits, flat_targets)
    loss.backward()
