import numpy as np

import tiramisu as tr


def test_gpt_training_step():
    vocab = 12
    seq = 4
    model = tr.nn.GPT(
        vocab_size=vocab,
        d_model=8,
        num_heads=2,
        num_layers=1,
        max_seq_len=seq,
    )
    tokens = np.mod(np.arange(seq + 1), vocab).astype(np.float32)
    ids = tr.from_numpy(tokens[:-1].reshape(1, seq))
    targets = tr.from_numpy(tokens[1:].reshape(-1))

    params = model.parameters()
    opt = tr.optim.Adam(params, lr=1e-2)
    opt.zero_grad()

    logits = model.forward(ids)
    flat_logits = tr.reshape(logits, [seq, vocab])
    loss = tr.nn.cross_entropy_loss(flat_logits, targets)
    loss.backward()

    assert any(p.grad is not None for p in params)
    weight_before = np.array(params[0], copy=True)
    opt.step()
    weight_after = np.asarray(params[0])
    assert not np.allclose(weight_before, weight_after)
    assert float(np.asarray(loss)[0]) > 0.0
