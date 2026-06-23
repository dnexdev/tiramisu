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
    ids = tr.from_numpy(np.mod(np.arange(seq), vocab).astype(np.float32).reshape(1, seq))
    logits = model.forward(ids)

    logits_np = np.asarray(logits)
    targets_np = np.asarray(ids)
    flat_logits = tr.from_numpy(logits_np[:, :-1, :].reshape(-1, vocab))
    flat_targets = tr.from_numpy(targets_np[:, 1:].reshape(-1))
    loss = tr.nn.cross_entropy_loss(flat_logits, flat_targets)
    loss.backward()

    opt = tr.optim.Adam(model.parameters(), lr=1e-2)
    opt.zero_grad()
    loss = tr.nn.cross_entropy_loss(flat_logits, flat_targets)
    loss.backward()
    opt.step()

    assert float(np.asarray(loss)[0]) > 0.0
