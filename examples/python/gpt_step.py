#!/usr/bin/env python3
"""One GPT training step with explicit causal cross-entropy."""

import numpy as np

import tiramisu as tr

vocab_size = 65
seq_len = 8
batch_size = 1

model = tr.nn.GPT(
    vocab_size=vocab_size,
    d_model=32,
    num_heads=2,
    num_layers=1,
    max_seq_len=seq_len,
)

token_ids = np.mod(np.arange(batch_size * seq_len), vocab_size).astype(np.float32)
token_ids = token_ids.reshape(batch_size, seq_len)
input_ids = tr.from_numpy(token_ids)

logits = model.forward(input_ids)  # (batch, seq, vocab)

# Causal LM: predict token t+1 from position t.
logits_np = np.asarray(logits)
targets_np = np.asarray(input_ids)
flat_logits = tr.from_numpy(logits_np[:, :-1, :].reshape(-1, vocab_size))
flat_targets = tr.from_numpy(targets_np[:, 1:].reshape(-1))

loss = tr.nn.cross_entropy_loss(flat_logits, flat_targets)

opt = tr.optim.Adam(model.parameters(), lr=1e-3)
opt.zero_grad()
loss.backward()
opt.step()

print(f"loss={float(np.asarray(loss)[0]):.4f}")
