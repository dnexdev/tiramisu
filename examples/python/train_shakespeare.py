#!/usr/bin/env python3
"""Train a char-level GPT on Tiny Shakespeare using the pip-installable API.

Mirrors examples/train_shakespeare.cpp. Presets: tiny, 2m, 10m.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import tiramisu as tr


PRESETS = {
    "tiny": dict(d_model=64, num_heads=2, num_layers=2, seq_len=64, batch_size=8),
    "2m":   dict(d_model=200, num_heads=4, num_layers=4, seq_len=128, batch_size=16),
    "10m":  dict(d_model=384, num_heads=6, num_layers=6, seq_len=256, batch_size=16),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/tiny_shakespeare.txt")
    p.add_argument("--preset", choices=list(PRESETS), default="tiny")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--checkpoint-interval", type=int, default=200)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--max-batches", type=int, default=-1,
                   help="Cap total optimizer steps (for smoke tests).")
    p.add_argument("--resume", action="store_true",
                   help="With --checkpoint, load and continue training.")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                   help="Run model + tensors on cpu or cuda (requires CUDA build).")
    return p.parse_args()


def build_vocab(text: str) -> tuple[dict[str, int], list[str]]:
    chars: list[str] = []
    seen: dict[str, int] = {}
    for c in text:
        if c not in seen:
            seen[c] = len(chars)
            chars.append(c)
    return seen, chars


def encode(text: str, ctoi: dict[str, int]) -> np.ndarray:
    return np.fromiter((ctoi[c] for c in text), dtype=np.float32, count=len(text))


def iter_batches(ids: np.ndarray, batch_size: int, seq_len: int):
    """Yield (input, target) uint tensors of shape (B, S), sliding window over `ids`."""
    step = batch_size * seq_len
    n = len(ids)
    for start in range(0, n - seq_len - 1, step):
        rows_x, rows_y = [], []
        for b in range(batch_size):
            offset = start + b * seq_len
            if offset + seq_len + 1 > n:
                break
            rows_x.append(ids[offset:offset + seq_len])
            rows_y.append(ids[offset + 1:offset + seq_len + 1])
        if not rows_x:
            return
        yield np.stack(rows_x), np.stack(rows_y)


def evaluate(model: tr.nn.GPT, val_ids: np.ndarray, vocab_size: int,
             seq_len: int, batch_size: int, device: str,
             max_batches: int = 20) -> float:
    total, count = 0.0, 0
    for i, (bx, by) in enumerate(iter_batches(val_ids, batch_size, seq_len)):
        if i >= max_batches:
            break
        logits = model.forward(tr.from_numpy(bx, device=device))
        b, s = bx.shape
        flat = tr.reshape(logits, [b * s, vocab_size])
        targets = tr.from_numpy(by.reshape(-1), device=device)
        loss = tr.nn.cross_entropy_loss(flat, targets)
        total += float(loss.cpu().numpy()[0])
        count += 1
    return total / max(count, 1)


def main() -> None:
    args = parse_args()
    cfg = PRESETS[args.preset]

    text = Path(args.data).read_text()
    ctoi, itoc = build_vocab(text)
    vocab_size = len(itoc)
    ids = encode(text, ctoi)
    split = int(0.9 * len(ids))
    train_ids, val_ids = ids[:split], ids[split:]

    print(f"preset={args.preset} | vocab={vocab_size} | "
          f"train={len(train_ids):,} val={len(val_ids):,} tokens")

    if args.device == "cuda" and not tr.cuda_available():
        raise RuntimeError(
            "--device cuda requires a CUDA-enabled build. Reinstall with:\n"
            "  pip install --no-binary=tiramisu-ml \\\n"
            "    --config-settings=cmake.define.TIRAMISU_ENABLE_CUDA=ON \\\n"
            "    tiramisu-ml"
        )

    model = tr.nn.GPT(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["seq_len"],
        tie_weights=True,
        device=args.device,
    )
    print(f"device={args.device}")

    resume_step = 0
    resume_epoch = 0
    if args.checkpoint and args.resume:
        resume_step, resume_epoch = tr.serialize.load_gpt(args.checkpoint, model)
        print(f"resumed from step={resume_step} epoch={resume_epoch}")

    params = model.parameters()
    opt = tr.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    opt.step_count = resume_step

    steps_per_epoch = max(1, len(train_ids) // (cfg["batch_size"] * cfg["seq_len"]))
    scheduler = tr.optim.CosineAnnealingLR(
        base_lr=args.lr,
        total_steps=steps_per_epoch * args.epochs,
        min_lr=args.lr * 0.1,
    )

    global_step = resume_step
    start_epoch = resume_epoch if args.resume else 0
    target_epoch = (resume_epoch + args.epochs) if args.resume else args.epochs

    t0 = time.time()
    for epoch in range(start_epoch, target_epoch):
        epoch_loss = 0.0
        batches = 0
        for bx, by in iter_batches(train_ids, cfg["batch_size"], cfg["seq_len"]):
            if 0 <= args.max_batches <= global_step:
                break

            opt.zero_grad()
            logits = model.forward(tr.from_numpy(bx, device=args.device))
            b, s = bx.shape
            flat_logits = tr.reshape(logits, [b * s, vocab_size])
            flat_targets = tr.from_numpy(by.reshape(-1), device=args.device)
            loss = tr.nn.cross_entropy_loss(flat_logits, flat_targets)
            loss.backward()

            tr.optim.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            global_step += 1

            lr = scheduler.step()
            opt.lr = lr

            train_loss = float(loss.cpu().numpy()[0])
            epoch_loss += train_loss
            batches += 1

            if global_step % args.eval_interval == 0:
                val = evaluate(model, val_ids, vocab_size,
                               cfg["seq_len"], cfg["batch_size"], args.device)
                dt = time.time() - t0
                print(f"step {global_step:>5} | train {train_loss:.4f} | "
                      f"val {val:.4f} | lr {lr:.6f} | {dt:.1f}s")

            if args.checkpoint and global_step % args.checkpoint_interval == 0:
                tr.serialize.save_gpt(args.checkpoint, model, global_step, epoch)

        avg = epoch_loss / max(batches, 1)
        print(f"epoch {epoch}: avg_loss={avg:.4f}")
        if args.checkpoint:
            tr.serialize.save_gpt(args.checkpoint, model, global_step, epoch + 1)


if __name__ == "__main__":
    main()
