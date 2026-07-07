#!/usr/bin/env python3
"""Train a 2-layer MLP on MNIST using the pip-installable API.

Requires MNIST IDX files in a data directory (default: data/):
  train-images-idx3-ubyte, train-labels-idx1-ubyte

Download from http://yann.lecun.com/exdb/mnist/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tiramisu as tr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data", help="Directory with MNIST IDX files")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


def load_images(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        _, n, r, c = np.frombuffer(f.read(16), ">u4")
        return np.frombuffer(f.read(), np.uint8).reshape(n, r * c).astype(np.float32) / 255


def load_labels(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        _, n = np.frombuffer(f.read(8), ">u4")
        return np.frombuffer(f.read(), np.uint8).astype(np.float32)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    X = load_images(data_dir / "train-images-idx3-ubyte")
    y = load_labels(data_dir / "train-labels-idx1-ubyte")

    fc1 = tr.nn.Linear(784, 128)
    fc2 = tr.nn.Linear(128, 10)
    opt = tr.optim.Adam(fc1.parameters() + fc2.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        idx = np.random.permutation(len(X))
        for i in range(0, len(idx), args.batch_size):
            b = idx[i : i + args.batch_size]
            bx, by = tr.from_numpy(X[b]), tr.from_numpy(y[b])
            logits = fc2.forward(tr.relu(fc1.forward(bx)))
            loss = tr.nn.cross_entropy_loss(logits, by)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"epoch {epoch}: loss={float(np.asarray(loss)[0]):.4f}")


if __name__ == "__main__":
    main()
