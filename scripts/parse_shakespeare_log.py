#!/usr/bin/env python3
"""Parse train_shakespeare stdout into CSV for plotting."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

STEP_RE = re.compile(
    r"step (\d+) \| train_loss ([\d.]+) \| val_loss ([\d.]+) \| lr ([\d.]+)"
)
EPOCH_RE = re.compile(
    r"epoch (\d+) \| avg_train_loss ([\d.]+) \| val_loss ([\d.]+)"
)


def parse_log(text: str) -> tuple[list[dict], list[dict]]:
    steps: list[dict] = []
    epochs: list[dict] = []
    for line in text.splitlines():
        m = STEP_RE.search(line)
        if m:
            steps.append(
                {
                    "step": int(m.group(1)),
                    "train_loss": float(m.group(2)),
                    "val_loss": float(m.group(3)),
                    "lr": float(m.group(4)),
                }
            )
            continue
        m = EPOCH_RE.search(line)
        if m:
            epochs.append(
                {
                    "epoch": int(m.group(1)),
                    "avg_train_loss": float(m.group(2)),
                    "val_loss": float(m.group(3)),
                }
            )
    return steps, epochs


def write_csv(path: Path, steps: list[dict], epochs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "step", "epoch", "train_loss", "val_loss", "lr"])
        for row in steps:
            w.writerow(
                ["step", row["step"], "", row["train_loss"], row["val_loss"], row["lr"]]
            )
        for row in epochs:
            w.writerow(
                [
                    "epoch",
                    "",
                    row["epoch"],
                    row["avg_train_loss"],
                    row["val_loss"],
                    "",
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", type=Path, help="train_shakespeare stdout log")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: <log>.csv next to log)",
    )
    args = parser.parse_args()

    text = args.log_file.read_text()
    steps, epochs = parse_log(text)
    if not steps and not epochs:
        print("No step/epoch lines found.", file=sys.stderr)
        return 1

    out = args.output or args.log_file.with_suffix(".csv")
    write_csv(out, steps, epochs)
    print(f"Wrote {out} ({len(steps)} steps, {len(epochs)} epochs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
