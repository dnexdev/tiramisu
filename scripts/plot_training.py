#!/usr/bin/env python3
"""Parse examples/mnist stdout and render a training-loss GIF for the README."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation

EPOCH_RE = re.compile(r"Epoch\s+(\d+),\s*Avg Loss:\s*([\d.]+)")
ACCURACY_RE = re.compile(r"Test Accuracy:\s*([\d.]+)%")


def parse_log(text: str) -> tuple[list[int], list[float], float | None]:
    epochs: list[int] = []
    losses: list[float] = []
    accuracy: float | None = None

    for line in text.splitlines():
        m = EPOCH_RE.search(line)
        if m:
            epochs.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            continue
        a = ACCURACY_RE.search(line)
        if a:
            accuracy = float(a.group(1))

    if not epochs:
        raise ValueError("No 'Epoch N, Avg Loss: X' lines found in log.")
    return epochs, losses, accuracy


def render_gif(
    epochs: list[int],
    losses: list[float],
    accuracy: float | None,
    output: Path,
    fps: int = 2,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=120)
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#ffffff")
    fig.subplots_adjust(top=0.78)

    (line,) = ax.plot([], [], color="#2563eb", linewidth=2.5, marker="o", markersize=6)
    ax.set_xlim(-0.3, max(epochs) + 0.3)
    ymin = min(losses) * 0.95
    ymax = max(losses) * 1.08
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Average training loss", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle(
        "MNIST — 2-layer MLP (tiramisu)",
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )
    subtitle = fig.text(
        0.5,
        0.91,
        "784 → 128 → ReLU → 10 · Adam · batch 64",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    epoch_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    def update(frame: int):
        xs = epochs[: frame + 1]
        ys = losses[: frame + 1]
        line.set_data(xs, ys)
        epoch_text.set_text(f"Epoch {epochs[frame]}\nLoss {losses[frame]:.4f}")
        if frame == len(epochs) - 1 and accuracy is not None:
            subtitle.set_text(
                f"784 → 128 → ReLU → 10 · Adam · batch 64 · test acc {accuracy:.1f}%"
            )
        return line, epoch_text, subtitle

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(epochs),
        interval=1000 // fps,
        blit=False,
    )
    anim.save(output, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_file",
        type=Path,
        help="mnist stdout log (e.g. build/mnist_run.log)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("docs/assets/mnist_training.gif"),
        help="Output GIF path",
    )
    parser.add_argument("--fps", type=int, default=2, help="Animation frames per second")
    args = parser.parse_args()

    text = args.log_file.read_text()
    epochs, losses, accuracy = parse_log(text)
    render_gif(epochs, losses, accuracy, args.output, fps=args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
