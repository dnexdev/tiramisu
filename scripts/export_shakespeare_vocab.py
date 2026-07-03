#!/usr/bin/env python3
"""Write Shakespeare CharVocab id_to_char table for WASM inference."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_vocab(text: str) -> str:
    mapping: dict[str, int] = {}
    for ch in text:
        if ch not in mapping:
            mapping[ch] = len(mapping)
    return "".join(ch for ch, _ in sorted(mapping.items(), key=lambda item: item[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "corpus",
        type=Path,
        default=Path("data/tiny_shakespeare.txt"),
        nargs="?",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("examples/shakespeare_vocab.inc"),
    )
    args = parser.parse_args()

    chars = build_vocab(args.corpus.read_text())
    escaped = (
        chars.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        f'constexpr char kShakespeareIdToChar[] = "{escaped}";\n'
        f"constexpr int kShakespeareVocabSize = {len(chars)};\n"
    )
    print(f"Wrote {args.output} ({len(chars)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
