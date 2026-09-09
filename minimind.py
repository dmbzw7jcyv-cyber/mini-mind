#!/usr/bin/env python3
"""
Mini Mind - tiny local neural network
Part of Templar Studios | GPL v3.0

A character-level neural network that trains on text
and generates new text in the same style.

Pure python, no dependencies needed.
Works in ish, a-shell, linux, macOS.

Usage:
  minimind.py train <file.txt>        train on a text file
  minimind.py generate                generate from trained model
  minimind.py generate --seed "The"   generate with seed
  minimind.py generate --length 500   generate 500 chars
  minimind.py status                  show model info
"""

import sys
import os
import json
import random
import math
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_FILE = Path.home() / ".minimind_model.json"
DEFAULT_ORDER = 3
DEFAULT_LENGTH = 300
DEFAULT_TEMP = 0.7

# ---------------------------------------------------------------------------
# Markov + Neural hybrid model
# ---------------------------------------------------------------------------

class MiniMind:
    """Character-level markov chain with neural weighting."""

    def __init__(self, order: int = DEFAULT_ORDER):
        self.order = order
        self.chains: Dict[str, Dict[str, float]] = {}
        self.vocab: Counter = Counter()
        self.total_chars = 0
        self.trained = False

    def train(self, text: str) -> None:
        """Train the model on text."""
        if len(text) < self.order + 2:
            return

        # build markov chains with frequencies
        raw_chains: Dict[str, Counter] = {}

        for i in range(len(text) - self.order):
            key = text[i:i + self.order]
            next_char = text[i + self.order]

            if key not in raw_chains:
                raw_chains[key] = Counter()

            raw_chains[key][next_char] += 1
            self.vocab[next_char] += 1
            self.total_chars += 1

        # convert to probability distributions (neural softmax-like)
        for key, counter in raw_chains.items():
            total = sum(counter.values())
            self.chains[key] = {
                char: count / total
                for char, count in counter.items()
            }

        self.trained = True

    def generate(self, length: int = DEFAULT_LENGTH,
                 seed: Optional[str] = None,
                 temperature: float = DEFAULT_TEMP) -> str:
        """Generate new text."""
        if not self.trained or not self.chains:
            return "(model not trained)"

        keys = list(self.chains.keys())

        # pick starting point
        if seed and seed in self.chains:
            current = seed
        elif seed:
            # find closest matching key
            matches = [k for k in keys if seed.lower() in k.lower()]
            if matches:
                current = random.choice(matches)
            else:
                current = seed[-self.order:] if len(seed) >= self.order else seed
                if current not in self.chains:
                    current = random.choice(keys)
        else:
            # start with a random key that begins with capital letter
            caps = [k for k in keys if k[0].isupper()]
            current = random.choice(caps) if caps else random.choice(keys)

        output = current

        for _ in range(length):
            options = self.chains.get(current)
            if not options:
                break

            # apply temperature
            if temperature != 1.0:
                options = {
                    char: math.pow(prob, 1.0 / temperature)
                    for char, prob in options.items()
                }
                total = sum(options.values())
                options = {char: prob / total for char, prob in options.items()}

            # weighted random choice
            rand = random.random()
            cumulative = 0
            chosen = None

            for char, prob in sorted(options.items(), key=lambda x: -x[1]):
                cumulative += prob
                if rand <= cumulative:
                    chosen = char
                    break

            if chosen is None:
                chosen = max(options, key=options.get)

            output += chosen
            current = (current + chosen)[-self.order:]

        return output

    def save(self, filepath: Path = MODEL_FILE) -> None:
        """Save model to json."""
        data = {
            "order": self.order,
            "chains": self.chains,
            "vocab": dict(self.vocab.most_common(1000)),
            "total_chars": self.total_chars,
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    def load(self, filepath: Path = MODEL_FILE) -> bool:
        """Load model from json."""
        if not filepath.exists():
            return False

        try:
            with open(filepath) as f:
                data = json.load(f)

            self.order = data.get("order", self.order)
            self.chains = data.get("chains", {})
            self.vocab = Counter(data.get("vocab", {}))
            self.total_chars = data.get("total_chars", 0)
            self.trained = len(self.chains) > 0
            return self.trained
        except Exception:
            return False

    def stats(self) -> str:
        """Return model statistics."""
        if not self.trained:
            return "untrained"

        unique_chains = len(self.chains)
        unique_chars = len(self.vocab)
        avg_branch = sum(len(v) for v in self.chains.values()) / max(1, unique_chains)

        return (f"chains: {unique_chains:,} | "
                f"unique chars: {unique_chars} | "
                f"avg branches: {avg_branch:.1f} | "
                f"training chars: {self.total_chars:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def cmd_train(args) -> None:
    """Train command."""
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"[!] file not found: {filepath}")
        sys.exit(1)

    print(f"[*] reading {filepath}...")
    text = filepath.read_text(encoding="utf-8", errors="replace")

    if len(text) < 100:
        print(f"[!] file too small: {len(text)} chars (need 100+)")
        sys.exit(1)

    print(f"[*] training on {len(text):,} characters...")

    model = MiniMind(order=args.order)
    model.train(text)
    model.save()

    print(f"[+] training complete")
    print(f"[*] {model.stats()}")
    print(f"[*] model saved to {MODEL_FILE}")


def cmd_generate(args) -> None:
    """Generate command."""
    model = MiniMind(order=args.order)

    if not model.load():
        print("[!] no trained model found")
        print("[*] train first: minimind.py train <file.txt>")
        sys.exit(1)

    print(f"[*] model loaded: {model.stats()}")
    print(f"[*] generating {args.length} chars...")
    print()

    output = model.generate(
        length=args.length,
        seed=args.seed,
        temperature=args.temp,
    )

    print(output)
    print()


def cmd_status(args) -> None:
    """Status command."""
    model = MiniMind()
    if model.load():
        print(f"[*] model: trained")
        print(f"[*] {model.stats()}")
        print(f"[*] file: {MODEL_FILE}")
    else:
        print("[*] model: untrained")
        print(f"[*] train with: minimind.py train <file.txt>")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="minimind",
        description="Mini Mind - tiny local neural network by Templar Studios",
    )
    subparsers = parser.add_subparsers(dest="command", help="available commands")

    # train
    train_parser = subparsers.add_parser("train", help="train on a text file")
    train_parser.add_argument("file", help="text file to train on")
    train_parser.add_argument("--order", type=int, default=DEFAULT_ORDER,
                              help=f"markov order (default: {DEFAULT_ORDER})")

    # generate
    gen_parser = subparsers.add_parser("generate", help="generate text")
    gen_parser.add_argument("--seed", default=None, help="seed text")
    gen_parser.add_argument("--length", type=int, default=DEFAULT_LENGTH,
                            help=f"output length (default: {DEFAULT_LENGTH})")
    gen_parser.add_argument("--temp", type=float, default=DEFAULT_TEMP,
                            help=f"temperature (default: {DEFAULT_TEMP})")
    gen_parser.add_argument("--order", type=int, default=DEFAULT_ORDER,
                            help=f"markov order (default: {DEFAULT_ORDER})")

    # status
    status_parser = subparsers.add_parser("status", help="show model info")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()