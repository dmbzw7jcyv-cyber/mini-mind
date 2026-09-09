#!/usr/bin/env python3
"""
Mini Mind v3 - tiny local neural network with persistent memory
Part of Templar Studios | GPL v3.0

A character-level markov chain that:
  - trains on specific text files
  - stores multiple trained models persistently
  - can generate from a single model
  - can generate from merged memory of everything it has seen

Pure python, no dependencies.
Works in ish, a-shell, linux, macOS.

Usage:
  minimind.py train <file.txt>                 train and add to memory
  minimind.py generate                          generate from current model
  minimind.py generate --seed "The" --length 500
  minimind.py freegen                           generate from merged memory
  minimind.py freegen --length 500 --temp 1.0
  minimind.py models                            list all stored models
  minimind.py status                            show model info
"""

import sys
import os
import json
import math
import random
import time
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR = Path.home() / ".minimind_models"
DEFAULT_MODEL = "current"
DEFAULT_ORDER = 4
DEFAULT_LENGTH = 300
DEFAULT_TEMP = 0.7
PROGRESS_INTERVAL = 5000
BAR_WIDTH = 30

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MiniMind:
    """Character-level markov chain with persistent multi-model memory."""

    def __init__(self, order: int = DEFAULT_ORDER, name: str = DEFAULT_MODEL):
        self.order = order
        self.name = name
        self.chains: Dict[str, Dict[str, float]] = {}
        self.vocab: Counter = Counter()
        self.total_chars = 0
        self.trained = False
        self.spinner_chars = ["|", "/", "-", "\\"]
        self.spinner_idx = 0
        self.model_file = MODEL_DIR / f"{name}.json"

    # ──────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────

    def _progress_bar(self, current: int, total: int) -> str:
        pct = (current / total) * 100 if total > 0 else 0
        filled = int(pct / 100 * BAR_WIDTH)
        return f"[{'█' * filled}{'░' * (BAR_WIDTH - filled)}] {pct:5.1f}%"

    def _spinner(self) -> str:
        char = self.spinner_chars[self.spinner_idx % len(self.spinner_chars)]
        self.spinner_idx += 1
        return char

    def _show_progress(self, current: int, total: int, sample: str) -> None:
        bar = self._progress_bar(current, total)
        spinner = self._spinner()
        sys.stdout.write("\r\033[K")
        sys.stdout.write(
            f"  {spinner} {bar}  chains: {len(self.chains):,}  "
            f"chars: {len(self.vocab)}  [{sample[:30]}]"
        )
        sys.stdout.flush()

    def train(self, text: str, show_progress: bool = True,
              add_to_memory: bool = True) -> None:
        """Train on text and optionally merge into existing model."""
        if len(text) < self.order + 2:
            print("[!] text too small")
            return

        total = len(text) - self.order
        raw_chains: Dict[str, Counter] = {}

        print()
        print(f"  training on {len(text):,} characters...")
        print(f"  order: {self.order}")
        print()

        start_time = time.time()

        for i in range(total):
            key = text[i:i + self.order]
            next_char = text[i + self.order]

            if key not in raw_chains:
                raw_chains[key] = Counter()

            raw_chains[key][next_char] += 1
            self.vocab[next_char] += 1
            self.total_chars += 1

            if show_progress and (i % PROGRESS_INTERVAL == 0 or i == total - 1):
                self._show_progress(i + 1, total, key)

        print()

        if add_to_memory and self.chains:
            print()
            print("  merging with existing memory...")

        # convert raw counts to probabilities
        for key, counter in raw_chains.items():
            total_count = sum(counter.values())
            probs = {char: count / total_count for char, count in counter.items()}

            if add_to_memory and key in self.chains:
                # blend old and new
                blended = {}
                for char in set(self.chains[key]) | set(probs):
                    old_prob = self.chains[key].get(char, 0)
                    new_prob = probs.get(char, 0)
                    blended[char] = (old_prob + new_prob) / 2
                self.chains[key] = blended
            else:
                self.chains[key] = probs

        self.trained = True
        elapsed = time.time() - start_time

        print()
        print(f"  training complete in {elapsed:.1f}s")
        print(f"  chains: {len(self.chains):,}")
        print(f"  unique chars: {len(self.vocab)}")
        print(f"  total transitions: {self.total_chars:,}")
        print()

        self.save()

    # ──────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────

    def generate(self, length: int = DEFAULT_LENGTH,
                 seed: Optional[str] = None,
                 temperature: float = DEFAULT_TEMP,
                 show_progress: bool = True) -> str:
        """Generate text from this model."""
        if not self.trained or not self.chains:
            return "(model not trained)"

        keys = list(self.chains.keys())

        # pick starting point
        if seed and seed in self.chains:
            current = seed
        elif seed:
            matches = [k for k in keys if seed.lower() in k.lower()]
            if matches:
                current = random.choice(matches)
            else:
                current = seed[-self.order:] if len(seed) >= self.order else seed
                if current not in self.chains:
                    current = random.choice(keys)
        else:
            caps = [k for k in keys if k[0].isupper()]
            current = random.choice(caps) if caps else random.choice(keys)

        output = current

        if show_progress:
            print("  generating", end="", flush=True)

        for i in range(length):
            options = self.chains.get(current)
            if not options:
                break

            if temperature != 1.0:
                options = {
                    char: math.pow(prob, 1.0 / temperature)
                    for char, prob in options.items()
                }
                total = sum(options.values())
                if total > 0:
                    options = {char: prob / total for char, prob in options.items()}

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

            if show_progress and i % 50 == 0:
                print(".", end="", flush=True)

        if show_progress:
            print()

        return output

    # ──────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────

    def save(self) -> None:
        """Save model to disk."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "order": self.order,
            "chains": self.chains,
            "vocab": dict(self.vocab.most_common(3000)),
            "total_chars": self.total_chars,
            "name": self.name,
        }
        with open(self.model_file, "w") as f:
            json.dump(data, f)

    def load(self, name: Optional[str] = None) -> bool:
        """Load model from disk."""
        if name:
            self.name = name
            self.model_file = MODEL_DIR / f"{name}.json"

        if not self.model_file.exists():
            return False

        try:
            with open(self.model_file) as f:
                data = json.load(f)

            self.order = data.get("order", self.order)
            self.chains = data.get("chains", {})
            self.vocab = Counter(data.get("vocab", {}))
            self.total_chars = data.get("total_chars", 0)
            self.name = data.get("name", self.name)
            self.trained = len(self.chains) > 0
            return self.trained
        except Exception:
            return False

    # ──────────────────────────────────────────
    # Stats
    # ──────────────────────────────────────────

    def stats(self) -> str:
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
# Model management
# ---------------------------------------------------------------------------

def list_models() -> List[str]:
    """List all stored models."""
    if not MODEL_DIR.exists():
        return []
    return sorted(f.stem for f in MODEL_DIR.glob("*.json"))


def load_any_model() -> Optional[MiniMind]:
    """Load any available model."""
    models = list_models()
    if not models:
        return None

    # prefer current, fallback to any
    name = "current" if "current" in models else models[0]
    model = MiniMind()
    model.load(name)
    return model


def merge_all_models() -> MiniMind:
    """Merge all stored models into one."""
    models = list_models()
    merged = MiniMind(name="merged")
    first = True

    for name in models:
        m = MiniMind()
        if m.load(name):
            if first:
                merged.chains = dict(m.chains)
                merged.vocab = Counter(m.vocab)
                merged.total_chars = m.total_chars
                merged.order = m.order
                first = False
            else:
                # merge chains
                for key, probs in m.chains.items():
                    if key in merged.chains:
                        blended = {}
                        for char in set(merged.chains[key]) | set(probs):
                            old_prob = merged.chains[key].get(char, 0)
                            new_prob = probs.get(char, 0)
                            blended[char] = (old_prob + new_prob) / 2
                        merged.chains[key] = blended
                    else:
                        merged.chains[key] = dict(probs)

                merged.vocab.update(m.vocab)
                merged.total_chars += m.total_chars

    merged.trained = len(merged.chains) > 0
    return merged


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_train(args) -> None:
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"[!] file not found: {filepath}")
        sys.exit(1)

    print(f"[*] reading {filepath}...")
    text = filepath.read_text(encoding="utf-8", errors="replace")

    if len(text) < 100:
        print(f"[!] file too small: {len(text)} chars (need 100+)")
        sys.exit(1)

    # load existing model to merge into
    model = MiniMind(order=args.order)
    if not args.fresh:
        model.load("current")

    model.train(text, show_progress=True, add_to_memory=not args.fresh)
    model.name = "current"
    model.save()

    print(f"[+] model saved to {model.model_file}")


def cmd_generate(args) -> None:
    model = load_any_model()
    if not model:
        print("[!] no trained model found")
        print("[*] train first: minimind.py train <file.txt>")
        sys.exit(1)

    print(f"[*] model: {model.name}")
    print(f"[*] {model.stats()}")
    print(f"[*] length: {args.length} | temp: {args.temp}")
    print()

    output = model.generate(
        length=args.length,
        seed=args.seed,
        temperature=args.temp,
    )

    print()
    print("─" * 60)
    print(output)
    print("─" * 60)
    print()


def cmd_freegen(args) -> None:
    """Generate from merged memory."""
    print("[*] merging all models...")
    model = merge_all_models()

    if not model.trained:
        print("[!] no models to merge")
        sys.exit(1)

    print(f"[*] merged {len(list_models())} model(s)")
    print(f"[*] {model.stats()}")
    print(f"[*] length: {args.length} | temp: {args.temp}")
    print()

    output = model.generate(
        length=args.length,
        seed=args.seed,
        temperature=args.temp,
    )

    print()
    print("─" * 60)
    print(output)
    print("─" * 60)
    print()


def cmd_models(args) -> None:
    models = list_models()

    print()
    print("stored models:")
    print("─" * 60)

    if not models:
        print("  (none)")
    else:
        for name in models:
            m = MiniMind()
            if m.load(name):
                size_kb = m.model_file.stat().st_size / 1024
                print(f"  {name:<20} {m.stats()}  ({size_kb:.1f} KB)")

    print("─" * 60)
    print()


def cmd_status(args) -> None:
    model = load_any_model()
    if model:
        print(f"[*] model: {model.name}")
        print(f"[*] {model.stats()}")
        print(f"[*] file: {model.model_file}")
    else:
        print("[*] model: untrained")
        print(f"[*] train with: minimind.py train <file.txt>")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="minimind",
        description="Mini Mind v3 - tiny neural network by Templar Studios",
    )
    subparsers = parser.add_subparsers(dest="command", help="available commands")

    train_parser = subparsers.add_parser("train", help="train on a text file")
    train_parser.add_argument("file", help="text file to train on")
    train_parser.add_argument("--order", type=int, default=DEFAULT_ORDER,
                              help=f"markov order (default: {DEFAULT_ORDER})")
    train_parser.add_argument("--fresh", action="store_true",
                              help="start fresh instead of merging")

    gen_parser = subparsers.add_parser("generate", help="generate from current model")
    gen_parser.add_argument("--seed", default=None, help="seed text")
    gen_parser.add_argument("--length", type=int, default=DEFAULT_LENGTH,
                            help=f"output length (default: {DEFAULT_LENGTH})")
    gen_parser.add_argument("--temp", type=float, default=DEFAULT_TEMP,
                            help=f"temperature (default: {DEFAULT_TEMP})")

    freegen_parser = subparsers.add_parser("freegen", help="generate from merged memory")
    freegen_parser.add_argument("--seed", default=None, help="seed text")
    freegen_parser.add_argument("--length", type=int, default=DEFAULT_LENGTH,
                                help=f"output length (default: {DEFAULT_LENGTH})")
    freegen_parser.add_argument("--temp", type=float, default=1.0,
                                help="temperature (default: 1.0)")

    models_parser = subparsers.add_parser("models", help="list stored models")

    status_parser = subparsers.add_parser("status", help="show model info")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "freegen":
        cmd_freegen(args)
    elif args.command == "models":
        cmd_models(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()