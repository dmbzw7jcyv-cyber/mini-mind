#!/usr/bin/env python3
"""
Mini Mind v4 - tiny local neural network with parallel training
Part of Templar Studios | GPL v3.0

Features:
  - parallel training across multiple workers (like 5 bots reading at once)
  - lazy merge (training is fast, merge only happens on freegen)
  - auto-train on all txt files in a folder
  - persistent model memory

Pure python, no dependencies.
Works in ish, a-shell, linux, macOS.

Usage:
  minimind.py train <file.txt>                  train single file
  minimind.py autotrain                         train all .txt in current dir
  minimind.py generate --seed "The" --length 400
  minimind.py freegen --length 500
  minimind.py models                            list stored models
  minimind.py status
"""

import sys
import os
import json
import math
import random
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR = Path.home() / ".minimind_models"
DEFAULT_ORDER = 4
DEFAULT_LENGTH = 300
DEFAULT_TEMP = 0.7
DEFAULT_WORKERS = 4
PROGRESS_INTERVAL = 20000
BAR_WIDTH = 30

# ---------------------------------------------------------------------------
# Worker function (runs in separate process)
# ---------------------------------------------------------------------------

def train_chunk(args: Tuple[str, int, int]) -> Dict:
    """Train on a chunk of text. Runs in a worker process."""
    text, order, chunk_id = args

    if len(text) < order + 2:
        return {"id": chunk_id, "chains": {}, "vocab": Counter(), "chars": 0}

    raw_chains: Dict[str, Counter] = {}
    vocab: Counter = Counter()
    total = 0

    for i in range(len(text) - order):
        key = text[i:i + order]
        next_char = text[i + order]
        if key not in raw_chains:
            raw_chains[key] = Counter()
        raw_chains[key][next_char] += 1
        vocab[next_char] += 1
        total += 1

    # convert to probabilities
    chains: Dict[str, Dict[str, float]] = {}
    for key, counter in raw_chains.items():
        total_count = sum(counter.values())
        chains[key] = {c: n / total_count for c, n in counter.items()}

    return {
        "id": chunk_id,
        "chains": chains,
        "vocab": dict(vocab),
        "chars": total,
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MiniMind:
    """Character-level markov chain with parallel training."""

    def __init__(self, order: int = DEFAULT_ORDER, name: str = "current"):
        self.order = order
        self.name = name
        self.chains: Dict[str, Dict[str, float]] = {}
        self.vocab: Counter = Counter()
        self.total_chars = 0
        self.trained = False
        self.model_file = MODEL_DIR / f"{name}.json"

    def _split_text(self, text: str, workers: int) -> List[str]:
        """Split text into overlapping chunks."""
        if workers <= 1:
            return [text]

        chunk_size = len(text) // workers
        chunks = []
        for i in range(workers):
            start = i * chunk_size
            end = start + chunk_size if i < workers - 1 else len(text)
            # add overlap so chains connect across boundaries
            overlap_start = max(0, start - self.order)
            chunks.append(text[overlap_start:end])
        return chunks

    def train(self, text: str, show_progress: bool = True,
              workers: int = DEFAULT_WORKERS) -> None:
        """Train in parallel across workers."""
        if len(text) < self.order + 2:
            print("[!] text too small")
            return

        print()
        print(f"  training on {len(text):,} characters")
        print(f"  order: {self.order} | workers: {workers}")
        print()

        chunks = self._split_text(text, workers)
        tasks = [(chunk, self.order, i) for i, chunk in enumerate(chunks)]

        start_time = time.time()
        results = []
        completed = 0

        # if only one chunk, no need for process pool overhead
        if len(chunks) == 1:
            results.append(train_chunk(tasks[0]))
            completed = 1
            self._show_progress(1, 1)
        else:
            try:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futures = {ex.submit(train_chunk, t): t[2] for t in tasks}
                    for fut in as_completed(futures):
                        try:
                            results.append(fut.result())
                        except Exception as e:
                            print(f"\n  [!] worker failed: {e}")
                        completed += 1
                        self._show_progress(completed, len(chunks))
            except Exception:
                # fallback to sequential if multiprocessing fails (ish sandbox)
                print("  [!] parallel unavailable, running sequential")
                for t in tasks:
                    results.append(train_chunk(t))
                    completed += 1
                    self._show_progress(completed, len(chunks))

        print()
        print()
        print("  merging worker results...")

        # merge all worker chains
        merged_chains: Dict[str, Dict[str, float]] = {}
        merged_vocab: Counter = Counter()
        total_chars = 0

        for r in results:
            total_chars += r["chars"]
            merged_vocab.update(r["vocab"])
            for key, probs in r["chains"].items():
                if key in merged_chains:
                    # average the probabilities
                    old = merged_chains[key]
                    blended = {}
                    for char in set(old) | set(probs):
                        blended[char] = (old.get(char, 0) + probs.get(char, 0)) / 2
                    merged_chains[key] = blended
                else:
                    merged_chains[key] = probs

        self.chains = merged_chains
        self.vocab = merged_vocab
        self.total_chars = total_chars
        self.trained = True

        elapsed = time.time() - start_time

        print()
        print(f"  training complete in {elapsed:.1f}s")
        print(f"  chains: {len(self.chains):,}")
        print(f"  unique chars: {len(self.vocab)}")
        print(f"  total transitions: {self.total_chars:,}")
        print()

        self.save()

    def _show_progress(self, current: int, total: int) -> None:
        pct = (current / total) * 100 if total > 0 else 0
        filled = int(pct / 100 * BAR_WIDTH)
        bar = f"[{'█' * filled}{'░' * (BAR_WIDTH - filled)}]"
        sys.stdout.write(f"\r  {bar} {pct:5.1f}%  workers done: {current}/{total}")
        sys.stdout.flush()

    def generate(self, length: int = DEFAULT_LENGTH,
                 seed: Optional[str] = None,
                 temperature: float = DEFAULT_TEMP,
                 show_progress: bool = True) -> str:
        if not self.trained or not self.chains:
            return "(model not trained)"

        keys = list(self.chains.keys())

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
                options = {c: math.pow(p, 1.0 / temperature) for c, p in options.items()}
                total = sum(options.values())
                if total > 0:
                    options = {c: p / total for c, p in options.items()}

            rand = random.random()
            cum = 0
            chosen = None
            for char, prob in sorted(options.items(), key=lambda x: -x[1]):
                cum += prob
                if rand <= cum:
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

    def save(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "order": self.order,
            "chains": self.chains,
            "vocab": dict(self.vocab.most_common(5000)),
            "total_chars": self.total_chars,
            "name": self.name,
        }
        with open(self.model_file, "w") as f:
            json.dump(data, f)

    def load(self, name: Optional[str] = None) -> bool:
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

    def stats(self) -> str:
        if not self.trained:
            return "untrained"
        avg = sum(len(v) for v in self.chains.values()) / max(1, len(self.chains))
        return (f"chains: {len(self.chains):,} | "
                f"chars: {len(self.vocab)} | "
                f"avg: {avg:.1f} | "
                f"trained on: {self.total_chars:,}")


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

def list_models() -> List[str]:
    if not MODEL_DIR.exists():
        return []
    return sorted(f.stem for f in MODEL_DIR.glob("*.json"))


def clean_gutenberg(text: str) -> str:
    """Strip project gutenberg boilerplate."""
    lines = text.split("\n")
    cleaned = []
    skip_patterns = [
        "gutenberg", "ebook", "produced by", "distributed proofread",
        "updated editions", "www.gutenberg", "project gutenberg",
        "*** start of", "*** end of",
    ]
    for line in lines:
        lower = line.lower()
        if any(p in lower for p in skip_patterns):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def merge_all_models() -> MiniMind:
    """Merge all stored models into one (lazy merge - only on freegen)."""
    models = list_models()
    merged = MiniMind(name="merged")
    first = True

    for name in models:
        m = MiniMind()
        if not m.load(name):
            continue

        if first:
            merged.chains = {k: dict(v) for k, v in m.chains.items()}
            merged.vocab = Counter(m.vocab)
            merged.total_chars = m.total_chars
            merged.order = m.order
            first = False
        else:
            for key, probs in m.chains.items():
                if key in merged.chains:
                    old = merged.chains[key]
                    blended = {}
                    for char in set(old) | set(probs):
                        blended[char] = (old.get(char, 0) + probs.get(char, 0)) / 2
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
    text = clean_gutenberg(text)

    if len(text) < 100:
        print(f"[!] file too small: {len(text)} chars")
        sys.exit(1)

    model_name = filepath.stem.lower().replace(" ", "_")
    model = MiniMind(order=args.order, name=model_name)
    model.train(text, workers=args.workers)

    print(f"[+] saved as model: {model_name}")
    print(f"[*] generate with: minimind.py generate --seed \"...\"")
    print(f"[*] freegen to merge all models")


def cmd_autotrain(args) -> None:
    """Train on every .txt file in the folder."""
    folder = Path(args.folder) if args.folder else Path.cwd()
    files = [f for f in folder.glob("*.txt")
             if not f.stem.startswith("clean_")]

    if not files:
        print(f"[!] no .txt files in {folder}")
        sys.exit(1)

    print(f"[*] found {len(files)} text files in {folder}")
    print()

    for i, filepath in enumerate(files, 1):
        print(f"=== [{i}/{len(files)}] {filepath.name} ===")
        try:
            text = filepath.read_text(encoding="utf-8", errors="replace")
            text = clean_gutenberg(text)

            if len(text) < 100:
                print(f"  [!] too small, skipping")
                continue

            model_name = filepath.stem.lower().replace(" ", "_")
            model = MiniMind(order=args.order, name=model_name)
            model.train(text, workers=args.workers)
        except Exception as e:
            print(f"  [!] failed: {e}")

    print()
    print(f"[+] autotrain complete. {len(files)} models saved.")
    print(f"[*] run freegen to generate from all of them")


def cmd_generate(args) -> None:
    models = list_models()
    if not models:
        print("[!] no models found")
        sys.exit(1)

    name = args.model or ("current" if "current" in models else models[-1])
    model = MiniMind()
    if not model.load(name):
        print(f"[!] could not load model: {name}")
        sys.exit(1)

    print(f"[*] model: {model.name}")
    print(f"[*] {model.stats()}")
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
    print("[*] merging all models (lazy merge)...")
    start = time.time()
    model = merge_all_models()
    elapsed = time.time() - start

    if not model.trained:
        print("[!] no models to merge")
        sys.exit(1)

    print(f"[*] merged {len(list_models())} model(s) in {elapsed:.1f}s")
    print(f"[*] {model.stats()}")
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
                try:
                    size_kb = m.model_file.stat().st_size / 1024
                except Exception:
                    size_kb = 0
                print(f"  {name:<25} {m.stats()}  ({size_kb:.0f} KB)")
    print("─" * 60)
    print()


def cmd_status(args) -> None:
    models = list_models()
    if not models:
        print("[*] no models yet")
        return
    print(f"[*] {len(models)} model(s) stored")
    for name in models:
        m = MiniMind()
        if m.load(name):
            print(f"  {name}: {m.stats()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="minimind",
        description="Mini Mind v4 - parallel training by Templar Studios",
    )
    sub = parser.add_subparsers(dest="command", help="commands")

    t = sub.add_parser("train", help="train a single file")
    t.add_argument("file")
    t.add_argument("--order", type=int, default=DEFAULT_ORDER)
    t.add_argument("--workers", type=int, default=DEFAULT_WORKERS)

    at = sub.add_parser("autotrain", help="train all .txt files in folder")
    at.add_argument("folder", nargs="?", default=None)
    at.add_argument("--order", type=int, default=DEFAULT_ORDER)
    at.add_argument("--workers", type=int, default=DEFAULT_WORKERS)

    g = sub.add_parser("generate", help="generate from one model")
    g.add_argument("--model", default=None)
    g.add_argument("--seed", default=None)
    g.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    g.add_argument("--temp", type=float, default=DEFAULT_TEMP)

    fg = sub.add_parser("freegen", help="generate from all merged models")
    fg.add_argument("--seed", default=None)
    fg.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    fg.add_argument("--temp", type=float, default=1.0)

    sub.add_parser("models", help="list models")
    sub.add_parser("status", help="show stats")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "train":
        cmd_train(args)
    elif args.command == "autotrain":
        cmd_autotrain(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "freegen":
        cmd_freegen(args)
    elif args.command == "models":
        cmd_models(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    mp.freeze_support()  # for windows compatibility
    main()