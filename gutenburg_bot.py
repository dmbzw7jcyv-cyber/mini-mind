#!/usr/bin/env python3
"""
GutenbergBot v2 - batched auto-fetch, train, and clean for MiniMind
Part of Templar Studios | GPL v3.0

Phase-per-batch model:
  1. download whole batch (400 books) to disk
  2. train on every book in the batch
  3. delete txt files after training
  4. fetch next batch, repeat

Lower disk usage, faster overall, resumable.

Usage:
  python gutenberg_bot.py                        # top 400, default batch
  python gutenberg_bot.py --total 2000           # fetch 2000, in batches
  python gutenberg_bot.py --total 400 --batch 100
  python gutenberg_bot.py --ids 1342 84 11
  python gutenberg_bot.py --keep-txt
"""

import sys
import os
import re
import json
import time
import inspect
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, List, Dict

# ─── find minimind.py in the same folder ───
HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

try:
    from minimind import MiniMind, clean_gutenberg
except ImportError:
    print("[!] could not import minimind.py")
    print(f"    looking in: {HERE}")
    print("    make sure gutenberg_bot.py is in the same folder as minimind.py")
    sys.exit(1)

# ─── paths ───
BOT_DIR = Path.home() / "gutenberg_bot"
BOOKS_DIR = BOT_DIR / "books"
DONE_FILE = BOT_DIR / "done.json"

# ─── endpoints ───
GUTENDEX = "https://gutendex.com/books"
GUTENBERG_CACHE = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
GUTENBERG_FILES = "https://www.gutenberg.org/files/{id}/{id}-0.txt"

TIMEOUT = 60
USER_AGENT = "GutenbergBot/2.0 (Templar Studios; educational)"


# ───────────────────────────────────────
# Networking
# ───────────────────────────────────────

def fetch_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read().decode("utf-8", errors="replace")


def get_book_ids(count: int = 100) -> List[int]:
    """Fetch top N book IDs from Gutendex by popularity."""
    ids: List[int] = []
    page = 1

    while len(ids) < count:
        try:
            data = fetch_json(f"{GUTENDEX}?page={page}")
        except Exception as e:
            print(f"[!] gutendex page {page} error: {e}")
            break

        results = data.get("results", [])
        if not results:
            break

        for book in results:
            ids.append(book["id"])
            if len(ids) >= count:
                break

        page += 1
        if page > 100:
            break

        if page % 5 == 0:
            print(f"    fetched {len(ids)} ids so far...")

    return ids[:count]


def download_book(book_id: int) -> Optional[str]:
    """Try both gutenberg URL formats. Return raw text or None."""
    urls = [
        GUTENBERG_CACHE.format(id=book_id),
        GUTENBERG_FILES.format(id=book_id),
    ]

    for url in urls:
        try:
            text = fetch_text(url)
            if len(text) > 1000:
                return text
        except urllib.error.HTTPError:
            continue
        except Exception:
            continue

    return None


# ───────────────────────────────────────
# Progress tracking
# ───────────────────────────────────────

def load_done() -> Dict:
    if DONE_FILE.exists():
        try:
            with open(DONE_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_done(done: Dict) -> None:
    BOT_DIR.mkdir(parents=True, exist_ok=True)
    with open(DONE_FILE, "w") as f:
        json.dump(done, f, indent=2)


# ───────────────────────────────────────
# Training
# ───────────────────────────────────────

def make_model(order: int, name: str):
    """Create a MiniMind, compatible with v7.x and v8.x."""
    sig = inspect.signature(MiniMind.__init__)
    if "order" in sig.parameters:
        return MiniMind(order=order, name=name)
    else:
        return MiniMind(max_order=order, name=name)


# ───────────────────────────────────────
# Phases
# ───────────────────────────────────────

def phase_download(ids: List[int], done: Dict) -> List[Path]:
    """Download all books in a batch that aren't already done. Return saved paths."""
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    for i, book_id in enumerate(ids, 1):
        key = str(book_id)
        if key in done:
            continue

        name = f"gutenberg_{book_id}"
        txt_path = BOOKS_DIR / f"{name}.txt"

        if txt_path.exists() and txt_path.stat().st_size > 500:
            print(f"    [{i}/{len(ids)}] id={book_id} — already on disk")
            saved.append(txt_path)
            continue

        print(f"    [{i}/{len(ids)}] id={book_id} — downloading...")

        try:
            text = download_book(book_id)
        except KeyboardInterrupt:
            print("\n[!] interrupted during download")
            return saved
        except Exception as e:
            print(f"        [!] download error: {e}")
            continue

        if not text:
            print(f"        [!] no text available")
            continue

        text = clean_gutenberg(text)

        if len(text) < 500:
            print(f"        [!] too short ({len(text)} chars)")
            continue

        try:
            txt_path.write_text(text, encoding="utf-8")
            saved.append(txt_path)
            print(f"        ✓ saved {len(text):,} chars")
        except Exception as e:
            print(f"        [!] write failed: {e}")

    return saved


def phase_train(paths: List[Path], order: int, done: Dict,
                keep_txt: bool) -> int:
    """Train on all downloaded books. Delete txt after unless keep_txt."""
    trained = 0

    for i, txt_path in enumerate(paths, 1):
        name = txt_path.stem
        book_id_str = name.replace("gutenberg_", "")

        if book_id_str in done:
            print(f"    [{i}/{len(paths)}] {name} — already trained, skip")
            if not keep_txt:
                try:
                    txt_path.unlink()
                except Exception:
                    pass
            continue

        try:
            text = txt_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"    [{i}/{len(paths)}] {name} — read failed: {e}")
            continue

        if len(text) < 500:
            print(f"    [{i}/{len(paths)}] {name} — too short, skip")
            try:
                txt_path.unlink()
            except Exception:
                pass
            continue

        print(f"    [{i}/{len(paths)}] {name} — training ({len(text):,} chars)...")

        try:
            model = make_model(order=order, name=name)
            ok = model.train(text)
        except KeyboardInterrupt:
            print("\n[!] interrupted during training")
            return trained
        except Exception as e:
            print(f"        [!] training error: {e}")
            continue

        if not ok:
            print(f"        [!] training returned failure")
            continue

        if not keep_txt:
            try:
                txt_path.unlink()
            except Exception:
                pass

        done[book_id_str] = {
            "name": name,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "chars": len(text),
        }
        save_done(done)
        trained += 1
        print(f"        ✓ trained")

    return trained


# ───────────────────────────────────────
# Main
# ───────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gutenberg_bot",
        description="batched auto-fetch, train, and clean Gutenberg books",
    )
    parser.add_argument("--total", type=int, default=400,
                        help="total books to process across all batches (default: 400)")
    parser.add_argument("--batch", type=int, default=400,
                        help="books per batch (default: 400)")
    parser.add_argument("--order", type=int, default=5,
                        help="minimind order (default: 5)")
    parser.add_argument("--keep-txt", action="store_true",
                        help="keep .txt files after training")
    parser.add_argument("--ids", type=int, nargs="*",
                        help="specific book IDs instead of fetching top")
    parser.add_argument("--start-at", type=int, default=0,
                        help="skip first N books (for resuming)")

    args = parser.parse_args()

    BOT_DIR.mkdir(parents=True, exist_ok=True)
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)

    # ─── get list of book IDs ───
    if args.ids:
        ids = args.ids
        print(f"[*] using {len(ids)} specified book IDs")
    else:
        print(f"[*] fetching top {args.total} books from Gutendex...")
        ids = get_book_ids(args.total)
        print(f"[*] got {len(ids)} book IDs")

    if args.start_at > 0:
        ids = ids[args.start_at:]
        print(f"[*] starting at index {args.start_at}, {len(ids)} remaining")

    if not ids:
        print("[!] no book IDs to process")
        sys.exit(1)

    # ─── split into batches ───
    batch_size = max(1, args.batch)
    batches = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    print(f"[*] {len(batches)} batch(es) of up to {batch_size} books")
    print()

    done = load_done()
    print(f"[*] {len(done)} books already trained — will be skipped")
    print()

    total_trained = 0
    total_failed = 0
    overall_start = time.time()

    for bi, batch in enumerate(batches, 1):
        print("=" * 60)
        print(f"  BATCH {bi}/{len(batches)}  ({len(batch)} books)")
        print("=" * 60)
        print()

        # ─── phase 1: download ───
        print(f"[batch {bi}] phase 1: download")
        dl_start = time.time()
        paths = phase_download(batch, done)
        dl_elapsed = time.time() - dl_start
        print(f"[batch {bi}] downloaded {len(paths)} books in {dl_elapsed/60:.1f}m")
        print()

        if not paths:
            print(f"[batch {bi}] nothing to train, skipping")
            print()
            continue

        # ─── phase 2: train ───
        print(f"[batch {bi}] phase 2: train")
        tr_start = time.time()
        trained = phase_train(paths, args.order, done, args.keep_txt)
        tr_elapsed = time.time() - tr_start
        print(f"[batch {bi}] trained {trained} books in {tr_elapsed/60:.1f}m")
        print()

        # ─── phase 3: ensure txt files gone ───
        if not args.keep_txt:
            print(f"[batch {bi}] phase 3: clean")
            removed = 0
            for p in paths:
                if p.exists():
                    try:
                        p.unlink()
                        removed += 1
                    except Exception:
                        pass
            print(f"[batch {bi}] removed {removed} leftover txt file(s)")
            print()

        total_trained += trained
        total_failed += (len(batch) - trained)

    # ─── summary ───
    overall = time.time() - overall_start
    print()
    print("=" * 60)
    print(f"  DONE")
    print(f"  total trained:  {total_trained}")
    print(f"  total skipped:  {len(done) - total_trained if total_trained else 0}")
    print(f"  total failed:   {total_failed}")
    print(f"  overall time:   {overall/60:.1f} minutes")
    print("=" * 60)
    print()
    print("generate from everything with:")
    print("  py minimind.py freegen --length 400")


if __name__ == "__main__":
    main()