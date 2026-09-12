#!/usr/bin/env python3
"""
GutenbergBot v4 - round-based auto-fetch, train, and clean
Part of Templar Studios | GPL v3.0

Each round fetches a NEW batch of books from Gutendex, trains them,
deletes the txt files, then loops. Run for N rounds or forever.

Usage:
  python gutenberg_bot.py --rounds 10 --batch 100     # 10 rounds
  python gutenberg_bot.py --rounds 0 --batch 100      # infinite
  python gutenberg_bot.py --batch 50                  # 1 round default
  python gutenberg_bot.py --keep-txt
"""

import sys
import os
import json
import time
import inspect
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, List, Dict

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

try:
    from minimind import MiniMind, clean_gutenberg
except ImportError:
    print("[!] could not import minimind.py")
    print(f"    looking in: {HERE}")
    print("    make sure gutenberg_bot.py is in the same folder as minimind.py")
    sys.exit(1)

BOT_DIR = Path.home() / "gutenberg_bot"
BOOKS_DIR = BOT_DIR / "books"
DONE_FILE = BOT_DIR / "done.json"
STATE_FILE = BOT_DIR / "state.json"

GUTENDEX = "https://gutendex.com/books"
GUTENBERG_CACHE = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
GUTENBERG_FILES = "https://www.gutenberg.org/files/{id}/{id}-0.txt"

TIMEOUT = 60
USER_AGENT = "GutenbergBot/4.0 (Templar Studios; educational)"
ROUND_PAUSE = 3


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


def fetch_next_batch(count: int, state: Dict) -> List[int]:
    """
    Fetch the next `count` book IDs from Gutendex, advancing the page pointer.
    Returns a list of new (never-seen) IDs and updates state['page'].
    """
    ids: List[int] = []
    seen: set = set(state.get("seen_ids", []))
    page = state.get("page", 1)

    while len(ids) < count:
        try:
            data = fetch_json(f"{GUTENDEX}?page={page}")
        except Exception as e:
            print(f"[!] gutendex page {page} error: {e}")
            break

        results = data.get("results", [])
        if not results:
            print(f"[*] gutendex ran out at page {page}")
            break

        for book in results:
            bid = book["id"]
            if bid in seen:
                continue
            ids.append(bid)
            if len(ids) >= count:
                break

        page += 1
        if page > 1000:
            break

    state["page"] = page
    # record every id we handed out
    seen.update(ids)
    state["seen_ids"] = list(seen)

    return ids


def download_book(book_id: int) -> Optional[str]:
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
# State / progress
# ───────────────────────────────────────

def load_json(path: Path, default: Dict) -> Dict:
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return default
    return default


def save_json(path: Path, data: Dict) -> None:
    BOT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ───────────────────────────────────────
# Training
# ───────────────────────────────────────

def make_model(order: int, name: str):
    sig = inspect.signature(MiniMind.__init__)
    if "order" in sig.parameters:
        return MiniMind(order=order, name=name)
    else:
        return MiniMind(max_order=order, name=name)


# ───────────────────────────────────────
# Phases
# ───────────────────────────────────────

def phase_download(ids: List[int], done: Dict,
                   round_num: int, total_rounds: str) -> List[Path]:
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    for i, book_id in enumerate(ids, 1):
        key = str(book_id)
        if key in done:
            continue

        name = f"gutenberg_{book_id}"
        txt_path = BOOKS_DIR / f"{name}.txt"

        if txt_path.exists() and txt_path.stat().st_size > 500:
            print(f"    [R{round_num}/{total_rounds} {i}/{len(ids)}] "
                  f"id={book_id} — already on disk")
            saved.append(txt_path)
            continue

        print(f"    [R{round_num}/{total_rounds} {i}/{len(ids)}] "
              f"id={book_id} — downloading...")

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
                keep_txt: bool, round_num: int, total_rounds: str) -> int:
    trained = 0

    for i, txt_path in enumerate(paths, 1):
        name = txt_path.stem
        book_id_str = name.replace("gutenberg_", "")

        if book_id_str in done:
            print(f"    [R{round_num}/{total_rounds} {i}/{len(paths)}] "
                  f"{name} — already trained, skip")
            if not keep_txt:
                try:
                    txt_path.unlink()
                except Exception:
                    pass
            continue

        try:
            text = txt_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"    [R{round_num}/{total_rounds} {i}/{len(paths)}] "
                  f"{name} — read failed: {e}")
            continue

        if len(text) < 500:
            try:
                txt_path.unlink()
            except Exception:
                pass
            continue

        print(f"    [R{round_num}/{total_rounds} {i}/{len(paths)}] "
              f"{name} — training ({len(text):,} chars)...")

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
        save_json(DONE_FILE, done)
        trained += 1
        print(f"        ✓ trained")

    return trained


# ───────────────────────────────────────
# Main
# ───────────────────────────────────────

def run_round(round_num: int, total_rounds: str, batch_size: int,
              order: int, keep_txt: bool, state: Dict, done: Dict) -> Dict:
    """Run a single round. Returns stats dict."""
    stats = {"round": round_num, "downloaded": 0,
             "trained": 0, "ids": 0}

    round_start = time.time()
    batch_id = f"R{round_num}/{total_rounds}"

    print()
    print("╔" + "═" * 62 + "╗")
    print(f"║  ROUND {batch_id} — fetching {batch_size} new books"
          + " " * (62 - 30 - len(batch_id) - len(str(batch_size))) + "║")
    print("╚" + "═" * 62 + "╝")
    print()

    # ─── fetch new IDs ───
    print(f"[{batch_id}] fetching new book IDs from gutendex...")
    ids = fetch_next_batch(batch_size, state)
    save_json(STATE_FILE, state)

    if not ids:
        print(f"[{batch_id}] no new books available — stopping")
        stats["ids"] = 0
        return stats

    stats["ids"] = len(ids)
    print(f"[{batch_id}] got {len(ids)} new book IDs")
    print()

    # ─── phase 1: download ───
    print(f"[{batch_id}] PHASE 1/3 → DOWNLOAD")
    dl_start = time.time()
    paths = phase_download(ids, done, round_num, total_rounds)
    dl_elapsed = time.time() - dl_start
    stats["downloaded"] = len(paths)
    print(f"[{batch_id}] downloaded {len(paths)} book(s) in {dl_elapsed/60:.1f}m")
    print()

    if not paths:
        print(f"[{batch_id}] nothing to train, moving on")
        return stats

    # ─── phase 2: train ───
    print(f"[{batch_id}] PHASE 2/3 → TRAIN")
    tr_start = time.time()
    trained = phase_train(paths, order, done, keep_txt,
                          round_num, total_rounds)
    tr_elapsed = time.time() - tr_start
    stats["trained"] = trained
    print(f"[{batch_id}] trained {trained} book(s) in {tr_elapsed/60:.1f}m")
    print()

    # ─── phase 3: clean ───
    if not keep_txt:
        print(f"[{batch_id}] PHASE 3/3 → CLEAN")
        removed = 0
        for p in paths:
            if p.exists():
                try:
                    p.unlink()
                    removed += 1
                except Exception:
                    pass
        print(f"[{batch_id}] removed {removed} leftover txt file(s)")
        print()

    elapsed = time.time() - round_start
    print(f"[{batch_id}] round complete in {elapsed/60:.1f}m")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gutenberg_bot",
        description="round-based auto-fetch, train, and clean Gutenberg books",
    )
    parser.add_argument("--rounds", type=int, default=1,
                        help="how many rounds to run (0 = infinite)")
    parser.add_argument("--batch", type=int, default=100,
                        help="books per round (default: 100)")
    parser.add_argument("--order", type=int, default=5,
                        help="minimind order (default: 5)")
    parser.add_argument("--keep-txt", action="store_true",
                        help="keep .txt files after training")

    args = parser.parse_args()

    BOT_DIR.mkdir(parents=True, exist_ok=True)
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)

    # ─── load persistent state ───
    state = load_json(STATE_FILE, {"page": 1, "seen_ids": []})
    done = load_json(DONE_FILE, {})

    infinite = args.rounds == 0
    total_display = "∞" if infinite else str(args.rounds)

    print(f"[*] rounds: {'infinite' if infinite else args.rounds}")
    print(f"[*] batch size: {args.batch} books per round")
    print(f"[*] minimind order: {args.order}")
    print(f"[*] already trained: {len(done)} books")
    print(f"[*] gutendex page pointer: {state.get('page', 1)}")
    print()
    print("[*] ctrl+c anytime to stop — progress is saved")
    print()

    overall_start = time.time()
    total_trained = 0
    total_downloaded = 0
    round_num = 0

    try:
        while True:
            round_num += 1

            if not infinite and round_num > args.rounds:
                break

            stats = run_round(round_num, total_display, args.batch,
                              args.order, args.keep_txt, state, done)

            total_trained += stats["trained"]
            total_downloaded += stats["downloaded"]

            # stop if no new books
            if stats["ids"] == 0:
                print("[*] no new book IDs available — ending loop")
                break

            if not infinite and round_num >= args.rounds:
                break

            print(f"[*] waiting {ROUND_PAUSE}s before next round...")
            print()
            time.sleep(ROUND_PAUSE)

    except KeyboardInterrupt:
        print()
        print("[!] stopped by user")

    # ─── summary ───
    overall = time.time() - overall_start
    print()
    print("╔" + "═" * 62 + "╗")
    print("║                    SESSION COMPLETE                         ║")
    print("╚" + "═" * 62 + "╝")
    print()
    print(f"  rounds run:        {round_num}")
    print(f"  total downloaded:  {total_downloaded}")
    print(f"  total trained:     {total_trained}")
    print(f"  session time:      {overall/60:.1f} minutes")
    print(f"  total in database: {len(done)} books")
    print()
    print("  generate with:")
    print("    py minimind.py freegen --length 400")


if __name__ == "__main__":
    main()