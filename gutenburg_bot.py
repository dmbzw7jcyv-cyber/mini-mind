#!/usr/bin/env python3
"""
GutenbergBot v4.1 - round-based auto-fetch, train, and clean
Part of Templar Studios | GPL v3.0

FIXES over v4:
  - removed seen_ids tracking (was blocking infinite mode)
  - uses only done.json for skip logic
  - auto-resets page pointer if gutendex returns empty
  - small delay between gutendex requests (no rate limit)
  - --reset-state flag
  - better diagnostics

Usage:
  python gutenberg_bot.py --rounds 10 --batch 100
  python gutenberg_bot.py --rounds 0 --batch 100    # infinite
  python gutenberg_bot.py --reset-state             # clear state, keep models
  python gutenberg_bot.py --diagnostic              # show status only
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
    sys.exit(1)

BOT_DIR = Path.home() / "gutenberg_bot"
BOOKS_DIR = BOT_DIR / "books"
DONE_FILE = BOT_DIR / "done.json"
STATE_FILE = BOT_DIR / "state.json"

GUTENDEX = "https://gutendex.com/books"
GUTENBERG_CACHE = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
GUTENBERG_FILES = "https://www.gutenberg.org/files/{id}/{id}-0.txt"

TIMEOUT = 60
USER_AGENT = "GutenbergBot/4.1 (Templar Studios)"
ROUND_PAUSE = 2
GUTENDEX_DELAY = 0.5
MAX_PAGES = 5000  # gutendex has ~2200 pages of results


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
    Fetch the next `count` book IDs starting from state['page'].
    Advances the pointer. No seen_ids filtering — done.json handles skips.
    """
    ids: List[int] = []
    page = state.get("page", 1)

    if page < 1:
        page = 1

    pages_fetched = 0

    while len(ids) < count:
        try:
            data = fetch_json(f"{GUTENDEX}?page={page}")
        except urllib.error.HTTPError as e:
            print(f"    [!] gutendex page {page} HTTP {e.code}")
            if e.code == 404:
                # past the end — reset
                print(f"    [*] page {page} not found, resetting to page 1")
                page = 1
                state["page"] = 1
                time.sleep(GUTENDEX_DELAY)
                # try once more from page 1
                try:
                    data = fetch_json(f"{GUTENDEX}?page=1")
                except Exception as e2:
                    print(f"    [!] even page 1 failed: {e2}")
                    break
                page = 2
            else:
                time.sleep(GUTENDEX_DELAY * 2)
                break
        except Exception as e:
            print(f"    [!] gutendex fetch error: {e}")
            break

        results = data.get("results", [])
        if not results:
            print(f"    [*] no results at page {page}")
            if page > 1:
                print(f"    [*] resetting to page 1")
                page = 1
                state["page"] = 1
                time.sleep(GUTENDEX_DELAY)
                # one more attempt from page 1
                try:
                    data = fetch_json(f"{GUTENDEX}?page=1")
                except Exception as e:
                    print(f"    [!] page 1 failed: {e}")
                    break
                results = data.get("results", [])
                if not results:
                    print(f"    [!] gutendex appears empty")
                    break
                page = 2
            else:
                print(f"    [!] gutendex has no results — check connection")
                break

        for book in results:
            ids.append(book["id"])
            if len(ids) >= count:
                break

        page += 1
        pages_fetched += 1
        time.sleep(GUTENDEX_DELAY)

        if page > MAX_PAGES:
            print(f"    [*] hit max pages")
            break

        # safety: if no new ids added in several pages, give up
        if pages_fetched > 50:
            print(f"    [!] fetched 50 pages without reaching target")
            break

    state["page"] = page
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
    already_done = 0

    for i, book_id in enumerate(ids, 1):
        key = str(book_id)
        if key in done:
            already_done += 1
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

    if already_done:
        print(f"    [*] skipped {already_done} already-trained book(s)")

    return saved


def phase_train(paths: List[Path], order: int, done: Dict,
                keep_txt: bool, round_num: int, total_rounds: str) -> int:
    trained = 0

    for i, txt_path in enumerate(paths, 1):
        name = txt_path.stem
        book_id_str = name.replace("gutenberg_", "")

        if book_id_str in done:
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
# Round
# ───────────────────────────────────────

def run_round(round_num: int, total_rounds: str, batch_size: int,
              order: int, keep_txt: bool, state: Dict, done: Dict) -> Dict:
    stats = {"round": round_num, "downloaded": 0,
             "trained": 0, "ids": 0, "skipped": 0}

    round_start = time.time()
    batch_id = f"R{round_num}/{total_rounds}"

    print()
    print("╔" + "═" * 62 + "╗")
    header = f"  ROUND {batch_id} — fetching {batch_size} books from page {state.get('page', 1)}"
    print("║" + header.ljust(62) + "║")
    print("╚" + "═" * 62 + "╝")
    print()

    # ─── fetch IDs ───
    print(f"[{batch_id}] fetching book IDs...")
    ids = fetch_next_batch(batch_size, state)
    save_json(STATE_FILE, state)

    if not ids:
        print(f"[{batch_id}] no book IDs returned — cannot proceed")
        return stats

    stats["ids"] = len(ids)
    print(f"[{batch_id}] got {len(ids)} book IDs (page pointer now {state['page']})")

    # count how many are already done
    already = sum(1 for i in ids if str(i) in done)
    new_ids = len(ids) - already
    print(f"[{batch_id}] {new_ids} new, {already} already trained")
    print()

    if new_ids == 0:
        # all books in this batch already trained, return so we advance
        print(f"[{batch_id}] all books already trained, advancing to next round")
        stats["skipped"] = already
        return stats

    # ─── phase 1: download ───
    print(f"[{batch_id}] PHASE 1/3 → DOWNLOAD")
    dl_start = time.time()
    paths = phase_download(ids, done, round_num, total_rounds)
    dl_elapsed = time.time() - dl_start
    stats["downloaded"] = len(paths)
    print(f"[{batch_id}] downloaded {len(paths)} book(s) in {dl_elapsed/60:.1f}m")
    print()

    if not paths:
        print(f"[{batch_id}] nothing to train")
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


# ───────────────────────────────────────
# Main
# ───────────────────────────────────────

def cmd_diagnostic() -> None:
    """Show current state without doing any work."""
    state = load_json(STATE_FILE, {"page": 1})
    done = load_json(DONE_FILE, {})

    print()
    print("=" * 60)
    print("  GUTENBERG BOT — DIAGNOSTIC")
    print("=" * 60)
    print(f"  bot folder:        {BOT_DIR}")
    print(f"  state file:        {STATE_FILE}")
    print(f"  state exists:      {STATE_FILE.exists()}")
    print(f"  done file:         {DONE_FILE}")
    print(f"  done exists:       {DONE_FILE.exists()}")
    print()
    print(f"  current page ptr:  {state.get('page', 1)}")
    print(f"  books trained:     {len(done)}")
    print()

    # model count
    model_dir = Path.home() / ".minimind_models"
    if model_dir.exists():
        jsons = list(model_dir.glob("gutenberg_*.json"))
        print(f"  gutenberg models:  {len(jsons)}")

    # left in books folder
    if BOOKS_DIR.exists():
        txts = list(BOOKS_DIR.glob("*.txt"))
        print(f"  leftover txts:     {len(txts)}")
    print()

    # try a fetch to see if gutendex responds
    print(f"  testing gutendex page 1...")
    try:
        data = fetch_json(f"{GUTENDEX}?page=1")
        count = data.get("count", "?")
        results = len(data.get("results", []))
        print(f"  gutendex responds: yes")
        print(f"  total books avail: {count}")
        print(f"  results per page:  {results}")
    except Exception as e:
        print(f"  gutendex responds: NO ({e})")
    print()


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
    parser.add_argument("--reset-state", action="store_true",
                        help="reset page pointer (keeps models)")
    parser.add_argument("--diagnostic", action="store_true",
                        help="show status only, do nothing")

    args = parser.parse_args()

    BOT_DIR.mkdir(parents=True, exist_ok=True)
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)

    if args.diagnostic:
        cmd_diagnostic()
        return

    if args.reset_state:
        save_json(STATE_FILE, {"page": 1})
        print("[*] state reset — page pointer back to 1")
        print()

    state = load_json(STATE_FILE, {"page": 1})
    done = load_json(DONE_FILE, {})

    infinite = args.rounds == 0
    total_display = "∞" if infinite else str(args.rounds)

    print(f"[*] rounds: {'infinite' if infinite else args.rounds}")
    print(f"[*] batch size: {args.batch}")
    print(f"[*] order: {args.order}")
    print(f"[*] already trained: {len(done)} book(s)")
    print(f"[*] gutendex page pointer: {state.get('page', 1)}")
    print()
    print("[*] ctrl+c anytime to stop — progress is saved")
    print()

    overall_start = time.time()
    total_trained = 0
    total_downloaded = 0
    round_num = 0
    consecutive_empty = 0

    try:
        while True:
            round_num += 1

            if not infinite and round_num > args.rounds:
                break

            stats = run_round(round_num, total_display, args.batch,
                              args.order, args.keep_txt, state, done)

            total_trained += stats["trained"]
            total_downloaded += stats["downloaded"]

            # track consecutive empty rounds
            if stats["trained"] == 0 and stats["downloaded"] == 0:
                consecutive_empty += 1
            else:
                consecutive_empty = 0

            # stop if gutendex returned nothing at all
            if stats["ids"] == 0:
                print(f"[*] gutendex returned no IDs — ending loop")
                break

            # if we've had 3 empty rounds in a row, stop
            if consecutive_empty >= 3:
                print(f"[*] 3 consecutive empty rounds — ending loop")
                break

            if not infinite and round_num >= args.rounds:
                break

            print(f"[*] waiting {ROUND_PAUSE}s before next round...")
            print()
            time.sleep(ROUND_PAUSE)

    except KeyboardInterrupt:
        print()
        print("[!] stopped by user")

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