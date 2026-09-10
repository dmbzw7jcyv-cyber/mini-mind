#!/usr/bin/env python3
"""
Mini Mind v7 - word-level context model with semantic topic state
Part of Templar Studios | GPL v3.0

New over v6:
  - Semantic categories (CHARACTER, ACTION, OBJECT, ABSTRACT, QUALITY, PLACE, TIME)
  - Per-word category distributions learned during training
  - Topic state that shifts and decays as generation proceeds
  - Category-aware candidate reranking
  - Fabrication triggers on low-confidence, not just <UNK>
  - Fabricated words get categorized too

Pure python, no dependencies. Works in ish, a-shell, linux, macOS.
"""

import sys
import os
import json
import math
import re
import random
import time
import hashlib
import argparse
import multiprocessing as mp
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR = Path.home() / ".minimind_models_v7"
MANIFEST_FILE = MODEL_DIR / "_manifest.json"

DEFAULT_MAX_ORDER = 7
DEFAULT_MIN_ORDER = 1
DEFAULT_LENGTH = 300
DEFAULT_TEMP = 0.8

START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK = "<UNK>"

PUNCT_TOKENS = {".", ",", "!", "?", ";", ":", "\u2014", "\u2013", ")", "]", "}"}
CHAR_ORDER = 4
REP_WINDOW = 30
REP_PENALTY_BASE = 0.55

# fabrication
FABRICATE_MIN_PROB = 0.15
FABRICATE_CHANCE = 0.35
FABRICATE_MAX_LEN = 14

# topic state
TOPIC_DECAY = 0.90
TOPIC_BOOST = 0.7
TOPIC_MIN_COUNT = 5

CATEGORIES = ["CHARACTER", "ACTION", "OBJECT", "ABSTRACT",
              "QUALITY", "PLACE", "TIME", "OTHER"]

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text)
    return re.findall(
        r"<START>|<END>|[A-Za-zÀ-ÿ'-]+|[0-9]+|[.,!?;:\"()\[\]{}\u2014\u2013]",
        text
    )


def sentence_split(text: str) -> List[List[str]]:
    raw = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for s in raw:
        toks = tokenize(s)
        if len(toks) >= 3:
            out.append([START_TOKEN] + toks + [END_TOKEN])
    return out


# ---------------------------------------------------------------------------
# POS
# ---------------------------------------------------------------------------

DET = {"the", "a", "an", "this", "that", "these", "those", "my", "your",
       "his", "her", "its", "our", "their", "some", "any"}
PRON = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
        "us", "them", "who", "whom", "which", "what"}
PREP = {"of", "in", "on", "at", "by", "for", "with", "to", "from", "into",
        "through", "during", "before", "after", "above", "below", "between",
        "under", "over", "about", "against"}
CONJ = {"and", "or", "but", "nor", "for", "yet", "so", "because",
        "although", "while", "if", "when", "unless"}
AUX = {"is", "are", "was", "were", "be", "been", "being", "has", "have",
       "had", "do", "does", "did", "will", "would", "shall", "should",
       "can", "could", "may", "might", "must"}


def pos_tag(word: str) -> str:
    w = word.lower()
    if w in DET: return "DET"
    if w in PRON: return "PRON"
    if w in PREP: return "PREP"
    if w in CONJ: return "CONJ"
    if w in AUX: return "AUX"
    if w in PUNCT_TOKENS: return "PUNCT"
    if w.endswith("ly"): return "ADV"
    if w.endswith("ing") or w.endswith("ed"): return "VERB"
    if w.endswith(("tion", "ness", "ment")): return "NOUN"
    if w.endswith(("ous", "ful", "ive", "al")): return "ADJ"
    if w.isdigit(): return "NUM"
    return "WORD"


# ---------------------------------------------------------------------------
# Semantic categorizer
# ---------------------------------------------------------------------------

PLACE_WORDS = {"home", "house", "city", "town", "village", "country", "world",
               "room", "hall", "castle", "forest", "mountain", "river", "sea",
               "ocean", "street", "road", "garden", "church", "school",
               "england", "london", "paris", "europe", "america", "france",
               "germany", "russia", "china", "india", "spain", "italy"}

TIME_WORDS = {"day", "night", "morning", "evening", "afternoon", "year",
              "month", "week", "hour", "minute", "second", "moment", "time",
              "today", "tomorrow", "yesterday", "now", "then", "later",
              "soon", "always", "never", "sometimes", "often", "once"}

ABSTRACT_SUFFIXES = ("tion", "ness", "ment", "ity", "ance", "ence", "ism",
                     "hood", "ship", "dom", "age")

ABSTRACT_WORDS = {"truth", "love", "fear", "hope", "justice", "freedom",
                  "power", "knowledge", "wisdom", "beauty", "soul", "mind",
                  "spirit", "reason", "will", "life", "death", "peace",
                  "war", "law", "right", "duty", "honor", "glory", "faith"}


def categorize(word: str) -> str:
    w = word.lower()
    if not w or w in PUNCT_TOKENS or w in (START_TOKEN, END_TOKEN):
        return "OTHER"
    if w in DET or w in PRON or w in PREP or w in CONJ or w in AUX:
        return "OTHER"
    if w in PLACE_WORDS:
        return "PLACE"
    if w in TIME_WORDS:
        return "TIME"
    if w in ABSTRACT_WORDS:
        return "ABSTRACT"
    if word[0].isupper() and len(word) > 1:
        return "CHARACTER"
    if w.endswith(ABSTRACT_SUFFIXES):
        return "ABSTRACT"
    if w.endswith("ly"):
        return "QUALITY"
    if w.endswith(("ous", "ful", "ive", "able", "ible", "less")):
        return "QUALITY"
    if w.endswith(("ing", "ed")):
        return "ACTION"
    if w.endswith(("er", "or")) and len(w) > 3:
        return "CHARACTER"
    return "OBJECT"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MiniMind:
    def __init__(self, max_order: int = DEFAULT_MAX_ORDER, name: str = "current"):
        self.max_order = max_order
        self.min_order = DEFAULT_MIN_ORDER
        self.name = name

        self.transitions: Dict[int, Dict[Tuple[str, ...], Dict[str, int]]] = {
            o: {} for o in range(self.min_order, self.max_order + 1)
        }
        self.pos_transitions: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self.word_categories: Dict[str, Counter] = defaultdict(Counter)
        self.context_categories: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.char_chains: Dict[str, Counter] = defaultdict(Counter)

        self.vocab: Counter = Counter()
        self.total_tokens = 0
        self.trained = False
        self.model_file = MODEL_DIR / f"{name}.json"

    # ──────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────

    def _train_tokens(self, tokens: List[str]) -> None:
        tags = [pos_tag(t) for t in tokens]
        cats = [categorize(t) for t in tokens]

        for i, tok in enumerate(tokens):
            self.vocab[tok] += 1
            cat = cats[i]
            self.word_categories[tok][cat] += 1

            for order in range(self.min_order, self.max_order + 1):
                if i < order:
                    continue
                ctx = tuple(tokens[i - order:i])
                bucket = self.transitions[order].setdefault(ctx, {})
                bucket[tok] = bucket.get(tok, 0) + 1
                self.context_categories[ctx][cat] += 1

            if i >= 2:
                pos_ctx = (tags[i - 2], tags[i - 1])
                self.pos_transitions[pos_ctx][tags[i]] += 1

        self.total_tokens += len(tokens)

    def _train_chars(self, text: str) -> None:
        text = re.sub(r"\s+", " ", text)
        for i in range(CHAR_ORDER, len(text)):
            ctx = text[i - CHAR_ORDER:i]
            self.char_chains[ctx][text[i]] += 1

    def train(self, text: str, show_progress: bool = True) -> None:
        if not text.strip():
            print("[!] empty text")
            return
        print(f"  tokenizing {len(text):,} chars...")
        sentences = sentence_split(text)
        all_tokens = []
        for s in sentences:
            all_tokens.extend(s)
        if len(all_tokens) < 20:
            print("[!] not enough tokens")
            return

        print(f"  {len(all_tokens):,} tokens across {len(sentences):,} sentences")
        print(f"  max order: {self.max_order} | char order: {CHAR_ORDER}")
        print()

        start = time.time()
        self._train_tokens(all_tokens)
        print(f"  [1/3] word model trained")
        self._train_chars(text)
        print(f"  [2/3] char model trained")
        print(f"  [3/3] semantic categories indexed")
        self.trained = True
        elapsed = time.time() - start

        word_ctx = sum(len(b) for b in self.transitions.values())
        print()
        print(f"  training complete in {elapsed:.1f}s")
        print(f"  word contexts: {word_ctx:,}")
        print(f"  char contexts: {len(self.char_chains):,}")
        print(f"  unique words: {len(self.vocab):,}")
        print(f"  categorized words: {len(self.word_categories):,}")
        print(f"  total tokens: {self.total_tokens:,}")
        print()

        self.save()

    # ──────────────────────────────────────────
    # Sampling
    # ──────────────────────────────────────────

    def _sample(self, dist: Dict[str, float], temp: float) -> str:
        if not dist:
            return UNK
        if temp != 1.0:
            adj = {t: math.pow(p, 1.0 / temp) for t, p in dist.items()}
            total = sum(adj.values())
            if total > 0:
                adj = {t: p / total for t, p in adj.items()}
            dist = adj
        r = random.random()
        cum = 0
        for tok, prob in sorted(dist.items(), key=lambda x: -x[1]):
            cum += prob
            if r <= cum:
                return tok
        return max(dist, key=dist.get)

    def _word_category(self, word: str) -> str:
        cats = self.word_categories.get(word)
        if not cats:
            return categorize(word)
        return cats.most_common(1)[0][0]

    def _predict(self, context: List[str], temperature: float,
                 recent_words: List[str],
                 topic_state: Dict[str, float]) -> Tuple[str, float]:
        for order in range(self.max_order, self.min_order - 1, -1):
            if len(context) < order:
                continue
            key = tuple(context[-order:])
            bucket = self.transitions.get(order, {})
            if key not in bucket:
                continue

            counts = dict(bucket[key])

            # POS reranking
            if len(context) >= 2:
                pos_ctx = (pos_tag(context[-2]), pos_tag(context[-1]))
                pos_counts = self.pos_transitions.get(pos_ctx)
                if pos_counts:
                    total_pos = sum(pos_counts.values())
                    if total_pos > 0:
                        for tok in list(counts.keys()):
                            p = pos_counts.get(pos_tag(tok), 0) / total_pos
                            counts[tok] = counts[tok] * (0.3 + p * 3.0)

            # Category reranking
            ctx_cats = self.context_categories.get(key)
            if ctx_cats and sum(ctx_cats.values()) > 0:
                total_cat = sum(ctx_cats.values())
                for tok in list(counts.keys()):
                    tok_cat = self._word_category(tok)
                    cat_prob = ctx_cats.get(tok_cat, 0) / total_cat
                    counts[tok] = counts[tok] * (0.4 + cat_prob * 2.0)

            # Topic state boost (gentle)
            if topic_state:
                for tok in list(counts.keys()):
                    tok_cat = self._word_category(tok)
                    weight = topic_state.get(tok_cat, 0.0)
                    if weight > TOPIC_MIN_COUNT:
                        counts[tok] = counts[tok] * (1.0 + TOPIC_BOOST * math.log1p(weight))

            # Repetition penalty
            for tok in list(counts.keys()):
                if tok in PUNCT_TOKENS:
                    continue
                if tok in recent_words:
                    occ = recent_words.count(tok)
                    counts[tok] *= (REP_PENALTY_BASE ** occ)

            total = sum(counts.values())
            if total <= 0:
                continue
            probs = {t: c / total for t, c in counts.items()}

            top_token = max(probs, key=probs.get)
            top_prob = probs[top_token]

            chosen = self._sample(probs, temperature)
            return chosen, top_prob

        return UNK, 0.0

    def _char_generate(self, seed_char: Optional[str] = None,
                       max_len: int = FABRICATE_MAX_LEN) -> Optional[str]:
        if not self.char_chains:
            return None
        starts = [c for c in self.char_chains
                  if c.startswith(" ") and c[1:].isalpha()]
        if not starts:
            return None

        ctx = " " + (seed_char or random.choice("abcdefghijklmnoprstuvwy"))
        if ctx not in self.char_chains:
            ctx = random.choice(starts)

        chars: List[str] = []
        if seed_char:
            chars.append(seed_char)

        for _ in range(max_len):
            options = self.char_chains.get(ctx)
            if not options:
                break
            total = sum(options.values())
            if total <= 0:
                break
            r = random.random() * total
            cum = 0
            chosen = None
            for c, n in options.items():
                cum += n
                if r <= cum:
                    chosen = c
                    break
            if chosen is None:
                chosen = max(options, key=options.get)
            if chosen == " ":
                break
            chars.append(chosen)
            ctx = (ctx + chosen)[-CHAR_ORDER:]

        word = "".join(chars).strip()
        if len(word) < 3 or not any(c.isalpha() for c in word):
            return None
        return word

    def _update_topic(self, topic_state: Dict[str, float],
                      emitted_word: str) -> None:
        for k in list(topic_state.keys()):
            topic_state[k] *= TOPIC_DECAY
            if topic_state[k] < 0.1:
                del topic_state[k]
        cat = self._word_category(emitted_word)
        topic_state[cat] = topic_state.get(cat, 0.0) + 1.0

    # ──────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────

    def generate(self, length: int = DEFAULT_LENGTH,
                 seed: Optional[str] = None,
                 temperature: float = DEFAULT_TEMP,
                 show_progress: bool = True,
                 fabricate: bool = True) -> str:
        if not self.trained:
            return "(model not trained)"

        if seed:
            context = tokenize(seed) or [START_TOKEN]
        else:
            context = [START_TOKEN]

        output: List[str] = []
        recent: List[str] = []
        sentences: List[str] = []
        topic_state: Dict[str, float] = {}

        if show_progress:
            print("  generating", end="", flush=True)

        for step in range(length * 3):
            if len(sentences) >= length:
                break

            token, top_prob = self._predict(context, temperature, recent, topic_state)

            # fabrication on low confidence
            if fabricate and token != UNK and token not in PUNCT_TOKENS \
                    and token not in (START_TOKEN, END_TOKEN):
                if top_prob < FABRICATE_MIN_PROB and random.random() < FABRICATE_CHANCE:
                    base = token[0].lower() if token else None
                    made_up = self._char_generate(seed_char=base)
                    if made_up:
                        token = made_up

            if token == UNK:
                if fabricate:
                    made_up = self._char_generate()
                    if made_up:
                        token = made_up
                    else:
                        if len(context) > 1:
                            context = context[1:]
                        continue
                else:
                    if len(context) > 1:
                        context = context[1:]
                    continue

            if token == END_TOKEN:
                if output and output[-1] != START_TOKEN and output[-1] not in PUNCT_TOKENS:
                    sentences.append(_detokenize_sentence(output))
                    output = []
                    recent.clear()
                    topic_state.clear()
                    context = [START_TOKEN]
                    if show_progress and step % 20 == 0:
                        print(".", end="", flush=True)
                else:
                    context.append(token)
                    context = context[-self.max_order:]
                continue

            if token == START_TOKEN:
                if len(context) == 1 and context[0] == START_TOKEN:
                    continue
                context = [START_TOKEN]
                output = []
                continue

            if not output and token in PUNCT_TOKENS:
                continue

            output.append(token)
            context.append(token)
            context = context[-self.max_order:]
            recent.append(token)
            if len(recent) > REP_WINDOW:
                recent.pop(0)
            self._update_topic(topic_state, token)

            if show_progress and step % 20 == 0:
                print(".", end="", flush=True)

        if output and output[-1] not in (START_TOKEN, END_TOKEN):
            sentences.append(_detokenize_sentence(output))

        if show_progress:
            print()
        return "\n\n".join(sentences)

    # ──────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────

    def save(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        ser = {}
        for order, bucket in self.transitions.items():
            ser[str(order)] = {"\x00".join(k): v for k, v in bucket.items()}
        pos_ser = {"\x00".join(k): dict(v) for k, v in self.pos_transitions.items()}
        cat_ser = {k: dict(v) for k, v in self.word_categories.items()}
        ctxcat_ser = {"\x00".join(k): dict(v) for k, v in self.context_categories.items()}
        char_ser = {k: dict(v) for k, v in self.char_chains.items()}

        data = {
            "max_order": self.max_order,
            "min_order": self.min_order,
            "name": self.name,
            "transitions": ser,
            "pos_transitions": pos_ser,
            "word_categories": cat_ser,
            "context_categories": ctxcat_ser,
            "char_chains": char_ser,
            "vocab": dict(self.vocab.most_common(5000)),
            "total_tokens": self.total_tokens,
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
            self.max_order = data.get("max_order", self.max_order)
            self.min_order = data.get("min_order", self.min_order)
            self.name = data.get("name", self.name)
            self.total_tokens = data.get("total_tokens", 0)
            self.vocab = Counter(data.get("vocab", {}))
            self.transitions = {}
            for o_str, bucket in data.get("transitions", {}).items():
                restored = {tuple(k.split("\x00")): v for k, v in bucket.items()}
                self.transitions[int(o_str)] = restored
            self.pos_transitions = defaultdict(Counter)
            for k, v in data.get("pos_transitions", {}).items():
                self.pos_transitions[tuple(k.split("\x00"))] = Counter(v)
            self.word_categories = defaultdict(Counter)
            for k, v in data.get("word_categories", {}).items():
                self.word_categories[k] = Counter(v)
            self.context_categories = defaultdict(Counter)
            for k, v in data.get("context_categories", {}).items():
                self.context_categories[tuple(k.split("\x00"))] = Counter(v)
            self.char_chains = defaultdict(Counter)
            for k, v in data.get("char_chains", {}).items():
                self.char_chains[k] = Counter(v)
            self.trained = any(self.transitions.values())
            return self.trained
        except Exception:
            return False

    def stats(self) -> str:
        if not self.trained:
            return "untrained"
        contexts = sum(len(b) for b in self.transitions.values())
        return (f"word ctx: {contexts:,} | "
                f"categories: {len(self.word_categories):,} | "
                f"char ctx: {len(self.char_chains):,} | "
                f"tokens: {self.total_tokens:,}")


# ---------------------------------------------------------------------------
# Detokenizer
# ---------------------------------------------------------------------------

NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", ")", "]", "}", "'", "\u2014", "\u2013"}
NO_SPACE_AFTER = {"(", "[", "{", "'"}


def _detokenize_sentence(tokens: List[str]) -> str:
    out = []
    for tok in tokens:
        if tok in (START_TOKEN, END_TOKEN):
            continue
        if not out:
            out.append(tok)
            continue
        prev = out[-1]
        if tok in NO_SPACE_BEFORE:
            out.append(tok)
        elif prev and prev[-1] in NO_SPACE_AFTER:
            out.append(tok)
        else:
            out.append(" " + tok)
    text = "".join(out)
    if text:
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".!?":
        text += "."
    return text


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _file_hash(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> Dict:
    if not MANIFEST_FILE.exists():
        return {}
    try:
        with open(MANIFEST_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_manifest(m: Dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w") as f:
        json.dump(m, f, indent=2)


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

def list_models() -> List[str]:
    if not MODEL_DIR.exists():
        return []
    return sorted(f.stem for f in MODEL_DIR.glob("*.json") if f.stem != "_manifest")


def clean_gutenberg(text: str) -> str:
    lines = text.split("\n")
    out = []
    skip = ["gutenberg", "ebook", "produced by", "distributed proofread",
            "updated editions", "www.gutenberg", "project gutenberg",
            "*** start of", "*** end of"]
    for line in lines:
        if any(p in line.lower() for p in skip):
            continue
        out.append(line)
    return "\n".join(out)


def merge_all_models(max_order: int = DEFAULT_MAX_ORDER) -> MiniMind:
    models = list_models()
    merged = MiniMind(max_order=max_order, name="merged")
    merged.transitions = {o: {} for o in range(merged.min_order, merged.max_order + 1)}
    merged.pos_transitions = defaultdict(Counter)
    merged.word_categories = defaultdict(Counter)
    merged.context_categories = defaultdict(Counter)
    merged.char_chains = defaultdict(Counter)
    merged.vocab = Counter()

    for name in models:
        m = MiniMind()
        if not m.load(name):
            continue
        for order, bucket in m.transitions.items():
            if order not in merged.transitions:
                merged.transitions[order] = {}
            for ctx, counts in bucket.items():
                target = merged.transitions[order].setdefault(ctx, {})
                for tok, c in counts.items():
                    target[tok] = target.get(tok, 0) + c
        for ctx, counter in m.pos_transitions.items():
            for tag, c in counter.items():
                merged.pos_transitions[ctx][tag] += c
        for w, counter in m.word_categories.items():
            for cat, c in counter.items():
                merged.word_categories[w][cat] += c
        for ctx, counter in m.context_categories.items():
            for cat, c in counter.items():
                merged.context_categories[ctx][cat] += c
        for ctx, counter in m.char_chains.items():
            for ch, c in counter.items():
                merged.char_chains[ctx][ch] += c
        merged.vocab.update(m.vocab)
        merged.total_tokens += m.total_tokens

    merged.trained = any(merged.transitions.values())
    return merged


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_train(args) -> None:
    fp = Path(args.file)
    if not fp.exists():
        print(f"[!] file not found: {fp}")
        sys.exit(1)
    print(f"[*] reading {fp}...")
    text = clean_gutenberg(fp.read_text(encoding="utf-8", errors="replace"))
    name = fp.stem.lower().replace(" ", "_")
    model = MiniMind(max_order=args.order, name=name)
    model.train(text)
    manifest = load_manifest()
    manifest[name] = {
        "file": str(fp),
        "hash": _file_hash(fp),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_manifest(manifest)
    print(f"[+] saved as: {name}")


def cmd_autotrain(args) -> None:
    folder = Path(args.folder) if args.folder else Path.cwd()
    files = [f for f in folder.glob("*.txt") if not f.stem.startswith("clean_")]
    if not files:
        print(f"[!] no .txt files in {folder}")
        sys.exit(1)
    manifest = load_manifest()
    trained = 0
    skipped = 0
    print(f"[*] found {len(files)} files in {folder}")
    print()
    for i, fp in enumerate(files, 1):
        name = fp.stem.lower().replace(" ", "_")
        if name in manifest:
            try:
                if manifest[name].get("hash") == _file_hash(fp):
                    print(f"=== [{i}/{len(files)}] {fp.name} ===")
                    print(f"  [=] already trained, skipping\n")
                    skipped += 1
                    continue
            except Exception:
                pass
        print(f"=== [{i}/{len(files)}] {fp.name} ===")
        try:
            text = clean_gutenberg(fp.read_text(encoding="utf-8", errors="replace"))
            if len(text) < 200:
                print("  [!] too small, skipping\n")
                continue
            model = MiniMind(max_order=args.order, name=name)
            model.train(text)
            manifest[name] = {
                "file": str(fp),
                "hash": _file_hash(fp),
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_manifest(manifest)
            trained += 1
        except Exception as e:
            print(f"  [!] failed: {e}\n")
    print(f"[+] done — trained: {trained}, skipped: {skipped}")


def cmd_generate(args) -> None:
    models = list_models()
    if not models:
        print("[!] no models")
        sys.exit(1)
    name = args.model or ("current" if "current" in models else models[-1])
    model = MiniMind()
    if not model.load(name):
        print(f"[!] could not load {name}")
        sys.exit(1)
    print(f"[*] model: {model.name}")
    print(f"[*] {model.stats()}")
    print()
    out = model.generate(
        length=args.length,
        seed=args.seed,
        temperature=args.temp,
        fabricate=not args.no_fabricate,
    )
    print()
    print("─" * 60)
    print(out)
    print("─" * 60)
    print()


def cmd_freegen(args) -> None:
    print("[*] merging all models (count-weighted)...")
    start = time.time()
    model = merge_all_models()
    elapsed = time.time() - start
    if not model.trained:
        print("[!] nothing to merge")
        sys.exit(1)
    print(f"[*] merged {len(list_models())} model(s) in {elapsed:.1f}s")
    print(f"[*] {model.stats()}")
    print()
    out = model.generate(
        length=args.length,
        seed=args.seed,
        temperature=args.temp,
        fabricate=not args.no_fabricate,
    )
    print()
    print("─" * 60)
    print(out)
    print("─" * 60)
    print()


def cmd_models(args) -> None:
    models = list_models()
    manifest = load_manifest()
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
                    kb = m.model_file.stat().st_size / 1024
                except Exception:
                    kb = 0
                when = manifest.get(name, {}).get("trained_at", "?")
                print(f"  {name:<25} {m.stats()} ({kb:.0f} KB) [{when}]")
    print("─" * 60)
    print()


def cmd_status(args) -> None:
    models = list_models()
    if not models:
        print("[*] no models yet")
        return
    print(f"[*] {len(models)} model(s)")
    for name in models:
        m = MiniMind()
        if m.load(name):
            print(f"  {name}: {m.stats()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="minimind",
        description="Mini Mind v7 - semantic topic model by Templar Studios",
    )
    sub = parser.add_subparsers(dest="command", help="commands")

    t = sub.add_parser("train", help="train on one file")
    t.add_argument("file")
    t.add_argument("--order", type=int, default=DEFAULT_MAX_ORDER)

    at = sub.add_parser("autotrain", help="train all .txt files")
    at.add_argument("folder", nargs="?", default=None)
    at.add_argument("--order", type=int, default=DEFAULT_MAX_ORDER)

    g = sub.add_parser("generate", help="generate from one model")
    g.add_argument("--model", default=None)
    g.add_argument("--seed", default=None)
    g.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    g.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    g.add_argument("--no-fabricate", action="store_true")

    fg = sub.add_parser("freegen", help="generate from merged memory")
    fg.add_argument("--seed", default=None)
    fg.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    fg.add_argument("--temp", type=float, default=1.0)
    fg.add_argument("--no-fabricate", action="store_true")

    sub.add_parser("models")
    sub.add_parser("status")

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
    mp.freeze_support()
    main()