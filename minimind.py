#!/usr/bin/env python3
"""
Mini Mind v8.1 - word-level language model
Part of Templar Studios | GPL v3.0

A complete rewrite. Fixes every known issue from v5-v8.

Key design:
  - Control tokens (START/END) never pollute the predictive context
  - Infallible backoff: order N -> ... -> unigram -> random
  - Nucleus (top-p) sampling for natural output
  - Topic state that drifts naturally
  - Optional char-level fabrication for invented words
  - Sentence-aware detokenizer

Pure python. Works in ish, a-shell, linux, macOS.
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
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR = Path.home() / ".minimind_models"
MANIFEST_FILE = MODEL_DIR / "_manifest.json"

DEFAULT_ORDER = 5
DEFAULT_LENGTH = 250          # target words
DEFAULT_TEMP = 0.85
DEFAULT_TOP_P = 0.92
DEFAULT_FABRICATE = True

START_TOKEN = "<START>"
END_TOKEN = "<END>"

PUNCT = {".", ",", "!", "?", ";", ":", "\u2014", "\u2013"}
SENTENCE_END = {".", "!", "?"}

CHAR_ORDER = 4
REP_WINDOW = 40
REP_PENALTY = 0.55
BIGRAM_PENALTY = 0.30
BIGRAM_WINDOW = 12

TOPIC_DECAY = 0.88
TOPIC_BOOST = 0.7
TOPIC_MIN = 5

FABRICATE_MIN_PROB = 0.08
FABRICATE_CHANCE = 0.25
FABRICATE_MAX_LEN = 14
FABRICATE_MAX_RATIO = 0.15

TOP_K = 40
PRUNE_MIN = 2
CONTEXT_CAT_CAP = 8000
VOCAB_CAP = 4000
MAX_JSON_BYTES = 40 * 1024 * 1024

MAX_END_RETRIES = 8
MAX_STEP_MULTIPLIER = 6

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(
    r"<START>|<END>|[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'\-]*|[0-9]+|[.,!?;:\"()\[\]{}\u2014\u2013\u201c\u201d]"
)


def tokenize(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text)
    return TOKEN_RE.findall(text)


def split_sentences(text: str) -> List[List[str]]:
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
       "his", "her", "its", "our", "their", "some", "any", "each", "every",
       "either", "neither", "no", "both", "all", "few", "many", "much"}

PRON = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
        "us", "them", "who", "whom", "which", "what", "myself", "yourself",
        "himself", "herself", "itself", "ourselves", "themselves"}

PREP = {"of", "in", "on", "at", "by", "for", "with", "to", "from", "into",
        "through", "during", "before", "after", "above", "below", "between",
        "under", "over", "about", "against", "toward", "towards", "upon",
        "within", "without", "along", "across", "behind", "beyond"}

CONJ = {"and", "or", "but", "nor", "for", "yet", "so", "because",
        "although", "while", "if", "when", "unless", "since", "though",
        "whereas", "whether", "then", "than"}

AUX = {"is", "are", "was", "were", "be", "been", "being", "has", "have",
       "had", "having", "do", "does", "did", "will", "would", "shall",
       "should", "can", "could", "may", "might", "must", "ought"}


def pos_tag(word: str) -> str:
    if word in (START_TOKEN, END_TOKEN):
        return "BOUNDARY"
    if word in PUNCT:
        return "PUNCT"
    w = word.lower()
    if w in DET: return "DET"
    if w in PRON: return "PRON"
    if w in PREP: return "PREP"
    if w in CONJ: return "CONJ"
    if w in AUX: return "AUX"
    if w.endswith("ly") and len(w) > 3: return "ADV"
    if w.endswith("ing") or w.endswith("ed"): return "VERB"
    if w.endswith(("tion", "ness", "ment", "ity", "ance", "ence")): return "NOUN"
    if w.endswith(("ous", "ful", "ive", "able", "ible", "less")): return "ADJ"
    if w.isdigit(): return "NUM"
    return "WORD"


# ---------------------------------------------------------------------------
# Semantic categories
# ---------------------------------------------------------------------------

PLACE_WORDS = {"home", "house", "city", "town", "village", "country", "world",
               "room", "hall", "castle", "forest", "mountain", "river", "sea",
               "ocean", "street", "road", "garden", "church", "school",
               "england", "london", "paris", "europe", "america", "france",
               "germany", "russia", "china", "india", "spain", "italy",
               "state", "nation", "republic", "kingdom", "empire", "land"}

TIME_WORDS = {"day", "night", "morning", "evening", "afternoon", "year",
              "month", "week", "hour", "minute", "second", "moment", "time",
              "today", "tomorrow", "yesterday", "now", "then", "later",
              "soon", "always", "never", "sometimes", "often", "once",
              "history", "century", "age", "epoch", "era"}

ABSTRACT_SUFFIXES = ("tion", "ness", "ment", "ity", "ance", "ence", "ism",
                     "hood", "ship", "dom", "age")

ABSTRACT_WORDS = {"truth", "love", "fear", "hope", "justice", "freedom",
                  "power", "knowledge", "wisdom", "beauty", "soul", "mind",
                  "spirit", "reason", "will", "life", "death", "peace",
                  "war", "law", "right", "duty", "honor", "glory", "faith",
                  "society", "class", "state", "property", "capital",
                  "labor", "labour", "production", "exchange", "value",
                  "liberty", "equality", "rights", "struggle", "revolution",
                  "movement", "system", "order", "oppression",
                  "exploitation", "bourgeoisie", "proletariat"}


def categorize(word: str) -> str:
    if word in (START_TOKEN, END_TOKEN):
        return "BOUNDARY"
    if word in PUNCT:
        return "OTHER"
    w = word.lower()
    if not w:
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
    if w.endswith("ly") and len(w) > 3:
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
    def __init__(self, order: int = DEFAULT_ORDER, name: str = "current"):
        self.order = order
        self.name = name

        # word transitions: transitions[order][(w1,...,wN)] = {next_word: count}
        self.transitions: Dict[int, Dict[Tuple[str, ...], Dict[str, int]]] = {
            o: {} for o in range(1, order + 1)
        }

        # char transitions: char_chains[ctx_str] = {next_char: count}
        self.char_chains: Dict[str, Counter] = defaultdict(Counter)

        # POS: pos_transitions[(pos1,pos2)] = {pos3: count}
        self.pos_transitions: Dict[Tuple[str, str], Counter] = defaultdict(Counter)

        # category per word
        self.word_categories: Dict[str, Counter] = defaultdict(Counter)

        # category given context
        self.context_categories: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)

        # vocab + unigram fallback
        self.vocab: Counter = Counter()
        self.unigram: Dict[str, float] = {}
        self.total_tokens = 0
        self.trained = False

        self.model_file = MODEL_DIR / f"{name}.json"

        # caches
        self._cat_cache: Dict[str, str] = {}
        self._pos_cache: Dict[str, str] = {}
        self._char_starts: Optional[List[str]] = None

    # ─────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────

    def train(self, text: str, show_progress: bool = True) -> bool:
        if not text.strip():
            print("[!] empty text")
            return False

        print(f"  tokenizing {len(text):,} chars...")
        sentences = split_sentences(text)
        if not sentences:
            print("[!] no valid sentences found")
            return False

        # flatten tokens for training (START/END included as boundary markers)
        all_tokens: List[str] = []
        for s in sentences:
            all_tokens.extend(s)

        print(f"  {len(all_tokens):,} tokens across {len(sentences):,} sentences")
        print(f"  order: {self.order}")
        print()

        start = time.time()

        # phase 1: word transitions
        print("  [1/3] word transitions...")
        tags = [pos_tag(t) for t in all_tokens]
        cats = [categorize(t) for t in all_tokens]

        for i, tok in enumerate(all_tokens):
            self.vocab[tok] += 1
            self.word_categories[tok][cats[i]] += 1

            # lower-order contexts
            for order in range(1, self.order + 1):
                if i < order:
                    continue
                ctx = tuple(all_tokens[i - order:i])
                bucket = self.transitions[order].setdefault(ctx, {})
                bucket[tok] = bucket.get(tok, 0) + 1
                self.context_categories[ctx][cats[i]] += 1

            # POS context
            if i >= 2:
                pos_ctx = (tags[i - 2], tags[i - 1])
                self.pos_transitions[pos_ctx][tags[i]] += 1

        self.total_tokens += len(all_tokens)

        # phase 2: char transitions
        print("  [2/3] character model...")
        clean_text = re.sub(r"\s+", " ", text)
        for i in range(CHAR_ORDER, len(clean_text)):
            ctx = clean_text[i - CHAR_ORDER:i]
            self.char_chains[ctx][clean_text[i]] += 1

        # phase 3: prune + unigram
        print("  [3/3] pruning + unigram...")
        self._prune()
        self._build_unigram()

        self.trained = True
        elapsed = time.time() - start

        word_ctx = sum(len(b) for b in self.transitions.values())
        print()
        print(f"  training complete in {elapsed:.1f}s")
        print(f"  word contexts: {word_ctx:,}")
        print(f"  char contexts: {len(self.char_chains):,}")
        print(f"  unique words: {len(self.vocab):,}")
        print(f"  total tokens: {self.total_tokens:,}")
        print()

        return self.save()

    def _prune(self) -> None:
        for order in self.transitions:
            pruned = {}
            for ctx, counts in self.transitions[order].items():
                if sum(counts.values()) >= PRUNE_MIN:
                    pruned[ctx] = counts
            self.transitions[order] = pruned

        if len(self.context_categories) > CONTEXT_CAT_CAP:
            scored = sorted(
                self.context_categories.items(),
                key=lambda kv: -sum(kv[1].values())
            )[:CONTEXT_CAT_CAP]
            self.context_categories = defaultdict(Counter, dict(scored))

    def _build_unigram(self) -> None:
        total = sum(self.vocab.values())
        if total == 0:
            self.unigram = {}
            return
        self.unigram = {
            w: c / total
            for w, c in self.vocab.items()
            if c >= 2 and w not in (START_TOKEN, END_TOKEN)
        }
        if not self.unigram:
            # if nothing qualifies, use full vocab
            self.unigram = {
                w: c / total
                for w, c in self.vocab.items()
                if w not in (START_TOKEN, END_TOKEN)
            }

    # ─────────────────────────────────────────
    # Caches
    # ─────────────────────────────────────────

    def _cat(self, word: str) -> str:
        c = self._cat_cache.get(word)
        if c is not None:
            return c
        cats = self.word_categories.get(word)
        c = cats.most_common(1)[0][0] if cats else categorize(word)
        self._cat_cache[word] = c
        return c

    def _pos(self, word: str) -> str:
        p = self._pos_cache.get(word)
        if p is None:
            p = pos_tag(word)
            self._pos_cache[word] = p
        return p

    # ─────────────────────────────────────────
    # Sampling
    # ─────────────────────────────────────────

    def _sample(self, dist: Dict[str, float],
                temp: float, top_p: float) -> Optional[str]:
        """Nucleus sampling. Returns None if dist empty."""
        if not dist:
            return None
        if len(dist) == 1:
            return next(iter(dist))

        items = sorted(dist.items(), key=lambda x: -x[1])

        # temperature on raw probs
        if temp != 1.0 and temp > 0:
            items = [(t, math.pow(p, 1.0 / temp)) for t, p in items]
            total = sum(p for _, p in items)
            if total > 0:
                items = [(t, p / total) for t, p in items]

        # nucleus cut
        kept = []
        cum = 0.0
        for t, p in items:
            kept.append((t, p))
            cum += p
            if cum >= top_p:
                break

        total = sum(p for _, p in kept)
        if total <= 0:
            return kept[0][0] if kept else None

        kept = [(t, p / total) for t, p in kept]

        r = random.random()
        cum = 0.0
        for t, p in kept:
            cum += p
            if r <= cum:
                return t
        return kept[-1][0]

    # ─────────────────────────────────────────
    # Prediction (infallible)
    # ─────────────────────────────────────────

    def _predict(self, context: List[str],
                 temp: float, top_p: float,
                 recent: Dict[str, int],
                 recent_bigrams: set,
                 topic_factors: Dict[str, float]
                 ) -> Tuple[Optional[str], float]:
        """
        Try highest-order context first. Back off until a hit.
        Returns (token, top_prob_after_rerank).
        Falls back to unigram, then None.
        """
        for order in range(min(self.order, len(context)), 0, -1):
            key = tuple(context[-order:])
            bucket = self.transitions.get(order, {})
            if key not in bucket:
                continue

            raw = bucket[key]
            if not raw:
                continue

            # top-K by raw count
            if len(raw) > TOP_K:
                counts = dict(sorted(raw.items(), key=lambda x: -x[1])[:TOP_K])
            else:
                counts = dict(raw)

            if len(counts) == 1:
                only = next(iter(counts))
                return only, 1.0

            # POS rerank
            if len(context) >= 2:
                pos_ctx = (self._pos(context[-2]), self._pos(context[-1]))
                pos_counts = self.pos_transitions.get(pos_ctx)
                if pos_counts:
                    total_pos = sum(pos_counts.values())
                    if total_pos > 0:
                        for tok in counts:
                            p = pos_counts.get(self._pos(tok), 0) / total_pos
                            counts[tok] *= (0.3 + p * 3.0)

            # category rerank from context
            ctx_cats = self.context_categories.get(key)
            if ctx_cats:
                total_cat = sum(ctx_cats.values())
                if total_cat > 0:
                    for tok in counts:
                        cp = ctx_cats.get(self._cat(tok), 0) / total_cat
                        counts[tok] *= (0.4 + cp * 2.0)

            # topic boost
            if topic_factors:
                for tok in counts:
                    f = topic_factors.get(self._cat(tok))
                    if f is not None:
                        counts[tok] *= f

            # word repetition
            if recent:
                for tok in counts:
                    if tok in PUNCT or tok in (START_TOKEN, END_TOKEN):
                        continue
                    occ = recent.get(tok, 0)
                    if occ:
                        counts[tok] *= (REP_PENALTY ** occ)

            # bigram repetition
            if recent_bigrams and context:
                prev = context[-1]
                for tok in counts:
                    if tok in PUNCT or tok in (START_TOKEN, END_TOKEN):
                        continue
                    if (prev, tok) in recent_bigrams:
                        counts[tok] *= BIGRAM_PENALTY

            total = sum(counts.values())
            if total <= 0:
                continue
            probs = {t: c / total for t, c in counts.items()}
            top = max(probs, key=probs.get)
            top_prob = probs[top]

            chosen = self._sample(probs, temp, top_p)
            if chosen is not None:
                return chosen, top_prob

        # fallback: unigram
        if self.unigram:
            chosen = self._sample(self.unigram, temp, top_p)
            if chosen is not None:
                return chosen, 0.05

        # last resort: random vocab
        if self.vocab:
            candidates = [w for w in self.vocab
                          if w not in (START_TOKEN, END_TOKEN)]
            if candidates:
                return random.choice(candidates), 0.02

        return None, 0.0

    # ─────────────────────────────────────────
    # Character fabrication
    # ─────────────────────────────────────────

    def _fabricate(self, seed_prefix: Optional[str] = None) -> Optional[str]:
        """Invent a word using char chains. seed_prefix = 1-2 chars to start with."""
        if not self.char_chains:
            return None

        if self._char_starts is None:
            self._char_starts = [
                c for c in self.char_chains
                if c.startswith(" ") and c[1:2].isalpha()
            ]
        if not self._char_starts:
            return None

        if seed_prefix:
            ctx = (" " + seed_prefix)[-CHAR_ORDER:].rjust(CHAR_ORDER, " ")
            if ctx not in self.char_chains:
                ctx = random.choice(self._char_starts)
                seed_prefix = None
        else:
            ctx = random.choice(self._char_starts)

        chars: List[str] = list(seed_prefix) if seed_prefix else []

        for _ in range(FABRICATE_MAX_LEN):
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

    # ─────────────────────────────────────────
    # Topic state
    # ─────────────────────────────────────────

    def _topic_decay(self, state: Dict[str, float]) -> None:
        for k in list(state.keys()):
            state[k] *= TOPIC_DECAY
            if state[k] < 0.1:
                del state[k]

    def _topic_factors(self, state: Dict[str, float]) -> Dict[str, float]:
        return {
            cat: 1.0 + TOPIC_BOOST * math.log1p(w)
            for cat, w in state.items() if w > TOPIC_MIN
        }

    # ─────────────────────────────────────────
    # Generation
    # ─────────────────────────────────────────

    def generate(self,
                 length: int = DEFAULT_LENGTH,
                 seed: Optional[str] = None,
                 temperature: float = DEFAULT_TEMP,
                 top_p: float = DEFAULT_TOP_P,
                 fabricate: bool = DEFAULT_FABRICATE,
                 show_progress: bool = True) -> str:
        """
        Generate ~length words of text.
        Never returns UNK. Never loops on END. Never stops after one sentence.
        """
        if not self.trained:
            return "(model not trained)"

        if not self.unigram:
            self._build_unigram()

        sentences: List[str] = []
        word_total = 0
        max_steps = max(500, length * MAX_STEP_MULTIPLIER)

        # per-sentence state
        def new_sentence_state():
            return {
                "context": [START_TOKEN],
                "output": [],
                "recent": {},
                "recent_order": [],
                "bigrams": [],
                "topic": {},
                "words": 0,
                "fab": 0,
                "end_retries": 0,
            }

        st = new_sentence_state()

        if seed:
            seed_tokens = tokenize(seed) or [START_TOKEN]
            st["context"] = list(seed_tokens)
            st["output"] = list(seed_tokens)

        if show_progress:
            print("  generating", end="", flush=True)

        for step in range(max_steps):
            if word_total >= length:
                break

            factors = self._topic_factors(st["topic"])
            token, top_prob = self._predict(
                st["context"], temperature, top_p,
                st["recent"], set(st["bigrams"]), factors
            )

            # ─── NULL: model completely empty ───
            if token is None:
                break

            # ─── END handling ───
            if token == END_TOKEN:
                # if we have real content, close the sentence
                if st["output"] and st["output"][-1] not in PUNCT and st["output"][-1] != START_TOKEN:
                    sentences.append(_detokenize(st["output"]))
                    word_total += st["words"]
                    st = new_sentence_state()
                    if show_progress and step % 20 == 0:
                        print(".", end="", flush=True)
                    continue

                # empty sentence → do NOT pollute context
                st["end_retries"] += 1
                if st["end_retries"] >= MAX_END_RETRIES:
                    # force bootstrap with a common word
                    boot = None
                    if self.unigram:
                        boot = self._sample(self.unigram, temperature, top_p)
                    if boot is None:
                        break
                    st["output"].append(boot)
                    st["context"] = [START_TOKEN, boot]
                    st["recent"][boot] = 1
                    st["recent_order"].append(boot)
                    st["topic"][self._cat(boot)] = st["topic"].get(self._cat(boot), 0.0) + 1.0
                    st["words"] = 1
                    st["end_retries"] = 0
                else:
                    st["context"] = [START_TOKEN]
                continue

            # ─── START token inside generation ───
            if token == START_TOKEN:
                if len(st["context"]) == 1 and st["context"][0] == START_TOKEN:
                    continue
                st["context"] = [START_TOKEN]
                st["output"] = []
                continue

            # ─── punctuation guard at sentence start ───
            if not st["output"] and token in PUNCT:
                continue

            # ─── fabrication on low confidence ───
            fabricated = False
            if (fabricate and token not in PUNCT
                    and token not in (START_TOKEN, END_TOKEN)
                    and st["words"] > 0
                    and (st["fab"] / max(1, st["words"])) < FABRICATE_MAX_RATIO
                    and top_prob < FABRICATE_MIN_PROB
                    and random.random() < FABRICATE_CHANCE):
                seed_prefix = None
                if st["output"]:
                    prev = st["output"][-1]
                    if prev not in PUNCT and len(prev) > 3:
                        seed_prefix = prev[-2:].lower()
                made_up = self._fabricate(seed_prefix=seed_prefix)
                if made_up and made_up.lower() != token.lower():
                    token = made_up
                    fabricated = True

            # ─── emit ───
            st["end_retries"] = 0

            if st["output"]:
                st["bigrams"].append((st["output"][-1], token))
                if len(st["bigrams"]) > BIGRAM_WINDOW:
                    st["bigrams"].pop(0)

            st["output"].append(token)
            st["context"].append(token)
            st["context"] = st["context"][-self.order:]

            st["recent"][token] = st["recent"].get(token, 0) + 1
            st["recent_order"].append(token)
            if len(st["recent_order"]) > REP_WINDOW:
                old = st["recent_order"].pop(0)
                c = st["recent"].get(old, 1) - 1
                if c <= 0:
                    st["recent"].pop(old, None)
                else:
                    st["recent"][old] = c

            self._topic_decay(st["topic"])
            cat = self._cat(token)
            st["topic"][cat] = st["topic"].get(cat, 0.0) + 1.0

            if token not in PUNCT and token not in (START_TOKEN, END_TOKEN):
                st["words"] += 1
                if fabricated:
                    st["fab"] += 1

            if show_progress and step % 20 == 0:
                print(".", end="", flush=True)

        # flush trailing sentence
        if st["output"] and st["output"][-1] not in (START_TOKEN, END_TOKEN):
            sentences.append(_detokenize(st["output"]))
            word_total += st["words"]

        if show_progress:
            print()

        return "\n\n".join(s for s in sentences if s.strip())

    # ─────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────

    def save(self) -> bool:
        try:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"[!] cannot create dir: {e}", file=sys.stderr)
            return False

        try:
            data = {
                "order": self.order,
                "name": self.name,
                "transitions": {
                    str(o): {"\x00".join(k): v for k, v in b.items()}
                    for o, b in self.transitions.items()
                },
                "pos_transitions": {
                    "\x00".join(k): dict(v)
                    for k, v in self.pos_transitions.items()
                },
                "word_categories": {
                    k: dict(v) for k, v in self.word_categories.items()
                },
                "context_categories": {
                    "\x00".join(k): dict(v)
                    for k, v in self.context_categories.items()
                },
                "char_chains": {
                    k: dict(v) for k, v in self.char_chains.items()
                },
                "vocab": dict(self.vocab.most_common(VOCAB_CAP)),
                "total_tokens": self.total_tokens,
            }

            with open(self.model_file, "w") as f:
                json.dump(data, f)
        except OSError as e:
            print(f"[!] write failed: {e}", file=sys.stderr)
            return False
        except (TypeError, ValueError) as e:
            print(f"[!] serialize failed: {e}", file=sys.stderr)
            return False

        try:
            size = self.model_file.stat().st_size
            if size > MAX_JSON_BYTES:
                print(f"[!] warning: model file is {size/1024/1024:.1f} MB",
                      file=sys.stderr)
        except OSError:
            pass

        return True

    def load(self, name: Optional[str] = None) -> bool:
        if name:
            self.name = name
            self.model_file = MODEL_DIR / f"{name}.json"

        if not self.model_file.exists():
            return False

        try:
            size = self.model_file.stat().st_size
            if size > MAX_JSON_BYTES:
                print(f"[!] model too large ({size/1024/1024:.1f} MB)",
                      file=sys.stderr)
                return False
        except OSError as e:
            print(f"[!] stat failed: {e}", file=sys.stderr)
            return False

        try:
            with open(self.model_file) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[!] corrupt JSON: {e}", file=sys.stderr)
            return False
        except MemoryError:
            print("[!] out of memory loading", file=sys.stderr)
            return False
        except Exception as e:
            print(f"[!] load failed: {type(e).__name__}: {e}", file=sys.stderr)
            return False

        try:
            self.order = data.get("order", self.order)
            self.name = data.get("name", self.name)
            self.total_tokens = data.get("total_tokens", 0)
            self.vocab = Counter(data.get("vocab", {}))

            self.transitions = {}
            for o_str, bucket in data.get("transitions", {}).items():
                try:
                    o = int(o_str)
                except ValueError:
                    continue
                self.transitions[o] = {
                    tuple(k.split("\x00")): v for k, v in bucket.items()
                }

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

            self._cat_cache.clear()
            self._pos_cache.clear()
            self._char_starts = None

            self.trained = any(self.transitions.values())
            if self.trained:
                self._build_unigram()
            return self.trained

        except MemoryError:
            print("[!] out of memory rebuilding", file=sys.stderr)
            return False
        except Exception as e:
            print(f"[!] rebuild failed: {type(e).__name__}: {e}",
                  file=sys.stderr)
            return False

    def stats(self) -> str:
        if not self.trained:
            return "untrained"
        ctx = sum(len(b) for b in self.transitions.values())
        return (f"ctx: {ctx:,} | vocab: {len(self.vocab):,} | "
                f"chars: {len(self.char_chains):,} | "
                f"tokens: {self.total_tokens:,}")


# ---------------------------------------------------------------------------
# Detokenizer
# ---------------------------------------------------------------------------

NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", ")", "]", "}",
                   "'s", "n't", "'re", "'ve", "'ll", "'d", "'m", "'",
                   "\u2014", "\u2013"}
NO_SPACE_AFTER = {"(", "[", "{", "\u201c", "\u2018"}


def _detokenize(tokens: List[str]) -> str:
    out: List[str] = []
    capitalize_next = True
    in_quote = False

    for tok in tokens:
        if tok in (START_TOKEN, END_TOKEN):
            continue

        # quote handling
        if tok in ('"', "\u201c", "\u201d", "\u2018", "\u2019"):
            if not out:
                out.append(tok)
                in_quote = True
                continue
            prev = out[-1]
            if prev and prev[-1] in ("(", "[", "{"):
                out.append(tok)
                in_quote = True
            elif in_quote:
                out.append(tok)
                in_quote = False
            else:
                out.append(" " + tok)
                in_quote = True
            continue

        if capitalize_next and tok and tok[0].isalpha():
            tok = tok[0].upper() + tok[1:]
        capitalize_next = False

        if not out:
            out.append(tok)
        else:
            prev = out[-1]
            if tok in NO_SPACE_BEFORE:
                out.append(tok)
            elif prev and prev[-1] in NO_SPACE_AFTER:
                out.append(tok)
            else:
                out.append(" " + tok)

        if tok in SENTENCE_END:
            capitalize_next = True

    text = "".join(out).strip()

    if text and text[-1] not in ".!?":
        if text[-1] not in "\"')]":
            text += "."

    return text


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
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
# Utilities
# ---------------------------------------------------------------------------

def list_models() -> List[str]:
    if not MODEL_DIR.exists():
        return []
    return sorted(
        f.stem for f in MODEL_DIR.glob("*.json") if f.stem != "_manifest"
    )


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


def merge_all_models() -> MiniMind:
    models = list_models()
    merged = MiniMind()
    merged.transitions = {}
    merged.pos_transitions = defaultdict(Counter)
    merged.word_categories = defaultdict(Counter)
    merged.context_categories = defaultdict(Counter)
    merged.char_chains = defaultdict(Counter)
    merged.vocab = Counter()
    max_order = 1

    for name in models:
        m = MiniMind()
        if not m.load(name):
            print(f"[!] skipping unloadable: {name}", file=sys.stderr)
            continue

        max_order = max(max_order, m.order)

        for o, bucket in m.transitions.items():
            if o not in merged.transitions:
                merged.transitions[o] = {}
            for ctx, counts in bucket.items():
                target = merged.transitions[o].setdefault(ctx, {})
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

    merged.order = max_order
    # ensure all orders exist up to max
    for o in range(1, max_order + 1):
        if o not in merged.transitions:
            merged.transitions[o] = {}

    merged.trained = any(merged.transitions.values())
    merged._prune()
    if merged.trained:
        merged._build_unigram()
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
    if len(text) < 100:
        print(f"[!] file too small: {len(text)} chars")
        sys.exit(1)

    name = fp.stem.lower().replace(" ", "_")
    model = MiniMind(order=args.order, name=name)
    if not model.train(text):
        print("[!] training failed")
        sys.exit(1)

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
                    print("  [=] already trained, skipping\n")
                    skipped += 1
                    continue
            except Exception:
                pass

        print(f"=== [{i}/{len(files)}] {fp.name} ===")
        try:
            text = clean_gutenberg(
                fp.read_text(encoding="utf-8", errors="replace")
            )
            if len(text) < 200:
                print("  [!] too small, skipping\n")
                continue

            model = MiniMind(order=args.order, name=name)
            if not model.train(text):
                print("  [!] training failed\n")
                continue

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

    t0 = time.time()
    out = model.generate(
        length=args.length,
        seed=args.seed,
        temperature=args.temp,
        top_p=args.top_p,
        fabricate=not args.no_fabricate,
    )
    elapsed = time.time() - t0

    print()
    print("─" * 60)
    print(out)
    print("─" * 60)
    print(f"[*] generated in {elapsed:.2f}s")
    print()


def cmd_freegen(args) -> None:
    print("[*] merging all models...")
    t0 = time.time()
    model = merge_all_models()
    merge_time = time.time() - t0

    if not model.trained:
        print("[!] nothing to merge")
        sys.exit(1)

    print(f"[*] merged {len(list_models())} model(s) in {merge_time:.1f}s")
    print(f"[*] {model.stats()}")
    print()

    t1 = time.time()
    out = model.generate(
        length=args.length,
        seed=args.seed,
        temperature=args.temp,
        top_p=args.top_p,
        fabricate=not args.no_fabricate,
    )
    gen_time = time.time() - t1

    print()
    print("─" * 60)
    print(out)
    print("─" * 60)
    print(f"[*] generated in {gen_time:.2f}s")
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
                print(f"  {name:<22} {m.stats()}")
                print(f"  {'':<22} ({kb:.0f} KB, {when})")
            else:
                print(f"  {name:<22} [!] could not load")
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="minimind",
        description="Mini Mind v8.1 - language model by Templar Studios",
    )
    sub = parser.add_subparsers(dest="command", help="commands")

    t = sub.add_parser("train", help="train on one file")
    t.add_argument("file")
    t.add_argument("--order", type=int, default=DEFAULT_ORDER)

    at = sub.add_parser("autotrain", help="train all .txt files in folder")
    at.add_argument("folder", nargs="?", default=None)
    at.add_argument("--order", type=int, default=DEFAULT_ORDER)

    g = sub.add_parser("generate", help="generate from one model")
    g.add_argument("--model", default=None)
    g.add_argument("--seed", default=None)
    g.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    g.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    g.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    g.add_argument("--no-fabricate", action="store_true")

    fg = sub.add_parser("freegen", help="generate from merged memory")
    fg.add_argument("--seed", default=None)
    fg.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    fg.add_argument("--temp", type=float, default=1.0)
    fg.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    fg.add_argument("--no-fabricate", action="store_true")

    sub.add_parser("models", help="list stored models")
    sub.add_parser("status", help="show model stats")

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
    main()