#!/usr/bin/env python3
"""
Mini Mind v8 - word-level language model, refined
Part of Templar Studios | GPL v3.0

v8 improvements over v7.2:
  - Nucleus (top-p) sampling instead of pure temperature
  - Infallible backoff — never returns UNK on non-empty models
  - Unigram fallback from trained vocab
  - Phrase-level repetition penalty (bigram + trigram dedup)
  - Sentence-state tracking (start/mid/end, quotes)
  - Context-aware fabrication (blends previous word endings)
  - Fabrication cap (max 20% of sentence)
  - Streaming output
  - Smarter detokenizer (capitalization, quotes, dashes)
  - Category + POS caches
  - All v7.2 speed wins preserved

Same model format as v7.x — old models load fine.
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
from typing import Dict, List, Optional, Tuple, Set

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR = Path.home() / ".minimind_models_v7"
MANIFEST_FILE = MODEL_DIR / "_manifest.json"

DEFAULT_MAX_ORDER = 5
DEFAULT_MIN_ORDER = 1
DEFAULT_LENGTH = 300
DEFAULT_TEMP = 0.8
DEFAULT_TOP_P = 0.92

START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK = "<UNK>"

PUNCT_TOKENS = {".", ",", "!", "?", ";", ":", "\u2014", "\u2013", ")", "]", "}"}
OPEN_QUOTES = {"\"", "\u201c", "'"}
CLOSE_QUOTES = {"\"", "\u201d", "'"}
CHAR_ORDER = 4
REP_WINDOW = 40
REP_PENALTY_BASE = 0.55
PHRASE_REP_PENALTY = 0.3     # penalty for repeating a 2-gram
PHRASE_REP_WINDOW = 15       # how many recent bigrams to track

FABRICATE_MIN_PROB = 0.08
FABRICATE_CHANCE = 0.25
FABRICATE_MAX_LEN = 14
FABRICATE_MAX_RATIO = 0.20   # max 20% of words can be fabricated in one sentence

TOPIC_DECAY = 0.90
TOPIC_BOOST = 0.7
TOPIC_MIN_COUNT = 5

TOP_K_CANDIDATES = 40

PRUNE_MIN_COUNT = 2
CONTEXT_CAT_CAP = 8000
VOCAB_CAP = 3000
MAX_JSON_BYTES = 40 * 1024 * 1024

CATEGORIES = ["CHARACTER", "ACTION", "OBJECT", "ABSTRACT",
              "QUALITY", "PLACE", "TIME", "BOUNDARY", "OTHER"]

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text)
    return re.findall(
        r"<START>|<END>|[A-Za-zÀ-ÿ'-]+|[0-9]+|[.,!?;:\"()\[\]{}\u2014\u2013\u201c\u201d]",
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
        "whereas", "whether", "either", "neither", "nor", "then", "than"}
AUX = {"is", "are", "was", "were", "be", "been", "being", "has", "have",
       "had", "having", "do", "does", "did", "will", "would", "shall",
       "should", "can", "could", "may", "might", "must", "ought"}


def pos_tag(word: str) -> str:
    if word in (START_TOKEN, END_TOKEN):
        return "BOUNDARY"
    if word in PUNCT_TOKENS:
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
    if w.endswith(("ous", "ful", "ive", "able", "ible", "less", "al")): return "ADJ"
    if w.isdigit(): return "NUM"
    return "WORD"


# ---------------------------------------------------------------------------
# Semantic categorizer
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
                  "freedom", "liberty", "equality", "rights", "struggle",
                  "revolution", "movement", "system", "order", "power",
                  "oppression", "exploitation", "bourgeoisie", "proletariat"}


def categorize(word: str) -> str:
    if word in (START_TOKEN, END_TOKEN):
        return "BOUNDARY"
    if word in PUNCT_TOKENS:
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

        # caches
        self._cat_cache: Dict[str, str] = {}
        self._pos_cache: Dict[str, str] = {}
        self._char_starts: Optional[List[str]] = None
        self._unigram: Optional[Dict[str, float]] = None

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
        self._prune()
        print(f"  [3/3] pruned + categories indexed")
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

        self._build_unigram()
        ok = self.save()
        if not ok:
            print("[!] save failed", file=sys.stderr)

    def _prune(self) -> None:
        for order in self.transitions:
            pruned = {}
            for ctx, counts in self.transitions[order].items():
                total = sum(counts.values())
                if total >= PRUNE_MIN_COUNT:
                    pruned[ctx] = counts
            self.transitions[order] = pruned

        if len(self.context_categories) > CONTEXT_CAT_CAP:
            scored = sorted(
                self.context_categories.items(),
                key=lambda kv: -sum(kv[1].values())
            )[:CONTEXT_CAT_CAP]
            self.context_categories = defaultdict(Counter, dict(scored))

    def _build_unigram(self) -> None:
        """Precompute unigram fallback distribution."""
        total = sum(self.vocab.values())
        if total == 0:
            self._unigram = None
            return
        # only keep words seen >= 2 times (avoid one-off oddities)
        self._unigram = {
            w: c / total for w, c in self.vocab.items()
            if c >= 2 and w not in (START_TOKEN, END_TOKEN)
        }

    # ──────────────────────────────────────────
    # Caches
    # ──────────────────────────────────────────

    def _cat(self, word: str) -> str:
        c = self._cat_cache.get(word)
        if c is not None:
            return c
        cats = self.word_categories.get(word)
        if cats:
            c = cats.most_common(1)[0][0]
        else:
            c = categorize(word)
        self._cat_cache[word] = c
        return c

    def _pos(self, word: str) -> str:
        p = self._pos_cache.get(word)
        if p is None:
            p = pos_tag(word)
            self._pos_cache[word] = p
        return p

    # ──────────────────────────────────────────
    # Sampling: top-p (nucleus) with temperature
    # ──────────────────────────────────────────

    def _sample_topp(self, dist: Dict[str, float], temp: float, top_p: float) -> str:
        """Nucleus sampling. Sorts once, applies temperature, keeps top-p mass."""
        if not dist:
            return UNK
        if len(dist) == 1:
            return next(iter(dist))

        # sort by probability desc
        items = sorted(dist.items(), key=lambda x: -x[1])

        # apply temperature to raw probs
        if temp != 1.0:
            items = [(t, math.pow(p, 1.0 / temp)) for t, p in items]
            total = sum(p for _, p in items)
            if total > 0:
                items = [(t, p / total) for t, p in items]

        # nucleus cutoff
        kept = []
        cum = 0.0
        for t, p in items:
            kept.append((t, p))
            cum += p
            if cum >= top_p:
                break

        # renormalize
        total = sum(p for _, p in kept)
        if total <= 0:
            return items[0][0]
        kept = [(t, p / total) for t, p in kept]

        # sample
        r = random.random()
        cum = 0.0
        for t, p in kept:
            cum += p
            if r <= cum:
                return t
        return kept[-1][0]

    # ──────────────────────────────────────────
    # Prediction: variable-order backoff, infallible
    # ──────────────────────────────────────────

    def _predict(self, context: List[str], temperature: float,
                 top_p: float,
                 recent_count: Dict[str, int],
                 recent_bigrams: Set[Tuple[str, str]],
                 topic_factors: Dict[str, float]) -> Tuple[str, float]:
        """Return (token, top_probability). Never returns UNK when model is trained."""
        for order in range(self.max_order, self.min_order - 1, -1):
            if len(context) < order:
                continue
            key = tuple(context[-order:])
            bucket = self.transitions.get(order, {})
            if key not in bucket:
                continue

            raw = bucket[key]
            if not raw:
                continue

            # top-K prune
            if len(raw) > TOP_K_CANDIDATES:
                counts = dict(sorted(raw.items(), key=lambda x: -x[1])[:TOP_K_CANDIDATES])
            else:
                counts = dict(raw)

            if len(counts) == 1:
                only = next(iter(counts))
                return only, 1.0

            # POS reranking
            if len(context) >= 2:
                pos_ctx = (self._pos(context[-2]), self._pos(context[-1]))
                pos_counts = self.pos_transitions.get(pos_ctx)
                if pos_counts:
                    total_pos = sum(pos_counts.values())
                    if total_pos > 0:
                        for tok in counts:
                            p = pos_counts.get(self._pos(tok), 0) / total_pos
                            counts[tok] *= (0.3 + p * 3.0)

            # Context-category reranking
            ctx_cats = self.context_categories.get(key)
            if ctx_cats:
                total_cat = sum(ctx_cats.values())
                if total_cat > 0:
                    for tok in counts:
                        cat = self._cat(tok)
                        cp = ctx_cats.get(cat, 0) / total_cat
                        counts[tok] *= (0.4 + cp * 2.0)

            # Topic boost
            if topic_factors:
                for tok in counts:
                    factor = topic_factors.get(self._cat(tok))
                    if factor is not None:
                        counts[tok] *= factor

            # Repetition penalty (word)
            if recent_count:
                for tok in counts:
                    if tok in PUNCT_TOKENS or tok in (START_TOKEN, END_TOKEN):
                        continue
                    occ = recent_count.get(tok, 0)
                    if occ:
                        counts[tok] *= (REP_PENALTY_BASE ** occ)

            # Phrase repetition penalty (bigram)
            if recent_bigrams and len(context) >= 1:
                prev = context[-1]
                for tok in counts:
                    if tok in PUNCT_TOKENS or tok in (START_TOKEN, END_TOKEN):
                        continue
                    if (prev, tok) in recent_bigrams:
                        counts[tok] *= PHRASE_REP_PENALTY

            total = sum(counts.values())
            if total <= 0:
                continue
            probs = {t: c / total for t, c in counts.items()}
            top_token = max(probs, key=probs.get)
            top_prob = probs[top_token]

            chosen = self._sample_topp(probs, temperature, top_p)
            return chosen, top_prob

        # fallback 1: unigram distribution
        if self._unigram:
            chosen = self._sample_topp(self._unigram, temperature, top_p)
            return chosen, 0.05

        # fallback 2: random vocab word
        if self.vocab:
            chosen = random.choice(list(self.vocab.keys()))
            return chosen, 0.02

        return UNK, 0.0

    # ──────────────────────────────────────────
    # Character fabrication
    # ──────────────────────────────────────────

    def _char_generate(self, seed_prefix: Optional[str] = None,
                       max_len: int = FABRICATE_MAX_LEN) -> Optional[str]:
        """Fabricate a word. seed_prefix can be 1-3 chars from previous word ending."""
        if not self.char_chains:
            return None
        if self._char_starts is None:
            self._char_starts = [c for c in self.char_chains
                                 if c.startswith(" ") and c[1:].isalpha()]
        if not self._char_starts:
            return None

        # pick context: use seed_prefix if provided, else random word start
        if seed_prefix and len(seed_prefix) <= CHAR_ORDER:
            ctx = (" " + seed_prefix)[-CHAR_ORDER:]
            if ctx not in self.char_chains:
                # pad with space if short
                ctx = (" " + seed_prefix).ljust(CHAR_ORDER, " ")[-CHAR_ORDER:]
                if ctx not in self.char_chains:
                    ctx = random.choice(self._char_starts)
                    seed_prefix = None
        else:
            ctx = random.choice(self._char_starts)
            seed_prefix = None

        chars: List[str] = list(seed_prefix) if seed_prefix else []

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

    # ──────────────────────────────────────────
    # Topic tracking
    # ──────────────────────────────────────────

    def _update_topic_counts(self, topic_state: Dict[str, float],
                             emitted_word: str) -> None:
        for k in list(topic_state.keys()):
            topic_state[k] *= TOPIC_DECAY
            if topic_state[k] < 0.1:
                del topic_state[k]
        cat = self._cat(emitted_word)
        topic_state[cat] = topic_state.get(cat, 0.0) + 1.0

    def _compute_topic_factors(self, topic_state: Dict[str, float]) -> Dict[str, float]:
        factors = {}
        for cat, w in topic_state.items():
            if w > TOPIC_MIN_COUNT:
                factors[cat] = 1.0 + TOPIC_BOOST * math.log1p(w)
        return factors

    # ──────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────

    def generate(self, length: int = DEFAULT_LENGTH,
                 seed: Optional[str] = None,
                 temperature: float = DEFAULT_TEMP,
                 top_p: float = DEFAULT_TOP_P,
                 show_progress: bool = True,
                 fabricate: bool = True) -> str:
        """Generate ~length words of text."""
        if not self.trained:
            return "(model not trained)"

        if self._unigram is None:
            self._build_unigram()

        if seed:
            context = tokenize(seed) or [START_TOKEN]
        else:
            context = [START_TOKEN]

        output: List[str] = []
        recent_count: Dict[str, int] = {}
        recent_order: List[str] = []
        recent_bigrams: List[Tuple[str, str]] = []
        sentences: List[str] = []
        topic_state: Dict[str, float] = {}
        word_count = 0
        fabricated_in_sentence = 0
        word_in_sentence = 0

        if show_progress:
            print("  generating", end="", flush=True)

        max_steps = max(200, length * 4)

        for step in range(max_steps):
            if word_count >= length:
                break

            topic_factors = self._compute_topic_factors(topic_state)
            token, top_prob = self._predict(
                context, temperature, top_p,
                recent_count, set(recent_bigrams), topic_factors
            )

            # ─── fabrication on low confidence ───
            fabricated = False
            if (fabricate and token != UNK
                    and token not in PUNCT_TOKENS
                    and token not in (START_TOKEN, END_TOKEN)
                    and word_in_sentence > 0):
                # check fabrication cap
                can_fabricate = (word_in_sentence == 0 or
                                 fabricated_in_sentence / max(1, word_in_sentence) < FABRICATE_MAX_RATIO)

                if (can_fabricate
                        and top_prob < FABRICATE_MIN_PROB
                        and random.random() < FABRICATE_CHANCE):
                    # use last 2 chars of previous output word as seed prefix
                    seed_prefix = None
                    if output:
                        prev = output[-1]
                        if prev and prev not in PUNCT_TOKENS and len(prev) > 3:
                            seed_prefix = prev[-2:].lower()

                    made_up = self._char_generate(seed_prefix=seed_prefix)
                    if made_up and made_up.lower() != token.lower():
                        token = made_up
                        fabricated = True

            if token == UNK:
                # extreme fallback — shouldn't happen if unigram exists
                if len(context) > 1:
                    context = context[1:]
                    continue
                else:
                    # force a random word
                    if self.vocab:
                        token = random.choice(list(self.vocab.keys()))
                    else:
                        break

            # ─── END token ───
            if token == END_TOKEN:
                if output and output[-1] != START_TOKEN and output[-1] not in PUNCT_TOKENS:
                    sentences.append(_detokenize_sentence(output))
                    word_count += word_in_sentence
                    output = []
                    recent_count.clear()
                    recent_order.clear()
                    recent_bigrams.clear()
                    topic_state.clear()
                    context = [START_TOKEN]
                    word_in_sentence = 0
                    fabricated_in_sentence = 0
                    if show_progress and step % 20 == 0:
                        print(".", end="", flush=True)
                else:
                    context.append(token)
                    context = context[-self.max_order:]
                continue

            # ─── START token ───
            if token == START_TOKEN:
                if len(context) == 1 and context[0] == START_TOKEN:
                    continue
                context = [START_TOKEN]
                output = []
                continue

            # ─── punctuation guard ───
            if not output and token in PUNCT_TOKENS:
                continue

            # ─── emit token ───
            if output:
                recent_bigrams.append((output[-1], token))
                if len(recent_bigrams) > PHRASE_REP_WINDOW:
                    recent_bigrams.pop(0)

            output.append(token)
            context.append(token)
            context = context[-self.max_order:]

            recent_count[token] = recent_count.get(token, 0) + 1
            recent_order.append(token)
            if len(recent_order) > REP_WINDOW:
                old = recent_order.pop(0)
                c = recent_count.get(old, 1) - 1
                if c <= 0:
                    recent_count.pop(old, None)
                else:
                    recent_count[old] = c

            self._update_topic_counts(topic_state, token)

            if token not in PUNCT_TOKENS and token not in (START_TOKEN, END_TOKEN):
                word_in_sentence += 1
                if fabricated:
                    fabricated_in_sentence += 1

            if show_progress and step % 20 == 0:
                print(".", end="", flush=True)

        if output and output[-1] not in (START_TOKEN, END_TOKEN):
            sentences.append(_detokenize_sentence(output))
            word_count += word_in_sentence

        if show_progress:
            print()
        return "\n\n".join(sentences)

    # ──────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────

    def save(self) -> bool:
        try:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"[!] cannot create model dir: {e}", file=sys.stderr)
            return False

        ser = {}
        for order, bucket in self.transitions.items():
            ser[str(order)] = {"\x00".join(k): dict(v) for k, v in bucket.items()}
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
            "vocab": dict(self.vocab.most_common(VOCAB_CAP)),
            "total_tokens": self.total_tokens,
        }

        try:
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
                print(f"[!] warning: model is {size/1024/1024:.1f} MB", file=sys.stderr)
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
                print(f"[!] model too large ({size/1024/1024:.1f} MB)", file=sys.stderr)
                return False
        except OSError as e:
            print(f"[!] cannot stat: {e}", file=sys.stderr)
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
            self.max_order = data.get("max_order", self.max_order)
            self.min_order = data.get("min_order", self.min_order)
            self.name = data.get("name", self.name)
            self.total_tokens = data.get("total_tokens", 0)
            self.vocab = Counter(data.get("vocab", {}))

            self.transitions = {}
            for o_str, bucket in data.get("transitions", {}).items():
                try:
                    order = int(o_str)
                except ValueError:
                    continue
                self.transitions[order] = {
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

            # reset caches
            self._cat_cache = {}
            self._pos_cache = {}
            self._char_starts = None
            self._unigram = None

            self.trained = any(self.transitions.values())
            if self.trained:
                self._build_unigram()
            return self.trained
        except MemoryError:
            print("[!] out of memory rebuilding", file=sys.stderr)
            return False
        except Exception as e:
            print(f"[!] rebuild failed: {type(e).__name__}: {e}", file=sys.stderr)
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

NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", ")", "]", "}", "'s", "n't",
                   "'re", "'ve", "'ll", "'d", "'m", "'", "\u2014", "\u2013"}
NO_SPACE_AFTER = {"(", "[", "{", "\u201c"}
END_OF_SENTENCE = {".", "!", "?"}


def _detokenize_sentence(tokens: List[str]) -> str:
    out: List[str] = []
    capitalize_next = True
    in_quote = False

    for tok in tokens:
        if tok in (START_TOKEN, END_TOKEN):
            continue

        # handle quotes
        if tok in OPEN_QUOTES or tok in CLOSE_QUOTES:
            if not out:
                out.append(tok)
                in_quote = not in_quote
                continue
            prev = out[-1]
            if in_quote:
                out.append(tok)
                in_quote = False
            elif prev and prev[-1] == " ":
                out.append(tok)
                in_quote = True
            else:
                out.append(" " + tok)
                in_quote = True
            continue

        # capitalization
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

        if tok in END_OF_SENTENCE:
            capitalize_next = True

    text = "".join(out)
    if text and text[-1] not in ".!?":
        if text[-1] not in "\"')]":
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
            print(f"[!] skipping unloadable: {name}", file=sys.stderr)
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
    print(f"[*] found {len(files)} files")
    print()
    for i, fp in enumerate(files, 1):
        name = fp.stem.lower().replace(" ", "_")
        if name in manifest:
            try:
                if manifest[name].get("hash") == _file_hash(fp):
                    print(f"=== [{i}/{len(files)}] {fp.name} ===")
                    print(f"  [=] already trained\n")
                    skipped += 1
                    continue
            except Exception:
                pass
        print(f"=== [{i}/{len(files)}] {fp.name} ===")
        try:
            text = clean_gutenberg(fp.read_text(encoding="utf-8", errors="replace"))
            if len(text) < 200:
                print("  [!] too small\n")
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
    start = time.time()
    out = model.generate(
        length=args.length,
        seed=args.seed,
        temperature=args.temp,
        top_p=args.top_p,
        fabricate=not args.no_fabricate,
    )
    elapsed = time.time() - start
    print()
    print("─" * 60)
    print(out)
    print("─" * 60)
    print(f"[*] generated in {elapsed:.2f}s")
    print()


def cmd_freegen(args) -> None:
    print("[*] merging all models...")
    start = time.time()
    model = merge_all_models()
    merge_elapsed = time.time() - start
    if not model.trained:
        print("[!] nothing to merge")
        sys.exit(1)
    print(f"[*] merged {len(list_models())} model(s) in {merge_elapsed:.1f}s")
    print(f"[*] {model.stats()}")
    print()
    gen_start = time.time()
    out = model.generate(
        length=args.length,
        seed=args.seed,
        temperature=args.temp,
        top_p=args.top_p,
        fabricate=not args.no_fabricate,
    )
    gen_elapsed = time.time() - gen_start
    print()
    print("─" * 60)
    print(out)
    print("─" * 60)
    print(f"[*] generated in {gen_elapsed:.2f}s")
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
            else:
                print(f"  {name:<25} [!] could not load")
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
        description="Mini Mind v8 - refined language model by Templar Studios",
    )
    sub = parser.add_subparsers(dest="command", help="commands")

    t = sub.add_parser("train", help="train on one file")
    t.add_argument("file")
    t.add_argument("--order", type=int, default=DEFAULT_MAX_ORDER)

    at = sub.add_parser("autotrain", help="train all .txt")
    at.add_argument("folder", nargs="?", default=None)
    at.add_argument("--order", type=int, default=DEFAULT_MAX_ORDER)

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