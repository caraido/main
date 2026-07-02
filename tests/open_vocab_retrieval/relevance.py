# -*- coding: utf-8 -*-
"""
tests/open_vocab_retrieval/relevance.py
=======================================
Step 5 of the guide: INDEPENDENT graded-relevance functions for the near-miss
analysis (Claim 3).

The circularity trap: if "how semantically similar is a retrieved word to the
true word" is graded in the SAME embedding used for decoding, then sensible
neighbours are guaranteed by construction and prove nothing about the brain.
The relevance signal must be independent of the decode target (GloVe).

These graders use **WordNet** (Miller 1995) — a non-embedding, externally
defined taxonomy:

  * ``wup``    Wu & Palmer (1994) — depth of the least common subsumer.
  * ``path``   inverse shortest-path length in the taxonomy.
  * ``lin``    Lin (1998) information-content similarity (Brown-corpus IC).
  * ``resnik`` Resnik (1995) IC of the least common subsumer.
  * ``category`` binary: do the two words share a noun hypernym at/above a depth
                 threshold (superordinate category membership)?  Immune to the
                 circularity trap and enables category-level Hit@k / mAP.

Word similarity is the MAX over noun-synset pairs (standard practice).  A missing
value (no common path) maps to ``0.0`` relevance — a defined, documented outcome
(no relatable meaning), not a silenced error.  Results are cached per unordered
word pair.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Callable, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_THIS_DIR)
_MAIN_DIR = os.path.dirname(_TESTS_DIR)
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

RELEVANCE_KINDS = ("wup", "path", "lin", "resnik", "category")

_WN = None
_IC = None


def _wordnet():
    global _WN
    if _WN is None:
        from nltk.corpus import wordnet as wn
        wn.ensure_loaded()
        _WN = wn
    return _WN


def _ic():
    global _IC
    if _IC is None:
        from nltk.corpus import wordnet_ic
        _IC = wordnet_ic.ic("ic-brown.dat")
    return _IC


def _noun_synsets(word: str):
    wn = _wordnet()
    return wn.synsets(word, pos=wn.NOUN)


def _max_pair(a: str, b: str, fn) -> Optional[float]:
    """Max of ``fn(sa, sb)`` over the two words' noun-synset pairs (None if no
    pair yields a value)."""
    sa = _noun_synsets(a)
    sb = _noun_synsets(b)
    best: Optional[float] = None
    for x in sa:
        for y in sb:
            v = fn(x, y)
            if v is not None and (best is None or v > best):
                best = v
    return best


@lru_cache(maxsize=2_000_000)
def wup_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    v = _max_pair(a, b, lambda x, y: x.wup_similarity(y))
    return 0.0 if v is None else float(v)


@lru_cache(maxsize=2_000_000)
def path_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    v = _max_pair(a, b, lambda x, y: x.path_similarity(y))
    return 0.0 if v is None else float(v)


@lru_cache(maxsize=2_000_000)
def lin_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    ic = _ic()
    def fn(x, y):
        try:
            return x.lin_similarity(y, ic)
        except Exception:
            # Lin is undefined across POS / disjoint IC roots — a defined "no
            # comparable meaning" outcome, not a silenced numerical error.
            return None
    v = _max_pair(a, b, fn)
    return 0.0 if v is None else float(v)


@lru_cache(maxsize=2_000_000)
def resnik_similarity(a: str, b: str) -> float:
    ic = _ic()
    def fn(x, y):
        try:
            return x.res_similarity(y, ic)
        except Exception:
            return None
    v = _max_pair(a, b, fn)
    return 0.0 if v is None else float(v)


@lru_cache(maxsize=2_000_000)
def shares_category(a: str, b: str, min_depth: int = 4) -> float:
    """1.0 if the two words share a noun hypernym at depth >= ``min_depth``.

    Uses the least common subsumer of the most-frequent noun sense of each word;
    ``min_depth`` keeps the shared node specific enough to be a real superordinate
    (e.g. 'animal', 'tool') rather than the trivial root 'entity'.
    """
    if a == b:
        return 1.0
    sa = _noun_synsets(a)
    sb = _noun_synsets(b)
    if not sa or not sb:
        return 0.0
    x, y = sa[0], sb[0]
    lcs = x.lowest_common_hypernyms(y)
    if not lcs:
        return 0.0
    return 1.0 if max(s.min_depth() for s in lcs) >= min_depth else 0.0


_KIND_TO_FN = {
    "wup": wup_similarity,
    "path": path_similarity,
    "lin": lin_similarity,
    "resnik": resnik_similarity,
    "category": shares_category,
}


def make_relevance_fn(kind: str = "wup") -> Callable[[str, str], float]:
    """Return a cached ``rel_fn(true_word, other_word) -> float`` for a kind."""
    if kind not in _KIND_TO_FN:
        raise ValueError(f"Unknown relevance kind {kind!r}; options {RELEVANCE_KINDS}")
    return _KIND_TO_FN[kind]


def has_wordnet_noun(word: str) -> bool:
    return len(_noun_synsets(word)) > 0
