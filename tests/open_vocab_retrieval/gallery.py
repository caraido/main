# -*- coding: utf-8 -*-
"""
tests/open_vocab_retrieval/gallery.py
=====================================
Step 1 of the open-vocabulary retrieval guide: build the open gallery.

Produces, for a requested size ``N`` and variant:

  * ``gallery_words``  list[str]  length N, lemmatized, deduped, INCLUDES the
                       stimulus wordset (so the true word is always retrievable).
  * ``gallery_emb``    ndarray (N, D)  same embedding model as the decode target
                       (GloVe 840B), L2 usable; rows are the raw GloVe vectors.
  * ``gallery_meta``   DataFrame indexed like ``gallery_words`` with per-word
                       ``freq_rank`` (GloVe frequency rank), ``log_freq``
                       (frequency proxy, higher = more frequent), ``concreteness``
                       (NaN if norms unavailable), ``pos`` and ``is_stimulus``.

Design choices / provenance
---------------------------
* **Decode target = language embedding** (GloVe 840B), per guide §3.1.  The
  project's kernel_pls runs regress neural HGA onto exactly these GloVe vectors,
  so the gallery must use the same model for query/gallery to share a space.
* **Frequency = GloVe vocabulary rank.**  torchtext's GloVe ``itos`` list is
  ordered by descending corpus frequency (``the`` -> rank ~2, ``dog`` -> ~1157,
  ``platypus`` -> ~109072), so the index into ``itos`` is a monotone proxy for
  word frequency.  This keeps the pipeline self-contained (no SUBTLEX download).
  ``log_freq = -log10(rank + 1)`` is higher for more frequent words.  An external
  SUBTLEX file can be supplied via :func:`load_subtlex` to override the proxy.
* **Concreteness** (Brysbaert et al. 2014) is optional: supplied via a norms
  file (:func:`load_concreteness`).  Without it, the *matched* gallery falls back
  to POS(noun) + frequency-band matching and the concreteness filter is SKIPPED
  with a loud warning (never silently) — the ``concreteness`` column is NaN.
* **POS filter = WordNet noun.**  A word is kept as a noun iff it has at least
  one noun synset (offline, no tagger model needed); matches the concrete-noun
  stimuli (guide §3.3).

Two gallery variants (guide §3.3), both include the stimulus wordset and are
lemmatized/deduped:
  ``raw``     : the N most frequent content nouns, minimally filtered.
  ``matched`` : POS-matched concrete nouns, concreteness-filtered (if norms
                given), sampled within the stimulus frequency band.

References: Brysbaert et al. (2014) concreteness; Brysbaert & New (2009) SUBTLEX.
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# ── Path setup (match the repo convention) ────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_THIS_DIR)
_MAIN_DIR = os.path.dirname(_TESTS_DIR)
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

PROJECT_ROOT = Path(_MAIN_DIR)
DEFAULT_GLOVE_CACHE = PROJECT_ROOT / ".vector_cache"
GLOVE_DIM = 300
GLOVE_NAME = "840B"

# Suffix disambiguation stripping: labels look like ``mouse(object/tool)``.
_SUFFIX_RE = re.compile(r"\(.*?\)")


# ══════════════════════════════════════════════════════════════════════════
# Embedding model (GloVe) — the decode target and gallery share this
# ══════════════════════════════════════════════════════════════════════════

_GLOVE_SINGLETON = None


def clean_word(label: str) -> str:
    """Strip a ``(category)`` disambiguation suffix and normalise a label.

    ``mouse(object/tool)`` -> ``mouse``.  Lower-cased and whitespace-stripped so
    gallery lookups are stable.
    """
    return _SUFFIX_RE.sub("", str(label)).strip().lower()


def load_glove(cache: Optional[Path] = None):
    """Load (and memoise) the torchtext GloVe 840B model from the local cache.

    The vectors are already downloaded under ``main/.vector_cache`` by the main
    pipeline; loading is ~10 s and ~5 GB RAM.
    """
    global _GLOVE_SINGLETON
    if _GLOVE_SINGLETON is not None:
        return _GLOVE_SINGLETON
    cache = Path(cache) if cache is not None else DEFAULT_GLOVE_CACHE
    if not cache.exists():
        raise FileNotFoundError(
            f"GloVe cache not found at {cache}. Expected the project's "
            "main/.vector_cache with glove.840B.300d vectors.")
    import torchtext
    try:
        torchtext.disable_torchtext_deprecation_warning()
    except Exception:
        pass
    from torchtext.vocab import GloVe
    _GLOVE_SINGLETON = GloVe(dim=GLOVE_DIM, name=GLOVE_NAME, cache=str(cache))
    return _GLOVE_SINGLETON


def glove_vector(glove, word: str) -> np.ndarray:
    """GloVe vector for a single word (zero vector if OOV)."""
    return glove[word].numpy().astype(np.float64)


def glove_embed(glove, words: Sequence[str]):
    """Embed a list of words. Returns ``(emb (n,D) float64, oov_mask (n,) bool)``.

    OOV words (absent from GloVe) receive a zero vector and are flagged in
    ``oov_mask`` so the caller can drop/report them rather than silently keep a
    meaningless zero row.
    """
    emb = np.zeros((len(words), GLOVE_DIM), dtype=np.float64)
    oov = np.zeros(len(words), dtype=bool)
    for i, w in enumerate(words):
        v = glove[w].numpy()
        emb[i] = v
        if not np.any(v):
            oov[i] = True
    return emb, oov


def frequency_rank(glove, word: str) -> Optional[int]:
    """GloVe vocabulary rank (0 = most frequent). ``None`` if the word is OOV."""
    return glove.stoi.get(word, None)


# ══════════════════════════════════════════════════════════════════════════
# Lexical filters (WordNet POS + lemmatization)
# ══════════════════════════════════════════════════════════════════════════

_WN = None
_LEMMATIZER = None
_STOPWORDS = None


def _wordnet():
    global _WN
    if _WN is None:
        from nltk.corpus import wordnet as wn
        wn.ensure_loaded()
        _WN = wn
    return _WN


def _lemmatizer():
    global _LEMMATIZER
    if _LEMMATIZER is None:
        from nltk.stem import WordNetLemmatizer
        _LEMMATIZER = WordNetLemmatizer()
    return _LEMMATIZER


def _stopwords() -> set:
    global _STOPWORDS
    if _STOPWORDS is None:
        from nltk.corpus import stopwords
        _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


def is_noun(word: str) -> bool:
    """True iff the word has at least one WordNet NOUN synset (offline POS)."""
    wn = _wordnet()
    return len(wn.synsets(word, pos=wn.NOUN)) > 0


def is_noun_dominant(word: str) -> bool:
    """True iff the word's WordNet senses are majority NOUN.

    Stricter than :func:`is_noun`: excludes words whose noun reading is marginal
    (e.g. 'run', 'have', 'be'), which keeps the gallery a concrete-noun lexicon
    and rejects function words that merely happen to have an obscure noun sense
    (``in`` -> indium, ``a`` -> angstrom).
    """
    wn = _wordnet()
    syns = wn.synsets(word)
    if not syns:
        return False
    n_noun = sum(1 for s in syns if s.pos() == "n")
    return n_noun > 0 and n_noun >= (len(syns) - n_noun)


def lemmatize(word: str) -> str:
    """Noun-lemmatize (``mice`` -> ``mouse``); lower-cased."""
    return _lemmatizer().lemmatize(word.lower(), pos="n")


def lemmatize_and_dedupe(words: Sequence[str]) -> List[str]:
    """Lemmatize each word (noun sense), lower-case, and dedupe preserving the
    first (most frequent, when the input is frequency-ordered) occurrence."""
    seen: set = set()
    out: List[str] = []
    for w in words:
        lw = lemmatize(w)
        if lw and lw not in seen:
            seen.add(lw)
            out.append(lw)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Optional external norms (concreteness, SUBTLEX frequency)
# ══════════════════════════════════════════════════════════════════════════

def load_concreteness(path: Optional[Path]) -> Optional[Dict[str, float]]:
    """Load Brysbaert et al. (2014) concreteness ratings from a norms file.

    Accepts the canonical tab/comma file with a ``Word`` column and a
    ``Conc.M`` (mean concreteness) column.  Returns ``{lemma: concreteness}``
    or ``None`` if *path* is falsy / missing (caller must warn, not silence).
    """
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        warnings.warn(f"Concreteness norms file not found: {path}. "
                      "Concreteness filtering will be skipped.")
        return None
    sep = "\t" if path.suffix.lower() in {".txt", ".tsv"} else ","
    df = pd.read_csv(path, sep=sep)
    wcol = next((c for c in df.columns if c.lower() in {"word", "lemma"}), None)
    ccol = next((c for c in df.columns
                 if c.lower().replace(" ", "") in {"conc.m", "concm", "concreteness"}), None)
    if wcol is None or ccol is None:
        raise ValueError(
            f"Concreteness file {path} lacks recognisable Word/Conc.M columns "
            f"(found {list(df.columns)}).")
    out = {}
    for w, c in zip(df[wcol].astype(str), pd.to_numeric(df[ccol], errors="coerce")):
        if not np.isnan(c):
            out[w.strip().lower()] = float(c)
    return out


def load_subtlex(path: Optional[Path]) -> Optional[Dict[str, float]]:
    """Load SUBTLEX-US log-frequency (Lg10WF) from a norms file, if provided.

    Returns ``{lemma: log10_word_frequency}`` or ``None``.  When absent the
    GloVe-rank proxy is used instead (documented in the module docstring).
    """
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        warnings.warn(f"SUBTLEX file not found: {path}. Using GloVe-rank "
                      "frequency proxy instead.")
        return None
    sep = "\t" if path.suffix.lower() in {".txt", ".tsv"} else ","
    df = pd.read_csv(path, sep=sep)
    wcol = next((c for c in df.columns if c.lower() in {"word", "lemma"}), None)
    fcol = next((c for c in df.columns
                 if c.lower().replace(" ", "") in {"lg10wf", "log10wf", "logfreq"}), None)
    if wcol is None or fcol is None:
        raise ValueError(
            f"SUBTLEX file {path} lacks Word / Lg10WF columns "
            f"(found {list(df.columns)}).")
    out = {}
    for w, f in zip(df[wcol].astype(str), pd.to_numeric(df[fcol], errors="coerce")):
        if not np.isnan(f):
            out[w.strip().lower()] = float(f)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Gallery object
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Gallery:
    """A built open gallery: words, embeddings, per-word meta, and lookup."""
    variant: str
    words: List[str]
    emb: np.ndarray                      # (N, D) float64
    meta: pd.DataFrame                   # per-word: freq_rank, log_freq, concreteness, pos, is_stimulus
    word_to_index: Dict[str, int] = field(default_factory=dict)

    @property
    def N(self) -> int:
        return len(self.words)

    def index_of(self, word: str) -> Optional[int]:
        return self.word_to_index.get(clean_word(word), None)


def _log_freq_from_rank(rank: int) -> float:
    """Monotone frequency proxy from GloVe rank (higher = more frequent)."""
    return float(-np.log10(rank + 1.0))


_CANDIDATE_CACHE: Dict[tuple, pd.DataFrame] = {}


def _candidate_nouns(glove, max_scan: int, concreteness: Optional[Dict[str, float]],
                     min_concreteness: float, require_concreteness: bool
                     ) -> pd.DataFrame:
    """Scan the GloVe vocabulary (frequency order) and collect alphabetic noun
    lemmas with their rank/log_freq/concreteness.  Deduped by lemma (keeps the
    most frequent surface form).

    The scan is the expensive step (hundreds of thousands of WordNet lookups), so
    the result is memoised per (vocab, scan-depth, concreteness filter) — a
    gallery-size sweep reuses one scan.
    """
    key = (id(glove), int(max_scan), bool(require_concreteness),
           float(min_concreteness), id(concreteness) if require_concreteness else 0)
    if key in _CANDIDATE_CACHE:
        return _CANDIDATE_CACHE[key].copy()
    stops = _stopwords()
    seen: set = set()
    recs: List[dict] = []
    for rank, tok in enumerate(glove.itos):
        if rank >= max_scan:
            break
        tok = tok.lower()
        if not tok.isalpha() or len(tok) < 3 or tok in stops:
            continue
        lw = lemmatize(tok)
        if lw in seen or len(lw) < 3 or lw in stops:
            continue
        if not is_noun_dominant(lw):
            continue
        # The gallery key is the lemmatized/lower-cased form, but GloVe 840B is
        # case-sensitive: a token like ``Ephesian`` has a vector while its lemma
        # ``ephesian`` does not.  Require the KEY itself to embed, so no gallery
        # word is ever a zero (OOV) row.
        if not np.any(glove[lw].numpy()):
            continue
        conc = concreteness.get(lw) if concreteness is not None else np.nan
        if require_concreteness and (conc is None or np.isnan(conc) or conc < min_concreteness):
            continue
        seen.add(lw)
        recs.append({"word": lw, "freq_rank": rank,
                     "log_freq": _log_freq_from_rank(rank),
                     "concreteness": np.nan if conc is None else conc})
    df = pd.DataFrame(recs)
    _CANDIDATE_CACHE[key] = df
    return df.copy()


def build_gallery(glove, stimulus_words: Sequence[str], n: int = 5000,
                  variant: str = "matched",
                  concreteness: Optional[Dict[str, float]] = None,
                  min_concreteness: float = 4.0,
                  freq_band_quantiles=(0.05, 0.95),
                  max_scan: int = 400_000,
                  subtlex: Optional[Dict[str, float]] = None) -> Gallery:
    """Build an open gallery of size ~``n`` (guide §3.3, Step 1).

    Parameters
    ----------
    stimulus_words : the decode stimulus lemmas (already cleaned/lemmatized by
        the caller); always included so the true word is retrievable.
    variant : ``"raw"`` (top-N frequency nouns, minimal filter) or ``"matched"``
        (concreteness-filtered, frequency-band-matched to the stimuli).
    concreteness : ``{lemma: score}`` or None.  If None and variant=="matched",
        the concreteness filter is skipped with a warning.
    """
    if variant not in {"raw", "matched"}:
        raise ValueError(f"variant must be 'raw' or 'matched', got {variant!r}")

    stim = lemmatize_and_dedupe([clean_word(w) for w in stimulus_words])
    stim_emb, stim_oov = glove_embed(glove, stim)
    if stim_oov.any():
        bad = [w for w, o in zip(stim, stim_oov) if o]
        warnings.warn(f"{stim_oov.sum()} stimulus word(s) are OOV in GloVe and "
                      f"cannot be gallery entries: {bad}. They are dropped from "
                      "the gallery; trials with these true words will be flagged.")
    stim = [w for w, o in zip(stim, stim_oov) if not o]

    require_conc = (variant == "matched" and concreteness is not None)
    if variant == "matched" and concreteness is None:
        warnings.warn("Matched gallery requested but no concreteness norms were "
                      "provided; skipping the concreteness filter (POS + "
                      "frequency-band matching only). concreteness column = NaN.")

    cand = _candidate_nouns(glove, max_scan, concreteness, min_concreteness, require_conc)
    if cand.empty:
        raise RuntimeError("No candidate nouns found while scanning GloVe — "
                           "check WordNet data and the scan limit.")

    # Frequency-band matching for the 'matched' variant.
    if variant == "matched":
        stim_ranks = np.array([frequency_rank(glove, w) for w in stim
                               if frequency_rank(glove, w) is not None], dtype=float)
        if len(stim_ranks) >= 2:
            lo = np.quantile(stim_ranks, freq_band_quantiles[0])
            hi = np.quantile(stim_ranks, freq_band_quantiles[1])
            in_band = cand[(cand["freq_rank"] >= lo) & (cand["freq_rank"] <= hi)]
            if len(in_band) >= n:
                cand = in_band
            else:
                warnings.warn(
                    f"Frequency band [{lo:.0f},{hi:.0f}] yields only "
                    f"{len(in_band)} matched nouns (< N={n}); widening beyond the "
                    "band to reach N.")
                # keep in-band first, then nearest-rank out-of-band fillers
                cand = pd.concat([in_band, cand.drop(in_band.index)], ignore_index=True)

    cand = cand.sort_values("freq_rank").reset_index(drop=True)

    # Take the top (most frequent) candidates up to N, minus the slots reserved
    # for stimulus words that are not already in the pool.
    stim_set = set(stim)
    non_stim = cand[~cand["word"].isin(stim_set)]
    n_reserve = len(stim_set)
    n_take = max(0, n - n_reserve)
    chosen_non_stim = non_stim.head(n_take)

    # Stimulus rows (their true rank/log_freq/concreteness).
    stim_recs = []
    for w in stim:
        r = frequency_rank(glove, w)
        conc = concreteness.get(w) if concreteness is not None else np.nan
        stim_recs.append({"word": w,
                          "freq_rank": np.nan if r is None else r,
                          "log_freq": np.nan if r is None else _log_freq_from_rank(r),
                          "concreteness": np.nan if conc is None else conc})
    stim_df = pd.DataFrame(stim_recs)

    meta = pd.concat([stim_df, chosen_non_stim], ignore_index=True)
    meta = meta.drop_duplicates(subset="word", keep="first").reset_index(drop=True)
    meta["is_stimulus"] = meta["word"].isin(stim_set)
    meta["pos"] = "noun"

    # SUBTLEX override of log_freq where available.
    if subtlex is not None:
        meta["log_freq"] = [subtlex.get(w, lf)
                            for w, lf in zip(meta["word"], meta["log_freq"])]

    words = meta["word"].tolist()
    emb, oov = glove_embed(glove, words)
    if oov.any():
        # Should not happen (candidates came from GloVe), but never keep zero rows.
        raise RuntimeError(f"{oov.sum()} gallery words unexpectedly OOV in GloVe: "
                           f"{[w for w, o in zip(words, oov) if o][:10]}")

    word_to_index = {w: i for i, w in enumerate(words)}
    return Gallery(variant=variant, words=words, emb=emb, meta=meta,
                   word_to_index=word_to_index)
