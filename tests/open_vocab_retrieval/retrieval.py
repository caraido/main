# -*- coding: utf-8 -*-
"""
tests/open_vocab_retrieval/retrieval.py
=======================================
Step 3 of the guide: nearest-neighbour retrieval of the decoded embedding
against the open gallery, and tie-safe rank computation.

Cosine similarity is kept as the metric (consistent with the ``kernel_pls_cosine``
training objective).  The project's canonical mean-centre convention — subtract
the gallery centroid from BOTH the gallery and the query predictions before
cosine — is reused from ``utils.retrieval`` (a model that always predicts the
centroid then scores at chance rather than deceptively well).

Ranks use *competition ranking* with strict ``>`` on similarity: the true word's
rank is ``1 + #{gallery words strictly more similar than the true word}``.  Ties
therefore never inflate the rank, and the best possible rank is 1.
"""

from __future__ import annotations

import os
import sys
from typing import Sequence

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_THIS_DIR)
_MAIN_DIR = os.path.dirname(_TESTS_DIR)
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

from utils.retrieval import mean_center_db, normalize_rows  # noqa: E402


def similarity_matrix(pred_emb: np.ndarray, gallery_emb: np.ndarray,
                      center: bool = True) -> np.ndarray:
    """Cosine similarity ``(T, N)`` between predictions and the gallery.

    When ``center`` (default) the gallery centroid is removed from both sides
    first — the canonical convention shared with ``utils.retrieval``.
    """
    pred_emb = np.asarray(pred_emb, dtype=np.float64)
    gallery_emb = np.asarray(gallery_emb, dtype=np.float64)
    if center:
        gallery_emb, pred_emb, _ = mean_center_db(gallery_emb, pred_emb)
    sims = normalize_rows(pred_emb) @ normalize_rows(gallery_emb).T
    if not np.all(np.isfinite(sims)):
        raise FloatingPointError(
            "Non-finite cosine similarities — check for zero/NaN embedding rows.")
    return sims


def true_indices(true_word: Sequence[str], word_to_index: dict) -> np.ndarray:
    """Map each trial's true (clean) word to its gallery index.

    Missing words yield ``-1``; the caller must decide how to handle them rather
    than silently dropping — a true word absent from the gallery has no rank.
    """
    return np.array([word_to_index.get(str(w), -1) for w in true_word], dtype=np.int64)


def compute_ranks(sims: np.ndarray, true_idx: np.ndarray) -> np.ndarray:
    """Tie-safe competition rank of the true word per trial (1 = best).

    ``rank_t = 1 + #{ j : sims[t, j] > sims[t, true_idx[t]] }``.
    Trials whose ``true_idx`` is ``-1`` (true word not in gallery) get rank ``-1``
    so downstream metrics can exclude them explicitly.
    """
    sims = np.asarray(sims, dtype=np.float64)
    true_idx = np.asarray(true_idx, dtype=np.int64)
    T = sims.shape[0]
    if len(true_idx) != T:
        raise ValueError(f"true_idx length {len(true_idx)} != n_trials {T}")
    rank = np.full(T, -1, dtype=np.int64)
    valid = true_idx >= 0
    if valid.any():
        rows = np.where(valid)[0]
        true_sim = sims[rows, true_idx[valid]][:, None]
        rank[rows] = 1 + (sims[rows] > true_sim).sum(axis=1)
    return rank


def ranked_indices(sims: np.ndarray) -> np.ndarray:
    """Full gallery ranking per trial (indices sorted by descending similarity).

    Used by graded / qualitative analyses (nDCG, top-5 tables).
    """
    return np.argsort(-np.asarray(sims, dtype=np.float64), axis=1)


def retrieve(pred_emb: np.ndarray, gallery, true_word: Sequence[str],
             center: bool = True):
    """Convenience wrapper: returns ``(sims, rank, true_idx)`` for a Gallery.

    ``gallery`` is a :class:`gallery.Gallery`.
    """
    sims = similarity_matrix(pred_emb, gallery.emb, center=center)
    tidx = true_indices(true_word, gallery.word_to_index)
    rank = compute_ranks(sims, tidx)
    return sims, rank, tidx
