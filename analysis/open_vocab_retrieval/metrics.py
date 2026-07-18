# -*- coding: utf-8 -*-
"""
tests/open_vocab_retrieval/metrics.py
=====================================
Steps 4-5 of the guide: rank-based retrieval metrics (primary) and graded
near-miss metrics with INDEPENDENT relevance.

Rank metrics (Step 4) — provenance noted for reviewers:
  * ``median_percentile`` = median(rank / N)  — PRIMARY, ~N-invariant headline
    (adding unrelated distractors scales rank ~linearly with N, so rank/N is
    stable); chance = 0.5.
  * ``top{k}`` = mean(rank <= k) = CMC(k) (Zheng et al. 2015) = top-k rank
    accuracy from neural decoding; chance = k/N.
  * ``MRR`` = mean(1/rank); ``MedR`` = median rank (CLIP-style cross-modal read).

Graded metrics (Step 5):
  * ``ndcg_independent`` — nDCG (Järvelin & Kekäläinen 2002) of the neural
    ranking under a relevance grade from a space INDEPENDENT of the decode
    target (WordNet); guards Claim 3 against circularity.
  * ``category_hit_at_k`` / ``category_ap`` — binary category-membership metrics.
  * ``near_miss_similarity`` — mean independent similarity of the top-k retrieved
    words to the true word (compared to a matched null in ``stats``).

All rank inputs are the tie-safe competition ranks from ``retrieval``; trials
with rank ``-1`` (true word not in gallery) are excluded EXPLICITLY with a
reported count — never silently.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np


# ── Step 4: rank metrics ──────────────────────────────────────────────────

def _valid_ranks(rank: np.ndarray):
    rank = np.asarray(rank, dtype=np.int64)
    valid = rank > 0
    return rank[valid], int((~valid).sum())


def rank_metrics(rank: np.ndarray, N: int,
                 ks: Sequence[int] = (1, 5, 10, 50, 100)) -> Dict[str, float]:
    """Aggregate rank metrics for a set of trials.

    Returns the metrics plus ``n_trials`` and ``n_excluded`` (true word not in
    gallery).  Chance references: ``mean_percentile``/``median_percentile`` = 0.5,
    ``top{k}`` = k/N, ``median_rank`` = N/2.
    """
    r, n_excl = _valid_ranks(rank)
    if len(r) == 0:
        out = {"median_rank": np.nan, "mean_rank": np.nan,
               "mean_percentile": np.nan, "median_percentile": np.nan,
               "MedR": np.nan, "MRR": np.nan}
        out.update({f"top{k}": np.nan for k in ks})
        out.update({"n_trials": 0, "n_excluded": n_excl, "N_gallery": int(N)})
        return out
    pct = r / float(N)
    out = {
        "median_rank": float(np.median(r)),
        "mean_rank": float(np.mean(r)),
        "mean_percentile": float(np.mean(pct)),
        "median_percentile": float(np.median(pct)),
        "MedR": float(np.median(r)),
        "MRR": float(np.mean(1.0 / r)),
    }
    for k in ks:
        out[f"top{k}"] = float(np.mean(r <= k))
    out.update({"n_trials": int(len(r)), "n_excluded": int(n_excl),
                "N_gallery": int(N)})
    return out


def chance_rank_metrics(N: int, ks: Sequence[int] = (1, 5, 10, 50, 100)
                        ) -> Dict[str, float]:
    """Analytic chance references for the rank metrics at gallery size N."""
    out = {"median_rank": N / 2.0, "mean_rank": (N + 1) / 2.0,
           "mean_percentile": 0.5, "median_percentile": 0.5,
           "MedR": N / 2.0, "MRR": np.nan}
    for k in ks:
        out[f"top{k}"] = k / float(N)
    return out


# ── Step 5: graded near-miss metrics (independent relevance) ──────────────

def _dcg(rels: np.ndarray) -> float:
    rels = np.asarray(rels, dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, len(rels) + 2))
    return float(np.sum(rels * discounts))


def ndcg_independent(order_row: np.ndarray, true_word: str,
                     gallery_words: Sequence[str],
                     rel_fn: Callable[[str, str], float], k: int = 100) -> float:
    """nDCG@k of ONE trial's neural ranking under an independent relevance grade.

    ``order_row`` is the gallery indices sorted by descending neural similarity
    (from ``retrieval.ranked_indices``).  ``rel_fn`` is a WordNet grader
    independent of the decode embedding.  Returns NaN if the ideal DCG is 0
    (no gallery word has any relevance to the true word — undefined, reported).
    """
    order_row = np.asarray(order_row, dtype=np.int64)
    topk = order_row[:k]
    rels = np.array([rel_fn(true_word, gallery_words[j]) for j in topk], dtype=np.float64)
    # Ideal ordering: the k highest relevances over the WHOLE gallery.
    all_rel = np.array([rel_fn(true_word, w) for w in gallery_words], dtype=np.float64)
    ideal = np.sort(all_rel)[::-1][:k]
    idcg = _dcg(ideal)
    if idcg <= 0:
        return np.nan
    return _dcg(rels) / idcg


def category_hit_at_k(order_row: np.ndarray, true_word: str,
                      gallery_words: Sequence[str],
                      rel_fn: Callable[[str, str], float], k: int = 10) -> float:
    """Fraction of the top-k retrieved words that share the true word's category.

    ``rel_fn`` should be the binary ``category`` grader (1.0 = same superordinate).
    The true word itself, if present in the top-k, is excluded so a perfect rank-1
    does not trivially count as its own neighbour.
    """
    order_row = np.asarray(order_row, dtype=np.int64)
    hits, seen = [], 0
    for j in order_row:
        w = gallery_words[j]
        if w == true_word:
            continue
        hits.append(rel_fn(true_word, w))
        seen += 1
        if seen >= k:
            break
    return float(np.mean(hits)) if hits else np.nan


def near_miss_similarity(order_row: np.ndarray, true_word: str,
                         gallery_words: Sequence[str],
                         rel_fn: Callable[[str, str], float], k: int = 10) -> float:
    """Mean independent similarity between the true word and its top-k retrieved
    neighbours (true word itself excluded).  Compared to a matched null in
    ``stats.near_miss_null``."""
    order_row = np.asarray(order_row, dtype=np.int64)
    sims, seen = [], 0
    for j in order_row:
        w = gallery_words[j]
        if w == true_word:
            continue
        sims.append(rel_fn(true_word, w))
        seen += 1
        if seen >= k:
            break
    return float(np.mean(sims)) if sims else np.nan


def aggregate_graded(order: np.ndarray, true_word: Sequence[str],
                     gallery_words: Sequence[str],
                     rel_fn: Callable[[str, str], float],
                     valid: np.ndarray, k: int = 100) -> Dict[str, float]:
    """Mean nDCG@k and near-miss similarity over the valid trials of a patient."""
    order = np.asarray(order)
    ndcgs, nms = [], []
    for t in np.where(valid)[0]:
        ndcgs.append(ndcg_independent(order[t], true_word[t], gallery_words, rel_fn, k=k))
        nms.append(near_miss_similarity(order[t], true_word[t], gallery_words,
                                        rel_fn, k=min(k, 10)))
    ndcgs = np.array(ndcgs, dtype=np.float64)
    nms = np.array(nms, dtype=np.float64)
    return {
        "ndcg_mean": float(np.nanmean(ndcgs)) if np.any(~np.isnan(ndcgs)) else np.nan,
        "ndcg_n_defined": int(np.sum(~np.isnan(ndcgs))),
        "near_miss_sim_mean": float(np.nanmean(nms)) if np.any(~np.isnan(nms)) else np.nan,
        "n_trials": int(valid.sum()),
    }
