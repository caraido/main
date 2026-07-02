# -*- coding: utf-8 -*-
"""
tests/open_vocab_retrieval/stats.py
===================================
Step 6 of the guide: significance and confound control.

  (a) **Permutation null** (Maris & Oostenveld 2007, the framework already used
      in the project).  The trial->true-word correspondence is shuffled WITHIN
      patient and WITHIN cv-fold, so a trial's predicted embedding is scored
      against a random other trial's true word; the aggregate statistic is
      recomputed B times.  Applied to the rank metric (median percentile) and to
      the graded near-miss statistic.

  (b) **Group-level inference** across patients: per-patient statistics ->
      Wilcoxon signed-rank against chance (never pool trials), plus bootstrap CIs.

  (d) **Frequency confound**: rare/OOV words are intrinsically harder; regress
      per-trial percentile rank on the (log) word frequency and report the
      partial effect before attributing any in-vocab->held-out drop to semantics.

Directions: for percentile rank, LOWER is better (alternative="less"); for top-k,
MRR, nDCG and near-miss similarity, HIGHER is better (alternative="greater").
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, Optional, Sequence

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_THIS_DIR)
_MAIN_DIR = os.path.dirname(_TESTS_DIR)
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

from .retrieval import compute_ranks  # noqa: E402
from .metrics import near_miss_similarity  # noqa: E402


# ── (a) permutation nulls ─────────────────────────────────────────────────

def _permute_within_groups(values: np.ndarray, groups: np.ndarray,
                           rng: np.random.Generator) -> np.ndarray:
    """Return a copy of ``values`` with entries shuffled within each group.

    Used to permute the trial->true-word assignment while respecting the
    cv-fold (and patient) structure.
    """
    values = np.asarray(values)
    groups = np.asarray(groups)
    out = values.copy()
    for g in np.unique(groups):
        pos = np.where(groups == g)[0]
        out[pos] = values[pos][rng.permutation(len(pos))]
    return out


def permutation_pvalue(observed: float, null: np.ndarray,
                       alternative: str = "greater") -> float:
    """One-sided permutation p-value with the +1 correction.

    ``alternative='greater'``: p = (#{null >= observed} + 1) / (B + 1).
    ``alternative='less'``:    p = (#{null <= observed} + 1) / (B + 1).
    """
    null = np.asarray(null, dtype=np.float64)
    null = null[~np.isnan(null)]
    B = len(null)
    if B == 0 or np.isnan(observed):
        return np.nan
    if alternative == "greater":
        c = int(np.sum(null >= observed))
    elif alternative == "less":
        c = int(np.sum(null <= observed))
    else:
        raise ValueError("alternative must be 'greater' or 'less'")
    return (c + 1) / (B + 1)


def rank_permutation_null(sims: np.ndarray, true_idx: np.ndarray,
                          cv_fold: np.ndarray,
                          stat_fn: Callable[[np.ndarray], float],
                          n_perm: int = 1000, seed: int = 0) -> np.ndarray:
    """Null distribution of a rank statistic under trial->word permutation.

    ``stat_fn`` maps the per-trial rank array (competition ranks, -1 for
    out-of-gallery) to a scalar (e.g. median percentile over valid trials).
    """
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm = _permute_within_groups(true_idx, cv_fold, rng)
        null[i] = stat_fn(compute_ranks(sims, perm))
    return null


def graded_permutation_null(order: np.ndarray, true_word: np.ndarray,
                            cv_fold: np.ndarray, gallery_words: Sequence[str],
                            rel_fn: Callable[[str, str], float],
                            valid: np.ndarray, k: int = 10,
                            n_perm: int = 1000, seed: int = 0) -> np.ndarray:
    """Null for the mean near-miss similarity under trial->true-word permutation.

    Each trial keeps its retrieved ranking (``order`` row) but is graded against a
    permuted true word, so the null asks: are the true word's actual neighbours
    more related than a random word's would be?
    """
    rng = np.random.default_rng(seed)
    true_word = np.asarray(true_word)
    valid_pos = np.where(valid)[0]
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm_words = _permute_within_groups(true_word, cv_fold, rng)
        vals = [near_miss_similarity(order[t], perm_words[t], gallery_words, rel_fn, k=k)
                for t in valid_pos]
        vals = np.array(vals, dtype=np.float64)
        null[i] = np.nanmean(vals) if np.any(~np.isnan(vals)) else np.nan
    return null


# ── (b) group-level inference ─────────────────────────────────────────────

def wilcoxon_vs_chance(values: Sequence[float], chance: float,
                       alternative: str = "greater") -> Dict[str, float]:
    """Wilcoxon signed-rank test of per-patient values against a chance constant.

    ``alternative`` is in terms of (value - chance): use 'greater' for metrics
    where higher beats chance, 'less' for percentile rank.
    """
    from scipy.stats import wilcoxon
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    diffs = v - chance
    out = {"n": int(len(v)), "median": float(np.median(v)) if len(v) else np.nan,
           "chance": float(chance)}
    if len(v) < 1 or np.allclose(diffs, 0):
        out.update({"statistic": np.nan, "p_value": np.nan})
        return out
    try:
        stat, p = wilcoxon(diffs, alternative=alternative)
        out.update({"statistic": float(stat), "p_value": float(p)})
    except ValueError as exc:
        # e.g. all differences zero / too few samples — reported, not silenced.
        out.update({"statistic": np.nan, "p_value": np.nan, "note": str(exc)})
    return out


def bootstrap_ci(values: Sequence[float], n_boot: int = 5000, ci: float = 0.95,
                 seed: int = 0) -> Dict[str, float]:
    """Percentile bootstrap CI of the mean of per-patient values."""
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(rng.choice(v, len(v), replace=True))
                      for _ in range(n_boot)])
    a = (1 - ci) / 2
    return {"mean": float(np.mean(v)),
            "lo": float(np.quantile(means, a)),
            "hi": float(np.quantile(means, 1 - a)),
            "n": int(len(v))}


# ── (d) frequency confound ────────────────────────────────────────────────

def frequency_partial_effect(percentile: np.ndarray, log_freq: np.ndarray
                             ) -> Dict[str, float]:
    """OLS of per-trial percentile rank on log word-frequency.

    Returns slope, Pearson r and p (scipy ``linregress``).  A significant
    negative slope means more frequent words are easier (lower percentile);
    reviewers expect this to be checked before crediting semantics for any
    in-vocab->held-out gap.  Trials with NaN in either input are dropped and
    counted.
    """
    from scipy.stats import linregress
    percentile = np.asarray(percentile, dtype=np.float64)
    log_freq = np.asarray(log_freq, dtype=np.float64)
    m = np.isfinite(percentile) & np.isfinite(log_freq)
    n_drop = int((~m).sum())
    if m.sum() < 3 or np.allclose(log_freq[m], log_freq[m][0]):
        return {"slope": np.nan, "r": np.nan, "p_value": np.nan,
                "n": int(m.sum()), "n_dropped": n_drop}
    lr = linregress(log_freq[m], percentile[m])
    return {"slope": float(lr.slope), "r": float(lr.rvalue),
            "p_value": float(lr.pvalue), "n": int(m.sum()), "n_dropped": n_drop}
