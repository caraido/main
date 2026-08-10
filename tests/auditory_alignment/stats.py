# -*- coding: utf-8 -*-
"""Significance for the alignment comparison.

n patients caps a one-sided Wilcoxon signed-rank at an exact minimum p of 1/2**n
(config.WILCOXON_FLOOR; 0.0156 at the n=6 this module was written for, 0.00098 at the
current n=10 — so ** / *** went from mathematically unreachable to reachable, and the
report derives the sentence rather than stating it). The PRIMARY group per-bin
test is Fisher's combination of the per-patient permutation p-values (each
reaches ≈1/(n_epochs+1) from the shuffled null the fit already computed) — this goes well
below the Wilcoxon floor and states "consistent across patients AND above each patient's
own null". A distribution-free Wilcoxon vs chance is reported alongside as a secondary
check (with the floor stated in the report).

perbin_significance / the per-bin permutation p mirror
figures_for_paper/semantic_regression/semantic_regression_panels.py:perbin_significance,
made nan-safe here for the few-trial patients (AA/DR word top-k can be NaN).
"""

import warnings

import numpy as np
import pandas as pd


# ── Per-patient, per-bin permutation test (nan-safe) ──────────────────────────

def perbin_perm(obs, null, pctile=99):
    """obs/null: (n_epochs, n_bins). Returns (sig, p_perm, thr, obs_mean, null_mean),
    all length n_bins. p_perm = P(null >= observed epoch-mean), one-sided. If null is
    None (e.g. cosine, no stored null) everything null-derived is NaN and sig is False.
    All-NaN bins (few-trial patients, e.g. AA/DR word top-k) collapse to NaN quietly."""
    obs = np.asarray(obs, dtype=float)
    n_bins = obs.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN bins -> NaN, expected
        obs_mean = np.nanmean(obs, axis=0)
        if null is None:
            nan = np.full(n_bins, np.nan)
            return np.zeros(n_bins, dtype=bool), nan.copy(), nan.copy(), obs_mean, nan.copy()
        null = np.asarray(null, dtype=float)
        null_mean = np.nanmean(null, axis=0)
        thr = np.nanpercentile(null, pctile, axis=0)
    p_perm = np.empty(n_bins)
    for b in range(n_bins):
        col = null[:, b]
        col = col[~np.isnan(col)]
        if col.size == 0 or not np.isfinite(obs_mean[b]):
            p_perm[b] = np.nan
        else:
            p_perm[b] = (np.sum(col >= obs_mean[b]) + 1) / (col.size + 1)
    sig = (obs_mean > thr) & np.isfinite(obs_mean) & np.isfinite(thr)
    return sig, p_perm, thr, obs_mean, null_mean


# ── p-value combination / correction ──────────────────────────────────────────

def fisher_combine(pvals):
    """Fisher's method: chi2 = -2 Sum ln(p), df = 2k. NaN/None dropped. NaN if none valid."""
    from scipy.stats import chi2
    p = np.asarray([x for x in pvals if x is not None and np.isfinite(x)], dtype=float)
    if p.size == 0:
        return np.nan
    p = np.clip(p, 1e-12, 1.0)
    stat = -2.0 * np.sum(np.log(p))
    return float(chi2.sf(stat, df=2 * p.size))


def benjamini_hochberg(pvals, alpha=0.05):
    """BH-FDR over a 1-D array (NaN entries ignored). Returns (rejected_mask, qvalues)."""
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    rej = np.zeros(p.shape, dtype=bool)
    idx = np.where(np.isfinite(p))[0]
    if idx.size == 0:
        return rej, q
    pv = p[idx]
    order = np.argsort(pv)
    m = pv.size
    qv = pv[order] * m / np.arange(1, m + 1)
    qv = np.minimum.accumulate(qv[::-1])[::-1]   # step-up monotonicity
    qv = np.clip(qv, 0, 1)
    q[idx[order]] = qv
    rej[idx] = q[idx] <= alpha
    return rej, q


# ── Wilcoxon helpers (adapted from figures_for_paper/extendability) ────────────

def wilcoxon_vs_chance(values, chance, alternative="greater"):
    """One-sided Wilcoxon signed-rank of `values` vs a scalar (or per-element) chance."""
    from scipy.stats import wilcoxon
    v = np.asarray(values, dtype=float)
    c = np.asarray(chance, dtype=float)
    diffs = v - c
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size < 1 or np.allclose(diffs, 0):
        return np.nan, int(diffs.size)
    try:
        _, p = wilcoxon(diffs, alternative=alternative)
    except ValueError:
        return np.nan, int(diffs.size)
    return float(p), int(diffs.size)


def wilcoxon_paired(a, b, alternative="two-sided"):
    """One-sided/two-sided paired Wilcoxon between two per-participant vectors."""
    from scipy.stats import wilcoxon
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if a.size < 1 or np.allclose(a - b, 0):
        return np.nan, int(a.size)
    try:
        _, p = wilcoxon(a, b, alternative=alternative)
    except ValueError:
        return np.nan, int(a.size)
    return float(p), int(a.size)


def stars(p):
    """p -> significance string (house convention)."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# ── Group per-bin significance for one (cue, metric) ──────────────────────────

def group_perbin(records, cue_key, metric, patients, alpha=0.05):
    """Combine per-patient per-bin results on the integer bin-offset grid.

    Only bins covered by ALL `patients` are tested (ragged windows differ per patient).
    Returns a DataFrame [k, t_s, n, group_obs, group_null, p_fisher, p_wilcoxon, sig_fdr].
    p_fisher/p_wilcoxon/sig_fdr are NaN/False for metrics without a stored null (cosine).
    """
    # k -> lists across patients
    from collections import defaultdict
    p_by_k = defaultdict(list)      # per-patient permutation p
    obs_by_k = defaultdict(list)
    null_by_k = defaultdict(list)
    t_of_k = {}
    for p in patients:
        rec = records.get((cue_key, p))
        if rec is None or metric not in rec["metrics"]:
            continue
        md = rec["metrics"][metric]
        for k, t, om, nm, pp in zip(md["k"], md["t_s"], md["obs_mean"],
                                    md["null_mean"], md["p_perm"]):
            p_by_k[int(k)].append(pp)
            obs_by_k[int(k)].append(om)
            null_by_k[int(k)].append(nm)
            t_of_k[int(k)] = float(t)

    n_want = len(patients)
    rows = []
    for k in sorted(p_by_k):
        obs = np.asarray(obs_by_k[k], dtype=float)
        nul = np.asarray(null_by_k[k], dtype=float)
        pp = p_by_k[k]
        n = int(np.sum(~np.isnan(obs)))
        if len(obs) < n_want:            # not all patients cover this bin
            continue
        has_null = np.any(np.isfinite(nul))
        p_fisher = fisher_combine(pp) if has_null else np.nan
        if has_null:
            p_wil, _ = wilcoxon_vs_chance(obs, nul, alternative="greater")
        else:
            p_wil = np.nan
        rows.append(dict(k=k, t_s=t_of_k[k], n=n,
                         group_obs=float(np.nanmean(obs)),
                         group_null=(float(np.nanmean(nul)) if has_null else np.nan),
                         p_fisher=p_fisher, p_wilcoxon=p_wil))
    df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    if len(df):
        rej, q = benjamini_hochberg(df["p_fisher"].values, alpha=alpha)
        df["sig_fdr"] = rej
        df["q_fisher"] = q
    else:
        df["sig_fdr"] = pd.Series(dtype=bool)
        df["q_fisher"] = pd.Series(dtype=float)
    return df


def paired_wilcoxon_peaks(peak_df, metric, cue_a, cue_b, alternative="two-sided"):
    """Paired Wilcoxon between two alignments on per-patient peak values. Returns (p, n)."""
    a = (peak_df[(peak_df["metric"] == metric) & (peak_df["cue_key"] == cue_a)]
         .set_index("patient")["peak_val"])
    b = (peak_df[(peak_df["metric"] == metric) & (peak_df["cue_key"] == cue_b)]
         .set_index("patient")["peak_val"])
    common = a.index.intersection(b.index)
    if len(common) < 1:
        return np.nan, 0
    return wilcoxon_paired(a.loc[common].values, b.loc[common].values, alternative)
