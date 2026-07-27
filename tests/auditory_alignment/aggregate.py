# -*- coding: utf-8 -*-
"""Load all (cue, patient) cells and build the comparison-ready aggregates.

Cross-patient aggregation joins on the INTEGER bin-offset-from-cue `k` (exact common
grid; every patient uses 100 ms bins with the cue on a boundary), not on a bin index or a
rounded float time — so ragged per-patient windows line up exactly. Group curves are
reported only where all requested patients cover a bin.
"""

import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

from tests.auditory_alignment import config
from tests.auditory_alignment import metrics as M
from tests.auditory_alignment import stats as S


def _cell_dir(cue_key, patient):
    from utils.paths import results_dir
    return str(results_dir(config.ANALYSIS, cue_key, patient, create=False))


def load_all(cues, patients, pctile=99):
    """Return records[(cue_key, patient)] = {'meta':..., 'metrics': {key: {...}}}.

    Each metric dict holds per-bin arrays aligned to the cue: k, t_s, obs_mean, obs_sem,
    null_mean, p_perm, sig. Missing cells are skipped (and reported)."""
    records = {}
    missing = []
    for cue_key in cues:
        for patient in patients:
            d = _cell_dir(cue_key, patient)
            if not M.is_done(d):
                missing.append((cue_key, patient))
                continue
            arrays, meta = M.load_perbin(d)
            k = np.asarray(meta["k"], dtype=int)
            t_s = np.asarray(meta["t_center_s"], dtype=float)
            mdict = {}
            for key, _label, _obs_attr, _null_attr, _fam in config.METRICS:
                if key not in arrays:
                    continue
                obs = arrays[key]["obs"]
                null = arrays[key].get("null")
                sig, p_perm, thr, obs_mean, null_mean = S.perbin_perm(obs, null, pctile=pctile)
                n_valid = np.sum(~np.isnan(obs), axis=0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN / ddof bins
                    obs_sem = np.nanstd(obs, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1))
                mdict[key] = dict(
                    k=k, t_s=t_s,
                    obs_mean=obs_mean, obs_sem=obs_sem,
                    null_mean=null_mean, p_perm=p_perm, sig=sig,
                )
            records[(cue_key, patient)] = {"meta": meta, "metrics": mdict}
    if missing:
        print(f"[aggregate] {len(missing)} cell(s) not computed yet: "
              f"{', '.join(f'{c}/{p}' for c, p in missing[:12])}"
              f"{' ...' if len(missing) > 12 else ''}")
    return records


def present_patients(records, cue_key, metric, patients):
    """Patients that have a computed cell for this (cue, metric)."""
    return [p for p in patients if (cue_key, p) in records
            and metric in records[(cue_key, p)]["metrics"]]


def group_timecourse(records, cue_key, metric, patients):
    """Across-patient mean±sem of obs_mean on the integer-k grid, kept only where ALL
    `patients` cover the bin. Returns DataFrame[k, t_s, mean, sem, null, n]."""
    obs_by_k = defaultdict(list)
    null_by_k = defaultdict(list)
    t_of_k = {}
    for p in patients:
        rec = records.get((cue_key, p))
        if rec is None or metric not in rec["metrics"]:
            continue
        md = rec["metrics"][metric]
        for k, t, om, nm in zip(md["k"], md["t_s"], md["obs_mean"], md["null_mean"]):
            obs_by_k[int(k)].append(om)
            null_by_k[int(k)].append(nm)
            t_of_k[int(k)] = float(t)
    n_want = len(patients)
    rows = []
    for k in sorted(obs_by_k):
        vals = np.asarray(obs_by_k[k], dtype=float)
        nul = np.asarray(null_by_k[k], dtype=float)
        if len(vals) < n_want:
            continue
        good = vals[~np.isnan(vals)]
        if good.size == 0:
            continue
        mean = float(np.mean(good))
        sem = float(np.std(good, ddof=1) / np.sqrt(good.size)) if good.size > 1 else np.nan
        rows.append(dict(k=k, t_s=t_of_k[k], mean=mean, sem=sem,
                         null=float(np.nanmean(nul)) if np.any(np.isfinite(nul)) else np.nan,
                         n=int(good.size)))
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def _fwhm(t_s, y, peak_idx, baseline, bin_s):
    """Full width (s) of the contiguous supra-half-max run containing the peak.
    half = baseline + 0.5*(peak - baseline). NaN if ill-defined."""
    peak = y[peak_idx]
    if not np.isfinite(peak) or not np.isfinite(baseline) or peak <= baseline:
        return np.nan
    half = baseline + 0.5 * (peak - baseline)
    lo = peak_idx
    while lo - 1 >= 0 and np.isfinite(y[lo - 1]) and y[lo - 1] >= half:
        lo -= 1
    hi = peak_idx
    while hi + 1 < len(y) and np.isfinite(y[hi + 1]) and y[hi + 1] >= half:
        hi += 1
    return float((hi - lo + 1) * bin_s)


def peak_table(records, cues, patients, metrics=None):
    """Per (cue_key, metric, patient): peak height / latency / width over the FULL window.

    Peak searched over ALL bins (not just post-cue): when a decoder is aligned to the true
    trigger the informative bin lands at a consistent small offset; aligning to a jittery
    non-trigger cue smears it (lower per-patient peak, larger cross-patient latency spread)
    — both signals live in this table. Returns a tidy DataFrame."""
    if metrics is None:
        metrics = config.METRIC_KEYS
    rows = []
    for cue_key in cues:
        for p in patients:
            rec = records.get((cue_key, p))
            if rec is None:
                continue
            bin_s = rec["meta"]["bin_size_ms"] / 1000.0
            for metric in metrics:
                if metric not in rec["metrics"]:
                    continue
                md = rec["metrics"][metric]
                y = np.asarray(md["obs_mean"], dtype=float)
                t = np.asarray(md["t_s"], dtype=float)
                if not np.any(np.isfinite(y)):
                    continue
                idx = int(np.nanargmax(y))
                nm = md["null_mean"]
                baseline = float(nm[idx]) if np.any(np.isfinite(nm)) else 0.0
                rows.append(dict(
                    cue_key=cue_key, metric=metric, patient=p,
                    peak_val=float(y[idx]), peak_k=int(md["k"][idx]),
                    peak_t_s=float(t[idx]),
                    baseline=baseline,
                    fwhm_s=_fwhm(t, y, idx, baseline, bin_s),
                    n_trials=rec["meta"].get("n_trials"),
                ))
    return pd.DataFrame(rows)


def peak_summary(peak_df, patients):
    """Group summary per (cue_key, metric): peak height mean±sem across patients, and the
    cross-patient SD of peak latency (the temporal-locking signal). Returns DataFrame."""
    rows = []
    for (cue_key, metric), g in peak_df.groupby(["cue_key", "metric"], sort=False):
        vals = g.set_index("patient")["peak_val"].reindex(patients).values.astype(float)
        lat = g.set_index("patient")["peak_t_s"].reindex(patients).values.astype(float)
        fw = g.set_index("patient")["fwhm_s"].reindex(patients).values.astype(float)
        v = vals[~np.isnan(vals)]
        l = lat[~np.isnan(lat)]
        rows.append(dict(
            cue_key=cue_key, metric=metric, n=int(v.size),
            peak_mean=float(np.mean(v)) if v.size else np.nan,
            peak_sem=float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan,
            latency_mean_s=float(np.mean(l)) if l.size else np.nan,
            latency_sd_s=float(np.std(l, ddof=1)) if l.size > 1 else np.nan,
            fwhm_mean_s=float(np.nanmean(fw)) if np.any(np.isfinite(fw)) else np.nan,
        ))
    return pd.DataFrame(rows)


def argmax_vote(peak_df, patients, metrics=None):
    """Per (metric, patient): which alignment gives the highest peak (robust to across-
    patient scale). Returns long DataFrame[metric, patient, winning_cue]."""
    if metrics is None:
        metrics = config.METRIC_KEYS
    rows = []
    for metric in metrics:
        for p in patients:
            g = peak_df[(peak_df["metric"] == metric) & (peak_df["patient"] == p)]
            if len(g) == 0 or not np.any(np.isfinite(g["peak_val"].values)):
                continue
            win = g.loc[g["peak_val"].idxmax(), "cue_key"]
            rows.append(dict(metric=metric, patient=p, winning_cue=win))
    return pd.DataFrame(rows)


def vote_tally(vote_df, cues, metrics=None):
    """Counts of winning cue per metric: DataFrame indexed by metric, columns = cue keys."""
    if metrics is None:
        metrics = config.METRIC_KEYS
    tally = pd.DataFrame(0, index=metrics, columns=list(cues))
    for _, r in vote_df.iterrows():
        if r["metric"] in tally.index and r["winning_cue"] in tally.columns:
            tally.loc[r["metric"], r["winning_cue"]] += 1
    return tally


def cue_bands(records, cue_key, patients):
    """Cross-patient mean±std position (s, relative to the aligned cue) of every OTHER
    cue, from each cell's meta['rel_cues']. Mirrors semantic_regression_panels._aggregate_cues.
    Returns {sr_cue_name: (mean_s, std_s)} excluding the aligned cue itself."""
    align_cue = config.CUES[cue_key]
    per_cue = defaultdict(list)
    for p in patients:
        rec = records.get((cue_key, p))
        if rec is None:
            continue
        rel = rec["meta"].get("rel_cues") or {}
        for name, v in rel.items():
            if name == align_cue:
                continue
            m = v.get("mean") if isinstance(v, dict) else None
            if m is not None and np.isfinite(m):
                per_cue[name].append(float(m))
    out = {}
    for name, vals in per_cue.items():
        vals = np.asarray(vals, dtype=float)
        if vals.size >= 1:
            out[name] = (float(np.mean(vals)),
                         float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0)
    return out
