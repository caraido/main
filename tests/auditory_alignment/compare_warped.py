# -*- coding: utf-8 -*-
"""The warped auditory run as a fifth arm alongside the pilot's four cue alignments.

The pilot re-references raw data to each of four candidate cues with ``--warp none``. The
shipped auditory decoder instead **time-warps** each trial so the spoken prompt occupies a
common duration, then aligns to ``aud_stim_onset``. That run is pinned as
``utils.config.AUD_RUN``. This module reads its per-bin curves and emits them in the pilot's
own ``peak_table`` schema with ``cue_key='warped'``, so the two treatments of the time axis
sit in one table.

Why two sources, not one
------------------------
``category_indep`` and ``word_top1/3/5`` come from
``figures_for_paper/semantic_regression/source_data/source_data.csv`` — the same file behind
the shipped figure, so the comparison cannot drift from what the paper reports. **Cosine is
not in that file at all**, so it comes from ``AUD_RUN/<patient>/per_time_scores.csv``
(``cosine_mean``). That column is ``br.all_cosine_sim.mean(0)`` (semantic_regression.py:1795)
— the identical attribute and the identical epoch-mean the pilot extracts for its own cosine
metric, so the two sides are the same quantity despite the different file.

``per_time_scores.csv`` is NOT a substitute for the other four: it carries no ``word_top1``
column (``word_balanced_acc`` is a different quantity, not top-1), and it has no time axis —
only ``bin_index``. The time axis is joined in from ``source_data.csv`` on
``(patient, bin_index)``.

Two peak estimators, and why both are here
------------------------------------------
``warped_peak_table`` mirrors ``aggregate.peak_table`` exactly: per patient, argmax of the
per-bin epoch-mean over the FULL window. That is the only estimator that makes the
comparison valid, because it is the operation the pilot applies to its own four arms.

``warped_group_anchor`` reproduces the *different* estimator behind
``figures_for_paper/semantic_regression/source_data/peak_rise_stats.csv``: the cohort mean at
a single common bin t*, where t* is the argmax of the across-participant mean curve
restricted to t>=0 and to bins every participant covers. It is here purely as a provenance
check — it reproduces that file to six decimals, which proves this module is reading the
run the paper reports. **The two estimators do not agree and are not meant to**: a
per-patient argmax capitalises on each patient's own noise, so it sits above the common-bin
estimate. Do not quote a number from one and compare it to the other.
"""

import os
import sys

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAIN_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

# AUD_RUN_FIGURE, not AUD_RUN, and this is load-bearing rather than a preference. The four
# retrieval metrics below come from figures_for_paper/semantic_regression's source_data.csv,
# and that figure's auditory arm was repointed to the 10-bin run on 2026-08-11. Cosine comes
# from the run directory directly. If the two disagreed, the "warped" arm assembled here
# would be a chimera -- cosine from one run, retrieval from another -- with nothing in the
# output saying so. Both must track the same constant.
from utils.config import AUD_RUN_FIGURE as AUD_RUN           # noqa: E402
from utils.paths import results_dir                       # noqa: E402
from tests.auditory_alignment import config               # noqa: E402
from tests.auditory_alignment.aggregate import _fwhm      # noqa: E402

#: The cue_key this arm occupies in the pilot's tables.
WARPED_KEY = "warped"
WARPED_LABEL = "Warped (AUD_RUN)"

#: Metrics available from source_data.csv, keyed exactly as config.METRICS.
_PANEL_METRICS = ("category_indep", "word_top1", "word_top3", "word_top5")

_PANEL_CSV = os.path.join(_MAIN_DIR, "figures_for_paper", "semantic_regression",
                          "source_data", "source_data.csv")


def _run_dir():
    return str(results_dir("semantic_regression", AUD_RUN, create=False))


def load_panel_curves():
    """Per-bin auditory curves for the four retrieval metrics, from the shipped figure's
    source data. Returns a tidy frame: patient, metric, bin_index, time_s, obs_mean,
    chance_mean."""
    if not os.path.isfile(_PANEL_CSV):
        raise FileNotFoundError(
            f"{_PANEL_CSV} not found. It is written by "
            "figures_for_paper/semantic_regression/semantic_regression_panels.py; "
            "regenerate that figure before running the warped comparison.")
    d = pd.read_csv(_PANEL_CSV)
    d = d[d["task"] == "auditory"].copy()
    if d.empty:
        raise ValueError(f"{_PANEL_CSV} contains no task=='auditory' rows.")
    return d[["patient", "metric", "bin_index", "time_s", "obs_mean", "chance_mean"]]


def load_cosine_curves(patients, embedding="GloVe"):
    """Per-bin cosine for each patient, from AUD_RUN/<pat>/per_time_scores.csv.

    The file interleaves embeddings (GloVe and Word2Vec both present in the pinned run), so
    filtering to one is mandatory — without it every bin appears twice and the max is taken
    across models."""
    root = _run_dir()
    rows = []
    missing = []
    for p in patients:
        f = os.path.join(root, p, "per_time_scores.csv")
        if not os.path.isfile(f):
            missing.append(p)
            continue
        d = pd.read_csv(f)
        if "embedding" in d.columns:
            d = d[d["embedding"] == embedding]
        if d.empty:
            missing.append(p)
            continue
        rows.append(pd.DataFrame(dict(
            patient=p, metric="cosine", bin_index=d["bin_index"].values,
            obs_mean=d["cosine_mean"].values,
            chance_mean=np.nan,          # cosine has no stored per-bin null (same as pilot)
        )))
    if missing:
        print(f"  [warn] cosine unavailable for: {', '.join(missing)} "
              f"(no per_time_scores.csv with embedding=={embedding!r} under {root})",
              flush=True)
    if not rows:
        return pd.DataFrame(columns=["patient", "metric", "bin_index", "time_s",
                                     "obs_mean", "chance_mean"])
    return pd.concat(rows, ignore_index=True)


def _attach_time_axis(cos_df, panel_df):
    """Give the cosine rows the same time axis as the retrieval metrics, joined on
    (patient, bin_index). per_time_scores.csv stores no time column, and the axis is
    per patient (participants have different trial lengths), so it cannot be assumed."""
    if cos_df.empty:
        return cos_df
    axis = (panel_df[["patient", "bin_index", "time_s"]]
            .drop_duplicates(subset=["patient", "bin_index"]))
    out = cos_df.drop(columns=[c for c in ("time_s",) if c in cos_df.columns]) \
                .merge(axis, on=["patient", "bin_index"], how="left")
    n_unmatched = int(out["time_s"].isna().sum())
    if n_unmatched:
        print(f"  [warn] {n_unmatched} cosine bins had no matching time in the panel source "
              "data (bin grids disagree); those bins are dropped from the cosine arm.",
              flush=True)
        out = out[out["time_s"].notna()]
    return out


def warped_peak_table(patients=None, bin_size_ms=None):
    """Per (metric, patient) peak of the warped run, in `aggregate.peak_table` schema.

    Mirrors that function's definition exactly — argmax of the per-bin epoch-mean over the
    full window, baseline read at the peak bin, FWHM via the same `_fwhm` helper — so the
    warped arm and the four cue arms are the same operation on the same quantity."""
    patients = list(patients or config.AUD_PATIENTS)
    bin_size_ms = int(bin_size_ms or config.DEFAULTS["bin_size"])
    bin_s = bin_size_ms / 1000.0

    panel = load_panel_curves()
    cos = _attach_time_axis(load_cosine_curves(patients), panel)
    allc = pd.concat([panel, cos], ignore_index=True, sort=False)

    rows = []
    for metric in config.METRIC_KEYS:
        sub_m = allc[allc["metric"] == metric]
        if sub_m.empty:
            continue          # r2 is in neither source; absent, not zero
        for p in patients:
            s = sub_m[sub_m["patient"] == p].sort_values("bin_index")
            if s.empty:
                continue
            y = s["obs_mean"].to_numpy(dtype=float)
            t = s["time_s"].to_numpy(dtype=float)
            k = s["bin_index"].to_numpy(dtype=int)
            nm = s["chance_mean"].to_numpy(dtype=float)
            if not np.any(np.isfinite(y)):
                continue
            idx = int(np.nanargmax(y))
            baseline = float(nm[idx]) if np.any(np.isfinite(nm)) else 0.0
            rows.append(dict(
                cue_key=WARPED_KEY, metric=metric, patient=p,
                peak_val=float(y[idx]), peak_k=int(k[idx]), peak_t_s=float(t[idx]),
                baseline=baseline,
                fwhm_s=_fwhm(t, y, idx, baseline, bin_s),
                n_trials=np.nan,     # not recorded per patient in the run's source data
            ))
    return pd.DataFrame(rows)


def warped_group_anchor(patients=None):
    """Reproduce peak_rise_stats.csv's estimator — provenance check only, see module docstring.

    t* = argmax of the across-participant mean curve over t>=0, restricted to bins every
    participant covers; the reported value is the cohort mean at that single bin."""
    patients = list(patients or config.AUD_PATIENTS)
    panel = load_panel_curves()
    panel = panel[panel["patient"].isin(patients)]
    n_pat = panel["patient"].nunique()
    rows = []
    for metric in _PANEL_METRICS:
        s = panel[panel["metric"] == metric]
        if s.empty:
            continue
        g = s.groupby("time_s").agg(mean=("obs_mean", "mean"),
                                    n=("patient", "nunique")).reset_index()
        ok = g[(g["time_s"] >= 0) & (g["n"] == n_pat)]
        if ok.empty:
            continue
        t_star = float(ok.loc[ok["mean"].idxmax(), "time_s"])
        at = s[np.isclose(s["time_s"], t_star)]
        rows.append(dict(
            metric=metric, t_star_s=t_star, n_patients=int(len(at)),
            peak_acc_mean=float(at["obs_mean"].mean()),
            peak_acc_sem=(float(at["obs_mean"].std(ddof=1) / np.sqrt(len(at)))
                          if len(at) > 1 else np.nan),
            emp_chance=float(at["chance_mean"].mean()),
        ))
    return pd.DataFrame(rows)


def warped_provenance():
    """Facts about AUD_RUN that the comparison's caveats depend on. Read from the run's own
    meta.json — never restated from prose."""
    import json
    f = os.path.join(_run_dir(), "meta.json")
    if not os.path.isfile(f):
        return {}
    m = json.load(open(f, "r", encoding="utf-8"))
    return {k: m.get(k) for k in (
        "run_id", "task", "auditory_warp", "auditory_warp_scope",
        "auditory_warp_target_sec", "auditory_warp_target_source", "align_cue",
        "roi_atlas", "n_bins_history", "bin_size_ms", "n_epochs", "patients")}


def write_source_csv(out_dir, patients=None):
    """Write warped_comparison.csv (+ the anchor check) under the pilot's source_data dir."""
    os.makedirs(out_dir, exist_ok=True)
    pk = warped_peak_table(patients)
    pk.to_csv(os.path.join(out_dir, "warped_comparison.csv"), index=False)
    an = warped_group_anchor(patients)
    an.to_csv(os.path.join(out_dir, "warped_group_anchor.csv"), index=False)
    return pk, an


if __name__ == "__main__":
    prov = warped_provenance()
    print("AUD_RUN provenance:")
    for k, v in prov.items():
        print(f"  {k:28s} {v}")
    pk = warped_peak_table()
    print(f"\nwarped_peak_table: {len(pk)} rows, "
          f"{pk['patient'].nunique()} patients, metrics={sorted(pk['metric'].unique())}")
    print(pk.groupby("metric")["peak_val"].agg(["count", "mean", "min", "max"]).to_string())
    print("\nGroup anchor (must match peak_rise_stats.csv):")
    print(warped_group_anchor().to_string(index=False))
