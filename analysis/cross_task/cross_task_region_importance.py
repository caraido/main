"""Brain-region (ROI) importance for the co-trained (pooled pic+aud) semantic model.

Answers, at BRAIN-REGION granularity (``primary_roi`` from {PAT}_*channels.pkl):
which regions drive *both* tasks' retrieval accuracy, which are picture-only, and
which are auditory-only — for the SAME pooled kernel-PLS model used in
``cross_task_cotrain.py``.  Single-channel attribution is deliberately NOT
reported: under the Nystroem-RBF dilution information is spread redundantly across
electrodes, so dropping any one channel barely moves accuracy (almost every
channel lands in the ``neither`` significance bucket).  The population-level
region view is the one that reads cleanly, so it is the only view produced here.

All three attributions are computed per region as a TOTAL (summed over the
region's channels), evaluated on the picture test set and the auditory test set
separately:

  1. Permutation importance (Δmetric when the region's whole history block is
     shuffled jointly across trials — the population-level accuracy drop when an
     entire region is removed).  Significance: a per-bootstrap label-shuffle null
     gives the noise floor of Δacc; one-sided p-values are pooled across
     bootstraps and BH-FDR corrected.  A whole-brain (all channels) block is also
     knocked out as the "ceiling" — the total accuracy the model attributes to
     the neural data — against which each region's Δacc / share is read.

  2. Analytic Jacobian sensitivity (mean ‖∂ŷ/∂x‖ back-propagated through the
     Nystroem-RBF + PLS affine map), summed over the region's channels.  Scores
     sensitivity of the predicted GloVe embedding rather than accuracy, so it is
     a cross-check.

  3. Neural-GloVe cross-covariance (``--analysis covariance`` / ``both``): per
     feature ‖zscore(X)^T (Y-Ybar)/(n-1)‖, summed over the region's columns, for
     each task separately.  The rawest form of the PLS objective (PLS maximises
     cov(X, Y)) and the only **model-free** measure here — no fit, no split, no
     resampling — so it cannot be an artifact of the Nystroem approximation.
     Metric-independent; merged into the region table on (patient, region).
     ``cov_nc_*`` subtracts the finite-sample floor; prefer it cross-participant.

     (Plain-PLS VIP was measure 3 until 2026-07-23 and is gone — see the note
     above ``_build_channel_map``.  Covariance used to be computed inside the VIP
     function and is now standalone in ``analyze_patient_region_cov``.)

Every ROI atlas is present now (all 6 cross-task patients AA/AZ/LH/WBH/DR/RB have
a {PAT}_*channels.pkl region file), so the region path runs for all of them.

READ REGION SCORES PER ELECTRODE, NOT AS TOTALS (external audit, 2026-07-23).
Region totals for the magnitude measures are electrode-count proxies: within
patient, ρ(total, n_channels) = 0.99 (Jacobian), 0.98 (VIP), 0.96 (covariance) —
against 0.19 for the two knockouts, which are the only size-robust measures. The
apparent picture↔auditory agreement of VIP (ρ=0.95) and covariance (ρ=0.96) is
entirely that size artifact: dividing by n_channels collapses it to −0.13 and
−0.09. The Jacobian is different — it stays at +0.99 per electrode because the
co-trained model scores both tasks through ONE shared map, so its picture-vs-
auditory diagonal is structural and is NOT evidence of amodality. Use the
_solo columns (--single-modality: two independently trained decoders) for any
task-specificity claim, and normalize before pooling ROIs across participants.

The retrieval-aligned Jacobian (jac_dir) was REMOVED for the same audit: it was a
constant rescaling of jac_sens (CV 0.8–6.7 % within patient/task, ρ=0.99). Only
its scalar diagnostics survive — see ``jacobian_measures``.

Grouping (permutation-null significance):
    both        : sig. positive Δacc in BOTH tasks
    picture_only: sig. positive Δacc in pic only
    auditory_only: sig. positive Δacc in aud only
    neither     : sig. in neither

Memory note: this loads the per-patient semantic_regression_results.pkl
(100 MB – 2.6 GB each) via cross_task_cotrain.load_patient, so run it on a
machine with enough RAM (the project README recommends 16 GB+).  Run e.g.:

    python -m analysis.cross_task.cross_task_region_importance --analysis both
    python -m analysis.cross_task.cross_task_region_importance --patient RB
    python -m analysis.cross_task.cross_task_region_importance \
        --n-bootstrap 20 --n-perm-repeats 5 --region-null-shuffles 20
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path setup (mirror cross_task_cotrain so `tests`/`utils` resolve when run
#    either as a module or as a script) ──────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_THIS_DIR)
_MAIN_DIR = os.path.dirname(_TESTS_DIR)
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

# Reuse the co-training pipeline so the model + data are identical.
from analysis.cross_task.cross_task_cotrain import (
    load_patient, make_model, _build_db, _score, _norm,
    _stratified_word_split, _balance_pooled,
    PIC_RUN_DEFAULT, AUD_RUN_DEFAULT, SHARED_PATIENTS, OUT_ROOT,
)
from utils.retrieval import compute_retrieval_metrics
from utils.config import ALPHA


# ── helpers ────────────────────────────────────────────────────────────────
METRICS = ("cat_indep_bal_acc", "word_bal_acc")   # default first (more robust)
_METRIC_TAG = {"cat_indep_bal_acc": "catindep", "word_bal_acc": "word"}


def _metric_value(Y_pred: np.ndarray, words: np.ndarray, cats: np.ndarray,
                  db: tuple, metric: str) -> float:
    m = compute_retrieval_metrics(Y_pred, words, cats, *db)
    return float(m[metric])


def _channel_columns(c: int, n_ch: int, n_hist: int) -> np.ndarray:
    """Column indices of channel *c* across all history bins (X layout is
    ``channels + b*n_ch`` — see _word_means in cross_task_cotrain)."""
    return c + n_ch * np.arange(n_hist)


def _region_columns(chan_idx: np.ndarray, n_ch: int, n_hist: int) -> np.ndarray:
    """Column indices for a *group* of channels (a region) across history bins."""
    return np.concatenate([_channel_columns(int(c), n_ch, n_hist) for c in chan_idx])


def _grouped_permutation_importance(model, X_te, words_te, cats_te, db,
                                    group_cols, n_repeats: int,
                                    rng: np.random.Generator, metric: str) -> np.ndarray:
    """Δ<metric> (baseline − permuted) when each *group* of feature columns is
    jointly shuffled across trials (one row-perm for the whole block), averaged
    over repeats.  Groups can be single channels or whole brain regions; the
    joint shuffle removes the group while preserving its within-group structure.
    """
    base = _metric_value(model.predict(X_te), words_te, cats_te, db, metric)
    n_te = X_te.shape[0]
    drops = np.zeros(len(group_cols))
    for gi, cols in enumerate(group_cols):
        acc = 0.0
        for _ in range(n_repeats):
            Xp = X_te.copy()
            perm = rng.permutation(n_te)
            Xp[:, cols] = Xp[perm][:, cols]      # same row-perm across the block
            acc += _metric_value(model.predict(Xp), words_te, cats_te, db, metric)
        drops[gi] = base - acc / n_repeats
    return drops


def _grouped_permutation_importance_multi(model, X_te, words_te, cats_te, db,
                                          group_cols, n_repeats: int,
                                          rng: np.random.Generator,
                                          metrics) -> dict:
    """Like _grouped_permutation_importance but evaluates SEVERAL metrics from the
    SAME shuffled predictions — the marginal cost of an extra metric is just its
    (cheap) retrieval eval, since model.predict dominates.  Returns
    {metric: drops_array} with one Δ<metric> per group."""
    def _all(Y_pred):
        m = compute_retrieval_metrics(Y_pred, words_te, cats_te, *db)
        return {k: float(m[k]) for k in metrics}
    base = _all(model.predict(X_te))
    n_te = X_te.shape[0]
    drops = {k: np.zeros(len(group_cols)) for k in metrics}
    for gi, cols in enumerate(group_cols):
        acc = {k: 0.0 for k in metrics}
        for _ in range(n_repeats):
            Xp = X_te.copy()
            perm = rng.permutation(n_te)
            Xp[:, cols] = Xp[perm][:, cols]
            mv = _all(model.predict(Xp))
            for k in metrics:
                acc[k] += mv[k]
        for k in metrics:
            drops[k][gi] = base[k] - acc[k] / n_repeats
    return drops


def _grouped_null_importance(model, X_te, words_te, cats_te, db,
                             group_cols, n_shuffles: int,
                             rng: np.random.Generator, metric: str) -> np.ndarray:
    """Pooled null of Δacc across *groups*: under shuffled trial labels every
    group is irrelevant, so its Δacc reflects only sampling noise.  Returns a
    flat array of null Δacc values (n_shuffles × n_groups), pooled across groups
    of the same kind (channels with channels, regions with regions)."""
    nulls: List[float] = []
    n_te = X_te.shape[0]
    base_pred = model.predict(X_te)
    for _ in range(n_shuffles):
        sh = rng.permutation(n_te)            # break Y_pred <-> label alignment
        w_sh, c_sh = words_te[sh], cats_te[sh]
        base = _metric_value(base_pred, w_sh, c_sh, db, metric)
        for cols in group_cols:
            Xp = X_te.copy()
            Xp[:, cols] = Xp[rng.permutation(n_te)][:, cols]
            nulls.append(base - _metric_value(model.predict(Xp), w_sh, c_sh, db, metric))
    return np.asarray(nulls)


def _pls_affine(model, n_targets: int) -> Tuple[np.ndarray, np.ndarray]:
    """Recover the PLS affine map  ŷ = φ @ A + b  empirically (orientation-
    agnostic w.r.t. sklearn's coef_ convention)."""
    pls = model.named_steps["pls"]
    n_feat = model.named_steps["nys"].n_components
    b = pls.predict(np.zeros((1, n_feat)))[0]            # (n_targets,)
    A = pls.predict(np.eye(n_feat))                       # (n_feat, n_targets)
    A = A - b[None, :]
    return A, b


def jacobian_measures(model, X_te, y_te, ybar, n_ch: int, n_hist: int):
    """Jacobian sensitivity of the co-trained model, plus the two rank-collapse
    diagnostics that retired the retrieval-aligned variant. ONE per-trial pass
    (the Jacobian J is built once per trial).

    Returns ``(sens_per_channel, align, pr_A)``:

      sens  — mean ‖∂ŷ/∂x‖₂ per feature (isotropic output sensitivity), returned
              aggregated to per-channel (summed over history bins). THE region
              measure.

      align — mean over trials of Σ_j|J_j·û| / Σ_j‖J_j‖, where û is the
              mean-centred unit true-GloVe direction (û = (y-ybar)/‖y-ybar‖).
              ONE SCALAR per (patient, task), not a region column.

      pr_A  — participation ratio (Σσ²)²/Σσ⁴ of A's singular spectrum: the
              effective number of output dimensions the PLS map actually uses.

    Why the retrieval-aligned Jacobian |∂(ŷ·û)/∂x| is NO LONGER reported per
    region (retired 2026-07-23 after external audit).

    Empirically: the per-region ratio jac_dir/jac_sens was constant to CV 0.8–6.7 %
    within every patient and task (ρ = 0.99 as region totals, 0.95 per channel), so
    jac_dir was a constant rescaling of jac_sens carrying no independent regional
    information. That is the finding; the rest is why it is structural rather than a
    property of this dataset.

    Structurally: every gradient row factors through the SAME map, J_j = Aᵀv_j with
    v_j = (N·∂k/∂x)_j, and A has rank ≤ n_pls = 10. So the only thing that can vary
    across features is the direction of v_j inside a ≤10-d subspace, and the v_j all
    share the common kernel factor k(x) — they differ only in the elementwise
    (x_j − Z_{:,j}) term. The ratio |v_jᵀ(Aû)| / ‖Aᵀv_j‖ is therefore near-constant in
    j, leaving a per-TRIAL quantity with no channel index. No reprojection fixes this:
    a margin gradient ∂/∂x[cos(ŷ,y_true) − cos(ŷ,y_distractor)] collapses identically,
    because the channel-dependence lives in that same barely-varying direction.

    NOTE — this is NOT the "leading singular value dominates" story. That was the
    audit's proposed mechanism and it is not what happens: on a synthetic fit with a
    nearly flat spectrum (pr_A ≈ 9.7 of a possible 10, i.e. σ₁ emphatically does NOT
    dominate) the ratio still collapses to CV 1.9 %. The collapse follows from the
    shared factorization above, not from spectral concentration. pr_A is recorded as
    a diagnostic, not as the explanation — do not cite it as the cause.

    φ(x) = k(x) @ N^T  with  k_j(x) = exp(-γ‖x - z_j‖²),  z_j = landmarks.
    ∂φ_m/∂x = Σ_j N_{mj} k_j (-2γ)(x - z_j);   ∂ŷ/∂x = Jφ^T @ A."""
    nys = model.named_steps["nys"]
    Z = nys.components_                                   # (L, d)
    N = nys.normalization_                                # (L, L)
    gamma = nys.gamma if nys.gamma is not None else 1.0 / nys.n_features_in_
    A, _ = _pls_affine(model, n_targets=None)             # (L, n_targets)

    s = np.linalg.svd(A, compute_uv=False)
    s2 = s ** 2
    pr_A = float(s2.sum() ** 2 / (s2 ** 2).sum()) if s2.sum() > 0 else np.nan

    d = X_te.shape[1]
    sens = np.zeros(d)
    ratios = []
    for x, y in zip(X_te, y_te):
        diff = x[None, :] - Z                             # (L, d)
        k = np.exp(-gamma * np.sum(diff * diff, axis=1))  # (L,)
        dk = (-2.0 * gamma) * (k[:, None] * diff)         # ∂k_j/∂x  (L, d)
        Jphi = N @ dk                                     # (L, d)
        J = Jphi.T @ A                                    # (d, n_targets)
        rows = np.linalg.norm(J, axis=1)                  # ‖J_j‖  (d,)
        sens += rows
        u = y - ybar
        nu = np.linalg.norm(u)
        den = rows.sum()
        if nu > 1e-12 and den > 1e-12:
            ratios.append(float(np.abs(J @ (u / nu)).sum() / den))
    sens /= X_te.shape[0]
    align = float(np.mean(ratios)) if ratios else np.nan
    return sens.reshape(n_hist, n_ch).sum(axis=0), align, pr_A   # (n_ch,), scalar, scalar


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        adj[i] = prev
    return adj


# NOTE (2026-07-23): plain-PLS VIP (`pls_vip` / `_pls_component_ssy`) was deleted
# here. It attributed a *linear surrogate* model that the paper does not report —
# there is no well-defined input-space VIP under the Nystroem map, which destroys
# the input↔feature correspondence — and as a region total it was an electrode-count
# proxy (within patient, ρ with n_channels = 0.98). Its one real use, "is the region
# ranking a Nystroem artifact?", is a linear-decoder control, not an importance
# measure, and is not worth a whole PLS path in this module.
#
# Covariance used to be computed inside the same function (`analyze_patient_region_vip`)
# and is now standalone in `analyze_patient_region_cov`, which needs no PLS fit at all.


# ── channel name resolution ───────────────────────────────────────────────
_DATA_DIR = Path(_MAIN_DIR) / "data"

# Mirror semantic_regression.py's patient-specific shank exclusions (see
# _PATIENT_EXCLUDE_PREFIXES there). Those channels are physically deleted from the
# model's data, so when load_patient falls back to positional ``ch{N}`` labels the
# name resolution must drop the same prefixes or ``ch{N}`` points at the wrong
# electrode. NOTE: only applied to the anatomical-name branch below. RB resolves
# names by integer position into its dataframe, whose channel order matches the
# (V-inclusive) data — RB's exclusion never fired at the SR stage because its
# channels are integer-named there — so RB must NOT be filtered here.
_PATIENT_EXCLUDE_PREFIXES = {
    "LH": ("O", "V", "P", "Q", "R"),
    "RB": ("V",),
}


def _build_channel_map(pat: str) -> dict:
    """Return {csv_label: electrode_name} for a patient. Returns {} on failure.

    AZ / LH / WBH : ch{N} -> clean_channel_names[N] from *_channels.pkl
    DR            : int N -> channel_names[N] from DR_picture_naming_df.pkl
    RB            : int N -> channel_names[N] from RB_picture_naming_combined_df.pkl
    AA            : names are already correct
    """
    try:
        if pat in ("AZ", "LH", "WBH"):
            pkls = sorted((_DATA_DIR / pat).glob(f"{pat}_*channels*.pkl"))
            if not pkls:
                return {}
            ch_df = pd.read_pickle(pkls[0])
            clean = ch_df[ch_df["clean"]]["channel_name"].astype(str).tolist()
            prefixes = _PATIENT_EXCLUDE_PREFIXES.get(pat)
            if prefixes:                       # drop the same shanks SR deleted
                clean = [c for c in clean if not c.startswith(prefixes)]
            return {f"ch{n}": name for n, name in enumerate(clean)}
        elif pat == "DR":
            import dill
            with open(_DATA_DIR / "DR" / "DR_picture_naming_df.pkl", "rb") as fh:
                df = dill.load(fh)
            cnames = df.iloc[0]["channel_names"]
            return {str(n): str(cnames[n]) for n in range(len(cnames))}
        elif pat == "RB":
            import dill
            with open(_DATA_DIR / "RB" / "RB_picture_naming_combined_df.pkl", "rb") as fh:
                df = dill.load(fh)
            cnames = df.iloc[0]["channel_names"]
            return {str(n): str(cnames[n]) for n in range(len(cnames))}
    except Exception as e:
        print(f"WARNING: could not resolve channel names for {pat}: {e}")
    return {}


# ── brain-region resolution ───────────────────────────────────────────────
def _elec_to_region(pat: str) -> dict:
    """{electrode_name: primary_roi} from {PAT}_*channels.pkl. {} if no region
    file exists (e.g. DR / RB have no *_channels.pkl)."""
    try:
        pkls = sorted((_DATA_DIR / pat).glob(f"{pat}_*channels*.pkl"))
        if not pkls:
            return {}
        ch_df = pd.read_pickle(pkls[0])
        if "primary_roi" not in ch_df.columns or "channel_name" not in ch_df.columns:
            return {}
        return {str(name): str(roi) for name, roi
                in zip(ch_df["channel_name"], ch_df["primary_roi"])}
    except Exception as e:
        print(f"WARNING: could not load brain regions for {pat}: {e}")
        return {}


# Atlas naming variants that denote the same ROI (normalised everywhere).
_ROI_NORMALIZE = {"temporo-occipital": "temporooccipital"}


def _normalize_roi(label) -> str:
    """Collapse atlas naming variants to one canonical spelling (always applied)."""
    s = str(label).strip()
    return _ROI_NORMALIZE.get(s, s)


def _merge_roi(label) -> str:
    """Coarser ROI: normalise, then strip a single-letter anterior/posterior prefix
    (aFus->Fus, pMTG->MTG, ...). Spelled-out 'ant depth' / 'post depth', 'frontal',
    'IPL', 'temporooccipital' etc. are left unchanged."""
    s = _normalize_roi(label)
    if len(s) >= 2 and s[0] in ("a", "p") and s[1].isupper():
        return s[1:]
    return s


def _build_region_labels(pat: str, chan_names: np.ndarray, merge: bool = False):
    """Brain region (primary_roi) per model channel index, or None if no region
    file exists.  Resolves each raw channel label -> electrode name (reusing the
    same _build_channel_map logic, so post-exclusion ch{N} positions line up) ->
    primary_roi.  Channels with no region match fall in 'unknown'.  Naming variants
    are always normalised; ``merge=True`` additionally collapses anterior/posterior
    gyral pairs into one coarser ROI (see _merge_roi)."""
    e2r = _elec_to_region(pat)
    if not e2r:
        return None
    chan_map = _build_channel_map(pat)            # raw label -> electrode (id for AA)
    labels = np.array([e2r.get(str(chan_map.get(str(c), c)), "unknown")
                       for c in chan_names], dtype=object)
    relabel = _merge_roi if merge else _normalize_roi
    labels = np.array([x if x == "unknown" else relabel(x) for x in labels],
                      dtype=object)
    if (labels == "unknown").all():
        return None
    return labels


def _significance_from_null(imp_pic: np.ndarray, imp_aud: np.ndarray,
                            null_pic: list, null_aud: list, alpha: float):
    """Group units (channels or regions) by permutation-null significance.

    One-sided p per unit = each bootstrap's observed Δacc vs. THAT bootstrap's
    pooled null (avoids the sqrt(n_bootstrap) scale-mismatch — see the fixed bug
    in CLAUDE.md), averaged across bootstraps, then BH-FDR corrected. A unit is
    significant for a task iff q < alpha and observed Δacc > 0.

    Returns (obs_pic, obs_aud, p_pic, p_aud, q_pic, q_aud, group)."""
    used, n_units = imp_pic.shape
    obs_pic, obs_aud = imp_pic.mean(0), imp_aud.mean(0)
    if null_pic:
        p_pic_boots = np.zeros((used, n_units))
        p_aud_boots = np.zeros((used, n_units))
        for b in range(used):
            nl_p, nl_a = null_pic[b], null_aud[b]
            for u in range(n_units):
                p_pic_boots[b, u] = (1 + np.sum(nl_p >= imp_pic[b, u])) / (1 + len(nl_p))
                p_aud_boots[b, u] = (1 + np.sum(nl_a >= imp_aud[b, u])) / (1 + len(nl_a))
        p_pic, p_aud = p_pic_boots.mean(0), p_aud_boots.mean(0)
    else:
        p_pic, p_aud = np.ones(n_units), np.ones(n_units)
    q_pic, q_aud = _bh_fdr(p_pic), _bh_fdr(p_aud)
    sig_pic = (q_pic < alpha) & (obs_pic > 0)
    sig_aud = (q_aud < alpha) & (obs_aud > 0)
    group = np.where(sig_pic & sig_aud, "both",
             np.where(sig_pic, "picture_only",
             np.where(sig_aud, "auditory_only", "neither")))
    return obs_pic, obs_aud, p_pic, p_aud, q_pic, q_aud, group


# ── per-patient analysis ─────────────────────────────────────────────────
def analyze_patient(patient: str, pic_run: str, aud_run: str,
                    n_bootstrap: int, test_frac: float, zero_shot_frac: float,
                    balance: str, n_perm_repeats: int,
                    alpha: float, rng_seed: int, metric: str,
                    region_null_shuffles: int = 20, merge: bool = False,
                    single_modality: bool = False, wb_null_shuffles: int = 200):
    """Bootstrapped kernel-PLS region-knockout permutation importance + region
    Jacobian sensitivity.

    Returns region_df (one row per region), or None when the patient has no
    {PAT}_*channels.pkl atlas (or every channel is unmatched). Each region's
    whole history block is shuffled jointly, so Δacc measures the population-
    level drop when an entire region is removed — robust to the redundancy that
    makes any single channel near-dispensable. A whole-brain block is knocked
    out as the ceiling (total Δacc the model attributes to the neural data),
    against which each region's Δacc / share is read.

    region_null_shuffles: the region null is pooled over only ~10 regions, so it
    needs more label shuffles than a per-channel pool for comparable p-value
    resolution. It uses a separate rng stream from the data-split rng.

    wb_null_shuffles: shuffles for the WHOLE-BRAIN ceiling test only. That test is
    a single group (not pooled over regions), so its p-value resolution is 1/(k+1)
    and at the region default (20) it quantizes to 0.0476 — which is where most
    patients' wb_p_pic landed, indistinguishable from "as low as this test can
    report". One group is cheap, so it gets its own, larger draw (default 200 →
    resolution 0.005). Set equal to region_null_shuffles to restore the old
    behaviour."""
    pic, aud = load_patient(patient, pic_run, aud_run)
    n_ch, n_hist = pic["n_channels"], pic["n_hist"]
    chan_names = pic["chan_names"]
    db_pic, db_aud = _build_db(pic), _build_db(aud)
    shared = np.array(sorted(set(pic["words"]) & set(aud["words"])))
    rng = np.random.default_rng(rng_seed)

    # Region grouping (None if this patient has no *_channels.pkl atlas). A
    # SEPARATE rng stream drives the region shuffles.  The whole-brain (all
    # channels) block is appended as one extra group: its Δacc is the "ceiling" —
    # the total accuracy the model attributes to the neural data — against which
    # each region's Δacc should be read.
    wb_cols = np.arange(n_ch * n_hist)
    region_labels = _build_region_labels(patient, np.asarray(chan_names)[:n_ch], merge=merge)
    if region_labels is None:
        return None
    rng_reg = np.random.default_rng(rng_seed + 99991)
    region_order = sorted(set(region_labels.tolist()))
    reg_idx = {r: np.where(region_labels == r)[0] for r in region_order}
    reg_cols = [_region_columns(reg_idx[r], n_ch, n_hist) for r in region_order]
    imp_groups = reg_cols + [wb_cols]  # regions then whole-brain (last col)

    rimp_pic = np.zeros((n_bootstrap, len(imp_groups)))    # primary metric (Δacc)
    rimp_aud = np.zeros((n_bootstrap, len(imp_groups)))
    cimp_pic = np.zeros((n_bootstrap, len(imp_groups)))    # Δcosine knockout
    cimp_aud = np.zeros((n_bootstrap, len(imp_groups)))
    rnull_pic, rnull_aud = [], []          # pooled region null (primary metric)
    wbnull_pic, wbnull_aud = [], []        # whole-brain null (its own scale)
    jac_pic = np.zeros((n_bootstrap, n_ch))  # ‖∂ŷ/∂x‖; per-channel, summed to region below
    jac_aud = np.zeros((n_bootstrap, n_ch))
    # rank-collapse diagnostics (scalars per bootstrap, NOT region columns) — these
    # replace the retired per-region retrieval-aligned Jacobian. See jacobian_measures.
    algn_pic, algn_aud, prA = [], [], []
    ybar_pic, ybar_aud = db_pic[0].mean(0), db_aud[0].mean(0)  # task GloVe centroids
    # single-modality (picture-only / auditory-only) decoders, each evaluated on its
    # OWN task's test set (same splits as co-trained). NaN rows where a fit failed.
    solo = {k: np.full((n_bootstrap, len(imp_groups) if k.startswith(("rimp", "cimp")) else n_ch), np.nan)
            for k in ("rimp_pic_s", "cimp_pic_s", "jac_pic_s",
                      "rimp_aud_s", "cimp_aud_s", "jac_aud_s")}
    used = 0

    for b in range(n_bootstrap):
        n_zs = int(round(len(shared) * zero_shot_frac))
        unseen = (set(rng.choice(shared, n_zs, replace=False).tolist())
                  if n_zs > 0 else set())
        p_tr, p_te = _stratified_word_split(pic["words"], unseen, test_frac, rng)
        a_tr, a_te = _stratified_word_split(aud["words"], unseen, test_frac, rng)
        if min(len(p_tr), len(a_tr), len(p_te), len(a_te)) < 3:
            continue
        bp, ba = _balance_pooled(p_tr, a_tr, balance, rng)
        X_pool = np.vstack([pic["X"][bp], aud["X"][ba]])
        y_pool = np.vstack([pic["y"][bp], aud["y"][ba]])

        model = make_model("kernel_pls", X_pool.shape[0], None)
        model.fit(X_pool, y_pool)

        jac_pic[used], _ap, _pr = jacobian_measures(
            model, pic["X"][p_te], pic["y"][p_te], ybar_pic, n_ch, n_hist)
        jac_aud[used], _aa, _ = jacobian_measures(
            model, aud["X"][a_te], aud["y"][a_te], ybar_aud, n_ch, n_hist)
        algn_pic.append(_ap); algn_aud.append(_aa); prA.append(_pr)

        kd_pic = _grouped_permutation_importance_multi(
            model, pic["X"][p_te], pic["words"][p_te], pic["cats"][p_te],
            db_pic, imp_groups, n_perm_repeats, rng_reg, [metric, "cosine_mean"])
        kd_aud = _grouped_permutation_importance_multi(
            model, aud["X"][a_te], aud["words"][a_te], aud["cats"][a_te],
            db_aud, imp_groups, n_perm_repeats, rng_reg, [metric, "cosine_mean"])
        rimp_pic[used], cimp_pic[used] = kd_pic[metric], kd_pic["cosine_mean"]
        rimp_aud[used], cimp_aud[used] = kd_aud[metric], kd_aud["cosine_mean"]

        if single_modality:
            for tag, tX, ty, twords, tcats, tdb, ybar, tr, te in (
                ("pic", pic["X"], pic["y"], pic["words"], pic["cats"], db_pic, ybar_pic, p_tr, p_te),
                ("aud", aud["X"], aud["y"], aud["words"], aud["cats"], db_aud, ybar_aud, a_tr, a_te)):
                try:  # own-task decoder, no cross-task balancing; may be tiny (auditory)
                    ms = make_model("kernel_pls", len(tr), None)
                    ms.fit(tX[tr], ty[tr])
                    js, _, _ = jacobian_measures(ms, tX[te], ty[te], ybar, n_ch, n_hist)
                    kd = _grouped_permutation_importance_multi(
                        ms, tX[te], twords[te], tcats[te], tdb, imp_groups,
                        n_perm_repeats, rng_reg, [metric, "cosine_mean"])
                    solo[f"jac_{tag}_s"][used] = js
                    solo[f"rimp_{tag}_s"][used] = kd[metric]
                    solo[f"cimp_{tag}_s"][used] = kd["cosine_mean"]
                except Exception as exc:
                    print(f"    [{patient}] {tag}-only fit failed (b{used}): {type(exc).__name__}")

        if region_null_shuffles > 0:
            rnull_pic.append(_grouped_null_importance(
                model, pic["X"][p_te], pic["words"][p_te], pic["cats"][p_te],
                db_pic, reg_cols, region_null_shuffles, rng_reg, metric))
            rnull_aud.append(_grouped_null_importance(
                model, aud["X"][a_te], aud["words"][a_te], aud["cats"][a_te],
                db_aud, reg_cols, region_null_shuffles, rng_reg, metric))
            # whole-brain ceiling is ONE group, so its p-value resolution is
            # 1/(k+1) with no pooling to help — give it its own, larger draw.
            wb_k = max(region_null_shuffles, wb_null_shuffles)
            wbnull_pic.append(_grouped_null_importance(
                model, pic["X"][p_te], pic["words"][p_te], pic["cats"][p_te],
                db_pic, [wb_cols], wb_k, rng_reg, metric))
            wbnull_aud.append(_grouped_null_importance(
                model, aud["X"][a_te], aud["words"][a_te], aud["cats"][a_te],
                db_aud, [wb_cols], wb_k, rng_reg, metric))

        used += 1
        print(f"  {patient}: bootstrap {used}/{n_bootstrap}")

    if used == 0:
        return None
    rimp_pic, rimp_aud = rimp_pic[:used], rimp_aud[:used]
    cimp_pic, cimp_aud = cimp_pic[:used], cimp_aud[:used]
    jac_pic, jac_aud = jac_pic[:used], jac_aud[:used]
    n_reg = len(region_order)
    # regions occupy the first n_reg cols; whole-brain is the last one
    (robs_pic, robs_aud, rp_pic, rp_aud,
     rq_pic, rq_aud, rgroup) = _significance_from_null(
        rimp_pic[:, :n_reg], rimp_aud[:, :n_reg], rnull_pic, rnull_aud, alpha)
    # whole-brain ceiling: total Δacc the model attributes to the neural data
    # (its own label-shuffle null, since a whole-brain block is a different
    # scale than a single region and can't share the pooled region null).
    (wbo_pic, wbo_aud, wbp_pic, wbp_aud,
     _, _, _) = _significance_from_null(
        rimp_pic[:, n_reg:n_reg + 1], rimp_aud[:, n_reg:n_reg + 1],
        wbnull_pic, wbnull_aud, alpha)
    wb_pic, wb_aud = float(wbo_pic[0]), float(wbo_aud[0])
    n_ch_in_reg = np.array([len(reg_idx[r]) for r in region_order])
    # region Jacobian = total (sum) sensitivity over the region's channels
    rjac_pic = np.array([jac_pic[:, reg_idx[r]].sum(1).mean() for r in region_order])
    rjac_aud = np.array([jac_aud[:, reg_idx[r]].sum(1).mean() for r in region_order])
    # Δcosine knockout (region cols only; no separate null / significance)
    cos_pic = np.nanmean(cimp_pic[:, :n_reg], axis=0)
    cos_aud = np.nanmean(cimp_aud[:, :n_reg], axis=0)
    # each region's Δacc as a fraction of the whole-brain ceiling
    frac_pic = robs_pic / wb_pic if abs(wb_pic) > 1e-9 else np.full(n_reg, np.nan)
    frac_aud = robs_aud / wb_aud if abs(wb_aud) > 1e-9 else np.full(n_reg, np.nan)

    # single-modality region totals (same aggregation as co-trained; nanmean over
    # bootstraps so failed/degenerate fits don't poison the estimate).
    solo_cols = {}
    if single_modality:
        def _knock_solo(a):   # region cols → per-region bootstrap nanmean
            return np.nanmean(a[:used, :n_reg], axis=0)
        def _jac_solo(a):     # per-channel → region-sum → bootstrap nanmean
            a = a[:used]
            return np.array([np.nanmean(np.nansum(a[:, reg_idx[r]], axis=1)) for r in region_order])
        for tag in ("pic", "aud"):
            solo_cols[f"perm_imp_{tag}_solo"] = _knock_solo(solo[f"rimp_{tag}_s"])
            solo_cols[f"cos_imp_{tag}_solo"] = _knock_solo(solo[f"cimp_{tag}_s"])
            solo_cols[f"jac_sens_{tag}_solo"] = _jac_solo(solo[f"jac_{tag}_s"])

    region_df = pd.DataFrame({
        "patient": patient,
        "metric": metric,
        "region": region_order,
        "n_channels": n_ch_in_reg,
        "perm_imp_pic": robs_pic, "perm_imp_aud": robs_aud,
        # per-channel-normalised Δacc separates "matters because it's big"
        # from "matters per electrode"
        "perm_imp_pic_per_ch": robs_pic / n_ch_in_reg,
        "perm_imp_aud_per_ch": robs_aud / n_ch_in_reg,
        "p_pic": rp_pic, "p_aud": rp_aud, "q_pic": rq_pic, "q_aud": rq_aud,
        "cos_imp_pic": cos_pic, "cos_imp_aud": cos_aud,
        "jac_sens_pic": rjac_pic, "jac_sens_aud": rjac_aud,
        # Rank-collapse diagnostics (constant per patient — NOT region measures).
        # jac_align = Σ|J·û|/Σ‖J‖ (what the retired per-region jac_dir reduced to);
        # jac_pr_A = effective output dimensionality of the PLS map. See
        # jacobian_measures for why the per-region jac_dir column was dropped.
        "jac_align_pic": float(np.nanmean(algn_pic)) if algn_pic else np.nan,
        "jac_align_aud": float(np.nanmean(algn_aud)) if algn_aud else np.nan,
        "jac_pr_A": float(np.nanmean(prA)) if prA else np.nan,
        "group": rgroup,
        # whole-brain ceiling (broadcast per patient) + each region's share
        "wb_imp_pic": wb_pic, "wb_imp_aud": wb_aud,
        "wb_p_pic": float(wbp_pic[0]), "wb_p_aud": float(wbp_aud[0]),
        "frac_wb_pic": frac_pic, "frac_wb_aud": frac_aud,
        **solo_cols,   # single-modality _solo columns (empty unless single_modality)
    }).sort_values(["group", "perm_imp_pic"], ascending=[True, False])
    return region_df


def _region_sum(feat: np.ndarray, reg_cols: list) -> np.ndarray:
    """Total (sum) of a per-feature score over each region's feature columns
    (feature layout: index = channel + n_ch·bin — see _channel_columns)."""
    return np.array([float(feat[cols].sum()) for cols in reg_cols])


def _feature_cov(X: np.ndarray, Y: np.ndarray, subtract_floor: bool = False) -> np.ndarray:
    """Per-feature neural↔GloVe covariance magnitude — the rawest form of the PLS
    objective (PLS maximises cov(X, Y)).  X columns are z-scored (amplitude-
    independent, like VIP scale=True); Y is mean-centred (raw GloVe geometry kept).
    Returns, per input feature j, ‖ Xc[:,j]ᵀ · Yc / (n-1) ‖₂  (L2 over GloVe dims).

    ``subtract_floor``: subtract the finite-sample null floor. With z-scored X, an
    uncoupled feature still has E‖cov_j‖ ≈ sqrt(trace(Cov Y)/(n-1)) (a scalar set by
    the trial count n), which is what makes the raw covariance scale ~1/sqrt(n) and
    cluster by participant. Subtracting it (clipped at 0) leaves covariance ABOVE
    chance — the trial-count artifact removed."""
    n = X.shape[0]
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Yc = Y - Y.mean(0)
    C = Xc.T @ Yc / max(n - 1, 1)                 # (n_features, n_targets)
    mag = np.linalg.norm(C, axis=1)               # (n_features,)
    if subtract_floor:
        floor = np.sqrt(np.sum(np.var(Y, axis=0, ddof=1)) / max(n - 1, 1))
        mag = np.maximum(mag - floor, 0.0)
    return mag


def analyze_patient_region_cov(patient: str, pic_run: str, aud_run: str,
                               merge: bool = False):
    """Region-total neural↔GloVe cross-covariance, per task.  Returns a per-region
    DataFrame, or None when the patient has no {PAT}_*channels.pkl atlas.

    The rawest form of the PLS objective (PLS maximises cov(X, Y)) and the only
    **model-free** region measure here — no fit, no split, no resampling, so it
    cannot be an artifact of the Nystroem approximation or of any train/test
    choice.  Metric-independent, therefore computed once per patient and broadcast
    across metrics by ``run_region_analysis``.

    Two flavours per task (see ``_feature_cov``): ``cov_*`` raw, and ``cov_nc_*``
    with the finite-sample floor √(trace(Cov Y)/(n−1)) subtracted and clipped at 0.
    Prefer ``cov_nc_*`` for anything cross-participant — the raw floor scales as
    1/√n_trials, so raw covariance largely sorts patients by trial count.

    History (2026-07-23): this was carved out of ``analyze_patient_region_vip``,
    which computed plain-PLS VIP and covariance together.  VIP was deleted (see the
    note above ``_build_channel_map``); covariance never used the VIP machinery —
    ``_feature_cov`` reads each task's own X/y directly — so the pooled fit, the
    ``balance`` resampling, the PLS knobs and the rng all disappeared with it.
    """
    pic, aud = load_patient(patient, pic_run, aud_run)
    n_ch, n_hist = pic["n_channels"], pic["n_hist"]
    chan_names = np.asarray(pic["chan_names"])[:n_ch]

    region_labels = _build_region_labels(patient, chan_names, merge=merge)
    if region_labels is None:
        return None
    region_order = sorted(set(region_labels.tolist()))
    reg_idx = {r: np.where(region_labels == r)[0] for r in region_order}
    reg_cols = [_region_columns(reg_idx[r], n_ch, n_hist) for r in region_order]
    n_ch_in_reg = np.array([len(reg_idx[r]) for r in region_order])

    # Each task's own trials, unbalanced and unresampled — covariance is a property
    # of the data, so there is nothing to fit and no split to make.
    cov_pic = _region_sum(_feature_cov(pic["X"], pic["y"]), reg_cols)
    cov_aud = _region_sum(_feature_cov(aud["X"], aud["y"]), reg_cols)
    # null-corrected (finite-sample floor removed) — the trial-count artifact that
    # makes raw covariance cluster by participant is subtracted out.
    cov_nc_pic = _region_sum(_feature_cov(pic["X"], pic["y"], subtract_floor=True), reg_cols)
    cov_nc_aud = _region_sum(_feature_cov(aud["X"], aud["y"], subtract_floor=True), reg_cols)

    df = pd.DataFrame({
        "patient": patient,
        "region": region_order,
        "n_channels": n_ch_in_reg,
        "cov_pic": cov_pic, "cov_aud": cov_aud,
        "cov_nc_pic": cov_nc_pic, "cov_nc_aud": cov_nc_aud,
        "n_trials_pic": int(pic["X"].shape[0]),
        "n_trials_aud": int(aud["X"].shape[0]),
    })
    return df.sort_values("cov_nc_pic", ascending=False).reset_index(drop=True)


# ── plotting ───────────────────────────────────────────────────────────────
def _grouped_barh(ax, sub, pic_col, aud_col, xlabel, wb=None):
    """Grouped horizontal pic/aud bars on `ax`, one row per region (order as
    given by `sub`). `wb=(pic, aud)` draws whole-brain ceiling reference lines."""
    y = np.arange(len(sub)); h = 0.4
    ax.barh(y + h / 2, sub[pic_col].values, height=h,
            color="#1f77b4", alpha=0.9, label="picture")
    ax.barh(y - h / 2, sub[aud_col].values, height=h,
            color="#d62728", alpha=0.9, label="auditory")
    ax.axvline(0, color="k", lw=0.6)
    if wb is not None:
        ax.axvline(wb[0], color="#1f77b4", lw=1.1, ls="--",
                   label=f"whole-brain pic ({wb[0]:+.3f})")
        ax.axvline(wb[1], color="#d62728", lw=1.1, ls="--",
                   label=f"whole-brain aud ({wb[1]:+.3f})")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r}  (n={n})" for r, n in
                        zip(sub["region"], sub["n_channels"])], fontsize=8)
    ax.set_xlabel(xlabel); ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=7, frameon=False, loc="lower right")


def _region_panels(df: pd.DataFrame, metric_tag: str, title: str, out: Path,
                   has_perm: bool, has_cov: bool) -> None:
    """One figure per patient/metric: up to three side-by-side panels sharing a
    single region y-order — permutation Δacc (pic/aud, with whole-brain ceiling),
    region Jacobian sensitivity (pic/aud), and region neural-GloVe covariance
    (null-corrected, pic/aud).  Regions are ordered by whichever primary method is
    present (perm Δacc, else covariance)."""
    sort_col = "perm_imp_pic" if has_perm else "cov_nc_pic"
    sub = df.sort_values(sort_col, ascending=True).reset_index(drop=True)
    panels = []
    if has_perm:
        panels.append("perm"); panels.append("jac")
    if has_cov:
        panels.append("cov")
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, max(3.0, 0.46 * len(sub))),
                             squeeze=False)
    axes = axes[0]
    for ax, kind in zip(axes, panels):
        if kind == "perm":
            wb = (float(sub["wb_imp_pic"].iloc[0]), float(sub["wb_imp_aud"].iloc[0])) \
                if "wb_imp_pic" in sub else None
            _grouped_barh(ax, sub, "perm_imp_pic", "perm_imp_aud",
                          f"Δ{metric_tag}  (entire region shuffled)", wb=wb)
            ax.set_title("permutation Δacc")
        elif kind == "jac":
            _grouped_barh(ax, sub, "jac_sens_pic", "jac_sens_aud",
                          "Σ ‖∂ŷ/∂x‖ over region")
            ax.set_title("Jacobian sensitivity")
        elif kind == "cov":
            _grouped_barh(ax, sub, "cov_nc_pic", "cov_nc_aud",
                          "Σ ‖cov(X, GloVe)‖ over region")
            ax.set_title("neural–GloVe covariance\n(null-corrected)")
    fig.suptitle(title, y=1.01)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── runner ──────────────────────────────────────────────────────────────
_COV_COLS = ["cov_pic", "cov_aud", "cov_nc_pic", "cov_nc_aud",
             "n_trials_pic", "n_trials_aud"]


def run_region_analysis(args, out_root: Path) -> None:
    """Region-level importance: kernel-PLS permutation Δacc + Jacobian
    (bootstrapped, per metric) and/or model-free neural↔GloVe covariance
    (metric-independent), merged per (patient, region) into a single
    region_importance_all.csv."""
    do_perm = args.analysis in ("permutation", "both")
    do_cov = args.analysis in ("covariance", "both")
    merge = getattr(args, "merge_regions", False)
    stem = "region_importance_merged" if merge else "region_importance"
    metrics = args.metric if do_perm else [args.metric[0]]  # covariance is metric-indep
    region_rows = []

    for pat in args.patient:
        # covariance is metric-independent — compute once per patient, broadcast
        cov_df = None
        if do_cov:
            print(f"[{pat}] region neural-GloVe covariance (merge={merge}) …")
            try:
                cov_df = analyze_patient_region_cov(
                    pat, args.pic_run, args.aud_run, merge=merge)
            except Exception as exc:
                print(f"  [{pat}] covariance FAILED: {type(exc).__name__}: {exc}")
            if cov_df is None:
                print(f"  [{pat}] no {{PAT}}_*channels.pkl atlas — covariance skipped")

        for metric in metrics:
            tag = _METRIC_TAG[metric]
            region_df = None
            if do_perm:
                print(f"[{pat}] region permutation importance "
                      f"(metric={metric}, merge={merge}) …")
                try:
                    region_df = analyze_patient(
                        pat, args.pic_run, args.aud_run, args.n_bootstrap,
                        args.test_frac, args.zero_shot_frac, args.balance,
                        args.n_perm_repeats, args.alpha, args.seed, metric,
                        region_null_shuffles=args.region_null_shuffles, merge=merge,
                        single_modality=getattr(args, "single_modality", False),
                        wb_null_shuffles=args.wb_null_shuffles)
                except Exception as exc:
                    print(f"  [{pat}] FAILED: {type(exc).__name__}: {exc}")
                if region_df is None:
                    print(f"  [{pat}] no {{PAT}}_*channels.pkl atlas — region skipped")

            # ── combine the two views on region ────────────────────────────
            if region_df is not None and cov_df is not None:
                merged = region_df.merge(
                    cov_df[["region"] + _COV_COLS], on="region", how="left")
            elif region_df is not None:
                merged = region_df
            elif cov_df is not None:
                merged = cov_df.copy()
                merged.insert(1, "metric", metric)
            else:
                continue

            pdir = out_root / pat
            pdir.mkdir(parents=True, exist_ok=True)
            merged.to_csv(pdir / f"{stem}_{pat}_{tag}.csv", index=False)
            _region_panels(merged, tag, f"{pat} · region importance (Δ {metric})",
                           pdir / f"{stem}_{pat}_{tag}.png",
                           has_perm=region_df is not None,
                           has_cov=cov_df is not None)
            region_rows.append(merged)

            if region_df is not None:
                best = region_df.sort_values("perm_imp_pic", ascending=False).iloc[0]
                print(f"  [{pat}/{tag}] {len(region_df)} regions; "
                      f"top pic region: {best['region']} "
                      f"(d_acc={best['perm_imp_pic']:+.4f}, n_ch={int(best['n_channels'])})")
            gc.collect()

    if region_rows:
        regdf = pd.concat(region_rows, ignore_index=True)
        regdf.to_csv(out_root / f"{stem}_all.csv", index=False)
        if "group" in regdf.columns:
            rsummary = (regdf.groupby(["patient", "metric", "group"]).size()
                        .unstack(fill_value=0))
            rsummary.to_csv(out_root / f"{stem}_group_counts.csv")
            print("Region group counts per patient:\n", rsummary)
        print(f"\nWrote region importance -> {out_root / (stem + '_all.csv')}")


# ── main ─────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient", nargs="*", default=SHARED_PATIENTS)
    ap.add_argument("--pic-run", default=PIC_RUN_DEFAULT)
    ap.add_argument("--aud-run", default=AUD_RUN_DEFAULT)
    ap.add_argument("--n-bootstrap", type=int, default=20)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--zero-shot-frac", type=float, default=0.3)
    ap.add_argument("--balance", default="none",
                    choices=["none", "downsample", "upsample"])
    ap.add_argument("--n-perm-repeats", type=int, default=5)
    ap.add_argument("--region-null-shuffles", type=int, default=20,
                    help="Label-shuffle null draws for the region significance test "
                         "(default 20). The region null is pooled over few units "
                         "(~10 regions) so it needs more shuffles than a per-channel "
                         "pool for comparable p-value resolution.")
    ap.add_argument("--wb-null-shuffles", type=int, default=200,
                    help="Label-shuffle draws for the WHOLE-BRAIN ceiling test only "
                         "(default 200). It is a single group with no pooling, so at "
                         "the region default (20) its p-value quantizes to 1/21=0.0476 "
                         "— which is exactly where most patients' wb_p_pic sat. One "
                         "group is cheap; this lifts it off that floor.")
    ap.add_argument("--alpha", type=float, default=ALPHA,
                    help=f"BH-FDR level (default {ALPHA:g}, from utils/config.py)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metric", nargs="+", default=["cat_indep_bal_acc"],
                    choices=list(METRICS),
                    help="Retrieval metric(s) driving the permutation importance "
                         "+ grouping. cat_indep_bal_acc (default) is more robust "
                         "than word_bal_acc; pass both to run each.")
    ap.add_argument("--analysis", choices=["permutation", "covariance", "both"],
                    default="both",
                    help="permutation (kernel-PLS region-knockout Δacc/Δcosine + "
                         "Jacobian), covariance (model-free neural-GloVe "
                         "cross-covariance; no fit, cheap), or both (default; merged "
                         "into one region_importance_all.csv). NOTE: 'vip' was removed "
                         "2026-07-23 — covariance used to be produced by that path and "
                         "is now its own.")
    ap.add_argument("--merge-regions", action="store_true",
                    help="Coarser ROIs: merge anterior/posterior gyral pairs "
                         "(aFus+pFus->Fus, aMTG+pMTG->MTG, ...) into one region and "
                         "normalise naming variants (temporo-occipital->temporooccipital). "
                         "'ant depth'/'post depth' are kept separate. Writes "
                         "region_importance_merged_all.csv instead of "
                         "region_importance_all.csv.")
    ap.add_argument("--single-modality", action="store_true",
                    help="Also train a picture-only and an auditory-only decoder per "
                         "patient (same splits as the co-trained model) and add "
                         "perm_imp/cos_imp/jac_sens _solo columns, each evaluated "
                         "on its own task. ~2-2.5x cost. Auditory-only is underpowered for "
                         "patients with few auditory trials (AA/DR). These columns are the "
                         "only two-independent-decoders control in the CSV — they are what "
                         "distinguishes genuine task specificity from the co-trained model "
                         "scoring both tasks through one shared map.")
    ap.add_argument("--out", default=None,
                    help="Output directory. Default: <results>/cross_task_cotrain/"
                         "balance_<BALANCE>/ — i.e. keyed on --balance, so the "
                         "resampling settings sit in parallel folders instead of one "
                         "of them colonising the analysis root. Pass this to override.")
    args = ap.parse_args()

    # Output is keyed on --balance so `none` and `downsample` are symmetric. NOTE the
    # per-patient subdirs at the analysis ROOT are shared with cross_task_cotrain.py
    # (which writes cotrain_{PAT}_*.png/csv into results/cross_task_cotrain/{PAT}/),
    # which is the other reason not to write region files there.
    out_root = Path(args.out) if args.out else (OUT_ROOT / f"balance_{args.balance}")
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {out_root}")

    run_region_analysis(args, out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
