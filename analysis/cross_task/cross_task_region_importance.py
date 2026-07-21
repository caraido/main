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

  3. Plain-PLS VIP (``--analysis vip`` / ``both``): a *plain* linear PLS fit on
     ALL pooled (pic+aud, peak-bin) trials — no train/test split — with per-input
     VIP (Variable Importance in Projection) summed over the region's feature
     columns.  Feature-level mean(VIP^2)=1, so a region's summed VIP scales with
     how much above-average signal it carries.  Metric-independent; merged into
     the region table on (patient, region).  A fast linear complement to the
     kernel-PLS permutation / Jacobian above.

Every ROI atlas is present now (all 6 cross-task patients AA/AZ/LH/WBH/DR/RB have
a {PAT}_*channels.pkl region file), so the region path runs for all of them.

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
from sklearn.cross_decomposition import PLSRegression

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
    N_PLS_COMPONENTS,
)
from utils.retrieval import compute_retrieval_metrics


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
    """Two per-channel Jacobian sensitivities of the co-trained model, computed in
    ONE per-trial pass (the Jacobian J is built once per trial):

      norm    — mean ‖∂ŷ/∂x‖₂  per feature (isotropic output sensitivity).
      aligned — mean |∂(ŷ·û)/∂x| = |J @ û| per feature, where û is the mean-centred
                unit true-GloVe direction of that trial (û = (y - ybar)/‖y - ybar‖).
                This is the *retrieval-aligned* gradient: how much a feature moves
                the prediction along the correct-answer direction, not just its
                magnitude.

    φ(x) = k(x) @ N^T  with  k_j(x) = exp(-γ‖x - z_j‖²),  z_j = landmarks.
    ∂φ_m/∂x = Σ_j N_{mj} k_j (-2γ)(x - z_j);   ∂ŷ/∂x = Jφ^T @ A.
    Both returned aggregated to per-channel (summed over history bins)."""
    nys = model.named_steps["nys"]
    Z = nys.components_                                   # (L, d)
    N = nys.normalization_                                # (L, L)
    gamma = nys.gamma if nys.gamma is not None else 1.0 / nys.n_features_in_
    A, _ = _pls_affine(model, n_targets=None)             # (L, n_targets)

    d = X_te.shape[1]
    sens = np.zeros(d)
    aln = np.zeros(d)
    for x, y in zip(X_te, y_te):
        diff = x[None, :] - Z                             # (L, d)
        k = np.exp(-gamma * np.sum(diff * diff, axis=1))  # (L,)
        dk = (-2.0 * gamma) * (k[:, None] * diff)         # ∂k_j/∂x  (L, d)
        Jphi = N @ dk                                     # (L, d)
        J = Jphi.T @ A                                    # (d, n_targets)
        sens += np.linalg.norm(J, axis=1)
        u = y - ybar
        nu = np.linalg.norm(u)
        if nu > 1e-12:
            aln += np.abs(J @ (u / nu))                   # |∂(ŷ·û)/∂x|  (d,)
    n = X_te.shape[0]
    sens /= n; aln /= n
    per_ch = lambda v: v.reshape(n_hist, n_ch).sum(axis=0)
    return per_ch(sens), per_ch(aln)                      # (n_ch,), (n_ch,)


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


# ── plain-PLS VIP channel importance (linear, trained on ALL pooled trials) ──
def _pls_component_ssy(pls) -> np.ndarray:
    """GloVe sum-of-squares explained by each latent component:
    SSY_a = ‖t_a‖² · ‖q_a‖²  (t = x_scores_, q = y_loadings_)."""
    return (pls.x_scores_ ** 2).sum(0) * (pls.y_loadings_ ** 2).sum(0)


def pls_vip(pls) -> np.ndarray:
    """Variable Importance in Projection, per input feature.

    VIP_j = sqrt( p · Σ_a SSY_a (w_{ja}/‖w_a‖)² / Σ_a SSY_a ).
    By construction mean(VIP²)=1, so VIP>1 flags above-average features — the
    usual PLS importance threshold.  w = x_weights_ (directions chosen for max
    X↔Y covariance), re-weighted by the GloVe variance each component explains:
    VIP credits features that build the components that actually predict GloVe,
    not merely those that explain neural (X) variance.
    """
    W = pls.x_weights_                                    # (p, A)
    ssy = _pls_component_ssy(pls)                         # (A,)
    Wn = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    return np.sqrt(W.shape[0] * ((Wn ** 2) * ssy[None, :]).sum(1) / (ssy.sum() + 1e-12))


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
                    region_null_shuffles: int = 20, merge: bool = False):
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
    resolution. It uses a separate rng stream from the data-split rng."""
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
    jdir_pic = np.zeros((n_bootstrap, n_ch))  # retrieval-aligned |∂(ŷ·û)/∂x|
    jdir_aud = np.zeros((n_bootstrap, n_ch))
    ybar_pic, ybar_aud = db_pic[0].mean(0), db_aud[0].mean(0)  # task GloVe centroids
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

        jac_pic[used], jdir_pic[used] = jacobian_measures(
            model, pic["X"][p_te], pic["y"][p_te], ybar_pic, n_ch, n_hist)
        jac_aud[used], jdir_aud[used] = jacobian_measures(
            model, aud["X"][a_te], aud["y"][a_te], ybar_aud, n_ch, n_hist)

        kd_pic = _grouped_permutation_importance_multi(
            model, pic["X"][p_te], pic["words"][p_te], pic["cats"][p_te],
            db_pic, imp_groups, n_perm_repeats, rng_reg, [metric, "cosine_mean"])
        kd_aud = _grouped_permutation_importance_multi(
            model, aud["X"][a_te], aud["words"][a_te], aud["cats"][a_te],
            db_aud, imp_groups, n_perm_repeats, rng_reg, [metric, "cosine_mean"])
        rimp_pic[used], cimp_pic[used] = kd_pic[metric], kd_pic["cosine_mean"]
        rimp_aud[used], cimp_aud[used] = kd_aud[metric], kd_aud["cosine_mean"]
        if region_null_shuffles > 0:
            rnull_pic.append(_grouped_null_importance(
                model, pic["X"][p_te], pic["words"][p_te], pic["cats"][p_te],
                db_pic, reg_cols, region_null_shuffles, rng_reg, metric))
            rnull_aud.append(_grouped_null_importance(
                model, aud["X"][a_te], aud["words"][a_te], aud["cats"][a_te],
                db_aud, reg_cols, region_null_shuffles, rng_reg, metric))
            wbnull_pic.append(_grouped_null_importance(
                model, pic["X"][p_te], pic["words"][p_te], pic["cats"][p_te],
                db_pic, [wb_cols], region_null_shuffles, rng_reg, metric))
            wbnull_aud.append(_grouped_null_importance(
                model, aud["X"][a_te], aud["words"][a_te], aud["cats"][a_te],
                db_aud, [wb_cols], region_null_shuffles, rng_reg, metric))

        used += 1
        print(f"  {patient}: bootstrap {used}/{n_bootstrap}")

    if used == 0:
        return None
    rimp_pic, rimp_aud = rimp_pic[:used], rimp_aud[:used]
    cimp_pic, cimp_aud = cimp_pic[:used], cimp_aud[:used]
    jac_pic, jac_aud = jac_pic[:used], jac_aud[:used]
    jdir_pic, jdir_aud = jdir_pic[:used], jdir_aud[:used]
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
    # retrieval-aligned Jacobian, same region-total aggregation
    rjdir_pic = np.array([jdir_pic[:, reg_idx[r]].sum(1).mean() for r in region_order])
    rjdir_aud = np.array([jdir_aud[:, reg_idx[r]].sum(1).mean() for r in region_order])
    # Δcosine knockout (region cols only; no separate null / significance)
    cos_pic = np.nanmean(cimp_pic[:, :n_reg], axis=0)
    cos_aud = np.nanmean(cimp_aud[:, :n_reg], axis=0)
    # each region's Δacc as a fraction of the whole-brain ceiling
    frac_pic = robs_pic / wb_pic if abs(wb_pic) > 1e-9 else np.full(n_reg, np.nan)
    frac_aud = robs_aud / wb_aud if abs(wb_aud) > 1e-9 else np.full(n_reg, np.nan)
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
        "jac_dir_pic": rjdir_pic, "jac_dir_aud": rjdir_aud,
        "group": rgroup,
        # whole-brain ceiling (broadcast per patient) + each region's share
        "wb_imp_pic": wb_pic, "wb_imp_aud": wb_aud,
        "wb_p_pic": float(wbp_pic[0]), "wb_p_aud": float(wbp_aud[0]),
        "frac_wb_pic": frac_pic, "frac_wb_aud": frac_aud,
    }).sort_values(["group", "perm_imp_pic"], ascending=[True, False])
    return region_df


def _region_sum(feat: np.ndarray, reg_cols: list) -> np.ndarray:
    """Total (sum) of a per-feature score over each region's feature columns
    (feature layout: index = channel + n_ch·bin — see _channel_columns)."""
    return np.array([float(feat[cols].sum()) for cols in reg_cols])


def _feature_cov(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per-feature neural↔GloVe covariance magnitude — the rawest form of the PLS
    objective (PLS maximises cov(X, Y)).  X columns are z-scored (amplitude-
    independent, like VIP scale=True); Y is mean-centred (raw GloVe geometry kept).
    Returns, per input feature j, ‖ Xc[:,j]ᵀ · Yc / (n-1) ‖₂  (L2 over GloVe dims)."""
    n = X.shape[0]
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Yc = Y - Y.mean(0)
    C = Xc.T @ Yc / max(n - 1, 1)                 # (n_features, n_targets)
    return np.linalg.norm(C, axis=1)              # (n_features,)


def analyze_patient_region_vip(patient: str, pic_run: str, aud_run: str,
                               balance: str, n_components: int, scale: bool,
                               n_bootstrap: int, rng_seed: int, merge: bool = False):
    """Fit a plain linear PLS on ALL pooled (pic+aud, peak-bin) trials and rank
    brain REGIONS by total VIP (Variable Importance in Projection), summed over
    each region's feature columns.  Returns a per-region DataFrame, or None when
    the patient has no {PAT}_*channels.pkl atlas.

    No train/test split: this inspects what the linear model *leans on*, so the
    most stable estimate uses every trial.  ``n_bootstrap`` (resampling the
    pooled trials) only adds a stability std to the ranking; the headline VIP
    comes from the single all-data fit.  VIP is metric-independent.

    ``scale=True`` (PLS standardises features) makes per-feature VIP comparable
    regardless of each channel's HGA amplitude — recommended for an importance
    comparison.  ``balance`` mirrors cross_task_cotrain so picture trials don't
    simply outvote auditory ones in the pooled fit.
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

    rng = np.random.default_rng(rng_seed)
    ip, ia = np.arange(len(pic["words"])), np.arange(len(aud["words"]))
    bp, ba = _balance_pooled(ip, ia, balance, rng)
    X = np.vstack([pic["X"][bp], aud["X"][ba]])
    y = np.vstack([pic["y"][bp], aud["y"][ba]])
    n_comp = max(1, min(n_components, X.shape[0] - 1, X.shape[1]))

    pls = PLSRegression(n_components=n_comp, scale=scale).fit(X, y)
    vip = _region_sum(pls_vip(pls), reg_cols)          # TOTAL per region

    vip_std = np.full(len(region_order), np.nan)
    if n_bootstrap > 0:
        vb = np.zeros((n_bootstrap, len(region_order)))
        N = X.shape[0]
        for b in range(n_bootstrap):
            sel = rng.integers(0, N, N)
            try:
                p = PLSRegression(n_components=n_comp, scale=scale).fit(X[sel], y[sel])
            except Exception:                  # degenerate resample — fall back
                vb[b] = vip
                continue
            vb[b] = _region_sum(pls_vip(p), reg_cols)
        vip_std = vb.std(0)

    # ── per-task VIP (measure 5) + neural↔GloVe covariance (measure 6) ──────
    # Separate picture-only / auditory-only PLS fits (not the pooled one above) —
    # what each task's decoder leans on — and the raw per-task cross-covariance.
    def _task_vip(tX, ty):
        nc = max(1, min(n_components, tX.shape[0] - 1, tX.shape[1]))
        try:
            tp = PLSRegression(n_components=nc, scale=scale).fit(tX, ty)
            return _region_sum(pls_vip(tp), reg_cols)
        except Exception:                          # too few trials to fit
            return np.full(len(region_order), np.nan)
    vip_pic = _task_vip(pic["X"], pic["y"])
    vip_aud = _task_vip(aud["X"], aud["y"])
    cov_pic = _region_sum(_feature_cov(pic["X"], pic["y"]), reg_cols)
    cov_aud = _region_sum(_feature_cov(aud["X"], aud["y"]), reg_cols)

    df = pd.DataFrame({
        "patient": patient,
        "region": region_order,
        "n_channels": n_ch_in_reg,
        "vip": vip, "vip_std": vip_std,
        "vip_pic": vip_pic, "vip_aud": vip_aud,
        "cov_pic": cov_pic, "cov_aud": cov_aud,
        "n_components": n_comp, "scaled": scale,
        "n_train_pooled": int(X.shape[0]),
    })
    df["vip_rank"] = df["vip"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("vip", ascending=False).reset_index(drop=True)


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
                   has_perm: bool, has_vip: bool) -> None:
    """One figure per patient/metric: up to three side-by-side panels sharing a
    single region y-order — permutation Δacc (pic/aud, with whole-brain ceiling),
    region-total Jacobian sensitivity (pic/aud), and region-total VIP.  Regions
    are ordered by whichever primary method is present (perm Δacc, else VIP)."""
    sort_col = "perm_imp_pic" if has_perm else "vip"
    sub = df.sort_values(sort_col, ascending=True).reset_index(drop=True)
    panels = []
    if has_perm:
        panels.append("perm"); panels.append("jac")
    if has_vip:
        panels.append("vip")
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
        elif kind == "vip":
            yv = np.arange(len(sub))
            xerr = sub["vip_std"].values if sub["vip_std"].notna().all() else None
            ax.barh(yv, sub["vip"].values, xerr=xerr, color="#4c72b0", alpha=0.85,
                    error_kw=dict(lw=0.7, ecolor="#444"))
            ax.set_yticks(yv)
            ax.set_yticklabels([f"{r}  (n={n_})" for r, n_ in
                                zip(sub["region"], sub["n_channels"])], fontsize=8)
            ax.set_xlabel("Σ VIP over region"); ax.grid(axis="x", alpha=0.3)
            ax.set_title("plain-PLS VIP")
    fig.suptitle(title, y=1.01)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── runner ──────────────────────────────────────────────────────────────
_VIP_COLS = ["vip", "vip_std", "vip_rank", "vip_pic", "vip_aud", "cov_pic", "cov_aud"]


def run_region_analysis(args, out_root: Path) -> None:
    """Region-level importance: kernel-PLS permutation Δacc + Jacobian
    (bootstrapped, per metric) and/or plain-PLS VIP (metric-independent), merged
    per (patient, region) into a single region_importance_all.csv."""
    do_perm = args.analysis in ("permutation", "both")
    do_vip = args.analysis in ("vip", "both")
    scale = not args.no_pls_scale
    merge = getattr(args, "merge_regions", False)
    stem = "region_importance_merged" if merge else "region_importance"
    metrics = args.metric if do_perm else [args.metric[0]]  # vip is metric-indep
    region_rows = []

    for pat in args.patient:
        # VIP is metric-independent — compute once per patient, broadcast to metrics
        vip_df = None
        if do_vip:
            print(f"[{pat}] plain-PLS region VIP "
                  f"(components={args.pls_components}, scale={scale}, "
                  f"bootstrap={args.pls_bootstrap}, merge={merge}) …")
            try:
                vip_df = analyze_patient_region_vip(
                    pat, args.pic_run, args.aud_run, args.balance,
                    args.pls_components, scale, args.pls_bootstrap, args.seed,
                    merge=merge)
            except Exception as exc:
                print(f"  [{pat}] VIP FAILED: {type(exc).__name__}: {exc}")
            if vip_df is None:
                print(f"  [{pat}] no {{PAT}}_*channels.pkl atlas — VIP skipped")

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
                        region_null_shuffles=args.region_null_shuffles, merge=merge)
                except Exception as exc:
                    print(f"  [{pat}] FAILED: {type(exc).__name__}: {exc}")
                if region_df is None:
                    print(f"  [{pat}] no {{PAT}}_*channels.pkl atlas — region skipped")

            # ── combine the two views on region ────────────────────────────
            if region_df is not None and vip_df is not None:
                merged = region_df.merge(
                    vip_df[["region"] + _VIP_COLS], on="region", how="left")
            elif region_df is not None:
                merged = region_df
            elif vip_df is not None:
                merged = vip_df.copy()
                merged.insert(1, "metric", metric)
            else:
                continue

            pdir = out_root / pat
            pdir.mkdir(parents=True, exist_ok=True)
            merged.to_csv(pdir / f"{stem}_{pat}_{tag}.csv", index=False)
            _region_panels(merged, tag, f"{pat} · region importance (Δ {metric})",
                           pdir / f"{stem}_{pat}_{tag}.png",
                           has_perm=region_df is not None,
                           has_vip=vip_df is not None)
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
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metric", nargs="+", default=["cat_indep_bal_acc"],
                    choices=list(METRICS),
                    help="Retrieval metric(s) driving the permutation importance "
                         "+ grouping. cat_indep_bal_acc (default) is more robust "
                         "than word_bal_acc; pass both to run each.")
    ap.add_argument("--analysis", choices=["permutation", "vip", "both"],
                    default="both",
                    help="permutation (kernel-PLS region-knockout Δacc + Jacobian), "
                         "vip (plain-PLS region VIP on ALL pooled trials), or both "
                         "(default; merged into one region_importance_all.csv).")
    ap.add_argument("--pls-components", type=int, default=N_PLS_COMPONENTS,
                    help="Latent components for the plain-PLS VIP analysis "
                         f"(default {N_PLS_COMPONENTS}, the project PLS default).")
    ap.add_argument("--pls-bootstrap", type=int, default=100,
                    help="Resamples of the pooled trials for VIP stability std "
                         "(0 = single all-data fit only).")
    ap.add_argument("--no-pls-scale", action="store_true",
                    help="Disable PLS feature scaling. Default scales features so "
                         "per-channel VIP is comparable across channels.")
    ap.add_argument("--merge-regions", action="store_true",
                    help="Coarser ROIs: merge anterior/posterior gyral pairs "
                         "(aFus+pFus->Fus, aMTG+pMTG->MTG, ...) into one region and "
                         "normalise naming variants (temporo-occipital->temporooccipital). "
                         "'ant depth'/'post depth' are kept separate. Writes "
                         "region_importance_merged_all.csv instead of "
                         "region_importance_all.csv.")
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    run_region_analysis(args, out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
