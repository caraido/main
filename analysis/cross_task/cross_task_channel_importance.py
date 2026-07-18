"""Per-channel importance for the co-trained (pooled pic+aud) semantic model.

Answers: which channels drive *both* tasks' retrieval accuracy, which are
picture-only, and which are auditory-only — for the SAME pooled kernel-PLS
model used in ``cross_task_cotrain.py``.

Two complementary attributions are produced per channel, evaluated on the
picture test set and the auditory test set separately:

  1. Permutation importance  (Δ word_bal_acc when the channel's full history
     block is shuffled across trials).  This is the "contributes to accuracy"
     measure and is robust to the RBF non-linearity.  Significance: a per-
     bootstrap label-shuffle null gives the noise floor of Δacc; one-sided
     p-values are pooled across bootstraps and BH-FDR corrected.

     The SAME permutation test is also run at BRAIN-REGION granularity
     (region_importance_*.csv): all channels in a region (primary_roi from
     {PAT}_*channels.pkl) are shuffled jointly, measuring the population-level
     accuracy drop when an entire region is removed.  This is the right scale
     when information is distributed redundantly across electrodes, so dropping
     any single channel barely moves accuracy but dropping a whole region does.
     Runs automatically for patients with a region file (AA/AZ/LH/WBH); DR/RB
     have none and are channel-only.

  2. Analytic Jacobian sensitivity  (mean ‖∂ŷ/∂x‖ back-propagated through the
     Nystroem-RBF map and the PLS affine map, aggregated over the channel's
     history columns).  This is the faithful local "back-projection through the
     kernel"; it scores sensitivity of the predicted GloVe embedding, not
     accuracy, so it is reported as a cross-check rather than for the grouping.

A third, independent view (``--analysis pls`` / ``both``) fits a *plain* linear
PLS on ALL pooled (pic+aud, peak-bin) trials — no train/test split — and ranks
channels by VIP (Variable Importance in Projection), the standard PLS importance:
mean(VIP^2)=1, so VIP>1 marks above-average channels.  VIP uses x_weights_ (the
directions chosen for max neural<->GloVe covariance) re-weighted by the GloVe
variance each latent component explains, so it credits channels that build the
components that actually predict the target, then averages over history bins.
This is a global "what the linear model leans on" ranking (no significance test),
a fast linear complement to the kernel-PLS permutation / Jacobian above.

The synthesis across all three methods (VIP + permutation Δacc + Jacobian) is
assembled by cross_task_channel_importance_report.py into a single HTML report.

Grouping (permutation-null significance):
    both        : sig. positive Δacc in BOTH tasks
    picture_only: sig. positive Δacc in pic only
    auditory_only: sig. positive Δacc in aud only
    neither     : sig. in neither

Memory note: this loads the per-patient semantic_regression_results.pkl
(100 MB – 2.6 GB each) via cross_task_cotrain.load_patient, so run it on a
machine with enough RAM (the project README recommends 16 GB+).  Run e.g.:

    python -m main.analysis.cross_task.cross_task_channel_importance
    python -m main.analysis.cross_task.cross_task_channel_importance --patient RB
    python -m main.analysis.cross_task.cross_task_channel_importance \
        --n-bootstrap 20 --n-perm-repeats 5 --null-shuffles 3
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


def permutation_importance(model, X_te, words_te, cats_te, db,
                           n_ch: int, n_hist: int, n_repeats: int,
                           rng: np.random.Generator, metric: str) -> np.ndarray:
    """Δ <metric> per channel (baseline − permuted), averaged over repeats."""
    group_cols = [_channel_columns(c, n_ch, n_hist) for c in range(n_ch)]
    return _grouped_permutation_importance(
        model, X_te, words_te, cats_te, db, group_cols, n_repeats, rng, metric)


def null_importance(model, X_te, words_te, cats_te, db,
                    n_ch: int, n_hist: int, n_shuffles: int,
                    rng: np.random.Generator, metric: str) -> np.ndarray:
    """Pooled (across channels) null of per-channel Δacc — see
    _grouped_null_importance."""
    group_cols = [_channel_columns(c, n_ch, n_hist) for c in range(n_ch)]
    return _grouped_null_importance(
        model, X_te, words_te, cats_te, db, group_cols, n_shuffles, rng, metric)


def _pls_affine(model, n_targets: int) -> Tuple[np.ndarray, np.ndarray]:
    """Recover the PLS affine map  ŷ = φ @ A + b  empirically (orientation-
    agnostic w.r.t. sklearn's coef_ convention)."""
    pls = model.named_steps["pls"]
    n_feat = model.named_steps["nys"].n_components
    b = pls.predict(np.zeros((1, n_feat)))[0]            # (n_targets,)
    A = pls.predict(np.eye(n_feat))                       # (n_feat, n_targets)
    A = A - b[None, :]
    return A, b


def jacobian_sensitivity(model, X_te, n_ch: int, n_hist: int) -> np.ndarray:
    """Mean ‖∂ŷ/∂x‖₂ per input feature, aggregated to per-channel.

    φ(x) = k(x) @ N^T  with  k_j(x) = exp(-γ‖x - z_j‖²),  z_j = landmarks.
    ∂φ_m/∂x = Σ_j N_{mj} k_j (-2γ)(x - z_j);   ∂ŷ/∂x = Jφ^T @ A.
    """
    nys = model.named_steps["nys"]
    Z = nys.components_                                   # (L, d)
    N = nys.normalization_                                # (L, L)
    gamma = nys.gamma if nys.gamma is not None else 1.0 / nys.n_features_in_
    A, _ = _pls_affine(model, n_targets=None)             # (L, n_targets)

    d = X_te.shape[1]
    sens = np.zeros(d)
    for x in X_te:
        diff = x[None, :] - Z                             # (L, d)
        k = np.exp(-gamma * np.sum(diff * diff, axis=1))  # (L,)
        dk = (-2.0 * gamma) * (k[:, None] * diff)         # ∂k_j/∂x  (L, d)
        Jphi = N @ dk                                     # (L, d)
        J = Jphi.T @ A                                    # (d, n_targets)
        sens += np.linalg.norm(J, axis=1)
    sens /= X_te.shape[0]
    return sens.reshape(n_hist, n_ch).sum(axis=0)         # per-channel


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


def _per_channel_mean(feat: np.ndarray, n_ch: int, n_hist: int) -> np.ndarray:
    """Average a per-feature score over a channel's history bins
    (feature layout: index = channel + n_ch·bin — see _channel_columns)."""
    return feat.reshape(n_hist, n_ch).mean(axis=0)


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


def _build_region_labels(pat: str, chan_names: np.ndarray):
    """Brain region (primary_roi) per model channel index, or None if no region
    file exists.  Resolves each raw channel label -> electrode name (reusing the
    same _build_channel_map logic, so post-exclusion ch{N} positions line up) ->
    primary_roi.  Channels with no region match fall in 'unknown'."""
    e2r = _elec_to_region(pat)
    if not e2r:
        return None
    chan_map = _build_channel_map(pat)            # raw label -> electrode (id for AA)
    labels = np.array([e2r.get(str(chan_map.get(str(c), c)), "unknown")
                       for c in chan_names], dtype=object)
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
                    balance: str, n_perm_repeats: int, null_shuffles: int,
                    alpha: float, rng_seed: int, metric: str,
                    regions: bool = True, region_null_shuffles: int = 20):
    """Bootstrapped kernel-PLS permutation importance + Jacobian sensitivity.

    Returns (channel_df, region_df). region_df groups the SAME permutation test
    by brain region (primary_roi from {PAT}_*channels.pkl): each region's whole
    history block is shuffled jointly, so it measures the population-level drop
    when an entire region is removed — robust to the redundancy that makes any
    single channel near-dispensable. region_df is None when no region file
    exists (DR / RB) or regions=False.

    region_null_shuffles is independent of the channel null_shuffles: the region
    null is pooled over only ~10 regions (vs. ~90 channels), so it needs more
    label shuffles for comparable p-value resolution. It uses a separate rng, so
    raising it does not perturb the channel-level results at all."""
    pic, aud = load_patient(patient, pic_run, aud_run)
    n_ch, n_hist = pic["n_channels"], pic["n_hist"]
    chan_names = pic["chan_names"]
    db_pic, db_aud = _build_db(pic), _build_db(aud)
    shared = np.array(sorted(set(pic["words"]) & set(aud["words"])))
    rng = np.random.default_rng(rng_seed)

    # Region grouping (None if this patient has no *_channels.pkl). A SEPARATE
    # rng stream is used for the region shuffles so adding the region analysis
    # leaves the channel-level results bit-for-bit unchanged.  The whole-brain
    # (all channels) block is appended as one extra group: its Δacc is the
    # "ceiling" — the total accuracy the model attributes to the neural data —
    # against which each region's Δacc should be read.
    wb_cols = np.arange(n_ch * n_hist)
    region_labels = _build_region_labels(patient, np.asarray(chan_names)[:n_ch]) if regions else None
    do_reg = region_labels is not None
    region_order, reg_idx, reg_cols, imp_groups = [], {}, [], []
    rimp_pic = np.zeros((n_bootstrap, 0))
    rimp_aud = np.zeros((n_bootstrap, 0))
    rnull_pic, rnull_aud = [], []          # pooled region null (regions only)
    wbnull_pic, wbnull_aud = [], []        # whole-brain null (its own scale)
    rng_reg = np.random.default_rng(rng_seed + 99991)
    if do_reg:
        region_order = sorted(set(region_labels.tolist()))
        reg_idx = {r: np.where(region_labels == r)[0] for r in region_order}
        reg_cols = [_region_columns(reg_idx[r], n_ch, n_hist) for r in region_order]
        imp_groups = reg_cols + [wb_cols]  # regions then whole-brain (last col)
        rimp_pic = np.zeros((n_bootstrap, len(imp_groups)))
        rimp_aud = np.zeros((n_bootstrap, len(imp_groups)))

    imp_pic = np.zeros((n_bootstrap, n_ch))
    imp_aud = np.zeros((n_bootstrap, n_ch))
    jac_pic = np.zeros((n_bootstrap, n_ch))
    jac_aud = np.zeros((n_bootstrap, n_ch))
    null_pic, null_aud = [], []
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

        imp_pic[used] = permutation_importance(
            model, pic["X"][p_te], pic["words"][p_te], pic["cats"][p_te],
            db_pic, n_ch, n_hist, n_perm_repeats, rng, metric)
        imp_aud[used] = permutation_importance(
            model, aud["X"][a_te], aud["words"][a_te], aud["cats"][a_te],
            db_aud, n_ch, n_hist, n_perm_repeats, rng, metric)
        jac_pic[used] = jacobian_sensitivity(model, pic["X"][p_te], n_ch, n_hist)
        jac_aud[used] = jacobian_sensitivity(model, aud["X"][a_te], n_ch, n_hist)
        if null_shuffles > 0:
            null_pic.append(null_importance(
                model, pic["X"][p_te], pic["words"][p_te], pic["cats"][p_te],
                db_pic, n_ch, n_hist, null_shuffles, rng, metric))
            null_aud.append(null_importance(
                model, aud["X"][a_te], aud["words"][a_te], aud["cats"][a_te],
                db_aud, n_ch, n_hist, null_shuffles, rng, metric))

        if do_reg:
            rimp_pic[used] = _grouped_permutation_importance(
                model, pic["X"][p_te], pic["words"][p_te], pic["cats"][p_te],
                db_pic, imp_groups, n_perm_repeats, rng_reg, metric)
            rimp_aud[used] = _grouped_permutation_importance(
                model, aud["X"][a_te], aud["words"][a_te], aud["cats"][a_te],
                db_aud, imp_groups, n_perm_repeats, rng_reg, metric)
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

    imp_pic, imp_aud = imp_pic[:used], imp_aud[:used]
    jac_pic, jac_aud = jac_pic[:used], jac_aud[:used]

    obs_pic, obs_aud, p_pic, p_aud, q_pic, q_aud, group = _significance_from_null(
        imp_pic, imp_aud, null_pic, null_aud, alpha)

    chan_df = pd.DataFrame({
        "patient": patient,
        "metric": metric,
        "channel": chan_names[:n_ch],
        "perm_imp_pic": obs_pic, "perm_imp_aud": obs_aud,
        "p_pic": p_pic, "p_aud": p_aud, "q_pic": q_pic, "q_aud": q_aud,
        "jac_sens_pic": jac_pic.mean(0), "jac_sens_aud": jac_aud.mean(0),
        "group": group,
    }).sort_values(["group", "perm_imp_pic"], ascending=[True, False])

    region_df = None
    if do_reg:
        rimp_pic, rimp_aud = rimp_pic[:used], rimp_aud[:used]
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
            "jac_sens_pic": rjac_pic, "jac_sens_aud": rjac_aud,
            "group": rgroup,
            # whole-brain ceiling (broadcast per patient) + each region's share
            "wb_imp_pic": wb_pic, "wb_imp_aud": wb_aud,
            "wb_p_pic": float(wbp_pic[0]), "wb_p_aud": float(wbp_aud[0]),
            "frac_wb_pic": frac_pic, "frac_wb_aud": frac_aud,
        }).sort_values(["group", "perm_imp_pic"], ascending=[True, False])

    return chan_df, region_df


def analyze_patient_pls_vip(patient: str, pic_run: str, aud_run: str,
                            balance: str, n_components: int, scale: bool,
                            n_bootstrap: int, rng_seed: int) -> pd.DataFrame:
    """Fit a plain linear PLS on ALL pooled (pic+aud, peak-bin) trials and rank
    channels by VIP (Variable Importance in Projection), averaged over bins.

    No train/test split: this inspects what the linear model *leans on*, so the
    most stable estimate uses every trial.  ``n_bootstrap`` (resampling the
    pooled trials) only adds a stability std to the ranking; the headline VIP
    comes from the single all-data fit.

    ``scale=True`` (PLS standardises features) makes per-channel VIP comparable
    regardless of each channel's HGA amplitude — recommended for an importance
    comparison.  ``balance`` mirrors cross_task_cotrain so picture trials don't
    simply outvote auditory ones in the pooled fit.
    """
    pic, aud = load_patient(patient, pic_run, aud_run)
    n_ch, n_hist = pic["n_channels"], pic["n_hist"]
    chan_names = np.asarray(pic["chan_names"])[:n_ch]
    rng = np.random.default_rng(rng_seed)

    ip, ia = np.arange(len(pic["words"])), np.arange(len(aud["words"]))
    bp, ba = _balance_pooled(ip, ia, balance, rng)
    X = np.vstack([pic["X"][bp], aud["X"][ba]])
    y = np.vstack([pic["y"][bp], aud["y"][ba]])
    n_comp = max(1, min(n_components, X.shape[0] - 1, X.shape[1]))

    pls = PLSRegression(n_components=n_comp, scale=scale).fit(X, y)
    vip = _per_channel_mean(pls_vip(pls), n_ch, n_hist)

    vip_std = np.full(n_ch, np.nan)
    if n_bootstrap > 0:
        vb = np.zeros((n_bootstrap, n_ch))
        N = X.shape[0]
        for b in range(n_bootstrap):
            sel = rng.integers(0, N, N)
            try:
                p = PLSRegression(n_components=n_comp, scale=scale).fit(X[sel], y[sel])
            except Exception:                  # degenerate resample — fall back
                vb[b] = vip
                continue
            vb[b] = _per_channel_mean(pls_vip(p), n_ch, n_hist)
        vip_std = vb.std(0)

    df = pd.DataFrame({
        "patient": patient,
        "channel": chan_names,
        "vip": vip, "vip_std": vip_std,
        "n_components": n_comp, "scaled": scale,
        "n_train_pooled": int(X.shape[0]),
    })
    df["vip_rank"] = df["vip"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("vip", ascending=False).reset_index(drop=True)


# ── plotting ───────────────────────────────────────────────────────────────
_GCOL = {"both": "#2ca02c", "picture_only": "#1f77b4",
         "auditory_only": "#d62728", "neither": "#bbbbbb"}


def _scatter(df: pd.DataFrame, xcol: str, ycol: str, title: str,
             xlab: str, ylab: str, out: Path, annotate_top: int = 5) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    for g, sub in df.groupby("group"):
        ax.scatter(sub[xcol], sub[ycol], s=26, c=_GCOL.get(g, "#777"),
                   label=f"{g} (n={len(sub)})", alpha=0.8, edgecolors="none")
    top = df.assign(_m=df[[xcol, ycol]].min(axis=1)).nlargest(annotate_top, "_m")
    for _, r in top.iterrows():
        ax.annotate(str(r["channel"]), (r[xcol], r[ycol]), fontsize=8,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _bar_top(df: pd.DataFrame, col: str, err: str, title: str, xlabel: str,
             out: Path, top: int = 20, vline: float | None = None) -> None:
    """Horizontal bar of the top-`top` channels by `col` (highest at top)."""
    sub = df.nlargest(top, col).iloc[::-1]
    xerr = sub[err].values if (err in sub and sub[err].notna().all()) else None
    fig, ax = plt.subplots(figsize=(5.6, max(3.2, 0.32 * len(sub))))
    ax.barh(np.arange(len(sub)), sub[col].values, xerr=xerr,
            color="#4c72b0", alpha=0.85, error_kw=dict(lw=0.7, ecolor="#444"))
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub["channel"].astype(str), fontsize=8)
    ax.set_xlabel(xlabel); ax.set_title(title); ax.grid(axis="x", alpha=0.3)
    if vline is not None:
        ax.axvline(vline, color="#d62728", lw=1.0, ls="--",
                   label=f"threshold ({vline:g})")
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _region_bar(df: pd.DataFrame, metric_tag: str, title: str, out: Path) -> None:
    """Grouped horizontal bars of Δacc per region (picture vs auditory), regions
    sorted by picture importance (largest at top). Each y-label notes the number
    of channels in the region (the population that gets knocked out)."""
    sub = df.sort_values("perm_imp_pic", ascending=True)
    y = np.arange(len(sub)); h = 0.4
    fig, ax = plt.subplots(figsize=(6.4, max(3.0, 0.46 * len(sub))))
    ax.barh(y + h / 2, sub["perm_imp_pic"].values, height=h,
            color="#1f77b4", alpha=0.9, label="picture")
    ax.barh(y - h / 2, sub["perm_imp_aud"].values, height=h,
            color="#d62728", alpha=0.9, label="auditory")
    ax.axvline(0, color="k", lw=0.6)
    # whole-brain ceiling (total attributable Δacc) as dashed reference lines
    if "wb_imp_pic" in sub:
        wbp, wba = float(sub["wb_imp_pic"].iloc[0]), float(sub["wb_imp_aud"].iloc[0])
        ax.axvline(wbp, color="#1f77b4", lw=1.1, ls="--",
                   label=f"whole-brain pic ({wbp:+.3f})")
        ax.axvline(wba, color="#d62728", lw=1.1, ls="--",
                   label=f"whole-brain aud ({wba:+.3f})")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r}  (n={n})" for r, n in
                        zip(sub["region"], sub["n_channels"])], fontsize=8)
    ax.set_xlabel(f"Δ{metric_tag}  (entire region shuffled)")
    ax.set_title(title); ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── runners ──────────────────────────────────────────────────────────────
def run_permutation_analysis(args, out_root: Path) -> None:
    """Kernel-PLS permutation importance + Jacobian sensitivity (bootstrapped),
    at both single-channel and brain-region granularity."""
    all_rows, region_rows = [], []
    for metric in args.metric:
        tag = _METRIC_TAG[metric]
        for pat in args.patient:
            print(f"[{pat}] analysing channel importance (metric={metric}) …")
            try:
                df, region_df = analyze_patient(
                    pat, args.pic_run, args.aud_run, args.n_bootstrap,
                    args.test_frac, args.zero_shot_frac, args.balance,
                    args.n_perm_repeats, args.null_shuffles, args.alpha,
                    args.seed, metric, regions=not args.no_regions,
                    region_null_shuffles=args.region_null_shuffles)
            except Exception as exc:
                print(f"  [{pat}] FAILED: {type(exc).__name__}: {exc}")
                continue
            chan_map = _build_channel_map(pat)
            if chan_map:
                df["channel"] = df["channel"].map(lambda x, m=chan_map: m.get(str(x), str(x)))
            pdir = out_root / pat
            pdir.mkdir(parents=True, exist_ok=True)
            df.to_csv(pdir / f"channel_importance_{pat}_{tag}.csv", index=False)
            _scatter(df, "perm_imp_pic", "perm_imp_aud",
                     f"{pat} · permutation importance (Δ {metric})",
                     f"Δ{tag} picture", f"Δ{tag} auditory",
                     pdir / f"channel_importance_{pat}_{tag}.png",
                     annotate_top=10)
            _scatter(df, "jac_sens_pic", "jac_sens_aud",
                     f"{pat} - Jacobian sensitivity (|grad-y / grad-x|)",
                     "sensitivity picture", "sensitivity auditory",
                     pdir / f"channel_jacobian_{pat}_{tag}.png")
            all_rows.append(df)
            print(f"  [{pat}/{tag}] groups: " +
                  ", ".join(f"{g}={int((df['group']==g).sum())}"
                            for g in ["both", "picture_only", "auditory_only", "neither"]))

            if region_df is not None:
                region_df.to_csv(pdir / f"region_importance_{pat}_{tag}.csv", index=False)
                _region_bar(region_df, tag,
                            f"{pat} · region permutation importance (Δ {metric})",
                            pdir / f"region_importance_{pat}_{tag}.png")
                region_rows.append(region_df)
                best = region_df.sort_values("perm_imp_pic", ascending=False).iloc[0]
                print(f"  [{pat}/{tag}] {len(region_df)} regions; "
                      f"top pic region: {best['region']} "
                      f"(d_acc={best['perm_imp_pic']:+.4f}, n_ch={int(best['n_channels'])})")
            elif not args.no_regions:
                print(f"  [{pat}] no *_channels.pkl region file — region analysis skipped")
            gc.collect()

    if all_rows:
        alldf = pd.concat(all_rows, ignore_index=True)
        alldf.to_csv(out_root / "channel_importance_all.csv", index=False)
        summary = (alldf.groupby(["patient", "metric", "group"]).size()
                   .unstack(fill_value=0))
        summary.to_csv(out_root / "channel_importance_group_counts.csv")
        print("\nGroup counts per patient:\n", summary)

    if region_rows:
        regdf = pd.concat(region_rows, ignore_index=True)
        regdf.to_csv(out_root / "region_importance_all.csv", index=False)
        rsummary = (regdf.groupby(["patient", "metric", "group"]).size()
                    .unstack(fill_value=0))
        rsummary.to_csv(out_root / "region_importance_group_counts.csv")
        print(f"\nWrote region importance -> {out_root / 'region_importance_all.csv'}")
        print("Region group counts per patient:\n", rsummary)


def run_pls_vip_analysis(args, out_root: Path) -> None:
    """Plain-PLS VIP channel importance on ALL pooled trials."""
    scale = not args.no_pls_scale
    rows = []
    for pat in args.patient:
        print(f"[{pat}] plain-PLS VIP importance "
              f"(components={args.pls_components}, scale={scale}, "
              f"bootstrap={args.pls_bootstrap}) …")
        try:
            df = analyze_patient_pls_vip(
                pat, args.pic_run, args.aud_run, args.balance,
                args.pls_components, scale, args.pls_bootstrap, args.seed)
        except Exception as exc:
            print(f"  [{pat}] FAILED: {type(exc).__name__}: {exc}")
            continue
        chan_map = _build_channel_map(pat)
        if chan_map:
            df["channel"] = df["channel"].map(lambda x, m=chan_map: m.get(str(x), str(x)))
        pdir = out_root / pat
        pdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(pdir / f"channel_pls_vip_{pat}.csv", index=False)
        _bar_top(df, "vip", "vip_std", f"{pat} · plain-PLS VIP (top channels)",
                 "VIP", pdir / f"channel_pls_vip_{pat}.png", vline=1.0)
        rows.append(df)
        top = df.head(5)["channel"].astype(str).tolist()
        print(f"  [{pat}] top-5 by VIP: {', '.join(top)}  "
              f"(VIP>1: {int((df['vip'] > 1).sum())}/{len(df)} channels)")
        gc.collect()

    if rows:
        alldf = pd.concat(rows, ignore_index=True)
        out = out_root / "channel_pls_vip_all.csv"
        alldf.to_csv(out, index=False)
        print(f"\nWrote plain-PLS VIP importance -> {out}")


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
    ap.add_argument("--null-shuffles", type=int, default=3)
    ap.add_argument("--no-regions", action="store_true",
                    help="Skip the brain-region permutation analysis (by default it "
                         "runs alongside the per-channel one for patients that have a "
                         "{PAT}_*channels.pkl region file: AA/AZ/LH/WBH).")
    ap.add_argument("--region-null-shuffles", type=int, default=20,
                    help="Label-shuffle null draws for the REGION significance test "
                         "(default 20). Independent of --null-shuffles and uses a "
                         "separate rng, so it never changes the channel results; the "
                         "region null is pooled over far fewer units (~10 regions) so "
                         "it needs more shuffles for comparable p-value resolution.")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metric", nargs="+", default=["cat_indep_bal_acc"],
                    choices=list(METRICS),
                    help="Retrieval metric(s) driving the permutation importance "
                         "+ grouping. cat_indep_bal_acc (default) is more robust "
                         "than word_bal_acc; pass both to run each.")
    ap.add_argument("--analysis", choices=["permutation", "pls", "both"],
                    default="permutation",
                    help="permutation (kernel-PLS Δacc + Jacobian, default), "
                         "pls (plain-PLS VIP importance on ALL pooled trials), "
                         "or both.")
    ap.add_argument("--pls-components", type=int, default=N_PLS_COMPONENTS,
                    help="Latent components for the plain-PLS VIP analysis "
                         f"(default {N_PLS_COMPONENTS}, the project PLS default).")
    ap.add_argument("--pls-bootstrap", type=int, default=100,
                    help="Resamples of the pooled trials for VIP/loading "
                         "stability std (0 = single all-data fit only).")
    ap.add_argument("--no-pls-scale", action="store_true",
                    help="Disable PLS feature scaling. Default scales features so "
                         "per-channel loadings are comparable across channels.")
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.analysis in ("permutation", "both"):
        run_permutation_analysis(args, out_root)
    if args.analysis in ("pls", "both"):
        run_pls_vip_analysis(args, out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
