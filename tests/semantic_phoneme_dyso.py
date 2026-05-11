# -*- coding: utf-8 -*-
"""
tests/semantic_phoneme_dyso.py
==============================
Isolate semantic vs phoneme geometry in neural activity using DySO in
embedding space + ridge-decoded neural axes per time bin.

Pipeline (per patient × bin)
----------------------------
1. Build per-trial semantic (GloVe, default) and phoneme (panphon) embeddings.
2. PCA each to a common dimensionality d_common; run DySO on the pair to
   obtain orthonormal embedding-side bases:
       U_S_emb   — semantic-unique
       U_P_emb   — phoneme-unique
       U_sh_emb  — shared (semantic AND phoneme)
3. Build per-trial targets in each embedding subspace:
       T_S_target  = S_pca @ U_S_emb @ U_S_emb.T
       T_P_target  = P_pca @ U_P_emb @ U_P_emb.T
       T_sh_target = mean of S_pca, P_pca projected onto U_sh_emb
4. Train ridge regressions neural -> each target. Use the regression
   coefficient matrices to recover neural-side axes; QR-orthonormalize
   each to get orthonormal neural-axis bases.
5. Project trial-level neural activity onto each neural-axis basis to get
   trial-level scores T_sem, T_phon, T_shared (each (n_trials, k_each)).
6. Score per-subspace R^2 against the held-out portion of S and P (KFold
   on trials), giving the time-resolved "is this subspace really semantic
   / phoneme / shared" curves.
7. Permutation null: shuffle word labels (preserving X), redo steps 2–6,
   record the null R^2 distribution per subspace per bin.

CLI
---
    python -m main.tests.semantic_phoneme_dyso                     # all patients, all bins
    python -m main.tests.semantic_phoneme_dyso --patients AA       # one patient
    python -m main.tests.semantic_phoneme_dyso --patient AA --bin 20 --smoke
    python -m main.tests.semantic_phoneme_dyso --task auditory_naming --run <auditory_run>

Output (under <project>/tests/results/semantic_phoneme_dyso/<patient>/):
    per_bin_metrics.csv     # bin, k_sem, k_phon, k_shared, R2_S_on_*, R2_P_on_*
    perm_null.csv           # bin, n_perm, mean and 95th-percentile null R2
    projections_peak.pkl    # neural-axis bases + trial projections at peak bin
    figures/dyso_traces.png # R^2 vs time per subspace
    figures/scatter_3d.png  # static 3D scatter at peak bins
    figures/quiver.png      # word-mean trajectories in 2D sem x phon plane
"""

from __future__ import annotations
import argparse
import gc
import os
import pickle as pk
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# project import
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))     # project root
sys.path.insert(0, str(_HERE.parents[0]))     # main/

from utils.dyso import dyso  # noqa: E402


# ── Constants ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEM_REG_DIR = PROJECT_ROOT / "semantic_regression"
DEFAULT_OUT = PROJECT_ROOT / "tests" / "results" / "semantic_phoneme_dyso"

DEFAULT_PIC_RUN = "2026-04-08_17-05-14_kernel_pls_cosine_50ep"
DEFAULT_AUD_RUN = "2026-05-06_19-12-48_auditory_naming_warp-linear_kernel_pls_cosine_50ep"

SHARED_PATIENTS = ["AA", "AZ", "DR", "LH", "RB", "WBH"]

D_COMMON_DEFAULT = 12       # PCA dim for both embeddings before DySO
DYSO_VAR_CUTOFF  = 99.0
RIDGE_ALPHA      = 1.5
N_PERM_DEFAULT   = 200
N_KFOLD          = 5

CATEGORY_COLORS = {
    "animal": "#1f77b4", "body part": "#ff7f0e", "food/fruit": "#2ca02c",
    "nature": "#d62728", "object/tool": "#9467bd", "vehicle": "#8c564b",
    "clothing": "#e377c2", "tool": "#7f7f7f", "other": "#bcbd22",
}
# Phoneme cluster by initial consonant (rough articulatory grouping)
_INITIAL_CLUSTERS = {
    "stop_voiceless": set("ptck"),
    "stop_voiced":    set("bdg"),
    "fricative":      set("fvszh"),
    "nasal":          set("mn"),
    "liquid":         set("lr"),
    "glide":          set("wy"),
    "vowel":          set("aeiou"),
}
def initial_cluster(word: str) -> str:
    if not word:
        return "vowel"
    c = word.strip().lower()[0]
    for name, chars in _INITIAL_CLUSTERS.items():
        if c in chars:
            return name
    return "other"

MARKER_BY_CLUSTER = {
    "stop_voiceless": "o", "stop_voiced": "s", "fricative": "^",
    "nasal": "D", "liquid": "v", "glide": "P", "vowel": "*", "other": "X",
}


# ── 1. Data loading ──────────────────────────────────────────────────────
def load_results_pkl(run_folder: str, patient: str) -> dict:
    """Load the heavy semantic_regression_results.pkl (needs project models pkg)."""
    path = SEM_REG_DIR / run_folder / patient / "semantic_regression_results.pkl"
    with open(path, "rb") as f:
        return pk.load(f)


def load_panphon_embeddings(words: np.ndarray, project_root: Path = PROJECT_ROOT) -> np.ndarray:
    """Map a (n_trials,) array of words to a (n_trials, 24) panphon embedding.
    Reuses the project's PWESuite cache. Falls back to NaN if the word is missing
    (drops via caller)."""
    sys.path.insert(0, str(project_root / "main"))
    from phoneme_regression import _map_phoneme_embed
    pkl = project_root / "main" / "embeddings" / "pictureNaming extended all" / "pwesuite_panphon_embeddings.pk"
    with open(pkl, "rb") as f:
        embed_dict = pk.load(f)
    return _map_phoneme_embed(embed_dict, words)


def get_neural_at_bin(reg, bin_idx: int) -> np.ndarray:
    return reg.X_to_use[bin_idx]


def get_trial_metadata(reg) -> pd.DataFrame:
    words = np.asarray(reg.labels).astype(str)
    w2i, i2c, wi2ci = reg.word_to_index, reg.index_to_category, reg.word_index_to_category_index
    cats = []
    for w in words:
        wi = w2i.get(str(w), w2i.get(w, 0))
        cats.append(str(i2c[wi2ci[wi]]))
    clusters = np.array([initial_cluster(w) for w in words])
    return pd.DataFrame({"word": words, "category": cats, "cluster": clusters})


# ── 2. DySO embedding decomposition ──────────────────────────────────────
def dyso_decompose_embeddings(S: np.ndarray, P: np.ndarray,
                              d_common: int = D_COMMON_DEFAULT,
                              var_cutoff: float = DYSO_VAR_CUTOFF,
                              ) -> dict:
    """PCA both embeddings to d_common dims; run DySO. Returns bases + projections."""
    n = S.shape[0]
    d_common = min(d_common, n - 1, S.shape[1], P.shape[1])

    pca_S = PCA(n_components=d_common, random_state=42).fit(S)
    pca_P = PCA(n_components=d_common, random_state=42).fit(P)
    S_pca = pca_S.transform(S)
    P_pca = pca_P.transform(P)

    res = dyso([S_pca, P_pca], var_cutoff=var_cutoff, combinations_mode="single")
    U_S = res.unique.get((0,), np.zeros((d_common, 0)))
    U_P = res.unique.get((1,), np.zeros((d_common, 0)))
    U_sh = res.shared

    return {
        "pca_S": pca_S, "pca_P": pca_P,
        "S_pca": S_pca, "P_pca": P_pca,
        "U_S_emb": U_S, "U_P_emb": U_P, "U_sh_emb": U_sh,
        "var_explained": res.var_explained,
        "d_common": d_common,
    }


def build_targets(emb_dec: dict) -> dict:
    """Project per-trial embeddings into each DySO subspace to get neural-target arrays."""
    S_pca, P_pca = emb_dec["S_pca"], emb_dec["P_pca"]
    U_S, U_P, U_sh = emb_dec["U_S_emb"], emb_dec["U_P_emb"], emb_dec["U_sh_emb"]

    # Semantic-private target: semantic content NOT shared with phoneme
    T_S_target = S_pca @ U_S @ U_S.T if U_S.shape[1] else np.zeros_like(S_pca)
    # Phoneme-private target
    T_P_target = P_pca @ U_P @ U_P.T if U_P.shape[1] else np.zeros_like(P_pca)
    # Shared content (average of both embeddings' projection onto the shared basis)
    if U_sh.shape[1]:
        T_sh_target = 0.5 * (S_pca @ U_sh @ U_sh.T + P_pca @ U_sh @ U_sh.T)
    else:
        T_sh_target = np.zeros_like(S_pca)
    return {"T_S_target": T_S_target, "T_P_target": T_P_target,
            "T_sh_target": T_sh_target}


# ── 3. Neural axes via ridge regression + QR ─────────────────────────────
def fit_neural_axes(X: np.ndarray, target: np.ndarray, alpha: float = RIDGE_ALPHA
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Ridge-regress X -> target. Return (W, U_orth) where W = coefficient matrix
    (n_features, k_target) and U_orth = QR-orthonormalized columns (n_features, rank)."""
    if target.shape[1] == 0:
        return np.zeros((X.shape[1], 0)), np.zeros((X.shape[1], 0))
    model = Ridge(alpha=alpha, fit_intercept=True).fit(X, target)
    W = model.coef_.T            # (n_features, k_target)
    Q, R = np.linalg.qr(W)
    # Truncate to numerical rank
    diag = np.abs(np.diag(R))
    keep = diag > 1e-8 * diag.max() if diag.size else np.array([], dtype=bool)
    U_orth = Q[:, keep] if keep.any() else Q[:, :1]
    return W, U_orth


def decompose_neural_at_bin(X: np.ndarray, S: np.ndarray, P: np.ndarray,
                             d_common: int = D_COMMON_DEFAULT,
                             ridge_alpha: float = RIDGE_ALPHA,
                             shared_cos_threshold: float = 0.5,
                             ) -> dict:
    """TDR-style decomposition of neural space into semantic-private,
    phoneme-private, and shared subspaces.

    1. PCA both embeddings to d_common dims (regularizes regression).
    2. Ridge-regress neural -> S_pca and neural -> P_pca, giving coefficient
       matrices W_S, W_P (n_neural, d_common). Each column is a regression-axis
       in neural space.
    3. QR-orthonormalize each to get axis bases Q_S, Q_P.
    4. Principal angles between Q_S and Q_P (via SVD of Q_S.T @ Q_P) reveal
       which directions are SHARED (cosine > threshold) vs PRIVATE.
    5. Shared basis = mean of matched pairs from each side; private bases are
       what remains after projecting the shared component out. All three bases
       are pairwise orthogonal in neural space by construction.
    """
    n_trials, n_neural = X.shape
    d_common = min(d_common, n_trials - 1, S.shape[1], P.shape[1])

    pca_S = PCA(n_components=d_common, random_state=42).fit(S)
    pca_P = PCA(n_components=d_common, random_state=42).fit(P)
    S_pca = pca_S.transform(S)
    P_pca = pca_P.transform(P)

    ridge_S = Ridge(alpha=ridge_alpha, fit_intercept=True).fit(X, S_pca)
    ridge_P = Ridge(alpha=ridge_alpha, fit_intercept=True).fit(X, P_pca)
    W_S = ridge_S.coef_.T
    W_P = ridge_P.coef_.T

    def _qr_safe(W):
        if W.shape[1] == 0:
            return np.zeros((W.shape[0], 0))
        Q, R = np.linalg.qr(W)
        diag = np.abs(np.diag(R))
        if diag.size == 0:
            return Q[:, :0]
        keep = diag > 1e-8 * diag.max()
        return Q[:, keep] if keep.any() else Q[:, :1]

    Q_S = _qr_safe(W_S)
    Q_P = _qr_safe(W_P)

    if Q_S.shape[1] == 0 or Q_P.shape[1] == 0:
        zero = np.zeros((n_neural, 0))
        return {
            "pca_S": pca_S, "pca_P": pca_P, "S_pca": S_pca, "P_pca": P_pca,
            "W_S": W_S, "W_P": W_P,
            "U_sem_neural": Q_S, "U_phon_neural": Q_P, "U_shared_neural": zero,
            "T_sem": X @ Q_S, "T_phon": X @ Q_P, "T_shared": X @ zero,
            "principal_cosines": np.zeros(0), "is_shared_direction": np.zeros(0, bool),
            "d_common": d_common,
        }

    # Principal angles between the two regression-axis bases
    M = Q_S.T @ Q_P
    U_M, sig, Vt = np.linalg.svd(M, full_matrices=False)
    is_shared = sig > shared_cos_threshold
    k_shared = int(is_shared.sum())

    # Shared neural basis: average of matched pairs from each side
    if k_shared > 0:
        shared_S = Q_S @ U_M[:, :k_shared]
        shared_P = Q_P @ Vt[:k_shared, :].T
        shared = (shared_S + shared_P) / 2.0
        U_shared = _qr_safe(shared)
    else:
        U_shared = np.zeros((n_neural, 0))

    def _project_out(Q, U):
        return Q if U.shape[1] == 0 else Q - U @ (U.T @ Q)

    U_sem = _qr_safe(_project_out(Q_S, U_shared))
    U_phon = _qr_safe(_project_out(Q_P, U_shared))
    # Also enforce U_sem perp U_phon (asymptotically true; numerically clean up)
    if U_phon.shape[1] and U_sem.shape[1]:
        U_phon = _qr_safe(_project_out(U_phon, U_sem))

    T_sem = X @ U_sem
    T_phon = X @ U_phon
    T_shared = X @ U_shared

    return {
        "pca_S": pca_S, "pca_P": pca_P, "S_pca": S_pca, "P_pca": P_pca,
        "W_S": W_S, "W_P": W_P,
        "U_sem_neural": U_sem, "U_phon_neural": U_phon, "U_shared_neural": U_shared,
        "T_sem": T_sem, "T_phon": T_phon, "T_shared": T_shared,
        "principal_cosines": sig, "is_shared_direction": is_shared,
        "d_common": d_common,
    }


# ── 4. Cross-validated R² ────────────────────────────────────────────────
def _r2_from_target(T_proj: np.ndarray, target: np.ndarray) -> float:
    """Linear-regression R² of target ~ T_proj (closed form, in-sample).
    Used WITH train/test splits in the caller."""
    if T_proj.size == 0 or target.size == 0 or T_proj.shape[1] == 0:
        return 0.0
    Y_hat = T_proj @ np.linalg.lstsq(T_proj, target, rcond=None)[0]
    ss_res = float(np.sum((target - Y_hat) ** 2))
    ss_tot = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def cross_validated_r2(X: np.ndarray, S: np.ndarray, P: np.ndarray,
                        d_common: int = D_COMMON_DEFAULT,
                        ridge_alpha: float = RIDGE_ALPHA,
                        n_splits: int = N_KFOLD,
                        random_state: int = 42) -> dict:
    """K-fold CV: fit DySO + neural axes on train, evaluate R² on test for each
    subspace × each target. Returns averaged R² across folds."""
    n_trials = X.shape[0]
    if n_trials < n_splits + 2:
        n_splits = max(2, n_trials // 2)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    keys = ["R2_S_on_sem", "R2_S_on_phon", "R2_S_on_shared",
            "R2_P_on_sem", "R2_P_on_phon", "R2_P_on_shared"]
    accum = {k: [] for k in keys}
    accum["k_sem"], accum["k_phon"], accum["k_shared"] = [], [], []

    for tr_idx, te_idx in kf.split(X):
        Xtr, Xte = X[tr_idx], X[te_idx]
        Str, Ste = S[tr_idx], S[te_idx]
        Ptr, Pte = P[tr_idx], P[te_idx]

        dec = decompose_neural_at_bin(Xtr, Str, Ptr,
                                       d_common=d_common, ridge_alpha=ridge_alpha)
        # Apply train-side PCAs to test embeddings
        Ste_pca = dec["pca_S"].transform(Ste)
        Pte_pca = dec["pca_P"].transform(Pte)

        # Project test neural onto train-derived axes
        T_sem_te    = Xte @ dec["U_sem_neural"]
        T_phon_te   = Xte @ dec["U_phon_neural"]
        T_shared_te = Xte @ dec["U_shared_neural"]

        # Evaluate each test-fold against its OWN held-out embeddings
        accum["R2_S_on_sem"]   .append(_r2_from_target(T_sem_te,    Ste_pca))
        accum["R2_S_on_phon"]  .append(_r2_from_target(T_phon_te,   Ste_pca))
        accum["R2_S_on_shared"].append(_r2_from_target(T_shared_te, Ste_pca))
        accum["R2_P_on_sem"]   .append(_r2_from_target(T_sem_te,    Pte_pca))
        accum["R2_P_on_phon"]  .append(_r2_from_target(T_phon_te,   Pte_pca))
        accum["R2_P_on_shared"].append(_r2_from_target(T_shared_te, Pte_pca))
        accum["k_sem"]   .append(dec["U_sem_neural"].shape[1])
        accum["k_phon"]  .append(dec["U_phon_neural"].shape[1])
        accum["k_shared"].append(dec["U_shared_neural"].shape[1])
    return {k: float(np.mean(v)) for k, v in accum.items()}


# ── 5. Permutation null ──────────────────────────────────────────────────
def permutation_null(X: np.ndarray, S: np.ndarray, P: np.ndarray,
                      n_perm: int = N_PERM_DEFAULT,
                      d_common: int = D_COMMON_DEFAULT,
                      ridge_alpha: float = RIDGE_ALPHA,
                      n_splits: int = N_KFOLD,
                      seed: int = 0) -> dict:
    """Shuffle trial-level labels (S and P together) against neural X. Returns
    mean and 95th percentile of null R² for each subspace x target combination."""
    rng = np.random.default_rng(seed)
    n_trials = X.shape[0]
    keys = ["R2_S_on_sem", "R2_S_on_phon", "R2_S_on_shared",
            "R2_P_on_sem", "R2_P_on_phon", "R2_P_on_shared"]
    null = {k: [] for k in keys}
    for i in range(n_perm):
        idx = rng.permutation(n_trials)
        res = cross_validated_r2(X, S[idx], P[idx],
                                  d_common=d_common, ridge_alpha=ridge_alpha,
                                  n_splits=n_splits, random_state=seed + i)
        for k in keys:
            null[k].append(res[k])
    out = {}
    for k, vals in null.items():
        a = np.asarray(vals)
        out[f"{k}_null_mean"] = float(a.mean())
        out[f"{k}_null_p95"]  = float(np.percentile(a, 95))
        out[f"{k}_null_std"]  = float(a.std())
    return out


# ── 6. Plotting ──────────────────────────────────────────────────────────
def plot_traces(df: pd.DataFrame, out_path: Path, bin_size_ms: int = 100) -> None:
    """Per-bin R² trace for each subspace × target."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    t = df["bin_index"].values * bin_size_ms / 1000.0
    # Left panel: S target
    for col, color, label in [
        ("R2_S_on_sem",    "#1f77b4", "U_sem ← S (private)"),
        ("R2_S_on_phon",   "#d62728", "U_phon ← S (off-target)"),
        ("R2_S_on_shared", "#7f7f7f", "U_shared ← S"),
    ]:
        axes[0].plot(t, df[col].values, label=label, color=color, lw=1.5)
        if f"{col}_null_p95" in df.columns:
            axes[0].fill_between(t, df[f"{col}_null_mean"], df[f"{col}_null_p95"],
                                 color=color, alpha=0.12)
    axes[0].set_title("Semantic embedding (S) prediction by subspace")
    axes[0].set_xlabel("time (s)"); axes[0].set_ylabel("R² (CV)")
    axes[0].axhline(0, color="grey", lw=0.5); axes[0].legend(fontsize=8)

    for col, color, label in [
        ("R2_P_on_phon",   "#d62728", "U_phon ← P (private)"),
        ("R2_P_on_sem",    "#1f77b4", "U_sem ← P (off-target)"),
        ("R2_P_on_shared", "#7f7f7f", "U_shared ← P"),
    ]:
        axes[1].plot(t, df[col].values, label=label, color=color, lw=1.5)
        if f"{col}_null_p95" in df.columns:
            axes[1].fill_between(t, df[f"{col}_null_mean"], df[f"{col}_null_p95"],
                                 color=color, alpha=0.12)
    axes[1].set_title("Phoneme embedding (P) prediction by subspace")
    axes[1].set_xlabel("time (s)"); axes[1].axhline(0, color="grey", lw=0.5)
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def plot_3d_scatter(T_sem: np.ndarray, T_phon: np.ndarray, T_shared: np.ndarray,
                     meta: pd.DataFrame, out_path: Path) -> None:
    """3D scatter with axes (U_sem[0], U_phon[0], U_shared[0])."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111, projection="3d")
    x = T_sem[:, 0] if T_sem.shape[1] else np.zeros(len(meta))
    y = T_phon[:, 0] if T_phon.shape[1] else np.zeros(len(meta))
    z = T_shared[:, 0] if T_shared.shape[1] else np.zeros(len(meta))
    for cat in sorted(meta["category"].unique()):
        for clu in sorted(meta["cluster"].unique()):
            mask = (meta["category"].values == cat) & (meta["cluster"].values == clu)
            if mask.sum() == 0: continue
            ax.scatter(x[mask], y[mask], z[mask],
                       c=CATEGORY_COLORS.get(cat, "#999"),
                       marker=MARKER_BY_CLUSTER.get(clu, "o"),
                       s=46, alpha=0.85, edgecolors="white", linewidths=0.4)
    for i in range(len(meta)):
        ax.text(x[i], y[i], z[i], str(meta["word"].iloc[i]),
                fontsize=6, alpha=0.55)
    ax.set_xlabel("U_sem PC1"); ax.set_ylabel("U_phon PC1"); ax.set_zlabel("U_shared PC1")
    # Legend (categories only; cluster legend kept short in caption)
    handles = [Line2D([0],[0], marker='o', linestyle='',
                       color=CATEGORY_COLORS.get(c, "#999"), label=c)
               for c in sorted(meta["category"].unique())]
    ax.legend(handles=handles, fontsize=7, loc="upper left", bbox_to_anchor=(1.05, 1.0))
    ax.set_title("Trials in orthogonal neural space\n(color=category, shape=initial-phoneme cluster)")
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_word_trajectory_quiver(per_bin_proj: dict, meta: pd.DataFrame,
                                  bin_range: tuple[int, int], out_path: Path,
                                  bin_size_ms: int = 100) -> None:
    """For each unique word, plot its trajectory in the (sem, phon) plane
    across bin_range. Color = category."""
    if not per_bin_proj:
        return
    bins = sorted(per_bin_proj.keys())
    fig, ax = plt.subplots(figsize=(7, 6))
    drawn = set()
    unique_words = sorted(meta["word"].unique())
    for word in unique_words:
        rows = (meta["word"].values == word)
        cat = meta.loc[rows, "category"].iloc[0]
        color = CATEGORY_COLORS.get(cat, "#999")
        traj_x, traj_y = [], []
        for b in bins:
            if b < bin_range[0] or b > bin_range[1]: continue
            d = per_bin_proj[b]
            T_sem = d["T_sem"][rows]; T_phon = d["T_phon"][rows]
            if T_sem.shape[1] == 0 or T_phon.shape[1] == 0: continue
            traj_x.append(T_sem[:, 0].mean())
            traj_y.append(T_phon[:, 0].mean())
        if len(traj_x) < 2: continue
        ax.plot(traj_x, traj_y, "-", color=color, alpha=0.6, lw=1.0)
        ax.scatter(traj_x[0], traj_y[0], s=18, c=color, marker="o", edgecolors="white")
        ax.scatter(traj_x[-1], traj_y[-1], s=46, c=color, marker=">", edgecolors="white")
        if cat not in drawn:
            ax.scatter([], [], c=color, s=42, label=cat); drawn.add(cat)
        ax.text(traj_x[-1], traj_y[-1], word, fontsize=6, alpha=0.7)
    ax.set_xlabel("U_sem PC1"); ax.set_ylabel("U_phon PC1")
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.set_title(f"Word trajectories (bins {bin_range[0]}–{bin_range[1]}, "
                 f"= {bin_range[0]*bin_size_ms/1000:.2f}–{bin_range[1]*bin_size_ms/1000:.2f}s)\n"
                 "o = first bin, ▶ = last bin")
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


# ── 7. Per-patient driver ────────────────────────────────────────────────
def analyze_patient(patient: str, run_folder: str, out_dir: Path,
                     bins: list[int] | None = None,
                     d_common: int = D_COMMON_DEFAULT,
                     ridge_alpha: float = RIDGE_ALPHA,
                     n_perm: int = 0,
                     n_splits: int = N_KFOLD,
                     save_figs: bool = True,
                     verbose: bool = True) -> pd.DataFrame:
    """Run DySO-based semantic/phoneme decomposition for one patient.
    Returns the per-bin metrics dataframe."""
    out_dir = out_dir / patient
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    print(f"\n=== {patient} ({run_folder}) ===", flush=True)
    d = load_results_pkl(run_folder, patient)
    reg = d["regressors"]["GloVe"]
    meta = get_trial_metadata(reg)
    words = meta["word"].values

    S = reg.y                                    # GloVe per trial (300d, before any PCA)
    P_full = load_panphon_embeddings(words)
    keep = ~np.any(np.isnan(P_full), axis=1)
    if (~keep).any():
        print(f"  dropping {int((~keep).sum())} trials with missing panphon", flush=True)
    S = S[keep]; P_full = P_full[keep]; meta = meta.loc[keep].reset_index(drop=True)

    n_bins = reg.n_bins
    bins = bins if bins is not None else list(range(reg.n_bins_history, n_bins))
    rows = []
    per_bin_proj = {}
    for b in bins:
        if verbose: print(f"  bin {b}/{n_bins-1} ", end="", flush=True)
        X = reg.X_to_use[b][keep]
        try:
            cv = cross_validated_r2(X, S, P_full, d_common=d_common,
                                     ridge_alpha=ridge_alpha, n_splits=n_splits)
            full_dec = decompose_neural_at_bin(X, S, P_full,
                                                d_common=d_common,
                                                ridge_alpha=ridge_alpha)
            per_bin_proj[b] = {"T_sem": full_dec["T_sem"],
                               "T_phon": full_dec["T_phon"],
                               "T_shared": full_dec["T_shared"]}
            row = {"patient": patient, "bin_index": b, **cv}
            if n_perm > 0:
                null = permutation_null(X, S, P_full, n_perm=n_perm,
                                         d_common=d_common, ridge_alpha=ridge_alpha,
                                         n_splits=n_splits, seed=b)
                row.update(null)
            rows.append(row)
            if verbose:
                print(f"R²(S|sem)={cv['R2_S_on_sem']:.3f}  "
                      f"R²(P|phon)={cv['R2_P_on_phon']:.3f}  "
                      f"R²(S|phon)={cv['R2_S_on_phon']:+.3f}  "
                      f"R²(P|sem)={cv['R2_P_on_sem']:+.3f}", flush=True)
        except Exception as e:
            print(f"  ERR at bin {b}: {type(e).__name__}: {e}", flush=True)
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_bin_metrics.csv", index=False)

    if save_figs and len(df) > 1:
        bin_size_ms = int(d.get("bin_size_ms", 100))
        plot_traces(df, out_dir / "figures" / "dyso_traces.png",
                    bin_size_ms=bin_size_ms)
        # Peak bin = argmax of mean(R²_S_on_sem, R²_P_on_phon) — joint quality
        joint = (df["R2_S_on_sem"].values + df["R2_P_on_phon"].values) / 2
        peak_b = int(df["bin_index"].values[np.argmax(joint)])
        peak = per_bin_proj.get(peak_b)
        if peak is not None:
            plot_3d_scatter(peak["T_sem"], peak["T_phon"], peak["T_shared"],
                             meta, out_dir / "figures" / "scatter_3d.png")
        # Quiver across the upper-half of bins
        plot_word_trajectory_quiver(per_bin_proj, meta,
                                     bin_range=(bins[0], bins[-1]),
                                     out_path=out_dir / "figures" / "quiver.png",
                                     bin_size_ms=bin_size_ms)
        # Save the peak-bin projections for downstream interactive viewing
        with open(out_dir / "projections_peak.pkl", "wb") as f:
            pk.dump({"peak_bin": peak_b, "meta": meta, **peak}, f)
        print(f"  peak bin = {peak_b}", flush=True)

    return df


# ── 8. CLI ───────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patients", nargs="*", default=None)
    p.add_argument("--patient",  default=None,
                   help="Single patient shorthand for --patients <PAT>")
    p.add_argument("--task", choices=["picture_naming", "auditory_naming"],
                   default="picture_naming")
    p.add_argument("--run", default=None,
                   help="Override run folder (default: matched to --task)")
    p.add_argument("--bin", type=int, default=None,
                   help="Single bin index (smoke test)")
    p.add_argument("--smoke", action="store_true",
                   help="Quick test: 1 patient × 1 bin × no permutation")
    p.add_argument("--d-common", type=int, default=D_COMMON_DEFAULT)
    p.add_argument("--ridge-alpha", type=float, default=RIDGE_ALPHA)
    p.add_argument("--n-perm", type=int, default=0)
    p.add_argument("--n-splits", type=int, default=N_KFOLD)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--no-figs", action="store_true")
    args = p.parse_args()

    run = args.run or (DEFAULT_AUD_RUN if args.task == "auditory_naming" else DEFAULT_PIC_RUN)
    patients = args.patients or ([args.patient] if args.patient else SHARED_PATIENTS)
    if args.smoke:
        patients = patients[:1]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for pat in patients:
        bins = [args.bin] if args.bin is not None else None
        try:
            df = analyze_patient(pat, run, out_dir, bins=bins,
                                  d_common=args.d_common,
                                  ridge_alpha=args.ridge_alpha,
                                  n_perm=(0 if args.smoke else args.n_perm),
                                  n_splits=args.n_splits,
                                  save_figs=not args.no_figs)
            all_rows.append(df)
        except Exception as e:
            print(f"  ERROR for {pat}: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
        finally:
            gc.collect()

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(out_dir / "cross_patient_metrics.csv", index=False)
        print(f"\nSaved combined: {out_dir / 'cross_patient_metrics.csv'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
