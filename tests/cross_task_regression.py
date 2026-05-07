# -*- coding: utf-8 -*-
"""
tests/cross_task_regression.py
==============================
Cross-task semantic regression: compare picture-naming vs auditory-naming
PLS subspaces at each task's peak loose-semantic-category bin, per patient.

Pipeline (per patient):
  (1) Find peak `category_balanced_acc` bin in each task (uses per_time_scores.csv).
  (2) Train fresh kernel-PLS at the peak bin on ALL trials per task.
  (3) Compare projection geometry:
        - alignment index (mean cos^2 of principal angles)
        - principal angles between word-averaged PLS subspaces
        - CCA: canonical correlations + per-component direction vectors
        - quiver visualization of the 10 PLS axes mapped onto the first 2
          CCA dims, before vs after CCA alignment
  (4) Co-project trials into a common 2D space (CCA) and visualize before/after.
  (5) Cross-task decoding: train task-A pipeline at task-A peak, evaluate on
      task-B trials at task-B peak. Compare to within-task accuracy at peak.

Outputs (under semantic_regression_figures/cross_task_regression/<patient>/):
  - peaks.csv, alignment_metrics.csv, cca_canonical_correlations.csv,
    cross_task_accuracy.csv, projection_2d_trials.csv
  - figures: quiver_align.png, scatter_2d.png, principal_angles.png,
    cross_task_bars.png, peak_traces.png

Usage:
    python -m main.tests.cross_task_regression                    # all patients
    python -m main.tests.cross_task_regression --patient AA       # one patient
    python -m main.tests.cross_task_regression --patient AA --no-figs

The folder pair is fixed (kernel_pls April-8 picture / May-6 auditory). Override
via --pic-run / --aud-run if needed.
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

from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from scipy.linalg import subspace_angles

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # .../Neuroscience of speech and language
SEM_REG_DIR = PROJECT_ROOT / "semantic_regression"
OUT_ROOT = PROJECT_ROOT / "semantic_regression_figures" / "cross_task_regression"

PIC_RUN_DEFAULT = "2026-04-08_17-05-14_kernel_pls_cosine_50ep"
AUD_RUN_DEFAULT = "2026-05-06_19-12-48_auditory_naming_warp-linear_kernel_pls_cosine_50ep"

SHARED_PATIENTS = ["AA", "AZ", "DR", "LH", "RB", "WBH"]
SHARED_EMBEDDINGS = ["GloVe", "FastText", "Word2Vec", "ConceptNet"]

PEAK_METRIC = "category_balanced_acc"   # column in per_time_scores.csv ("loose semantic category")
N_PLS_COMPONENTS = 10
NYSTROEM_N_COMPONENTS = 100             # auto-clamped to n_samples-1 by sklearn
PEAK_EMBEDDING = "GloVe"                # primary embedding for alignment / cross-decoding

CATEGORY_COLORS = {
    "animal": "#1f77b4", "body part": "#ff7f0e", "food/fruit": "#2ca02c",
    "nature": "#d62728", "object/tool": "#9467bd", "vehicle": "#8c564b",
    "clothing": "#e377c2", "tool": "#7f7f7f", "other": "#bcbd22",
}


# ── 1. Loading ───────────────────────────────────────────────────────────
def load_per_time_scores(run_folder: str, patient: str) -> pd.DataFrame:
    """Read per-time-bin retrieval scores; lightweight CSV — safe to load anywhere."""
    csv_path = SEM_REG_DIR / run_folder / patient / "per_time_scores.csv"
    return pd.read_csv(csv_path)


def find_peak_bin(scores_df: pd.DataFrame, embedding: str = PEAK_EMBEDDING,
                  metric: str = PEAK_METRIC) -> tuple[int, float]:
    """Return (peak_bin_index, peak_value) for a given embedding and metric."""
    sub = scores_df[scores_df["embedding"] == embedding]
    if sub.empty:
        raise ValueError(f"No rows for embedding={embedding!r}")
    row = sub.loc[sub[metric].idxmax()]
    return int(row["bin_index"]), float(row[metric])


def load_results_pkl(run_folder: str, patient: str) -> dict:
    """Unpickle the heavy results pkl. Requires the project's `models` package on PYTHONPATH.

    Memory note: pkl files are 100MB-2.6GB. Run on a machine with enough RAM
    (16GB+ recommended). If you only need peak bins (not the trained models),
    use `load_per_time_scores` instead.
    """
    pkl_path = SEM_REG_DIR / run_folder / patient / "semantic_regression_results.pkl"
    with open(pkl_path, "rb") as f:
        return pk.load(f)


def get_neural_at_bin(reg, bin_idx: int) -> np.ndarray:
    """Return X[bin] of shape (n_trials, n_features). Uses pre-computed history-concat."""
    return reg.X_to_use[bin_idx]


def build_X_at_bin_with_channel_subset(reg, bin_idx: int,
                                        channel_subset_idx: np.ndarray) -> np.ndarray:
    """Reconstruct X for a given bin using only a channel subset.

    `reg.data` has shape (n_trials, n_bins, n_channels). For bin >= n_bins_history,
    the original X_to_use concatenated bins [bin - history + 1, bin]; we replicate
    that with only the requested channels.
    """
    n_history = reg.n_bins_history
    data = reg.data[:, :, channel_subset_idx]   # (n_trials, n_bins, n_chan_subset)
    if bin_idx < n_history - 1:
        # Pad earlier bins with zeros (mirrors the sklearn-style early-bin behavior)
        # Original code in the project usually only fits at bins >= history-1; cross-task
        # decoding at peak bins should be safely past that point.
        return data[:, bin_idx, :]
    start = bin_idx - n_history + 1
    end = bin_idx + 1
    return data[:, start:end, :].reshape(data.shape[0], -1)


def common_channels(reg_A, reg_B) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (chan_idx_A, chan_idx_B, common_names) for shared channel names."""
    # `clean_channel_names` lives on the parent dict, but the reg also has channel info via data shape.
    # We rely on the parent dict providing it; here we accept the names lists explicitly.
    raise NotImplementedError("Use _common_channels_from_names instead.")


def _common_channels_from_names(names_A: np.ndarray, names_B: np.ndarray
                                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indices into A's and B's channel arrays for the intersection (same order)."""
    nA = np.asarray([str(n) for n in names_A])
    nB = np.asarray([str(n) for n in names_B])
    common = np.array(sorted(set(nA.tolist()) & set(nB.tolist())))
    idx_A = np.array([np.where(nA == c)[0][0] for c in common])
    idx_B = np.array([np.where(nB == c)[0][0] for c in common])
    return idx_A, idx_B, common


def get_trial_metadata(reg) -> pd.DataFrame:
    """Per-trial metadata: word label and category."""
    words = np.asarray(reg.labels)
    word_to_idx = reg.word_to_index
    word_idx_to_cat_idx = reg.word_index_to_category_index
    idx_to_cat = reg.index_to_category
    cats = []
    for w in words:
        wi = word_to_idx[str(w)] if str(w) in word_to_idx else word_to_idx[w]
        cats.append(str(idx_to_cat[word_idx_to_cat_idx[wi]]))
    return pd.DataFrame({"word": words.astype(str), "category": cats})


# ── 2. PLS fitting on full data ──────────────────────────────────────────
def fit_full_pls(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Pipeline:
    """Fresh kernel-PLS on ALL trials (no holdout)."""
    n_nys = min(NYSTROEM_N_COMPONENTS, max(2, X.shape[0] - 1))
    pipe = Pipeline([
        ("nystroem", Nystroem(kernel="rbf", n_components=n_nys,
                              random_state=random_state)),
        ("pls", PLSRegression(n_components=N_PLS_COMPONENTS, scale=False)),
    ])
    pipe.fit(X, y)
    return pipe


def transform_pls(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    """Project neural matrix into the PLS 10D score space."""
    return pipe.transform(X)


# ── 3. Alignment, principal angles, CCA ──────────────────────────────────
def per_word_average(T: np.ndarray, words: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average T per unique word. Returns (n_unique_words, K), unique_words array."""
    unique_words = np.array(sorted(set(words.tolist())))
    out = np.zeros((len(unique_words), T.shape[1]))
    for i, w in enumerate(unique_words):
        out[i] = T[words == w].mean(axis=0)
    return out, unique_words


def shared_words(words_A: np.ndarray, words_B: np.ndarray) -> np.ndarray:
    """Sorted intersection of unique words present in both tasks."""
    return np.array(sorted(set(words_A.tolist()) & set(words_B.tolist())))


def matched_word_average(T_A: np.ndarray, words_A: np.ndarray,
                         T_B: np.ndarray, words_B: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build word-averaged matrices for the SHARED words across tasks.

    Returns (M_A, M_B, shared_word_array) each (n_shared, K).
    """
    sw = shared_words(words_A, words_B)
    M_A = np.zeros((len(sw), T_A.shape[1]))
    M_B = np.zeros((len(sw), T_B.shape[1]))
    for i, w in enumerate(sw):
        M_A[i] = T_A[words_A == w].mean(axis=0)
        M_B[i] = T_B[words_B == w].mean(axis=0)
    return M_A, M_B, sw


def alignment_metrics(M_A: np.ndarray, M_B: np.ndarray) -> dict:
    """Principal angles + alignment index between column spaces of M_A and M_B.

    Both are (n_words, K). Each represents a K-dim subspace in word-space.
    """
    # Orthonormalize columns first (subspace_angles handles internally too)
    angles_rad = subspace_angles(M_A, M_B)
    cos2 = np.cos(angles_rad) ** 2
    align_idx = float(np.mean(cos2))                       # 1 = identical, 0 = orthogonal
    grassmann_dist = float(np.linalg.norm(angles_rad))     # smaller = closer
    return {
        "principal_angles_deg": np.rad2deg(angles_rad),
        "alignment_index": align_idx,
        "grassmann_distance": grassmann_dist,
    }


def cca_align(M_A: np.ndarray, M_B: np.ndarray, n_components: int | None = None
              ) -> tuple[CCA, np.ndarray, np.ndarray, np.ndarray]:
    """Train CCA on word-averaged matrices. Returns (cca, A_canonical, B_canonical, corr)."""
    n_max = min(M_A.shape[1], M_B.shape[1], M_A.shape[0] - 1)
    n = n_components or n_max
    cca = CCA(n_components=n, max_iter=2000)
    A_c, B_c = cca.fit_transform(M_A, M_B)
    corr = np.array([np.corrcoef(A_c[:, i], B_c[:, i])[0, 1] for i in range(n)])
    return cca, A_c, B_c, corr


def _cca_x_mean(cca: CCA) -> np.ndarray:
    """Return CCA's x-side mean (handles sklearn version differences)."""
    return getattr(cca, "x_mean_", None) or getattr(cca, "_x_mean")


def _cca_y_mean(cca: CCA) -> np.ndarray:
    return getattr(cca, "y_mean_", None) or getattr(cca, "_y_mean")


def project_trials_to_cca_2d(cca: CCA,
                             T_A_trials: np.ndarray,
                             T_B_trials: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Project trial-level PLS scores to the first 2 CCA components."""
    A2 = cca.transform(T_A_trials)[:, :2]
    B2 = (T_B_trials - _cca_y_mean(cca)) @ cca.y_rotations_[:, :2]
    return A2, B2


# ── 4. Cross-task decoding ───────────────────────────────────────────────
def category_retrieval(y_pred: np.ndarray, y_true_db: np.ndarray, db_words: np.ndarray,
                       trial_words: np.ndarray, trial_categories: np.ndarray,
                       word_to_category: dict) -> dict:
    """1-NN cosine retrieval: for each predicted embedding, find nearest
    word in the database, classify category. Returns balanced acc + f1."""
    # Cosine similarity
    p = y_pred / (np.linalg.norm(y_pred, axis=1, keepdims=True) + 1e-12)
    d = y_true_db / (np.linalg.norm(y_true_db, axis=1, keepdims=True) + 1e-12)
    sim = p @ d.T
    nn_idx = sim.argmax(axis=1)
    pred_words = db_words[nn_idx]
    pred_cats = np.array([word_to_category[w] for w in pred_words])
    bal = balanced_accuracy_score(trial_categories, pred_cats)
    macro_f1 = f1_score(trial_categories, pred_cats, average="macro")
    return {
        "category_balanced_acc": float(bal),
        "category_f1": float(macro_f1),
        "pred_words": pred_words,
        "pred_categories": pred_cats,
    }


def predict_y(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    """Apply pipeline up to the PLS predict step → returns embedding-space prediction."""
    return pipe.predict(X)


def _build_word_category_map(words: np.ndarray, idx_to_cat: np.ndarray,
                              word_idx_to_cat_idx: np.ndarray,
                              word_to_index: dict) -> dict:
    """Map each unique word -> category name for words present in word_to_index."""
    out = {}
    for w in np.unique(words):
        ws = str(w)
        if ws not in word_to_index:
            continue
        wi = word_to_index[ws]
        out[ws] = str(idx_to_cat[word_idx_to_cat_idx[wi]])
    return out


def evaluate_pipe_on_task(pipe, X, words, y_db,
                           idx_to_cat, word_idx_to_cat_idx, word_to_index) -> dict:
    """Apply pipe to X (from one task), retrieve categories using THAT task's
    word/category database. Used for both within- and cross-task evaluation
    (caller chooses which task's mappings to pass)."""
    y_pred = predict_y(pipe, X)
    db_embeds, db_words = _db_from_trials(y_db, words)
    word_to_cat = _build_word_category_map(words, idx_to_cat,
                                            word_idx_to_cat_idx, word_to_index)
    keep = np.array([str(w) in word_to_cat for w in db_words])
    db_words = db_words[keep]
    db_embeds = db_embeds[keep]
    trial_cats = np.array([word_to_cat.get(str(w), "?") for w in words])
    return category_retrieval(y_pred, db_embeds, db_words,
                               words, trial_cats, word_to_cat)


def cross_task_decode(pipe_train, X_train_peak, y_train, words_train,
                      pipe_eval_unused,
                      X_test_peak, y_test, words_test,
                      idx_to_cat_test, word_idx_to_cat_idx_test, word_to_index_test
                      ) -> dict:
    """Cross-task decoding: pipe_train applied to test trials at test peak bin.

    Evaluation uses the TEST-side database (test trial words + their true embeddings).
    Returns the cross-task result dict.
    """
    return evaluate_pipe_on_task(
        pipe_train, X_test_peak, words_test, y_test,
        idx_to_cat_test, word_idx_to_cat_idx_test, word_to_index_test,
    )


def _db_from_trials(y: np.ndarray, words: np.ndarray):
    """Helper: build DB embeddings + DB-words from per-trial arrays."""
    unique_words, first_idx = np.unique(words, return_index=True)
    return y[first_idx], unique_words


# ── 5. Plotting ──────────────────────────────────────────────────────────
def plot_peak_traces(pic_scores: pd.DataFrame, aud_scores: pd.DataFrame,
                     pic_peak: int, aud_peak: int, out_path: Path,
                     embedding: str = PEAK_EMBEDDING) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    sub_p = pic_scores[pic_scores["embedding"] == embedding]
    sub_a = aud_scores[aud_scores["embedding"] == embedding]
    bin_size_ms = 100
    t_p = sub_p["bin_index"].values * bin_size_ms / 1000.0
    t_a = sub_a["bin_index"].values * bin_size_ms / 1000.0
    ax.plot(t_p, sub_p[PEAK_METRIC].values, label="picture", lw=1.7, color="#1f77b4")
    ax.plot(t_a, sub_a[PEAK_METRIC].values, label="auditory", lw=1.7, color="#d62728")
    ax.axvline(pic_peak * bin_size_ms / 1000.0, ls="--", color="#1f77b4", alpha=0.5)
    ax.axvline(aud_peak * bin_size_ms / 1000.0, ls="--", color="#d62728", alpha=0.5)
    ax.set_xlabel("Time relative to onset (s)")
    ax.set_ylabel(f"{PEAK_METRIC}")
    ax.set_title(f"Loose-category retrieval traces ({embedding}); peak bins marked")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_principal_angles(angles_deg: np.ndarray, alignment_idx: float,
                          out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.bar(np.arange(1, len(angles_deg) + 1), angles_deg,
           color=["#2ca02c" if a < 30 else "#ff7f0e" if a < 60 else "#d62728"
                  for a in angles_deg])
    ax.axhline(45, ls=":", color="grey", lw=1)
    ax.set_xlabel("PLS dimension")
    ax.set_ylabel("Principal angle (deg)")
    ax.set_title(f"Principal angles between PLS subspaces  |  align idx = {alignment_idx:.2f}")
    ax.set_ylim(0, 95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_quiver_align(M_A: np.ndarray, M_B: np.ndarray,
                      cca, A_c: np.ndarray, B_c: np.ndarray,
                      out_path: Path) -> None:
    """Visualize how each task's 10 PLS axes map onto the first 2 CCA dimensions.

    Each arrow = one PLS dim k of task X projected through CCA to a 2D point.
    """
    # PLS axes in CCA space: take canonical projections of identity-shaped basis
    # Build axis-as-row vectors of length K, project via CCA-fitted matrix
    K = M_A.shape[1]
    eye_A = np.eye(K)
    eye_B = np.eye(K)
    A_axes = (eye_A - _cca_x_mean(cca)) @ cca.x_rotations_[:, :2]
    B_axes = (eye_B - _cca_y_mean(cca)) @ cca.y_rotations_[:, :2]

    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    # Plot CCA-aligned word centroids as dots
    ax.scatter(A_c[:, 0], A_c[:, 1], s=18, color="#1f77b4", alpha=0.5, label="picture words")
    ax.scatter(B_c[:, 0], B_c[:, 1], s=18, color="#d62728", alpha=0.5, label="auditory words")
    # Quivers
    origin = np.zeros(K)
    for k in range(K):
        ax.annotate("", xy=A_axes[k], xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.0, alpha=0.9))
        ax.annotate("", xy=B_axes[k], xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0, alpha=0.9))
        ax.text(*A_axes[k] * 1.05, f"P{k+1}", fontsize=7, color="#1f77b4")
        ax.text(*B_axes[k] * 1.05, f"A{k+1}", fontsize=7, color="#d62728")
    ax.axhline(0, ls=":", color="grey", lw=0.6)
    ax.axvline(0, ls=":", color="grey", lw=0.6)
    ax.set_xlabel("CCA dim 1")
    ax.set_ylabel("CCA dim 2")
    ax.set_title("PLS axes mapped through CCA (P=picture, A=auditory)")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_2d_trials(T_A_2d_pre: np.ndarray, T_B_2d_pre: np.ndarray,
                   T_A_2d_post: np.ndarray, T_B_2d_post: np.ndarray,
                   meta_A: pd.DataFrame, meta_B: pd.DataFrame,
                   out_path: Path) -> None:
    """4-panel: pre-CCA pic, post-CCA pic, pre-CCA aud, post-CCA aud (color by category)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    panels = [
        (axes[0, 0], T_A_2d_pre, meta_A, "Picture · pre-CCA (PCA of PLS scores)"),
        (axes[0, 1], T_A_2d_post, meta_A, "Picture · post-CCA"),
        (axes[1, 0], T_B_2d_pre, meta_B, "Auditory · pre-CCA (PCA of PLS scores)"),
        (axes[1, 1], T_B_2d_post, meta_B, "Auditory · post-CCA"),
    ]
    cats_seen = sorted(set(meta_A["category"].tolist()) | set(meta_B["category"].tolist()))
    for ax, T2, meta, title in panels:
        for cat in cats_seen:
            mask = (meta["category"].values == cat)
            if mask.sum() == 0: continue
            color = CATEGORY_COLORS.get(cat, "#999999")
            ax.scatter(T2[mask, 0], T2[mask, 1], s=42, color=color, alpha=0.75,
                       edgecolors="white", linewidths=0.4, label=cat)
            for j in np.where(mask)[0]:
                ax.text(T2[j, 0], T2[j, 1], str(meta["word"].iloc[j]),
                        fontsize=6, alpha=0.7, color="black")
        ax.set_title(title, fontsize=10)
        ax.axhline(0, ls=":", color="grey", lw=0.5)
        ax.axvline(0, ls=":", color="grey", lw=0.5)
    handles = [Line2D([0], [0], marker='o', linestyle='', color=CATEGORY_COLORS.get(c, "#999"), label=c)
               for c in cats_seen]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_cross_task_bars(within_pic: float, within_aud: float,
                         cross_pic_to_aud: float, cross_aud_to_pic: float,
                         within_pic_holdout: float | None, within_aud_holdout: float | None,
                         out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    labels, values, colors = [], [], []
    if within_pic_holdout is not None:
        labels.append("pic→pic\n(holdout)"); values.append(within_pic_holdout); colors.append("#9ecae1")
    labels.append("pic→pic\n(full fit)");  values.append(within_pic);  colors.append("#1f77b4")
    labels.append("pic→aud\n(cross)");     values.append(cross_pic_to_aud); colors.append("#7f7f7f")
    if within_aud_holdout is not None:
        labels.append("aud→aud\n(holdout)"); values.append(within_aud_holdout); colors.append("#fdae6b")
    labels.append("aud→aud\n(full fit)");  values.append(within_aud);  colors.append("#d62728")
    labels.append("aud→pic\n(cross)");     values.append(cross_aud_to_pic); colors.append("#7f7f7f")

    xs = np.arange(len(labels))
    ax.bar(xs, values, color=colors)
    for i, v in enumerate(values):
        ax.text(xs[i], v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, max(0.6, max(values) * 1.2))
    ax.set_ylabel(PEAK_METRIC)
    ax.set_title("Within- vs cross-task category retrieval (peak bin)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ── 6. Per-patient pipeline ──────────────────────────────────────────────
def analyze_patient(patient: str, pic_run: str, aud_run: str,
                    embedding: str = PEAK_EMBEDDING,
                    out_dir: Path | None = None,
                    save_figs: bool = True) -> dict:
    """Run the full per-patient analysis. Returns a dict of metrics for cross-patient summary."""
    print(f"\n=== {patient} : {pic_run}  vs  {aud_run} ===", flush=True)
    out_dir = out_dir or (OUT_ROOT / patient)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Find peaks (cheap, from CSV)
    pic_scores = load_per_time_scores(pic_run, patient)
    aud_scores = load_per_time_scores(aud_run, patient)
    pic_peak, pic_peak_val = find_peak_bin(pic_scores, embedding=embedding)
    aud_peak, aud_peak_val = find_peak_bin(aud_scores, embedding=embedding)
    pic_peak_chance = float(pic_scores[(pic_scores["embedding"] == embedding) &
                                        (pic_scores["bin_index"] == pic_peak)]["chance_mean"].iloc[0])
    aud_peak_chance = float(aud_scores[(aud_scores["embedding"] == embedding) &
                                        (aud_scores["bin_index"] == aud_peak)]["chance_mean"].iloc[0])
    print(f"  picture peak bin = {pic_peak} ({pic_peak_val:.3f})  |  "
          f"auditory peak bin = {aud_peak} ({aud_peak_val:.3f})", flush=True)

    if save_figs:
        plot_peak_traces(pic_scores, aud_scores, pic_peak, aud_peak,
                         out_dir / "peak_traces.png", embedding=embedding)

    # Step 2: Load pkl, get neural data + train fresh PLS
    pic_d = load_results_pkl(pic_run, patient)
    pic_reg = pic_d["regressors"][embedding]
    pic_chan_names = np.asarray(pic_d.get("clean_channel_names", [])).astype(str)
    pic_meta = get_trial_metadata(pic_reg)
    X_pic = get_neural_at_bin(pic_reg, pic_peak)
    y_pic = pic_reg.y
    words_pic = np.asarray(pic_reg.labels).astype(str)
    pipe_pic = fit_full_pls(X_pic, y_pic, random_state=42)
    T_pic = transform_pls(pipe_pic, X_pic)
    word_to_index_pic = {str(k): v for k, v in pic_reg.word_to_index.items()}
    word_idx_to_cat_idx_pic = pic_reg.word_index_to_category_index
    idx_to_cat_pic = pic_reg.index_to_category
    del pic_d; gc.collect()

    aud_d = load_results_pkl(aud_run, patient)
    aud_reg = aud_d["regressors"][embedding]
    aud_chan_names = np.asarray(aud_d.get("clean_channel_names", [])).astype(str)
    aud_meta = get_trial_metadata(aud_reg)
    X_aud = get_neural_at_bin(aud_reg, aud_peak)
    y_aud = aud_reg.y
    words_aud = np.asarray(aud_reg.labels).astype(str)
    pipe_aud = fit_full_pls(X_aud, y_aud, random_state=42)
    T_aud = transform_pls(pipe_aud, X_aud)
    word_to_index_aud = {str(k): v for k, v in aud_reg.word_to_index.items()}
    word_idx_to_cat_idx_aud = aud_reg.word_index_to_category_index
    idx_to_cat_aud = aud_reg.index_to_category
    del aud_d; gc.collect()

    # Step 3: alignment + CCA on word-averaged matched matrices
    M_pic, M_aud, sw = matched_word_average(T_pic, words_pic, T_aud, words_aud)
    print(f"  shared words: {len(sw)}", flush=True)
    if len(sw) < N_PLS_COMPONENTS + 1:
        print(f"  WARNING: only {len(sw)} shared words; CCA will use min(K, n-1) components", flush=True)
    align = alignment_metrics(M_pic, M_aud)
    n_cca = min(N_PLS_COMPONENTS, len(sw) - 1)
    cca, A_c, B_c, canon_corr = cca_align(M_pic, M_aud, n_components=n_cca)
    print(f"  alignment_index={align['alignment_index']:.3f}  "
          f"first_canon_corr={canon_corr[0]:.3f}", flush=True)

    if save_figs:
        plot_principal_angles(align["principal_angles_deg"],
                              align["alignment_index"],
                              out_dir / "principal_angles.png")
        # Quiver: re-fit a 2-component CCA so x_rotations_/y_rotations_ are 2D
        cca2 = CCA(n_components=2, max_iter=2000)
        Ac2, Bc2 = cca2.fit_transform(M_pic, M_aud)
        plot_quiver_align(M_pic, M_aud, cca2, Ac2, Bc2,
                          out_dir / "quiver_align.png")

    # Step 4: Co-project trials to 2D pre/post CCA
    # PRE-CCA: PCA of trial PLS scores per task (to 2D)
    from sklearn.decomposition import PCA
    pca_pic = PCA(n_components=2).fit(T_pic)
    pca_aud = PCA(n_components=2).fit(T_aud)
    T_pic_2d_pre = pca_pic.transform(T_pic)
    T_aud_2d_pre = pca_aud.transform(T_aud)
    # POST-CCA: project trial scores using fitted 2D CCA
    T_pic_2d_post = (T_pic - _cca_x_mean(cca2)) @ cca2.x_rotations_[:, :2]
    T_aud_2d_post = (T_aud - _cca_y_mean(cca2)) @ cca2.y_rotations_[:, :2]
    if save_figs:
        plot_2d_trials(T_pic_2d_pre, T_aud_2d_pre,
                       T_pic_2d_post, T_aud_2d_post,
                       pic_meta, aud_meta,
                       out_dir / "scatter_2d.png")

    # Step 5: Cross-task decoding at each task's own peak.
    # Picture and auditory may have different channel counts (different rejection
    # masks). Restrict to common channels and train a separate kernel-PLS on those.
    if len(pic_chan_names) > 0 and len(aud_chan_names) > 0:
        idx_pic, idx_aud, common = _common_channels_from_names(pic_chan_names, aud_chan_names)
        print(f"  common channels: {len(common)} / pic={len(pic_chan_names)} / aud={len(aud_chan_names)}", flush=True)
    else:
        # Fallback: assume identical channel order, take min
        n = min(pic_reg.data.shape[2], aud_reg.data.shape[2])
        idx_pic = np.arange(n); idx_aud = np.arange(n); common = np.arange(n)
        print(f"  channel names unavailable; using first {n} channels (fallback)", flush=True)

    X_pic_common = build_X_at_bin_with_channel_subset(pic_reg, pic_peak, idx_pic)
    X_aud_common = build_X_at_bin_with_channel_subset(aud_reg, aud_peak, idx_aud)
    pipe_pic_common = fit_full_pls(X_pic_common, y_pic, random_state=42)
    pipe_aud_common = fit_full_pls(X_aud_common, y_aud, random_state=42)

    within_pic = evaluate_pipe_on_task(
        pipe_pic_common, X_pic_common, words_pic, y_pic,
        idx_to_cat_pic, word_idx_to_cat_idx_pic, word_to_index_pic,
    )
    within_aud = evaluate_pipe_on_task(
        pipe_aud_common, X_aud_common, words_aud, y_aud,
        idx_to_cat_aud, word_idx_to_cat_idx_aud, word_to_index_aud,
    )
    cross_pic_to_aud = cross_task_decode(
        pipe_pic_common, X_pic_common, y_pic, words_pic,
        pipe_aud_common, X_aud_common, y_aud, words_aud,
        idx_to_cat_aud, word_idx_to_cat_idx_aud, word_to_index_aud,
    )
    cross_aud_to_pic = cross_task_decode(
        pipe_aud_common, X_aud_common, y_aud, words_aud,
        pipe_pic_common, X_pic_common, y_pic, words_pic,
        idx_to_cat_pic, word_idx_to_cat_idx_pic, word_to_index_pic,
    )

    # Holdout reference values from per_time_scores.csv (already-computed average over 50 epochs)
    within_pic_holdout = pic_peak_val
    within_aud_holdout = aud_peak_val

    if save_figs:
        plot_cross_task_bars(
            within_pic["category_balanced_acc"], within_aud["category_balanced_acc"],
            cross_pic_to_aud["category_balanced_acc"], cross_aud_to_pic["category_balanced_acc"],
            within_pic_holdout, within_aud_holdout,
            out_dir / "cross_task_bars.png",
        )

    # ── Save CSVs ─────────────────────────────────────────────────────
    pd.DataFrame([{
        "patient": patient, "embedding": embedding,
        "pic_peak_bin": pic_peak, "pic_peak_acc": pic_peak_val, "pic_peak_chance": pic_peak_chance,
        "aud_peak_bin": aud_peak, "aud_peak_acc": aud_peak_val, "aud_peak_chance": aud_peak_chance,
        "n_shared_words": len(sw),
    }]).to_csv(out_dir / "peaks.csv", index=False)

    pd.DataFrame({
        "dim": np.arange(1, len(align["principal_angles_deg"]) + 1),
        "principal_angle_deg": align["principal_angles_deg"],
    }).to_csv(out_dir / "principal_angles.csv", index=False)

    pd.DataFrame([{
        "patient": patient, "embedding": embedding,
        "alignment_index": align["alignment_index"],
        "grassmann_distance": align["grassmann_distance"],
        "first_canon_corr": float(canon_corr[0]),
        "mean_canon_corr": float(canon_corr.mean()),
    }]).to_csv(out_dir / "alignment_metrics.csv", index=False)

    pd.DataFrame({"dim": np.arange(1, len(canon_corr) + 1),
                  "canon_corr": canon_corr}).to_csv(out_dir / "cca_canonical_correlations.csv", index=False)

    pd.DataFrame([{
        "patient": patient, "embedding": embedding,
        "within_pic_full_fit": within_pic["category_balanced_acc"],
        "within_aud_full_fit": within_aud["category_balanced_acc"],
        "within_pic_holdout": within_pic_holdout,
        "within_aud_holdout": within_aud_holdout,
        "cross_pic_to_aud": cross_pic_to_aud["category_balanced_acc"],
        "cross_aud_to_pic": cross_aud_to_pic["category_balanced_acc"],
        "delta_pic": cross_aud_to_pic["category_balanced_acc"] - within_pic_holdout,
        "delta_aud": cross_pic_to_aud["category_balanced_acc"] - within_aud_holdout,
    }]).to_csv(out_dir / "cross_task_accuracy.csv", index=False)

    # 2D trial projections (for downstream report use)

    df_pic_2d = pd.DataFrame({
        "task": "picture", "trial_idx": np.arange(len(T_pic)),
        "word": pic_meta["word"], "category": pic_meta["category"],
        "pre_x": T_pic_2d_pre[:, 0], "pre_y": T_pic_2d_pre[:, 1],
        "post_x": T_pic_2d_post[:, 0], "post_y": T_pic_2d_post[:, 1],
    })
    df_aud_2d = pd.DataFrame({
        "task": "auditory", "trial_idx": np.arange(len(T_aud)),
        "word": aud_meta["word"], "category": aud_meta["category"],
        "pre_x": T_aud_2d_pre[:, 0], "pre_y": T_aud_2d_pre[:, 1],
        "post_x": T_aud_2d_post[:, 0], "post_y": T_aud_2d_post[:, 1],
    })
    pd.concat([df_pic_2d, df_aud_2d], ignore_index=True).to_csv(
        out_dir / "projection_2d_trials.csv", index=False)

    return {
        "patient": patient, "embedding": embedding,
        "pic_peak_bin": pic_peak, "pic_peak_acc": pic_peak_val,
        "aud_peak_bin": aud_peak, "aud_peak_acc": aud_peak_val,
        "n_shared_words": len(sw),
        "alignment_index": align["alignment_index"],
        "grassmann_distance": align["grassmann_distance"],
        "first_canon_corr": float(canon_corr[0]),
        "mean_canon_corr": float(canon_corr.mean()),
        "within_pic_holdout": within_pic_holdout,
        "within_aud_holdout": within_aud_holdout,
        "within_pic_full_fit": within_pic["category_balanced_acc"],
        "within_aud_full_fit": within_aud["category_balanced_acc"],
        "cross_pic_to_aud": cross_pic_to_aud["category_balanced_acc"],
        "cross_aud_to_pic": cross_aud_to_pic["category_balanced_acc"],
    }


# ── 7. Driver ────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patient", default=None)
    p.add_argument("--pic-run", default=PIC_RUN_DEFAULT)
    p.add_argument("--aud-run", default=AUD_RUN_DEFAULT)
    p.add_argument("--embedding", default=PEAK_EMBEDDING)
    p.add_argument("--no-figs", action="store_true")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    out_root = Path(args.out_dir) if args.out_dir else OUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)
    patients = [args.patient] if args.patient else SHARED_PATIENTS
    rows = []
    for pat in patients:
        try:
            row = analyze_patient(pat, args.pic_run, args.aud_run,
                                  embedding=args.embedding,
                                  out_dir=out_root / pat,
                                  save_figs=not args.no_figs)
            rows.append(row)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR for {pat}: {type(e).__name__}: {e}", flush=True)
        finally:
            gc.collect()
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_root / "cross_patient_summary.csv", index=False)
        print(f"\nWrote summary: {out_root / 'cross_patient_summary.csv'}", flush=True)
        print(df.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
