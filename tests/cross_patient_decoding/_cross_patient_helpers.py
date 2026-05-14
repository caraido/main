# -*- coding: utf-8 -*-
"""
tests/cross_patient_decoding/_cross_patient_helpers.py
======================================================
Shared utilities for cross-patient few-shot transfer-learning experiments.

Both arms of the experiment use ridge regression at peak time:

    Arm 1 (TRANSFER)    : ridge  HGA_X -> T_RB[word]   followed by RB's frozen
                          decoder Q_RB^T to recover the embedding.
    Arm 2 (NO-TRANSFER) : ridge  HGA_X -> embedding[word]   (X-only baseline).

Design choices:
    - Anchor = unique word with trial-averaged HGA at peak (word-anchored).
    - Both arms use ridge regression for a fair architecture match.
    - Per-bootstrap full M_X coefficients can be pickled for SVD / quiver plots.
"""

from __future__ import annotations

import os
import sys
import gc
import pickle as pk
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Ensure main/ is on the path ──────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_THIS_DIR)
_MAIN_DIR = os.path.dirname(_TESTS_DIR)
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

from tests.helpers._phoneme_semantic_helpers import (  # noqa: E402
    load_semantic_embeddings_for_patient,
    filter_nan_phoneme_trials,
    build_retrieval_db,
    compute_retrieval_metrics,
    N_BINS_HISTORY,
    PLS_COMPONENTS,
    SEMANTIC_EMBEDDINGS_TO_USE as _DEFAULT_EMBEDDINGS,
)
from utils.utils import reformat  # noqa: E402
from utils.patient_data import find_df_path  # noqa: E402


# ── Constants ────────────────────────────────────────────────────────────
DEFAULT_SOURCE_PATIENT = "RB"
DEFAULT_TARGET_PATIENTS = ["AA","AP","AZ","CP","DR","EH","EM","MM","VB", "WBH", "LH",]
DEFAULT_SOURCE_TASKS: Tuple[str, ...] = ("picture_naming",)
DEFAULT_TARGET_TASK = "picture_naming"
DEFAULT_EMBEDDINGS = list(_DEFAULT_EMBEDDINGS)  # ['GloVe']

DEFAULT_K_VALUES = [3, 5, 8, 12, 16, 20, 24]
DEFAULT_RIDGE_ALPHA = 1.0
DEFAULT_TEST_FRAC = 0.2
DEFAULT_N_BOOTSTRAP_PEAK = 200
DEFAULT_N_BOOTSTRAP_TIMECOURSE = 20
DEFAULT_N_BOOTSTRAP_MAPS = 20
DEFAULT_ARM3_RESULTS_ROOT = os.path.join(_MAIN_DIR, "results", "semantic_regression")
DEFAULT_PCA_COMPONENTS = 10
DEFAULT_ALIGN_START_BIN = 10   # first bin of alignment window (~1 s post trial-onset)
DEFAULT_ARMS = ["transfer", "no_transfer", "pca_align"]


# ── Data loading ─────────────────────────────────────────────────────────

def _load_patient_data_for_task(patient: str, task: str) -> dict:
    """Load patient data for an arbitrary task by temporarily swapping
    ``semantic_regression.TASK``."""
    import semantic_regression as sr
    saved_task = sr.TASK
    try:
        sr.TASK = task
        pdata = sr.load_patient_data(patient)
    finally:
        sr.TASK = saved_task
    return pdata


def load_patient_combined(patient: str, tasks: Sequence[str]) -> dict:
    """Load and (optionally) pool data across multiple tasks for a patient."""
    parts = []
    for t in tasks:
        patient_folder = os.path.join("data", patient)
        if find_df_path(patient_folder, patient, t) is None:
            print(f"  [INFO] {patient} has no {t} data; skipping.")
            continue
        pdata_t = _load_patient_data_for_task(patient, t)
        parts.append((t, pdata_t))
        gc.collect()

    if not parts:
        raise FileNotFoundError(
            f"No task data found for {patient} in {list(tasks)}"
        )
    if len(parts) == 1:
        out = parts[0][1].copy()
        n_trials = len(out["clean_answer_labels"])
        out["task_per_trial"] = np.array([parts[0][0]] * n_trials)
        return out

    base_shape = parts[0][1]["clean_data_binned"].shape
    base_n_ch, base_n_bins = base_shape[1], base_shape[2]
    for t, p in parts[1:]:
        s = p["clean_data_binned"].shape
        if s[1] != base_n_ch or s[2] != base_n_bins:
            raise ValueError(
                f"{patient}: task {t} has shape {s} but {parts[0][0]} has "
                f"{base_shape}. Cannot pool — investigate alignment / channel "
                f"masking before pooling."
            )

    binned = np.concatenate([p["clean_data_binned"] for _, p in parts], axis=0)
    labels = np.concatenate([p["clean_answer_labels"] for _, p in parts], axis=0)
    cats = np.concatenate([p["clean_word_category"] for _, p in parts], axis=0)
    task_per_trial = np.concatenate(
        [np.array([t] * len(p["clean_answer_labels"])) for t, p in parts], axis=0
    )
    return {
        "clean_data_binned": binned,
        "clean_answer_labels": labels,
        "clean_word_category": cats,
        "task_per_trial": task_per_trial,
    }


def get_features_per_bin(pdata: dict, n_bins_history: int = N_BINS_HISTORY) -> List[np.ndarray]:
    """Build per-bin lagged feature matrices.

    Returns list of length n_bins, each (n_trials, n_channels * n_bins_history).
    Early bins (with fewer available lags) are zero-padded on the left so that
    every bin has the same feature dimensionality and the time axis is preserved.
    """
    X = pdata["clean_data_binned"].swapaxes(1, 2)
    raw = reformat(X, n_bins_history)
    n_trials, n_channels = X.shape[0], X.shape[2]
    full_dim = n_channels * n_bins_history
    padded = []
    for feat in raw:
        if feat.shape[1] < full_dim:
            pad = np.zeros((n_trials, full_dim - feat.shape[1]), dtype=feat.dtype)
            feat = np.concatenate([pad, feat], axis=1)
        padded.append(feat)
    return padded


def get_shared_vocabulary(label_sets: Sequence[np.ndarray]) -> np.ndarray:
    common = set(np.unique(label_sets[0]).tolist())
    for s in label_sets[1:]:
        common &= set(np.unique(s).tolist())
    return np.array(sorted(common))


# ── PLS encoder / decoder extraction ─────────────────────────────────────

def fit_source_pls(X: np.ndarray, Y: np.ndarray, n_components: int = PLS_COMPONENTS) -> PLSRegression:
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X, Y)
    return pls


def encoder_matrix(pls: PLSRegression) -> np.ndarray:
    """E s.t. T = (X - x_mean_) @ E. E = W_x @ pinv(P_x^T W_x)."""
    W_x = pls.x_weights_
    P_x = pls.x_loadings_
    return W_x @ np.linalg.pinv(P_x.T @ W_x)


def decoder_matrix(pls: PLSRegression) -> np.ndarray:
    """D s.t. Y = (T @ D) + y_mean_, shape (n_components, n_targets)."""
    return pls.y_loadings_.T


def project_to_T(pls: PLSRegression, X: np.ndarray) -> np.ndarray:
    return (X - pls._x_mean) @ encoder_matrix(pls)


def decode_from_T(pls: PLSRegression, T: np.ndarray) -> np.ndarray:
    return T @ decoder_matrix(pls) + pls._y_mean


# ── Anchors ──────────────────────────────────────────────────────────────

def compute_T_anchors(
    pls_src: PLSRegression,
    X_src_peak: np.ndarray,
    labels_src: np.ndarray,
    shared_vocab: np.ndarray,
) -> Dict[str, np.ndarray]:
    T_anchors: Dict[str, np.ndarray] = {}
    labels_src = np.asarray(labels_src)
    for w in shared_vocab:
        mask = (labels_src == w)
        if mask.sum() == 0:
            continue
        x_mean = X_src_peak[mask].mean(axis=0, keepdims=True)
        t = project_to_T(pls_src, x_mean)[0]
        T_anchors[str(w)] = t
    return T_anchors


def compute_Y_anchors(
    Y_src: np.ndarray,
    labels_src: np.ndarray,
    shared_vocab: np.ndarray,
) -> Dict[str, np.ndarray]:
    Y_anchors: Dict[str, np.ndarray] = {}
    labels_src = np.asarray(labels_src)
    for w in shared_vocab:
        mask = (labels_src == w)
        if mask.sum() == 0:
            continue
        Y_anchors[str(w)] = Y_src[mask].mean(axis=0)
    return Y_anchors


def compute_X_src_anchors(
    X_src_peak: np.ndarray,
    labels_src: np.ndarray,
    shared_vocab: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute mean source HGA per word for each word in shared_vocab."""
    anchors: Dict[str, np.ndarray] = {}
    labels_src = np.asarray(labels_src)
    for w in shared_vocab:
        mask = (labels_src == w)
        if mask.sum() > 0:
            anchors[str(w)] = X_src_peak[mask].mean(axis=0)
    return anchors


# ── Peak finder ──────────────────────────────────────────────────────────

def find_peak_bin_source(
    X_features: List[np.ndarray],
    Y: np.ndarray,
    labels: np.ndarray,
    cats: np.ndarray,
    n_components: int = PLS_COMPONENTS,
    metric: str = "word_bal_acc",
    # Deprecated parameters kept for call-site compatibility; ignored.
    holdout_frac: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Tuple[int, np.ndarray]:
    """Find the peak time bin using ALL source trials (no train/test split).

    Trains PLS on every trial and evaluates retrieval on the same data so that
    the peak bin reflects the best fit of RB's model, not a noisy holdout.
    """
    db_embeds, unique_words_db, word_to_cat_idx, unique_cats, word_to_idx = \
        build_retrieval_db(Y, labels, cats)

    n_bins = len(X_features)
    metric_per_bin = np.full(n_bins, np.nan)
    for b in range(n_bins):
        Xb = X_features[b]
        try:
            pls = PLSRegression(n_components=n_components, scale=False)
            pls.fit(Xb, Y)
            Y_pred = pls.predict(Xb)
        except Exception:
            continue
        m = compute_retrieval_metrics(
            Y_pred, labels, cats,
            db_embeds, unique_words_db, word_to_cat_idx,
            unique_cats, word_to_idx,
        )
        metric_per_bin[b] = m.get(metric, np.nan)

    if np.all(np.isnan(metric_per_bin)):
        raise RuntimeError("Peak finder produced all-NaN metric_per_bin")
    peak_bin = int(np.nanargmax(metric_per_bin))
    return peak_bin, metric_per_bin


# ── Sampling + fitting ───────────────────────────────────────────────────

def stratified_train_test_split(
    labels: np.ndarray,
    test_frac: float = DEFAULT_TEST_FRAC,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng(0)
    n = len(labels)
    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    for w in np.unique(labels):
        idx = np.where(labels == w)[0].copy()
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac))) if len(idx) >= 2 else 0
        test_mask[idx[:n_test]] = True
        train_mask[idx[n_test:]] = True
    return np.where(train_mask)[0], np.where(test_mask)[0]


def sample_k_anchor_words(
    train_labels: np.ndarray,
    shared_vocab: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    available = np.array(sorted(set(train_labels.tolist()) & set(shared_vocab.tolist())))
    if k > len(available):
        return available
    chosen = rng.choice(available, size=k, replace=False)
    return chosen


def sample_k_anchor_words_from_vocab(
    shared_vocab: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample k anchor words directly from shared_vocab (no train set required)."""
    if k >= len(shared_vocab):
        return shared_vocab.copy()
    return rng.choice(shared_vocab, size=k, replace=False)


def word_based_split(
    labels: np.ndarray,
    anchor_words: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split trial indices by word: anchor word trials -> train, all others -> test."""
    anchor_set = set(str(w) for w in anchor_words)
    train_mask = np.array([str(lb) in anchor_set for lb in labels])
    return np.where(train_mask)[0], np.where(~train_mask)[0]


def build_anchored_inputs(
    X_train: np.ndarray,
    train_labels: np.ndarray,
    anchor_words: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    rows = []
    kept_words = []
    for w in anchor_words:
        mask = (train_labels == w)
        if mask.sum() == 0:
            continue
        rows.append(X_train[mask].mean(axis=0))
        kept_words.append(str(w))
    return np.stack(rows, axis=0), kept_words


def build_arm1_train_inputs(
    X_tgt_train: np.ndarray,
    labels_tgt_train: np.ndarray,
    anchor_words: np.ndarray,
    T_anchors: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """All individual target trials for anchor words, with mean source T_RB as Y.

    Returns (X_train, Y_T_rb_train, kept_words). Rows are trial-aligned.
    Each anchor word's trials in X are paired with the same (tiled) mean
    source PLS score vector (T_RB) for that word in Y.
    """
    rows_X: List[np.ndarray] = []
    rows_Y: List[np.ndarray] = []
    kept: List[str] = []
    for w in anchor_words:
        key = str(w)
        if key not in T_anchors:
            continue
        mask = (labels_tgt_train == w)
        if mask.sum() == 0:
            continue
        rows_X.append(X_tgt_train[mask])
        rows_Y.append(np.tile(T_anchors[key], (int(mask.sum()), 1)))
        kept.append(key)
    if not rows_X:
        t_dim = next(iter(T_anchors.values())).shape[0] if T_anchors else 1
        return np.empty((0, X_tgt_train.shape[1])), np.empty((0, t_dim)), []
    return np.concatenate(rows_X, axis=0), np.concatenate(rows_Y, axis=0), kept


def build_arm2_train_targets(
    labels_tgt_train: np.ndarray,
    kept_words: List[str],
    Y_tgt_train: np.ndarray,
) -> np.ndarray:
    """Word embeddings in the same trial order as build_arm1_train_inputs."""
    rows: List[np.ndarray] = []
    for w in kept_words:
        mask = (labels_tgt_train == w)
        if mask.sum() == 0:
            continue
        rows.append(Y_tgt_train[mask])
    if not rows:
        return np.empty((0, Y_tgt_train.shape[1]))
    return np.concatenate(rows, axis=0)


def fit_ridge(X: np.ndarray, Y: np.ndarray, alpha: float = DEFAULT_RIDGE_ALPHA) -> Ridge:
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(X, Y)
    return m


def fit_kernel_ridge(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float = DEFAULT_RIDGE_ALPHA,
) -> KernelRidge:
    """KernelRidge with RBF kernel — same nonlinear capacity as Arm 2's kernel PLS."""
    m = KernelRidge(kernel="rbf", alpha=alpha)
    m.fit(X, Y)
    return m


def fit_kernel_pls(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int = PLS_COMPONENTS,
) -> Pipeline:
    """Nystroem-RBF kernel approximation + PLS regression pipeline."""
    n_comp = max(1, min(n_components, X.shape[0] - 1))
    pipe = Pipeline([
        ("nystroem", Nystroem(kernel="rbf", random_state=0)),
        ("pls", PLSRegression(n_components=n_comp, scale=False)),
    ])
    pipe.fit(X, Y)
    return pipe


# ── PCA-alignment arm helpers ─────────────────────────────────────────────

def fit_pca_from_lagged(
    X_lagged: np.ndarray,
    n_channels: int,
    n_components: int = DEFAULT_PCA_COMPONENTS,
) -> PCA:
    """Fit a shared PCA on pooled per-bin slices of a lagged-feature matrix.

    X_lagged : (n_trials, n_channels * n_hist)  — each row is n_hist bins of HGA
                concatenated.  The bins are split out, stacked, and the PCA is
                fitted on the combined (n_trials * n_hist, n_channels) pool so
                that one PCA captures the per-channel structure across the whole
                alignment window rather than a single snapshot.
    """
    n_hist = X_lagged.shape[1] // n_channels
    bins = [X_lagged[:, b * n_channels:(b + 1) * n_channels] for b in range(n_hist)]
    X_pooled = np.concatenate(bins, axis=0)   # (n_trials * n_hist, n_channels)
    n_comp = min(n_components, X_pooled.shape[0], X_pooled.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(X_pooled)
    return pca


def project_to_multibin_pcs(X_lagged: np.ndarray, pca: PCA) -> np.ndarray:
    """Project each per-bin chunk of X_lagged through pca independently.

    Returns (n_trials, n_components * n_hist) — same time structure as the
    input but in PC space.  Works for any n_hist implied by pca.n_features_in_.
    """
    n_ch = pca.n_features_in_
    n_hist = X_lagged.shape[1] // n_ch
    chunks = [pca.transform(X_lagged[:, b * n_ch:(b + 1) * n_ch]) for b in range(n_hist)]
    return np.concatenate(chunks, axis=1)   # (n, n_components * n_hist)


def reconstruct_from_multibin_pcs(X_multibin: np.ndarray, pca: PCA) -> np.ndarray:
    """Inverse of project_to_multibin_pcs — back to (n, n_channels * n_hist)."""
    n_comp = pca.n_components_
    n_hist = X_multibin.shape[1] // n_comp
    chunks = [
        pca.inverse_transform(X_multibin[:, b * n_comp:(b + 1) * n_comp])
        for b in range(n_hist)
    ]
    return np.concatenate(chunks, axis=1)   # (n, n_channels * n_hist)


def compute_src_pca_anchors(
    pca_src: PCA,
    X_src_align: np.ndarray,
    labels_src: np.ndarray,
    shared_vocab: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Mean source multibin PC representation per shared word."""
    anchors: Dict[str, np.ndarray] = {}
    labels_src = np.asarray(labels_src)
    for w in shared_vocab:
        mask = (labels_src == w)
        if mask.sum() > 0:
            x_mean = X_src_align[mask].mean(axis=0, keepdims=True)
            anchors[str(w)] = project_to_multibin_pcs(x_mean, pca_src)[0]
    return anchors


def build_arm_pca_train_inputs(
    pca_tgt,   # PCA | None — caller guarantees non-None when run_pca is True
    X_tgt_align_train: np.ndarray,
    labels_tgt_train: np.ndarray,
    anchor_words: np.ndarray,
    src_pca_anchors,   # Dict[str, np.ndarray] | None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Project individual target trials to multibin PC space; pair with mean source PCs.

    Returns (X_tgt_pcs, Y_src_pcs, kept_words).
    X rows are per-trial target multibin PCs (n_comp * n_hist dims);
    Y rows are the tiled mean source multibin PCs for that word.
    """
    rows_X: List[np.ndarray] = []
    rows_Y: List[np.ndarray] = []
    kept: List[str] = []
    for w in anchor_words:
        key = str(w)
        if key not in src_pca_anchors:
            continue
        mask = (labels_tgt_train == w)
        if mask.sum() == 0:
            continue
        tgt_pcs = project_to_multibin_pcs(X_tgt_align_train[mask], pca_tgt)
        rows_X.append(tgt_pcs)
        rows_Y.append(np.tile(src_pca_anchors[key], (int(mask.sum()), 1)))
        kept.append(key)
    if not rows_X:
        d = next(iter(src_pca_anchors.values())).shape[0] if src_pca_anchors else 1
        n_hist = X_tgt_align_train.shape[1] // pca_tgt.n_features_in_
        return np.empty((0, pca_tgt.n_components_ * n_hist)), np.empty((0, d)), []
    return np.concatenate(rows_X, axis=0), np.concatenate(rows_Y, axis=0), kept


def predict_arm_pca_embedding(
    pls_src: PLSRegression,
    pca_src,   # PCA | None — caller guarantees non-None
    pca_tgt,   # PCA | None — caller guarantees non-None
    ridge_pca: Ridge,
    X_target_test: np.ndarray,
) -> np.ndarray:
    """target HGA → target multibin PCs → ridge → src multibin PCs → src HGA → PLS_RB → embedding."""
    tgt_multibin = project_to_multibin_pcs(X_target_test, pca_tgt)
    src_multibin = ridge_pca.predict(tgt_multibin)
    src_hga_hat = reconstruct_from_multibin_pcs(src_multibin, pca_src)
    return decode_from_T(pls_src, project_to_T(pls_src, src_hga_hat))


def predict_arm1_embedding(
    pls_src: PLSRegression,
    kernel_ridge_to_T: KernelRidge,
    X_target_test: np.ndarray,
) -> np.ndarray:
    """Map target HGA → predicted T_RB via kernel ridge → embedding via frozen decoder."""
    T_hat = kernel_ridge_to_T.predict(X_target_test)
    return decode_from_T(pls_src, T_hat)


def predict_arm2_embedding(ridge_y: Ridge, X_target_test: np.ndarray) -> np.ndarray:
    return ridge_y.predict(X_target_test)


def predict_arm2_kpls(kpls_pipeline: Pipeline, X_target_test: np.ndarray) -> np.ndarray:
    """Predict embedding via kernel PLS pipeline trained directly on target HGA."""
    return np.asarray(kpls_pipeline.predict(X_target_test))


# ── Scoring ──────────────────────────────────────────────────────────────

def score_predictions(
    Y_pred: np.ndarray,
    labels_test: np.ndarray,
    cats_test: np.ndarray,
    db_embeds: np.ndarray,
    unique_words_db: np.ndarray,
    word_to_cat_idx: np.ndarray,
    unique_cats: np.ndarray,
    word_to_idx: dict,
    anchor_words: Sequence[str],
) -> dict:
    m = compute_retrieval_metrics(
        Y_pred, labels_test, cats_test,
        db_embeds, unique_words_db, word_to_cat_idx, unique_cats, word_to_idx,
    )
    out = {
        "cosine_mean": m["cosine_mean"],
        "word_bal_acc": m["word_bal_acc"],
        "cat_indep_bal_acc": m["cat_indep_bal_acc"],
    }
    anchor_set = set(str(w).strip().lower() for w in anchor_words)
    seen_mask = np.array(
        [str(w).strip().lower() in anchor_set for w in labels_test], dtype=bool
    )
    if seen_mask.any():
        m_s = compute_retrieval_metrics(
            Y_pred[seen_mask], labels_test[seen_mask], cats_test[seen_mask],
            db_embeds, unique_words_db, word_to_cat_idx, unique_cats, word_to_idx,
        )
        out.update({
            "cosine_seen": m_s["cosine_mean"],
            "word_acc_seen": m_s["word_bal_acc"],
            "cat_acc_seen": m_s["cat_indep_bal_acc"],
        })
    else:
        out.update({"cosine_seen": np.nan, "word_acc_seen": np.nan, "cat_acc_seen": np.nan})

    if (~seen_mask).any():
        m_u = compute_retrieval_metrics(
            Y_pred[~seen_mask], labels_test[~seen_mask], cats_test[~seen_mask],
            db_embeds, unique_words_db, word_to_cat_idx, unique_cats, word_to_idx,
        )
        out.update({
            "cosine_unseen": m_u["cosine_mean"],
            "word_acc_unseen": m_u["word_bal_acc"],
            "cat_acc_unseen": m_u["cat_indep_bal_acc"],
        })
    else:
        out.update({"cosine_unseen": np.nan, "word_acc_unseen": np.nan, "cat_acc_unseen": np.nan})
    return out


# ── Existing-baseline loader (Arm 3) ─────────────────────────────────────

def load_arm3_baseline(
    patient: str,
    embedding: str,
    pic_run_folder: str,
    results_root: str = None,
) -> pd.DataFrame | None:
    """Read per_time_scores.csv from an existing kernel-PLS run for Arm 3."""
    if results_root is None:
        results_root = os.path.join(_MAIN_DIR, "results", "semantic_regression")
    csv_path = os.path.join(results_root, pic_run_folder, patient, "per_time_scores.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    sub = df[df["embedding"] == embedding]
    if sub.empty:
        return None
    return sub.reset_index(drop=True)


def load_arm3_chance(
    patient: str,
    embedding: str,
    pic_run_folder: str,
    results_root: str = None,
) -> dict:
    """Return chance levels for the given (patient, embedding) Arm 3 run.

    Output dict keys (any may be missing / None if the underlying CSV is absent):
        cosine_chance_per_bin   -- ndarray (n_bins,) of shuffled-cosine baseline
                                   from per_time_scores.csv['chance_mean']
        cosine_chance_bins      -- ndarray (n_bins,) of bin_index for the above
        word_chance             -- float, 1 / n_unique_words from top1 source CSV
        cat_chance              -- float, 1 / n_unique_categories from top1 source CSV
        n_unique_words          -- int (for label / debugging)
        n_unique_categories     -- int
    """
    if results_root is None:
        results_root = os.path.join(_MAIN_DIR, "results", "semantic_regression")
    out = {
        "cosine_chance_per_bin": None,
        "cosine_chance_bins": None,
        "word_chance": None,
        "cat_chance": None,
        "n_unique_words": None,
        "n_unique_categories": None,
    }
    # Per-bin shuffled-cosine chance from per_time_scores.csv
    pts_path = os.path.join(results_root, pic_run_folder, patient,
                            "per_time_scores.csv")
    if os.path.exists(pts_path):
        try:
            pts = pd.read_csv(pts_path)
            sub = pts[pts["embedding"] == embedding]
            if not sub.empty and "chance_mean" in sub.columns:
                out["cosine_chance_per_bin"] = sub["chance_mean"].values
                out["cosine_chance_bins"] = sub["bin_index"].values
        except Exception:
            pass
    # Word/category chance from top1_decoding_source_data.csv (trial-level)
    top1_path = os.path.join(results_root, pic_run_folder, patient,
                             "top1_decoding_source_data.csv")
    if os.path.exists(top1_path):
        try:
            t1 = pd.read_csv(top1_path, usecols=["embedding", "true_word",
                                                 "true_category"])
            sub = t1[t1["embedding"] == embedding]
            if not sub.empty:
                n_words = int(sub["true_word"].nunique())
                n_cats = int(sub["true_category"].nunique())
                out["n_unique_words"] = n_words
                out["n_unique_categories"] = n_cats
                if n_words > 0:
                    out["word_chance"] = 1.0 / n_words
                if n_cats > 0:
                    out["cat_chance"] = 1.0 / n_cats
        except Exception:
            pass
    return out


# -- Map-record bookkeeping for quiver / SVD analysis --

def svd_summary(M):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s2 = s ** 2
    eff_rank = float((s2.sum() ** 2) / (np.sum(s2 ** 2) + 1e-12))
    return {
        "U": U.astype(np.float32),
        "s": s.astype(np.float32),
        "Vt": Vt.astype(np.float32),
        "effective_rank": eff_rank,
        "normalized_spectrum": (s / (s.sum() + 1e-12)).astype(np.float32),
    }


def build_map_record(*, arm, k, bootstrap_id, ridge_model, HGA_anchored,
                     anchor_words, T_targets=None, Y_targets=None):
    pred_anchors = ridge_model.predict(HGA_anchored).astype(np.float32)

    # KernelRidge has dual_coef_ (n_train × n_out); Ridge has coef_ (n_out × n_feat).
    # Compute an effective linear weight matrix for SVD in both cases.
    if hasattr(ridge_model, "coef_"):
        coef = np.asarray(ridge_model.coef_, dtype=np.float32)
        intercept = np.asarray(ridge_model.intercept_, dtype=np.float32)
    else:
        # Effective primal weights: X_train^T @ alpha  (n_feat × n_out)
        dual = np.asarray(ridge_model.dual_coef_, dtype=np.float32)   # (n_train, n_out)
        coef = (HGA_anchored.T.astype(np.float32) @ dual).T           # (n_out, n_feat)
        intercept = np.zeros(coef.shape[0], dtype=np.float32)

    rec = {
        "arm": arm, "k": int(k), "bootstrap_id": int(bootstrap_id),
        "anchor_words": list(anchor_words),
        "coef": coef, "intercept": intercept, "pred_anchors": pred_anchors,
        "svd": svd_summary(coef),
    }
    if T_targets is not None:
        rec["T_RB_anchors"] = np.asarray(T_targets, dtype=np.float32)
    if Y_targets is not None:
        rec["Y_anchors_target"] = np.asarray(Y_targets, dtype=np.float32)
    return rec


def save_map_records(map_records, metadata, out_path):
    payload = {"metadata": metadata, "records": map_records}
    with open(out_path, "wb") as f:
        pk.dump(payload, f, protocol=pk.HIGHEST_PROTOCOL)


def load_map_records(in_path):
    with open(in_path, "rb") as f:
        return pk.load(f)


# -- Output helpers --

def get_out_dir(args_out_dir=None):
    base = os.path.join(_MAIN_DIR, "test_results")
    out = args_out_dir or base
    os.makedirs(out, exist_ok=True)
    return out


def header(msg):
    print(f"\n{'=' * 60}\n{msg}\n{'=' * 60}")


def step(msg):
    print(f"  {msg}")
