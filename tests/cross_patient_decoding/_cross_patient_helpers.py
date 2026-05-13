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
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")

# ── Ensure main/ is on the path ──────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_THIS_DIR)
_MAIN_DIR = os.path.dirname(_TESTS_DIR)
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

from tests.helpers._phoneme_semantic_helpers import (  # noqa: E402
    load_phoneme_embeddings_for_patient,
    filter_nan_phoneme_trials,
    build_retrieval_db,
    compute_retrieval_metrics,
    N_BINS_HISTORY,
    PLS_COMPONENTS,
    PHONEME_EMBEDDINGS as _DEFAULT_PHONEME_EMBEDDINGS,
)
from utils.utils import reformat  # noqa: E402
from utils.patient_data import find_df_path  # noqa: E402


# ── Constants ────────────────────────────────────────────────────────────
DEFAULT_SOURCE_PATIENT = "RB"
DEFAULT_TARGET_PATIENTS = ["VB", "WBH", "LH"]
DEFAULT_SOURCE_TASKS: Tuple[str, ...] = ("picture_naming",)
DEFAULT_TARGET_TASK = "picture_naming"
PHONEME_EMBEDDINGS = list(_DEFAULT_PHONEME_EMBEDDINGS) + ["token_ipa"]

DEFAULT_K_VALUES = [3, 5, 8, 12, 16, 20, 24]
DEFAULT_RIDGE_ALPHA = 1.0
DEFAULT_TEST_FRAC = 0.2
DEFAULT_N_BOOTSTRAP_PEAK = 200
DEFAULT_N_BOOTSTRAP_TIMECOURSE = 20
DEFAULT_N_BOOTSTRAP_MAPS = 20


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
    """
    X = pdata["clean_data_binned"].swapaxes(1, 2)
    return reformat(X, n_bins_history)


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
    return (X - pls.x_mean_) @ encoder_matrix(pls)


def decode_from_T(pls: PLSRegression, T: np.ndarray) -> np.ndarray:
    return T @ decoder_matrix(pls) + pls.y_mean_


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


# ── Peak finder ──────────────────────────────────────────────────────────

def find_peak_bin_source(
    X_features: List[np.ndarray],
    Y: np.ndarray,
    labels: np.ndarray,
    cats: np.ndarray,
    n_components: int = PLS_COMPONENTS,
    holdout_frac: float = 0.2,
    rng: np.random.Generator | None = None,
    metric: str = "word_bal_acc",
) -> Tuple[int, np.ndarray]:
    rng = rng or np.random.default_rng(0)
    n_trials = Y.shape[0]
    unique_words = np.unique(labels)
    train_mask = np.zeros(n_trials, dtype=bool)
    test_mask = np.zeros(n_trials, dtype=bool)
    for w in unique_words:
        idx = np.where(labels == w)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * holdout_frac)))
        if len(idx) < 2:
            train_mask[idx] = True
            continue
        test_mask[idx[:n_test]] = True
        train_mask[idx[n_test:]] = True

    db_embeds, unique_words_db, word_to_cat_idx, unique_cats, word_to_idx = \
        build_retrieval_db(Y, labels, cats)

    n_bins = len(X_features)
    metric_per_bin = np.full(n_bins, np.nan)
    for b in range(n_bins):
        Xb = X_features[b]
        try:
            pls = PLSRegression(n_components=n_components, scale=False)
            pls.fit(Xb[train_mask], Y[train_mask])
            Y_pred = pls.predict(Xb[test_mask])
        except Exception:
            continue
        m = compute_retrieval_metrics(
            Y_pred, labels[test_mask], cats[test_mask],
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


def fit_ridge(X: np.ndarray, Y: np.ndarray, alpha: float = DEFAULT_RIDGE_ALPHA) -> Ridge:
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(X, Y)
    return m


def predict_arm1_embedding(pls_src: PLSRegression, ridge_t: Ridge, X_target_test: np.ndarray) -> np.ndarray:
    T_hat = ridge_t.predict(X_target_test)
    return decode_from_T(pls_src, T_hat)


def predict_arm2_embedding(ridge_y: Ridge, X_target_test: np.ndarray) -> np.ndarray:
    return ridge_y.predict(X_target_test)


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


# ── Map-record bookkeeping for quiver / SVD analysis ────────────────────

def svd_summary(M: np.ndarray) -> dict:
    """SVD of ridge coefficient matrix M; returns U, s, Vt, effective_rank."""
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


def build_map_record(
    *,
    arm: str,
    k: int,
    bootstrap_id: int,
    ridge_model,
    HGA_anchored: np.ndarray,
    anchor_words: List[str],
    T_targets: np.ndarray | None = None,
    Y_targets: np.ndarray | None = None,
) -> dict:
    coef = np.asarray(ridge_model.coef_, dtype=np.float32)
    intercept = np.asarray(ridge_model.intercept_, dtype=np.float32)
    pred_anchors = ridge_model.predict(HGA_anchored).astype(np.float32)
    rec = {
        "arm": arm,
        "k": int(k),
        "bootstrap_id": int(bootstrap_id),
        "anchor_words": list(anchor_words),
        "coef": coef,
        "intercept": intercept,
        "pred_anchors": pred_anchors,
        "svd": svd_summary(coef),
    }
    if T_targets is not None:
        rec["T_RB_anchors"] = np.asarray(T_targets, dtype=np.float32)
    if Y_targets is not None:
        rec["Y_anchors_target"] = np.asarray(Y_targets, dtype=np.float32)
    return rec


def save_map_records(map_records: List[dict], metadata: dict, out_path: str) -> None:
    payload = {"metadata": metadata, "records": map_records}
    with open(out_path, "wb") as f:
        pk.dump(payload, f, protocol=pk.HIGHEST_PROTOCOL)


def load_map_records(in_path: str) -> dict:
    with open(in_path, "rb") as f:
        return pk.load(f)


# ── Output helpers ───────────────────────────────────────────────────────

def get_out_dir(args_out_dir: str | None = None) -> str:
    base = os.path.join(_MAIN_DIR, "test_results")
    out = args_out_dir or base
    os.makedirs(out, exist_ok=True)
    return out


def header(msg: str) -> None:
    print(f"\n{'=' * 60}\n{msg}\n{'=' * 60}")


def step(msg: str) -> None:
    print(f"  {msg}")
