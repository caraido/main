# -*- coding: utf-8 -*-
"""
tests/cross_patient_decoding/cross_patient_aligned_semantic.py
===============================================================
Cross-patient semantic decoding using the aligned_decoding library
(AlignCCA, JointPCA, AlignMCCA) instead of the bespoke ridge-CCA approach.

Alignment is fitted on a 1-second window starting at trial onset (condition-
averaged over shared vocabulary words).  After alignment, a kernel PLS
decoder is trained on pooled (source shared-only + target train) trials
and evaluated on held-out target test trials.

Arms:
    cca_align   : AlignCCA maps source -> target channel space
    joint_pca   : JointPCA finds a shared low-d space
    mcca        : AlignMCCA (multiview CCA) shared space
    no_transfer : kernel PLS trained on target-only train trials

Usage (from main/):
    python -m _archive.cross_patient_decoding.cross_patient_aligned_semantic
    python -m _archive.cross_patient_decoding.cross_patient_aligned_semantic \\
        --target-patients VB WBH --embeddings GloVe --arms cca_align no_transfer
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAIN_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
_REPO_DIR = os.path.join(
    _MAIN_DIR, "supportive_repos", "cross_patient_speech_decoding"
)
for _p in [_MAIN_DIR, _REPO_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aligned_decoding.alignment.AlignCCA import AlignCCA          # noqa: E402
from aligned_decoding.alignment.JointPCA import JointPCA          # noqa: E402
from aligned_decoding.alignment.AlignMCCA import AlignMCCA        # noqa: E402

from analysis.helpers._phoneme_semantic_helpers import (              # noqa: E402
    build_retrieval_db,
    N_BINS_HISTORY,
    PLS_COMPONENTS,
    load_semantic_embeddings_for_patient,
    filter_nan_phoneme_trials,
)
from analysis.helpers._cross_patient_helpers import (  # noqa: E402
    DEFAULT_SOURCE_PATIENT,
    DEFAULT_TARGET_PATIENTS,
    DEFAULT_TARGET_TASK,
    DEFAULT_RIDGE_ALPHA,
    DEFAULT_N_BOOTSTRAP_PEAK,
    DEFAULT_N_BOOTSTRAP_TIMECOURSE,
    DEFAULT_EMBEDDINGS,
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_CCA_COMPONENTS,
    load_patient_combined,
    get_features_per_bin,
    get_shared_vocabulary,
    find_peak_bin_source,
    fit_kernel_pls,
    predict_arm2_kpls,
    score_predictions,
    word_based_split,
    get_out_dir,
    header,
    step,
)

# ── Constants ────────────────────────────────────────────────────────────
DEFAULT_VOCAB_TRAIN_FRAC = 0.7   # fraction of shared vocab used as train words
DEFAULT_N_COMPONENTS = 10
DEFAULT_MCCA_REGS = 0.5
DEFAULT_ARMS = ["cca_align", "joint_pca", "mcca", "no_transfer"]
ALIGN_WINDOW_SEC = 1.0           # seconds


# ── Alignment-window helpers ─────────────────────────────────────────────

def _get_adjusted_fs(pdata: dict) -> int:
    """Bins per second (e.g. 10 for 100 ms bins)."""
    return int(pdata.get("adjusted_fs", 10))


def _get_trial_onset_bin(pdata: dict) -> int:
    """Return the bin index corresponding to trial onset.

    When ALIGN_CUE='none' (default), actual_back_sec is None and bin 0 is
    already trial-onset-aligned.
    """
    actual_back_sec = pdata.get("actual_back_sec", None)
    if actual_back_sec is None:
        return 0
    rel_cues = pdata.get("rel_cues", {})
    trial_onset_mean = rel_cues.get("trial_onset", {}).get("mean", 0.0)
    adjusted_fs = _get_adjusted_fs(pdata)
    return int(round((actual_back_sec + trial_onset_mean) * adjusted_fs))


def get_alignment_window_3d(pdata: dict) -> np.ndarray:
    """Return (n_trials, n_align_bins, n_channels) array for the 1-s window.

    clean_data_binned is stored as (n_trials, n_channels, n_bins); we
    transpose to (n_trials, n_bins, n_channels) and slice.
    """
    X = np.asarray(pdata["clean_data_binned"])  # (n_trials, n_ch, n_bins)
    X_t = X.transpose(0, 2, 1)                  # (n_trials, n_bins, n_ch)
    adjusted_fs = _get_adjusted_fs(pdata)
    onset = _get_trial_onset_bin(pdata)
    n_align = max(1, int(round(ALIGN_WINDOW_SEC * adjusted_fs)))
    end = min(onset + n_align, X_t.shape[1])
    return X_t[:, onset:end, :]                  # (n_trials, n_align, n_ch)


# ── Alignment diagnostics ──────────────────────────────────────────────────

def _alignment_diagnostics(X_src_aligned, labels_src, X_tgt, labels_tgt):
    """Same-word centroid cosine + Spearman RSA between (1-cosine) RDMs."""
    from scipy.stats import spearmanr
    shared_words = sorted(set(labels_src) & set(labels_tgt))
    src_cents, tgt_cents = [], []
    for w in shared_words:
        s = X_src_aligned[labels_src == w]
        t = X_tgt[labels_tgt == w]
        if len(s) == 0 or len(t) == 0:
            continue
        src_cents.append(s.mean(0))
        tgt_cents.append(t.mean(0))
    n_words = len(src_cents)
    if n_words < 2:
        return {"diag_centroid_cos": np.nan, "diag_rsa": np.nan}
    src_cents = np.stack(src_cents)
    tgt_cents = np.stack(tgt_cents)
    src_n = src_cents / (np.linalg.norm(src_cents, axis=1, keepdims=True) + 1e-10)
    tgt_n = tgt_cents / (np.linalg.norm(tgt_cents, axis=1, keepdims=True) + 1e-10)
    centroid_cos = float(np.mean(np.sum(src_n * tgt_n, axis=1)))
    if n_words < 3:
        rsa_val = np.nan
    else:
        src_rdm = 1.0 - src_n @ src_n.T
        tgt_rdm = 1.0 - tgt_n @ tgt_n.T
        triu = np.triu_indices(n_words, k=1)
        r = spearmanr(src_rdm[triu], tgt_rdm[triu])
        rsa_val = float(r.statistic if hasattr(r, "statistic") else r.correlation)
    return {"diag_centroid_cos": centroid_cos, "diag_rsa": rsa_val}


# ── run_one ───────────────────────────────────────────────────────────────

def run_one(
    source_pdata: dict,
    source_sem: Dict[str, np.ndarray],
    source_X_per_bin: List[np.ndarray],
    target_patient: str,
    target_pdata: dict,
    target_sem: Dict[str, np.ndarray],
    target_X_per_bin: List[np.ndarray],
    embedding: str,
    args,
    rng_master: np.random.Generator,
    peak_bin: int,
    src_metric_per_bin: np.ndarray,
) -> pd.DataFrame:
    """Run all arms for one (target_patient, embedding) pair.

    Returns a DataFrame of per-bootstrap rows.
    """
    Y_src = source_sem[embedding]
    labels_src = np.asarray(source_pdata["clean_answer_labels"])
    cats_src = np.asarray(source_pdata["clean_word_category"])

    step(f"[{target_patient}/{embedding}]  peak_bin={peak_bin}  "
         f"(src metric={src_metric_per_bin[peak_bin]:.4f})")

    labels_tgt = np.asarray(target_pdata["clean_answer_labels"])
    cats_tgt = np.asarray(target_pdata["clean_word_category"])
    Y_tgt = target_sem[embedding]

    # Drop target trials whose embedding vector is NaN
    valid_tgt = ~np.isnan(Y_tgt).any(axis=1)
    if not valid_tgt.all():
        n_drop = int((~valid_tgt).sum())
        step(f"  [{target_patient}/{embedding}] dropping {n_drop} NaN-embedding trials")
        labels_tgt = labels_tgt[valid_tgt]
        cats_tgt   = cats_tgt[valid_tgt]
        Y_tgt      = Y_tgt[valid_tgt]
        tgt_X_all  = [X[valid_tgt] for X in target_X_per_bin]
        tgt_pdata_local = dict(target_pdata)
        tgt_pdata_local["clean_data_binned"] = (
            np.asarray(target_pdata["clean_data_binned"])[valid_tgt]
        )
    else:
        tgt_X_all         = list(target_X_per_bin)
        tgt_pdata_local   = target_pdata

    # Drop source trials whose embedding vector is NaN
    valid_src = ~np.isnan(Y_src).any(axis=1)
    if not valid_src.all():
        n_drop = int((~valid_src).sum())
        step(f"  [source/{embedding}] dropping {n_drop} NaN-embedding source trials")
        labels_src   = labels_src[valid_src]
        cats_src     = cats_src[valid_src]
        Y_src        = Y_src[valid_src]
        src_X_all    = [X[valid_src] for X in source_X_per_bin]
        src_pdata_local = dict(source_pdata)
        src_pdata_local["clean_data_binned"] = (
            np.asarray(source_pdata["clean_data_binned"])[valid_src]
        )
    else:
        src_X_all        = list(source_X_per_bin)
        src_pdata_local  = source_pdata

    shared_vocab = get_shared_vocabulary([labels_src, labels_tgt])
    if len(shared_vocab) < 3:
        step(f"  [{target_patient}/{embedding}] only {len(shared_vocab)} shared words; skip")
        return pd.DataFrame()

    n_train_words = max(2, int(round(len(shared_vocab) * args.vocab_train_frac)))

    # Retrieval DB built on ALL target embeddings (used for scoring)
    db_embeds, unique_words_db, word_to_cat_idx, unique_cats, word_to_idx = (
        build_retrieval_db(Y_tgt, labels_tgt, cats_tgt)
    )

    # Pre-compute alignment windows (source: all trials; target: indexed per bootstrap)
    X_src_align_3d = get_alignment_window_3d(src_pdata_local)   # (n_src, n_align, n_ch_src)
    X_tgt_align_3d = get_alignment_window_3d(tgt_pdata_local)   # (n_tgt, n_align, n_ch_tgt)

    n_align = X_src_align_3d.shape[1]

    # Peak-bin index relative to the alignment window start
    onset_tgt = _get_trial_onset_bin(tgt_pdata_local)
    peak_in_window = max(0, min(peak_bin - onset_tgt, n_align - 1))

    # Single-bin raw-channel slices at the true global peak_bin.
    # The alignment transforms are time-agnostic (channel dimension only),
    # so peak_bin need not lie within the 1-second alignment window.
    X_src_peak_raw = np.asarray(
        src_pdata_local["clean_data_binned"]
    ).transpose(0, 2, 1)[:, peak_bin, :]   # (n_src, n_ch_src)
    X_tgt_peak_raw = np.asarray(
        tgt_pdata_local["clean_data_binned"]
    ).transpose(0, 2, 1)[:, peak_bin, :]   # (n_tgt, n_ch_tgt)

    # Full-trial 3D array for extended timecourse (n_trials, n_bins, n_ch_tgt).
    # Only materialised when --full-timecourse is requested.
    full_timecourse = getattr(args, "full_timecourse", False)
    if full_timecourse:
        X_tgt_full_3d = np.asarray(
            tgt_pdata_local["clean_data_binned"]
        ).transpose(0, 2, 1)   # (n_tgt, n_bins, n_ch_tgt)
    else:
        X_tgt_full_3d = None

    n_bins_total = len(tgt_X_all)

    rows: list = []

    for b_idx in range(args.n_bootstrap_peak):
        rng = np.random.default_rng(rng_master.integers(0, 2**31))

        # --- sample train words ---
        train_words = rng.choice(shared_vocab, size=n_train_words, replace=False)
        train_idx, test_idx = word_based_split(labels_tgt, train_words)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        # Source trials that belong to a shared (train) word
        shared_src_mask = np.isin(labels_src, train_words)
        shared_src_idx  = np.where(shared_src_mask)[0]
        if len(shared_src_idx) == 0:
            continue

        # Alignment windows for this bootstrap
        X_tgt_align_train = X_tgt_align_3d[train_idx]   # (n_train, n_align, n_ch_tgt)
        y_tgt_train_words  = labels_tgt[train_idx]
        y_src_words        = labels_src                  # all source labels

        # Single-bin peak features for target
        X_tgt_train_peak = tgt_X_all[peak_bin][train_idx]
        X_tgt_test_peak  = tgt_X_all[peak_bin][test_idx]
        Y_tgt_train      = Y_tgt[train_idx]

        run_timecourse = (b_idx < args.n_bootstrap_timecourse)

        for arm in args.arms:
            if arm == "no_transfer":
                kpls = fit_kernel_pls(X_tgt_train_peak, Y_tgt_train)
                Yhat = predict_arm2_kpls(kpls, X_tgt_test_peak)
                scores = score_predictions(
                    Yhat, labels_tgt[test_idx], cats_tgt[test_idx],
                    db_embeds, unique_words_db, word_to_cat_idx, unique_cats,
                    word_to_idx, list(train_words),
                )
                rows.append(_make_row(arm, b_idx, peak_bin, "peak", scores,
                                      target_patient, embedding,
                                      args.source_patient,
                                      len(train_words), len(test_idx)))
                if run_timecourse:
                    for t in range(n_bins_total):
                        if t == peak_bin:
                            continue
                        Yhat_t = predict_arm2_kpls(kpls, tgt_X_all[t][test_idx])
                        sc_t = score_predictions(
                            Yhat_t, labels_tgt[test_idx], cats_tgt[test_idx],
                            db_embeds, unique_words_db, word_to_cat_idx, unique_cats,
                            word_to_idx, list(train_words),
                        )
                        rows.append(_make_row(arm, b_idx, t, "timecourse", sc_t,
                                              target_patient, embedding,
                                              args.source_patient,
                                              len(train_words), len(test_idx)))

            elif arm == "cca_align":
                aligner = AlignCCA(type="class", return_space="b_to_a")
                # target = a (reference space), source = b
                aligner.fit(
                    X_tgt_align_train, X_src_align_3d,
                    y_tgt_train_words, y_src_words,
                )
                # Map source at true global peak_bin into target channel space.
                X_src_peak_mapped = aligner.transform(
                    X_src_peak_raw[:, np.newaxis, :]
                )[:, 0, :]                                      # (n_src, n_ch_tgt)
                X_src_shared_peak    = X_src_peak_mapped[shared_src_idx]
                X_tgt_train_peak_raw = X_tgt_peak_raw[train_idx]
                X_tgt_test_peak_raw  = X_tgt_peak_raw[test_idx]
                X_pool = np.vstack([X_src_shared_peak, X_tgt_train_peak_raw])
                Y_pool = np.vstack([Y_src[shared_src_idx], Y_tgt_train])
                kpls = fit_kernel_pls(X_pool, Y_pool)
                Yhat = predict_arm2_kpls(kpls, X_tgt_test_peak_raw)
                scores = score_predictions(
                    Yhat, labels_tgt[test_idx], cats_tgt[test_idx],
                    db_embeds, unique_words_db, word_to_cat_idx, unique_cats,
                    word_to_idx, list(train_words),
                )
                diag = _alignment_diagnostics(
                    X_src_peak_mapped[shared_src_idx], labels_src[shared_src_idx],
                    X_tgt_peak_raw[train_idx], labels_tgt[train_idx],
                )
                rows.append(_make_row(arm, b_idx, peak_bin, "peak", scores,
                                      target_patient, embedding,
                                      args.source_patient,
                                      len(train_words), len(test_idx), diag=diag))
                if run_timecourse:
                    if full_timecourse:
                        _tc_bins_cca = range(n_bins_total)
                        _skip_cca = peak_bin
                        def _tgt_cca(t):
                            return X_tgt_full_3d[test_idx, t, :]
                    else:
                        _tc_bins_cca = range(n_align)
                        _skip_cca = peak_in_window
                        def _tgt_cca(t):
                            return X_tgt_align_3d[test_idx, t, :]
                    for t in _tc_bins_cca:
                        if t == _skip_cca:
                            continue
                        global_t = t if full_timecourse else onset_tgt + t
                        Yhat_t = predict_arm2_kpls(kpls, _tgt_cca(t))
                        sc_t = score_predictions(
                            Yhat_t, labels_tgt[test_idx], cats_tgt[test_idx],
                            db_embeds, unique_words_db, word_to_cat_idx, unique_cats,
                            word_to_idx, list(train_words),
                        )
                        rows.append(_make_row(arm, b_idx, global_t, "timecourse", sc_t,
                                              target_patient, embedding,
                                              args.source_patient,
                                              len(train_words), len(test_idx)))

            elif arm == "joint_pca":
                j = JointPCA(n_components=args.n_components)
                j.fit_transform(
                    [X_tgt_align_train, X_src_align_3d],
                    [y_tgt_train_words, y_src_words],
                )
                # Evaluate at true global peak_bin (not window-clamped).
                X_src_peak_j = j.transform(
                    X_src_peak_raw[:, np.newaxis, :], idx=1
                )[:, 0, :]                                      # (n_src, n_comp)
                X_src_shared_peak_j = X_src_peak_j[shared_src_idx]
                X_tgt_peak_j = j.transform(
                    X_tgt_peak_raw[:, np.newaxis, :], idx=0
                )[:, 0, :]                                      # (n_tgt, n_comp)
                X_tgt_train_peak_j  = X_tgt_peak_j[train_idx]
                X_pool = np.vstack([X_src_shared_peak_j, X_tgt_train_peak_j])
                Y_pool = np.vstack([Y_src[shared_src_idx], Y_tgt_train])
                kpls = fit_kernel_pls(X_pool, Y_pool)

                # Transform test set (target = idx 0); keep for window timecourse
                X_tgt_test_j = j.transform(X_tgt_align_3d[test_idx], idx=0)
                X_tgt_test_peak_j = X_tgt_peak_j[test_idx]
                Yhat = predict_arm2_kpls(kpls, X_tgt_test_peak_j)
                scores = score_predictions(
                    Yhat, labels_tgt[test_idx], cats_tgt[test_idx],
                    db_embeds, unique_words_db, word_to_cat_idx, unique_cats,
                    word_to_idx, list(train_words),
                )
                diag = _alignment_diagnostics(
                    X_src_peak_j[shared_src_idx], labels_src[shared_src_idx],
                    X_tgt_peak_j[train_idx], labels_tgt[train_idx],
                )
                rows.append(_make_row(arm, b_idx, peak_bin, "peak", scores,
                                      target_patient, embedding,
                                      args.source_patient,
                                      len(train_words), len(test_idx), diag=diag))
                if run_timecourse:
                    if full_timecourse:
                        _tc_bins_j = range(n_bins_total)
                        _skip_j = peak_bin
                        def _tgt_jpca(t, _j=j):
                            return _j.transform(
                                X_tgt_full_3d[test_idx, t:t+1, :], idx=0
                            )[:, 0, :]
                    else:
                        _tc_bins_j = range(n_align)
                        _skip_j = peak_in_window
                        def _tgt_jpca(t):
                            return X_tgt_test_j[:, t, :]
                    for t in _tc_bins_j:
                        if t == _skip_j:
                            continue
                        global_t = t if full_timecourse else onset_tgt + t
                        Yhat_t = predict_arm2_kpls(kpls, _tgt_jpca(t))
                        sc_t = score_predictions(
                            Yhat_t, labels_tgt[test_idx], cats_tgt[test_idx],
                            db_embeds, unique_words_db, word_to_cat_idx, unique_cats,
                            word_to_idx, list(train_words),
                        )
                        rows.append(_make_row(arm, b_idx, global_t, "timecourse", sc_t,
                                              target_patient, embedding,
                                              args.source_patient,
                                              len(train_words), len(test_idx)))

            elif arm == "mcca":
                m = AlignMCCA(n_components=args.n_components, regs=args.mcca_regs)
                m.fit(
                    [X_tgt_align_train, X_src_align_3d],
                    [y_tgt_train_words, y_src_words],
                )
                # Evaluate at true global peak_bin (not window-clamped).
                X_src_peak_m = m.transform(
                    X_src_peak_raw[:, np.newaxis, :], idx=1
                )[:, 0, :]                                      # (n_src, n_comp)
                X_src_shared_peak_m = X_src_peak_m[shared_src_idx]
                X_tgt_peak_m = m.transform(
                    X_tgt_peak_raw[:, np.newaxis, :], idx=0
                )[:, 0, :]                                      # (n_tgt, n_comp)
                X_tgt_train_peak_m  = X_tgt_peak_m[train_idx]
                X_pool = np.vstack([X_src_shared_peak_m, X_tgt_train_peak_m])
                Y_pool = np.vstack([Y_src[shared_src_idx], Y_tgt_train])
                kpls = fit_kernel_pls(X_pool, Y_pool)

                # Keep for window timecourse path
                X_tgt_test_m = m.transform(X_tgt_align_3d[test_idx], idx=0)
                X_tgt_test_peak_m = X_tgt_peak_m[test_idx]
                Yhat = predict_arm2_kpls(kpls, X_tgt_test_peak_m)
                scores = score_predictions(
                    Yhat, labels_tgt[test_idx], cats_tgt[test_idx],
                    db_embeds, unique_words_db, word_to_cat_idx, unique_cats,
                    word_to_idx, list(train_words),
                )
                diag = _alignment_diagnostics(
                    X_src_peak_m[shared_src_idx], labels_src[shared_src_idx],
                    X_tgt_peak_m[train_idx], labels_tgt[train_idx],
                )
                rows.append(_make_row(arm, b_idx, peak_bin, "peak", scores,
                                      target_patient, embedding,
                                      args.source_patient,
                                      len(train_words), len(test_idx), diag=diag))
                if run_timecourse:
                    if full_timecourse:
                        _tc_bins_m = range(n_bins_total)
                        _skip_m = peak_bin
                        def _tgt_mcca(t, _m=m):
                            return _m.transform(
                                X_tgt_full_3d[test_idx, t:t+1, :], idx=0
                            )[:, 0, :]
                    else:
                        _tc_bins_m = range(n_align)
                        _skip_m = peak_in_window
                        def _tgt_mcca(t):
                            return X_tgt_test_m[:, t, :]
                    for t in _tc_bins_m:
                        if t == _skip_m:
                            continue
                        global_t = t if full_timecourse else onset_tgt + t
                        Yhat_t = predict_arm2_kpls(kpls, _tgt_mcca(t))
                        sc_t = score_predictions(
                            Yhat_t, labels_tgt[test_idx], cats_tgt[test_idx],
                            db_embeds, unique_words_db, word_to_cat_idx, unique_cats,
                            word_to_idx, list(train_words),
                        )
                        rows.append(_make_row(arm, b_idx, global_t, "timecourse", sc_t,
                                              target_patient, embedding,
                                              args.source_patient,
                                              len(train_words), len(test_idx)))

    return pd.DataFrame(rows)


def _make_row(arm, bootstrap_id, bin_idx, phase, scores,
              target_patient, embedding, source_patient,
              n_train_words, n_test_trials, diag=None) -> dict:
    row = {
        "arm": arm,
        "bootstrap_id": bootstrap_id,
        "bin_index": bin_idx,
        "phase": phase,
        "target_patient": target_patient,
        "source_patient": source_patient,
        "embedding": embedding,
        "n_train_words": n_train_words,
        "n_test_trials": n_test_trials,
        **scores,
        "diag_centroid_cos": diag["diag_centroid_cos"] if diag else float("nan"),
        "diag_rsa": diag["diag_rsa"] if diag else float("nan"),
    }
    return row


# ── Build helpers ────────────────────────────────────────────────────────

def _build_source(args, shared_models):
    src_tasks = ["picture_naming"]
    if args.pool_flashing:
        src_tasks.append("picture_flashing")
    header(f"Loading SOURCE {args.source_patient}  tasks = {src_tasks}")
    src_pdata = load_patient_combined(args.source_patient, src_tasks)
    src_sem = load_semantic_embeddings_for_patient(src_pdata, shared_models)
    src_pdata, src_sem = filter_nan_phoneme_trials(src_pdata, src_sem)
    step(f"  {args.source_patient}: trials={len(src_pdata['clean_answer_labels'])}"
         f"  channels={src_pdata['clean_data_binned'].shape[1]}"
         f"  bins={src_pdata['clean_data_binned'].shape[2]}")
    src_X_per_bin = get_features_per_bin(src_pdata, n_bins_history=N_BINS_HISTORY)
    return src_pdata, src_sem, src_X_per_bin


def _build_target(patient: str, shared_models):
    header(f"Loading TARGET {patient}  task = {DEFAULT_TARGET_TASK}")
    tgt_pdata = load_patient_combined(patient, [DEFAULT_TARGET_TASK])
    tgt_sem = load_semantic_embeddings_for_patient(tgt_pdata, shared_models)
    step(f"  {patient}: trials={len(tgt_pdata['clean_answer_labels'])}"
         f"  channels={tgt_pdata['clean_data_binned'].shape[1]}"
         f"  bins={tgt_pdata['clean_data_binned'].shape[2]}")
    tgt_X_per_bin = get_features_per_bin(tgt_pdata, n_bins_history=N_BINS_HISTORY)
    return tgt_pdata, tgt_sem, tgt_X_per_bin


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-patient semantic decoding with aligned_decoding library."
    )
    parser.add_argument("--source-patient", default=DEFAULT_SOURCE_PATIENT)
    parser.add_argument("--target-patients", nargs="+", default=DEFAULT_TARGET_PATIENTS)
    parser.add_argument("--embeddings", nargs="+", default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--pool-flashing", action="store_true")
    parser.add_argument("--arms", nargs="+", default=DEFAULT_ARMS,
                        choices=DEFAULT_ARMS)
    parser.add_argument("--vocab-train-frac", type=float,
                        default=DEFAULT_VOCAB_TRAIN_FRAC,
                        help="Fraction of shared vocabulary assigned to train split.")
    parser.add_argument("--n-components", type=int, default=DEFAULT_N_COMPONENTS,
                        help="Dimensionality for JointPCA / AlignMCCA.")
    parser.add_argument("--mcca-regs", type=float, default=DEFAULT_MCCA_REGS,
                        help="Regularisation for AlignMCCA.")
    parser.add_argument("--n-bootstrap-peak", type=int,
                        default=DEFAULT_N_BOOTSTRAP_PEAK)
    parser.add_argument("--n-bootstrap-timecourse", type=int,
                        default=DEFAULT_N_BOOTSTRAP_TIMECOURSE)
    parser.add_argument("--full-timecourse", action="store_true", default=False,
                        help="Extend alignment-arm timecourse to all trial bins by "
                             "applying the fitted aligner to each bin of the full "
                             "clean_data_binned array. Default: window-only (~10 bins).")
    parser.add_argument("--pls-components", type=int, default=PLS_COMPONENTS)
    parser.add_argument("--peak-metric", default="cat_indep_bal_acc",
                        choices=["word_bal_acc", "cat_indep_bal_acc", "cosine_mean"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip (target, embedding) combos whose CSV already exists.")
    args = parser.parse_args()

    if args.embeddings == ["all"]:
        args.embeddings = DEFAULT_EMBEDDINGS

    out_dir = get_out_dir(args.out_dir)
    header("CROSS-PATIENT ALIGNED SEMANTIC DECODING")
    print(f"  source            : {args.source_patient}   "
          f"pool_flashing={args.pool_flashing}")
    print(f"  targets           : {args.target_patients}")
    print(f"  embeddings        : {args.embeddings}")
    print(f"  arms              : {args.arms}")
    print(f"  vocab_train_frac  : {args.vocab_train_frac}")
    print(f"  n_components      : {args.n_components}")
    print(f"  mcca_regs         : {args.mcca_regs}")
    print(f"  bootstrap         : peak={args.n_bootstrap_peak}  "
          f"time={args.n_bootstrap_timecourse}")
    print(f"  pls_components    : {args.pls_components}")
    print(f"  peak_metric       : {args.peak_metric}")
    print(f"  out dir           : {out_dir}")

    from semantic_regression import load_shared_embedding_models  # noqa: E402
    shared_models = load_shared_embedding_models()

    src_pdata, src_sem, src_X_per_bin = _build_source(args, shared_models)
    rng_master = np.random.default_rng(args.seed)

    # Source peak bin found once per embedding (deterministic, no holdout)
    src_peak_cache: dict = {}

    all_dfs: list = []

    for tgt in args.target_patients:
        if tgt == args.source_patient:
            step(f"  Skipping {tgt} (== source)")
            continue
        try:
            tgt_pdata, tgt_sem, tgt_X_per_bin = _build_target(tgt, shared_models)
        except FileNotFoundError as e:
            step(f"  {tgt}: cannot load ({e}); skipping")
            continue

        for emb in args.embeddings:
            csv_path = os.path.join(
                out_dir,
                f"cross_patient_aligned_semantic_{args.source_patient}_to_{tgt}_{emb}.csv",
            )
            if args.resume and os.path.exists(csv_path):
                step(f"  RESUME: {csv_path} exists, skipping.")
                all_dfs.append(pd.read_csv(csv_path))
                continue

            if emb not in src_peak_cache:
                Y_src = src_sem[emb]
                labels_src = np.asarray(src_pdata["clean_answer_labels"])
                cats_src   = np.asarray(src_pdata["clean_word_category"])
                valid_src  = ~np.isnan(Y_src).any(axis=1)
                step(f"[source/{emb}] finding source peak bin...")
                peak_bin, src_metric_per_bin = find_peak_bin_source(
                    [X[valid_src] for X in src_X_per_bin],
                    Y_src[valid_src],
                    labels_src[valid_src],
                    cats_src[valid_src],
                    n_components=args.pls_components,
                    metric=args.peak_metric,
                )
                step(f"  source peak_bin={peak_bin}  "
                     f"({args.peak_metric}={src_metric_per_bin[peak_bin]:.4f})")
                src_peak_cache[emb] = (peak_bin, src_metric_per_bin)
            peak_bin, src_metric_per_bin = src_peak_cache[emb]

            t0 = time.time()
            df = run_one(
                src_pdata, src_sem, src_X_per_bin,
                tgt, tgt_pdata, tgt_sem, tgt_X_per_bin,
                emb, args, rng_master, peak_bin, src_metric_per_bin,
            )
            elapsed = time.time() - t0

            if df.empty:
                step(f"  [{tgt}/{emb}] no results produced; skipping save")
                continue

            df.to_csv(csv_path, index=False)
            all_dfs.append(df)
            step(f"  [{tgt}/{emb}] saved {len(df)} rows → {csv_path}  "
                 f"({elapsed:.1f}s)")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = os.path.join(
            out_dir,
            f"cross_patient_aligned_semantic_{args.source_patient}_combined.csv",
        )
        combined.to_csv(combined_path, index=False)
        header(f"Combined results saved → {combined_path}")


if __name__ == "__main__":
    main()
