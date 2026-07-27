# -*- coding: utf-8 -*-
"""
tests/open_vocab_retrieval/predict_io.py
========================================
Step 2 of the guide: produce per-trial PREDICTED embeddings with cross-
validation folds and a zero-shot (held-out-word) flag, from the existing
per-patient semantic-regression pkls.

The decoder (kernel-PLS: Nystroem-RBF + PLSRegression, regressing neural HGA at
the loose-category peak bin onto GloVe) is re-fit under cross-validation so that
every trial receives an out-of-fold predicted embedding.  Two test regimes are
distinguished (guide Claim 2):

  * **in-vocab** trial — its word appears (via other trials) in the training set
    of the fold that predicted it.
  * **held-out** trial — its word is withheld from ALL training folds (zero-shot).

Held-out status is resolved from the model's own trial labels (``reg.labels``,
already lemmatized) with the ``(category)`` disambiguation suffix stripped, at
the *clean-word* level — so both senses of e.g. ``mouse`` are held out together
and no sense leaks into training (matches the seen/unseen caveat in the repo).

Reused project infrastructure:
  * ``analysis.helpers.load_results_pkl``                    — unpickle results
  * ``cross_task.cross_task_regression.find_peak_bin/get_trial_metadata`` — peak bin + categories
  * ``cross_task.cross_task_cotrain.make_model``          — kernel-PLS builder

Output: a :class:`TrialPredictions` bundle (parallel arrays) that the retrieval
and metrics steps consume.  Nothing is written to disk here.
"""

from __future__ import annotations

import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_THIS_DIR)
_MAIN_DIR = os.path.dirname(_TESTS_DIR)
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

from .gallery import clean_word  # noqa: E402
from utils.config import AUD_RUN, PIC_RUN_50EP  # noqa: E402

PROJECT_ROOT = Path(_MAIN_DIR)
SEM_REG_DIR = PROJECT_ROOT / "results" / "semantic_regression"

# Pinned in utils/config.py — do not retype a run id here.
PIC_RUN_DEFAULT = PIC_RUN_50EP
AUD_RUN_DEFAULT = AUD_RUN
SHARED_PATIENTS = ["AA", "AZ", "DR", "LH", "RB", "WBH"]

HELD_OUT_FOLD = -1   # cv_fold value marking the zero-shot (held-out-word) pool


@dataclass
class TrialPredictions:
    """Per-trial decoded embeddings and metadata for one patient/task.

    Arrays are parallel, length T (number of trials):
      pred_emb   (T, D) float64   out-of-fold predicted GloVe embedding
      true_word  (T,)   str       clean stimulus lemma (gallery key)
      true_label (T,)   str       original suffixed label (kept for provenance)
      category   (T,)   str       loose semantic category
      is_held_out(T,)   bool      True if the word was withheld from training
      cv_fold    (T,)   int       fold that produced the prediction (HELD_OUT_FOLD
                                  for zero-shot trials)
    """
    patient: str
    task: str
    pred_emb: np.ndarray
    true_word: np.ndarray
    true_label: np.ndarray
    category: np.ndarray
    is_held_out: np.ndarray
    cv_fold: np.ndarray

    def __len__(self) -> int:
        return len(self.true_word)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "patient": self.patient, "task": self.task,
            "true_word": self.true_word, "true_label": self.true_label,
            "category": self.category, "is_held_out": self.is_held_out,
            "cv_fold": self.cv_fold,
        })
        # predicted embedding columns pe0..pe{D-1}
        for j in range(self.pred_emb.shape[1]):
            df[f"pe{j}"] = self.pred_emb[:, j]
        return df


def _peak_bin(run_folder: str, patient: str, embedding: str) -> int:
    from analysis.cross_task.cross_task_regression import find_peak_bin
    csv_path = SEM_REG_DIR / run_folder / patient / "per_time_scores.csv"
    scores = pd.read_csv(csv_path)
    peak, _ = find_peak_bin(scores, embedding=embedding)
    return int(peak)


def make_predictions(patient: str, run_folder: str, task: str,
                     embedding: str = "GloVe",
                     n_folds: int = 5, held_out_frac: float = 0.3,
                     model: str = "kernel_pls", seed: int = 0
                     ) -> TrialPredictions:
    """Cross-validated per-trial predicted embeddings with a zero-shot split.

    A fraction ``held_out_frac`` of the *unique clean words* is withheld from all
    training folds (their trials are predicted by every fold's model and the
    predictions averaged).  The remaining words' trials go through standard
    ``n_folds`` K-fold CV, each trial predicted out-of-fold.
    """
    from analysis.helpers import load_results_pkl
    from analysis.cross_task.cross_task_regression import get_neural_at_bin, get_trial_metadata
    from analysis.cross_task.cross_task_cotrain import make_model

    peak = _peak_bin(run_folder, patient, embedding)
    d = load_results_pkl(run_folder, patient)
    reg = d["regressors"][embedding]

    X = np.asarray(get_neural_at_bin(reg, peak), dtype=np.float64)   # (T, n_features)
    y = np.asarray(reg.y, dtype=np.float64)                          # (T, D) GloVe
    labels = np.asarray(reg.labels).astype(str)                     # suffixed
    meta = get_trial_metadata(reg)                                  # word/category (suffixed word)
    categories = meta["category"].to_numpy().astype(str)
    del d, reg
    gc.collect()

    T = len(labels)
    if not (X.shape[0] == y.shape[0] == T == len(categories)):
        raise ValueError(
            f"{patient}/{task}: trial-count mismatch "
            f"(X={X.shape[0]}, y={y.shape[0]}, labels={T}, cats={len(categories)}).")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{patient}/{task}: non-finite values in neural features X.")
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{patient}/{task}: non-finite values in GloVe targets y.")

    clean = np.array([clean_word(w) for w in labels])
    unique_clean = np.unique(clean)

    rng = np.random.default_rng(seed)
    n_ho = int(round(len(unique_clean) * held_out_frac))
    held_out_words = set(rng.choice(unique_clean, n_ho, replace=False).tolist()) \
        if n_ho > 0 else set()
    is_held_out = np.array([w in held_out_words for w in clean], dtype=bool)

    ho_idx = np.where(is_held_out)[0]
    reg_idx = np.where(~is_held_out)[0]
    if len(reg_idx) < n_folds:
        raise ValueError(
            f"{patient}/{task}: only {len(reg_idx)} in-vocab trials for "
            f"{n_folds}-fold CV (held_out_frac={held_out_frac} too high?).")

    pred_emb = np.full((T, y.shape[1]), np.nan, dtype=np.float64)
    cv_fold = np.full(T, HELD_OUT_FOLD, dtype=np.int64)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    ho_accum = np.zeros((len(ho_idx), y.shape[1]), dtype=np.float64)
    n_models = 0
    for f, (tr_local, te_local) in enumerate(kf.split(reg_idx)):
        tr = reg_idx[tr_local]
        te = reg_idx[te_local]
        est = make_model(model, len(tr))
        est.fit(X[tr], y[tr])
        pred_emb[te] = est.predict(X[te])
        cv_fold[te] = f
        if len(ho_idx):
            ho_accum += est.predict(X[ho_idx])
        n_models += 1

    if len(ho_idx):
        pred_emb[ho_idx] = ho_accum / n_models   # averaged zero-shot prediction
        cv_fold[ho_idx] = HELD_OUT_FOLD

    if np.any(np.isnan(pred_emb)):
        n_bad = int(np.isnan(pred_emb).any(axis=1).sum())
        raise RuntimeError(
            f"{patient}/{task}: {n_bad} trials received no prediction — CV "
            "coverage is incomplete.")

    return TrialPredictions(
        patient=patient, task=task, pred_emb=pred_emb,
        true_word=clean, true_label=labels, category=categories,
        is_held_out=is_held_out, cv_fold=cv_fold)


def make_predictions_all(patients: Sequence[str], run_folder: str, task: str,
                         embedding: str = "GloVe", n_folds: int = 5,
                         held_out_frac: float = 0.3, model: str = "kernel_pls",
                         seed: int = 0) -> List[TrialPredictions]:
    """Run :func:`make_predictions` for each patient, keeping per-patient
    structure (never pooled — group inference happens later)."""
    out: List[TrialPredictions] = []
    for pat in patients:
        print(f"  [predict] {pat} ({task}) ...", flush=True)
        tp = make_predictions(pat, run_folder, task, embedding=embedding,
                              n_folds=n_folds, held_out_frac=held_out_frac,
                              model=model, seed=seed)
        n_ho = int(tp.is_held_out.sum())
        print(f"    T={len(tp)}  held_out_trials={n_ho}  "
              f"unique_words={len(np.unique(tp.true_word))}", flush=True)
        out.append(tp)
        gc.collect()
    return out
