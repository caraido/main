# -*- coding: utf-8 -*-
"""
tests/cross_task/cross_task_cotrain.py
======================================
Cross-task CO-TRAINING: treat picture naming and auditory naming of the same
patient as one task and train a single semantic regressor (neural HGA -> GloVe)
on the pooled trials, then ask three questions:

  (1) Is the semantic representation the SAME across tasks?
        -> compare cross-task decoding (train A, test B) to within-task, and
           compare pooled-trained to task-specific decoders; plus RSA of the
           per-word neural geometry between tasks.
  (2) Are there brain regions encoding an AMODAL semantic representation?
        -> per-electrode semantic encoding (RSA vs GloVe) computed in EACH task
           independently, plus cross-task tuning consistency; electrodes high in
           both tasks and consistent across them are amodal candidates.
  (3) Can ONE decoder serve both tasks?
        -> evaluate the pooled-trained decoder on held-out trials of each task
           and compare to the task-specific decoders.

Design choices (confirmed with the user):
  - Per-task peak bin alignment: each task uses its own loose-category peak bin
    (mirrors cross_task_regression / cross_task_transfer).  Channels are the
    intersection of the two tasks' channel names, arranged identically in both.
  - Default model = kernel PLS (Nystroem-RBF + PLSRegression) regressing to
    GloVe.  Other models (plain PLS, ridge, kernel ridge) are pluggable via a
    registry and can be run separately (--models ridge) or in bulk
    (--models kernel_pls ridge krr).
  - Class imbalance (picture >> auditory) is handled by a --balance switch
    applied to the POOLED training set: none | downsample | upsample.

Evaluation conditions (per bootstrap, shared test sets):
  within_pic, within_aud      — train & test the same task (ceiling per task)
  cross_p2a, cross_a2p        — train one task, test the other  (Q1)
  pooled_pic, pooled_aud      — train pooled, test each task     (Q3)
Each is scored with word_bal_acc, cat_indep_bal_acc, cosine, and split into
seen vs unseen (zero-shot) words relative to that model's training vocabulary.

Outputs: each invocation creates its own run folder so prior runs are never
overwritten —
  OUT_ROOT/<run>/                      run = <timestamp>_<models>_balance-<b>_<N>boot[...]
    run_metadata.json                  — full parameter set of this run
    cotrain_conditions_summary.csv     — cross-patient aggregate (mean/sem)
    cotrain_rsa_summary.csv            — cross-patient RSA summary
    <patient>/
      cotrain_conditions_<patient>.csv — per-bootstrap rows (model x condition)
      cotrain_rsa_<patient>.csv        — per-patient RSA summary
      cotrain_electrodes_<patient>.csv — per-electrode amodal scores
      cotrain_<patient>_bars.png       — condition bar chart (default model)
      cotrain_<patient>_electrodes.png — rsa_pic vs rsa_aud scatter

Usage:
    python -m main.analysis.cross_task.cross_task_cotrain
    python -m main.analysis.cross_task.cross_task_cotrain --patient RB
    python -m main.analysis.cross_task.cross_task_cotrain --models kernel_pls ridge krr
    python -m main.analysis.cross_task.cross_task_cotrain --balance downsample --n-bootstrap 50
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_THIS_DIR)
_MAIN_DIR = os.path.dirname(_TESTS_DIR)
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

# Light import (numpy + sklearn only); the heavy data stack is imported lazily
# inside the loading functions so the compute/analysis functions stay testable.
from utils.retrieval import build_retrieval_db, compute_retrieval_metrics  # noqa: E402
from utils.paths import results_dir  # noqa: E402
from utils.config import AUD_RUN, PIC_RUN_50EP  # noqa: E402

# ── Config (mirrors cross_task_regression) ────────────────────────────────
PROJECT_ROOT = Path(_MAIN_DIR)
SEM_REG_DIR = PROJECT_ROOT / "results" / "semantic_regression"
OUT_ROOT = results_dir("cross_task_cotrain", create=False)

# Pinned in utils/config.py; these names are kept because several modules import
# them. AUD_RUN moved to the group-warped 100-epoch run on 2026-07-27 while the
# picture side is still the 50-epoch run — see the epoch-asymmetry note in
# .claude/open-questions.md before reading a cross-task null too closely.
PIC_RUN_DEFAULT = PIC_RUN_50EP
AUD_RUN_DEFAULT = AUD_RUN
SHARED_PATIENTS = ["AA", "AZ", "DR", "LH", "RB", "WBH"]
PEAK_EMBEDDING = "GloVe"

DEFAULT_N_BOOTSTRAP = 50
DEFAULT_TEST_FRAC = 0.3
DEFAULT_ZERO_SHOT_FRAC = 0.3     # fraction of shared words held fully out (zero-shot)
DEFAULT_BALANCE = "none"         # none | downsample | upsample
N_PLS_COMPONENTS = 10
NYSTROEM_N_COMPONENTS = 100

CONDITIONS = ["within_pic", "within_aud", "cross_p2a", "cross_a2p",
              "pooled_pic", "pooled_aud"]


# ══════════════════════════════════════════════════════════════════════════
# Model registry  — name -> builder(n_samples, **hp) -> sklearn estimator
# ══════════════════════════════════════════════════════════════════════════

def _build_kernel_pls(n_samples: int, n_components: int = N_PLS_COMPONENTS,
                      nystroem_components: int = NYSTROEM_N_COMPONENTS,
                      gamma: Optional[float] = None,
                      random_state: int = 0) -> Pipeline:
    n_nys = max(2, min(nystroem_components, n_samples - 1))
    n_pls = max(1, min(n_components, n_nys))
    return Pipeline([
        ("nys", Nystroem(kernel="rbf", n_components=n_nys, gamma=gamma,
                         random_state=random_state)),
        ("pls", PLSRegression(n_components=n_pls, scale=False)),
    ])


def _build_pls(n_samples: int, n_components: int = N_PLS_COMPONENTS) -> PLSRegression:
    return PLSRegression(n_components=max(1, min(n_components, n_samples - 1)),
                         scale=False)


def _build_ridge(n_samples: int, alpha: float = 1.0) -> Ridge:
    return Ridge(alpha=alpha)


def _build_krr(n_samples: int, alpha: float = 1.0,
               gamma: Optional[float] = None) -> KernelRidge:
    return KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)


MODEL_REGISTRY: Dict[str, Callable[..., object]] = {
    "kernel_pls": _build_kernel_pls,
    "pls": _build_pls,
    "ridge": _build_ridge,
    "krr": _build_krr,
}
DEFAULT_MODEL = "kernel_pls"


def make_model(name: str, n_samples: int, hp: Optional[dict] = None):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Options: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](n_samples, **(hp or {}))


# ══════════════════════════════════════════════════════════════════════════
# Data loading  (heavy deps imported lazily)
# ══════════════════════════════════════════════════════════════════════════

def _load_per_time_scores(run_folder: str, patient: str) -> pd.DataFrame:
    return pd.read_csv(SEM_REG_DIR / run_folder / patient / "per_time_scores.csv")


def _task_arrays(reg, chan_idx: np.ndarray, peak_bin: int,
                 chan_names_common: np.ndarray) -> dict:
    """Build X (lagged HGA at peak bin, common channels), y, words, cats."""
    from analysis.cross_task.cross_task_regression import (
        build_X_at_bin_with_channel_subset,
    )
    words = np.asarray(reg.labels).astype(str)
    word_to_index = {str(k): v for k, v in reg.word_to_index.items()}
    cats = np.array([
        str(reg.index_to_category[
            reg.word_index_to_category_index[word_to_index[str(w)]]
        ])
        for w in words
    ])
    X = build_X_at_bin_with_channel_subset(reg, peak_bin, chan_idx)
    y = np.asarray(reg.y)
    n_channels = int(len(chan_idx))
    return {
        "X": X, "y": y, "words": words, "cats": cats,
        "n_channels": n_channels, "n_hist": X.shape[1] // n_channels,
        "chan_names": np.asarray(chan_names_common).astype(str),
    }


def load_patient(patient: str, pic_run: str, aud_run: str,
                 embedding: str = PEAK_EMBEDDING) -> Tuple[dict, dict]:
    """Load both tasks for a patient, intersect channels, build peak-bin arrays.

    Returns (pic, aud) dicts.  Both share the same channel set (intersection of
    names, arranged identically) so their feature matrices are directly poolable
    iff the two runs use the same history window.
    """
    from analysis.helpers import load_results_pkl
    from analysis.cross_task.cross_task_regression import (
        find_peak_bin, _common_channels_from_names,
    )

    pic_scores = _load_per_time_scores(pic_run, patient)
    aud_scores = _load_per_time_scores(aud_run, patient)
    pic_peak, _ = find_peak_bin(pic_scores, embedding=embedding)
    aud_peak, _ = find_peak_bin(aud_scores, embedding=embedding)

    dp = load_results_pkl(pic_run, patient)
    pic_reg = dp["regressors"][embedding]
    pic_names = np.asarray(dp.get("clean_channel_names", [])).astype(str)
    da = load_results_pkl(aud_run, patient)
    aud_reg = da["regressors"][embedding]
    aud_names = np.asarray(da.get("clean_channel_names", [])).astype(str)

    if len(pic_names) and len(aud_names):
        idx_pic, idx_aud, common = _common_channels_from_names(pic_names, aud_names)
        if len(common) == 0:
            n = min(pic_reg.data.shape[2], aud_reg.data.shape[2])
            idx_pic = idx_aud = np.arange(n, dtype=np.int64)
            common = np.array([f"ch{i}" for i in range(n)])
    else:
        n = min(pic_reg.data.shape[2], aud_reg.data.shape[2])
        idx_pic = idx_aud = np.arange(n, dtype=np.int64)
        common = np.array([f"ch{i}" for i in range(n)])

    pic = _task_arrays(pic_reg, idx_pic, pic_peak, common)
    aud = _task_arrays(aud_reg, idx_aud, aud_peak, common)
    del dp, da, pic_reg, aud_reg
    gc.collect()

    if pic["X"].shape[1] != aud["X"].shape[1]:
        raise ValueError(
            f"{patient}: pooled feature dim mismatch "
            f"(pic={pic['X'].shape[1]}, aud={aud['X'].shape[1]}). The two runs "
            "use different history windows; re-export with matching n_bins_history "
            "before co-training.")
    return pic, aud


# ══════════════════════════════════════════════════════════════════════════
# Splitting / balancing
# ══════════════════════════════════════════════════════════════════════════

def _stratified_word_split(words: np.ndarray, exclude_words: set,
                           test_frac: float, rng: np.random.Generator
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-word trial split. Words in *exclude_words* go entirely to test."""
    n = len(words)
    train = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)
    for w in np.unique(words):
        idx = np.where(words == w)[0].copy()
        rng.shuffle(idx)
        if w in exclude_words:
            test[idx] = True
            continue
        n_te = max(1, int(round(len(idx) * test_frac))) if len(idx) >= 2 else 0
        test[idx[:n_te]] = True
        train[idx[n_te:]] = True
    return np.where(train)[0], np.where(test)[0]


def _balance_pooled(idx_pic: np.ndarray, idx_aud: np.ndarray, mode: str,
                    rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Balance the two tasks' training-trial counts. Returns (pic_idx, aud_idx)
    possibly resampled. Indices index into each task's own arrays."""
    if mode == "none" or len(idx_pic) == 0 or len(idx_aud) == 0:
        return idx_pic, idx_aud
    n_p, n_a = len(idx_pic), len(idx_aud)
    if mode == "downsample":
        m = min(n_p, n_a)
        ip = rng.choice(idx_pic, m, replace=False) if n_p > m else idx_pic
        ia = rng.choice(idx_aud, m, replace=False) if n_a > m else idx_aud
        return ip, ia
    if mode == "upsample":
        m = max(n_p, n_a)
        ip = rng.choice(idx_pic, m, replace=True) if n_p < m else idx_pic
        ia = rng.choice(idx_aud, m, replace=True) if n_a < m else idx_aud
        return ip, ia
    raise ValueError(f"Unknown balance mode '{mode}'")


# ══════════════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════════════

def _build_db(task: dict) -> tuple:
    """Retrieval database (per-word mean GloVe + category map) for a task."""
    return build_retrieval_db(task["y"], task["words"], task["cats"])


def _norm(w) -> str:
    return str(w).strip().lower()


def _score(Y_pred: np.ndarray, words_te: np.ndarray, cats_te: np.ndarray,
           db: tuple, train_vocab: set) -> dict:
    """word/cat/cosine metrics, split into seen vs unseen (zero-shot) relative
    to the fitted model's training vocabulary."""
    m = compute_retrieval_metrics(Y_pred, words_te, cats_te, *db)
    out = {"word_bal_acc": m["word_bal_acc"],
           "cat_indep_bal_acc": m["cat_indep_bal_acc"],
           "cosine_mean": m["cosine_mean"]}
    tv = {_norm(w) for w in train_vocab}
    seen_mask = np.array([_norm(w) in tv for w in words_te], dtype=bool)
    for label, mask in [("seen", seen_mask), ("unseen", ~seen_mask)]:
        if mask.any():
            ms = compute_retrieval_metrics(Y_pred[mask], words_te[mask],
                                           cats_te[mask], *db)
            out[f"word_acc_{label}"] = ms["word_bal_acc"]
            out[f"cat_acc_{label}"] = ms["cat_indep_bal_acc"]
            out[f"cosine_{label}"] = ms["cosine_mean"]
        else:
            out[f"word_acc_{label}"] = np.nan
            out[f"cat_acc_{label}"] = np.nan
            out[f"cosine_{label}"] = np.nan
    return out


# ══════════════════════════════════════════════════════════════════════════
# Bootstrap over conditions
# ══════════════════════════════════════════════════════════════════════════

def _fit_predict(model_name: str, X_tr: np.ndarray, y_tr: np.ndarray,
                 X_te: np.ndarray, hp: Optional[dict]) -> np.ndarray:
    model = make_model(model_name, X_tr.shape[0], hp)
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def run_conditions(pic: dict, aud: dict, models: Sequence[str],
                   n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
                   test_frac: float = DEFAULT_TEST_FRAC,
                   zero_shot_frac: float = DEFAULT_ZERO_SHOT_FRAC,
                   balance: str = DEFAULT_BALANCE,
                   model_hp: Optional[Dict[str, dict]] = None,
                   rng_seed: int = 42) -> List[dict]:
    """Run the 6 evaluation conditions for each model across bootstraps."""
    rng = np.random.default_rng(rng_seed)
    model_hp = model_hp or {}
    db_pic = _build_db(pic)
    db_aud = _build_db(aud)
    shared_vocab = np.array(sorted(set(pic["words"]) & set(aud["words"])))
    rows: List[dict] = []

    for boot in range(n_bootstrap):
        # words held fully out of training (zero-shot), from the shared vocab
        n_zs = int(round(len(shared_vocab) * zero_shot_frac))
        unseen = set(rng.choice(shared_vocab, n_zs, replace=False).tolist()) \
            if n_zs > 0 else set()

        p_tr, p_te = _stratified_word_split(pic["words"], unseen, test_frac, rng)
        a_tr, a_te = _stratified_word_split(aud["words"], unseen, test_frac, rng)
        if min(len(p_tr), len(a_tr), len(p_te), len(a_te)) < 3:
            continue

        # pooled training set (with imbalance handling)
        bp_tr, ba_tr = _balance_pooled(p_tr, a_tr, balance, rng)
        X_pool = np.vstack([pic["X"][bp_tr], aud["X"][ba_tr]])
        y_pool = np.vstack([pic["y"][bp_tr], aud["y"][ba_tr]])
        vocab_pic = set(pic["words"][p_tr]); vocab_aud = set(aud["words"][a_tr])
        vocab_pool = vocab_pic | vocab_aud

        for mdl in models:
            hp = model_hp.get(mdl)
            try:
                specs = [
                    ("within_pic", pic["X"][p_tr], pic["y"][p_tr], pic["X"][p_te],
                     pic["words"][p_te], pic["cats"][p_te], db_pic, vocab_pic),
                    ("within_aud", aud["X"][a_tr], aud["y"][a_tr], aud["X"][a_te],
                     aud["words"][a_te], aud["cats"][a_te], db_aud, vocab_aud),
                    ("cross_p2a", pic["X"][p_tr], pic["y"][p_tr], aud["X"][a_te],
                     aud["words"][a_te], aud["cats"][a_te], db_aud, vocab_pic),
                    ("cross_a2p", aud["X"][a_tr], aud["y"][a_tr], pic["X"][p_te],
                     pic["words"][p_te], pic["cats"][p_te], db_pic, vocab_aud),
                    ("pooled_pic", X_pool, y_pool, pic["X"][p_te],
                     pic["words"][p_te], pic["cats"][p_te], db_pic, vocab_pool),
                    ("pooled_aud", X_pool, y_pool, aud["X"][a_te],
                     aud["words"][a_te], aud["cats"][a_te], db_aud, vocab_pool),
                ]
                for (cond, Xtr, ytr, Xte, wte, cte, db, tv) in specs:
                    Yp = _fit_predict(mdl, Xtr, ytr, Xte, hp)
                    sc = _score(Yp, wte, cte, db, tv)
                    rows.append({"model": mdl, "condition": cond,
                                 "bootstrap_id": boot, "n_train": int(Xtr.shape[0]),
                                 "n_test": int(Xte.shape[0]), **sc})
            except Exception as exc:  # keep the loop alive
                print(f"    [{mdl}] boot={boot}: {type(exc).__name__}: {exc}")
    return rows


# ══════════════════════════════════════════════════════════════════════════
# RSA: cross-task geometry (Q1)
# ══════════════════════════════════════════════════════════════════════════

def _rdm(M: np.ndarray) -> np.ndarray:
    """Cosine-distance RDM (condensed upper triangle) for rows of M."""
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    S = Mn @ Mn.T
    iu = np.triu_indices(len(M), k=1)
    return (1.0 - S)[iu]


def _word_means(task: dict, vocab: np.ndarray, channels: Optional[np.ndarray] = None
                ) -> np.ndarray:
    """Per-word mean feature vector. If *channels* given, restrict to those
    channel columns (across all history bins)."""
    n_ch, n_hist = task["n_channels"], task["n_hist"]
    if channels is not None:
        cols = np.concatenate([channels + b * n_ch for b in range(n_hist)])
        X = task["X"][:, cols]
    else:
        X = task["X"]
    rows = []
    for w in vocab:
        m = (task["words"] == w)
        rows.append(X[m].mean(axis=0))
    return np.stack(rows)


def rsa_analysis(pic: dict, aud: dict) -> dict:
    """RDM correlations between tasks and against GloVe over shared words."""
    from scipy.stats import spearmanr
    vocab = np.array(sorted(set(pic["words"]) & set(aud["words"])))
    if len(vocab) < 4:
        return {"n_shared_words": int(len(vocab))}
    H_pic = _word_means(pic, vocab)
    H_aud = _word_means(aud, vocab)
    G = np.stack([pic["y"][pic["words"] == w].mean(0) for w in vocab])  # GloVe
    r_pa = spearmanr(_rdm(H_pic), _rdm(H_aud)).correlation
    r_pg = spearmanr(_rdm(H_pic), _rdm(G)).correlation
    r_ag = spearmanr(_rdm(H_aud), _rdm(G)).correlation
    return {"n_shared_words": int(len(vocab)),
            "rdm_pic_vs_aud": float(r_pa),
            "rdm_pic_vs_glove": float(r_pg),
            "rdm_aud_vs_glove": float(r_ag)}


# ══════════════════════════════════════════════════════════════════════════
# Amodal electrode localization (Q2)
# ══════════════════════════════════════════════════════════════════════════

def electrode_amodal_scores(pic: dict, aud: dict,
                            n_perm: int = 0,
                            rng_seed: int = 0) -> pd.DataFrame:
    """Per-electrode semantic encoding (RSA vs GloVe) in each task + cross-task
    tuning consistency.  Amodal candidates score high in BOTH tasks and have
    consistent tuning across them.

    If *n_perm* > 0, a word-label permutation null gives p-values per task.
    """
    from scipy.stats import spearmanr
    rng = np.random.default_rng(rng_seed)
    vocab = np.array(sorted(set(pic["words"]) & set(aud["words"])))
    n_ch = pic["n_channels"]
    G = np.stack([pic["y"][pic["words"] == w].mean(0) for w in vocab])
    rdm_g = _rdm(G)

    recs = []
    for c in range(n_ch):
        Hp = _word_means(pic, vocab, channels=np.array([c]))  # (n_words, n_hist_pic)
        Ha = _word_means(aud, vocab, channels=np.array([c]))  # (n_words, n_hist_aud)
        rsa_p = spearmanr(_rdm(Hp), rdm_g).correlation
        rsa_a = spearmanr(_rdm(Ha), rdm_g).correlation
        # cross-task tuning consistency: per-word activity (mean over history)
        tp = Hp.mean(axis=1); ta = Ha.mean(axis=1)
        cons = spearmanr(tp, ta).correlation
        rec = {"channel": pic["chan_names"][c] if c < len(pic["chan_names"]) else f"ch{c}",
               "rsa_pic": float(rsa_p), "rsa_aud": float(rsa_a),
               "cross_task_consistency": float(cons),
               "amodal_score": float(min(rsa_p, rsa_a) * max(0.0, cons))}
        if n_perm > 0:
            null_p = np.empty(n_perm); null_a = np.empty(n_perm)
            for i in range(n_perm):
                perm = rng.permutation(len(vocab))
                null_p[i] = spearmanr(_rdm(Hp), _rdm(G[perm])).correlation
                null_a[i] = spearmanr(_rdm(Ha), _rdm(G[perm])).correlation
            rec["p_pic"] = float((np.sum(null_p >= rsa_p) + 1) / (n_perm + 1))
            rec["p_aud"] = float((np.sum(null_a >= rsa_a) + 1) / (n_perm + 1))
        recs.append(rec)
    df = pd.DataFrame(recs).sort_values("amodal_score", ascending=False)
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════

def _plot_conditions(df: pd.DataFrame, patient: str, model: str, out_dir: Path):
    sub = df[df["model"] == model]
    metrics = ["word_bal_acc", "cat_indep_bal_acc", "cosine_mean"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, met in zip(axes, metrics):
        means = [sub[sub["condition"] == c][met].mean() for c in CONDITIONS]
        sems = [sub[sub["condition"] == c][met].sem() for c in CONDITIONS]
        xs = np.arange(len(CONDITIONS))
        ax.bar(xs, means, yerr=sems, capsize=3, color="#4c72b0", alpha=0.85)
        ax.set_xticks(xs); ax.set_xticklabels(CONDITIONS, rotation=40, ha="right", fontsize=8)
        ax.set_title(met); ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{patient} · co-training conditions ({model})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"cotrain_{patient}_bars.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_electrodes(df: pd.DataFrame, patient: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(df["rsa_pic"], df["rsa_aud"], c=df["cross_task_consistency"],
               cmap="viridis", s=24, alpha=0.85)
    lim = [min(df["rsa_pic"].min(), df["rsa_aud"].min(), 0) - 0.02,
           max(df["rsa_pic"].max(), df["rsa_aud"].max()) + 0.02]
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("RSA vs GloVe (picture)"); ax.set_ylabel("RSA vs GloVe (auditory)")
    ax.set_title(f"{patient} · per-electrode amodal encoding")
    cb = fig.colorbar(ax.collections[0], ax=ax); cb.set_label("cross-task consistency")
    fig.tight_layout()
    fig.savefig(out_dir / f"cotrain_{patient}_electrodes.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Per-patient driver
# ══════════════════════════════════════════════════════════════════════════

def analyze_patient(patient: str, pic_run: str, aud_run: str,
                    embedding: str = PEAK_EMBEDDING,
                    models: Sequence[str] = (DEFAULT_MODEL,),
                    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
                    test_frac: float = DEFAULT_TEST_FRAC,
                    zero_shot_frac: float = DEFAULT_ZERO_SHOT_FRAC,
                    balance: str = DEFAULT_BALANCE,
                    n_perm: int = 0,
                    out_dir: Optional[Path] = None,
                    save_figs: bool = True,
                    rng_seed: int = 42) -> Dict[str, pd.DataFrame]:
    print(f"\n=== {patient} : {embedding} (co-train) ===", flush=True)
    out_dir = out_dir or (OUT_ROOT / patient)
    out_dir.mkdir(parents=True, exist_ok=True)

    pic, aud = load_patient(patient, pic_run, aud_run, embedding)
    print(f"  pic trials={len(pic['words'])}  aud trials={len(aud['words'])}  "
          f"common_ch={pic['n_channels']}  feat_dim={pic['X'].shape[1]}  "
          f"shared_vocab={len(set(pic['words']) & set(aud['words']))}", flush=True)

    # conditions
    cond_rows = run_conditions(pic, aud, models, n_bootstrap=n_bootstrap,
                               test_frac=test_frac, zero_shot_frac=zero_shot_frac,
                               balance=balance, rng_seed=rng_seed)
    cond_df = pd.DataFrame(cond_rows)
    cond_df["patient"] = patient
    cond_df.to_csv(out_dir / f"cotrain_conditions_{patient}.csv", index=False)

    # RSA (Q1)
    rsa = rsa_analysis(pic, aud); rsa["patient"] = patient
    rsa_df = pd.DataFrame([rsa])
    rsa_df.to_csv(out_dir / f"cotrain_rsa_{patient}.csv", index=False)

    # electrodes (Q2)
    elec_df = electrode_amodal_scores(pic, aud, n_perm=n_perm, rng_seed=rng_seed)
    elec_df["patient"] = patient
    elec_df.to_csv(out_dir / f"cotrain_electrodes_{patient}.csv", index=False)

    if save_figs and not cond_df.empty:
        default = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]
        _plot_conditions(cond_df, patient, default, out_dir)
        _plot_electrodes(elec_df, patient, out_dir)

    print(f"  saved CSVs + figures -> {out_dir}", flush=True)
    return {"conditions": cond_df, "rsa": rsa_df, "electrodes": elec_df}


# ══════════════════════════════════════════════════════════════════════════
# Run directory + metadata
# ══════════════════════════════════════════════════════════════════════════

def _run_dir_name(args, timestamp: str) -> str:
    """Folder name encoding the run's key parameters.

    e.g. ``2026-06-30_14-22-01_kernel_pls_balance-none_50boot`` (single-patient
    runs get the patient appended).  Timestamp prefix keeps runs lexically
    sortable by recency.
    """
    parts = [timestamp, "-".join(args.models),
             f"balance-{args.balance}", f"{args.n_bootstrap}boot"]
    if args.patient:
        parts.append(args.patient)
    if args.n_perm:
        parts.append(f"{args.n_perm}perm")
    return "_".join(parts)


def _write_metadata(run_dir: Path, args, patients: List[str], timestamp: str) -> None:
    """Dump the full parameter set of this run to ``run_metadata.json``."""
    meta = {
        "timestamp": timestamp,
        "script": "cross_task_cotrain.py",
        "command": "python -m main.analysis.cross_task.cross_task_cotrain " + " ".join(sys.argv[1:]),
        "patients": patients,
        "pic_run": args.pic_run,
        "aud_run": args.aud_run,
        "embedding": args.embedding,
        "models": args.models,
        "n_bootstrap": args.n_bootstrap,
        "test_frac": args.test_frac,
        "zero_shot_frac": args.zero_shot_frac,
        "balance": args.balance,
        "n_perm": args.n_perm,
        "seed": args.seed,
        "save_figs": not args.no_figs,
        "conditions": CONDITIONS,
        "model_hyperparameters": {
            "n_pls_components": N_PLS_COMPONENTS,
            "nystroem_n_components": NYSTROEM_N_COMPONENTS,
        },
    }
    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def main() -> int:
    p = argparse.ArgumentParser(description="Cross-task co-training: pic + aud naming")
    p.add_argument("--patient", default=None, help="One patient (default: all)")
    p.add_argument("--pic-run", default=PIC_RUN_DEFAULT)
    p.add_argument("--aud-run", default=AUD_RUN_DEFAULT)
    p.add_argument("--embedding", default=PEAK_EMBEDDING)
    p.add_argument("--models", nargs="+", default=[DEFAULT_MODEL],
                   choices=list(MODEL_REGISTRY),
                   help="One or more models to run (bulk).")
    p.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    p.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    p.add_argument("--zero-shot-frac", type=float, default=DEFAULT_ZERO_SHOT_FRAC)
    p.add_argument("--balance", default=DEFAULT_BALANCE,
                   choices=["none", "downsample", "upsample"])
    p.add_argument("--n-perm", type=int, default=0,
                   help="Permutations for per-electrode p-values (0 = skip).")
    p.add_argument("--no-figs", action="store_true")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Each invocation gets its own run folder so previous runs are never
    # overwritten.  --out-dir overrides the parent under which runs are grouped.
    out_parent = Path(args.out_dir) if args.out_dir else OUT_ROOT
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    patients = [args.patient] if args.patient else SHARED_PATIENTS
    out_root = out_parent / _run_dir_name(args, timestamp)
    out_root.mkdir(parents=True, exist_ok=True)
    _write_metadata(out_root, args, patients, timestamp)
    print(f"Run dir: {out_root}", flush=True)

    cond_all, rsa_all, elec_all = [], [], []
    for pat in patients:
        try:
            res = analyze_patient(
                pat, args.pic_run, args.aud_run, embedding=args.embedding,
                models=args.models, n_bootstrap=args.n_bootstrap,
                test_frac=args.test_frac, zero_shot_frac=args.zero_shot_frac,
                balance=args.balance, n_perm=args.n_perm,
                out_dir=out_root / pat, save_figs=not args.no_figs,
                rng_seed=args.seed)
            cond_all.append(res["conditions"]); rsa_all.append(res["rsa"])
            elec_all.append(res["electrodes"])
        except Exception:
            import traceback; traceback.print_exc()
            print(f"  ERROR for {pat}", flush=True)
        finally:
            gc.collect()

    if cond_all:
        cond = pd.concat(cond_all, ignore_index=True)
        agg = (cond.groupby(["patient", "model", "condition"])
               [["word_bal_acc", "cat_indep_bal_acc", "cosine_mean",
                 "word_acc_seen", "word_acc_unseen"]]
               .agg(["mean", "sem"]))
        agg.columns = ["_".join(c) for c in agg.columns]
        agg.reset_index().to_csv(out_root / "cotrain_conditions_summary.csv", index=False)
        pd.concat(rsa_all, ignore_index=True).to_csv(
            out_root / "cotrain_rsa_summary.csv", index=False)
        print(f"\nWrote summaries -> {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
