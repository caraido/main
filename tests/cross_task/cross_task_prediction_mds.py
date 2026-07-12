# -*- coding: utf-8 -*-
"""
tests/cross_task/cross_task_prediction_mds.py
=============================================
Semantic-organization map of the TWO SEPARATE per-task decoders.

Motivation panel for the cross-task story: *before* co-training, do the
picture-naming and auditory-naming decoders — trained entirely independently —
already organize their neural data the same way semantically?  If so, a single
co-trained decoder is a natural next step.

Method (no co-trainer, no balancing):
  1. load_patient -> peak-bin lagged HGA (X), true GloVe (y), words, categories,
     for each task, on the shared channel set (each task at its OWN peak bin).
  2. For EACH task separately, fit a per-task kernel-PLS (Nystroem-RBF + PLS,
     identical architecture to the shipped runs) with WORD-grouped K-fold CV so
     every trial receives one out-of-fold (held-out) predicted GloVe vector
     (300-D).  No word ever appears in both train and test of the same fold.
  3. Stack both tasks' predicted trial vectors, take cosine distances, and run
     metric MDS -> 2D.  Both tasks live in ONE shared 2D space; we split them
     into two subplots so co-location of a category across tasks is visible.
  4. Quantify with the cross-task category-centroid alignment: cosine between the
     picture- and auditory-predicted centroid of each shared category, tested
     against a category-label shuffle.

Outputs (a fresh timestamped run dir under tests/results/cross_task_cotrain/):
  <run>/prediction_mds_<patient>.csv        per-trial (task, word, category, mds1, mds2)
  <run>/<patient>/prediction_mds_<patient>.png
  <run>/prediction_mds_alignment_summary.csv   one row per patient
  <run>/run_metadata.json

Usage (Speech conda env, from project root d:/.../Speech):
    python -m main.tests.cross_task.cross_task_prediction_mds
    python -m main.tests.cross_task.cross_task_prediction_mds --patient RB
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import GroupKFold, KFold

warnings.filterwarnings("ignore")

# ── Path setup (mirror cross_task_cotrain) ─────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_THIS_DIR)
_MAIN_DIR = os.path.dirname(_TESTS_DIR)
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

from tests.cross_task.cross_task_cotrain import (  # noqa: E402
    load_patient, make_model, SHARED_PATIENTS,
    PIC_RUN_DEFAULT, AUD_RUN_DEFAULT, OUT_ROOT,
)

TASKS = ("picture", "auditory")
DEFAULT_N_FOLDS = 5
DEFAULT_N_SHUFFLE = 500
DEFAULT_SEED = 42

# One run truncated a category label at 10 chars ("object/too" for "object/tool");
# normalise so the same category is never split across two colours.
_CATEGORY_FIX = {"object/too": "object/tool"}


def _fix_cats(cats: np.ndarray) -> np.ndarray:
    return np.array([_CATEGORY_FIX.get(str(c), str(c)) for c in cats])


# ══════════════════════════════════════════════════════════════════════════
# Out-of-fold predicted embeddings from a per-task decoder
# ══════════════════════════════════════════════════════════════════════════

def oof_predictions(X: np.ndarray, y: np.ndarray, words: np.ndarray,
                    n_folds: int, seed: int,
                    group_by_word: bool = False) -> np.ndarray:
    """Out-of-fold predictions of a per-task kernel-PLS — every trial is held out
    exactly once and predicted from a model that never saw it.

    Default is trial-level K-fold (``group_by_word=False``): a held-out *trial*,
    matching what the separate per-task runs report; words may recur across
    train/test (auditory naming has few repeats, so word-grouping would force it
    into a pure zero-shot regime and swamp the predictions in noise).  With
    ``group_by_word=True`` all trials of a word share a fold (stricter zero-shot).
    Returns predicted GloVe (n_trials, D).
    """
    n = X.shape[0]
    pred = np.full_like(y, np.nan, dtype=float)
    if group_by_word:
        uniq = np.unique(words)
        k = int(max(2, min(n_folds, len(uniq))))
        rng = np.random.default_rng(seed)
        order = {w: i for i, w in enumerate(rng.permutation(uniq))}
        grp = np.array([order[w] for w in words], dtype=int)
        splitter = GroupKFold(n_splits=k).split(X, y, groups=grp)
    else:
        k = int(max(2, min(n_folds, n)))
        splitter = KFold(n_splits=k, shuffle=True, random_state=seed).split(X)
    for tr, te in splitter:
        model = make_model("kernel_pls", len(tr))
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
    return pred


# ══════════════════════════════════════════════════════════════════════════
# Cross-task category-centroid alignment (shared-organization statistic)
# ══════════════════════════════════════════════════════════════════════════

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _category_centroids(pred: np.ndarray, cats: np.ndarray,
                        keep: List[str]) -> Dict[str, np.ndarray]:
    return {c: pred[cats == c].mean(axis=0) for c in keep}


def category_centroid_alignment(pic_pred: np.ndarray, pic_cats: np.ndarray,
                                aud_pred: np.ndarray, aud_cats: np.ndarray,
                                n_shuffle: int, seed: int
                                ) -> Tuple[float, float, int, Dict[str, float]]:
    """Mean cosine between the picture- and auditory-predicted *displacement* of
    each shared category from its task mean, plus a category-shuffle p-value.

    Each task's predictions are mean-centered first (kernel-PLS shrinks all
    predictions toward the GloVe mean, so raw category centroids are dominated by
    that common offset and look alike even under shuffling).  After centering, a
    category centroid is its direction away from the mean; cosine>0 across tasks
    means both decoders push that category the same way in GloVe space.  Higher
    (and shuffle-significant) => the two tasks organize categories alike."""
    shared = sorted(set(pic_cats) & set(aud_cats))
    if len(shared) < 2:
        return float("nan"), float("nan"), len(shared), {}
    picc = pic_pred - pic_pred.mean(axis=0, keepdims=True)
    audc = aud_pred - aud_pred.mean(axis=0, keepdims=True)
    pic_c = _category_centroids(picc, pic_cats, shared)
    aud_c = _category_centroids(audc, aud_cats, shared)
    per_cat = {c: float(np.dot(_unit(pic_c[c]), _unit(aud_c[c]))) for c in shared}
    obs = float(np.mean(list(per_cat.values())))

    rng = np.random.default_rng(seed + 7)
    ge = 1  # +1 (observed counts) for a conservative permutation p-value
    for _ in range(n_shuffle):
        perm = rng.permutation(aud_cats)
        a_c = _category_centroids(audc, perm, shared)
        try:
            m = np.mean([np.dot(_unit(pic_c[c]), _unit(a_c[c])) for c in shared])
        except Exception:
            continue
        if m >= obs:
            ge += 1
    p = ge / (n_shuffle + 1)
    return obs, float(p), len(shared), per_cat


# ══════════════════════════════════════════════════════════════════════════
# Per-patient analysis
# ══════════════════════════════════════════════════════════════════════════

def analyze_patient(patient: str, pic_run: str, aud_run: str,
                    n_folds: int, n_shuffle: int, seed: int,
                    group_by_word: bool = False
                    ) -> Tuple[pd.DataFrame, dict]:
    pic, aud = load_patient(patient, pic_run, aud_run)
    pic["cats"] = _fix_cats(pic["cats"])
    aud["cats"] = _fix_cats(aud["cats"])

    pic_pred = oof_predictions(pic["X"], pic["y"], pic["words"], n_folds, seed,
                               group_by_word)
    aud_pred = oof_predictions(aud["X"], aud["y"], aud["words"], n_folds, seed,
                               group_by_word)

    # every reducer is fit on BOTH tasks jointly, so the two tasks share ONE
    # space (only then can co-location be read off directly).  2D coords feed the
    # main/S1/S2 figures; the separate 3-component fits feed the 3D supplements.
    # (PCA is nested — pc1/pc2 are identical whether we ask for 2 or 3 comps — but
    # MDS is NOT, so a 3D MDS needs its own fit.)
    combined = np.vstack([pic_pred, aud_pred])
    D = cosine_distances(combined)
    mds = MDS(n_components=2, dissimilarity="precomputed", metric=True,
              random_state=seed, n_init=4, max_iter=300, normalized_stress=False)
    coords = mds.fit_transform(D)
    coords3 = MDS(n_components=3, dissimilarity="precomputed", metric=True,
                  random_state=seed, n_init=4, max_iter=300,
                  normalized_stress=False).fit_transform(D)
    pca_coords = PCA(n_components=3).fit_transform(combined)
    n_pic = pic_pred.shape[0]

    df = pd.DataFrame({
        "patient": patient,
        "task": ["picture"] * n_pic + ["auditory"] * aud_pred.shape[0],
        "word": np.concatenate([pic["words"], aud["words"]]),
        "category": np.concatenate([pic["cats"], aud["cats"]]),
        "mds1": coords[:, 0],
        "mds2": coords[:, 1],
        "mds3d_1": coords3[:, 0],
        "mds3d_2": coords3[:, 1],
        "mds3d_3": coords3[:, 2],
        "pc1": pca_coords[:, 0],
        "pc2": pca_coords[:, 1],
        "pc3": pca_coords[:, 2],
    })

    align, p_align, n_shared, per_cat = category_centroid_alignment(
        pic_pred, pic["cats"], aud_pred, aud["cats"], n_shuffle, seed)
    summary = {
        "patient": patient,
        "n_pic_trials": int(n_pic),
        "n_aud_trials": int(aud_pred.shape[0]),
        "n_shared_categories": int(n_shared),
        "cat_centroid_alignment": align,
        "cat_centroid_alignment_p": p_align,
        "mds_stress": float(mds.stress_),
    }

    del pic, aud, combined, D
    gc.collect()
    return df, summary


# ══════════════════════════════════════════════════════════════════════════
# Plotting (QC — the paper panels are rendered from source_data separately)
# ══════════════════════════════════════════════════════════════════════════

def _category_palette(cats: np.ndarray) -> Dict[str, tuple]:
    order = sorted(np.unique(cats))
    cmap = plt.get_cmap("tab10")
    return {c: cmap(i % 10) for i, c in enumerate(order)}


def plot_patient(df: pd.DataFrame, summary: dict, out_png: Path) -> None:
    pal = _category_palette(df["category"].to_numpy())
    x_all, y_all = df["mds1"].to_numpy(), df["mds2"].to_numpy()
    xpad = 0.05 * (np.nanmax(x_all) - np.nanmin(x_all) + 1e-9)
    ypad = 0.05 * (np.nanmax(y_all) - np.nanmin(y_all) + 1e-9)
    xlim = (np.nanmin(x_all) - xpad, np.nanmax(x_all) + xpad)
    ylim = (np.nanmin(y_all) - ypad, np.nanmax(y_all) + ypad)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), sharex=True, sharey=True)
    for ax, task in zip(axes, TASKS):
        sub = df[df["task"] == task]
        for c in sorted(sub["category"].unique()):
            cc = sub[sub["category"] == c]
            ax.scatter(cc["mds1"], cc["mds2"], s=26, alpha=0.8,
                       color=pal[c], edgecolor="none", label=c)
        ax.set_title(f"{task} decoder  (n={len(sub)})")
        ax.set_xlabel("MDS 1")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.axhline(0, color="0.85", lw=0.6, zorder=0)
        ax.axvline(0, color="0.85", lw=0.6, zorder=0)
    axes[0].set_ylabel("MDS 2")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                   fontsize=8, frameon=False, title="category")
    a = summary["cat_centroid_alignment"]
    p = summary["cat_centroid_alignment_p"]
    fig.suptitle(f"{summary['patient']} — separate-decoder predicted GloVe "
                 f"(cosine-MDS)\ncross-task category-centroid alignment "
                 f"= {a:.3f} (p = {p:.3g})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.86, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Run dir + main
# ══════════════════════════════════════════════════════════════════════════

def _run_dir_name(n_folds: int, seed: int) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{ts}_prediction_mds_separate_kfold{n_folds}_seed{seed}"


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient", nargs="*", default=None,
                    help="patients (default: all shared PN+AN patients)")
    ap.add_argument("--pic-run", default=PIC_RUN_DEFAULT)
    ap.add_argument("--aud-run", default=AUD_RUN_DEFAULT)
    ap.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument("--n-shuffle", type=int, default=DEFAULT_N_SHUFFLE)
    ap.add_argument("--group-by-word", action="store_true",
                    help="hold out whole words (stricter zero-shot) instead of "
                         "trial-level K-fold")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-dir", default=None,
                    help="override run dir (default: fresh timestamped dir)")
    args = ap.parse_args(argv)

    patients = args.patient or SHARED_PATIENTS
    run_dir = (Path(args.out_dir) if args.out_dir
               else OUT_ROOT / _run_dir_name(args.n_folds, args.seed))
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prediction_mds] run dir: {run_dir}")

    summaries = []
    for pat in patients:
        print(f"[prediction_mds] {pat} ...", flush=True)
        try:
            df, summary = analyze_patient(
                pat, args.pic_run, args.aud_run,
                args.n_folds, args.n_shuffle, args.seed, args.group_by_word)
        except Exception as exc:  # keep going; report which patient failed
            print(f"  !! {pat} failed: {exc}")
            continue
        df.to_csv(run_dir / f"prediction_mds_{pat}.csv", index=False)
        plot_patient(df, summary, run_dir / pat / f"prediction_mds_{pat}.png")
        summaries.append(summary)
        print(f"  {pat}: align={summary['cat_centroid_alignment']:.3f} "
              f"p={summary['cat_centroid_alignment_p']:.3g} "
              f"(n_pic={summary['n_pic_trials']}, n_aud={summary['n_aud_trials']})")

    if summaries:
        pd.DataFrame(summaries).to_csv(
            run_dir / "prediction_mds_alignment_summary.csv", index=False)
    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "analysis": "prediction_mds_separate_decoders",
            "created": datetime.now().isoformat(timespec="seconds"),
            "patients": patients,
            "pic_run": args.pic_run, "aud_run": args.aud_run,
            "n_folds": args.n_folds, "n_shuffle": args.n_shuffle,
            "seed": args.seed, "group_by_word": bool(args.group_by_word),
            "method": ("%s KFold OOF per-task kernel-PLS -> cosine MDS"
                       % ("word-grouped" if args.group_by_word else "trial-level")),
        }, f, indent=2)
    print(f"[prediction_mds] done -> {run_dir}")


if __name__ == "__main__":
    main()
