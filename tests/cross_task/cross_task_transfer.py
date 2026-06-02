# -*- coding: utf-8 -*-
"""
tests/cross_task/cross_task_transfer.py
=======================================
Cross-task transfer learning: apply the 3-arm transfer framework from
cross_patient_decoding to the within-patient cross-task setting.

Tasks compared: picture naming vs auditory naming (same patient, same channels).

Arms:
  transfer    — plain PLSRegression on source-task train data; ridge maps
                X_target_anchor → T_source_anchor; frozen source decoder
                reconstructs the embedding.
  no_transfer — kernel PLS (Nystroem + PLSRegression) fitted on target-task
                train data → embedding (within-task baseline).
  cca         — time-resolved HGA CCA (Spalding/Cogan AlignCCA convention):
                aligns class-averaged latent dynamics where each (anchor word,
                time bin) is a separate observation (k*n_hist samples), channels
                as variables; CCA maps target HGA → source HGA per bin; source
                PLS then predicts the embedding.
  pca_cca     — like cca but first reduces channels to PCA components per task
                (samples = trials*bins, variables = channels); CCA aligns the
                per-(word, bin) PC trajectories; prediction per bin:
                X_tgt → PCA_tgt → CCA → inverse PCA_src → source HGA bin;
                bins re-concatenated → source PLS → embedding.

Both directions are run per patient:
  pic_to_aud  — source = picture naming, target = auditory naming
  aud_to_pic  — source = auditory naming, target = picture naming

Bootstrap loop (per direction):
  1. Stratified train/test split on both tasks (shared-vocabulary trials only).
  2. Sample k anchor words from the shared train vocabulary.
  3. Fit and evaluate all 3 arms on the target test set.
  4. Record cosine_mean, word_bal_acc, cat_indep_bal_acc + seen/unseen splits.

Outputs (under OUT_ROOT/<patient>/):
  cross_task_transfer_<patient>.csv   — per-bootstrap rows, all arms + directions
  transfer_arms_bars.png              — bar plot mean ± SEM per arm × direction

Usage:
    python -m main.tests.cross_task.cross_task_transfer
    python -m main.tests.cross_task.cross_task_transfer --patient RB
    python -m main.tests.cross_task.cross_task_transfer --patient RB --k 8 --n-bootstrap 50 --no-figs

The run folder defaults match cross_task_regression.py. Override via
--pic-run / --aud-run if needed.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import warnings
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_THIS_DIR)
_MAIN_DIR = os.path.dirname(_TESTS_DIR)
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

# ── Internal imports ──────────────────────────────────────────────────────
from tests.helpers import load_results_pkl  # noqa: E402
from tests.helpers._phoneme_semantic_helpers import (  # noqa: E402
    build_retrieval_db,
    compute_retrieval_metrics,
)
from tests.cross_patient_decoding._cross_patient_helpers import (  # noqa: E402
    stratified_train_test_split,
    sample_k_anchor_words_from_vocab,
    fit_source_pls,
    compute_T_anchors,
    build_arm1_train_inputs,
    build_arm1_cca_train_inputs,
    fit_ridge,
    fit_cca,
    fit_kernel_pls,
    predict_arm1_embedding,
    predict_arm1_cca_embedding,
    score_predictions,
    fit_pca_from_lagged,
    project_to_multibin_pcs,
    reconstruct_from_multibin_pcs,
    compute_src_pca_anchors,
    build_cca_timeobs_inputs,
    predict_cca_timeobs,
)
from tests.cross_task.cross_task_regression import (  # noqa: E402
    find_peak_bin,
    _common_channels_from_names,
    build_X_at_bin_with_channel_subset,
    PIC_RUN_DEFAULT,
    AUD_RUN_DEFAULT,
    SHARED_PATIENTS,
    PEAK_EMBEDDING,
    PEAK_METRIC,
)

# ── Constants ─────────────────────────────────────────────────────────────
SEM_REG_DIR = Path(_MAIN_DIR) / "results" / "semantic_regression"
OUT_ROOT = Path(_MAIN_DIR) / "tests" / "results" / "cross_task_transfer"


def load_per_time_scores(run_folder: str, patient: str) -> pd.DataFrame:
    """Read per-time-bin retrieval scores from the correct SEM_REG_DIR."""
    csv_path = SEM_REG_DIR / run_folder / patient / "per_time_scores.csv"
    return pd.read_csv(csv_path)

DEFAULT_K = 10
DEFAULT_N_BOOTSTRAP = 50
DEFAULT_TEST_FRAC = 0.3
DEFAULT_RIDGE_ALPHA = 2.0
DEFAULT_CCA_COMPONENTS = 10
DEFAULT_PCA_COMPONENTS = 10


# ── Data loading helpers ─────────────────────────────────────────────────

def _load_reg(run_folder: str, patient: str, embedding: str):
    """Load regressor object and channel names.

    Returns (reg, chan_names).  Frees the parent pkl dict from memory.
    """
    d = load_results_pkl(run_folder, patient)
    reg = d["regressors"][embedding]
    chan_names = np.asarray(d.get("clean_channel_names", [])).astype(str)
    del d
    gc.collect()
    return reg, chan_names


def _build_task_arrays(reg, chan_idx: np.ndarray | None, peak_bin: int) -> dict:
    """Extract X, y, words, cats arrays from a regressor at *peak_bin*.

    If *chan_idx* is given (e.g. restricted to common channels with the other
    task), ``build_X_at_bin_with_channel_subset`` is used; otherwise the
    pre-computed ``reg.X_to_use[peak_bin]`` is returned directly.

    Returns a dict with keys:
        X, y, words, cats,
        db_embeds, unique_words, word_to_cat_idx, unique_cats, word_to_idx
    """
    words = np.asarray(reg.labels).astype(str)
    word_to_index = {str(k): v for k, v in reg.word_to_index.items()}
    cats = np.array([
        str(reg.index_to_category[
            reg.word_index_to_category_index[word_to_index[str(w)]]
        ])
        for w in words
    ])

    if chan_idx is not None and len(chan_idx) > 0:
        n_channels = int(len(chan_idx))
        X = build_X_at_bin_with_channel_subset(reg, peak_bin, chan_idx)
    else:
        n_channels = int(reg.data.shape[2])
        X = np.array(reg.X_to_use[peak_bin])

    y = np.array(reg.y)
    db_embeds, unique_words, word_to_cat_idx, unique_cats, word_to_idx = (
        build_retrieval_db(y, words, cats)
    )

    return {
        "X": X,
        "y": y,
        "words": words,
        "cats": cats,
        "n_channels": n_channels,
        "db_embeds": db_embeds,
        "unique_words": unique_words,
        "word_to_cat_idx": word_to_cat_idx,
        "unique_cats": unique_cats,
        "word_to_idx": word_to_idx,
    }


# ── Bootstrap experiment (one direction) ─────────────────────────────────

def _run_bootstrap_direction(
    src: dict,
    tgt: dict,
    k: int,
    n_bootstrap: int,
    test_frac: float,
    ridge_alpha: float,
    cca_components: int,
    pca_components: int,
    rng_seed: int,
) -> List[dict]:
    """Run the 4-arm bootstrap experiment in one direction (source → target).

    Parameters
    ----------
    src, tgt : dict
        Task data dicts returned by ``_build_task_arrays``.
    k : int
        Number of anchor words to sample per bootstrap.
    n_bootstrap : int
        Number of bootstrap iterations.
    test_frac : float
        Fraction of each word's trials held out as test.
    ridge_alpha : float
        Ridge regularisation strength for the transfer ridge model.
    cca_components : int
        Max CCA components (clipped to n_anchors - 1).
    pca_components : int
        Number of PCA components per task for the pca_cca arm.
    rng_seed : int
        Base random seed (each bootstrap uses a derived state).

    Returns
    -------
    List of row dicts (arm × bootstrap), ready for pd.DataFrame.
    """
    rng = np.random.default_rng(rng_seed)

    # Shared vocabulary: words present in BOTH tasks
    shared_vocab = np.array(sorted(
        set(src["words"].tolist()) & set(tgt["words"].tolist())
    ))
    if len(shared_vocab) < 3:
        print(f"    WARNING: only {len(shared_vocab)} shared words — skipping direction.")
        return []
    k_eff = min(k, len(shared_vocab) - 2)
    if k_eff < k:
        print(f"    k capped from {k} to {k_eff} (shared vocab size = {len(shared_vocab)})")

    rows: List[dict] = []

    for boot_id in range(n_bootstrap):
        # ── Stratified train/test split on each task ───────────────────
        src_train_idx, src_test_idx = stratified_train_test_split(
            src["words"], test_frac=test_frac, rng=rng
        )
        tgt_train_idx, tgt_test_idx = stratified_train_test_split(
            tgt["words"], test_frac=test_frac, rng=rng
        )

        X_src_tr = src["X"][src_train_idx]
        y_src_tr = src["y"][src_train_idx]
        w_src_tr = src["words"][src_train_idx]

        X_tgt_tr = tgt["X"][tgt_train_idx]
        y_tgt_tr = tgt["y"][tgt_train_idx]
        w_tgt_tr = tgt["words"][tgt_train_idx]

        X_tgt_te = tgt["X"][tgt_test_idx]
        w_tgt_te = tgt["words"][tgt_test_idx]
        c_tgt_te = tgt["cats"][tgt_test_idx]

        # ── Anchor words: shared words present in BOTH train sets ──────
        train_shared = np.array(sorted(
            set(w_src_tr.tolist())
            & set(w_tgt_tr.tolist())
            & set(shared_vocab.tolist())
        ))
        if len(train_shared) < 2:
            continue
        anchor_words = sample_k_anchor_words_from_vocab(
            train_shared, k=k_eff, rng=rng
        )

        # ── Common scoring kwargs (target task database) ───────────────
        score_kwargs = dict(
            db_embeds=tgt["db_embeds"],
            unique_words_db=tgt["unique_words"],
            word_to_cat_idx=tgt["word_to_cat_idx"],
            unique_cats=tgt["unique_cats"],
            word_to_idx=tgt["word_to_idx"],
            anchor_words=anchor_words,
        )

        # ── Arm: no_transfer (within-target baseline) ──────────────────
        try:
            kpls = fit_kernel_pls(X_tgt_tr, y_tgt_tr)
            Y_pred = kpls.predict(X_tgt_te)
            sc = score_predictions(Y_pred, w_tgt_te, c_tgt_te, **score_kwargs)
            rows.append({
                "arm": "no_transfer",
                "bootstrap_id": boot_id,
                "k_anchors": int(len(anchor_words)),
                **sc,
            })
        except Exception as exc:
            print(f"    no_transfer boot={boot_id}: {type(exc).__name__}: {exc}")

        # ── Fit source PLS once; reused by transfer + cca arms ─────────
        try:
            pls_src = fit_source_pls(X_src_tr, y_src_tr)
        except Exception as exc:
            print(f"    fit_source_pls boot={boot_id}: {type(exc).__name__}: {exc}")
            continue

        # ── Arm: transfer (ridge in T-space) ──────────────────────────
        try:
            T_anchors = compute_T_anchors(pls_src, X_src_tr, w_src_tr, anchor_words)
            X_anch, T_anch_tgt, kept_tr = build_arm1_train_inputs(
                X_tgt_tr, w_tgt_tr, anchor_words, T_anchors
            )
            if len(kept_tr) < 2:
                raise ValueError(f"Only {len(kept_tr)} anchor(s) kept")
            ridge = fit_ridge(X_anch, T_anch_tgt, alpha=ridge_alpha)
            Y_pred = predict_arm1_embedding(pls_src, ridge, X_tgt_te)
            sc = score_predictions(Y_pred, w_tgt_te, c_tgt_te, **score_kwargs)
            rows.append({
                "arm": "transfer",
                "bootstrap_id": boot_id,
                "k_anchors": int(len(kept_tr)),
                **sc,
            })
        except Exception as exc:
            print(f"    transfer boot={boot_id}: {type(exc).__name__}: {exc}")

        # ── Arm: cca (time-resolved HGA CCA alignment) ────────────────
        # Canonical convention (Spalding/Cogan AlignCCA): align class-averaged
        # latent dynamics with each (word, time-bin) as an alignment sample,
        # so k anchors give k*n_hist observations rather than k.
        try:
            L_tgt, L_src, kept_cca = build_cca_timeobs_inputs(
                X_src_tr, w_src_tr, X_tgt_tr, w_tgt_tr, anchor_words,
                n_channels_src=src["n_channels"],
                n_channels_tgt=tgt["n_channels"],
            )
            if len(kept_cca) < 2:
                raise ValueError(f"Only {len(kept_cca)} CCA anchor(s) kept")
            n_comp = max(1, min(cca_components, L_tgt.shape[0] - 1,
                                L_tgt.shape[1], L_src.shape[1]))
            cca_model = fit_cca(L_tgt, L_src, n_components=n_comp)
            n_hist_src = X_src_tr.shape[1] // src["n_channels"]
            Y_pred = predict_cca_timeobs(
                pls_src, cca_model, X_tgt_te,
                n_channels_tgt=tgt["n_channels"],
                n_channels_src=src["n_channels"],
                n_hist_src=n_hist_src,
            )
            sc = score_predictions(Y_pred, w_tgt_te, c_tgt_te, **score_kwargs)
            rows.append({
                "arm": "cca",
                "bootstrap_id": boot_id,
                "k_anchors": int(len(kept_cca)),
                **sc,
            })
        except Exception as exc:
            print(f"    cca boot={boot_id}: {type(exc).__name__}: {exc}")

        # ── Arm: pca_cca (channel-PCA + time-resolved CCA) ────────────
        # Reduce channels per task with PCA (DimRedReshape convention:
        # samples = trials*bins, variables = channels), then align the
        # per-(word, bin) PC trajectories with CCA exactly as the cca arm.
        # Prediction path (per bin): X_tgt → pca_tgt → CCA → pca_src⁻¹
        #   → src HGA bin; bins re-concatenated → pls_src → embedding.
        try:
            pca_src = fit_pca_from_lagged(
                X_src_tr, src["n_channels"], n_components=pca_components
            )
            pca_tgt = fit_pca_from_lagged(
                X_tgt_tr, tgt["n_channels"], n_components=pca_components
            )
            L_tgt, L_src, kept_pc = build_cca_timeobs_inputs(
                X_src_tr, w_src_tr, X_tgt_tr, w_tgt_tr, anchor_words,
                n_channels_src=src["n_channels"],
                n_channels_tgt=tgt["n_channels"],
                pca_src=pca_src, pca_tgt=pca_tgt,
            )
            if len(kept_pc) < 2:
                raise ValueError(f"Only {len(kept_pc)} pca_cca anchor(s) kept")
            n_comp_pc = max(1, min(cca_components, L_tgt.shape[0] - 1,
                                   L_tgt.shape[1], L_src.shape[1]))
            cca_pc = fit_cca(L_tgt, L_src, n_components=n_comp_pc)
            n_hist_src = X_src_tr.shape[1] // src["n_channels"]
            Y_pred = predict_cca_timeobs(
                pls_src, cca_pc, X_tgt_te,
                n_channels_tgt=tgt["n_channels"],
                n_channels_src=src["n_channels"],
                n_hist_src=n_hist_src,
                pca_tgt=pca_tgt, pca_src=pca_src,
            )
            sc = score_predictions(Y_pred, w_tgt_te, c_tgt_te, **score_kwargs)
            rows.append({
                "arm": "pca_cca",
                "bootstrap_id": boot_id,
                "k_anchors": int(len(kept_pc)),
                **sc,
            })
        except Exception as exc:
            print(f"    pca_cca boot={boot_id}: {type(exc).__name__}: {exc}")

    return rows


# ── Plotting ──────────────────────────────────────────────────────────────

def _plot_summary(df: pd.DataFrame, patient: str, out_dir: Path) -> None:
    """Bar plot: mean ± SEM of word_bal_acc per arm and direction."""
    arm_order = ["no_transfer", "transfer", "cca", "pca_cca"]
    colors = {
        "no_transfer": "#7f7f7f",
        "transfer": "#1f77b4",
        "cca": "#2ca02c",
        "pca_cca": "#9467bd",
    }
    metric = "word_bal_acc"
    directions = ["pic_to_aud", "aud_to_pic"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, direction in zip(axes, directions):
        sub = df[df["direction"] == direction]
        means, sems = [], []
        for arm in arm_order:
            vals = sub[sub["arm"] == arm][metric].dropna()
            means.append(float(vals.mean()) if len(vals) > 0 else np.nan)
            sems.append(float(vals.sem()) if len(vals) > 1 else 0.0)

        xs = np.arange(len(arm_order))
        ax.bar(xs, means, color=[colors[a] for a in arm_order], alpha=0.85, zorder=2)
        ax.errorbar(xs, means, yerr=sems, fmt="none", color="black",
                    capsize=4, lw=1.5, zorder=3)
        for xi, (m, s) in enumerate(zip(means, sems)):
            if not np.isnan(m):
                ax.text(xi, m + (s or 0) + 0.01, f"{m:.3f}",
                        ha="center", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(arm_order, fontsize=9)
        ax.set_xlabel("Arm")
        ax.set_ylabel(metric)
        ax.set_title(f"{patient}  ·  {direction.replace('_', ' → ')}")
        valid_means = [m for m in means if not np.isnan(m)]
        ax.set_ylim(0, max(0.4, (max(valid_means) + 0.15)) if valid_means else 0.4)
        ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle(f"Cross-task transfer: {patient}", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "transfer_arms_bars.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── Per-patient pipeline ──────────────────────────────────────────────────

def analyze_patient(
    patient: str,
    pic_run: str,
    aud_run: str,
    embedding: str = PEAK_EMBEDDING,
    k: int = DEFAULT_K,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    test_frac: float = DEFAULT_TEST_FRAC,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    cca_components: int = DEFAULT_CCA_COMPONENTS,
    pca_components: int = DEFAULT_PCA_COMPONENTS,
    out_dir: Path | None = None,
    save_figs: bool = True,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Run the full cross-task transfer analysis for one patient.

    Returns a DataFrame of per-bootstrap rows (all arms + both directions).
    Also writes a CSV and optional summary figure to *out_dir*.
    """
    print(f"\n=== {patient} : {embedding} ===", flush=True)
    out_dir = out_dir or (OUT_ROOT / patient)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: peak bins from CSV (cheap) ────────────────────────────
    pic_scores = load_per_time_scores(pic_run, patient)
    aud_scores = load_per_time_scores(aud_run, patient)
    pic_peak, pic_peak_val = find_peak_bin(pic_scores, embedding=embedding)
    aud_peak, aud_peak_val = find_peak_bin(aud_scores, embedding=embedding)
    print(f"  pic_peak={pic_peak} ({pic_peak_val:.3f})  "
          f"aud_peak={aud_peak} ({aud_peak_val:.3f})", flush=True)

    # ── Step 2: load regressors ───────────────────────────────────────
    print("  loading pkl files ...", flush=True)
    pic_reg, pic_chan = _load_reg(pic_run, patient, embedding)
    aud_reg, aud_chan = _load_reg(aud_run, patient, embedding)

    # ── Step 3: find common channels ─────────────────────────────────
    if len(pic_chan) > 0 and len(aud_chan) > 0:
        idx_pic, idx_aud, common = _common_channels_from_names(pic_chan, aud_chan)
        if len(common) == 0:
            n = min(pic_reg.data.shape[2], aud_reg.data.shape[2])
            idx_pic = np.arange(n, dtype=np.int64)
            idx_aud = np.arange(n, dtype=np.int64)
            print(f"  no overlapping channel names; fallback to first {n} channels",
                  flush=True)
        else:
            print(f"  common channels: {len(common)} "
                  f"(pic={len(pic_chan)}, aud={len(aud_chan)})", flush=True)
    else:
        n = min(pic_reg.data.shape[2], aud_reg.data.shape[2])
        idx_pic = np.arange(n, dtype=np.int64)
        idx_aud = np.arange(n, dtype=np.int64)
        print(f"  channel names unavailable; fallback to first {n} channels",
              flush=True)

    # ── Step 4: build task-level arrays ──────────────────────────────
    pic = _build_task_arrays(pic_reg, idx_pic, pic_peak)
    aud = _build_task_arrays(aud_reg, idx_aud, aud_peak)
    del pic_reg, aud_reg
    gc.collect()
    print(f"  pic trials={len(pic['words'])}  "
          f"shared_vocab={len(set(pic['words'].tolist()) & set(aud['words'].tolist()))}",
          flush=True)

    # ── Step 5: bootstrap both directions ────────────────────────────
    all_rows: List[dict] = []
    for direction, src, tgt in [
        ("pic_to_aud", pic, aud),
        ("aud_to_pic", aud, pic),
    ]:
        print(f"  [{direction}] n_bootstrap={n_bootstrap} k={k} ...", flush=True)
        rows = _run_bootstrap_direction(
            src=src,
            tgt=tgt,
            k=k,
            n_bootstrap=n_bootstrap,
            test_frac=test_frac,
            ridge_alpha=ridge_alpha,
            cca_components=cca_components,
            pca_components=pca_components,
            rng_seed=rng_seed,
        )
        for r in rows:
            r.update({
                "patient": patient,
                "embedding": embedding,
                "direction": direction,
                "pic_peak_bin": pic_peak,
                "aud_peak_bin": aud_peak,
            })
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    # ── Step 6: save CSV ──────────────────────────────────────────────
    csv_path = out_dir / f"cross_task_transfer_{patient}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  saved: {csv_path}  ({len(df)} rows)", flush=True)

    # ── Step 7: optional figure ───────────────────────────────────────
    if save_figs and not df.empty:
        _plot_summary(df, patient, out_dir)

    return df


# ── Driver ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-task transfer: picture naming ↔ auditory naming"
    )
    parser.add_argument("--patient", default=None,
                        help="Run one patient (default: all SHARED_PATIENTS)")
    parser.add_argument("--pic-run", default=PIC_RUN_DEFAULT)
    parser.add_argument("--aud-run", default=AUD_RUN_DEFAULT)
    parser.add_argument("--embedding", default=PEAK_EMBEDDING)
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="Number of anchor words per bootstrap")
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    parser.add_argument("--ridge-alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    parser.add_argument("--cca-components", type=int, default=DEFAULT_CCA_COMPONENTS)
    parser.add_argument("--pca-components", type=int, default=DEFAULT_PCA_COMPONENTS)
    parser.add_argument("--no-figs", action="store_true")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_root = Path(args.out_dir) if args.out_dir else OUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    patients = [args.patient] if args.patient else SHARED_PATIENTS
    summary_dfs: List[pd.DataFrame] = []

    for pat in patients:
        try:
            df = analyze_patient(
                patient=pat,
                pic_run=args.pic_run,
                aud_run=args.aud_run,
                embedding=args.embedding,
                k=args.k,
                n_bootstrap=args.n_bootstrap,
                test_frac=args.test_frac,
                ridge_alpha=args.ridge_alpha,
                cca_components=args.cca_components,
                pca_components=args.pca_components,
                out_dir=out_root / pat,
                save_figs=not args.no_figs,
                rng_seed=args.seed,
            )
            summary_dfs.append(df)
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"  ERROR for {pat}", flush=True)
        finally:
            gc.collect()

    if summary_dfs:
        summary = pd.concat(summary_dfs, ignore_index=True)
        agg = (
            summary
            .groupby(["patient", "direction", "arm"])[
                ["word_bal_acc", "cat_indep_bal_acc", "cosine_mean",
                 "word_acc_seen", "word_acc_unseen"]
            ]
            .agg(["mean", "sem"])
        )
        agg.columns = ["_".join(c) for c in agg.columns]
        agg = agg.reset_index()
        summary_path = out_root / "cross_task_transfer_summary.csv"
        agg.to_csv(summary_path, index=False)
        print(f"\nWrote summary: {summary_path}", flush=True)
        print(agg.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
