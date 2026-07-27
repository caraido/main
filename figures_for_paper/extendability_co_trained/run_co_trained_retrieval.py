# -*- coding: utf-8 -*-
"""
figures_for_paper/extendability_co_trained/run_co_trained_retrieval.py
======================================================================
Open-vocabulary (zero-shot) word-retrieval pipeline driven by the **CO-TRAINED**
cross-task decoder, evaluated on picture-naming and auditory-naming trials
*separately*.

This is the co-trained analogue of ``tests/open_vocab_retrieval/run.py``.  The
*only* thing that changes versus the shipped picture-naming figure is the source
of the per-trial predicted GloVe vectors: instead of a picture-only kernel-PLS,
a single kernel-PLS is co-trained on pooled picture + auditory trials (the
``pooled_pic`` / ``pooled_aud`` condition of ``cross_task_cotrain.py``), produced
here in a genuine out-of-fold (OOF) form so every trial receives a prediction
from a model that never saw it.  Everything downstream — gallery construction,
cosine retrieval, rank / graded metrics, permutation nulls, group Wilcoxon,
sweeps — is REUSED unchanged from the ``open_vocab_retrieval`` package.

Co-trained design (per patient):
  * Channels are the intersection of the two tasks' electrodes (``load_patient``);
    features are lagged HGA at each task's own peak bin.  Restricted to the 6
    shared PN∩AN patients (AA, AZ, DR, LH, RB, WBH).
  * Zero-shot split is over the UNION vocabulary: a fraction ``held_out_frac`` of
    the unique clean words across BOTH tasks is withheld from ALL training in both
    modalities — a held-out word is truly unseen in either task.
  * For target task T, each CV fold trains on (T's in-vocab train trials for that
    fold) ∪ (ALL of the OTHER task's in-vocab trials) and predicts T's held-out
    fold trials out-of-fold; zero-shot T trials are predicted by every fold's
    model and averaged.  So an "in-vocab" T trial is one whose word was seen in
    training — possibly only cross-modally (the co-training payoff).

Run in the Speech conda env (needs the project pkls + torchtext GloVe):
    C:/Users/Owner/miniconda3/envs/Speech/python.exe \
        figures_for_paper/extendability_co_trained/run_co_trained_retrieval.py

Outputs -> figures_for_paper/extendability_co_trained/source_data/  (task-suffixed,
identical schema to the open_vocab_retrieval run):
    trial_predictions_{task}.csv, per_patient_metrics_{task}.csv, sweep_{task}.csv,
    sweep_summary_{task}.csv, group_inference_{task}.json, group_panel_{task}.csv,
    gallery_meta_matched_N5000.csv, qualitative_top5_{task}.csv, run_metadata_{task}.json
Quick QC figures -> figures_for_paper/extendability_co_trained/pipeline_qc/
(the paper panels themselves are rendered by extendability_panels.py).
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
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS_ROOT = os.path.dirname(HERE)                              # …/figures_for_paper
MAIN_DIR = os.path.dirname(FIGS_ROOT)                          # …/main
if MAIN_DIR not in sys.path:
    sys.path.insert(0, MAIN_DIR)

from analysis.open_vocab_retrieval import gallery as gallery_mod          # noqa: E402
from analysis.open_vocab_retrieval import retrieval, metrics, relevance, stats, sweeps, figures  # noqa: E402
from analysis.open_vocab_retrieval.run import (                          # noqa: E402
    evaluate_patient, group_inference, DEFAULT_KS, DEFAULT_NS)
from analysis.open_vocab_retrieval.predict_io import (                   # noqa: E402
    TrialPredictions, HELD_OUT_FOLD, PIC_RUN_DEFAULT, AUD_RUN_DEFAULT, SHARED_PATIENTS)
from analysis.open_vocab_retrieval.gallery import clean_word             # noqa: E402
from analysis.cross_task.cross_task_cotrain import load_patient, make_model  # noqa: E402

PROJECT_ROOT = Path(MAIN_DIR)
FIG_DIR = Path(HERE)
SRC_DIR = FIG_DIR / "source_data"
QC_DIR = FIG_DIR / "pipeline_qc"

TASKS = ("picture_naming", "auditory_naming")


# ══════════════════════════════════════════════════════════════════════════
# Co-trained out-of-fold predictions  (the only new logic)
# ══════════════════════════════════════════════════════════════════════════

def _oof_cotrained(target: dict, target_clean: np.ndarray,
                   other: dict, other_clean: np.ndarray,
                   held_out_words: set, task_name: str,
                   patient: str, n_folds: int, model: str, seed: int
                   ) -> TrialPredictions:
    """OOF predicted GloVe for one *target* task, co-trained with the *other* task.

    Each fold trains on (target in-vocab train) ∪ (ALL other in-vocab) and predicts
    the target's held-out fold trials.  Zero-shot target trials are predicted by
    every fold's model and averaged.  Held-out words are excluded from BOTH tasks'
    training so they are genuinely unseen.
    """
    X_t, y_t = target["X"], target["y"]
    labels_t = target["words"]                # suffixed labels (provenance)
    cats_t = target["cats"]
    D = y_t.shape[1]

    if not (np.all(np.isfinite(X_t)) and np.all(np.isfinite(y_t))):
        raise ValueError(f"{patient}/{task_name}: non-finite neural features or GloVe targets.")

    ho_set = held_out_words                   # set of clean words
    is_held_out = np.array([w in ho_set for w in target_clean], dtype=bool)

    # Other task's in-vocab trials (word not held out) — always in training.
    other_in = np.array([w not in ho_set for w in other_clean], dtype=bool)
    X_other = other["X"][other_in]
    y_other = other["y"][other_in]
    if len(X_other) == 0:
        warnings.warn(f"{patient}/{task_name}: the other task contributes no in-vocab "
                      "trials — falling back to within-task CV (no co-training).")

    ho_idx = np.where(is_held_out)[0]
    reg_idx = np.where(~is_held_out)[0]
    if len(reg_idx) < 2:
        raise ValueError(
            f"{patient}/{task_name}: only {len(reg_idx)} in-vocab trials — cannot "
            f"cross-validate (held_out_frac too high for this task?).")

    n_folds_eff = int(max(2, min(n_folds, len(reg_idx))))
    if n_folds_eff < n_folds:
        warnings.warn(f"{patient}/{task_name}: {len(reg_idx)} in-vocab trials -> "
                      f"reducing CV folds {n_folds}->{n_folds_eff}.")

    pred_emb = np.full((len(target_clean), D), np.nan, dtype=np.float64)
    cv_fold = np.full(len(target_clean), HELD_OUT_FOLD, dtype=np.int64)
    ho_accum = np.zeros((len(ho_idx), D), dtype=np.float64)
    n_models = 0

    kf = KFold(n_splits=n_folds_eff, shuffle=True, random_state=seed)
    for f, (tr_local, te_local) in enumerate(kf.split(reg_idx)):
        tr = reg_idx[tr_local]
        te = reg_idx[te_local]
        X_tr = np.vstack([X_t[tr], X_other]) if len(X_other) else X_t[tr]
        y_tr = np.vstack([y_t[tr], y_other]) if len(X_other) else y_t[tr]
        est = make_model(model, X_tr.shape[0])
        est.fit(X_tr, y_tr)
        pred_emb[te] = est.predict(X_t[te])
        cv_fold[te] = f
        if len(ho_idx):
            ho_accum += est.predict(X_t[ho_idx])
        n_models += 1

    if len(ho_idx):
        pred_emb[ho_idx] = ho_accum / n_models
        cv_fold[ho_idx] = HELD_OUT_FOLD

    if np.any(np.isnan(pred_emb)):
        n_bad = int(np.isnan(pred_emb).any(axis=1).sum())
        raise RuntimeError(f"{patient}/{task_name}: {n_bad} trials received no prediction.")

    return TrialPredictions(
        patient=patient, task=task_name, pred_emb=pred_emb,
        true_word=target_clean, true_label=labels_t, category=cats_t,
        is_held_out=is_held_out, cv_fold=cv_fold)


def make_cotrained_predictions(patient: str, pic_run: str, aud_run: str,
                               embedding: str = "GloVe", n_folds: int = 5,
                               held_out_frac: float = 0.3, model: str = "kernel_pls",
                               seed: int = 0) -> Tuple[TrialPredictions, TrialPredictions]:
    """Return (picture_tp, auditory_tp) OOF predictions from the co-trained decoder.

    Loads both tasks' pkls exactly once (via ``load_patient``) and produces both
    tasks' predictions from that single load.
    """
    pic, aud = load_patient(patient, pic_run, aud_run, embedding=embedding)
    pic_clean = np.array([clean_word(w) for w in pic["words"]])
    aud_clean = np.array([clean_word(w) for w in aud["words"]])

    union = np.unique(np.concatenate([pic_clean, aud_clean]))
    rng = np.random.default_rng(seed)
    n_ho = int(round(len(union) * held_out_frac))
    held_out_words = set(rng.choice(union, n_ho, replace=False).tolist()) if n_ho > 0 else set()

    pic_tp = _oof_cotrained(pic, pic_clean, aud, aud_clean, held_out_words,
                            "picture_naming", patient, n_folds, model, seed)
    aud_tp = _oof_cotrained(aud, aud_clean, pic, pic_clean, held_out_words,
                            "auditory_naming", patient, n_folds, model, seed)
    del pic, aud
    gc.collect()
    return pic_tp, aud_tp


# ══════════════════════════════════════════════════════════════════════════
# Scoring + source-data output  (mirrors run.run steps 3-8, new output dir)
# ══════════════════════════════════════════════════════════════════════════

def score_and_write(task: str, predictions: List[TrialPredictions], glove,
                    concreteness, subtlex, rel_graded, rel_category,
                    headline_N: int, headline_variant: str, Ns: Sequence[int],
                    variants: Sequence[str], n_perm: int, n_perm_graded: int,
                    k_ndcg: int, center: bool, seed: int, save_figs: bool) -> dict:
    """Evaluate one task's co-trained predictions and write the source-data CSVs."""
    stim_words = sorted(set(w for tp in predictions for w in tp.true_word))
    print(f"  [{task}] stimulus wordset (union, clean): {len(stim_words)} words", flush=True)

    pd.concat([tp.to_frame() for tp in predictions], ignore_index=True).to_csv(
        SRC_DIR / f"trial_predictions_{task}.csv", index=False)

    headline_gallery = gallery_mod.build_gallery(
        glove, stim_words, n=headline_N, variant=headline_variant,
        concreteness=concreteness, subtlex=subtlex)
    headline_gallery.meta.to_csv(
        SRC_DIR / f"gallery_meta_{headline_variant}_N{headline_N}_{task}.csv", index=False)
    print(f"  [{task}] gallery N_effective={headline_gallery.N} "
          f"(stimulus={int(headline_gallery.meta['is_stimulus'].sum())})", flush=True)

    patient_rows = []
    for tp in predictions:
        row = evaluate_patient(tp, headline_gallery, rel_graded, rel_category,
                               ks=DEFAULT_KS, center=center, k_ndcg=k_ndcg,
                               n_perm=n_perm, n_perm_graded=n_perm_graded, seed=seed)
        patient_rows.append(row)
        print(f"    {tp.patient} ({task}): median%rank all={row.get('median_percentile_all'):.3f} "
              f"invocab={row.get('median_percentile_invocab'):.3f} "
              f"heldout={row.get('median_percentile_heldout'):.3f} "
              f"top10_all={row.get('top10_all'):.3f} "
              f"nDCG={row.get('graded_ndcg_mean'):.3f}", flush=True)

    patient_df = pd.DataFrame(patient_rows)
    patient_df.to_csv(SRC_DIR / f"per_patient_metrics_{task}.csv", index=False)

    grp = group_inference(patient_rows)
    with open(SRC_DIR / f"group_inference_{task}.json", "w", encoding="utf-8") as f:
        json.dump(grp, f, indent=2, default=float)

    group_panel = patient_df[["patient",
                              "median_percentile_invocab", "median_percentile_heldout",
                              "graded_ndcg_mean", "graded_near_miss_sim_mean"]].copy()
    group_panel = group_panel.rename(columns={
        "graded_ndcg_mean": "ndcg_mean",
        "graded_near_miss_sim_mean": "near_miss_sim_mean"})
    if "near_miss_null_mean" in patient_df.columns:
        group_panel["near_miss_null_mean"] = patient_df["near_miss_null_mean"].values
    if "ndcg_null_mean" in patient_df.columns:
        group_panel["ndcg_null_mean"] = patient_df["ndcg_null_mean"].values
    group_panel.to_csv(SRC_DIR / f"group_panel_{task}.csv", index=False)

    print(f"  [{task}] sweeps (N x variant) ...", flush=True)
    sweep_df = sweeps.sweep_gallery_size(
        predictions, glove, stim_words, Ns=Ns, variants=variants,
        concreteness=concreteness, ks=DEFAULT_KS, center=center, subtlex=subtlex)
    sweep_df.to_csv(SRC_DIR / f"sweep_{task}.csv", index=False)
    sweep_summary = sweeps.summarize_sweep(sweep_df, ks=DEFAULT_KS)
    sweep_summary.to_csv(SRC_DIR / f"sweep_summary_{task}.csv", index=False)

    if save_figs:
        QC_DIR.mkdir(parents=True, exist_ok=True)
        figures.plot_metric_vs_N(sweep_summary, QC_DIR / f"01_metric_vs_N_{task}.png")
        figures.plot_cmc(sweep_df, headline_N, headline_variant,
                         QC_DIR / f"02_cmc_{task}.png", ks=DEFAULT_KS)
        figures.plot_invocab_vs_heldout(group_panel, QC_DIR / f"03_invocab_vs_heldout_{task}.png")
        figures.plot_near_miss(group_panel, QC_DIR / f"04_near_miss_{task}.png")
        figures.qualitative_table(
            predictions, headline_gallery, rel_graded,
            SRC_DIR / f"qualitative_top5_{task}.csv",
            out_html=QC_DIR / f"06_qualitative_top5_{task}.html",
            center=center, seed=seed)
    return grp


def run(patients: Sequence[str], pic_run: str, aud_run: str, embedding: str = "GloVe",
        headline_N: int = 5000, headline_variant: str = "matched",
        Ns: Sequence[int] = DEFAULT_NS, variants: Sequence[str] = ("matched", "raw"),
        n_folds: int = 5, held_out_frac: float = 0.3,
        n_perm: int = 1000, n_perm_graded: int = 200,
        relevance_kind: str = "wup", k_ndcg: int = 100,
        concreteness_file: Optional[str] = None, subtlex_file: Optional[str] = None,
        center: bool = True, save_figs: bool = True, seed: int = 0) -> None:
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("[1/4] GloVe + norms ...", flush=True)
    glove = gallery_mod.load_glove()
    concreteness = gallery_mod.load_concreteness(concreteness_file)
    subtlex = gallery_mod.load_subtlex(subtlex_file)
    rel_graded = relevance.make_relevance_fn(relevance_kind)
    rel_category = relevance.make_relevance_fn("category")

    print("[2/4] Co-trained OOF predictions (per patient, both tasks) ...", flush=True)
    pic_preds: List[TrialPredictions] = []
    aud_preds: List[TrialPredictions] = []
    for pat in patients:
        print(f"  [predict] {pat} (co-trained pic+aud) ...", flush=True)
        p_tp, a_tp = make_cotrained_predictions(
            pat, pic_run, aud_run, embedding=embedding, n_folds=n_folds,
            held_out_frac=held_out_frac, seed=seed)
        for tp in (p_tp, a_tp):
            print(f"    {tp.task}: T={len(tp)} held_out={int(tp.is_held_out.sum())} "
                  f"unique_words={len(np.unique(tp.true_word))}", flush=True)
        pic_preds.append(p_tp)
        aud_preds.append(a_tp)
        gc.collect()

    print("[3/4] Score + write source data (per task) ...", flush=True)
    grp_by_task = {}
    for task, preds in [("picture_naming", pic_preds), ("auditory_naming", aud_preds)]:
        grp_by_task[task] = score_and_write(
            task, preds, glove, concreteness, subtlex, rel_graded, rel_category,
            headline_N, headline_variant, Ns, variants, n_perm, n_perm_graded,
            k_ndcg, center, seed, save_figs)

    print("[4/4] Metadata ...", flush=True)
    meta = {
        "timestamp": ts, "decoder": "co_trained_kernel_pls (pooled pic+aud, OOF)",
        "tasks": list(TASKS), "pic_run": pic_run, "aud_run": aud_run,
        "patients": list(patients), "embedding": embedding,
        "headline_N": headline_N, "headline_variant": headline_variant,
        "Ns": list(Ns), "variants": list(variants), "n_folds": n_folds,
        "held_out_frac": held_out_frac, "held_out_scope": "union vocabulary (both tasks)",
        "n_perm": n_perm, "n_perm_graded": n_perm_graded,
        "relevance_kind": relevance_kind, "k_ndcg": k_ndcg, "center": center, "seed": seed,
        "concreteness_file": concreteness_file, "subtlex_file": subtlex_file,
        "command": "python figures_for_paper/extendability_co_trained/run_co_trained_retrieval.py "
                   + " ".join(sys.argv[1:]),
    }
    for task in TASKS:
        m = dict(meta, task=task)
        with open(SRC_DIR / f"run_metadata_{task}.json", "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)
    print(f"\nDone. Source data -> {SRC_DIR}\n      QC figures  -> {QC_DIR}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Co-trained open-vocabulary retrieval (pic + aud)")
    p.add_argument("--patient", default=None, help="One patient (default: all 6 shared)")
    p.add_argument("--pic-run", default=PIC_RUN_DEFAULT)
    p.add_argument("--aud-run", default=AUD_RUN_DEFAULT)
    p.add_argument("--embedding", default="GloVe")
    p.add_argument("--headline-N", type=int, default=5000)
    p.add_argument("--headline-variant", default="matched", choices=["matched", "raw"])
    p.add_argument("--Ns", type=int, nargs="+", default=list(DEFAULT_NS))
    p.add_argument("--variants", nargs="+", default=["matched", "raw"], choices=["matched", "raw"])
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--held-out-frac", type=float, default=0.3)
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--n-perm-graded", type=int, default=200)
    p.add_argument("--relevance", default="wup", choices=list(relevance.RELEVANCE_KINDS))
    p.add_argument("--k-ndcg", type=int, default=100)
    p.add_argument("--concreteness-file", default=None)
    p.add_argument("--subtlex-file", default=None)
    p.add_argument("--no-center", action="store_true")
    p.add_argument("--no-figs", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    patients = [args.patient] if args.patient else SHARED_PATIENTS
    run(patients, args.pic_run, args.aud_run, embedding=args.embedding,
        headline_N=args.headline_N, headline_variant=args.headline_variant,
        Ns=args.Ns, variants=args.variants, n_folds=args.n_folds,
        held_out_frac=args.held_out_frac, n_perm=args.n_perm,
        n_perm_graded=args.n_perm_graded, relevance_kind=args.relevance,
        k_ndcg=args.k_ndcg, concreteness_file=args.concreteness_file,
        subtlex_file=args.subtlex_file, center=not args.no_center,
        save_figs=not args.no_figs, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
