# -*- coding: utf-8 -*-
"""
tests/open_vocab_retrieval/run.py
=================================
Orchestration for the open-vocabulary (zero-shot) retrieval analysis.

Pipeline (per the implementation guide):
  1. Cross-validated per-trial predicted embeddings, per patient, with a zero-shot
     held-out-word split                                    (predict_io)
  2. Build the open gallery (matched + raw), including the stimulus wordset (gallery)
  3. Cosine retrieval + tie-safe ranks                      (retrieval)
  4. Rank metrics (median percentile rank headline, CMC/top-k, MRR, MedR) (metrics)
  5. Graded near-miss with INDEPENDENT WordNet relevance    (relevance, metrics)
  6. Significance: within-patient permutation nulls; group-level Wilcoxon vs
     chance; frequency-confound regression                  (stats)
  7. Gallery-size / variant sweeps                          (sweeps)
  8. Figures + qualitative table                            (figures)

Outputs (repo figure convention): PNGs -> ``figures/open_vocab_retrieval/``,
all CSV source data -> ``figures/open_vocab_retrieval/source_data/``.

Usage:
    python -m main.analysis.open_vocab_retrieval.run
    python -m main.analysis.open_vocab_retrieval.run --patient AA --n-perm 200
    python -m main.analysis.open_vocab_retrieval.run --task auditory --headline-N 3000
    python -m main.analysis.open_vocab_retrieval.run --concreteness-file data/concreteness.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_THIS_DIR)
_MAIN_DIR = os.path.dirname(_TESTS_DIR)
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

from utils.paths import figures_dir
from analysis.open_vocab_retrieval import gallery as gallery_mod
from analysis.open_vocab_retrieval import predict_io, retrieval, metrics, relevance, stats, sweeps, figures
from analysis.open_vocab_retrieval.predict_io import (
    PIC_RUN_DEFAULT, AUD_RUN_DEFAULT, SHARED_PATIENTS, PICTURE_PATIENTS)

PROJECT_ROOT = Path(_MAIN_DIR)
# NB these are WRITE targets, and two paper pipelines read SRC_DIR --
# figures_for_paper/extendability/{compute_extendability_data,extendability_panels}.py and
# semantic_regression/compute_within_category_null.py. See the untracked-inputs table in
# docs/repo_layout.md: figures/ is gitignored, so this is a tracked figure depending on an
# untracked input. Routed through utils.paths so the location is at least single-sourced.
FIG_DIR = figures_dir("open_vocab_retrieval", create=False)
SRC_DIR = figures_dir("open_vocab_retrieval", "source_data", create=False)

DEFAULT_KS = (1, 5, 10, 50, 100)
DEFAULT_NS = (200, 500, 1000, 2000, 5000)
CHANCE_PCT = 0.5


# ══════════════════════════════════════════════════════════════════════════
# Per-patient evaluation
# ══════════════════════════════════════════════════════════════════════════
 
def _median_percentile(rank: np.ndarray, N: int) -> float:
    r = rank[rank > 0]
    return float(np.median(r / float(N))) if len(r) else np.nan


def _subset(tp: predict_io.TrialPredictions, mask: np.ndarray):
    return (tp.pred_emb[mask], tp.true_word[mask], tp.cv_fold[mask])


def evaluate_patient(tp: predict_io.TrialPredictions, gallery: "gallery_mod.Gallery",
                     rel_graded, rel_category,
                     ks: Sequence[int] = DEFAULT_KS, center: bool = True,
                     k_ndcg: int = 100, k_nearmiss: int = 10,
                     n_perm: int = 1000, n_perm_graded: int = 200,
                     seed: int = 0) -> Dict[str, object]:
    """Full per-patient open-vocabulary evaluation against one gallery."""
    sims, rank, tidx = retrieval.retrieve(tp.pred_emb, gallery, tp.true_word, center=center)
    valid = tidx >= 0
    n_missing = int((~valid).sum())
    if n_missing:
        missing = sorted(set(w for w, v in zip(tp.true_word, valid) if not v))
        warnings.warn(f"{tp.patient}: {n_missing} trials whose true word is not in "
                      f"the gallery (excluded from rank metrics): {missing[:8]}")

    N = gallery.N
    out: Dict[str, object] = {"patient": tp.patient, "task": tp.task,
                              "N": N, "variant": gallery.variant,
                              "n_trials": int(len(tp)), "n_missing": n_missing,
                              "n_held_out": int(tp.is_held_out.sum())}

    # Rank metrics: all / in-vocab / held-out
    for label, mask in [("all", np.ones(len(tp), bool)),
                        ("invocab", ~tp.is_held_out),
                        ("heldout", tp.is_held_out)]:
        m = metrics.rank_metrics(rank[mask], N, ks=ks)
        for key, val in m.items():
            out[f"{key}_{label}"] = val

    # Within-patient permutation null on median percentile (all trials)
    stat_fn = lambda rk: _median_percentile(rk, N)
    if n_perm > 0:
        null = stats.rank_permutation_null(sims, tidx, tp.cv_fold, stat_fn,
                                           n_perm=n_perm, seed=seed)
        out["perm_p_median_percentile_all"] = stats.permutation_pvalue(
            stat_fn(rank), null, alternative="less")

    # Graded near-miss (independent WordNet relevance) on valid trials
    order = retrieval.ranked_indices(sims)
    graded = metrics.aggregate_graded(order, tp.true_word, gallery.words,
                                      rel_graded, valid, k=k_ndcg)
    out.update({f"graded_{k}": v for k, v in graded.items()})
    # category-level near-miss (immune to circularity)
    cat_hits = [metrics.category_hit_at_k(order[t], tp.true_word[t], gallery.words,
                                          rel_category, k=k_nearmiss)
                for t in np.where(valid)[0]]
    cat_hits = np.array(cat_hits, dtype=np.float64)
    out["category_hit_at_k"] = float(np.nanmean(cat_hits)) if np.any(~np.isnan(cat_hits)) else np.nan

    # Permutation nulls for the graded §5 statistics (matched trial->word null)
    if n_perm_graded > 0 and valid.any():
        obs_nm = graded["near_miss_sim_mean"]
        null_nm = stats.graded_permutation_null(
            order, tp.true_word, tp.cv_fold, gallery.words, rel_graded, valid,
            k=k_nearmiss, n_perm=n_perm_graded, seed=seed,
            trial_stat=metrics.near_miss_similarity)
        out["near_miss_null_mean"] = float(np.nanmean(null_nm))
        out["perm_p_near_miss"] = stats.permutation_pvalue(obs_nm, null_nm,
                                                           alternative="greater")

        # nDCG@k: absolute value is uninterpretable (chance nDCG != 0), so test
        # the observed mean against the same matched permutation null.
        obs_ndcg = graded["ndcg_mean"]
        null_ndcg = stats.graded_permutation_null(
            order, tp.true_word, tp.cv_fold, gallery.words, rel_graded, valid,
            k=k_ndcg, n_perm=n_perm_graded, seed=seed,
            trial_stat=metrics.ndcg_independent)
        out["ndcg_null_mean"] = float(np.nanmean(null_ndcg))
        out["perm_p_ndcg"] = stats.permutation_pvalue(obs_ndcg, null_ndcg,
                                                      alternative="greater")

    # Frequency confound: per-trial percentile vs log word-frequency
    lf_map = dict(zip(gallery.meta["word"], gallery.meta["log_freq"]))
    pct = np.where(rank > 0, rank / float(N), np.nan)
    log_freq = np.array([lf_map.get(w, np.nan) for w in tp.true_word], dtype=np.float64)
    freq = stats.frequency_partial_effect(pct, log_freq)
    out.update({f"freq_{k}": v for k, v in freq.items()})

    return out


# ══════════════════════════════════════════════════════════════════════════
# Group-level aggregation
# ══════════════════════════════════════════════════════════════════════════

def group_inference(patient_rows: List[dict]) -> Dict[str, object]:
    """Cross-patient Wilcoxon vs chance + bootstrap CIs (never pool trials)."""
    df = pd.DataFrame(patient_rows)
    res: Dict[str, object] = {}
    # median percentile rank (lower better): test < 0.5
    for label in ["all", "invocab", "heldout"]:
        col = f"median_percentile_{label}"
        if col in df.columns:
            w = stats.wilcoxon_vs_chance(df[col].values, CHANCE_PCT, alternative="less")
            ci = stats.bootstrap_ci(df[col].values)
            res[f"median_percentile_{label}"] = {**w, **{f"ci_{k}": v for k, v in ci.items()}}
    # near-miss similarity (higher better): test > 0 via per-patient minus null
    if "graded_near_miss_sim_mean" in df.columns and "near_miss_null_mean" in df.columns:
        delta = df["graded_near_miss_sim_mean"].values - df["near_miss_null_mean"].values
        res["near_miss_vs_null"] = stats.wilcoxon_vs_chance(delta, 0.0, alternative="greater")
    if "graded_ndcg_mean" in df.columns:
        res["ndcg"] = stats.bootstrap_ci(df["graded_ndcg_mean"].values)
    # nDCG vs matched permutation null (higher better): per-patient minus null
    if "graded_ndcg_mean" in df.columns and "ndcg_null_mean" in df.columns:
        delta = df["graded_ndcg_mean"].values - df["ndcg_null_mean"].values
        res["ndcg_vs_null"] = stats.wilcoxon_vs_chance(delta, 0.0, alternative="greater")
    return res


# ══════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════

def run(patients: Sequence[str], run_folder: str, task: str,
        embedding: str = "GloVe", headline_N: int = 5000,
        headline_variant: str = "matched", Ns: Sequence[int] = DEFAULT_NS,
        variants: Sequence[str] = ("matched", "raw"),
        n_folds: int = 5, held_out_frac: float = 0.3,
        n_perm: int = 1000, n_perm_graded: int = 200,
        relevance_kind: str = "wup", k_ndcg: int = 100,
        concreteness_file: Optional[str] = None, subtlex_file: Optional[str] = None,
        center: bool = True, save_figs: bool = True, seed: int = 0) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("[1/8] GloVe + norms ...", flush=True)
    glove = gallery_mod.load_glove()
    concreteness = gallery_mod.load_concreteness(concreteness_file)
    subtlex = gallery_mod.load_subtlex(subtlex_file)
    rel_graded = relevance.make_relevance_fn(relevance_kind)
    rel_category = relevance.make_relevance_fn("category")

    print("[2/8] Predicted embeddings (CV, zero-shot split) ...", flush=True)
    predictions = predict_io.make_predictions_all(
        patients, run_folder, task, embedding=embedding, n_folds=n_folds,
        held_out_frac=held_out_frac, seed=seed)

    stim_words = sorted(set(w for tp in predictions for w in tp.true_word))
    print(f"      stimulus wordset (union, clean): {len(stim_words)} words", flush=True)

    # Persist per-trial predictions (source data)
    pd.concat([tp.to_frame() for tp in predictions], ignore_index=True).to_csv(
        SRC_DIR / f"trial_predictions_{task}.csv", index=False)

    print(f"[3-6/8] Headline gallery ({headline_variant}, N={headline_N}) + metrics ...", flush=True)
    headline_gallery = gallery_mod.build_gallery(
        glove, stim_words, n=headline_N, variant=headline_variant,
        concreteness=concreteness, subtlex=subtlex)
    headline_gallery.meta.to_csv(SRC_DIR / f"gallery_meta_{headline_variant}_N{headline_N}.csv", index=False)
    print(f"      gallery N_effective={headline_gallery.N} "
          f"(stimulus={int(headline_gallery.meta['is_stimulus'].sum())})", flush=True)

    patient_rows = []
    for tp in predictions:
        row = evaluate_patient(tp, headline_gallery, rel_graded, rel_category,
                               ks=DEFAULT_KS, center=center, k_ndcg=k_ndcg,
                               n_perm=n_perm, n_perm_graded=n_perm_graded, seed=seed)
        patient_rows.append(row)
        print(f"      {tp.patient}: median%rank all={row.get('median_percentile_all'):.3f} "
              f"invocab={row.get('median_percentile_invocab'):.3f} "
              f"heldout={row.get('median_percentile_heldout'):.3f} "
              f"top10_all={row.get('top10_all'):.3f} "
              f"nDCG={row.get('graded_ndcg_mean'):.3f}", flush=True)

    patient_df = pd.DataFrame(patient_rows)
    patient_df.to_csv(SRC_DIR / f"per_patient_metrics_{task}.csv", index=False)

    grp = group_inference(patient_rows)
    with open(SRC_DIR / f"group_inference_{task}.json", "w", encoding="utf-8") as f:
        json.dump(grp, f, indent=2, default=float)

    # Group panel table for figures (in-vocab vs held-out; near-miss)
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

    print("[7/8] Sweeps (N x variant) ...", flush=True)
    sweep_df = sweeps.sweep_gallery_size(
        predictions, glove, stim_words, Ns=Ns, variants=variants,
        concreteness=concreteness, ks=DEFAULT_KS, center=center, subtlex=subtlex)
    sweep_df.to_csv(SRC_DIR / f"sweep_{task}.csv", index=False)
    sweep_summary = sweeps.summarize_sweep(sweep_df, ks=DEFAULT_KS)
    sweep_summary.to_csv(SRC_DIR / f"sweep_summary_{task}.csv", index=False)

    if save_figs:
        print("[8/8] Figures ...", flush=True)
        figures.plot_metric_vs_N(sweep_summary, FIG_DIR / f"01_metric_vs_N_{task}.png")
        figures.plot_cmc(sweep_df, headline_N, headline_variant,
                         FIG_DIR / f"02_cmc_{task}.png", ks=DEFAULT_KS)
        figures.plot_invocab_vs_heldout(group_panel, FIG_DIR / f"03_invocab_vs_heldout_{task}.png")
        figures.plot_near_miss(group_panel, FIG_DIR / f"04_near_miss_{task}.png")
        figures.qualitative_table(
            predictions, headline_gallery, rel_graded,
            SRC_DIR / f"qualitative_top5_{task}.csv",
            out_html=FIG_DIR / f"06_qualitative_top5_{task}.html",
            center=center, seed=seed)

    meta = {
        "timestamp": ts, "task": task, "run_folder": run_folder,
        "patients": list(patients), "embedding": embedding,
        "headline_N": headline_N, "headline_variant": headline_variant,
        "Ns": list(Ns), "variants": list(variants), "n_folds": n_folds,
        "held_out_frac": held_out_frac, "n_perm": n_perm,
        "n_perm_graded": n_perm_graded, "relevance_kind": relevance_kind,
        "k_ndcg": k_ndcg, "center": center, "seed": seed,
        "concreteness_file": concreteness_file, "subtlex_file": subtlex_file,
        "command": "python -m main.analysis.open_vocab_retrieval.run " + " ".join(sys.argv[1:]),
    }
    with open(SRC_DIR / f"run_metadata_{task}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. Figures -> {FIG_DIR}\n      Source data -> {SRC_DIR}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Open-vocabulary (zero-shot) word retrieval")
    p.add_argument("--patient", default=None, help="One patient (default: all shared)")
    p.add_argument("--task", default="picture", choices=["picture", "auditory"])
    p.add_argument("--pic-run", default=PIC_RUN_DEFAULT)
    p.add_argument("--aud-run", default=AUD_RUN_DEFAULT)
    p.add_argument("--embedding", default="GloVe")
    p.add_argument("--headline-N", type=int, default=5000)
    p.add_argument("--headline-variant", default="matched", choices=["matched", "raw"])
    p.add_argument("--Ns", type=int, nargs="+", default=list(DEFAULT_NS))
    p.add_argument("--variants", nargs="+", default=["matched", "raw"],
                   choices=["matched", "raw"])
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--held-out-frac", type=float, default=0.3)
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--n-perm-graded", type=int, default=200)
    p.add_argument("--relevance", default="wup",
                   choices=list(relevance.RELEVANCE_KINDS))
    p.add_argument("--k-ndcg", type=int, default=100)
    p.add_argument("--concreteness-file", default=None)
    p.add_argument("--subtlex-file", default=None)
    p.add_argument("--no-center", action="store_true",
                   help="Disable gallery-centroid mean-centring (not recommended).")
    p.add_argument("--no-figs", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_folder = args.pic_run if args.task == "picture" else args.aud_run
    task = "picture_naming" if args.task == "picture" else "auditory_naming"
    # Cohort follows the TASK. The picture arm needs no auditory data, so it runs on every
    # picture participant (N=13); only the auditory arm is restricted to those who have both
    # (N=8). Before 2026-07-30 both used SHARED_PATIENTS, so a bare invocation silently
    # rebuilt trial_predictions_picture_naming.csv at 8 patients instead of 13 -- and this
    # writer OVERWRITES rather than appends, so the larger cohort was simply lost.
    default_patients = PICTURE_PATIENTS if args.task == "picture" else SHARED_PATIENTS
    patients = [args.patient] if args.patient else default_patients
    print(f"[cohort] task={task}  n_patients={len(patients)}  {patients}", flush=True)

    run(patients, run_folder, task, embedding=args.embedding,
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
