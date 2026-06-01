# -*- coding: utf-8 -*-
"""
tests/cross_patient_decoding/cross_patient_few_shot.py
======================================================
Few-shot cross-patient transfer-learning experiment.

Pretrains a PLS regressor on patient RB (optionally pooling picture_naming +
picture_flashing). For each target patient, sweeps over shot count k and
bootstraps a per-(k, b) ridge-regression alignment:

    Arm 1 (TRANSFER)    : ridge  HGA_X -> T_RB[anchor_word]
                          predict embedding via RB's frozen decoder
    Arm 2 (NO-TRANSFER) : ridge  HGA_X -> embedding[anchor_word]   (X-only)

Outputs (under main/test_results/):
    cross_patient_few_shot_{src}_to_{tgt}_{emb}.csv
    cross_patient_few_shot_{src}_to_{tgt}_{emb}_maps.pkl   (full M_X + SVD)

Usage (from main/):
    python -m tests.cross_patient_decoding.cross_patient_few_shot
    python -m tests.cross_patient_decoding.cross_patient_few_shot \\
        --target-patients VB WBH --embeddings panphon --n-bootstrap-peak 100
    python -m tests.cross_patient_decoding.cross_patient_few_shot --pool-flashing --resume
"""

from __future__ import annotations

import argparse
import gc
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
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

from tests.helpers._phoneme_semantic_helpers import (   # noqa: E402
    build_retrieval_db,
    N_BINS_HISTORY,
    PLS_COMPONENTS,
    load_semantic_embeddings_for_patient,
    filter_nan_phoneme_trials,
)
from tests.cross_patient_decoding._cross_patient_helpers import (   # noqa: E402
    DEFAULT_SOURCE_PATIENT,
    DEFAULT_TARGET_PATIENTS,
    DEFAULT_TARGET_TASK,
    DEFAULT_K_VALUES,
    DEFAULT_RIDGE_ALPHA,
    DEFAULT_TEST_FRAC,
    DEFAULT_N_BOOTSTRAP_PEAK,
    DEFAULT_N_BOOTSTRAP_TIMECOURSE,
    DEFAULT_N_BOOTSTRAP_MAPS,
    DEFAULT_EMBEDDINGS,
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_CCA_COMPONENTS,
    DEFAULT_ALIGN_START_BIN,
    DEFAULT_ARMS,
    load_patient_combined,
    get_features_per_bin,
    get_shared_vocabulary,
    fit_source_pls,
    encoder_matrix,
    decoder_matrix,
    compute_T_anchors,
    find_peak_bin_source,
    sample_k_anchor_words_from_vocab,
    word_based_split,
    build_arm1_train_inputs,
    build_arm1_cca_train_inputs,
    build_arm2_train_targets,
    fit_ridge,
    fit_kernel_ridge,
    fit_cca,
    fit_kernel_pls,
    fit_pca_from_lagged,
    compute_src_pca_anchors,
    build_arm_pca_train_inputs,
    predict_arm_pca_embedding,
    predict_arm1_embedding,
    predict_arm1_cca_embedding,
    predict_arm2_kpls,
    score_predictions,
    build_map_record,
    save_map_records,
    get_out_dir,
    header,
    step,
)


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
):
    """Run the full sweep for one (target, embedding) pair.

    Returns (rows_df, map_records, map_metadata).
    """
    Y_src = source_sem[embedding]
    labels_src = np.asarray(source_pdata["clean_answer_labels"])
    cats_src = np.asarray(source_pdata["clean_word_category"])

    step(f"[{target_patient}/{embedding}]   peak bin = {peak_bin}  "
         f"({args.peak_metric} = {src_metric_per_bin[peak_bin]:.4f})")

    X_src_peak = source_X_per_bin[peak_bin]
    pls_src = fit_source_pls(X_src_peak, Y_src, n_components=args.pls_components)

    labels_tgt = np.asarray(target_pdata["clean_answer_labels"])
    cats_tgt = np.asarray(target_pdata["clean_word_category"])

    # Mean T_RB (PLS latent scores) per shared word (regression target for Arm 1)
    shared_vocab_initial = get_shared_vocabulary([labels_src, labels_tgt])
    T_anchors = compute_T_anchors(pls_src, X_src_peak, labels_src, shared_vocab_initial)

    Y_tgt = target_sem[embedding]
    # Per-embedding NaN filter (words not in embedding vocab)
    valid_tgt = ~np.isnan(Y_tgt).any(axis=1)
    if not valid_tgt.all():
        n_drop = int((~valid_tgt).sum())
        step(f"  [{target_patient}/{embedding}] dropping {n_drop} target trials with NaN embedding")
        labels_tgt = labels_tgt[valid_tgt]
        cats_tgt = cats_tgt[valid_tgt]
        Y_tgt = Y_tgt[valid_tgt]
        tgt_X_all = [X[valid_tgt] for X in target_X_per_bin]
    else:
        tgt_X_all = list(target_X_per_bin)

    shared_vocab = get_shared_vocabulary([labels_src, labels_tgt])
    if len(shared_vocab) < 3:
        step(f"  WARNING: only {len(shared_vocab)} shared words; skipping.")
        return pd.DataFrame(), [], {}

    db_embeds, unique_words_db, word_to_cat_idx, unique_cats, word_to_idx = \
        build_retrieval_db(Y_tgt, labels_tgt, cats_tgt)

    # Need at least 1 word remaining outside the anchor set for the test set
    k_values = sorted({k for k in args.k_values if 2 <= k <= len(shared_vocab) - 1})
    if not k_values:
        step(f"  WARNING: shared vocab ({len(shared_vocab)}) too small for any k.")
        return pd.DataFrame(), [], {}
    step(f"  shared vocab = {len(shared_vocab)} words;  sweeping k in {k_values}")

    tgt_X_peak = tgt_X_all[peak_bin]

    # Alignment window: N_BINS_HISTORY bins ending at align_end_bin.
    # Default (args.align_start_bin < 0): align_end_bin = peak_bin so the
    # alignment window coincides with PLS_RB's training window. Ridge then
    # maps  target HGA @ peak  ->  source PLS latent @ peak, with no temporal
    # asymmetry between input and target.
    n_ch_src = source_pdata["clean_data_binned"].shape[1]
    n_ch_tgt = target_pdata["clean_data_binned"].shape[1]
    if args.align_start_bin is None or args.align_start_bin < 0:
        align_end_bin = int(peak_bin)
    else:
        align_end_bin = min(
            args.align_start_bin + N_BINS_HISTORY - 1,
            len(source_X_per_bin) - 1,
            len(tgt_X_all) - 1,
        )
    step(f"  alignment window: bins {max(0, align_end_bin - N_BINS_HISTORY + 1)}..{align_end_bin}  "
         f"(peak_bin = {peak_bin})")
    X_src_align = source_X_per_bin[align_end_bin]
    tgt_X_align = tgt_X_all[align_end_bin]

    # PCA models fitted once on all data (unsupervised — no label leakage)
    run_pca = "pca_align" in args.arms
    pca_src = pca_tgt = src_pca_anchors = None
    if run_pca:
        pca_src = fit_pca_from_lagged(X_src_align, n_ch_src, n_components=args.pca_components)
        pca_tgt = fit_pca_from_lagged(tgt_X_align, n_ch_tgt, n_components=args.pca_components)
        src_pca_anchors = compute_src_pca_anchors(
            pca_src, X_src_align, labels_src, shared_vocab,
        )

    rows: List[dict] = []
    map_records: List[dict] = []
    n_bins = len(tgt_X_all)

    for k in k_values:
        for b in range(args.n_bootstrap_peak):
            rng = np.random.default_rng(seed=(args.seed, k, b))

            # Sample k anchor words; ALL their trials become train, rest becomes test
            anchor_words = sample_k_anchor_words_from_vocab(shared_vocab, k, rng)
            train_idx, test_idx = word_based_split(labels_tgt, anchor_words)

            if len(train_idx) < 2 or len(test_idx) < 2:
                continue

            X_tgt_align_train = tgt_X_align[train_idx]
            labels_tgt_train = labels_tgt[train_idx]

            # Arm 1 (CCA): word-averaged (HGA_tgt, HGA_src) pairs per anchor word
            X_cca_tgt, X_cca_src, kept_words = build_arm1_cca_train_inputs(
                X_src_align, labels_src, X_tgt_align_train, labels_tgt_train, anchor_words,
            )
            if len(kept_words) < 2:
                continue

            # Arm 2: individual target trials → word embeddings (kernel PLS)
            # Re-query individual trials for kept_words (word-averaged CCA pairs
            # differ from the per-trial design kernel PLS needs)
            X_arm2_indiv, _, _ = build_arm1_train_inputs(
                X_tgt_align_train, labels_tgt_train, np.array(kept_words), T_anchors,
            )
            Y_arm2 = build_arm2_train_targets(
                labels_tgt_train, kept_words, Y_tgt[train_idx],
            )

            cca_arm1 = fit_cca(X_cca_tgt, X_cca_src, n_components=args.cca_components) \
                if "transfer" in args.arms else None
            kpls_arm2 = fit_kernel_pls(X_arm2_indiv, Y_arm2, n_components=args.pls_components) \
                if "no_transfer" in args.arms else None

            # Arm pca_align: target PCs → source PCs (linear ridge, d×d)
            ridge_pca = None
            kept_pca: List[str] = []
            if run_pca:
                X_pca, Y_pca, kept_pca = build_arm_pca_train_inputs(
                    pca_tgt, X_tgt_align_train, labels_tgt_train, anchor_words, src_pca_anchors,
                )
                ridge_pca = fit_ridge(X_pca, Y_pca, alpha=args.ridge_alpha) \
                    if len(kept_pca) >= 2 else None

            if args.save_maps and b < args.n_bootstrap_maps and cca_arm1 is not None:
                map_records.append(build_map_record(
                    arm="transfer", k=k, bootstrap_id=b,
                    ridge_model=cca_arm1, HGA_anchored=X_cca_tgt,
                    anchor_words=kept_words, pls_src=pls_src,
                ))

            X_tgt_test = tgt_X_peak[test_idx]

            common_meta = dict(
                source_patient=args.source_patient,
                target_patient=target_patient,
                embedding=embedding,
                k=k,
                bootstrap_id=b,
                time_bin=peak_bin,
                is_peak=True,
                n_train=len(train_idx),
                n_test=len(test_idx),
                n_shared_vocab=len(shared_vocab),
                n_anchor_kept=len(kept_words),
                source_peak_bin=peak_bin,
                source_peak_metric=src_metric_per_bin[peak_bin],
            )

            if cca_arm1 is not None:
                Yhat_a1 = predict_arm1_cca_embedding(pls_src, cca_arm1, X_tgt_test)
                s1 = score_predictions(
                    Yhat_a1, labels_tgt[test_idx], cats_tgt[test_idx],
                    db_embeds, unique_words_db, word_to_cat_idx, unique_cats,
                    word_to_idx, kept_words,
                )
                rows.append(dict(common_meta, arm="transfer", **s1))

            if kpls_arm2 is not None:
                Yhat_a2 = predict_arm2_kpls(kpls_arm2, X_tgt_test)
                s2 = score_predictions(
                    Yhat_a2, labels_tgt[test_idx], cats_tgt[test_idx],
                    db_embeds, unique_words_db, word_to_cat_idx, unique_cats,
                    word_to_idx, kept_words,
                )
                rows.append(dict(common_meta, arm="no_transfer", **s2))

            if run_pca and ridge_pca is not None:
                Yhat_pca = predict_arm_pca_embedding(
                    pls_src, ridge_pca, pca_src, pca_tgt, X_tgt_test,
                )
                s_pca = score_predictions(
                    Yhat_pca, labels_tgt[test_idx], cats_tgt[test_idx],
                    db_embeds, unique_words_db, word_to_cat_idx, unique_cats,
                    word_to_idx, kept_pca,
                )
                rows.append(dict(common_meta, arm="pca_align",
                                 n_anchor_kept=len(kept_pca), **s_pca))

            if b < args.n_bootstrap_timecourse:
                for t in range(n_bins):
                    if t == peak_bin:
                        continue
                    X_tgt_test_t = tgt_X_all[t][test_idx]
                    meta_t = dict(common_meta, time_bin=t, is_peak=False)

                    if cca_arm1 is not None:
                        Yhat_a1_t = predict_arm1_cca_embedding(pls_src, cca_arm1, X_tgt_test_t)
                        s1_t = score_predictions(
                            Yhat_a1_t, labels_tgt[test_idx], cats_tgt[test_idx],
                            db_embeds, unique_words_db, word_to_cat_idx, unique_cats,
                            word_to_idx, kept_words,
                        )
                        rows.append(dict(meta_t, arm="transfer", **s1_t))

                    if kpls_arm2 is not None:
                        Yhat_a2_t = predict_arm2_kpls(kpls_arm2, X_tgt_test_t)
                        s2_t = score_predictions(
                            Yhat_a2_t, labels_tgt[test_idx], cats_tgt[test_idx],
                            db_embeds, unique_words_db, word_to_cat_idx, unique_cats,
                            word_to_idx, kept_words,
                        )
                        rows.append(dict(meta_t, arm="no_transfer", **s2_t))

                    if run_pca and ridge_pca is not None:
                        Yhat_pca_t = predict_arm_pca_embedding(
                            pls_src, ridge_pca, pca_src, pca_tgt, X_tgt_test_t,
                        )
                        s_pca_t = score_predictions(
                            Yhat_pca_t, labels_tgt[test_idx], cats_tgt[test_idx],
                            db_embeds, unique_words_db, word_to_cat_idx, unique_cats,
                            word_to_idx, kept_pca,
                        )
                        rows.append(dict(meta_t, arm="pca_align",
                                         n_anchor_kept=len(kept_pca), **s_pca_t))

        gc.collect()
        step(f"    k={k:>3d} done; running rows = {len(rows)}")

    map_metadata = {
        "source_patient": args.source_patient,
        "target_patient": target_patient,
        "embedding": embedding,
        "peak_bin": int(peak_bin),
        "peak_metric": args.peak_metric,
        "pls_components": int(args.pls_components),
        "shared_vocab": list(map(str, shared_vocab)),
        "n_features_target": int(tgt_X_peak.shape[1]),
        "T_anchors_full": {k_: v.astype(np.float32) for k_, v in T_anchors.items()},
        "pls_decoder": decoder_matrix(pls_src).astype(np.float32),
        "pls_encoder": encoder_matrix(pls_src).astype(np.float32),
        "pls_y_mean": pls_src._y_mean.astype(np.float32),
        "pls_x_mean": pls_src._x_mean.astype(np.float32),
    }
    return pd.DataFrame(rows), map_records, map_metadata


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


def main():
    parser = argparse.ArgumentParser(
        description="Few-shot cross-patient transfer learning experiment."
    )
    parser.add_argument("--source-patient", default=DEFAULT_SOURCE_PATIENT)
    parser.add_argument("--target-patients", nargs="+", default=DEFAULT_TARGET_PATIENTS)
    parser.add_argument("--embeddings", nargs="+", default=DEFAULT_EMBEDDINGS,
                        choices=DEFAULT_EMBEDDINGS + ["all"])  # default: ['GloVe']
    parser.add_argument("--pool-flashing", action="store_true",
                        help="Pool picture_flashing onto source picture_naming "
                             "(requires matching channel & bin counts).")
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    parser.add_argument("--n-bootstrap-peak", type=int, default=DEFAULT_N_BOOTSTRAP_PEAK)
    parser.add_argument("--n-bootstrap-timecourse", type=int, default=DEFAULT_N_BOOTSTRAP_TIMECOURSE)
    parser.add_argument("--n-bootstrap-maps", type=int, default=DEFAULT_N_BOOTSTRAP_MAPS,
                        help="Bootstraps per k whose full M_X coef + SVD get pickled.")
    parser.add_argument("--save-maps", dest="save_maps", action="store_true", default=True,
                        help="Save M_X coef + SVD pickles for quiver plots (default ON).")
    parser.add_argument("--no-save-maps", dest="save_maps", action="store_false",
                        help="Disable saving M_X pickles.")
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    parser.add_argument("--ridge-alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    parser.add_argument("--pls-components", type=int, default=PLS_COMPONENTS)
    parser.add_argument("--cca-components", type=int, default=DEFAULT_CCA_COMPONENTS,
                        help="Number of CCA components for the Arm 1 transfer alignment "
                             "(clipped to min(n_anchor_words-1, n_features); default 10).")
    parser.add_argument("--peak-metric", default="cat_indep_bal_acc",
                        choices=["word_bal_acc", "cat_indep_bal_acc", "cosine_mean"])
    parser.add_argument("--arms", nargs="+", default=DEFAULT_ARMS,
                        choices=DEFAULT_ARMS,
                        help="Which arms to run (default: all). E.g. "
                             "--arms transfer no_transfer pca_align")
    parser.add_argument("--pca-components", type=int, default=DEFAULT_PCA_COMPONENTS,
                        help="PCA dimensionality for the pca_align arm (default 10).")
    parser.add_argument("--align-start-bin", type=int, default=DEFAULT_ALIGN_START_BIN,
                        help="First bin of the alignment window (N_BINS_HISTORY bins "
                             "wide). Default -1 = align window ends at the source "
                             "peak_bin, so PLS_RB latent and target features share "
                             "the same time window. Non-negative integer pins the "
                             "window: e.g. 0 = trial-onset-aligned, 10 = ~1s post.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip (target, embedding) combos that already have a CSV.")
    args = parser.parse_args()

    if args.embeddings == ["all"]:
        args.embeddings = DEFAULT_EMBEDDINGS

    out_dir = get_out_dir(args.out_dir)
    header("CROSS-PATIENT FEW-SHOT TRANSFER LEARNING")
    print(f"  source            : {args.source_patient}   "
          f"pool_flashing={args.pool_flashing}")
    print(f"  targets           : {args.target_patients}")
    print(f"  embeddings        : {args.embeddings}")
    print(f"  arms              : {args.arms}")
    print(f"  k_values          : {args.k_values}")
    print(f"  bootstrap         : peak={args.n_bootstrap_peak}  "
          f"time={args.n_bootstrap_timecourse}  maps={args.n_bootstrap_maps}")
    print(f"  ridge alpha       : {args.ridge_alpha}")
    print(f"  pls components    : {args.pls_components}")
    print(f"  cca components    : {args.cca_components}")
    print(f"  pca components    : {args.pca_components}")
    print(f"  peak metric       : {args.peak_metric}")
    print(f"  align_start_bin   : {args.align_start_bin}  "
          f"(-1 = align window ends at peak_bin)")
    print(f"  save maps         : {args.save_maps}")
    print(f"  out dir           : {out_dir}")

    # Shared embedding models loaded once (reused across patients)
    from semantic_regression import load_shared_embedding_models
    shared_models = load_shared_embedding_models()

    src_pdata, src_sem, src_X_per_bin = _build_source(args, shared_models)
    rng_master = np.random.default_rng(args.seed)

    # Find source peak bin ONCE per embedding (deterministic, no holdout)
    src_peak_cache = {}

    all_dfs = []
    for tgt in args.target_patients:
        if tgt == args.source_patient:
            step(f"  Skipping target {tgt} (== source)")
            continue
        try:
            tgt_pdata, tgt_sem, tgt_X_per_bin = _build_target(tgt, shared_models)
        except FileNotFoundError as e:
            step(f"  {tgt}: cannot load ({e}); skipping")
            continue

        for emb in args.embeddings:
            csv_path = os.path.join(
                out_dir,
                f"cross_patient_few_shot_{args.source_patient}_to_{tgt}_{emb}.csv",
            )
            if args.resume and os.path.exists(csv_path):
                step(f"  RESUME: {csv_path} exists, skipping.")
                all_dfs.append(pd.read_csv(csv_path))
                continue

            if emb not in src_peak_cache:
                Y_src = src_sem[emb]
                labels_src = np.asarray(src_pdata["clean_answer_labels"])
                cats_src = np.asarray(src_pdata["clean_word_category"])
                step(f"[source/{emb}] finding source peak bin (no holdout)...")
                peak_bin, src_metric_per_bin = find_peak_bin_source(
                    src_X_per_bin, Y_src, labels_src, cats_src,
                    n_components=args.pls_components,
                    metric=args.peak_metric,
                )
                step(f"  source peak_bin={peak_bin}  "
                     f"({args.peak_metric}={src_metric_per_bin[peak_bin]:.4f})")
                src_peak_cache[emb] = (peak_bin, src_metric_per_bin)
            peak_bin, src_metric_per_bin = src_peak_cache[emb]

            t0 = time.time()
            df, map_records, map_meta = run_one(
                src_pdata, src_sem, src_X_per_bin,
                tgt, tgt_pdata, tgt_sem, tgt_X_per_bin,
                emb, args, rng_master, peak_bin, src_metric_per_bin,
            )
            if len(df) == 0:
                step(f"  {tgt}/{emb}: no rows produced.")
                continue
            df.to_csv(csv_path, index=False)
            step(f"  saved {csv_path}  ({len(df)} rows, {time.time()-t0:.0f}s)")
            all_dfs.append(df)

            if args.save_maps and map_records:
                pkl_path = csv_path.replace(".csv", "_maps.pkl")
                save_map_records(map_records, map_meta, pkl_path)
                step(f"  saved {pkl_path}  ({len(map_records)} map records)")

        del tgt_pdata, tgt_sem, tgt_X_per_bin
        gc.collect()

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = os.path.join(
            out_dir, f"cross_patient_few_shot_{args.source_patient}_all.csv",
        )
        combined.to_csv(combined_path, index=False)
        header("SUMMARY")
        print(f"  Combined CSV: {combined_path}")
        print(f"  Total rows  : {len(combined)}")



if __name__ == "__main__":
    main()
