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
    load_phoneme_embeddings_for_patient,
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
    PHONEME_EMBEDDINGS,
    load_patient_combined,
    get_features_per_bin,
    get_shared_vocabulary,
    fit_source_pls,
    decoder_matrix,
    decode_from_T,
    compute_T_anchors,
    compute_Y_anchors,
    find_peak_bin_source,
    stratified_train_test_split,
    sample_k_anchor_words,
    build_anchored_inputs,
    fit_ridge,
    predict_arm1_embedding,
    predict_arm2_embedding,
    score_predictions,
    build_map_record,
    save_map_records,
    get_out_dir,
    header,
    step,
)


def run_one(
    source_pdata: dict,
    source_phon: Dict[str, np.ndarray],
    source_X_per_bin: List[np.ndarray],
    target_patient: str,
    target_pdata: dict,
    target_phon: Dict[str, np.ndarray],
    target_X_per_bin: List[np.ndarray],
    embedding: str,
    args,
    rng_master: np.random.Generator,
):
    """Run the full sweep for one (target, embedding) pair.

    Returns (rows_df, map_records, map_metadata).
    """
    Y_src = source_phon[embedding]
    labels_src = np.asarray(source_pdata["clean_answer_labels"])
    cats_src = np.asarray(source_pdata["clean_word_category"])

    step(f"[{target_patient}/{embedding}] finding source peak bin …")
    peak_bin, src_metric_per_bin = find_peak_bin_source(
        source_X_per_bin, Y_src, labels_src, cats_src,
        n_components=args.pls_components,
        holdout_frac=0.2,
        rng=rng_master,
        metric=args.peak_metric,
    )
    step(f"[{target_patient}/{embedding}]   peak bin = {peak_bin}  "
         f"({args.peak_metric} = {src_metric_per_bin[peak_bin]:.4f})")

    X_src_peak = source_X_per_bin[peak_bin]
    pls_src = fit_source_pls(X_src_peak, Y_src, n_components=args.pls_components)

    labels_tgt = np.asarray(target_pdata["clean_answer_labels"])
    cats_tgt = np.asarray(target_pdata["clean_word_category"])
    shared_vocab = get_shared_vocabulary([labels_src, labels_tgt])
    if len(shared_vocab) < 3:
        step(f"  WARNING: only {len(shared_vocab)} shared words; skipping.")
        return pd.DataFrame(), [], {}

    T_anchors = compute_T_anchors(pls_src, X_src_peak, labels_src, shared_vocab)
    Y_src_anchors = compute_Y_anchors(Y_src, labels_src, shared_vocab)

    Y_tgt = target_phon[embedding]
    db_embeds, unique_words_db, word_to_cat_idx, unique_cats, word_to_idx = \
        build_retrieval_db(Y_tgt, labels_tgt, cats_tgt)

    k_values = sorted({k for k in args.k_values if 2 <= k <= len(shared_vocab)})
    if not k_values:
        step(f"  WARNING: shared vocab ({len(shared_vocab)}) too small for any k.")
        return pd.DataFrame(), [], {}
    step(f"  shared vocab = {len(shared_vocab)} words;  sweeping k in {k_values}")

    rows: List[dict] = []
    map_records: List[dict] = []
    n_bins = len(target_X_per_bin)

    for k in k_values:
        for b in range(args.n_bootstrap_peak):
            rng = np.random.default_rng(seed=(args.seed, k, b))
            train_idx, test_idx = stratified_train_test_split(
                labels_tgt, test_frac=args.test_frac, rng=rng,
            )
            if len(train_idx) < k or len(test_idx) < 2:
                continue
            anchor_words = sample_k_anchor_words(
                labels_tgt[train_idx], shared_vocab, k, rng,
            )
            if len(anchor_words) < 3:
                continue
            X_tgt_train_peak = target_X_per_bin[peak_bin][train_idx]
            HGA_anch, kept_words = build_anchored_inputs(
                X_tgt_train_peak, labels_tgt[train_idx], anchor_words,
            )
            if len(kept_words) < 3:
                continue
            T_targets = np.stack([T_anchors[w] for w in kept_words], axis=0)
            Y_targets_for_arm2 = np.stack([Y_src_anchors[w] for w in kept_words], axis=0)

            ridge_arm1 = fit_ridge(HGA_anch, T_targets, alpha=args.ridge_alpha)
            ridge_arm2 = fit_ridge(HGA_anch, Y_targets_for_arm2, alpha=args.ridge_alpha)

            if args.save_maps and b < args.n_bootstrap_maps:
                map_records.append(build_map_record(
                    arm="transfer", k=k, bootstrap_id=b,
                    ridge_model=ridge_arm1, HGA_anchored=HGA_anch,
                    anchor_words=kept_words, T_targets=T_targets,
                ))
                map_records.append(build_map_record(
                    arm="no_transfer", k=k, bootstrap_id=b,
                    ridge_model=ridge_arm2, HGA_anchored=HGA_anch,
                    anchor_words=kept_words, Y_targets=Y_targets_for_arm2,
                ))

            X_tgt_test_peak = target_X_per_bin[peak_bin][test_idx]
            Yhat_a1 = predict_arm1_embedding(pls_src, ridge_arm1, X_tgt_test_peak)
            Yhat_a2 = predict_arm2_embedding(ridge_arm2, X_tgt_test_peak)
            s1 = score_predictions(
                Yhat_a1, labels_tgt[test_idx], cats_tgt[test_idx],
                db_embeds, unique_words_db, word_to_cat_idx, unique_cats,
                word_to_idx, kept_words,
            )
            s2 = score_predictions(
                Yhat_a2, labels_tgt[test_idx], cats_tgt[test_idx],
                db_embeds, unique_words_db, word_to_cat_idx, unique_cats,
                word_to_idx, kept_words,
            )
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
            rows.append(dict(common_meta, arm="transfer", **s1))
            rows.append(dict(common_meta, arm="no_transfer", **s2))

            if b < args.n_bootstrap_timecourse:
                for t in range(n_bins):
                    if t == peak_bin:
                        continue
                    X_tgt_test_t = target_X_per_bin[t][test_idx]
                    Yhat_a1_t = predict_arm1_embedding(pls_src, ridge_arm1, X_tgt_test_t)
                    Yhat_a2_t = predict_arm2_embedding(ridge_arm2, X_tgt_test_t)
                    s1_t = score_predictions(
                        Yhat_a1_t, labels_tgt[test_idx], cats_tgt[test_idx],
                        db_embeds, unique_words_db, word_to_cat_idx, unique_cats,
                        word_to_idx, kept_words,
                    )
                    s2_t = score_predictions(
                        Yhat_a2_t, labels_tgt[test_idx], cats_tgt[test_idx],
                        db_embeds, unique_words_db, word_to_cat_idx, unique_cats,
                        word_to_idx, kept_words,
                    )
                    meta_t = dict(common_meta, time_bin=t, is_peak=False)
                    rows.append(dict(meta_t, arm="transfer", **s1_t))
                    rows.append(dict(meta_t, arm="no_transfer", **s2_t))

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
        "n_features_target": int(target_X_per_bin[peak_bin].shape[1]),
        "T_anchors_full": {str(k_): v.astype(np.float32) for k_, v in T_anchors.items()},
        "pls_decoder": decoder_matrix(pls_src).astype(np.float32),
        "pls_y_mean": pls_src.y_mean_.astype(np.float32),
        "pls_x_mean": pls_src.x_mean_.astype(np.float32),
    }
    return pd.DataFrame(rows), map_records, map_metadata


def _build_source(args):
    src_tasks = ["picture_naming"]
    if args.pool_flashing:
        src_tasks.append("picture_flashing")
    header(f"Loading SOURCE {args.source_patient}  tasks = {src_tasks}")
    src_pdata = load_patient_combined(args.source_patient, src_tasks)
    src_phon = load_phoneme_embeddings_for_patient(src_pdata)
    src_pdata, src_phon = filter_nan_phoneme_trials(src_pdata, src_phon)
    step(f"  {args.source_patient}: trials={len(src_pdata['clean_answer_labels'])}"
         f"  channels={src_pdata['clean_data_binned'].shape[1]}"
         f"  bins={src_pdata['clean_data_binned'].shape[2]}")
    src_X_per_bin = get_features_per_bin(src_pdata, n_bins_history=N_BINS_HISTORY)
    return src_pdata, src_phon, src_X_per_bin


def _build_target(patient: str):
    header(f"Loading TARGET {patient}  task = {DEFAULT_TARGET_TASK}")
    tgt_pdata = load_patient_combined(patient, [DEFAULT_TARGET_TASK])
    tgt_phon = load_phoneme_embeddings_for_patient(tgt_pdata)
    tgt_pdata, tgt_phon = filter_nan_phoneme_trials(tgt_pdata, tgt_phon)
    step(f"  {patient}: trials={len(tgt_pdata['clean_answer_labels'])}"
         f"  channels={tgt_pdata['clean_data_binned'].shape[1]}"
         f"  bins={tgt_pdata['clean_data_binned'].shape[2]}")
    tgt_X_per_bin = get_features_per_bin(tgt_pdata, n_bins_history=N_BINS_HISTORY)
    return tgt_pdata, tgt_phon, tgt_X_per_bin


def main():
    parser = argparse.ArgumentParser(
        description="Few-shot cross-patient transfer learning experiment."
    )
    parser.add_argument("--source-patient", default=DEFAULT_SOURCE_PATIENT)
    parser.add_argument("--target-patients", nargs="+", default=DEFAULT_TARGET_PATIENTS)
    parser.add_argument("--embeddings", nargs="+", default=PHONEME_EMBEDDINGS,
                        choices=PHONEME_EMBEDDINGS + ["all"])
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
    parser.add_argument("--peak-metric", default="word_bal_acc",
                        choices=["word_bal_acc", "cat_indep_bal_acc", "cosine_mean"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip (target, embedding) combos that already have a CSV.")
    args = parser.parse_args()

    if args.embeddings == ["all"]:
        args.embeddings = PHONEME_EMBEDDINGS

    out_dir = get_out_dir(args.out_dir)
    header("CROSS-PATIENT FEW-SHOT TRANSFER LEARNING")
    print(f"  source       : {args.source_patient}"
          f"   pool_flashing={args.pool_flashing}")
    print(f"  targets      : {args.target_patients}")
    print(f"  embeddings   : {args.embeddings}")
    print(f"  k_values     : {args.k_values}")
    print(f"  bootstrap    : peak={args.n_bootstrap_peak}  "
          f"time={args.n_bootstrap_timecourse}  maps={args.n_bootstrap_maps}")
    print(f"  ridge alpha  : {args.ridge_alpha}")
    print(f"  pls comp     : {args.pls_components}")
    print(f"  peak metric  : {args.peak_metric}")
    print(f"  save maps    : {args.save_maps}")
    print(f"  out dir      : {out_dir}")

    src_pdata, src_phon, src_X_per_bin = _build_source(args)

    rng_master = np.random.default_rng(args.seed)

    all_dfs = []
    for tgt in args.target_patients:
        if tgt == args.source_patient:
            step(f"  Skipping target {tgt} (== source)")
            continue
        try:
            tgt_pdata, tgt_phon, tgt_X_per_bin = _build_target(tgt)
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

            t0 = time.time()
            df, map_records, map_meta = run_one(
                src_pdata, src_phon, src_X_per_bin,
                tgt, tgt_pdata, tgt_phon, tgt_X_per_bin,
                emb, args, rng_master,
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

        del tgt_pdata, tgt_phon, tgt_X_per_bin
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

        peak = combined[combined["is_peak"]]
        if len(peak) > 0:
            tbl = (
                peak.groupby(["target_patient", "embedding", "arm", "k"])
                .agg(cosine=("cosine_mean", "mean"),
                     word_acc=("word_bal_acc", "mean"),
                     cat_acc=("cat_indep_bal_acc", "mean"),
                     n_boot=("bootstrap_id", "nunique"))
                .reset_index()
            )
            print("\n  Peak-time summary:")
            for _, r in tbl.iterrows():
                print(f"    {r.target_patient:>4} {r.embedding:>10} {r.arm:>12} "
                      f"k={r.k:<3d}  cos={r.cosine:.3f}  word={r.word_acc:.3f}  "
                      f"cat={r.cat_acc:.3f}  ({int(r.n_boot)} boots)")


if __name__ == "__main__":
    main()
