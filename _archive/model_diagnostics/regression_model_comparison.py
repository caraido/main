#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests.regression_model_comparison — Compare regression models on the same train/test splits.

Runs four model variants on the same data to isolate the effects of
nonlinearity (Nystroem kernel) and regularization strategy (Ridge vs PLS):

  A: Linear Ridge         (linear + L2 regularization)
  B: Kernel Ridge (KRR)   (nonlinear + L2 regularization)  ← current default
  C: PLS                  (linear + implicit regularization via n_components)
  D: Kernel PLS           (nonlinear + implicit regularization)

For each, reports:
  - Test R² at the best time bin
  - Category retrieval balanced accuracy
  - Word retrieval balanced accuracy
  - Prediction entropy (bias measure)
  - Per-model favorite word and fraction

This test answers two questions:
  1. Does the kernel (nonlinearity) help? Compare A vs B and C vs D.
  2. Does PLS fix the prediction bias? Compare A vs C and B vs D.

Usage:
    python -m analysis.regression_model_comparison --patients AA AZ --epochs 10
    python -m analysis.regression_model_comparison --patients AA VB --epochs 50 --out-dir tests/results

Output:
    tests/results/model_comparison.csv    — full results table
    tests/results/model_comparison.html   — visual summary
"""

import os
import sys
import argparse
import gc
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

# Add project root to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from analysis.helpers._phoneme_semantic_helpers import get_out_dir

# ─── Model configurations ────────────────────────────────────────────────────

MODELS = {
    'linear_ridge': {
        'label': 'A: Linear Ridge',
        'pipeline': lambda alpha: Pipeline([('ridge', Ridge(alpha=alpha))]),
        'use_pca': True,
        'nonlinear': False,
    },
    'krr': {
        'label': 'B: Kernel Ridge (KRR)',
        'pipeline': lambda alpha: Pipeline([
            ('nystroem', Nystroem(kernel='rbf')),
            ('ridge', Ridge(alpha=alpha)),
        ]),
        'use_pca': True,
        'nonlinear': True,
    },
    'pls': {
        'label': 'C: PLS',
        'pipeline': lambda n_comp: Pipeline([
            ('pls', PLSRegression(n_components=n_comp, scale=False)),
        ]),
        'use_pca': False,  # PLS handles dim reduction internally
        'nonlinear': False,
    },
    'kernel_pls': {
        'label': 'D: Kernel PLS',
        'pipeline': lambda n_comp: Pipeline([
            ('nystroem', Nystroem(kernel='rbf')),
            ('pls', PLSRegression(n_components=n_comp, scale=False)),
        ]),
        'use_pca': False,
        'nonlinear': True,
    },
}


def run_comparison(patient, pdata, embeddings, n_epochs=10, alpha=1.5,
                   pca_components=10, pls_components=10, closest='l2'):
    """
    Run all four model variants on one patient's data.

    Parameters
    ----------
    patient : str
        Patient ID.
    pdata : dict
        Patient data dict with 'clean_data_binned', 'clean_answer_labels', etc.
    embeddings : dict
        Dict of embedding_name → embedding_array.
    n_epochs : int
        Number of random train/test split epochs.
    alpha : float
        Ridge alpha (for Ridge-based models).
    pca_components : int
        PCA components (for Ridge-based models).
    pls_components : int
        PLS components (for PLS-based models).
    closest : str
        Retrieval metric: 'l2' or 'cosine'.

    Returns
    -------
    pd.DataFrame
        Results with columns: patient, model, embedding, test_r2, cat_bal_acc,
        word_bal_acc, pred_entropy_norm, top1_word, top1_frac.
    """
    from models.model import BasicRegressor

    EMBEDDING_NAMES = ['GloVe', 'FastText', 'Word2Vec', 'ConceptNet', 'DINOv2', 'SimCLR']
    X = pdata['clean_data_binned'].swapaxes(1, 2)
    labels = pdata['clean_answer_labels']
    category_labels = pdata['clean_word_category']

    records = []

    for model_key, model_cfg in MODELS.items():
        for emb_name in EMBEDDING_NAMES:
            if emb_name not in embeddings:
                continue

            print(f"    {model_cfg['label']} + {emb_name}...", flush=True)

            # Build pipeline
            if 'pls' in model_key:
                pipeline = model_cfg['pipeline'](pls_components)
            else:
                pipeline = model_cfg['pipeline'](alpha)

            # Build regressor
            y_reducer = PCA(pca_components) if model_cfg['use_pca'] else None
            br = BasicRegressor(pipeline, y_reducer=y_reducer)
            br.load_data(
                X, embeddings[emb_name],
                n_bins_history=10,
                labels=labels,
                category_labels=category_labels,
            )

            try:
                br.fit(
                    n_epochs=n_epochs,
                    parallel=None,  # sequential for reproducibility in tests
                    closest=closest,
                    compute_retrieval=True,
                    save_retrieval_pairs=True,
                    compute_top_k_accuracy=False,
                )
            except Exception as e:
                print(f"      ERROR: {e}")
                continue

            # Extract results at best category bin
            cat_acc = np.array(br.all_retrieval_category_balanced_acc)  # (n_epochs, n_bins)
            word_acc = np.array(br.all_retrieval_word_balanced_acc)
            test_scores = np.array(br.all_test_score)  # (n_epochs, n_bins)

            if cat_acc.size == 0:
                continue

            # Cosine similarity
            cosine_sim    = np.array(br.all_cosine_sim)       # (n_epochs, n_bins)
            train_cosine  = np.array(br.all_train_cosine_sim)
            chance_scores = np.array(br.all_chance)            # (n_epochs, n_bins)

            # Best bins
            cat_best_bin = int(np.argmax(cat_acc.mean(0)))
            word_best_bin = int(np.argmax(word_acc.mean(0)))
            r2_best_bin = int(np.argmax(test_scores.mean(0)))

            mean_cat    = float(cat_acc[:, cat_best_bin].mean())
            mean_word   = float(word_acc[:, word_best_bin].mean())
            mean_r2     = float(test_scores[:, r2_best_bin].mean())
            mean_chance = float(chance_scores[:, r2_best_bin].mean())
            delta_r2    = mean_r2 - mean_chance
            mean_cos    = float(cosine_sim[:, r2_best_bin].mean()) if cosine_sim.size > 0 else np.nan
            mean_train_cos = float(train_cosine[:, r2_best_bin].mean()) if train_cosine.size > 0 else np.nan

            # Prediction entropy from retrieval pairs
            entropy_norm = np.nan
            top1_word = '?'
            top1_frac = np.nan
            if hasattr(br, 'all_retrieval_pairs') and br.all_retrieval_pairs:
                # Collect predictions at best word bin
                pred_words = []
                for pair in br.all_retrieval_pairs:
                    if pair.get('bin_index') == word_best_bin:
                        if 'pred_word_labels' in pair:
                            pred_words.extend(pair['pred_word_labels'].tolist())
                        elif 'pred_word_idx' in pair:
                            pred_words.extend(pair['pred_word_idx'].tolist())

                if pred_words:
                    from collections import Counter
                    counts = Counter(pred_words)
                    total = sum(counts.values())
                    top1_word = counts.most_common(1)[0][0]
                    top1_frac = counts.most_common(1)[0][1] / total
                    n_unique_words = len(set(pred_words))

                    probs = np.array([c / total for c in counts.values()])
                    entropy = -np.sum(probs * np.log2(probs + 1e-12))
                    def _to_hashable(v):
                        if isinstance(v, np.ndarray):
                            return tuple(v.tolist())
                        return v
                    n_words = len(set(
                        _to_hashable(pair.get('true_word_labels', pair.get('true_word_idx', [])))
                        for pair in br.all_retrieval_pairs
                        if pair.get('bin_index') == word_best_bin
                    ))
                    # Use the number of unique true words as max entropy base
                    if isinstance(top1_word, (int, np.integer)):
                        # Map index to word if possible — handle both dict and numpy array
                        if hasattr(br, 'index_to_word'):
                            idx = int(top1_word)
                            itw = br.index_to_word
                            if isinstance(itw, dict):
                                top1_word = itw.get(idx, str(top1_word))
                            else:
                                top1_word = itw[idx] if 0 <= idx < len(itw) else str(top1_word)
                    entropy_norm = entropy / np.log2(max(n_unique_words, 2))

            records.append({
                'patient':           patient,
                'model':             model_key,
                'model_label':       model_cfg['label'],
                'nonlinear':         model_cfg['nonlinear'],
                'embedding':         emb_name,
                'test_r2':           mean_r2,
                'chance_r2':         mean_chance,
                'delta_r2':          delta_r2,
                'test_cosine':       mean_cos,
                'train_cosine':      mean_train_cos,
                'cat_bal_acc':       mean_cat,
                'word_bal_acc':      mean_word,
                'pred_entropy_norm': entropy_norm,
                'top1_word':         str(top1_word),
                'top1_frac':         top1_frac,
                'cat_best_bin':      cat_best_bin,
                'word_best_bin':     word_best_bin,
                'r2_best_bin':       r2_best_bin,
            })

            gc.collect()

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        prog='python -m analysis.regression_model_comparison',
        description='Compare regression models (KRR vs PLS vs linear variants)',
    )
    parser.add_argument('--patients', nargs='+', default=['AA', 'AZ'],
                        help='Patient IDs to test (default: AA AZ)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs (default: 10 for quick test)')
    parser.add_argument('--closest', choices=['l2', 'cosine'], default='l2',
                        help='Retrieval metric')
    parser.add_argument('--alpha', type=float, default=1.5,
                        help='Ridge alpha (default: 1.5)')
    parser.add_argument('--pca-components', type=int, default=10)
    parser.add_argument('--pls-components', type=int, default=10)
    parser.add_argument('--out-dir', default=None,
                        help='Output directory (default: main/tests/results)')
    args = parser.parse_args()

    os.chdir(_PROJECT_DIR)
    args.out_dir = get_out_dir(args.out_dir)

    # Import project modules (needs to be in main/ directory)
    from semantic_regression import load_patient_data, load_shared_embedding_models, build_patient_embeddings

    print("Loading shared embedding models...")
    shared = load_shared_embedding_models()

    all_results = []
    for patient in args.patients:
        print(f"\n{'='*60}")
        print(f"Patient: {patient}")
        print(f"{'='*60}")

        pdata = load_patient_data(patient)
        embeddings = build_patient_embeddings(pdata, shared)

        df = run_comparison(
            patient, pdata, embeddings,
            n_epochs=args.epochs,
            alpha=args.alpha,
            pca_components=args.pca_components,
            pls_components=args.pls_components,
            closest=args.closest,
        )
        all_results.append(df)
        gc.collect()

    results = pd.concat(all_results, ignore_index=True)
    csv_path = os.path.join(args.out_dir, 'model_comparison.csv')
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    MODEL_ORDER = ['linear_ridge', 'krr', 'pls', 'kernel_pls']
    HDR = f"    {'model':15s}  {'R²':>7}  {'chance':>7}  {'ΔR²':>7}  {'cos':>7}  {'cat_acc':>7}  {'word_acc':>8}"
    for patient in args.patients:
        sub = results[results.patient == patient]
        print(f"\n  {patient}:")
        print(HDR)
        for model in MODEL_ORDER:
            m = sub[sub.model == model]
            if len(m) == 0:
                continue
            r2     = m['test_r2'].mean()
            chance = m['chance_r2'].mean()
            dr2    = m['delta_r2'].mean()
            cos    = m['test_cosine'].mean()
            cat    = m['cat_bal_acc'].mean()
            word   = m['word_bal_acc'].mean()
            print(f"    {model:15s}  {r2:7.4f}  {chance:7.4f}  {dr2:7.4f}  {cos:7.4f}  {cat:7.4f}  {word:8.4f}")

    def _delta(sub, m1, m2, col):
        a = sub[sub.model == m1][col].mean()
        b = sub[sub.model == m2][col].mean()
        return a, b, b - a

    # Nonlinearity effect: linear→KRR and PLS→KernelPLS
    print("\n  Nonlinearity effect (kernel vs linear):")
    print(f"    {'patient':6s}  {'pair':28s}  {'R²Δ':>7}  {'ΔR²Δ':>7}  {'cosΔ':>7}  {'catΔ':>7}  {'wordΔ':>7}")
    for patient in args.patients:
        sub = results[results.patient == patient]
        for pair, m_lin, m_kern in [
            ('Linear Ridge → KRR',        'linear_ridge', 'krr'),
            ('PLS → Kernel PLS',           'pls',          'kernel_pls'),
        ]:
            r2a, r2b, r2d     = _delta(sub, m_lin, m_kern, 'test_r2')
            dr2a, dr2b, dr2d  = _delta(sub, m_lin, m_kern, 'delta_r2')
            cosd              = _delta(sub, m_lin, m_kern, 'test_cosine')[2]
            ca,  cb,  catd    = _delta(sub, m_lin, m_kern, 'cat_bal_acc')
            wa,  wb,  wordd   = _delta(sub, m_lin, m_kern, 'word_bal_acc')
            print(f"    {patient:6s}  {pair:28s}  {r2d:+7.4f}  {dr2d:+7.4f}  {cosd:+7.4f}  {catd:+7.4f}  {wordd:+7.4f}")

    # PLS effect: Ridge→PLS and KRR→KernelPLS
    print("\n  PLS effect (PLS vs Ridge regularization):")
    print(f"    {'patient':6s}  {'pair':28s}  {'R²Δ':>7}  {'ΔR²Δ':>7}  {'cosΔ':>7}  {'catΔ':>7}  {'wordΔ':>7}")
    for patient in args.patients:
        sub = results[results.patient == patient]
        for pair, m_ridge, m_pls in [
            ('Linear Ridge → PLS',         'linear_ridge', 'pls'),
            ('KRR → Kernel PLS',           'krr',          'kernel_pls'),
        ]:
            r2a, r2b, r2d     = _delta(sub, m_ridge, m_pls, 'test_r2')
            dr2a, dr2b, dr2d  = _delta(sub, m_ridge, m_pls, 'delta_r2')
            cosd              = _delta(sub, m_ridge, m_pls, 'test_cosine')[2]
            ca,  cb,  catd    = _delta(sub, m_ridge, m_pls, 'cat_bal_acc')
            wa,  wb,  wordd   = _delta(sub, m_ridge, m_pls, 'word_bal_acc')
            print(f"    {patient:6s}  {pair:28s}  {r2d:+7.4f}  {dr2d:+7.4f}  {cosd:+7.4f}  {catd:+7.4f}  {wordd:+7.4f}")


if __name__ == '__main__':
    main()
