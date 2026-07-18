# -*- coding: utf-8 -*-
"""
tests/pca_and_deflation_retrieval.py
=====================================
Investigate where word-level information lives in neural feature space.

Two complementary tests probe why vanilla (LOO nearest-centroid) word retrieval
in raw neural features outperforms model-based semantic regression:

  **Test 1 — PCA Vanilla Retrieval**
    Apply PCA (default 10 components) per time bin, then run LOO
    nearest-centroid retrieval on the PCs alone.  No regression model.
    If accuracy is preserved, word-separable structure is low-dimensional.

  **Test 2 — Deflated Vanilla Retrieval**
    Fit PLS(n_components) on (neural_features, semantic_embedding) to
    find the "semantic subspace", project it out, then run LOO retrieval
    on the residual.  If accuracy survives, word information is orthogonal
    to the embedding semantics; if it drops, the two overlap.

A baseline "vanilla" condition (unmodified features) is always run so that
all three conditions share identical trial sets and label bookkeeping.

Usage (from main/):
    python -m analysis.pca_and_deflation_retrieval
    python -m analysis.pca_and_deflation_retrieval --patients VB LH
    python -m analysis.pca_and_deflation_retrieval --pca-components 5 10 20
    python -m analysis.pca_and_deflation_retrieval --pls-components 10 --embeddings GloVe FastText

Output:
    tests/results/pca_deflation_{patient}.csv   (per-patient, all bins)
    tests/results/pca_deflation_all.csv         (combined)
"""

import argparse
import gc
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

warnings.filterwarnings("ignore")

# ── project imports ──────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MAIN_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _MAIN_DIR)

from semantic_vanilla_retrieval import (
    NeuralRetriever,
    load_patient_data,
    N_BINS_HISTORY,
)
from utils.utils import reformat

from analysis.helpers._phoneme_semantic_helpers import (
    header, step, get_out_dir, discover_patients,
)

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_PATIENTS = ['VB', 'LH', 'WBH']
DEFAULT_PCA_COMPONENTS = [10]
DEFAULT_PLS_COMPONENTS = 10
DEFAULT_SHUFFLES = 50
DEFAULT_EMBEDDINGS = ['GloVe', 'FastText', 'Word2Vec', 'ConceptNet']


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _extract_retriever_scores(nr, condition, embedding=''):
    """Extract per-bin metrics from a fitted NeuralRetriever into row dicts."""
    n_bins = nr.n_bins
    rows = []
    for b in range(n_bins):
        row = {
            'condition':  condition,
            'embedding':  embedding,
            'bin':        b,
            'word_top1':                float(nr.all_retrieval_top1[0, b]),
            'word_top3':                float(nr.all_retrieval_top3[0, b]),
            'word_top5':                float(nr.all_retrieval_top5[0, b]),
            'word_balanced_acc':        float(nr.all_retrieval_word_balanced_acc[0, b]),
            'word_f1':                  float(nr.all_retrieval_word_f1[0, b]),
            'category_balanced_acc':    float(nr.all_retrieval_category_balanced_acc[0, b]),
            'category_balanced_acc_indep': float(nr.all_retrieval_category_indep_balanced_acc[0, b]),
            'category_f1':              float(nr.all_retrieval_category_f1[0, b]),
            # Chance
            'chance_word_balanced_acc':     float(np.nanmean(nr.all_retrieval_chance_word_balanced_acc[:, b])),
            'chance_category_balanced_acc': float(np.nanmean(nr.all_retrieval_category_chance_balanced_acc[:, b])),
            'chance_word_balanced_acc_std': float(np.nanstd(nr.all_retrieval_chance_word_balanced_acc[:, b])),
            'chance_category_balanced_acc_std': float(np.nanstd(nr.all_retrieval_category_chance_balanced_acc[:, b])),
        }
        rows.append(row)
    return rows


def _run_retrieval_with_features(X_features, pdata, n_shuffles, closest='cosine'):
    """Create a NeuralRetriever, inject pre-computed features, fit, and return it.

    Parameters
    ----------
    X_features : list of ndarray
        Per-bin feature matrices, each (n_trials, n_feat).
    pdata : dict
        Patient data dict (from load_patient_data).
    n_shuffles : int
        Number of label permutations for chance distribution.
    closest : str
        'cosine' or 'l2'.

    Returns
    -------
    NeuralRetriever
    """
    # Use NeuralRetriever's load_data for label bookkeeping, then override features
    X_raw = pdata['clean_data_binned'].swapaxes(1, 2)
    nr = NeuralRetriever()
    nr.load_data(
        X_raw,
        n_bins_history=N_BINS_HISTORY,
        labels=pdata['target_concept'],
        category_labels=pdata['clean_word_category'],
    )
    # Override the reformatted features with our custom ones
    nr.X_to_use = X_features
    nr.n_bins = len(X_features)
    nr.fit(n_shuffles=n_shuffles, closest=closest, save_retrieval_pairs=False)
    return nr


# ═════════════════════════════════════════════════════════════════════════════
#  Test 1: PCA Vanilla Retrieval
# ═════════════════════════════════════════════════════════════════════════════

def run_pca_retrieval(pdata, n_pca_components, n_shuffles):
    """PCA per bin → LOO retrieval on top PCs.

    Parameters
    ----------
    pdata : dict
        Patient data.
    n_pca_components : int
        Number of PCA components to keep.
    n_shuffles : int
        Chance shuffles.

    Returns
    -------
    NeuralRetriever
    """
    step(f"  PCA vanilla retrieval (n_components={n_pca_components})")
    X = pdata['clean_data_binned'].swapaxes(1, 2)  # (n_trials, n_bins, n_ch)
    X_reformatted = reformat(X, N_BINS_HISTORY)

    X_pca = []
    for b, X_b in enumerate(X_reformatted):
        n_comp = min(n_pca_components, X_b.shape[0], X_b.shape[1])
        pca = PCA(n_components=n_comp)
        X_pca.append(pca.fit_transform(X_b))

    return _run_retrieval_with_features(X_pca, pdata, n_shuffles)


# ═════════════════════════════════════════════════════════════════════════════
#  Test 2: Deflated Vanilla Retrieval
# ═════════════════════════════════════════════════════════════════════════════

def compute_semantic_projection(X_feat, Y_sem, n_components):
    """Fit PLS on (X, Y_semantic) and return the orthogonal projection matrix.

    Returns
    -------
    proj : ndarray (n_features, n_features)
        Orthogonal projection onto the semantic subspace.
    """
    n_comp = min(n_components, X_feat.shape[0] - 1, X_feat.shape[1], Y_sem.shape[1])
    pls = PLSRegression(n_components=n_comp, scale=False)
    pls.fit(X_feat, Y_sem)
    W = pls.x_rotations_  # (n_features, n_components)
    proj = W @ np.linalg.pinv(W)
    return proj


def run_deflated_retrieval(pdata, Y_sem, emb_name, n_pls_components, n_shuffles):
    """Remove semantic subspace per bin → LOO retrieval on residual.

    Parameters
    ----------
    pdata : dict
        Patient data.
    Y_sem : ndarray (n_trials, D)
        Semantic embedding aligned to trials.
    emb_name : str
        Name of the embedding (for logging).
    n_pls_components : int
        PLS components for semantic subspace estimation.
    n_shuffles : int
        Chance shuffles.

    Returns
    -------
    NeuralRetriever
    """
    step(f"  Deflated retrieval (embedding={emb_name}, pls_comp={n_pls_components})")
    X = pdata['clean_data_binned'].swapaxes(1, 2)
    X_reformatted = reformat(X, N_BINS_HISTORY)

    X_deflated = []
    for b, X_b in enumerate(X_reformatted):
        try:
            proj = compute_semantic_projection(X_b, Y_sem, n_pls_components)
            X_deflated.append(X_b - X_b @ proj)
        except Exception:
            X_deflated.append(X_b)

    return _run_retrieval_with_features(X_deflated, pdata, n_shuffles)


# ═════════════════════════════════════════════════════════════════════════════
#  Per-patient runner
# ═════════════════════════════════════════════════════════════════════════════

def run_patient(patient, pdata, shared_models, args):
    """Run all conditions for one patient and return a DataFrame."""
    out_dir = get_out_dir(args.out_dir)
    records = []

    # ── Baseline: vanilla retrieval (unmodified features) ────────────────
    header(f"[{patient}] Vanilla baseline")
    X = pdata['clean_data_binned'].swapaxes(1, 2)
    X_reformatted = reformat(X, N_BINS_HISTORY)
    nr_vanilla = _run_retrieval_with_features(X_reformatted, pdata, args.shuffles)
    for row in _extract_retriever_scores(nr_vanilla, 'vanilla'):
        row['patient'] = patient
        records.append(row)
    best_b = int(np.nanargmax(nr_vanilla.all_retrieval_word_balanced_acc[0]))
    step(f"  Vanilla peak word_bal_acc = "
         f"{nr_vanilla.all_retrieval_word_balanced_acc[0, best_b]:.4f} @ bin {best_b}")
    del nr_vanilla
    gc.collect()

    # ── Test 1: PCA vanilla retrieval ────────────────────────────────────
    for n_pca in args.pca_components:
        header(f"[{patient}] PCA retrieval (n_components={n_pca})")
        nr_pca = run_pca_retrieval(pdata, n_pca, args.shuffles)
        cond_name = f'pca_{n_pca}'
        for row in _extract_retriever_scores(nr_pca, cond_name):
            row['patient'] = patient
            records.append(row)
        best_b = int(np.nanargmax(nr_pca.all_retrieval_word_balanced_acc[0]))
        step(f"  PCA({n_pca}) peak word_bal_acc = "
             f"{nr_pca.all_retrieval_word_balanced_acc[0, best_b]:.4f} @ bin {best_b}")
        del nr_pca
        gc.collect()

    # ── Test 2: Deflated vanilla retrieval ───────────────────────────────
    if shared_models is not None:
        from semantic_regression import build_patient_embeddings
        all_embeds = build_patient_embeddings(pdata, shared_models)

        for emb_name in args.embeddings:
            if emb_name not in all_embeds:
                step(f"  [WARN] Embedding '{emb_name}' not found, skipping")
                continue

            header(f"[{patient}] Deflated retrieval ({emb_name})")
            Y_sem = all_embeds[emb_name]
            nr_defl = run_deflated_retrieval(
                pdata, Y_sem, emb_name, args.pls_components, args.shuffles,
            )
            cond_name = f'deflated_{emb_name}'
            for row in _extract_retriever_scores(nr_defl, cond_name, emb_name):
                row['patient'] = patient
                records.append(row)
            best_b = int(np.nanargmax(nr_defl.all_retrieval_word_balanced_acc[0]))
            step(f"  Deflated({emb_name}) peak word_bal_acc = "
                 f"{nr_defl.all_retrieval_word_balanced_acc[0, best_b]:.4f} @ bin {best_b}")
            del nr_defl
            gc.collect()

        del all_embeds
        gc.collect()

    df = pd.DataFrame(records)
    pat_csv = os.path.join(out_dir, f'pca_deflation_{patient}.csv')
    df.to_csv(pat_csv, index=False)
    step(f"  Saved {pat_csv}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PCA & semantic-deflation vanilla retrieval tests")
    parser.add_argument('--patients', nargs='+', default=None,
                        help='Patient IDs (default: VB LH WBH). '
                             'Use "all" to discover all available patients.')
    parser.add_argument('--shuffles', type=int, default=DEFAULT_SHUFFLES,
                        help=f'Label permutations for chance (default: {DEFAULT_SHUFFLES})')
    parser.add_argument('--pca-components', type=int, nargs='+',
                        default=DEFAULT_PCA_COMPONENTS,
                        help='PCA component counts to test (default: [10])')
    parser.add_argument('--pls-components', type=int,
                        default=DEFAULT_PLS_COMPONENTS,
                        help=f'PLS components for deflation (default: {DEFAULT_PLS_COMPONENTS})')
    parser.add_argument('--embeddings', nargs='+', default=DEFAULT_EMBEDDINGS,
                        help='Semantic embeddings for deflation '
                             f'(default: {DEFAULT_EMBEDDINGS})')
    parser.add_argument('--skip-deflation', action='store_true',
                        help='Skip Test 2 (deflation) — only run PCA test')
    parser.add_argument('--out-dir', default=None,
                        help='Output directory (default: tests/results/)')
    args = parser.parse_args()

    # Resolve patients
    if args.patients is None:
        patients = DEFAULT_PATIENTS
    elif len(args.patients) == 1 and args.patients[0].lower() == 'all':
        patients = discover_patients()
    else:
        patients = args.patients

    header("PCA & DEFLATION VANILLA RETRIEVAL TESTS")
    print(f"  patients     = {patients}")
    print(f"  shuffles     = {args.shuffles}")
    print(f"  pca_comp     = {args.pca_components}")
    print(f"  pls_comp     = {args.pls_components}")
    print(f"  embeddings   = {args.embeddings}")
    print(f"  skip_deflation = {args.skip_deflation}")

    # Load shared embedding models (only if deflation is enabled)
    shared = None
    if not args.skip_deflation:
        step("Loading shared semantic embedding models (one-time cost)...")
        from semantic_regression import load_shared_embedding_models
        shared = load_shared_embedding_models()

    all_dfs = []
    for patient in patients:
        header(f"═══ Patient: {patient} ═══")
        t0 = time.time()
        pdata = load_patient_data(patient)
        df = run_patient(patient, pdata, shared, args)
        all_dfs.append(df)
        step(f"  {patient} completed in {time.time() - t0:.0f}s")
        del pdata
        gc.collect()

    # Save combined CSV
    out_dir = get_out_dir(args.out_dir)
    combined = pd.concat(all_dfs, ignore_index=True)
    combined_csv = os.path.join(out_dir, 'pca_deflation_all.csv')
    combined.to_csv(combined_csv, index=False)

    # Print summary
    header("SUMMARY")
    for cond in combined['condition'].unique():
        sub = combined[combined['condition'] == cond]
        for pat in sub['patient'].unique():
            psub = sub[sub['patient'] == pat]
            best_b = psub.loc[psub['word_balanced_acc'].idxmax()]
            step(f"  {pat}/{cond}: peak word_bal_acc = "
                 f"{best_b['word_balanced_acc']:.4f} @ bin {int(best_b['bin'])}")

    step(f"\nCombined results: {combined_csv}")
    print("\nDone!")


if __name__ == '__main__':
    main()
