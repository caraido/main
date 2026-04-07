"""
report.helper.word_bias_analysis — Word prediction bias analysis.

Detects the "favorite-word" effect: when the ridge regression model predicts
the same word for nearly all trials (due to L2 shrinkage toward the embedding
centroid). Computes prediction entropy and identifies dominant predicted words.

See ADR_prediction_bias_fix.md for the full analysis of why this happens and
proposed solutions (cosine retrieval, PLS regression, contrastive learning).
"""

import os
import numpy as np
import pandas as pd
from .config import EMBEDDING_NAMES


def compute_word_bias(run_dir):
    """
    Analyze word prediction bias at the peak decoding time bin.

    For each patient x embedding, examines the top1 predicted word distribution
    at the best word-decoding time bin.

    Parameters
    ----------
    run_dir : str
        Path to the run's results directory.

    Returns
    -------
    pd.DataFrame
        Columns: patient, embedding, top1_word, top1_frac, n_unique_pred,
        n_words, pred_entropy_norm.

        ``pred_entropy_norm`` is Shannon entropy normalized to [0, 1] where
        1.0 = perfectly uniform predictions across all words.
    """
    patients = sorted([
        d for d in os.listdir(run_dir)
        if os.path.isdir(os.path.join(run_dir, d)) and d != '__pycache__'
           and not d.endswith('.json')
    ])
    records = []

    for patient in patients:
        patient_dir = os.path.join(run_dir, patient)
        top1_path   = os.path.join(patient_dir, 'top1_decoding_source_data.csv')
        pts_path    = os.path.join(patient_dir, 'per_time_scores.csv')
        if not os.path.exists(top1_path) or not os.path.exists(pts_path):
            print(f"  {patient}: missing CSV files, skipping bias analysis")
            continue

        print(f"  {patient}: computing bias...", flush=True)
        pts = pd.read_csv(pts_path)

        # Find best word-decoding time bin per embedding
        best_bins = {}
        for emb in EMBEDDING_NAMES:
            sub = pts[pts['embedding'] == emb]
            if len(sub) == 0:
                continue
            best_bins[emb] = int(sub.loc[sub['word_balanced_acc'].idxmax(), 'bin_index'])

        # Chunked read — only keep rows at best bins
        needed = set(best_bins.values())
        chunks = []
        for chunk in pd.read_csv(top1_path, chunksize=500_000):
            f = chunk[chunk['bin_index'].isin(needed)]
            if len(f):
                chunks.append(f)
        if not chunks:
            continue
        top1 = pd.concat(chunks, ignore_index=True)

        for emb, best_bin in best_bins.items():
            sub = top1[(top1['embedding'] == emb) & (top1['bin_index'] == best_bin)]
            if len(sub) == 0:
                continue

            counts    = sub['pred_word'].value_counts()
            top1_word = counts.index[0]
            top1_frac = counts.iloc[0] / len(sub)
            n_unique  = sub['pred_word'].nunique()
            n_words   = sub['true_word'].nunique()

            # Normalized Shannon entropy: H / log2(N_words)
            probs = counts.values / counts.values.sum()
            entropy = -np.sum(probs * np.log2(probs + 1e-12))
            entropy_norm = entropy / np.log2(n_words) if n_words > 1 else 0.0

            records.append({
                'patient':           patient,
                'embedding':         emb,
                'top1_word':         top1_word,
                'top1_frac':         top1_frac,
                'n_unique_pred':     n_unique,
                'n_words':           n_words,
                'pred_entropy_norm': entropy_norm,
            })

    df = pd.DataFrame(records)
    print(f"[Bias] {len(df)} rows")
    return df
