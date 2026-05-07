# -*- coding: utf-8 -*-
"""
report.helper.embedding_norms — Embedding norm analysis.

Identifies which words have the smallest L2 norm in both raw and PCA-reduced
embedding space. These are the words ridge regression is biased toward
predicting when signal is weak (because L2 shrinkage pulls predictions toward
the origin, and the nearest neighbor to the origin is the smallest-norm word).

After the mean-centering fix (commit 0459d4c), the centroid is subtracted
before retrieval, but the bias may persist — just toward a different word.
After switching to PLS or cosine retrieval, this analysis serves as a
diagnostic to confirm the bias is reduced.
"""

import os
import numpy as np
import pandas as pd
from .config import EMBEDDING_NAMES
from .results_loader import load_pkl_raw

try:
    from sklearn.decomposition import PCA
except ImportError:
    os.system(f"pip install scikit-learn --break-system-packages -q")
    from sklearn.decomposition import PCA


def compute_norm_analysis(run_dir, top_n=10):
    """
    For each patient x embedding, find words with smallest L2 norm
    in both raw embedding space and PCA-reduced (centered) space.

    Parameters
    ----------
    run_dir : str
        Path to the run's results directory.
    top_n : int
        Number of top smallest-norm words to report per embedding.

    Returns
    -------
    pd.DataFrame
        Columns: patient, embedding, raw_norm_rank, raw_norm_word, raw_norm,
        centered_norm_rank, centered_norm_word, centered_norm.
    """
    patients = sorted([
        d for d in os.listdir(run_dir)
        if os.path.isdir(os.path.join(run_dir, d)) and d != '__pycache__'
           and not d.endswith('.json')
    ])
    records = []

    for patient in patients:
        pkl_path = os.path.join(run_dir, patient, 'semantic_regression_results.pkl')
        if not os.path.exists(pkl_path):
            continue
        try:
            data = load_pkl_raw(pkl_path)
            if data is None:
                continue
        except Exception as e:
            print(f"  {patient}: PKL error ({e}), skipping norm analysis")
            continue

        for emb in EMBEDDING_NAMES:
            if emb not in data.get('regressors', {}):
                continue
            br = data['regressors'][emb]

            try:
                raw_embeds = np.array(br._retrieval_db_embeds_raw)
                word_idx   = np.array(br._retrieval_db_word_idx)
                idx2word   = br.index_to_word  # numpy array or dict
            except AttributeError:
                continue

            def _word_label(wi):
                """Resolve a word index to its string label."""
                wi = int(wi)
                if isinstance(idx2word, dict):
                    return idx2word.get(wi, str(wi))
                elif hasattr(idx2word, '__getitem__') and wi < len(idx2word):
                    return str(idx2word[wi])
                return str(wi)

            # Raw norm ranking
            raw_norms = np.linalg.norm(raw_embeds, axis=1)
            raw_order = np.argsort(raw_norms)

            # PCA-reduced norm (refit PCA to avoid sklearn version mismatch
            # with the PCA object stored in the PKL)
            try:
                n_comp = min(10, raw_embeds.shape[0], raw_embeds.shape[1])
                pca = PCA(n_components=n_comp)
                pca_embeds = pca.fit_transform(raw_embeds)  # centered internally
                pca_norms  = np.linalg.norm(pca_embeds, axis=1)
                pca_order  = np.argsort(pca_norms)
            except Exception:
                pca_norms = raw_norms
                pca_order = raw_order

            for rank in range(min(top_n, len(raw_order))):
                ri = raw_order[rank]
                pi = pca_order[rank]
                records.append({
                    'patient':             patient,
                    'embedding':           emb,
                    'raw_norm_rank':       rank,
                    'raw_norm_word':       _word_label(word_idx[ri]),
                    'raw_norm':            float(raw_norms[ri]),
                    'centered_norm_rank':  rank,
                    'centered_norm_word':  _word_label(word_idx[pi]),
                    'centered_norm':       float(pca_norms[pi]),
                })
        print(f"  {patient}: norm analysis done", flush=True)

    df = pd.DataFrame(records)
    print(f"[Norm] {len(df)} rows")
    return df
