# -*- coding: utf-8 -*-
"""
report.helper.metric_dissociation — Metric dissociation analysis.

Examines whether R², category accuracy, and word accuracy peak at the same
time bin and with the same embedding model, or whether they dissociate.
Dissociation suggests different neural processes underlie regression quality
vs. categorical/word-level discrimination.
"""

import os
import numpy as np
import pandas as pd
from .config import EMBEDDING_NAMES


def compute_metric_dissociation(run_dir):
    """
    Compare best bins and peak values for R², category acc, and word acc.

    Parameters
    ----------
    run_dir : str
        Path to the run's results directory.

    Returns
    -------
    pd.DataFrame
        One row per patient x embedding with best bin and peak value for
        each metric: r2_best_bin, cat_best_bin, word_best_bin, best_r2,
        best_cat_acc, best_word_acc.
    """
    patients = sorted([
        d for d in os.listdir(run_dir)
        if os.path.isdir(os.path.join(run_dir, d)) and d != '__pycache__'
           and not d.endswith('.json')
    ])
    records = []

    for patient in patients:
        pts_path = os.path.join(run_dir, patient, 'per_time_scores.csv')
        if not os.path.exists(pts_path):
            continue
        pts = pd.read_csv(pts_path)

        for emb in pts['embedding'].unique():
            sub = pts[pts['embedding'] == emb]
            if len(sub) == 0:
                continue
            r2_idx   = sub['r2_mean'].idxmax()
            cat_idx  = sub['category_balanced_acc'].idxmax()
            word_idx = sub['word_balanced_acc'].idxmax()
            records.append({
                'patient':       patient,
                'embedding':     emb,
                'r2_best_bin':   int(sub.loc[r2_idx,   'bin_index']),
                'cat_best_bin':  int(sub.loc[cat_idx,  'bin_index']),
                'word_best_bin': int(sub.loc[word_idx, 'bin_index']),
                'best_r2':       float(sub.loc[r2_idx,   'r2_mean']),
                'best_cat_acc':  float(sub.loc[cat_idx,  'category_balanced_acc']),
                'best_word_acc': float(sub.loc[word_idx, 'word_balanced_acc']),
            })

    df = pd.DataFrame(records)
    print(f"[Dissoc] {len(df)} rows")
    return df
