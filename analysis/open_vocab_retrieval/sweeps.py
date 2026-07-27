# -*- coding: utf-8 -*-
"""
tests/open_vocab_retrieval/sweeps.py
====================================
Step 7 of the guide: sweep the gallery SIZE (N) and the gallery VARIANT
(matched vs raw), so "why this N?" is answered with the whole curve rather than
a single defended point.

The headline statistic is the median percentile rank, which is ~N-invariant by
construction; top-k accuracies are reported alongside with their N (mechanically
N-dependent, chance = k/N).  Everything is per-patient with a cross-patient mean.

These functions build each gallery ONCE per (N, variant) from a shared stimulus
wordset and re-score the already-computed per-trial predictions — no model
re-fitting is needed for a sweep.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .gallery import build_gallery
from .retrieval import similarity_matrix, true_indices, compute_ranks
from .metrics import rank_metrics


def _score_predictions(pred_emb, true_word, gallery, ks, center):
    sims = similarity_matrix(pred_emb, gallery.emb, center=center)
    tidx = true_indices(true_word, gallery.word_to_index)
    rank = compute_ranks(sims, tidx)
    return rank_metrics(rank, gallery.N, ks=ks)


def sweep_gallery_size(predictions: Sequence, glove, stimulus_words: Sequence[str],
                       Ns: Sequence[int] = (500, 1000, 2000, 5000, 10000),
                       variants: Sequence[str] = ("matched", "raw"),
                       concreteness: Optional[dict] = None,
                       ks: Sequence[int] = (1, 5, 10, 50, 100),
                       center: bool = True, subtlex: Optional[dict] = None
                       ) -> pd.DataFrame:
    """Sweep N and gallery variant; return a tidy per-(patient, variant, N) table.

    ``predictions`` is a list of :class:`predict_io.TrialPredictions` (one per
    patient).  Chance columns (median_percentile=0.5, top{k}=k/N) are included so
    figures can draw the chance line.
    """
    rows: List[dict] = []
    for variant in variants:
        for N in Ns:
            gallery = build_gallery(glove, stimulus_words, n=N, variant=variant,
                                    concreteness=concreteness, subtlex=subtlex)
            for tp in predictions:
                m = _score_predictions(tp.pred_emb, tp.true_word, gallery, ks, center)
                row = {"patient": tp.patient, "task": tp.task,
                       "variant": variant, "N": int(N), "N_effective": gallery.N}
                row.update(m)
                row.update({f"chance_top{k}": k / float(gallery.N) for k in ks})
                row["chance_median_percentile"] = 0.5
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_sweep(sweep_df: pd.DataFrame,
                    ks: Sequence[int] = (1, 5, 10, 50, 100)) -> pd.DataFrame:
    """Cross-patient mean/sem of the sweep for each (variant, N)."""
    metric_cols = (["median_percentile", "mean_percentile", "MRR", "median_rank"]
                   + [f"top{k}" for k in ks])
    metric_cols = [c for c in metric_cols if c in sweep_df.columns]
    agg = (sweep_df.groupby(["variant", "N"])[metric_cols]
           .agg(["mean", "sem"]))
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()
