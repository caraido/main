# -*- coding: utf-8 -*-
"""
utils.confusion_matrices — shared confusion-matrix utilities for the
regression/retrieval pipelines.

These were previously duplicated verbatim across phoneme_regression.py,
semantic_regression.py, and semantic_vanilla_retrieval.py.

Public API:
    _best_bin_from_top1      — pick the best time bin per (mode='peak'|'mean')
    _collect_pairs_at_bin    — collect (true, pred) label pairs at a given bin
    _make_cm                 — build a confusion matrix for a model at a bin
    _normalize_col           — column-normalize a CM
    _rank_labels_by_f1       — rank labels by F1 score
    _plot_cm_grid            — grid plot of CMs across models
    _per_word_stats          — per-word top-1 accuracy stats
    _per_word_f1_stats       — per-word F1 stats
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
import math

def _best_bin_from_top1(br, mode='word'):
    if mode == 'category':
        cat_top1 = np.asarray(br.all_retrieval_category_top1)
        if cat_top1.ndim == 2 and cat_top1.size > 0:
            return int(np.nanargmax(np.nanmean(cat_top1, axis=0)))
    top1 = np.asarray(br.all_retrieval_top1)
    return int(np.nanargmax(np.nanmean(top1, axis=0)))

def _collect_pairs_at_bin(br, bin_index):
    true_idx, pred_idx = [], []
    for rec in br.all_retrieval_pairs:
        if int(rec['bin_index']) == int(bin_index):
            true_idx.append(np.asarray(rec['true_word_idx'], dtype=np.int64))
            pred_idx.append(np.asarray(rec['pred_word_idx'], dtype=np.int64))
    if not true_idx:
        raise ValueError(f'No retrieval pairs found for bin {bin_index}')
    return np.concatenate(true_idx), np.concatenate(pred_idx)

def _make_cm(br, bin_index=None, mode='word'):
    if bin_index is None:
        bin_index = _best_bin_from_top1(br, mode=mode)
    y_true_w, y_pred_w = _collect_pairs_at_bin(br, bin_index)
    if mode == 'word':
        int_labels = np.arange(len(br.index_to_word), dtype=np.int64)
        y_true, y_pred, names = y_true_w, y_pred_w, np.asarray(br.index_to_word)
    else:
        int_labels = np.arange(len(br.index_to_category), dtype=np.int64)
        y_true = br.word_index_to_category_index[y_true_w].astype(np.int64)
        y_pred = br.word_index_to_category_index[y_pred_w].astype(np.int64)
        names  = np.asarray(br.index_to_category)
    cm = confusion_matrix(y_true, y_pred, labels=int_labels)
    return cm, names, bin_index

def _normalize_col(cm):
    cm      = cm.astype(float)
    col_sum = cm.sum(axis=0, keepdims=True)
    col_sum[col_sum == 0] = 1.0
    return cm / col_sum

def _rank_labels_by_f1(cm):
    tp      = np.diag(cm).astype(float)
    fp      = cm.sum(axis=0).astype(float) - tp
    fn      = cm.sum(axis=1).astype(float) - tp
    denom   = 2.0 * tp + fp + fn
    f1      = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    support = cm.sum(axis=1).astype(float)
    return np.lexsort((np.arange(len(f1)), -support, -f1))

def _plot_cm_grid(model_map, mode='word', normalize=True, cmap='viridis',
                  top_k_words_by_f1=None):
    n_models = len(model_map)
    n_cols   = min(3, n_models)
    n_rows   = math.ceil(n_models / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 7 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, (model_name, br) in zip(axes_flat, model_map.items()):
        cm, word_names, best_bin = _make_cm(br, mode=mode)
        total_n = int(cm.sum())
        if mode == 'word' and top_k_words_by_f1 is not None:
            k    = max(1, min(int(top_k_words_by_f1), len(word_names)))
            keep = _rank_labels_by_f1(cm)[:k]
            cm   = cm[np.ix_(keep, keep)]
            word_names = word_names[keep]
        shown_n  = int(cm.sum())
        cm_plot  = _normalize_col(cm) if normalize else cm.astype(float)
        vmin, vmax = (0.0, 1.0) if normalize else (0.0, max(float(cm_plot.max()), 1.0))
        im = ax.imshow(cm_plot, aspect='auto', cmap=cmap, origin='lower',
                       vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        n_ticks = len(word_names)
        ax.set_xticks(np.arange(n_ticks))
        ax.set_yticks(np.arange(n_ticks))
        ax.set_xticklabels(word_names, rotation=90, fontsize=9)
        ax.set_yticklabels(word_names, fontsize=9)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        title = f'{model_name} | best bin={best_bin}'
        title += f' | N={shown_n}' if shown_n == total_n else f' | shown={shown_n}/{total_n}'
        ax.set_title(title)

    for ax in axes_flat[n_models:]:
        ax.set_visible(False)

    mode_str = 'Single-word' if mode == 'word' else 'Category'
    fig.suptitle(
        f'{mode_str} retrieval confusion matrices (column-normalised)',
        fontsize=14,
    )
    plt.tight_layout()
    return fig

def _per_word_stats(br):
    bin_idx = _best_bin_from_top1(br, mode='word')
    y_true, y_pred = _collect_pairs_at_bin(br, bin_idx)
    n_words = len(br.index_to_word)
    counts  = np.zeros(n_words, dtype=int)
    correct = np.zeros(n_words, dtype=int)
    for wi in range(n_words):
        mask         = y_true == wi
        counts[wi]   = mask.sum()
        correct[wi]  = (y_pred[mask] == wi).sum()
    accuracy = np.where(counts > 0, correct / counts, np.nan)
    return br.index_to_word, counts, accuracy, bin_idx

def _per_word_f1_stats(br):
    cm, names, bin_idx = _make_cm(br, mode='word')
    tp      = np.diag(cm).astype(float)
    fp      = cm.sum(axis=0).astype(float) - tp
    fn      = cm.sum(axis=1).astype(float) - tp
    denom   = 2.0 * tp + fp + fn
    f1      = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    counts  = cm.sum(axis=1).astype(int)
    f1      = np.where(counts > 0, f1, np.nan)
    return names, counts, f1, bin_idx


def _plot_count_vs_metric(model_map, metric='accuracy'):
    n_models = len(model_map)
    n_cols   = min(3, n_models)
    n_rows   = math.ceil(n_models / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 5 * n_rows),
                             squeeze=False)
    axes_flat = axes.ravel()

    for ax, (model_name, br) in zip(axes_flat, model_map.items()):
        if metric == 'accuracy':
            words, counts, vals, best_bin = _per_word_stats(br)
            ylabel = 'Top-1 word accuracy'
        else:
            words, counts, vals, best_bin = _per_word_f1_stats(br)
            ylabel = 'Per-class F1'

        valid = ~np.isnan(vals)
        ax.scatter(counts[valid], vals[valid], s=60, alpha=0.75, zorder=3)
        for w, c, v in zip(words[valid], counts[valid], vals[valid]):
            if v > 0:
                ax.annotate(w, (c, v), textcoords='offset points',
                            xytext=(4, 3), fontsize=9, alpha=0.85)
        nonzero = valid & (vals > 0)
        if nonzero.sum() >= 2:
            r = np.corrcoef(counts[nonzero].astype(float), vals[nonzero])[0, 1]
            ax.set_title(f'{model_name}  |  best bin={best_bin}  |  r={r:.2f}')
        else:
            ax.set_title(f'{model_name}  |  best bin={best_bin}')
        ax.set_xlabel('Number of test samples')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n_models:]:
        ax.set_visible(False)

    fig.suptitle(f'Per-word: sample count vs. {ylabel}', fontsize=14, y=1.01)
    plt.tight_layout()
    return fig
