#!/usr/bin/env python3
"""
tests.visual_layer_sweep — Intermediate-layer regression for DINOv2 and SimCLR.

Tests whether intermediate layers of visual models produce embeddings that
predict neural HGA better than the default final (pooled) layer.

Experiment structure:
  1. Layer sweep: run regression at each layer independently per patient,
     recording cosine sim, retrieval accuracy, and R² at the best bin.
  2. Cross-patient consistency: check whether the same layers win across
     patients or whether optimal layer depends on electrode coverage.
  3. Layer combination: test concatenated multi-layer embeddings (with PCA
     to manage dimensionality) against the single best layer.
  4. Statistical comparison: paired Wilcoxon of each layer vs pooled.

Why this matters:
  - DINOv2 (ViT-B/14) has 13 transformer layers.  Early layers encode
    low-level visual features; later layers encode high-level semantics.
    iEEG electrodes in temporal/frontal cortex may preferentially align
    with mid-level representations (object parts, shape).
  - SimCLR (ResNet-50) has 5 CNN stages.  Similar hierarchy argument —
    stage 3/4 might outperform the final pooled representation.

Usage (from main/):
    python -m tests.visual_layer_sweep --patients AA --epochs 10
    python -m tests.visual_layer_sweep --patients AA AZ VB --epochs 20 --model pls
    python -m tests.visual_layer_sweep --patients AA --combine-layers --epochs 10

Output:
    tests/results/layer_sweep.csv          — full per-layer results
    tests/results/layer_sweep_stats.csv    — Wilcoxon vs pooled per layer
    tests/results/layer_sweep.html         — interactive report
"""

import os
import sys
import argparse
import gc
import warnings
import pickle as pk
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

warnings.filterwarnings('ignore')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline


# ── Helpers ──────────────────────────────────────────────────────────────────

def _layer_keys(d, prefix):
    """Sorted list of layer keys with a given prefix (e.g. 'dinov2_layer_')."""
    keys = [k for k in d if k.startswith(prefix + '_layer_')]
    return sorted(keys, key=lambda x: int(x.split('_')[-1]))


def _build_pipeline(model_mode='krr', alpha=1.5, pls_components=4):
    """Build regression pipeline matching the main codebase conventions."""
    if model_mode == 'krr':
        pipe = Pipeline([('nystroem', Nystroem(kernel='rbf')),
                         ('ridge', Ridge(alpha=alpha))])
        use_pca = True
    elif model_mode == 'pls':
        pipe = Pipeline([('pls', PLSRegression(n_components=pls_components,
                                               scale=False))])
        use_pca = False
    elif model_mode == 'kernel_pls':
        pipe = Pipeline([('nystroem', Nystroem(kernel='rbf')),
                         ('pls', PLSRegression(n_components=pls_components,
                                               scale=False))])
        use_pca = False
    else:
        pipe = Pipeline([('ridge', Ridge(alpha=alpha))])
        use_pca = True
    return pipe, use_pca


def load_layerwise_embeddings(pdata):
    """
    Load all available layer-wise embeddings for a patient.

    Returns
    -------
    dict[str, np.ndarray]
        Keys like 'dinov2_layer_00', 'dinov2_pooled', 'simclr_layer_02', etc.
        Values are (n_trials, D) arrays aligned to the patient's labels.
    """
    from semantic_regression import (_visual_embed_folders, _map_to_target,
                                     _normalize_tokens)

    labels = pdata['clean_target_labels']
    embed_folders = _visual_embed_folders(pdata['patient'])

    all_embeds = {}

    # ── DINOv2 ───────────────────────────────────────────────────────────
    dinov2_sources = []
    for folder in embed_folders:
        fpath = os.path.join(folder, 'dinov2_layerwise_embeddings.pk')
        if os.path.exists(fpath):
            with open(fpath, 'rb') as f:
                d = pk.load(f)
            dinov2_sources.append((d, np.array(d['words'])))

    if dinov2_sources:
        # Pooled
        all_embeds['dinov2_pooled'] = _map_to_target(
            dinov2_sources, 'dinov2_pooled', labels)

        # Per-layer
        layer_keys = _layer_keys(dinov2_sources[0][0], 'dinov2')
        for lk in layer_keys:
            try:
                arr = _map_to_target(dinov2_sources, lk, labels)
                if arr.size > 0:
                    all_embeds[lk] = arr
            except (KeyError, IndexError):
                pass

    # ── SimCLR ───────────────────────────────────────────────────────────
    simclr_sources = []
    for folder in embed_folders:
        fpath = os.path.join(folder, 'simclr_layerwise_embeddings.pk')
        if os.path.exists(fpath):
            with open(fpath, 'rb') as f:
                d = pk.load(f)
            simclr_sources.append((d, np.array(d['words'])))

    if simclr_sources:
        all_embeds['simclr_pooled'] = _map_to_target(
            simclr_sources, 'simclr_pooled', labels)

        layer_keys = _layer_keys(simclr_sources[0][0], 'simclr')
        for lk in layer_keys:
            try:
                arr = _map_to_target(simclr_sources, lk, labels)
                if arr.size > 0:
                    all_embeds[lk] = arr
            except (KeyError, IndexError):
                pass

    return all_embeds


def build_combined_embedding(layer_embeds, prefix, n_components=50):
    """
    Concatenate all layers of a model and reduce via PCA.

    Parameters
    ----------
    layer_embeds : dict
        All layer embeddings (from load_layerwise_embeddings).
    prefix : str
        'dinov2' or 'simclr'.
    n_components : int
        PCA components for the combined embedding.

    Returns
    -------
    np.ndarray or None
        (n_trials, n_components) array, or None if not enough layers.
    """
    keys = sorted([k for k in layer_embeds if k.startswith(f'{prefix}_layer_')],
                  key=lambda x: int(x.split('_')[-1]))
    if len(keys) < 2:
        return None

    concat = np.hstack([layer_embeds[k] for k in keys])
    n_comp = min(n_components, concat.shape[0] - 1, concat.shape[1])
    pca = PCA(n_components=n_comp)
    return pca.fit_transform(concat)


# ── Main sweep ───────────────────────────────────────────────────────────────

def run_layer_sweep(patient, pdata, layer_embeds, n_epochs=10,
                    model_mode='krr', alpha=1.5, pls_components=4,
                    closest='cosine', pca_components=10,
                    combine_layers=False):
    """
    Run regression for every available layer embedding.

    Returns a DataFrame with one row per (layer, epoch) containing:
    test_r2, test_cosine, cat_bal_acc, word_bal_acc.
    """
    from models.model import BasicRegressor

    X = pdata['clean_data_binned'].swapaxes(1, 2)
    labels = pdata['target_concept']
    category_labels = pdata['clean_word_category']

    # Determine which embeddings to test
    embed_keys = sorted(layer_embeds.keys())

    # Optionally add combined embeddings
    if combine_layers:
        for prefix in ['dinov2', 'simclr']:
            combined = build_combined_embedding(layer_embeds, prefix)
            if combined is not None:
                key = f'{prefix}_combined'
                layer_embeds[key] = combined
                embed_keys.append(key)

    records = []

    for emb_key in embed_keys:
        y = layer_embeds[emb_key]
        if y.ndim != 2 or y.shape[0] != X.shape[0]:
            print(f"    [skip] {emb_key}: shape mismatch "
                  f"({y.shape} vs {X.shape[0]} trials)")
            continue

        # Determine model family (for grouping)
        if emb_key.startswith('dinov2'):
            model_family = 'DINOv2'
        elif emb_key.startswith('simclr'):
            model_family = 'SimCLR'
        else:
            model_family = 'other'

        # Layer index for ordering
        if '_layer_' in emb_key:
            layer_idx = int(emb_key.split('_')[-1])
            layer_type = 'intermediate'
        elif '_pooled' in emb_key:
            layer_idx = 999  # sort last
            layer_type = 'pooled'
        elif '_combined' in emb_key:
            layer_idx = 1000
            layer_type = 'combined'
        else:
            layer_idx = -1
            layer_type = 'unknown'

        print(f"    {emb_key:30s} (dim={y.shape[1]:4d}) ...", end='', flush=True)

        pipe, use_pca = _build_pipeline(model_mode, alpha, pls_components)

        # PCA on target space for Ridge models (matching main pipeline)
        n_pca = min(pca_components, y.shape[0] - 1, y.shape[1])
        y_reducer = PCA(n_pca) if use_pca else None

        br = BasicRegressor(pipe, y_reducer=y_reducer)
        br.load_data(X, y, n_bins_history=10,
                     labels=labels, category_labels=category_labels)

        try:
            br.fit(
                n_epochs=n_epochs,
                parallel=None,
                closest=closest,
                compute_retrieval=True,
                save_retrieval_pairs=False,
                compute_top_k_accuracy=True,
                top_k_values=[1, 3, 5, 10],
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        # Extract at best bins
        test_r2    = np.array(br.all_test_score)
        cosine_sim = np.array(br.all_cosine_sim)
        cat_acc    = np.array(br.all_retrieval_category_balanced_acc)
        word_acc   = np.array(br.all_retrieval_word_balanced_acc)

        if test_r2.size == 0:
            print("  (no data)")
            continue

        r2_best   = int(np.argmax(test_r2.mean(0)))
        cos_best  = int(np.argmax(cosine_sim.mean(0))) if cosine_sim.size > 0 else r2_best
        cat_best  = int(np.argmax(cat_acc.mean(0)))
        word_best = int(np.argmax(word_acc.mean(0)))

        # Top-k accuracy at best word bin
        top_k_at_best = {}
        for k in [1, 3, 5, 10]:
            if k in br.all_top_k_accuracy and br.all_top_k_accuracy[k].size > 0:
                top_k_at_best[k] = float(br.all_top_k_accuracy[k][:, word_best].mean())

        # Per-epoch records (for statistical testing)
        for ep in range(test_r2.shape[0]):
            rec = {
                'patient':      patient,
                'model_family': model_family,
                'layer_key':    emb_key,
                'layer_idx':    layer_idx,
                'layer_type':   layer_type,
                'embed_dim':    y.shape[1],
                'epoch':        ep,
                'test_r2':      float(test_r2[ep, r2_best]),
                'test_cosine':  float(cosine_sim[ep, cos_best]) if cosine_sim.size > 0 else np.nan,
                'cat_bal_acc':  float(cat_acc[ep, cat_best]),
                'word_bal_acc': float(word_acc[ep, word_best]),
            }
            for k, v in top_k_at_best.items():
                rec[f'top{k}_acc'] = float(br.all_top_k_accuracy[k][ep, word_best])
            records.append(rec)

        mean_cos  = float(cosine_sim[:, cos_best].mean()) if cosine_sim.size > 0 else float('nan')
        mean_cat  = float(cat_acc[:, cat_best].mean())
        mean_word = float(word_acc[:, word_best].mean())
        top3_str  = f"  top3={top_k_at_best.get(3, float('nan')):.3f}" if 3 in top_k_at_best else ""
        print(f"  cos={mean_cos:.4f}  cat={mean_cat:.4f}  word={mean_word:.4f}{top3_str}")

        gc.collect()

    return pd.DataFrame(records)


def compute_vs_pooled_stats(df):
    """
    For each (patient, model_family, intermediate layer), run a paired
    Wilcoxon signed-rank test comparing its accuracy against the pooled layer.

    Returns a DataFrame with layer-level summary + p-values.
    """
    stat_records = []

    for (patient, family), grp in df.groupby(['patient', 'model_family']):
        pooled_key = f'{family.lower()}_pooled'
        pooled = grp[grp.layer_key == pooled_key]
        if len(pooled) == 0:
            continue

        pooled_cos  = pooled.sort_values('epoch')['test_cosine'].values
        pooled_word = pooled.sort_values('epoch')['word_bal_acc'].values
        pooled_cat  = pooled.sort_values('epoch')['cat_bal_acc'].values

        for layer_key, lgrp in grp.groupby('layer_key'):
            if layer_key == pooled_key:
                continue

            lgrp_sorted = lgrp.sort_values('epoch')
            layer_cos   = lgrp_sorted['test_cosine'].values
            layer_word  = lgrp_sorted['word_bal_acc'].values
            layer_cat   = lgrp_sorted['cat_bal_acc'].values

            n = min(len(pooled_cos), len(layer_cos))
            if n < 5:
                continue

            # Wilcoxon: is this layer BETTER than pooled?
            try:
                _, p_cos  = stats.wilcoxon(layer_cos[:n] - pooled_cos[:n],
                                           alternative='greater')
            except ValueError:
                p_cos = 1.0
            try:
                _, p_word = stats.wilcoxon(layer_word[:n] - pooled_word[:n],
                                           alternative='greater')
            except ValueError:
                p_word = 1.0
            try:
                _, p_cat  = stats.wilcoxon(layer_cat[:n] - pooled_cat[:n],
                                           alternative='greater')
            except ValueError:
                p_cat = 1.0

            stat_records.append({
                'patient':         patient,
                'model_family':    family,
                'layer_key':       layer_key,
                'layer_idx':       lgrp.iloc[0]['layer_idx'],
                'layer_type':      lgrp.iloc[0]['layer_type'],
                'embed_dim':       lgrp.iloc[0]['embed_dim'],
                'mean_cosine':     float(layer_cos.mean()),
                'pooled_cosine':   float(pooled_cos.mean()),
                'cos_delta':       float(layer_cos[:n].mean() - pooled_cos[:n].mean()),
                'cos_pval':        float(p_cos),
                'mean_word_acc':   float(layer_word.mean()),
                'pooled_word_acc': float(pooled_word.mean()),
                'word_delta':      float(layer_word[:n].mean() - pooled_word[:n].mean()),
                'word_pval':       float(p_word),
                'mean_cat_acc':    float(layer_cat.mean()),
                'pooled_cat_acc':  float(pooled_cat.mean()),
                'cat_delta':       float(layer_cat[:n].mean() - pooled_cat[:n].mean()),
                'cat_pval':        float(p_cat),
            })

    stat_df = pd.DataFrame(stat_records)
    if len(stat_df) > 0:
        # Bonferroni correction across all tests within each metric
        n_tests = len(stat_df)
        for col in ['cos_pval', 'word_pval', 'cat_pval']:
            stat_df[col.replace('pval', 'pval_bonf')] = np.minimum(
                stat_df[col] * n_tests, 1.0)
    return stat_df


def generate_html_report(df, stat_df, out_path):
    """Delegate to tests.helper.visual_layer_sweep_report (see that module for details)."""
    from main.tests.helper.visual_layer_sweep_report import generate_html_report as _gen
    _gen(df, stat_df, out_path)


def main():
    parser = argparse.ArgumentParser(
        prog='python -m tests.visual_layer_sweep',
        description='Test intermediate DINOv2/SimCLR layers for neural decoding',
    )
    parser.add_argument('--patients', nargs='+', default=['AA'],
                        help='Patient IDs (default: AA)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Epochs per layer (default: 10)')
    parser.add_argument('--model', choices=['krr', 'linear_ridge', 'pls', 'kernel_pls'],
                        default='pls', help='Regression model (default: pls)')
    parser.add_argument('--pls-components', type=int, default=4,
                        help='PLS n_components (default: 4)')
    parser.add_argument('--closest', choices=['l2', 'cosine'], default='cosine',
                        help='Retrieval metric (default: cosine)')
    parser.add_argument('--combine-layers', action='store_true',
                        help='Also test concatenated all-layer embeddings')
    parser.add_argument('--out-dir', default='tests/results')
    args = parser.parse_args()

    os.chdir(_PROJECT_DIR)
    os.makedirs(args.out_dir, exist_ok=True)

    from semantic_regression import load_patient_data

    all_results = []
    for patient in args.patients:
        print(f"\n{'='*60}")
        print(f"Patient: {patient}")
        print(f"{'='*60}")

        pdata = load_patient_data(patient)

        print("  Loading layer-wise embeddings...")
        layer_embeds = load_layerwise_embeddings(pdata)
        print(f"  Found {len(layer_embeds)} layer embeddings: "
              f"{sorted(layer_embeds.keys())}")

        if not layer_embeds:
            print("  [!] No layer-wise embeddings found, skipping")
            continue

        df = run_layer_sweep(
            patient, pdata, layer_embeds,
            n_epochs=args.epochs,
            model_mode=args.model,
            pls_components=args.pls_components,
            closest=args.closest,
            combine_layers=args.combine_layers,
        )
        all_results.append(df)
        gc.collect()

    if not all_results:
        print("\nNo results collected.")
        return

    results = pd.concat(all_results, ignore_index=True)
    csv_path = os.path.join(args.out_dir, 'layer_sweep.csv')
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    # Statistical comparison vs pooled
    stat_df = compute_vs_pooled_stats(results)
    if len(stat_df) > 0:
        stat_path = os.path.join(args.out_dir, 'layer_sweep_stats.csv')
        stat_df.to_csv(stat_path, index=False)
        print(f"Stats saved:   {stat_path}")

    # HTML report
    html_path = os.path.join(args.out_dir, 'layer_sweep.html')
    generate_html_report(results, stat_df, html_path)

    # ── Summary ──────────────────────────────────────────────────────────
    from main.tests.helper.visual_layer_sweep_report import print_console_summary
    print_console_summary(results)


if __name__ == '__main__':
    main()
