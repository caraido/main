# -*- coding: utf-8 -*-
"""
report.model_vs_vanilla_report — Compare model semantic regression vs vanilla neural retrieval.

Compares model-based semantic regression (embedding-based) against vanilla
leave-one-out nearest-centroid retrieval in raw neural space.

Takes model per_time_scores.csv and vanilla per_time_scores.csv results; reports:
  - Per-patient comparison tables (word/category accuracy)
  - Paired Wilcoxon signed-rank tests
  - Per-embedding comparison (mean across patients)
  - Time-series overlays (SVG) for visual inspection
  - Interpretation notes

Usage (from main/):
    python -m report.model_vs_vanilla_report --model_run_dir <path> --vanilla_run_dir <path>
    python -m report.model_vs_vanilla_report --model_run_dir results/semantic_regression/2026-04-06_14-30-00_krr_cosine_50ep \\
                                              --vanilla_run_dir results/semantic_vanilla_retrieval/2026-04-06_14-00-00_vanilla_50sh

Output: HTML file written to --out (default: model_vs_vanilla_report.html)
"""

import os
import sys
import json
import re
import base64
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- cleanup batch 1: imports added by automated migration ---
from report.helper.html_utils import _decode_bdata

# --- cleanup batch 2: extract_vanilla_html now lives in report.helper.html_utils ---
from report.helper.html_utils import extract_vanilla_html

try:
    # Package execution
    from .helper.config import EMBEDDING_NAMES
except ImportError:
    try:
        # Script execution from report/
        from helper.config import EMBEDDING_NAMES
    except ImportError:
        # Script execution from main/
        from report.helper.config import EMBEDDING_NAMES


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

BIN_SIZE = 100  # ms
N_BINS_HISTORY = 10
PATIENTS = ['AA', 'AP', 'AZ', 'CP', 'DR', 'EH', 'EM', 'KAW', 'LH', 'MM', 'RB', 'VB', 'WBH']

EMB_COLORS = {
    'GloVe':       '#1565C0',   # dark blue
    'FastText':    '#0288D1',   # sky blue
    'Word2Vec':    '#00838F',   # teal
    'ConceptNet':  '#2E7D32',   # green
}


def derive_model_label_from_run_dir(model_run_dir):
    """Derive a readable model label from the run directory basename."""
    run_name = os.path.basename(os.path.normpath(model_run_dir))
    if not run_name:
        return 'Model'

    # Strip common timestamp prefix: YYYY-MM-DD_HH-MM-SS_
    m = re.match(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(.+)$', run_name)
    core = m.group(1) if m else run_name

    # Use the first token as model identifier (e.g., krr, kernel_pls, ridge)
    model_token = core.split('_')[0] if core else ''
    if not model_token:
        return 'Model'

    token_upper_map = {
        'krr': 'KRR',
        'pls': 'PLS',
        'svm': 'SVM',
        'mlp': 'MLP',
        'cnn': 'CNN',
        'rnn': 'RNN',
    }

    parts = [p for p in model_token.replace('-', '_').split('_') if p]
    if not parts:
        return 'Model'

    pretty_parts = [token_upper_map.get(p.lower(), p.upper() if len(p) <= 3 else p.capitalize()) for p in parts]
    return '_'.join(pretty_parts)


# ═══════════════════════════════════════════════════════════════════════════════
# HTML extraction helpers
# ═══════════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_results(run_dir, patients):
    """
    Load model per_time_scores.csv for all patients.

    Returns
    -------
    dict[patient] → pd.DataFrame
        Rows: (embedding, bin_index), columns: word_balanced_acc, category_balanced_acc, ...
    """
    model_data = {}
    for patient in patients:
        csv_path = os.path.join(run_dir, patient, 'per_time_scores.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            model_data[patient] = df
        else:
            print(f"    [skip] {patient}: no per_time_scores.csv", flush=True)

    return model_data


def load_vanilla_results(run_dir, patients):
    """
    Load vanilla retrieval per_time_scores.csv files.

    Returns
    -------
    dict[patient] → {'word': (x, y, chance), 'category': (x, y, chance)}
    """
    vanilla_data = {}
    for patient in patients:
        csv_path = os.path.join(run_dir, patient, 'per_time_scores.csv')
        if not os.path.exists(csv_path):
            print(f"    [skip] {patient}: no vanilla per_time_scores.csv", flush=True)
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            print(f"    [skip] {patient}: failed to read vanilla per_time_scores.csv", flush=True)
            continue

        if len(df) == 0:
            print(f"    [skip] {patient}: empty vanilla per_time_scores.csv", flush=True)
            continue

        if 'bin_index' in df.columns:
            x_arr = (df['bin_index'].values.astype(np.float32) - N_BINS_HISTORY) * BIN_SIZE / 1000.0
        else:
            x_arr = (np.arange(len(df), dtype=np.float32) - N_BINS_HISTORY) * BIN_SIZE / 1000.0

        word_arr = df['word_balanced_acc'].values.astype(np.float32) if 'word_balanced_acc' in df.columns else None
        cat_arr = df['category_balanced_acc'].values.astype(np.float32) if 'category_balanced_acc' in df.columns else None
        word_chance = df['chance_word_balanced_acc'].values.astype(np.float32) if 'chance_word_balanced_acc' in df.columns else None
        cat_chance = df['chance_category_balanced_acc'].values.astype(np.float32) if 'chance_category_balanced_acc' in df.columns else None

        if word_arr is not None or cat_arr is not None:
            vanilla_data[patient] = {
                'word': (x_arr, word_arr, word_chance),
                'category': (x_arr, cat_arr, cat_chance),
            }
        else:
            print(f"    [skip] {patient}: missing vanilla word/category accuracy columns", flush=True)

    return vanilla_data


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison logic
# ═══════════════════════════════════════════════════════════════════════════════

def compare_models(model_data, vanilla_data):
    """
    Compare model (best embedding per patient) vs vanilla.

    Returns
    -------
    comparison : pd.DataFrame
        Columns: patient, metric (word/category), vanilla, krr_best, krr_embedding, delta
    """
    records = []

    for patient in PATIENTS:
        if patient not in model_data or patient not in vanilla_data:
            continue

        df_krr = model_data[patient]
        van_data = vanilla_data[patient]

        # Per metric (word, category)
        for metric in ['word_balanced_acc', 'category_balanced_acc']:
            metric_short = metric.split('_')[0]

            # Get vanilla peak accuracy
            if metric_short == 'word':
                x_van, y_van, y_chance = van_data['word']
            else:
                x_van, y_van, y_chance = van_data['category']

            if y_van is None:
                continue

            vanilla_peak = np.nanmax(y_van)

            # Get KRR best-embedding peak accuracy
            krr_peaks = []
            emb_peaks = {}
            for emb in EMBEDDING_NAMES:
                emb_df = df_krr[df_krr['embedding'] == emb]
                if len(emb_df) > 0:
                    peak = emb_df[metric].max()
                    krr_peaks.append(peak)
                    emb_peaks[emb] = peak
                else:
                    emb_peaks[emb] = np.nan

            if not krr_peaks or all(np.isnan(krr_peaks)):
                continue

            krr_best = np.nanmax(krr_peaks)
            best_emb = max(emb_peaks, key=lambda k: emb_peaks[k])
            delta = vanilla_peak - krr_best

            records.append({
                'patient': patient,
                'metric': metric_short,
                'vanilla': vanilla_peak,
                'krr_best': krr_best,
                'krr_embedding': best_emb,
                'delta': delta,
                'winner': 'Vanilla' if delta > 0 else 'KRR',
            })

    return pd.DataFrame(records)


def per_embedding_comparison(model_data, vanilla_data):
    """
    Compare each model embedding vs vanilla (mean across patients).

    Returns
    -------
    emb_comp : pd.DataFrame
        Columns: embedding, metric, vanilla_mean, emb_mean, n_patients, p_wilcoxon
    """
    records = []

    for metric in ['word_balanced_acc', 'category_balanced_acc']:
        metric_short = metric.split('_')[0]

        # Vanilla means per patient
        vanilla_pts = {}
        for patient in PATIENTS:
            if patient not in vanilla_data:
                continue
            van_data = vanilla_data[patient]
            if metric_short == 'word':
                _, y_van, _ = van_data['word']
            else:
                _, y_van, _ = van_data['category']

            if y_van is not None:
                vanilla_pts[patient] = np.nanmax(y_van)

        # Per embedding
        for emb in EMBEDDING_NAMES:
            emb_pts = {}
            for patient in PATIENTS:
                if patient not in model_data:
                    continue
                df_krr = model_data[patient]
                emb_df = df_krr[df_krr['embedding'] == emb]
                if len(emb_df) > 0:
                    emb_pts[patient] = emb_df[metric].max()

            # Compare paired patients
            shared_pts = set(vanilla_pts.keys()) & set(emb_pts.keys())
            if len(shared_pts) < 2:
                continue

            van_vals = [vanilla_pts[p] for p in sorted(shared_pts)]
            emb_vals = [emb_pts[p] for p in sorted(shared_pts)]

            if len(van_vals) > 1:
                try:
                    stat, pval = _scipy_stats.wilcoxon(
                        np.array(van_vals) - np.array(emb_vals),
                        alternative='two-sided'
                    )
                except Exception:
                    pval = np.nan

                records.append({
                    'embedding': emb,
                    'metric': metric_short,
                    'vanilla_mean': np.mean(van_vals),
                    'emb_mean': np.mean(emb_vals),
                    'n_patients': len(van_vals),
                    'p_wilcoxon': pval,
                })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# Time-series plotting
# ═══════════════════════════════════════════════════════════════════════════════

def create_timeseries_svg(model_data, vanilla_data, model_label='Model'):
    """
    Create a 3×4 small-multiples SVG showing vanilla vs model best-embedding accuracy over time.

    Returns
    -------
    svg_word : str
        SVG figure for word accuracy
    svg_category : str
        SVG figure for category accuracy
    """
    # --- Compute shared y-axis limits across all patients ---
    all_word_vals, all_cat_vals = [], []
    for patient in PATIENTS:
        if patient in vanilla_data:
            _, yw, _ = vanilla_data[patient]['word']
            _, yc, _ = vanilla_data[patient]['category']
            if yw is not None: all_word_vals.extend(yw[~np.isnan(yw)])
            if yc is not None: all_cat_vals.extend(yc[~np.isnan(yc)])
        if patient in model_data:
            df_k = model_data[patient]
            all_word_vals.extend(df_k['word_balanced_acc'].dropna().values)
            all_cat_vals.extend(df_k['category_balanced_acc'].dropna().values)
    ymax_word = max(all_word_vals) * 1.12 if all_word_vals else 0.3
    ymax_cat  = max(all_cat_vals) * 1.12 if all_cat_vals else 0.6

    fig_word, axes_word = plt.subplots(3, 4, figsize=(18, 12))
    fig_word.suptitle(f'Word Accuracy: Vanilla vs {model_label} (best embedding)', fontsize=14, fontweight='bold', y=0.98)

    fig_cat, axes_cat = plt.subplots(3, 4, figsize=(18, 12))
    fig_cat.suptitle(f'Category Accuracy: Vanilla vs {model_label} (best embedding)', fontsize=14, fontweight='bold', y=0.98)

    axes_word = axes_word.flatten()
    axes_cat = axes_cat.flatten()

    for idx, patient in enumerate(PATIENTS):
        ax_w = axes_word[idx]
        ax_c = axes_cat[idx]

        if patient not in model_data or patient not in vanilla_data:
            ax_w.text(0.5, 0.5, f'{patient}\n(no data)', ha='center', va='center', transform=ax_w.transAxes)
            ax_c.text(0.5, 0.5, f'{patient}\n(no data)', ha='center', va='center', transform=ax_c.transAxes)
            continue

        df_krr = model_data[patient]
        van_data = vanilla_data[patient]

        # Word accuracy
        x_van, y_van, y_chance_van = van_data['word']
        if x_van is None:
            x_van = np.arange(len(y_van)) if y_van is not None else None

        if y_van is not None:
            if x_van is None:
                x_van = np.arange(len(y_van))
            ax_w.plot(x_van, y_van, 'o-', color="#E03116", label='Vanilla', linewidth=2, markersize=4)
            if y_chance_van is not None:
                ax_w.axhline(np.nanmean(y_chance_van), color='gray', linestyle='--', label='Chance')

        # KRR best embedding
        krr_peaks = {}
        for emb in EMBEDDING_NAMES:
            emb_df = df_krr[df_krr['embedding'] == emb]
            if len(emb_df) > 0:
                krr_peaks[emb] = emb_df['word_balanced_acc'].max()

        if krr_peaks:
            best_emb = max(krr_peaks, key=lambda k: krr_peaks[k])
            emb_df = df_krr[df_krr['embedding'] == best_emb]
            if len(emb_df) > 0:
                emb_df = emb_df.sort_values('bin_index')
                x_krr = emb_df['bin_index'].values
                y_krr = emb_df['word_balanced_acc'].values
                # Convert bin indices to time (seconds from onset, -1.0s history)
                time_s = (x_krr - N_BINS_HISTORY) * BIN_SIZE / 1000.0
                ax_w.plot(time_s, y_krr, 's-', color=EMB_COLORS.get(best_emb, '#FF9800'),
                         label=f'{model_label} ({best_emb})', linewidth=2, markersize=4)

        ax_w.set_title(f'{patient}', fontweight='bold')
        ax_w.set_xlabel('Time (s)')
        ax_w.set_ylabel('Balanced Accuracy')
        ax_w.set_ylim([0, ymax_word])
        ax_w.legend(fontsize=8)
        ax_w.grid(True, alpha=0.3)

        # Category accuracy (same structure)
        x_van, y_van, y_chance_van = van_data['category']
        if x_van is None and y_van is not None:
            x_van = np.arange(len(y_van))

        if y_van is not None:
            ax_c.plot(x_van, y_van, 'o-', color="#E03116", label='Vanilla', linewidth=2, markersize=4)
            if y_chance_van is not None:
                ax_c.axhline(np.nanmean(y_chance_van), color='gray', linestyle='--', label='Chance')

        krr_peaks = {}
        for emb in EMBEDDING_NAMES:
            emb_df = df_krr[df_krr['embedding'] == emb]
            if len(emb_df) > 0:
                krr_peaks[emb] = emb_df['category_balanced_acc'].max()

        if krr_peaks:
            best_emb = max(krr_peaks, key=lambda k: krr_peaks[k])
            emb_df = df_krr[df_krr['embedding'] == best_emb]
            if len(emb_df) > 0:
                emb_df = emb_df.sort_values('bin_index')
                x_krr = emb_df['bin_index'].values
                y_krr = emb_df['category_balanced_acc'].values
                time_s = (x_krr - N_BINS_HISTORY) * BIN_SIZE / 1000.0
                ax_c.plot(time_s, y_krr, 's-', color=EMB_COLORS.get(best_emb, '#FF9800'),
                         label=f'{model_label} ({best_emb})', linewidth=2, markersize=4)

        ax_c.set_title(f'{patient}', fontweight='bold')
        ax_c.set_xlabel('Time (s)')
        ax_c.set_ylabel('Balanced Accuracy')
        ax_c.set_ylim([0, ymax_cat])
        ax_c.legend(fontsize=8)
        ax_c.grid(True, alpha=0.3)

    fig_word.subplots_adjust(hspace=0.45)
    fig_word.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig_cat.subplots_adjust(hspace=0.45)
    fig_cat.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    # Convert to SVG
    import io
    svg_word_io = io.StringIO()
    fig_word.savefig(svg_word_io, format='svg')
    svg_word = svg_word_io.getvalue()
    plt.close(fig_word)

    svg_cat_io = io.StringIO()
    fig_cat.savefig(svg_cat_io, format='svg')
    svg_cat = svg_cat_io.getvalue()
    plt.close(fig_cat)

    return svg_word, svg_cat


# ═══════════════════════════════════════════════════════════════════════════════
# HTML generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_html_report(comparison_df, emb_comp_df, svg_word, svg_cat, out_path,
                         model_label='KRR'):
    """Generate standalone HTML report comparing model vs vanilla."""

    # Summary statistics
    n_vanilla_win_word = (comparison_df[comparison_df['metric'] == 'word']['winner'] == 'Vanilla').sum()
    n_vanilla_win_cat = (comparison_df[comparison_df['metric'] == 'category']['winner'] == 'Vanilla').sum()
    n_patients = len(comparison_df[comparison_df['metric'] == 'word'])

    # Wilcoxon tests for overall word and category
    word_df = comparison_df[comparison_df['metric'] == 'word']
    cat_df = comparison_df[comparison_df['metric'] == 'category']

    if len(word_df) > 1:
        stat_w, pval_w = _scipy_stats.wilcoxon(
            word_df['vanilla'].values - word_df['krr_best'].values,
            alternative='two-sided'
        )
    else:
        pval_w = np.nan

    if len(cat_df) > 1:
        stat_c, pval_c = _scipy_stats.wilcoxon(
            cat_df['vanilla'].values - cat_df['krr_best'].values,
            alternative='two-sided'
        )
    else:
        pval_c = np.nan

    def _format_pvalue(pval):
        if isinstance(pval, tuple):
            pval = pval[-1] if len(pval) > 0 else np.nan
        try:
            p_num = float(pval)
        except (TypeError, ValueError):
            return 'N/A', True
        if np.isnan(p_num):
            return 'N/A', True
        return f'{p_num:.4f}', False

    pval_w_text, pval_w_is_nan = _format_pvalue(pval_w)
    pval_c_text, pval_c_is_nan = _format_pvalue(pval_c)

    # HTML template
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{model_label} vs Vanilla Retrieval Comparison</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #1565C0;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2em;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 0.95em;
        }}
        .summary-box {{
            background: white;
            border-left: 4px solid #1565C0;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 15px;
        }}
        .summary-item {{
            padding: 15px;
            background: #f9f9f9;
            border-radius: 4px;
            border-left: 3px solid #2196F3;
        }}
        .summary-item h3 {{
            color: #1565C0;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        .summary-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #2196F3;
        }}
        .pvalue {{
            font-size: 0.85em;
            color: #666;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            border-radius: 4px;
            overflow: hidden;
        }}
        thead {{
            background: #1565C0;
            color: white;
        }}
        th {{
            padding: 12px;
            text-align: left;
            font-weight: 600;
            font-size: 0.9em;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
        }}
        tbody tr:nth-child(even) {{
            background: #f9f9f9;
        }}
        tbody tr:hover {{
            background: #f0f7ff;
        }}
        .metric-label {{
            font-weight: 600;
            color: #1565C0;
            text-transform: capitalize;
        }}
        .winner-vanilla {{
            background: #c8e6c9;
            color: #1b5e20;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: 3px;
        }}
        .winner-krr {{
            background: #bbdefb;
            color: #0d47a1;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: 3px;
        }}
        .stat-sig {{
            color: #d32f2f;
            font-weight: bold;
        }}
        .stat-ns {{
            color: #666;
        }}
        h2 {{
            color: #1565C0;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 1.4em;
            border-bottom: 2px solid #1565C0;
            padding-bottom: 10px;
        }}
        .notes {{
            background: #fffde7;
            border-left: 4px solid #f57f17;
            padding: 15px;
            margin-bottom: 30px;
            border-radius: 4px;
        }}
        .notes h3 {{
            color: #f57f17;
            margin-bottom: 10px;
        }}
        .notes p {{
            margin-bottom: 10px;
            font-size: 0.95em;
        }}
        .notes p:last-child {{
            margin-bottom: 0;
        }}
        svg {{
            width: 100%;
            height: auto;
            margin-bottom: 30px;
        }}
        .figure-container {{
            background: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .figure-title {{
            font-weight: 600;
            color: #1565C0;
            margin-bottom: 15px;
            font-size: 1.1em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{model_label} Semantic Regression vs Vanilla Neural Retrieval</h1>
        <p class="subtitle">Comparison of embedding-based regression against direct neural feature nearest-neighbor matching</p>

        <div class="summary-box">
            <h2 style="margin-top: 0; border-bottom: none; padding-bottom: 0;">Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <h3>Word Accuracy</h3>
                    <div class="summary-value">{n_vanilla_win_word}/{n_patients}</div>
                    <p style="margin-top: 5px; font-size: 0.85em; color: #666;">Vanilla wins (out of {n_patients} patients)</p>
                    <div class="pvalue">Wilcoxon p = {pval_w_text} {'(two-sided)' if not pval_w_is_nan else '(N/A)'}</div>
                </div>
                <div class="summary-item">
                    <h3>Category Accuracy</h3>
                    <div class="summary-value">{n_vanilla_win_cat}/{n_patients}</div>
                    <p style="margin-top: 5px; font-size: 0.85em; color: #666;">Vanilla wins (out of {n_patients} patients)</p>
                    <div class="pvalue">Wilcoxon p = {pval_c_text} {'(two-sided)' if not pval_c_is_nan else '(N/A)'}</div>
                </div>
            </div>
        </div>

        <div class="notes">
            <h3>Methods</h3>
            <p><strong>Vanilla retrieval:</strong> Leave-one-out nearest-centroid matching in raw neural feature space (high-gamma 70–150 Hz, no embedding projection, no regression model).</p>
            <p><strong>{model_label} semantic regression:</strong> Mapping neural features to semantic embedding space (GloVe, FastText, Word2Vec, ConceptNet, DINOv2, SimCLR); cosine similarity retrieval.</p>
            <p><strong>Significance:</strong> Paired two-sided Wilcoxon signed-rank test across {n_patients} patients at each metric's peak time bin.</p>
        </div>

        <h2>Per-Patient Comparison: Word Accuracy</h2>
        <table>
            <thead>
                <tr>
                    <th>Patient</th>
                    <th>Vanilla</th>
                    <th>{model_label} Best</th>
                    <th>Embedding</th>
                    <th>Δ (V−M)</th>
                    <th>Winner</th>
                </tr>
            </thead>
            <tbody>
"""

    for _, row in word_df.sort_values('patient').iterrows():
        winner_class = 'winner-vanilla' if row['winner'] == 'Vanilla' else 'winner-krr'
        winner_label = row['winner'] if row['winner'] == 'Vanilla' else model_label
        html += f"""                <tr>
                    <td><strong>{row['patient']}</strong></td>
                    <td>{row['vanilla']:.3f}</td>
                    <td>{row['krr_best']:.3f}</td>
                    <td>{row['krr_embedding']}</td>
                    <td>{row['delta']:+.3f}</td>
                    <td><span class="{winner_class}">{winner_label}</span></td>
                </tr>
"""

    pval_w_cls = 'stat-sig' if (not pval_w_is_nan and float(pval_w_text) < 0.05) else 'stat-ns'
    html += f"""                <tr style="background: #f0f7ff; border-top: 2px solid #ccc;">
                    <td colspan="4"><strong>Wilcoxon signed-rank (two-sided, n={len(word_df)} patients)</strong></td>
                    <td colspan="2"><span class="{pval_w_cls}">p = {pval_w_text}</span></td>
                </tr>
            </tbody>
        </table>

        <h2>Per-Patient Comparison: Category Accuracy</h2>
        <table>
            <thead>
                <tr>
                    <th>Patient</th>
                    <th>Vanilla</th>
                    <th>{model_label} Best</th>
                    <th>Embedding</th>
                    <th>Δ (V−M)</th>
                    <th>Winner</th>
                </tr>
            </thead>
            <tbody>
"""

    for _, row in cat_df.sort_values('patient').iterrows():
        winner_class = 'winner-vanilla' if row['winner'] == 'Vanilla' else 'winner-krr'
        winner_label = row['winner'] if row['winner'] == 'Vanilla' else model_label
        html += f"""                <tr>
                    <td><strong>{row['patient']}</strong></td>
                    <td>{row['vanilla']:.3f}</td>
                    <td>{row['krr_best']:.3f}</td>
                    <td>{row['krr_embedding']}</td>
                    <td>{row['delta']:+.3f}</td>
                    <td><span class="{winner_class}">{winner_label}</span></td>
                </tr>
"""

    pval_c_cls = 'stat-sig' if (not pval_c_is_nan and float(pval_c_text) < 0.05) else 'stat-ns'
    html += f"""                <tr style="background: #f0f7ff; border-top: 2px solid #ccc;">
                    <td colspan="4"><strong>Wilcoxon signed-rank (two-sided, n={len(cat_df)} patients)</strong></td>
                    <td colspan="2"><span class="{pval_c_cls}">p = {pval_c_text}</span></td>
                </tr>
            </tbody>
        </table>

        <h2>Per-Embedding Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Embedding</th>
                    <th>Metric</th>
                    <th>Vanilla Mean</th>
                    <th>Embedding Mean</th>
                    <th>N Patients</th>
                    <th>Wilcoxon p</th>
                </tr>
            </thead>
            <tbody>
"""

    for _, row in emb_comp_df.sort_values(['metric', 'embedding']).iterrows():
        pval_class = 'stat-sig' if row['p_wilcoxon'] < 0.05 else 'stat-ns'
        html += f"""                <tr>
                    <td><strong>{row['embedding']}</strong></td>
                    <td class="metric-label">{row['metric']}</td>
                    <td>{row['vanilla_mean']:.3f}</td>
                    <td>{row['emb_mean']:.3f}</td>
                    <td>{int(row['n_patients'])}</td>
                    <td><span class="{pval_class}">{row['p_wilcoxon']:.4f}</span></td>
                </tr>
"""

    html += f"""            </tbody>
        </table>
"""

    html += f"""
        <h2>Time-Series Comparison</h2>
        <div class="figure-container">
            <div class="figure-title">Word Accuracy Over Time (3×4 patient grid)</div>
            {svg_word}
        </div>

        <div class="figure-container">
            <div class="figure-title">Category Accuracy Over Time (3×4 patient grid)</div>
            {svg_cat}
        </div>
"""

    html += f"""

        <div class="notes">
            <h3>Interpretation</h3>
            <p><strong>Vanilla strength:</strong> Direct neural feature matching has no bottleneck of embedding space projection; may capture patient-specific neural signatures that don't align with generic word embeddings.</p>
            <p><strong>{model_label} potential advantage:</strong> Embedding-based approaches can generalize across semantic relationships and leverage pre-trained semantic knowledge.</p>
            <p><strong>Time dynamics:</strong> Peaks may differ between methods due to different feature spaces and regularization. Vanilla peaks on raw neural features; {model_label} peaks in embedding space after regression.</p>
            <p><strong>Patient heterogeneity:</strong> Some patients (e.g., RB, CP per CLAUDE.md) may have more visual-area coverage, favoring image embeddings or direct neural matches. Individual electrode locations matter.</p>
        </div>

    </div>
</body>
</html>
"""

    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(html)

    print(f"  ✓ Report saved to: {out_path}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Compare model semantic regression vs vanilla neural retrieval.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m report.model_vs_vanilla_report \\
    --model_run_dir results/semantic_regression/2026-04-06_14-30-00_krr_cosine_50ep \\
        --vanilla_run_dir results/semantic_vanilla_retrieval/2026-04-06_14-00-00_vanilla_50sh

  python -m report.model_vs_vanilla_report \\
    --model_run_dir results/semantic_regression/latest \\
        --vanilla_run_dir results/semantic_vanilla_retrieval/latest \\
    --out comparison_report.html
        """,
    )
    parser.add_argument('--model-run-dir', '--model_run_dir', required=True,
                       help='Path to model results directory (results/semantic_regression/<run_id>)')
    parser.add_argument('--vanilla-run-dir', '--vanilla_run_dir', required=True,
                       help='Path to vanilla results directory (results/semantic_vanilla_retrieval/<run_id>)')
    parser.add_argument('--out', default=None,
                       help='Output HTML path (default: model_vs_vanilla_report.html in working directory)')
    parser.add_argument('--model-label', '--model_label', default=None,
                       help='Label for the model in report titles and tables (default: derived from --model_run_dir)')

    args = parser.parse_args()

    model_run_dir = args.model_run_dir

    # Validate inputs
    if not os.path.isdir(model_run_dir):
        print(f"ERROR: Model run directory not found: {model_run_dir}", flush=True)
        sys.exit(1)

    if not os.path.isdir(args.vanilla_run_dir):
        print(f"ERROR: Vanilla run directory not found: {args.vanilla_run_dir}", flush=True)
        sys.exit(1)

    out_path = args.out if args.out else 'model_vs_vanilla_report.html'
    model_label = args.model_label if args.model_label else derive_model_label_from_run_dir(model_run_dir)

    print(f"\n  Loading model results from: {model_run_dir}", flush=True)
    model_data = load_model_results(model_run_dir, PATIENTS)
    print(f"    Loaded {len(model_data)} patients", flush=True)

    print(f"\n  Loading vanilla results from: {args.vanilla_run_dir}", flush=True)
    vanilla_data = load_vanilla_results(args.vanilla_run_dir, PATIENTS)
    print(f"    Loaded {len(vanilla_data)} patients", flush=True)

    print(f"\n  Comparing models...", flush=True)
    comparison_df = compare_models(model_data, vanilla_data)
    print(f"    {len(comparison_df)} comparisons (word + category across patients)", flush=True)

    print(f"\n  Computing per-embedding comparison...", flush=True)
    emb_comp_df = per_embedding_comparison(model_data, vanilla_data)
    print(f"    {len(emb_comp_df)} embedding × metric combinations", flush=True)

    print(f"\n  Generating time-series SVG plots...", flush=True)
    svg_word, svg_cat = create_timeseries_svg(model_data, vanilla_data, model_label=model_label)
    print(f"    Created 3×4 small-multiples for word and category accuracy", flush=True)

    print(f"\n  Generating HTML report...", flush=True)
    generate_html_report(comparison_df, emb_comp_df, svg_word, svg_cat, out_path,
                         model_label=model_label)

    print(f"\n  Done!", flush=True)


if __name__ == '__main__':
    main()
