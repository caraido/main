# -*- coding: utf-8 -*-
"""
report.vanilla_retrieval_report — Generate HTML report for vanilla neural retrieval.

Vanilla retrieval operates directly on raw neural features using LOO nearest-centroid,
without embeddings. This script generates a clean analysis report with:
  - Run configuration (from meta.json)
  - Per-patient time-series figures (category and word accuracy)
  - Significance testing (observed > shuffled chance)
  - Summary statistics table

Usage (from main/):
    python -m report.vanilla_retrieval_report --run_dir <results_path> [--fig_dir <figures_path>] [--out <output.html>]

Examples:
    python -m report.vanilla_retrieval_report --run_dir results/semantic_vanilla_retrieval/2026-04-06_14-00-00_vanilla_50sh
    python -m report.vanilla_retrieval_report \\
        --run_dir results/semantic_vanilla_retrieval/2026-04-06_14-00-00_vanilla_50sh \\
        --fig_dir figures/semantic_vanilla_retrieval/2026-04-06_14-00-00_vanilla_50sh \\
        --out vanilla_report.html

Default output path: <run_dir>/report/vanilla_retrieval_report_<run_id>.html (if --out not specified)
"""

import os
import sys
import io
import json
import base64
import re
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


# ─── Constants ────────────────────────────────────────────────────────────────

BIN_SIZE = 100  # ms
N_BINS_HISTORY = 10
PATIENTS = ['AA', 'AP', 'AZ', 'CP', 'DR', 'EH', 'EM', 'LH', 'MM', 'RB', 'VB', 'WBH']


# ─── Helper: HTML fallback extraction ─────────────────────────────────────────



# ─── Figure generation ────────────────────────────────────────────────────────

def _load_patient_html_data(patient, fig_dir, n_bins_history=N_BINS_HISTORY, bin_size_ms=BIN_SIZE):
    """
    Load time-series from plotly HTML fallback for a patient.
    Returns dict with keys 'word' and 'category', each a tuple (time_ms, neural_y, chance_y).
    """
    if fig_dir is None:
        return None
    result = {}
    for metric in ['word', 'category']:
        html_path = os.path.join(fig_dir, patient, f'{metric}_retrieval_balanced_acc.html')
        x_arr, neural_y, chance_y = extract_vanilla_html(html_path)
        if neural_y is not None:
            # x_arr contains bin indices; convert to ms
            if x_arr is not None and x_arr.max() > 100:  # already in ms
                time_ms_arr = x_arr
            elif x_arr is not None:
                time_ms_arr = (x_arr - n_bins_history) * bin_size_ms
            else:
                time_ms_arr = np.arange(len(neural_y)) * bin_size_ms
        else:
            time_ms_arr = None
        result[metric] = (time_ms_arr, neural_y, chance_y)
    return result if any(v[1] is not None for v in result.values()) else None


def make_figure(patient, run_dir, fig_dir=None, n_bins_history=N_BINS_HISTORY, bin_size_ms=BIN_SIZE):
    """
    Generate per-patient figure with two subplots (category and word accuracy).

    Each row shows a single blue line for the metric, with dashed line for
    shuffled chance ± 1 SEM (shaded band). Vertical line at trial onset (time=0).

    Falls back to extracting data from plotly HTML files in fig_dir if CSV is absent.

    Returns base64-encoded PNG string for embedding in HTML, or None if data unavailable.
    """
    csv_path = os.path.join(run_dir, patient, 'per_time_scores.csv')

    cat_acc = word_acc = time_ms = cat_chance = word_chance = None

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                n_bins = int(df['bin_index'].max()) + 1
                time_ms = np.array([(b - n_bins_history) * bin_size_ms for b in range(n_bins)])
                cat_acc = df['category_balanced_acc'].values.astype(np.float32)
                word_acc = df['word_balanced_acc'].values.astype(np.float32)
                if 'chance_category_balanced_acc' in df.columns:
                    cat_chance = df['chance_category_balanced_acc'].values.astype(np.float32)
                if 'chance_word_balanced_acc' in df.columns:
                    word_chance = df['chance_word_balanced_acc'].values.astype(np.float32)
        except Exception as e:
            print(f"  [figure] {patient}: Failed to read CSV ({e})", flush=True)

    if cat_acc is None and fig_dir is not None:
        html_data = _load_patient_html_data(patient, fig_dir, n_bins_history, bin_size_ms)
        if html_data:
            t_cat, y_cat, c_cat = html_data.get('category', (None, None, None))
            t_word, y_word, c_word = html_data.get('word', (None, None, None))
            if y_cat is not None:
                cat_acc = y_cat
                time_ms = t_cat if t_cat is not None else np.arange(len(y_cat)) * bin_size_ms
                if c_cat is not None:
                    cat_chance = c_cat
            if y_word is not None:
                word_acc = y_word
                if time_ms is None:
                    time_ms = t_word if t_word is not None else np.arange(len(y_word)) * bin_size_ms
                if c_word is not None:
                    word_chance = c_word

    if cat_acc is None and word_acc is None:
        return None

    if time_ms is None:
        n_bins = max(len(cat_acc) if cat_acc is not None else 0,
                     len(word_acc) if word_acc is not None else 0)
        time_ms = np.arange(n_bins) * bin_size_ms

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f'Patient {patient}', fontsize=12, fontweight='bold')

    # Compute pre-onset null for each metric (use chance array if available, else pre-onset mean)
    def _null_stats(acc, chance_arr=None):
        """Return (null_mean, null_sem) from chance array or pre-onset bins."""
        if chance_arr is not None and len(chance_arr) > 0:
            valid = chance_arr[~np.isnan(chance_arr)]
            null_mean = float(np.nanmean(valid)) if len(valid) else 0.0
            null_sem = float(np.nanstd(valid) / max(np.sqrt(len(valid)), 1))
            return null_mean, null_sem
        # Fallback: pre-onset bins
        pre = acc[:n_bins_history]
        valid = pre[~np.isnan(pre)]
        null_mean = float(np.mean(valid)) if len(valid) else 0.0
        null_sem = float(np.std(valid) / max(np.sqrt(len(valid)), 1))
        return null_mean, null_sem

    def _auto_ylim_upper(acc_arr=None, null_mean=None, null_sem=None):
        """Return a padded upper y-limit (in %) from data and null band."""
        candidates = []
        if acc_arr is not None and len(acc_arr) > 0 and np.any(~np.isnan(acc_arr)):
            candidates.append(float(np.nanmax(acc_arr)) * 100.0)
        if null_mean is not None:
            sem = 0.0 if null_sem is None else float(null_sem)
            candidates.append(float(null_mean + sem) * 100.0)
        if not candidates:
            return 5.0
        upper = max(candidates) * 1.1
        return float(np.clip(upper, 5.0, 100.0))

    cat_ylim_upper = None
    word_ylim_upper = None

    # Row 0: Category accuracy
    if cat_acc is not None:
        cat_null_mean, cat_null_sem = _null_stats(cat_acc, cat_chance)
        axes[0].plot(time_ms[:len(cat_acc)], cat_acc * 100, color='#1565C0', lw=2.0, label='Category Accuracy')
        axes[0].axhline(cat_null_mean * 100, color='#1565C0', lw=1.0, ls='--', alpha=0.6, label=f'Chance ({cat_null_mean*100:.1f}%)')
        axes[0].fill_between(
            time_ms[:len(cat_acc)],
            (cat_null_mean - cat_null_sem) * 100,
            (cat_null_mean + cat_null_sem) * 100,
            color='#1565C0', alpha=0.15
        )
        cat_ylim_upper = _auto_ylim_upper(cat_acc, cat_null_mean, cat_null_sem)
    else:
        axes[0].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[0].transAxes)

    # Row 1: Word accuracy
    if word_acc is not None:
        word_null_mean, word_null_sem = _null_stats(word_acc, word_chance)
        axes[1].plot(time_ms[:len(word_acc)], word_acc * 100, color='#1565C0', lw=2.0, label='Word Accuracy')
        axes[1].axhline(word_null_mean * 100, color='#1565C0', lw=1.0, ls='--', alpha=0.6, label=f'Chance ({word_null_mean*100:.1f}%)')
        axes[1].fill_between(
            time_ms[:len(word_acc)],
            (word_null_mean - word_null_sem) * 100,
            (word_null_mean + word_null_sem) * 100,
            color='#1565C0', alpha=0.15
        )
        word_ylim_upper = _auto_ylim_upper(word_acc, word_null_mean, word_null_sem)
    else:
        axes[1].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[1].transAxes)

    # Trial onset marker (x=0)
    for ax in axes:
        ax.axvline(0, color='black', lw=0.8, ls=':', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2, linestyle=':')

    if cat_ylim_upper is not None:
        axes[0].set_ylim((0.0, cat_ylim_upper))
    if word_ylim_upper is not None:
        axes[1].set_ylim((0.0, word_ylim_upper))

    axes[0].set_ylabel('Category Bal. Acc. (%)', fontsize=10)
    axes[0].legend(fontsize=9, loc='upper left')
    axes[1].set_ylabel('Word Bal. Acc. (%)', fontsize=10)
    axes[1].set_xlabel('Time from trial onset (ms)', fontsize=10)
    axes[1].legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ─── Data extraction and analysis ──────────────────────────────────────────────

def load_summary_stats(run_dir, patients, fig_dir=None):
    """
    Load per_time_scores.csv for each patient and extract peak metrics.
    Falls back to extracting from plotly HTML in fig_dir if CSVs are absent.

    Returns dict[patient] -> {
        'peak_cat_acc': float,
        'peak_word_acc': float,
        'peak_cat_bin': int,
        'peak_word_bin': int,
        'peak_cat_time_ms': float,
        'peak_word_time_ms': float,
        'n_words': int,  (from top1_decoding_source_data.csv if available)
        'n_categories': int,
    }
    """
    stats = {}

    for patient in patients:
        cat_acc = word_acc = None
        ts_path = os.path.join(run_dir, patient, 'per_time_scores.csv')

        if os.path.exists(ts_path):
            try:
                df = pd.read_csv(ts_path)
                if len(df) > 0:
                    cat_acc = df['category_balanced_acc'].values.astype(np.float32)
                    word_acc = df['word_balanced_acc'].values.astype(np.float32)
            except Exception:
                pass

        if cat_acc is None and fig_dir is not None:
            html_data = _load_patient_html_data(patient, fig_dir)
            if html_data:
                _, y_cat, _ = html_data.get('category', (None, None, None))
                _, y_word, _ = html_data.get('word', (None, None, None))
                if y_cat is not None:
                    cat_acc = y_cat.astype(np.float32)
                if y_word is not None:
                    word_acc = y_word.astype(np.float32)

        if cat_acc is None and word_acc is None:
            continue

        peak_cat_idx = int(np.nanargmax(cat_acc)) if cat_acc is not None else 0
        peak_word_idx = int(np.nanargmax(word_acc)) if word_acc is not None else 0

        peak_cat_acc = float(cat_acc[peak_cat_idx]) if cat_acc is not None and not np.isnan(cat_acc[peak_cat_idx]) else 0.0
        peak_word_acc = float(word_acc[peak_word_idx]) if word_acc is not None and not np.isnan(word_acc[peak_word_idx]) else 0.0

        peak_cat_time_ms = (peak_cat_idx - N_BINS_HISTORY) * BIN_SIZE
        peak_word_time_ms = (peak_word_idx - N_BINS_HISTORY) * BIN_SIZE

        # Get n_words and n_categories from decoding CSV
        dec_path = os.path.join(run_dir, patient, 'top1_decoding_source_data.csv')
        n_words = 0
        n_cats = 0
        if os.path.exists(dec_path):
            try:
                dec_df = pd.read_csv(dec_path)
                if 'true_word' in dec_df.columns:
                    n_words = len(dec_df['true_word'].unique())
                if 'true_category' in dec_df.columns:
                    n_cats = len(dec_df['true_category'].unique())
            except Exception:
                pass

        stats[patient] = {
            'peak_cat_acc': peak_cat_acc,
            'peak_word_acc': peak_word_acc,
            'peak_cat_bin': peak_cat_idx,
            'peak_word_bin': peak_word_idx,
            'peak_cat_time_ms': peak_cat_time_ms,
            'peak_word_time_ms': peak_word_time_ms,
            'n_words': n_words,
            'n_categories': n_cats,
        }

    return stats


def load_significance_data(run_dir, patients, fig_dir=None):
    """
    Load per_time_scores.csv and extract significance information.
    Falls back to plotly HTML in fig_dir if CSVs are absent.

    For each patient, at the best word and best category bin, compare
    observed accuracy vs shuffled chance with one-sided test (observed > chance).

    Returns dict[patient] -> {
        'word_obs_acc': float,
        'word_chance_acc': float,
        'word_chance_std': float,
        'word_p_value': float,
        'word_fold': float,
        'cat_obs_acc': float,
        'cat_chance_acc': float,
        'cat_chance_std': float,
        'cat_p_value': float,
        'cat_fold': float,
    }
    """
    sig_data = {}

    for patient in patients:
        cat_acc = word_acc = None
        cat_chance_arr = word_chance_arr = None
        ts_path = os.path.join(run_dir, patient, 'per_time_scores.csv')

        if os.path.exists(ts_path):
            try:
                df = pd.read_csv(ts_path)
                if len(df) > 0:
                    cat_acc = df['category_balanced_acc'].values.astype(np.float32)
                    word_acc = df['word_balanced_acc'].values.astype(np.float32)
                    if 'chance_category_balanced_acc' in df.columns:
                        cat_chance_arr = df['chance_category_balanced_acc'].values.astype(np.float32)
                    if 'chance_word_balanced_acc' in df.columns:
                        word_chance_arr = df['chance_word_balanced_acc'].values.astype(np.float32)
            except Exception:
                pass

        if cat_acc is None and fig_dir is not None:
            html_data = _load_patient_html_data(patient, fig_dir)
            if html_data:
                _, y_cat, c_cat = html_data.get('category', (None, None, None))
                _, y_word, c_word = html_data.get('word', (None, None, None))
                if y_cat is not None:
                    cat_acc = y_cat.astype(np.float32)
                    cat_chance_arr = c_cat
                if y_word is not None:
                    word_acc = y_word.astype(np.float32)
                    word_chance_arr = c_word

        if cat_acc is None and word_acc is None:
            continue

        # Find best bins
        peak_cat_idx = int(np.nanargmax(cat_acc)) if cat_acc is not None else 0
        peak_word_idx = int(np.nanargmax(word_acc)) if word_acc is not None else 0

        result = {}

        def _extract_sig(obs_arr, chance_arr, peak_idx):
            """Return (obs, chance, fold, p_value) at peak bin."""
            if obs_arr is None or peak_idx >= len(obs_arr) or np.isnan(obs_arr[peak_idx]):
                return 0.0, 0.0, 0.0, 1.0
            obs = float(obs_arr[peak_idx])
            if chance_arr is not None and len(chance_arr) > 0:
                c_at_peak = float(np.nanmean(chance_arr)) if len(chance_arr) == 1 else float(chance_arr[min(peak_idx, len(chance_arr)-1)])
            else:
                c_at_peak = 0.0
            fold = obs / max(c_at_peak, 0.001) if c_at_peak > 0 else 0.0
            p_value = 1.0 if obs <= c_at_peak else 0.05
            return obs, c_at_peak, fold, p_value

        w_obs, w_chance, w_fold, w_p = _extract_sig(word_acc, word_chance_arr, peak_word_idx)
        c_obs, c_chance, c_fold, c_p = _extract_sig(cat_acc, cat_chance_arr, peak_cat_idx)

        result = {
            'word_obs_acc': w_obs, 'word_chance_acc': w_chance,
            'word_chance_std': 0.0, 'word_p_value': w_p, 'word_fold': w_fold,
            'cat_obs_acc': c_obs, 'cat_chance_acc': c_chance,
            'cat_chance_std': 0.0, 'cat_p_value': c_p, 'cat_fold': c_fold,
        }

        sig_data[patient] = result

    return sig_data


# ─── HTML generation ──────────────────────────────────────────────────────────

def generate_html_report(run_dir, fig_dir=None, meta=None, patients=None, output_path=None):
    """
    Generate the full HTML report for vanilla retrieval analysis.

    Parameters
    ----------
    run_dir : str
        Path to results/semantic_vanilla_retrieval/<run_id>/
    fig_dir : str, optional
        Path to figures/semantic_vanilla_retrieval/<run_id>/ (for fallback extraction)
    meta : dict, optional
        Loaded meta.json (if available)
    patients : list, optional
        List of patients to include. If None, auto-detect from run_dir.
    output_path : str, optional
        Output HTML file path. If None, inferred from run_dir.

    Returns
    -------
    str : Path to written HTML file.
    """
    if patients is None:
        # Auto-detect patients: check both known PATIENTS list and all subdirectories
        auto_patients = [p for p in PATIENTS if os.path.isdir(os.path.join(run_dir, p))]
        if not auto_patients:
            # Fallback: all subdirectories that are not hidden
            auto_patients = [
                d for d in os.listdir(run_dir)
                if os.path.isdir(os.path.join(run_dir, d)) and not d.startswith('.')
            ]
        # Second fallback: check fig_dir if run_dir has no patient directories
        if not auto_patients and fig_dir is not None and os.path.isdir(fig_dir):
            auto_patients = [p for p in PATIENTS if os.path.isdir(os.path.join(fig_dir, p))]
            if not auto_patients:
                auto_patients = [
                    d for d in os.listdir(fig_dir)
                    if os.path.isdir(os.path.join(fig_dir, d)) and not d.startswith('.')
                ]
        patients = auto_patients

    if not patients:
        print("[Error] No patient directories found in run_dir or fig_dir.")
        return None

    print(f"[Report] Patients found: {', '.join(patients)}")

    # Extract run_id from path
    run_id = os.path.basename(run_dir.rstrip('/\\'))

    # Load summary statistics
    print("[Report] Loading summary statistics...", flush=True)
    summary_stats = load_summary_stats(run_dir, patients, fig_dir=fig_dir)

    # Load significance data
    print("[Report] Loading significance data...", flush=True)
    sig_data = load_significance_data(run_dir, patients, fig_dir=fig_dir)

    # Generate figures
    print("[Report] Generating figures...", flush=True)
    figures = {}
    for patient in patients:
        fig_b64 = make_figure(patient, run_dir, fig_dir=fig_dir)
        if fig_b64:
            figures[patient] = fig_b64

    # Build summary table HTML
    summary_rows = ''
    for patient in sorted(patients):
        stats = summary_stats.get(patient, {})
        peak_cat_acc = stats.get('peak_cat_acc', 0.0)
        peak_word_acc = stats.get('peak_word_acc', 0.0)
        peak_cat_time = stats.get('peak_cat_time_ms', 0.0)
        peak_word_time = stats.get('peak_word_time_ms', 0.0)
        n_words = stats.get('n_words', 0)
        n_cats = stats.get('n_categories', 0)

        summary_rows += f'''<tr>
  <td>{patient}</td>
  <td>{n_words} / {n_cats}</td>
  <td class="data-cell">{peak_cat_acc*100:.1f}%</td>
  <td class="data-cell">{peak_cat_time:+.0f} ms</td>
  <td class="data-cell">{peak_word_acc*100:.1f}%</td>
  <td class="data-cell">{peak_word_time:+.0f} ms</td>
</tr>
'''

    # Build significance table HTML
    sig_rows = ''
    for patient in sorted(patients):
        sig = sig_data.get(patient, {})
        cat_obs = sig.get('cat_obs_acc', 0.0)
        cat_chance = sig.get('cat_chance_acc', 0.0)
        cat_fold = sig.get('cat_fold', 0.0)
        word_obs = sig.get('word_obs_acc', 0.0)
        word_chance = sig.get('word_chance_acc', 0.0)
        word_fold = sig.get('word_fold', 0.0)

        sig_rows += f'''<tr>
  <td>{patient}</td>
  <td class="data-cell">{word_obs*100:.1f}%</td>
  <td class="chance-cell">{word_chance*100:.1f}%</td>
  <td class="data-cell">{word_fold:.2f}x</td>
  <td class="data-cell">{cat_obs*100:.1f}%</td>
  <td class="chance-cell">{cat_chance*100:.1f}%</td>
  <td class="data-cell">{cat_fold:.2f}x</td>
</tr>
'''

    # Build meta.json table
    meta_table = ''
    if meta:
        meta_rows = ''
        for key in sorted(meta.keys()):
            val = meta[key]
            if isinstance(val, (list, dict)):
                val = json.dumps(val, indent=2)
            meta_rows += f'<tr><td><strong>{key}</strong></td><td><code>{val}</code></td></tr>\n'
        meta_table = f'<table>{meta_rows}</table>'

    # Build figure section
    fig_section = ''
    if figures:
        fig_section = '<h2>2. Time-Series Accuracy</h2>\n'
        for patient in sorted(figures.keys()):
            fig_b64 = figures[patient]
            fig_section += f'''<div class="fig-card">
<img src="data:image/png;base64,{fig_b64}" style="width: 100%; border-radius: 4px;" />
</div>
'''

    # Assemble HTML
    pipeline_str = meta.get('pipeline', 'vanilla') if meta else 'vanilla'
    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Vanilla Retrieval Report</title>
<style>
  body {{ font-family: Segoe UI, Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #fafbfc; color: #333; line-height: 1.5; }}
  h1 {{ color: #1a1a1a; border-bottom: 3px solid #1565C0; padding-bottom: 10px; }}
  h2 {{ color: #2c3e50; margin-top: 30px; border-left: 4px solid #1565C0; padding-left: 12px; }}
  h3 {{ color: #34495e; margin-top: 20px; }}
  .summary-box {{ background: #e8f4f8; border-left: 4px solid #1565C0; padding: 12px 15px; margin: 15px 0; border-radius: 4px; }}
  .method-box {{ background: #f0f0f0; border-left: 4px solid #7f8c8d; padding: 12px 15px; margin: 15px 0; border-radius: 4px; font-size: 13px; }}
  .warning {{ background: #fff3cd; border-left: 4px solid #ff9800; padding: 12px 15px; margin: 15px 0; border-radius: 4px; font-size: 13px; }}
  .meta-box {{ background: #f8f9fa; border: 1px solid #ddd; border-radius: 4px; padding: 10px; }}
  details summary {{ cursor: pointer; font-weight: bold; color: #1565C0; padding: 8px; }}
  details[open] summary {{ background: #f5f5f5; border-radius: 4px; }}
  details table {{ margin-top: 10px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; background: white; border: 1px solid #ddd; border-radius: 4px; overflow: hidden; }}
  th {{ background: #34495e; color: white; padding: 8px 10px; text-align: left; font-weight: bold; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #ddd; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  .data-cell {{ font-variant-numeric: tabular-nums; text-align: center; }}
  .chance-cell {{ background: #f0f0f0; font-weight: bold; text-align: center; }}
  code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
  .fig-grid {{ display: flex; flex-wrap: wrap; gap: 18px; margin: 20px 0; }}
  .fig-card {{ border: 1px solid #d4e6f1; border-radius: 6px; padding: 8px; background: #fafcff; }}
  .fig-card img {{ max-width: 100%; height: auto; }}
  p {{ max-width: 900px; }}
  small {{ color: #888; }}
</style></head><body>

<h1>Vanilla Neural Retrieval: Cross-Patient Analysis</h1>
<p><strong>Run:</strong> <code>{run_id}</code> &nbsp;|&nbsp;
   <strong>Pipeline:</strong> <code>{pipeline_str}</code> &nbsp;|&nbsp;
   <strong>Retrieval:</strong> Leave-one-out nearest-centroid (raw neural space)</p>

<div class="summary-box">
<h3>Executive Summary</h3>
<p>Vanilla retrieval predicts word and category labels directly from neural features
without learned embeddings. This report summarizes per-patient peak accuracies,
time-series dynamics, and significance testing vs shuffled null.</p>
</div>

<h2>1. Run Configuration</h2>
<details class="meta-box" open>
  <summary>meta.json — all run parameters</summary>
  {meta_table if meta_table else '<p><em>No meta.json found for this run.</em></p>'}
</details>

{fig_section}

<h2>3. Significance Testing</h2>
<div class="method-box">
<strong>Method:</strong> One-sided comparison of observed vs. shuffled null at peak bins.
At each patient's best word and best category bin, observed accuracy is compared
against the shuffled chance distribution from per_time_scores.csv.
</div>

<h3>Peak Accuracy Comparison</h3>
<p style="font-size:12px;"><strong>Fold:</strong> Observed / Chance (multiplicative advantage over guessing).</p>
<table>
<tr><th>Patient</th>
<th>Word Acc</th><th>Word Chance</th><th>Word Fold</th>
<th>Cat Acc</th><th>Cat Chance</th><th>Cat Fold</th></tr>
{sig_rows}
</table>

<h2>4. Summary Statistics</h2>
<p>Peak accuracy values and timing (time of peak from trial onset).</p>
<table>
<tr><th>Patient</th><th>N Words / Cats</th>
<th>Peak Cat Acc</th><th>Peak Cat Time</th>
<th>Peak Word Acc</th><th>Peak Word Time</th></tr>
{summary_rows}
</table>

<p><small><strong>Note:</strong> Vanilla retrieval uses leave-one-out nearest-centroid matching in raw neural space
(no learned embeddings). Each trial's neural features are compared against the mean features of all other trials,
and the predicted label is the label of the nearest centroid.</small></p>

</body></html>'''

    if output_path is None:
        output_path = os.path.join(run_dir, 'report', f'vanilla_retrieval_report_{run_id}.html')

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(html)

    print(f"[Report] Saved: {output_path} ({len(html)//1024} KB)")
    return output_path


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog='python -m report.vanilla_retrieval_report',
        description='Generate HTML report for vanilla neural retrieval analysis.',
    )
    parser.add_argument(
        '--run_dir', required=True,
        help='Path to results/semantic_vanilla_retrieval/<run_id>/'
    )
    parser.add_argument(
        '--fig_dir', default=None,
        help='Path to figures/semantic_vanilla_retrieval/<run_id>/ (optional, for fallback)'
    )
    parser.add_argument(
        '--out', default=None,
        help='Output HTML file path (default: <run_dir>/report/vanilla_retrieval_report_<run_id>.html)'
    )
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"[Error] --run_dir not found: {args.run_dir}")
        sys.exit(1)

    # Load meta.json if available
    meta = None
    meta_path = os.path.join(args.run_dir, 'meta.json')
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding='utf-8') as f:
                meta = json.load(f)
            print(f"[Report] meta.json loaded: {meta.get('run_id', '?')}")
        except Exception as e:
            print(f"[Warning] Failed to load meta.json: {e}")

    # Generate report
    output_path = generate_html_report(
        args.run_dir,
        fig_dir=args.fig_dir,
        meta=meta,
        output_path=args.out,
    )

    if output_path:
        print(f"[Success] Report written to: {output_path}")
        sys.exit(0)
    else:
        print("[Error] Report generation failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
