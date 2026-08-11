# -*- coding: utf-8 -*-
"""
report.pca_deflation_report — Where does word information live in neural space?

Reads the combined CSV from tests/pca_and_deflation_retrieval.py and generates
a standalone HTML report comparing:

  vanilla    — LOO nearest-centroid retrieval on raw neural features
  pca_N      — same retrieval on the top-N PCA components
  deflated_X — same retrieval after projecting out the PLS semantic subspace
               estimated from embedding X

The report includes:
  - Executive summary with key findings
  - Per-patient peak accuracy comparison table
  - Paired Wilcoxon signed-rank tests (PCA vs vanilla, deflated vs vanilla)
  - Time-series overlay plots (SVG) for each condition
  - Interpretation / discussion section

Usage (from main/):
    python -m report.pca_deflation_report
    python -m report.pca_deflation_report --csv <path to the combined CSV>
    python -m report.pca_deflation_report --out <path to the output HTML>

Defaults come from utils.paths, not from a hand-composed string; run with --help to see
where they resolve to on this machine.
"""

import argparse
import io
import os
import sys
import warnings

from utils.paths import report_path, results_dir

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── Defaults ─────────────────────────────────────────────────────────────────
# Both of these used to be composed by hand and both were wrong. The input named
# `tests/results/`, a root the 2026-07 reorganisation deleted -- and named it
# *relatively*, so it resolved against the working directory. The output was a bare
# filename, i.e. the current working directory, which is why .gitignore carries a
# `/*.html` rule for the repository root. Route both through utils.paths.
DEFAULT_CSV = str(results_dir('model_diagnostics', create=False) / 'pca_deflation_all.csv')
DEFAULT_OUT = str(report_path('model_diagnostics', 'pca_deflation_report', create=False))

CONDITION_COLORS = {
    'vanilla':           '#C62828',
    'pca_10':            '#6A1B9A',
    'pca_5':             '#7B1FA2',
    'pca_20':            '#4A148C',
    'deflated_GloVe':    '#2E7D32',
    'deflated_FastText':  '#558B2F',
    'deflated_Word2Vec':  '#00838F',
    'deflated_ConceptNet':'#6A1B9A',
}

CONDITION_LABELS = {
    'vanilla':            'Vanilla (raw)',
    'pca_10':             'PCA (10 PCs)',
    'pca_5':              'PCA (5 PCs)',
    'pca_20':             'PCA (20 PCs)',
    'deflated_GloVe':     'Deflated (GloVe)',
    'deflated_FastText':  'Deflated (FastText)',
    'deflated_Word2Vec':  'Deflated (Word2Vec)',
    'deflated_ConceptNet':'Deflated (ConceptNet)',
}


# ═════════════════════════════════════════════════════════════════════════════
#  Data loading & peak extraction
# ═════════════════════════════════════════════════════════════════════════════

def load_results(csv_path):
    """Load combined CSV and return DataFrame."""
    df = pd.read_csv(csv_path)
    return df


def extract_peaks(df, metric='word_balanced_acc'):
    """Extract peak metric per (patient, condition).

    Returns DataFrame with columns: patient, condition, embedding,
    peak_bin, peak_value, chance_at_peak.
    """
    rows = []
    for (pat, cond), g in df.groupby(['patient', 'condition']):
        best_idx = g[metric].idxmax()
        best_row = g.loc[best_idx]
        rows.append({
            'patient':       pat,
            'condition':     cond,
            'embedding':     best_row.get('embedding', ''),
            'peak_bin':      int(best_row['bin']),
            'peak_value':    float(best_row[metric]),
            'chance_at_peak': float(best_row.get('chance_word_balanced_acc', np.nan)),
        })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
#  Statistical tests
# ═════════════════════════════════════════════════════════════════════════════

def paired_wilcoxon(peaks_df, cond_a, cond_b, metric='peak_value'):
    """Paired Wilcoxon signed-rank test between two conditions.

    Returns (stat, p_value, n_pairs, mean_diff).
    """
    a = peaks_df[peaks_df['condition'] == cond_a].set_index('patient')[metric]
    b = peaks_df[peaks_df['condition'] == cond_b].set_index('patient')[metric]
    common = a.index.intersection(b.index)
    if len(common) < 2:
        return np.nan, np.nan, len(common), np.nan
    vals_a = a.loc[common].values
    vals_b = b.loc[common].values
    diff = vals_a - vals_b
    if np.all(diff == 0):
        return 0.0, 1.0, len(common), 0.0
    try:
        stat, pval = scipy_stats.wilcoxon(vals_a, vals_b, alternative='two-sided')
    except ValueError:
        stat, pval = np.nan, np.nan
    return float(stat), float(pval), len(common), float(np.mean(diff))


# ═════════════════════════════════════════════════════════════════════════════
#  Time-series SVG plotting
# ═════════════════════════════════════════════════════════════════════════════

def create_timeseries_svg(df, metric='word_balanced_acc', title_prefix='Word Accuracy', ylim=None):
    """Create small-multiples SVG: one subplot per patient, all conditions overlaid.

    Returns SVG string.
    """
    patients = sorted(df['patient'].unique())
    conditions = sorted(df['condition'].unique())
    n_patients = len(patients)

    n_cols = min(4, n_patients)
    n_rows = int(np.ceil(n_patients / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)

    # Shared y limits
    if ylim is not None:
        ymin, ymax = ylim
    else:
        all_vals = df[metric].dropna().values
        if len(all_vals) > 0:
            ymin = max(0, np.percentile(all_vals, 1) - 0.02)
            ymax = min(1, np.percentile(all_vals, 99) + 0.05)
        else:
            ymin, ymax = 0, 0.3

    for idx, patient in enumerate(patients):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        pat_df = df[df['patient'] == patient]

        for cond in conditions:
            cond_df = pat_df[pat_df['condition'] == cond].sort_values('bin')
            if cond_df.empty:
                continue
            color = CONDITION_COLORS.get(cond, '#888888')
            label = CONDITION_LABELS.get(cond, cond)
            lw = 2.5 if cond == 'vanilla' else 1.5
            ls = '-' if (cond == 'vanilla' or cond.startswith('pca_')) else '--'
            ax.plot(cond_df['bin'].values, cond_df[metric].values,
                    color=color, linewidth=lw, linestyle=ls, label=label, alpha=0.85)

        # Chance line (from vanilla condition)
        van_df = pat_df[pat_df['condition'] == 'vanilla'].sort_values('bin')
        if not van_df.empty and 'chance_word_balanced_acc' in van_df.columns:
            ax.plot(van_df['bin'].values, van_df['chance_word_balanced_acc'].values,
                    color='#616161', linewidth=2, linestyle='--', alpha=0.9, label='Chance')

        ax.set_title(patient, fontsize=12, fontweight='bold')
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('Time bin')
        ax.set_ylabel(title_prefix)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_patients, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    # Single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(6, len(conditions) + 1),
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'{title_prefix} Over Time: All Conditions',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))

    buf = io.StringIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    svg = buf.getvalue()
    plt.close(fig)
    return svg


# ═════════════════════════════════════════════════════════════════════════════
#  HTML report generation
# ═════════════════════════════════════════════════════════════════════════════

def _fmt_pval(p):
    if np.isnan(p):
        return 'N/A', True
    if p < 0.001:
        return f'{p:.2e}', False
    return f'{p:.4f}', False


def generate_html(df, out_path):
    """Generate standalone HTML report."""

    peaks = extract_peaks(df, 'word_balanced_acc')
    peaks_cat = extract_peaks(df, 'category_balanced_acc')

    conditions = sorted(df['condition'].unique())
    patients = sorted(df['patient'].unique())
    n_patients = len(patients)

    # Non-vanilla conditions for comparison
    other_conds = [c for c in conditions if c != 'vanilla']

    # ── Wilcoxon tests ───────────────────────────────────────────────────
    wilcoxon_rows = []
    for cond in other_conds:
        stat_w, p_w, n_w, diff_w = paired_wilcoxon(peaks, 'vanilla', cond)
        stat_c, p_c, n_c, diff_c = paired_wilcoxon(peaks_cat, 'vanilla', cond)
        wilcoxon_rows.append({
            'condition': cond,
            'word_stat': stat_w, 'word_p': p_w, 'word_n': n_w, 'word_diff': diff_w,
            'cat_stat': stat_c, 'cat_p': p_c, 'cat_n': n_c, 'cat_diff': diff_c,
        })

    # ── SVGs ─────────────────────────────────────────────────────────────
    svg_word = create_timeseries_svg(df, 'word_balanced_acc', 'Word Balanced Acc', ylim=(0, 0.26))
    svg_cat = create_timeseries_svg(df, 'category_balanced_acc', 'Category Balanced Acc')

    # ── Compute summary stats ────────────────────────────────────────────
    vanilla_peaks = peaks[peaks['condition'] == 'vanilla']
    vanilla_mean = vanilla_peaks['peak_value'].mean()

    pca_conds = [c for c in conditions if c.startswith('pca_')]
    defl_conds = [c for c in conditions if c.startswith('deflated_')]

    # ── Build HTML ───────────────────────────────────────────────────────
    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PCA & Deflation Retrieval Report</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #f5f5f5; color: #333; padding: 20px; line-height: 1.6;
}}
.container {{ max-width: 1400px; margin: 0 auto; }}
h1 {{ color: #1565C0; text-align: center; margin-bottom: 10px; font-size: 2em; }}
.subtitle {{ text-align: center; color: #666; margin-bottom: 30px; font-size: 0.95em; }}
h2 {{ color: #1565C0; margin-top: 40px; margin-bottom: 20px; font-size: 1.4em;
      border-bottom: 2px solid #1565C0; padding-bottom: 10px; }}
h3 {{ color: #374151; margin: 20px 0 10px; }}
.summary-box {{
  background: white; border-left: 4px solid #1565C0; padding: 20px;
  margin-bottom: 30px; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}
.summary-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-top: 15px; }}
.summary-item {{
  padding: 15px; background: #f9f9f9; border-radius: 4px; border-left: 3px solid #2196F3;
}}
.summary-item h4 {{ color: #1565C0; font-size: 0.85em; text-transform: uppercase;
                    letter-spacing: 0.5px; margin-bottom: 8px; }}
.summary-value {{ font-size: 1.6em; font-weight: bold; color: #2196F3; }}
.summary-detail {{ font-size: 0.85em; color: #666; margin-top: 5px; }}
table {{
  width: 100%; border-collapse: collapse; background: white;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 30px;
  border-radius: 4px; overflow: hidden; font-size: 0.9em;
}}
thead {{ background: #1565C0; color: white; }}
th {{ padding: 10px 12px; text-align: left; font-weight: 600; }}
td {{ padding: 8px 12px; border-bottom: 1px solid #eee; }}
tbody tr:nth-child(even) {{ background: #f9f9f9; }}
tbody tr:hover {{ background: #f0f7ff; }}
.winner {{ background: #c8e6c9; color: #1b5e20; font-weight: 600; padding: 2px 6px; border-radius: 3px; }}
.loser {{ background: #ffcdd2; color: #b71c1c; padding: 2px 6px; border-radius: 3px; }}
.sig {{ color: #d32f2f; font-weight: bold; }}
.ns {{ color: #999; }}
.methods {{
  background: #fffde7; border-left: 4px solid #f57f17; padding: 15px;
  margin-bottom: 30px; border-radius: 4px;
}}
.methods h3 {{ color: #f57f17; margin-bottom: 10px; }}
.methods p {{ margin-bottom: 10px; font-size: 0.95em; }}
.interpretation {{
  background: #e8f5e9; border-left: 4px solid #2E7D32; padding: 20px;
  margin: 30px 0; border-radius: 4px;
}}
.interpretation h3 {{ color: #2E7D32; margin-bottom: 10px; }}
.interpretation p {{ margin-bottom: 10px; font-size: 0.95em; }}
.interpretation ul {{ margin: 10px 0 10px 20px; font-size: 0.95em; }}
.figure-container {{
  background: white; padding: 20px; margin-bottom: 30px;
  border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}
svg {{ width: 100%; height: auto; }}
.footer {{ color: #999; font-size: 0.85em; margin-top: 40px; border-top: 1px solid #ddd; padding-top: 15px; text-align: center; }}
</style>
</head>
<body>
<div class="container">
<h1>Where Does Word Information Live in Neural Space?</h1>
<p class="subtitle">PCA dimensionality reduction &amp; semantic deflation analysis of LOO neural retrieval</p>
""")

    # ── Methods box ──────────────────────────────────────────────────────
    html_parts.append("""
<div class="methods">
  <h3>Methods</h3>
  <p><strong>Vanilla retrieval:</strong> Leave-one-out nearest-centroid matching in raw neural
     feature space (high-gamma 70–150 Hz, lagged feature concatenation). Cosine distance.</p>
  <p><strong>PCA retrieval (Test 1):</strong> Per time bin, fit PCA on all trials' neural features
     and retain only the top-N principal components. Then run the same LOO nearest-centroid
     retrieval on the reduced feature set. No regression model is trained.</p>
  <p><strong>Deflated retrieval (Test 2):</strong> Per time bin, fit PLS regression from neural
     features to a semantic embedding (e.g., GloVe 300-D). The PLS x-rotations define the
     "semantic subspace". This subspace is orthogonally projected out of the neural features.
     LOO nearest-centroid retrieval is then run on the residual.</p>
  <p><strong>Significance:</strong> Paired two-sided Wilcoxon signed-rank test on peak
     word balanced accuracy across patients.</p>
</div>
""")

    # ── Executive Summary ────────────────────────────────────────────────
    html_parts.append('<div class="summary-box">')
    html_parts.append('<h2 style="margin-top:0;border-bottom:none;padding-bottom:0;">Executive Summary</h2>')
    html_parts.append('<div class="summary-grid">')

    # Vanilla
    html_parts.append(f"""
<div class="summary-item">
  <h4>Vanilla Baseline</h4>
  <div class="summary-value">{vanilla_mean:.3f}</div>
  <div class="summary-detail">Mean peak word balanced accuracy ({n_patients} patients)</div>
</div>""")

    # PCA
    for pc in pca_conds:
        pc_peaks = peaks[peaks['condition'] == pc]
        if not pc_peaks.empty:
            pc_mean = pc_peaks['peak_value'].mean()
            diff = pc_mean - vanilla_mean
            html_parts.append(f"""
<div class="summary-item">
  <h4>{CONDITION_LABELS.get(pc, pc)}</h4>
  <div class="summary-value">{pc_mean:.3f}</div>
  <div class="summary-detail">Δ vs vanilla: {diff:+.3f}</div>
</div>""")

    # Deflated (mean across embeddings)
    if defl_conds:
        defl_peaks = peaks[peaks['condition'].isin(defl_conds)]
        if not defl_peaks.empty:
            defl_mean = defl_peaks.groupby('patient')['peak_value'].mean().mean()
            diff = defl_mean - vanilla_mean
            html_parts.append(f"""
<div class="summary-item">
  <h4>Deflated (avg across embeddings)</h4>
  <div class="summary-value">{defl_mean:.3f}</div>
  <div class="summary-detail">Δ vs vanilla: {diff:+.3f}</div>
</div>""")

    html_parts.append('</div></div>')  # close summary-grid and summary-box

    # ── Per-patient comparison table ─────────────────────────────────────
    html_parts.append('<h2>Per-Patient Peak Word Balanced Accuracy</h2>')
    html_parts.append('<table><thead><tr><th>Patient</th><th>Vanilla</th>')
    for cond in other_conds:
        html_parts.append(f'<th>{CONDITION_LABELS.get(cond, cond)}</th>')
    html_parts.append('</tr></thead><tbody>')

    for pat in patients:
        html_parts.append(f'<tr><td><strong>{pat}</strong></td>')
        van_val = peaks[(peaks['patient'] == pat) & (peaks['condition'] == 'vanilla')]['peak_value']
        van_v = float(van_val.iloc[0]) if not van_val.empty else np.nan
        html_parts.append(f'<td>{van_v:.4f}</td>')

        for cond in other_conds:
            cond_val = peaks[(peaks['patient'] == pat) & (peaks['condition'] == cond)]['peak_value']
            if cond_val.empty:
                html_parts.append('<td>—</td>')
            else:
                v = float(cond_val.iloc[0])
                delta = v - van_v
                cls = 'winner' if delta >= 0 else 'loser'
                html_parts.append(
                    f'<td><span class="{cls}">{v:.4f}</span> '
                    f'<small>({delta:+.4f})</small></td>')
        html_parts.append('</tr>')

    html_parts.append('</tbody></table>')

    # ── Wilcoxon test table ──────────────────────────────────────────────
    html_parts.append('<h2>Statistical Comparison vs Vanilla (Wilcoxon Signed-Rank)</h2>')
    html_parts.append("""<table><thead><tr>
        <th>Condition</th>
        <th>N</th>
        <th>Word Δ (V−C)</th><th>Word p</th>
        <th>Category Δ (V−C)</th><th>Category p</th>
    </tr></thead><tbody>""")

    for wr in wilcoxon_rows:
        cond_label = CONDITION_LABELS.get(wr['condition'], wr['condition'])
        wp_text, wp_nan = _fmt_pval(wr['word_p'])
        cp_text, cp_nan = _fmt_pval(wr['cat_p'])
        wp_cls = 'sig' if (not wp_nan and wr['word_p'] < 0.05) else 'ns'
        cp_cls = 'sig' if (not cp_nan and wr['cat_p'] < 0.05) else 'ns'
        html_parts.append(f"""<tr>
            <td><strong>{cond_label}</strong></td>
            <td>{wr['word_n']}</td>
            <td>{wr['word_diff']:+.4f}</td>
            <td class="{wp_cls}">{wp_text}</td>
            <td>{wr['cat_diff']:+.4f}</td>
            <td class="{cp_cls}">{cp_text}</td>
        </tr>""")

    html_parts.append('</tbody></table>')

    # ── Per-patient category accuracy table ──────────────────────────────
    html_parts.append('<h2>Per-Patient Peak Category Balanced Accuracy</h2>')
    html_parts.append('<table><thead><tr><th>Patient</th><th>Vanilla</th>')
    for cond in other_conds:
        html_parts.append(f'<th>{CONDITION_LABELS.get(cond, cond)}</th>')
    html_parts.append('</tr></thead><tbody>')

    for pat in patients:
        html_parts.append(f'<tr><td><strong>{pat}</strong></td>')
        van_val = peaks_cat[(peaks_cat['patient'] == pat) & (peaks_cat['condition'] == 'vanilla')]['peak_value']
        van_v = float(van_val.iloc[0]) if not van_val.empty else np.nan
        html_parts.append(f'<td>{van_v:.4f}</td>')

        for cond in other_conds:
            cond_val = peaks_cat[(peaks_cat['patient'] == pat) & (peaks_cat['condition'] == cond)]['peak_value']
            if cond_val.empty:
                html_parts.append('<td>—</td>')
            else:
                v = float(cond_val.iloc[0])
                delta = v - van_v
                cls = 'winner' if delta >= 0 else 'loser'
                html_parts.append(
                    f'<td><span class="{cls}">{v:.4f}</span> '
                    f'<small>({delta:+.4f})</small></td>')
        html_parts.append('</tr>')

    html_parts.append('</tbody></table>')

    # ── Time-series SVGs ─────────────────────────────────────────────────
    html_parts.append('<h2>Time-Series: Word Balanced Accuracy</h2>')
    html_parts.append(f'<div class="figure-container">{svg_word}</div>')

    html_parts.append('<h2>Time-Series: Category Balanced Accuracy</h2>')
    html_parts.append(f'<div class="figure-container">{svg_cat}</div>')

    # ── Interpretation / Discussion ──────────────────────────────────────
    # Compute the key metrics for the interpretation text
    pca_retained = {}
    for pc in pca_conds:
        pc_peaks = peaks[peaks['condition'] == pc]
        if not pc_peaks.empty:
            ratio = pc_peaks['peak_value'].mean() / vanilla_mean if vanilla_mean > 0 else np.nan
            pca_retained[pc] = ratio

    defl_retained = {}
    for dc in defl_conds:
        dc_peaks = peaks[peaks['condition'] == dc]
        if not dc_peaks.empty:
            ratio = dc_peaks['peak_value'].mean() / vanilla_mean if vanilla_mean > 0 else np.nan
            defl_retained[dc] = ratio

    html_parts.append("""
<div class="interpretation">
  <h3>Interpretation: Where Does Word Information Come From?</h3>

  <p>These two tests probe the structure of word-level neural representations and
     shed light on why vanilla LOO nearest-centroid retrieval in raw neural space
     achieves higher word accuracy than model-based semantic regression through
     word embeddings.</p>

  <h3>Test 1: PCA Dimensionality Reduction</h3>
""")

    if pca_retained:
        for pc, ratio in pca_retained.items():
            label = CONDITION_LABELS.get(pc, pc)
            pct = ratio * 100
            html_parts.append(
                f'<p><strong>{label}</strong> retains <strong>{pct:.1f}%</strong> '
                f'of vanilla peak word accuracy.</p>')

    html_parts.append("""
  <ul>
    <li>If PCA retains most accuracy (>80%), word-discriminable structure is concentrated
        in a low-dimensional subspace — the top principal components capture the axes along
        which word centroids separate.</li>
    <li>If PCA loses substantial accuracy (<50%), word information is distributed across
        many dimensions and requires the full high-dimensional feature space.</li>
    <li>PCA extracts the directions of <em>maximum variance</em>, not maximum class
        separability. If word boundaries align with high-variance directions (likely when
        different words activate distinct electrode populations), PCA preserves them.
        If word boundaries lie in low-variance directions, PCA will discard them.</li>
  </ul>

  <h3>Test 2: Semantic Subspace Deflation</h3>
""")

    if defl_retained:
        for dc, ratio in defl_retained.items():
            label = CONDITION_LABELS.get(dc, dc)
            pct = ratio * 100
            html_parts.append(
                f'<p><strong>{label}</strong> retains <strong>{pct:.1f}%</strong> '
                f'of vanilla peak word accuracy after removing its semantic subspace.</p>')

    html_parts.append("""
  <ul>
    <li>If deflation <em>preserves</em> word accuracy, the word-discriminating neural
        dimensions are largely orthogonal to the semantic embedding subspace. This means
        the LOO retrieval exploits neural features that encode word identity (e.g., phonological
        or articulatory codes) independently of the tested semantic space.</li>
    <li>If deflation <em>destroys</em> word accuracy, word-discriminating structure
        substantially overlaps with the semantic subspace — the same neural dimensions
        that predict word embeddings also separate word centroids.</li>
    <li>If different embeddings cause different deflation effects, it suggests those
        embedding spaces capture different aspects of word-level neural coding.</li>
  </ul>

  <h3>Why Does the Model Lose Word Accuracy?</h3>
  <p>Even when semantic subspaces overlap with word-separable structure, the regression
     model may lose word accuracy for several reasons:</p>
  <ul>
    <li><strong>Dimensionality bottleneck:</strong> The model projects neural features through
        a low-rank regression (e.g., PLS with 10 components or PCA to 10-D before Ridge),
        which discards neural dimensions that are word-discriminating but not well-predicted
        by the target embedding.</li>
    <li><strong>Embedding neighbourhood structure:</strong> Semantic embeddings group
        semantically-similar words close together (e.g., "cat" ≈ "dog"), collapsing
        word-level distinctions that the raw neural space preserves. The model's cosine
        retrieval in embedding space thus conflates within-category words.</li>
    <li><strong>Regression noise:</strong> The mapping from neural space to embedding space
        introduces prediction error that smears word centroids in embedding space,
        whereas LOO retrieval in neural space uses the raw centroids without a learned
        mapping.</li>
    <li><strong>Train/test mismatch:</strong> The regression model uses a train/test split,
        so some trials are never seen during training. The vanilla LOO centroid uses
        N−1 trials (all except the query), maximising the information per centroid.</li>
  </ul>
</div>
""")

    # ── Footer ───────────────────────────────────────────────────────────
    from datetime import datetime
    html_parts.append(f"""
<div class="footer">
  Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &mdash;
  report.pca_deflation_report
</div>
</div></body></html>""")

    html = '\n'.join(html_parts)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) if os.path.dirname(out_path) else '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Report written to {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Generate PCA & deflation retrieval comparison report')
    parser.add_argument('--csv', default=DEFAULT_CSV,
                        help=f'Combined results CSV (default: {DEFAULT_CSV})')
    parser.add_argument('--out', default=DEFAULT_OUT,
                        help=f'Output HTML path (default: {DEFAULT_OUT})')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: CSV not found: {args.csv}")
        print("Run  python -m analysis.pca_and_deflation_retrieval  first.")
        sys.exit(1)

    df = load_results(args.csv)
    print(f"Loaded {len(df):,} rows from {args.csv}")
    print(f"  Patients:   {sorted(df['patient'].unique())}")
    print(f"  Conditions: {sorted(df['condition'].unique())}")

    generate_html(df, args.out)


if __name__ == '__main__':
    main()
