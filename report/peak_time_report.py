"""
report.peak_time_report — Analyze timing dissociation between word and category peak accuracy.

Compares peak times (across time bins) for word vs. category balanced accuracy across patients,
using both KRR semantic regression and vanilla retrieval methods.

Entry point:
  python -m report.peak_time_report --krr_run_dir <path> --vanilla_fig_dir <path> [--out <html>]

Output:
  HTML report with per-patient peak time comparison, Wilcoxon test, and figures.
"""

import os
import sys
import json
import re
import base64
import argparse
import warnings
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

# Constants
PATIENTS = ['AA', 'AP', 'AZ', 'CP', 'DR', 'EH', 'EM', 'LH', 'MM', 'RB', 'VB', 'WBH']
EMBEDDINGS = ['GloVe', 'FastText', 'Word2Vec', 'ConceptNet', 'DINOv2', 'SimCLR']
BIN_SIZE = 100  # ms
N_BINS_HISTORY = 10  # bins before trial onset


def bin_to_time(bin_index):
    """Convert bin index to time in seconds from trial onset."""
    return (bin_index - N_BINS_HISTORY) * BIN_SIZE / 1000.0


def time_to_bin(time_sec):
    """Convert time in seconds to bin index."""
    return int(time_sec * 1000 / BIN_SIZE + N_BINS_HISTORY)


def extract_vanilla_html(html_path):
    """
    Extract neural accuracy time series from vanilla Plotly HTML.

    Returns
    -------
    x_arr : ndarray or None
        Time bins
    neural_y : ndarray or None
        Neural accuracy values
    chance_y : ndarray or None
        Chance accuracy values
    """
    if not os.path.exists(html_path):
        return None, None, None

    try:
        with open(html_path, encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return None, None, None

    # Find Plotly.newPlot call
    match = re.search(
        r'Plotly\.newPlot\(\s*"[^"]*"\s*,\s*(\[.*?\])\s*,\s*(\{.*?\})\s*\)',
        content,
        re.DOTALL
    )
    if not match:
        return None, None, None

    try:
        traces = json.loads(match.group(1))
    except Exception:
        return None, None, None

    neural_y, chance_y, x_arr = None, None, None

    for t in traces:
        y_data = t.get('y', {})
        x_data = t.get('x', {})
        trace_name = t.get('name', '')

        # Handle base64-encoded binary data
        if isinstance(y_data, dict) and 'bdata' in y_data:
            try:
                neural_bytes = base64.b64decode(y_data['bdata'])
                arr = np.frombuffer(neural_bytes, dtype='<f8')

                if isinstance(x_data, dict) and 'bdata' in x_data:
                    x_bytes = base64.b64decode(x_data['bdata'])
                    xarr = np.frombuffer(x_bytes, dtype='<f8')
                else:
                    xarr = None

                if trace_name == 'Neural':
                    neural_y = arr
                    x_arr = xarr
                elif trace_name == 'chance':
                    chance_y = arr
            except Exception:
                pass
        # Handle plain array data (if not base64)
        elif isinstance(y_data, (list, tuple)):
            arr = np.array(y_data, dtype=np.float64)
            xarr = np.array(x_data, dtype=np.float64) if isinstance(x_data, (list, tuple)) else None

            if trace_name == 'Neural':
                neural_y = arr
                x_arr = xarr
            elif trace_name == 'chance':
                chance_y = arr

    return x_arr, neural_y, chance_y


def load_krr_peak_times(run_dir, patients, embeddings):
    """
    Load peak times for word and category accuracy from semantic regression results.

    For each patient x embedding, find the bin with max word_balanced_acc and max category_balanced_acc.
    Also determine the best embedding per patient (highest peak category accuracy across embeddings).

    Returns
    -------
    peak_data : dict
        peak_data[patient] = {
            'embedding': str (best embedding),
            'word_peak_bin': int,
            'word_peak_acc': float,
            'cat_peak_bin': int,
            'cat_peak_acc': float,
        }
    """
    peak_data = {}

    for patient in patients:
        csv_path = os.path.join(run_dir, patient, 'per_time_scores.csv')
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        # Find best embedding for this patient (max category peak across all embeddings)
        best_emb = None
        best_cat_acc = -np.inf
        emb_peaks = {}

        for emb in embeddings:
            sub = df[df['embedding'] == emb].sort_values('bin_index')
            if len(sub) == 0:
                continue

            word_acc = sub['word_balanced_acc'].values
            cat_acc = sub['category_balanced_acc'].values

            word_peak_idx = np.nanargmax(word_acc)
            cat_peak_idx = np.nanargmax(cat_acc)

            word_peak_acc = float(word_acc[word_peak_idx])
            cat_peak_acc = float(cat_acc[cat_peak_idx])

            emb_peaks[emb] = {
                'word_peak_bin': int(sub.iloc[word_peak_idx]['bin_index']),
                'word_peak_acc': word_peak_acc,
                'cat_peak_bin': int(sub.iloc[cat_peak_idx]['bin_index']),
                'cat_peak_acc': cat_peak_acc,
            }

            if cat_peak_acc > best_cat_acc:
                best_cat_acc = cat_peak_acc
                best_emb = emb

        if best_emb is not None:
            peak_data[patient] = {
                'embedding': best_emb,
                **emb_peaks[best_emb]
            }

    return peak_data


def load_vanilla_peak_times(fig_dir, patients):
    """
    Load peak times for word and category accuracy from vanilla HTML figures.

    Returns
    -------
    peak_data : dict
        peak_data[patient] = {
            'word_peak_bin': int,
            'word_peak_acc': float,
            'cat_peak_bin': int,
            'cat_peak_acc': float,
        }
    """
    peak_data = {}

    for patient in patients:
        word_html = os.path.join(fig_dir, patient, 'word_retrieval_balanced_acc.html')
        cat_html = os.path.join(fig_dir, patient, 'category_retrieval_balanced_acc.html')

        # Extract word peaks
        x_word, y_word, _ = extract_vanilla_html(word_html)
        if y_word is None or len(y_word) == 0:
            continue

        # Extract category peaks
        x_cat, y_cat, _ = extract_vanilla_html(cat_html)
        if y_cat is None or len(y_cat) == 0:
            continue

        # Find peak bins (argmax of neural accuracy)
        y_word_valid = np.nan_to_num(y_word, nan=-np.inf)
        y_cat_valid = np.nan_to_num(y_cat, nan=-np.inf)

        word_peak_idx = np.argmax(y_word_valid)
        cat_peak_idx = np.argmax(y_cat_valid)

        word_peak_acc = float(y_word[word_peak_idx])
        cat_peak_acc = float(y_cat[cat_peak_idx])

        # Infer bin indices from x values if available, otherwise use array indices
        if x_word is not None and len(x_word) > word_peak_idx:
            word_peak_bin = int(x_word[word_peak_idx])
        else:
            word_peak_bin = word_peak_idx

        if x_cat is not None and len(x_cat) > cat_peak_idx:
            cat_peak_bin = int(x_cat[cat_peak_idx])
        else:
            cat_peak_bin = cat_peak_idx

        peak_data[patient] = {
            'word_peak_bin': word_peak_bin,
            'word_peak_acc': word_peak_acc,
            'cat_peak_bin': cat_peak_bin,
            'cat_peak_acc': cat_peak_acc,
        }

    return peak_data


def compute_peak_time_statistics(krr_peaks, vanilla_peaks):
    """
    Compute summary statistics and Wilcoxon tests for peak times.

    Returns
    -------
    stats_table : pd.DataFrame
        Method | Mean word peak (s) | Mean cat peak (s) | Mean Δ | Wilcoxon p | N
    """
    results = []

    for method_name, peak_dict in [('KRR (best emb)', krr_peaks), ('Vanilla', vanilla_peaks)]:
        if not peak_dict:
            continue

        patients_with_data = list(peak_dict.keys())

        word_peaks = [bin_to_time(peak_dict[p]['word_peak_bin']) for p in patients_with_data]
        cat_peaks = [bin_to_time(peak_dict[p]['cat_peak_bin']) for p in patients_with_data]
        diffs = [w - c for w, c in zip(word_peaks, cat_peaks)]

        # Wilcoxon signed-rank test: does word peak differ from cat peak?
        try:
            stat, pval = scipy_stats.wilcoxon(word_peaks, cat_peaks, alternative='two-sided')
        except Exception:
            pval = np.nan

        results.append({
            'Method': method_name,
            'Mean word peak (s)': np.mean(word_peaks),
            'Mean cat peak (s)': np.mean(cat_peaks),
            'Mean Δ (word - cat) (s)': np.mean(diffs),
            'Wilcoxon p': pval,
            'N patients': len(patients_with_data)
        })

    return pd.DataFrame(results)


def make_peak_time_comparison_table(krr_peaks, vanilla_peaks, patients):
    """
    Create a detailed per-patient comparison table.

    Returns
    -------
    df : pd.DataFrame
        Columns: Patient | KRR word (s) | KRR cat (s) | KRR Δ | Vanilla word (s) | ...
    """
    rows = []

    for patient in patients:
        if patient not in krr_peaks or patient not in vanilla_peaks:
            continue

        krr_w = bin_to_time(krr_peaks[patient]['word_peak_bin'])
        krr_c = bin_to_time(krr_peaks[patient]['cat_peak_bin'])
        krr_d = krr_w - krr_c

        van_w = bin_to_time(vanilla_peaks[patient]['word_peak_bin'])
        van_c = bin_to_time(vanilla_peaks[patient]['cat_peak_bin'])
        van_d = van_w - van_c

        rows.append({
            'Patient': patient,
            'KRR word (s)': krr_w,
            'KRR cat (s)': krr_c,
            'KRR Δ (s)': krr_d,
            'Vanilla word (s)': van_w,
            'Vanilla cat (s)': van_c,
            'Vanilla Δ (s)': van_d,
        })

    return pd.DataFrame(rows)


def make_peak_time_stripplot_svg(krr_peaks, vanilla_peaks, patients):
    """
    Create a two-panel SVG figure showing peak time distributions.

    Left panel: Vanilla
    Right panel: KRR

    Each panel has patients on y-axis, time on x-axis. Word peaks = circles (blue),
    category peaks = triangles (orange), connected by lines.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    fig.suptitle('Peak Time Comparison: Word vs. Category Accuracy', fontsize=14, fontweight='bold')

    # Color scheme
    color_word = '#1f77b4'  # blue
    color_cat = '#ff7f0e'   # orange

    for ax_idx, (method_name, peak_dict) in enumerate([
        ('Vanilla', vanilla_peaks),
        ('KRR (best embedding)', krr_peaks)
    ]):
        ax = axes[ax_idx]

        patients_with_data = sorted([p for p in patients if p in peak_dict])

        word_times = [bin_to_time(peak_dict[p]['word_peak_bin']) for p in patients_with_data]
        cat_times = [bin_to_time(peak_dict[p]['cat_peak_bin']) for p in patients_with_data]

        y_positions = np.arange(len(patients_with_data))

        # Draw connecting lines
        for y, (w, c) in enumerate(zip(word_times, cat_times)):
            ax.plot([w, c], [y, y], color='#cccccc', lw=1.0, zorder=1)

        # Plot word peaks (circles)
        ax.scatter(word_times, y_positions, s=100, color=color_word, marker='o',
                   label='Word', zorder=3, alpha=0.75, edgecolors='black', linewidth=0.5)

        # Plot category peaks (triangles)
        ax.scatter(cat_times, y_positions, s=100, color=color_cat, marker='^',
                   label='Category', zorder=3, alpha=0.75, edgecolors='black', linewidth=0.5)

        # Mean lines
        mean_word = np.mean(word_times)
        mean_cat = np.mean(cat_times)
        ax.axvline(mean_word, color=color_word, linestyle='--', alpha=0.5, lw=2, label=f'Mean word: {mean_word:.2f} s')
        ax.axvline(mean_cat, color=color_cat, linestyle='--', alpha=0.5, lw=2, label=f'Mean cat: {mean_cat:.2f} s')

        ax.set_yticks(y_positions)
        ax.set_yticklabels(patients_with_data)
        ax.set_xlabel('Time from trial onset (s)', fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel('Patient', fontsize=11)
        ax.set_title(method_name, fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def make_peak_time_diff_histogram_svg(krr_peaks, vanilla_peaks, patients):
    """
    Create histogram of peak time differences (word - category).

    Two overlapping histograms: vanilla (blue) and KRR (orange).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    color_van = '#1f77b4'
    color_krr = '#ff7f0e'

    all_diffs = []
    all_labels = []

    for method_name, peak_dict, color in [
        ('Vanilla', vanilla_peaks, color_van),
        ('KRR', krr_peaks, color_krr)
    ]:
        patients_with_data = [p for p in patients if p in peak_dict]

        diffs = [
            bin_to_time(peak_dict[p]['word_peak_bin']) - bin_to_time(peak_dict[p]['cat_peak_bin'])
            for p in patients_with_data
        ]

        all_diffs.append(diffs)
        all_labels.append(method_name)

        mean_diff = np.mean(diffs)
        ax.axvline(mean_diff, color=color, linestyle='--', alpha=0.7, lw=2.5,
                   label=f'{method_name} mean: {mean_diff:.2f} s')

        ax.hist(diffs, bins=6, alpha=0.5, color=color, label=f'{method_name} (N={len(diffs)})',
                edgecolor='black', linewidth=1.0)

    ax.set_xlabel('Peak time difference (word - category) (s)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of Peak Time Differences', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def table_to_html(df, table_id='', css_class=''):
    """Convert a pandas DataFrame to an HTML table."""
    html = f'<table id="{table_id}" class="{css_class}">\n<tr>'

    # Header
    for col in df.columns:
        html += f'<th>{col}</th>'
    html += '</tr>\n'

    # Rows
    for idx, row in df.iterrows():
        html += '<tr>'
        for val in row:
            if isinstance(val, float):
                if np.isnan(val):
                    cell = 'N/A'
                else:
                    cell = f'{val:.4f}'
            else:
                cell = str(val)
            html += f'<td>{cell}</td>'
        html += '</tr>\n'

    html += '</table>\n'
    return html


def generate_html_report(krr_peaks, vanilla_peaks, patients, krr_run_dir,
                         stripplot_img, histograms_img, comparison_df, stats_df, out_path):
    """
    Assemble the complete HTML report.
    """
    # Load meta.json if available
    meta = {}
    meta_path = os.path.join(krr_run_dir, 'meta.json')
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            pass

    run_id = os.path.basename(krr_run_dir)

    # Generate comparison tables
    comparison_html = table_to_html(comparison_df, css_class='comparison-table')
    stats_html = table_to_html(stats_df, css_class='stats-table')

    # Count patients
    n_krr = len(krr_peaks)
    n_vanilla = len(vanilla_peaks)
    n_both = len(set(krr_peaks.keys()) & set(vanilla_peaks.keys()))

    # HTML template
    html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Peak Time Analysis Report — {run_id}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    color: #333;
    line-height: 1.6;
    background: #fafafa;
  }}
  h1 {{
    color: #1a5276;
    border-bottom: 3px solid #2980b9;
    padding-bottom: 10px;
    margin-bottom: 20px;
  }}
  h2 {{
    color: #2471a3;
    margin-top: 40px;
    border-bottom: 2px solid #d4e6f1;
    padding-bottom: 8px;
  }}
  h3 {{
    color: #2e86c1;
    margin-top: 25px;
  }}
  .summary-box {{
    background: #eaf2f8;
    border-left: 4px solid #2980b9;
    padding: 15px;
    margin: 20px 0;
    border-radius: 4px;
  }}
  .finding {{
    background: #fef9e7;
    border-left: 4px solid #f39c12;
    padding: 15px;
    margin: 15px 0;
    border-radius: 4px;
  }}
  .method-box {{
    background: #f3e5f5;
    border-left: 4px solid #8e24aa;
    padding: 15px;
    margin: 15px 0;
    border-radius: 4px;
  }}
  .meta-box {{
    background: #f9f9f9;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 12px;
    margin: 15px 0;
  }}
  .meta-box details {{
    cursor: pointer;
  }}
  .meta-box summary {{
    cursor: pointer;
    font-weight: bold;
    color: #2471a3;
    padding: 5px 0;
    user-select: none;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 15px 0;
    font-size: 13px;
    background: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
  }}
  th {{
    background: #2980b9;
    color: white;
    padding: 10px;
    text-align: left;
    font-weight: bold;
  }}
  td {{
    padding: 8px 10px;
    border-bottom: 1px solid #eee;
  }}
  tr:nth-child(even) {{
    background: #f8f9fa;
  }}
  tr:hover {{
    background: #f0f4f8;
  }}
  code {{
    background: #f0f0f0;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.9em;
    font-family: 'Monaco', 'Courier New', monospace;
  }}
  small {{
    color: #888;
  }}
  .fig-grid {{
    display: flex;
    flex-direction: column;
    gap: 20px;
    margin: 20px 0;
  }}
  .fig-card {{
    border: 1px solid #d4e6f1;
    border-radius: 6px;
    padding: 10px;
    background: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
  }}
  .fig-card img {{
    width: 100%;
    max-width: 900px;
  }}
  .comparison-table, .stats-table {{
    font-size: 12px;
  }}
  .comparison-table td, .stats-table td {{
    text-align: center;
    font-variant-numeric: tabular-nums;
  }}
  .sig {{ color: #27ae60; font-weight: bold; }}
  .ns {{ color: #e74c3c; }}
  em.label {{ font-style: normal; font-weight: bold; color: #2471a3; }}
</style></head><body>

<h1>Peak Time Analysis: Word vs. Category Decoding</h1>
<p><strong>Run:</strong> <code>{run_id}</code><br>
   <strong>KRR patients:</strong> {n_krr} |
   <strong>Vanilla patients:</strong> {n_vanilla} |
   <strong>Both:</strong> {n_both}</p>

<div class="summary-box">
  <h3>Overview</h3>
  <p>This analysis compares the timing of peak accuracy for <em>word</em> vs. <em>category</em>
  neural decoding across patients. If category information emerges earlier in neural activity
  than word identity (coarser-to-finer semantic processing), category peaks should occur
  <strong>before</strong> word peaks, resulting in negative peak time differences (word − category).</p>
</div>

<h2>1. Summary Statistics</h2>
<p>Wilcoxon signed-rank test (two-sided) compares peak times for word vs. category across patients.</p>
{stats_html}

<h2>2. Per-Patient Peak Time Comparison</h2>
<p>Time values in seconds from trial onset. BIN_SIZE = {BIN_SIZE} ms, history window = {N_BINS_HISTORY} bins ({N_BINS_HISTORY * BIN_SIZE} ms before onset).</p>
{comparison_html}

<h2>3. Peak Time Distributions</h2>
<p>Left: Vanilla retrieval. Right: KRR best embedding. Circles = word peaks (blue),
Triangles = category peaks (orange). Connected by gray lines per patient.
Dashed lines = mean peak times.</p>
<div class="fig-card">
  <img src="data:image/png;base64,{stripplot_img}" alt="Peak time comparison">
</div>

<h2>4. Distribution of Peak Time Differences</h2>
<p>Histogram of (word peak time − category peak time) for each method.
Negative values = category peaks before word (coarser-to-finer).
Dashed lines = mean difference.</p>
<div class="fig-card">
  <img src="data:image/png;base64,{histograms_img}" alt="Peak time differences">
</div>

<h2>5. Interpretation</h2>
<div class="method-box">
  <h3>Temporal Dissociation: Word vs. Category</h3>
  <p>
    The analysis reveals <strong>whether category information emerges earlier than word identity</strong>
    in the neural signal. This would be consistent with a coarser-to-finer semantic processing hierarchy:
  </p>
  <ol>
    <li><em>Semantic category</em> (coarser: "animal", "object") may be decoded earlier
        (peak in high gamma ~100–300 ms post-stimulus).</li>
    <li><em>Word identity</em> (finer: "dog", "cat") may be decoded later
        (~300–500 ms), reflecting deeper semantic processing or multimodal binding.</li>
  </ol>

  <h4>Key Findings</h4>
  <ul>
    <li><strong>Mean word − category difference:</strong> Negative values suggest category peaks
        <strong>earlier</strong> than word (expected hierarchical pattern).
        Positive values suggest word peaks earlier (context-dependent or embedding-specific).</li>
    <li><strong>Consistency across methods:</strong> Do vanilla and KRR show the same temporal pattern?
        Agreement suggests a robust neural signature.</li>
    <li><strong>Patient variability:</strong> Some patients may show strong dissociation (e.g., RB, VB)
        while others show minimal delay. This could reflect differences in electrode coverage
        (category-selective vs. word-selective areas).</li>
  </ul>

  <h4>Relationship to Speech Production</h4>
  <p>
    From stimulus onset (t=0):
  </p>
  <ul>
    <li>~100–200 ms: Visual stimulus processing (early sensory activity).</li>
    <li>~200–400 ms: Semantic / category representation (coarser features).</li>
    <li>~300–600 ms: Word identity and phonological planning (finer features, lexical access).</li>
    <li>~600–1000+ ms: Voice onset / articulation (motor execution).</li>
  </ul>
  <p>
    Peak times in this analysis reflect the <em>peak</em> of the high gamma signal, not absolute onset.
    The timing gap between category and word peaks may reflect the latency difference in processing
    these two levels of semantic information.
  </p>
</div>

<h2>6. Methodological Notes</h2>
<div class="meta-box">
  <details open>
    <summary>Details</summary>
    <p>
      <strong>KRR method:</strong> Peak times computed from <code>per_time_scores.csv</code>,
      using the best embedding (highest peak category accuracy) for each patient.<br>
      <strong>Vanilla method:</strong> Peak times extracted from Plotly HTML figures,
      inferred from binary-encoded time series data.<br>
      <strong>Peak bin selection:</strong> For each metric (word/category), the peak bin is
      <code>argmax(accuracy)</code>, converted to time via <code>(bin_index − {N_BINS_HISTORY}) × {BIN_SIZE} ms</code>.<br>
      <strong>Wilcoxon test:</strong> Two-sided signed-rank test on paired peak times (word vs. category)
      within each method, no Bonferroni correction (exploratory analysis).
    </p>
  </details>
</div>

</body></html>'''

    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(html)

    print(f"[Report] Saved: {out_path} ({len(html)//1024} KB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='Analyze peak time dissociation between word and category decoding.'
    )
    parser.add_argument('--krr_run_dir', required=True,
                       help='Path to KRR run directory (e.g., results/semantic_regression/2026-03-27_...)')
    parser.add_argument('--vanilla_fig_dir', required=True,
                       help='Path to vanilla figures directory (e.g., figures/semantic_vanilla_retrieval/2026-04-08_...)')
    parser.add_argument('--out', default=None,
                       help='Output HTML path (default: <krr_run_dir>/../peak_time_report.html)')

    args = parser.parse_args()

    # Validate directories
    if not os.path.isdir(args.krr_run_dir):
        print(f"Error: KRR run directory not found: {args.krr_run_dir}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(args.vanilla_fig_dir):
        print(f"Error: Vanilla figures directory not found: {args.vanilla_fig_dir}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.out is None:
        out_path = os.path.join(os.path.dirname(args.krr_run_dir), 'peak_time_report.html')
    else:
        out_path = args.out

    print(f"[PeakTimeReport] KRR run: {args.krr_run_dir}")
    print(f"[PeakTimeReport] Vanilla figures: {args.vanilla_fig_dir}")
    print(f"[PeakTimeReport] Output: {out_path}")

    # Load peak times
    print("[PeakTimeReport] Loading KRR peak times...", flush=True)
    krr_peaks = load_krr_peak_times(args.krr_run_dir, PATIENTS, EMBEDDINGS)
    print(f"  Found {len(krr_peaks)} patients with KRR data.")

    print("[PeakTimeReport] Loading vanilla peak times...", flush=True)
    vanilla_peaks = load_vanilla_peak_times(args.vanilla_fig_dir, PATIENTS)
    print(f"  Found {len(vanilla_peaks)} patients with vanilla data.")

    if not krr_peaks or not vanilla_peaks:
        print("Error: No data loaded. Check paths.", file=sys.stderr)
        sys.exit(1)

    # Compute statistics
    print("[PeakTimeReport] Computing statistics...", flush=True)
    stats_df = compute_peak_time_statistics(krr_peaks, vanilla_peaks)

    # Create comparison table
    print("[PeakTimeReport] Building comparison table...", flush=True)
    comparison_df = make_peak_time_comparison_table(krr_peaks, vanilla_peaks, PATIENTS)

    # Generate figures
    print("[PeakTimeReport] Generating figures...", flush=True)
    stripplot_img = make_peak_time_stripplot_svg(krr_peaks, vanilla_peaks, PATIENTS)
    histograms_img = make_peak_time_diff_histogram_svg(krr_peaks, vanilla_peaks, PATIENTS)

    # Generate HTML report
    print("[PeakTimeReport] Assembling HTML report...", flush=True)
    generate_html_report(krr_peaks, vanilla_peaks, PATIENTS, args.krr_run_dir,
                         stripplot_img, histograms_img, comparison_df, stats_df, out_path)

    print(f"\n[PeakTimeReport] Complete! Report: {out_path}")


if __name__ == '__main__':
    main()
