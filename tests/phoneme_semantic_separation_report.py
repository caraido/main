"""
tests/phoneme_semantic_separation_report.py
============================================
Combined HTML report for all four phoneme-semantic separation tests.

Reads CSV outputs from:
  1. cross_category_generalization   → cross_cat_gen_all.csv
  2. semantic_residual_regression    → semantic_residual_all.csv
  3. partial_rsa                     → partial_rsa_all.csv
  4. subspace_angle_analysis         → subspace_angles_all.csv

Usage (run from main/):
    python -m tests.phoneme_semantic_separation_report
    python -m tests.phoneme_semantic_separation_report --in-dir test_results/

Output:
    test_results/phoneme_semantic_separation_report.html
"""

import os, sys, argparse, warnings, base64, io
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._phoneme_semantic_helpers import (
    PHONEME_EMBEDDINGS, N_BINS_HISTORY, header, get_out_dir,
)

EMB_COLORS = {'panphon': '#1565C0', 'token_ipa': '#E65100'}


# ── Plotting helpers ─────────────────────────────────────────────────────

def fig_to_base64(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── Section 1: Cross-category generalization ─────────────────────────────

def section_cross_cat(in_dir):
    csv_path = os.path.join(in_dir, 'cross_cat_gen_all.csv')
    if not os.path.exists(csv_path):
        return "<h2>1. Cross-Category Generalization</h2><p>No data found.</p>"

    df = pd.read_csv(csv_path)

    # Summary: mean across folds
    summary = df.groupby(['patient', 'embedding']).agg(
        word_acc=('word_bal_acc', 'mean'),
        word_std=('word_bal_acc', 'std'),
        cat_acc=('cat_indep_bal_acc', 'mean'),
        cat_std=('cat_indep_bal_acc', 'std'),
        cosine=('cosine_mean', 'mean'),
        word_chance=('word_chance', 'mean'),
        cat_chance=('cat_chance', 'mean'),
        n_folds=('fold_idx', 'count'),
    ).reset_index()

    # Table
    rows_html = ""
    for _, r in summary.iterrows():
        w_above = r['word_acc'] > r['word_chance']
        c_above = r['cat_acc'] > r['cat_chance']
        w_class = 'sig' if w_above else 'ns'
        c_class = 'sig' if c_above else 'ns'
        rows_html += f"""<tr>
            <td>{r['patient']}</td><td>{r['embedding']}</td>
            <td>{r['n_folds']:.0f}</td>
            <td class="{w_class}">{r['word_acc']:.4f} &plusmn; {r['word_std']:.4f}</td>
            <td>{r['word_chance']:.4f}</td>
            <td class="{c_class}">{r['cat_acc']:.4f} &plusmn; {r['cat_std']:.4f}</td>
            <td>{r['cat_chance']:.4f}</td>
            <td>{r['cosine']:.4f}</td>
        </tr>"""

    # Bar chart: word_acc vs chance across patients
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, metric_name, chance_col in [
        (axes[0], 'word_acc', 'Word Balanced Acc', 'word_chance'),
        (axes[1], 'cat_acc', 'Category Indep Acc', 'cat_chance'),
    ]:
        patients = summary['patient'].unique()
        x = np.arange(len(patients))
        width = 0.35
        for i, emb in enumerate(PHONEME_EMBEDDINGS):
            sub = summary[summary['embedding'] == emb]
            vals = [float(sub[sub['patient'] == p][metric].iloc[0])
                    if len(sub[sub['patient'] == p]) > 0 else 0
                    for p in patients]
            ax.bar(x + i * width, vals, width, label=emb,
                   color=EMB_COLORS.get(emb, f'C{i}'), alpha=0.8)
        # Chance line
        chance_val = summary[chance_col].mean()
        ax.axhline(chance_val, color='red', ls='--', alpha=0.7, label=f'chance={chance_val:.3f}')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(patients, rotation=45, ha='right')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} (cross-category)')
        ax.legend(fontsize=8)

    fig.tight_layout()
    img1 = fig_to_base64(fig)

    return f"""
    <h2>1. Cross-Category Generalization</h2>
    <p><b>Question:</b> Can phoneme decoding generalize to semantic categories
    never seen during training?</p>
    <p><b>Method:</b> Leave-K-categories-out CV for phoneme regression.
    Model trained on trials from a subset of categories, tested on held-out categories.</p>
    <p><b>Interpretation:</b> If word accuracy is above chance on held-out categories,
    the model uses phonological information that transfers across semantic contexts.
    If category-independent accuracy also drops toward chance, the semantic signal
    has been successfully excluded.</p>
    <img src="data:image/png;base64,{img1}" style="max-width:100%">
    <table class="data">
    <tr><th>Patient</th><th>Embedding</th><th>Folds</th>
        <th>Word Acc &plusmn; SD</th><th>Word Chance</th>
        <th>Cat Indep Acc &plusmn; SD</th><th>Cat Chance</th>
        <th>Cosine</th></tr>
    {rows_html}
    </table>
    """


# ── Section 2: Semantic residualization ──────────────────────────────────

def section_residual(in_dir):
    csv_path = os.path.join(in_dir, 'semantic_residual_all.csv')
    if not os.path.exists(csv_path):
        return "<h2>2. Semantic Residualization</h2><p>No data found.</p>"

    df = pd.read_csv(csv_path)

    # Grouped summary
    summary = df.groupby(['patient', 'phon_emb', 'condition']).agg(
        word_acc=('word_bal_acc', 'mean'),
        cat_acc=('cat_indep_bal_acc', 'mean'),
        cosine=('cosine_mean', 'mean'),
    ).reset_index()

    # Pivot for comparison
    rows_html = ""
    patients = summary['patient'].unique()
    for patient in patients:
        pat = summary[summary['patient'] == patient]
        for phon in pat['phon_emb'].unique():
            sub = pat[pat['phon_emb'] == phon]
            for _, r in sub.iterrows():
                cond = r['condition']
                rows_html += f"""<tr>
                    <td>{r['patient']}</td><td>{r['phon_emb']}</td>
                    <td><b>{cond}</b></td>
                    <td>{r['word_acc']:.4f}</td>
                    <td>{r['cat_acc']:.4f}</td>
                    <td>{r['cosine']:.4f}</td>
                </tr>"""

    # Bar chart: conditions side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    conditions = ['normal', 'residualized', 'sem_only']
    cond_colors = {'normal': '#2196F3', 'residualized': '#4CAF50', 'sem_only': '#FF9800'}

    for ax, metric, title in [
        (axes[0], 'word_acc', 'Word Balanced Accuracy'),
        (axes[1], 'cat_acc', 'Category Indep Accuracy'),
    ]:
        x = np.arange(len(patients))
        width = 0.25
        for i, cond in enumerate(conditions):
            vals = []
            for p in patients:
                sub = summary[(summary['patient'] == p) & (summary['condition'] == cond)]
                vals.append(float(sub[metric].mean()) if len(sub) > 0 else 0)
            ax.bar(x + i * width, vals, width, label=cond,
                   color=cond_colors[cond], alpha=0.8)
        ax.set_xticks(x + width)
        ax.set_xticklabels(patients, rotation=45, ha='right')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.tight_layout()
    img2 = fig_to_base64(fig)

    return f"""
    <h2>2. Semantic Residualization</h2>
    <p><b>Question:</b> Does phoneme decoding survive after surgically removing
    semantic neural dimensions?</p>
    <p><b>Method:</b> Fit semantic PLS to find "semantic directions" in neural space,
    project them out, then fit phoneme PLS on the residual. Three conditions:
    <em>normal</em> (unmodified X), <em>residualized</em> (semantic projected out),
    <em>sem_only</em> (only semantic subspace, sanity check).</p>
    <p><b>Interpretation:</b> If <em>residualized</em> retains word accuracy but
    category accuracy drops → phonological info is independent of semantic subspace.
    If <em>sem_only</em> shows no phoneme accuracy → semantic dimensions alone
    can't predict phonology.</p>
    <img src="data:image/png;base64,{img2}" style="max-width:100%">
    <table class="data">
    <tr><th>Patient</th><th>Phon Emb</th><th>Condition</th>
        <th>Word Acc</th><th>Cat Indep Acc</th><th>Cosine</th></tr>
    {rows_html}
    </table>
    """


# ── Section 3: Partial RSA ──────────────────────────────────────────────

def section_partial_rsa(in_dir):
    csv_path = os.path.join(in_dir, 'partial_rsa_all.csv')
    if not os.path.exists(csv_path):
        return "<h2>3. Partial RSA</h2><p>No data found.</p>"

    df = pd.read_csv(csv_path)

    # Time-course plots per patient
    patients = df['patient'].unique()
    figs_html = ""
    summary_rows = ""

    for patient in patients:
        pat = df[df['patient'] == patient]
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(f'{patient} — Partial RSA', fontsize=14)

        for j, phon in enumerate(PHONEME_EMBEDDINGS):
            sub = pat[pat['phon_emb'] == phon].sort_values('bin_index')
            if len(sub) == 0:
                continue

            t = sub['time_ms'].values
            # Full correlations
            axes[0, j].plot(t, sub['r_pred_phon'], color=EMB_COLORS[phon],
                           label='r(pred, phoneme)')
            axes[0, j].plot(t, sub['r_pred_sem'], color='#E91E63', ls='--',
                           label='r(pred, semantic)')
            axes[0, j].axhline(0, color='grey', lw=0.5)
            axes[0, j].axvline(0, color='grey', lw=0.5, ls=':')
            axes[0, j].set_title(f'{phon} — Full correlations')
            axes[0, j].set_ylabel('Spearman r')
            axes[0, j].legend(fontsize=7)

            # Partial correlations
            axes[1, j].plot(t, sub['r_partial_phon'], color=EMB_COLORS[phon],
                           label='r_partial(pred, phon | sem)')
            axes[1, j].plot(t, sub['r_partial_sem'], color='#E91E63', ls='--',
                           label='r_partial(pred, sem | phon)')
            axes[1, j].axhline(0, color='grey', lw=0.5)
            axes[1, j].axvline(0, color='grey', lw=0.5, ls=':')
            axes[1, j].set_title(f'{phon} — Partial correlations')
            axes[1, j].set_xlabel('Time from onset (ms)')
            axes[1, j].set_ylabel('Partial Spearman r')
            axes[1, j].legend(fontsize=7)

            # Summary: peak partial correlations
            valid = sub.dropna(subset=['r_partial_phon', 'r_partial_sem'])
            if len(valid) > 0:
                peak_phon_row = valid.loc[valid['r_partial_phon'].idxmax()]
                peak_sem_row  = valid.loc[valid['r_partial_sem'].idxmax()]
                summary_rows += f"""<tr>
                    <td>{patient}</td><td>{phon}</td>
                    <td>{peak_phon_row['r_partial_phon']:.3f}</td>
                    <td>{peak_phon_row['time_ms']:.0f}</td>
                    <td>{peak_sem_row['r_partial_sem']:.3f}</td>
                    <td>{peak_sem_row['time_ms']:.0f}</td>
                    <td>{sub['r_phon_sem'].iloc[0]:.3f}</td>
                </tr>"""

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        figs_html += f'<img src="data:image/png;base64,{fig_to_base64(fig)}" style="max-width:100%">'

    return f"""
    <h2>3. Partial RSA</h2>
    <p><b>Question:</b> What fraction of neural prediction geometry is uniquely
    phonological vs uniquely semantic?</p>
    <p><b>Method:</b> Spearman correlation between neural prediction RDM and
    phoneme/semantic ground-truth RDMs, with partial correlation to control for
    the other modality.</p>
    <p><b>Interpretation:</b> If <em>r_partial(pred, phon | sem)</em> is positive
    and peaks post-onset → genuine phonological representation independent of
    semantics. If <em>r_partial(pred, sem | phon)</em> is also positive →
    the neural signal carries both.</p>
    <table class="data">
    <tr><th>Patient</th><th>Phon Emb</th>
        <th>Peak r_partial_phon</th><th>@ ms</th>
        <th>Peak r_partial_sem</th><th>@ ms</th>
        <th>r(phon,sem)</th></tr>
    {summary_rows}
    </table>
    {figs_html}
    """


# ── Section 4: Subspace angles ──────────────────────────────────────────

def section_subspace_angles(in_dir):
    csv_path = os.path.join(in_dir, 'subspace_angles_all.csv')
    if not os.path.exists(csv_path):
        return "<h2>4. Subspace Angle Analysis</h2><p>No data found.</p>"

    df = pd.read_csv(csv_path)

    # Time-course plot: mean angle over time
    patients = df['patient'].unique()
    figs_html = ""
    summary_rows = ""

    fig, ax = plt.subplots(figsize=(12, 5))
    for patient in patients:
        pat = df[(df['patient'] == patient) & (df['phon_emb'] == 'panphon')]
        if len(pat) == 0 or 'mean_angle' not in pat:
            continue
        pat = pat.sort_values('bin_index')
        ax.plot(pat['time_ms'], pat['mean_angle'], alpha=0.6, label=patient)

    ax.axhline(90, color='green', ls=':', alpha=0.5, label='orthogonal (90°)')
    ax.axhline(45, color='orange', ls=':', alpha=0.5)
    ax.axvline(0, color='grey', lw=0.5, ls=':')
    ax.set_xlabel('Time from onset (ms)')
    ax.set_ylabel('Mean principal angle (deg)')
    ax.set_title('Phoneme vs Semantic subspace angles (panphon)')
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    figs_html = f'<img src="data:image/png;base64,{fig_to_base64(fig)}" style="max-width:100%">'

    # Summary table: post-onset mean angles
    post = df[df['bin_index'] >= N_BINS_HISTORY]
    if len(post) > 0 and 'mean_angle' in post:
        summary = post.groupby(['patient', 'phon_emb']).agg(
            mean_angle=('mean_angle', 'mean'),
            min_angle=('min_angle', 'min'),
            max_angle=('max_angle', 'max'),
        ).reset_index()
        for _, r in summary.iterrows():
            interpret = ('ORTHOGONAL' if r['mean_angle'] > 60
                        else 'ENTANGLED' if r['mean_angle'] < 30
                        else 'MIXED')
            summary_rows += f"""<tr>
                <td>{r['patient']}</td><td>{r['phon_emb']}</td>
                <td>{r['mean_angle']:.1f}°</td>
                <td>{r['min_angle']:.1f}°</td>
                <td>{r['max_angle']:.1f}°</td>
                <td><b>{interpret}</b></td>
            </tr>"""

    return f"""
    <h2>4. Subspace Angle Analysis</h2>
    <p><b>Question:</b> Do phonological and semantic PLS subspaces overlap in
    neural feature space?</p>
    <p><b>Method:</b> Fit Kernel PLS independently for phoneme and semantic targets,
    extract x_rotations_ (neural directions used by each), compute principal angles.</p>
    <p><b>Interpretation:</b> Mean angle near 90° = orthogonal subspaces (easy to
    separate). Near 0° = entangled (shared neural dimensions carry both).
    Below 30° flagged as ENTANGLED, above 60° as ORTHOGONAL.</p>
    {figs_html}
    <table class="data">
    <tr><th>Patient</th><th>Phon Emb</th>
        <th>Mean Angle</th><th>Min</th><th>Max</th><th>Interpretation</th></tr>
    {summary_rows}
    </table>
    """


# ── HTML assembly ────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Phoneme-Semantic Separation Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 2em; max-width: 1200px; }}
  h1 {{ color: #1a237e; border-bottom: 3px solid #1565C0; padding-bottom: 0.3em; }}
  h2 {{ color: #1565C0; margin-top: 2em; }}
  p {{ max-width: 800px; line-height: 1.5; }}
  table.data {{ border-collapse: collapse; margin: 1em 0; font-size: 0.9em; }}
  table.data th {{ background: #1565C0; color: white; padding: 6px 10px; }}
  table.data td {{ padding: 5px 10px; border-bottom: 1px solid #e0e0e0; }}
  table.data tr:hover {{ background: #f5f5f5; }}
  .sig {{ color: #2e7d32; font-weight: bold; }}
  .ns {{ color: #999; }}
  img {{ margin: 1em 0; border: 1px solid #e0e0e0; border-radius: 4px; }}
  .meta {{ color: #666; font-size: 0.85em; margin-bottom: 2em; }}
</style>
</head><body>
<h1>Phoneme-Semantic Separation Report</h1>
<div class="meta">Generated: {timestamp}<br>
Tests investigate whether phoneme regression decodes genuine phonological
information or merely reflects semantic co-variance in the neural signal.</div>

{section1}
{section2}
{section3}
{section4}

<hr>
<p class="meta"><em>Pipeline: Kernel PLS (Nystr&ouml;m RBF + PLS) with cosine retrieval.
Category-independent accuracy uses nearest-centroid matching in embedding space.</em></p>
</body></html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate combined HTML report for phoneme-semantic separation tests")
    parser.add_argument('--in-dir', default=None,
                        help='Input directory with CSV files (default: test_results/)')
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()

    in_dir = args.in_dir or get_out_dir(args.out_dir)
    out_dir = get_out_dir(args.out_dir)

    header("GENERATING PHONEME-SEMANTIC SEPARATION REPORT")
    print(f"  Input:  {in_dir}")
    print(f"  Output: {out_dir}")

    s1 = section_cross_cat(in_dir)
    s2 = section_residual(in_dir)
    s3 = section_partial_rsa(in_dir)
    s4 = section_subspace_angles(in_dir)

    html = HTML_TEMPLATE.format(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M'),
        section1=s1, section2=s2, section3=s3, section4=s4,
    )

    out_path = os.path.join(out_dir, 'phoneme_semantic_separation_report.html')
    with open(out_path, 'w') as f:
        f.write(html)

    print(f"\n  Report: {out_path}")
    print("Done!")


if __name__ == '__main__':
    main()
