"""
report.html_report — Assemble the full HTML analysis report.

Takes DataFrames produced by the other report modules (significance, bias,
dissociation, norms) and generates a self-contained HTML report with:
  - Executive summary
  - Significance tables (category + word, with Bonferroni stars)
  - Word prediction bias analysis
  - Embedding norm analysis
  - Metric dissociation
  - Semantic vs. visual comparison
"""

import os
import json
import numpy as np
import pandas as pd
from .config import EMBEDDING_NAMES, SEM_MODELS, VIS_MODELS


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _sig_class(s):
    """CSS class for a significance star string."""
    return {'***': 'star-three', '**': 'star-two', '*': 'star-one'}.get(s, 'star-ns')


def _patient_tier(p, sig_df):
    """Classify patient signal strength by mean category fold-over-null."""
    sub = sig_df[sig_df.patient == p]
    fold = (sub['mean_cat_obs'] / sub['mean_cat_null']).mean()
    if fold > 1.3:
        return 'patient-high'
    if fold > 1.1:
        return 'patient-moderate'
    return 'patient-low'


def _adaptive_col(df, preferred, fallback):
    """Pick the column name that exists in df."""
    return preferred if preferred in df.columns else fallback


# ─── Main report generator ────────────────────────────────────────────────────

def generate_report(sig_df, bias_df, dissoc_df, norm_df, out_dir, meta=None):
    """
    Generate the full HTML analysis report.

    Parameters
    ----------
    sig_df : pd.DataFrame
        Output of ``significance.compute_significance()``.
    bias_df : pd.DataFrame
        Output of ``bias.compute_word_bias()``.
    dissoc_df : pd.DataFrame
        Output of ``dissociation.compute_metric_dissociation()``.
    norm_df : pd.DataFrame
        Output of ``norms.compute_norm_analysis()``.
    out_dir : str
        Directory to write the HTML report and CSV files.
    meta : dict or None
        Run metadata (from meta.json) for display in the report header.

    Returns
    -------
    str
        Path to the generated HTML report.
    """
    os.makedirs(out_dir, exist_ok=True)

    if len(sig_df) == 0:
        print("[Report] No significance data — aborting")
        return None

    n_tests    = len(sig_df)
    n_patients = sig_df['patient'].nunique()
    patients_sorted = sorted(
        sig_df['patient'].unique(),
        key=lambda p: sig_df[sig_df.patient == p]['mean_cat_obs'].mean(),
        reverse=True,
    )
    n_cat_sig  = (sig_df['cat_sig']  != 'NS').sum()
    n_word_sig = (sig_df['word_sig'] != 'NS').sum()

    # Run info from meta.json
    run_id       = meta.get('run_id', 'unknown')      if meta else 'unknown'
    closest_mode = meta.get('closest', 'l2')           if meta else 'l2'
    pipeline_str = meta.get('regressor_pipeline', '?') if meta else '?'

    # ── Per-model significance counts ─────────────────────────────────────────
    sig_counts = {emb: {'cat': 0, 'word': 0} for emb in EMBEDDING_NAMES}
    for emb in EMBEDDING_NAMES:
        sub = sig_df[sig_df.embedding == emb]
        sig_counts[emb]['cat']  = (sub['cat_sig']  != 'NS').sum()
        sig_counts[emb]['word'] = (sub['word_sig'] != 'NS').sum()

    # ── Word bias summary ─────────────────────────────────────────────────────
    bias_summary = []
    if len(bias_df) > 0:
        ent_col = _adaptive_col(bias_df, 'pred_entropy_norm', 'pred_entropy')
        for emb in EMBEDDING_NAMES:
            sub = bias_df[bias_df.embedding == emb]
            if len(sub) == 0:
                continue
            top = sub.groupby('top1_word').size().sort_values(ascending=False)
            fav = top.index[0]
            n_fav = top.iloc[0]
            mean_pct = sub[sub.top1_word == fav]['top1_frac'].mean()
            mean_ent = sub[ent_col].mean()
            bias_summary.append({
                'emb': emb, 'fav_word': fav,
                'n_patients': f'{n_fav}/{n_patients}',
                'mean_pct':   f'{mean_pct*100:.1f}%',
                'mean_ent':   f'{mean_ent:.3f}',
            })

    # ── Norm-bias summary ─────────────────────────────────────────────────────
    norm_html = ''
    if len(norm_df) > 0:
        rank_col = _adaptive_col(norm_df, 'norm_rank', 'raw_norm_rank')
        word_col = _adaptive_col(norm_df, 'word', 'raw_norm_word')
        norm_col = _adaptive_col(norm_df, 'pca_norm', 'raw_norm')

        norm_html += '<h3>Embedding Norm vs. Predicted Words</h3>\n'
        norm_html += ('<p>Words with the smallest L2 norm in PCA-reduced embedding space '
                      'per model. Ridge regression is biased toward predicting these words.</p>\n')
        norm_html += '<table><tr><th>Model</th>'
        for r in range(5):
            norm_html += f'<th>Rank {r+1}</th>'
        norm_html += '</tr>\n'

        for emb in EMBEDDING_NAMES:
            sub = norm_df[(norm_df.embedding == emb) & (norm_df[rank_col] < 5)]
            if len(sub) == 0:
                continue
            cells = []
            for rank in range(5):
                rank_sub = sub[sub[rank_col] == rank]
                if len(rank_sub) == 0:
                    cells.append('—')
                else:
                    top_word = rank_sub.groupby(word_col).size().sort_values(ascending=False).index[0]
                    med_norm = rank_sub[rank_sub[word_col] == top_word][norm_col].median()
                    cells.append(f'{top_word} <small>(‖e‖={med_norm:.3f})</small>')
            norm_html += (f'<tr><td><strong>{emb}</strong></td>'
                          + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>\n')
        norm_html += '</table>\n'

        # Norm–bias match rate
        if len(bias_df) > 0:
            match_count = total_count = 0
            for emb in EMBEDDING_NAMES:
                for p in sig_df.patient.unique():
                    bias_row = bias_df[(bias_df.patient == p) & (bias_df.embedding == emb)]
                    norm_row = norm_df[(norm_df.patient == p) & (norm_df.embedding == emb)
                                       & (norm_df[rank_col] == 0)]
                    if len(bias_row) > 0 and len(norm_row) > 0:
                        total_count += 1
                        if bias_row.iloc[0]['top1_word'] == norm_row.iloc[0][word_col]:
                            match_count += 1
            if total_count > 0:
                pct = match_count / total_count
                norm_html += (
                    f'<div class="finding"><strong>Norm–bias correlation:</strong> '
                    f'{match_count}/{total_count} ({pct*100:.0f}%) match. ')
                if pct > 0.7:
                    norm_html += 'Ridge shrinkage is the dominant cause.</div>\n'
                elif pct > 0.3:
                    norm_html += 'Partial — shrinkage is one factor among several.</div>\n'
                else:
                    norm_html += ('Low — bias not primarily driven by norm proximity. '
                                  'Other embedding geometry factors dominate.</div>\n')

    # ── Build table rows ──────────────────────────────────────────────────────
    def _build_table_rows(metric='cat'):
        rows = []
        for p in patients_sorted:
            sub  = sig_df[sig_df.patient == p]
            tier = _patient_tier(p, sig_df)
            n_cats  = round(1 / sub['mean_cat_null'].mean()) if sub['mean_cat_null'].mean() > 0 else '?'
            n_words = round(1 / sub['mean_word_null'].mean()) if sub['mean_word_null'].mean() > 0 else '?'
            null_col = sub[f'mean_{metric}_null'].mean()
            cells = []
            for emb in EMBEDDING_NAMES:
                row = sub[sub.embedding == emb]
                if len(row) == 0:
                    cells.append('<td>—</td>')
                    continue
                r   = row.iloc[0]
                acc = r[f'mean_{metric}_obs']
                null = r[f'mean_{metric}_null']
                fc  = acc / null if null > 0 else 0
                sig = r[f'{metric}_sig']
                fmt = f'{acc*100:.1f}%' if metric == 'cat' else f'{acc*100:.2f}%'
                cells.append(
                    f'<td class="data-cell">{fmt} ({fc:.1f}×) '
                    f'<span class="{_sig_class(sig)}">{sig}</span></td>')
            fmt_null = f'{null_col*100:.1f}%' if metric == 'cat' else f'{null_col*100:.2f}%'
            rows.append(
                f'<tr class="{tier}"><td><strong>{p}</strong></td>'
                f'<td>{n_words} / {n_cats}</td>'
                + ''.join(cells)
                + f'<td class="chance-cell">{fmt_null}</td></tr>')
        return '\n'.join(rows)

    cat_rows  = _build_table_rows('cat')
    word_rows = _build_table_rows('word')

    # ── Overview table ────────────────────────────────────────────────────────
    overview_rows = ''
    for emb in EMBEDDING_NAMES:
        mtype = 'Semantic' if emb in SEM_MODELS else 'Visual'
        c = sig_counts[emb]['cat']
        w = sig_counts[emb]['word']
        c_cls = 'sig' if c >= 10 else ('ns' if c < 6 else '')
        w_cls = 'sig' if w >= 10 else ('ns' if w < 6 else '')
        overview_rows += (f'<tr><td><strong>{emb}</strong></td>'
                          f'<td class="{c_cls}">{c}/{n_patients}</td>'
                          f'<td class="{w_cls}">{w}/{n_patients}</td>'
                          f'<td>{mtype}</td></tr>\n')

    # ── Bias table ────────────────────────────────────────────────────────────
    bias_table = ''
    if bias_summary:
        bias_table = ('<table><tr><th>Model</th><th>Favorite Word</th>'
                      '<th>Patients</th><th>Mean % Predictions</th>'
                      '<th>Entropy (norm)</th></tr>\n')
        for b in bias_summary:
            bias_table += (f'<tr><td>{b["emb"]}</td><td><strong>"{b["fav_word"]}"</strong></td>'
                           f'<td>{b["n_patients"]}</td><td>{b["mean_pct"]}</td>'
                           f'<td>{b["mean_ent"]}</td></tr>\n')
        bias_table += '</table>'

    # ── Dissociation HTML ─────────────────────────────────────────────────────
    dissoc_html = ''
    if len(dissoc_df) > 0:
        consistent = 0
        total = dissoc_df.patient.nunique()
        for p in dissoc_df.patient.unique():
            sub = dissoc_df[dissoc_df.patient == p]
            if (sub.loc[sub.best_r2.idxmax(), 'embedding'] ==
                sub.loc[sub.best_cat_acc.idxmax(), 'embedding'] ==
                sub.loc[sub.best_word_acc.idxmax(), 'embedding']):
                consistent += 1
        dissoc_html = (f'<p><strong>{consistent}/{total}</strong> patients have the same '
                       f'model winning all three metrics.</p>')
        d2 = dissoc_df.copy()
        d2['r2_cat_gap']   = np.abs(d2.r2_best_bin - d2.cat_best_bin)
        d2['r2_word_gap']  = np.abs(d2.r2_best_bin - d2.word_best_bin)
        d2['cat_word_gap'] = np.abs(d2.cat_best_bin - d2.word_best_bin)
        dissoc_html += (f'<p>Mean bin gap: R²↔Cat = {d2.r2_cat_gap.mean():.1f}, '
                        f'R²↔Word = {d2.r2_word_gap.mean():.1f}, '
                        f'Cat↔Word = {d2.cat_word_gap.mean():.1f} bins.</p>')

    # ── Semantic vs visual ────────────────────────────────────────────────────
    sem_cat = sum(sig_counts[e]['cat'] for e in SEM_MODELS)
    vis_cat = sum(sig_counts[e]['cat'] for e in VIS_MODELS)

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Semantic Regression Report — {run_id}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; color: #333; line-height: 1.6; }}
  h1 {{ color: #1a5276; border-bottom: 3px solid #2980b9; padding-bottom: 10px; }}
  h2 {{ color: #2471a3; margin-top: 40px; border-bottom: 1px solid #d4e6f1; padding-bottom: 5px; }}
  h3 {{ color: #2e86c1; }}
  .summary-box {{ background: #eaf2f8; border-left: 4px solid #2980b9; padding: 15px; margin: 20px 0; border-radius: 4px; }}
  .finding {{ background: #fef9e7; border-left: 4px solid #f39c12; padding: 15px; margin: 15px 0; border-radius: 4px; }}
  .warning {{ background: #fdedec; border-left: 4px solid #e74c3c; padding: 15px; margin: 15px 0; border-radius: 4px; }}
  .method-box {{ background: #f3e5f5; border-left: 4px solid #8e24aa; padding: 15px; margin: 15px 0; border-radius: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }}
  th {{ background: #2980b9; color: white; padding: 8px 10px; text-align: left; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #ddd; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  .sig {{ color: #27ae60; font-weight: bold; }}
  .ns  {{ color: #e74c3c; }}
  code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
  small {{ color: #888; }}
  .data-cell {{ font-variant-numeric: tabular-nums; text-align: center; }}
  .chance-cell {{ background: #f0f0f0; font-weight: bold; text-align: center; }}
  .star-three {{ color: #1b5e20; font-weight: bold; }}
  .star-two   {{ color: #2e7d32; font-weight: bold; }}
  .star-one   {{ color: #388e3c; }}
  .star-ns    {{ color: #c62828; }}
  .patient-high     td:first-child {{ background: #e8f5e9; font-weight: bold; }}
  .patient-moderate td:first-child {{ background: #fff8e1; }}
  .patient-low      td:first-child {{ background: #ffebee; }}
  .sem-header {{ background: #1565C0; color: white; }}
  .vis-header {{ background: #E65100; color: white; }}
  #cat-table, #word-table {{ font-size: 12px; table-layout: fixed; }}
  #cat-table th, #word-table th {{ padding: 6px 5px; text-align: center; font-size: 11px; }}
  #cat-table td, #word-table td {{ padding: 5px; text-align: center; font-size: 11.5px; }}
</style></head><body>

<h1>Semantic Regression: Cross-Patient Analysis</h1>
<p><strong>Run:</strong> <code>{run_id}</code> &nbsp;|&nbsp;
   <strong>Pipeline:</strong> <code>{pipeline_str}</code> &nbsp;|&nbsp;
   <strong>Retrieval:</strong> {closest_mode} &nbsp;|&nbsp;
   <strong>Test:</strong> Wilcoxon vs. shuffled null, Bonferroni ({n_tests} tests)</p>

<div class="summary-box">
<h3>Executive Summary</h3>
<p><strong>Category: {n_cat_sig}/{n_tests} ({n_cat_sig*100//n_tests}%) significant</strong> after
Bonferroni correction. Word: {n_word_sig}/{n_tests} ({n_word_sig*100//n_tests}%).
Strongest: {", ".join(patients_sorted[:3])}.</p>
</div>

<h2>1. Significance Testing</h2>
<div class="method-box">
<strong>Method:</strong> Internal shuffled null preserves all pipeline biases.
At each patient x embedding's best bin, 50 obs vs 50 null epoch accuracies
are compared via one-sided Wilcoxon signed-rank, Bonferroni-corrected ({n_tests} tests).
</div>

<h3>Per-Model Significance</h3>
<table><tr><th>Model</th><th>Cat Sig</th><th>Word Sig</th><th>Type</th></tr>
{overview_rows}</table>

<h3>Category Decoding</h3>
<p style="font-size:12px;">
<span class="star-three">*** p&lt;0.001</span> &nbsp;
<span class="star-two">** p&lt;0.01</span> &nbsp;
<span class="star-one">* p&lt;0.05</span> &nbsp;
<span class="star-ns">NS</span> (Bonferroni)</p>
<table id="cat-table">
<tr><th>Patient</th><th>N words/cats</th>
<th class="sem-header">GloVe</th><th class="sem-header">FastText</th>
<th class="sem-header">Word2Vec</th><th class="sem-header">ConceptNet</th>
<th class="vis-header">DINOv2</th><th class="vis-header">SimCLR</th>
<th>Null</th></tr>
{cat_rows}</table>

<h3>Word Decoding</h3>
<div class="warning"><strong>Interpret with caution</strong> — word predictions may be
dominated by prediction bias (see Section 2).</div>
<table id="word-table">
<tr><th>Patient</th><th>N words/cats</th>
<th class="sem-header">GloVe</th><th class="sem-header">FastText</th>
<th class="sem-header">Word2Vec</th><th class="sem-header">ConceptNet</th>
<th class="vis-header">DINOv2</th><th class="vis-header">SimCLR</th>
<th>Null</th></tr>
{word_rows}</table>

<h2>2. Word Prediction Bias</h2>
{bias_table if bias_table else '<p><em>Bias analysis skipped.</em></p>'}

{norm_html}

<h2>3. Metric Dissociation</h2>
{dissoc_html if dissoc_html else '<p><em>No data.</em></p>'}

<h2>4. Semantic vs. Visual</h2>
<table><tr><th>Group</th><th>Cat Sig</th><th>Per Model</th></tr>
<tr><td>Semantic</td><td>{sem_cat}/{n_patients*4}</td>
<td>{"  |  ".join(f"{e}: {sig_counts[e]['cat']}/{n_patients}" for e in ['GloVe','FastText','Word2Vec','ConceptNet'])}</td></tr>
<tr><td>Visual</td><td>{vis_cat}/{n_patients*2}</td>
<td>{"  |  ".join(f"{e}: {sig_counts[e]['cat']}/{n_patients}" for e in ['DINOv2','SimCLR'])}</td></tr>
</table>

</body></html>'''

    out_path = os.path.join(out_dir, 'analysis_report.html')
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(html)
    print(f"[Report] Saved: {out_path} ({len(html)//1024} KB)")
    return out_path
