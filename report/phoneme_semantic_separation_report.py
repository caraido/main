# -*- coding: utf-8 -*-
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
    python -m analysis.phoneme_semantic_separation_report
    python -m analysis.phoneme_semantic_separation_report --in-dir test_results/

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

from analysis.helpers._phoneme_semantic_helpers import (
    PHONEME_EMBEDDINGS, N_BINS_HISTORY, header, get_out_dir,
)

# --- cleanup batch 1: imports added by automated migration ---
from report.helper.html_utils import fig_to_base64

EMB_COLORS = {'panphon': '#1565C0', 'token_ipa': '#E65100'}


# ── Plotting helpers ─────────────────────────────────────────────────────


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

    patients = summary['patient'].unique()

    # ── Fig 1: Bar + per-fold scatter ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, fold_col, metric_name, chance_col in [
        (axes[0], 'word_acc', 'word_bal_acc', 'Word Balanced Acc', 'word_chance'),
        (axes[1], 'cat_acc', 'cat_indep_bal_acc', 'Category Indep Acc', 'cat_chance'),
    ]:
        x = np.arange(len(patients))
        width = 0.35
        for i, emb in enumerate(PHONEME_EMBEDDINGS):
            sub = summary[summary['embedding'] == emb]
            vals = [float(sub[sub['patient'] == p][metric].iloc[0])
                    if len(sub[sub['patient'] == p]) > 0 else 0
                    for p in patients]
            ax.bar(x + i * width, vals, width, label=emb,
                   color=EMB_COLORS.get(emb, f'C{i}'), alpha=0.7)
            # Overlay per-fold scatter
            fold_sub = df[df['embedding'] == emb]
            for pi, p in enumerate(patients):
                fold_vals = fold_sub[fold_sub['patient'] == p][fold_col].values
                jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(fold_vals))
                ax.scatter(np.full(len(fold_vals), x[pi] + i * width) + jitter,
                           fold_vals, color='k', s=12, alpha=0.5, zorder=3)
        chance_val = summary[chance_col].mean()
        ax.axhline(chance_val, color='red', ls='--', alpha=0.7, label=f'chance={chance_val:.3f}')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(patients, rotation=45, ha='right')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} (cross-category)')
        ax.legend(fontsize=8)

    fig.tight_layout()
    img1 = fig_to_base64(fig)

    # ── Fig 2: Cat acc relative to chance heatmap ────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    for ax, emb in zip(axes2, PHONEME_EMBEDDINGS):
        sub = summary[summary['embedding'] == emb].set_index('patient')
        delta = (sub['cat_acc'] - sub['cat_chance']).reindex(patients)
        colors = ['#c62828' if v < 0 else '#1565C0' for v in delta]
        bars = ax.barh(np.arange(len(patients)), delta.values, color=colors, alpha=0.85)
        ax.axvline(0, color='k', lw=1)
        ax.set_yticks(np.arange(len(patients)))
        ax.set_yticklabels(patients)
        ax.set_xlabel('Cat Acc − Chance')
        ax.set_title(f'{emb}: below-chance = red, above = blue')
        # Annotate values
        for bar, v in zip(bars, delta.values):
            ax.text(v + (0.002 if v >= 0 else -0.002), bar.get_y() + bar.get_height() / 2,
                    f'{v:+.3f}', va='center', ha='left' if v >= 0 else 'right', fontsize=7)
    fig2.suptitle('Cross-category cat accuracy vs chance (negative = below chance)', fontsize=11)
    fig2.tight_layout()
    img2 = fig_to_base64(fig2)

    return f"""
    <h2>1. Cross-Category Generalization</h2>
    <p><b>Question:</b> Can phoneme decoding generalize to semantic categories
    never seen during training?</p>
    <p><b>Method:</b> Leave-K-categories-out CV for phoneme regression.
    Model trained on trials from a subset of categories, tested on held-out categories.
    Dots show individual fold values.</p>
    <p><b>Interpretation:</b> If word accuracy is above chance on held-out categories,
    the model uses phonological information that transfers across semantic contexts.
    Below-chance performance (red bars) indicates zero phonological generalization —
    the strongest causal evidence against phonological structure.</p>
    <img src="data:image/png;base64,{img1}" style="max-width:100%">
    <img src="data:image/png;base64,{img2}" style="max-width:100%">
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

    # ── Fig A: Bar chart with all three conditions ───────────────────────
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
    imgA = fig_to_base64(fig)

    # ── Fig B: Delta heatmap — accuracy drop (normal → residualized) ─────
    # One subplot per embedding, rows=patients, single column = cat_acc drop
    phon_embs = summary['phon_emb'].unique()
    fig2, axes2 = plt.subplots(1, len(phon_embs), figsize=(5 * len(phon_embs), max(4, len(patients) * 0.5 + 1)))
    if len(phon_embs) == 1:
        axes2 = [axes2]
    for ax, emb in zip(axes2, phon_embs):
        deltas = []
        for p in patients:
            norm_row = summary[(summary['patient'] == p) & (summary['phon_emb'] == emb) & (summary['condition'] == 'normal')]
            resid_row = summary[(summary['patient'] == p) & (summary['phon_emb'] == emb) & (summary['condition'] == 'residualized')]
            if len(norm_row) == 0 or len(resid_row) == 0:
                deltas.append(np.nan)
            else:
                deltas.append(float(resid_row['cat_acc'].values[0]) - float(norm_row['cat_acc'].values[0]))
        deltas = np.array(deltas)
        colors = ['#c62828' if np.isnan(v) or v < 0 else '#2e7d32' for v in deltas]
        bars = ax.barh(np.arange(len(patients)), deltas, color=colors, alpha=0.85)
        ax.axvline(0, color='k', lw=1)
        ax.set_yticks(np.arange(len(patients)))
        ax.set_yticklabels(patients)
        ax.set_xlabel('Δ Cat Acc (residualized − normal)')
        ax.set_title(f'{emb}\nred=drop, green=gain')
        for bar, v in zip(bars, deltas):
            if not np.isnan(v):
                ax.text(v + (0.001 if v >= 0 else -0.001), bar.get_y() + bar.get_height() / 2,
                        f'{v:+.3f}', va='center', ha='left' if v >= 0 else 'right', fontsize=7)
    fig2.suptitle('Cat accuracy change after projecting out semantic subspace', fontsize=11)
    fig2.tight_layout()
    imgB = fig_to_base64(fig2)

    # ── Fig C: sem_only vs normal scatter ────────────────────────────────
    fig3, axes3 = plt.subplots(1, len(phon_embs), figsize=(5 * len(phon_embs), 4))
    if len(phon_embs) == 1:
        axes3 = [axes3]
    for ax, emb in zip(axes3, phon_embs):
        norm_cats, sem_cats = [], []
        for p in patients:
            norm_row = summary[(summary['patient'] == p) & (summary['phon_emb'] == emb) & (summary['condition'] == 'normal')]
            sem_row  = summary[(summary['patient'] == p) & (summary['phon_emb'] == emb) & (summary['condition'] == 'sem_only')]
            if len(norm_row) > 0 and len(sem_row) > 0:
                norm_cats.append(float(norm_row['cat_acc'].values[0]))
                sem_cats.append(float(sem_row['cat_acc'].values[0]))
                ax.annotate(p, (float(norm_row['cat_acc'].values[0]), float(sem_row['cat_acc'].values[0])),
                            fontsize=7, alpha=0.8)
        if norm_cats:
            ax.scatter(norm_cats, sem_cats, color=EMB_COLORS.get(emb, 'C0'), s=60, zorder=3)
            lo = min(min(norm_cats), min(sem_cats)) - 0.02
            hi = max(max(norm_cats), max(sem_cats)) + 0.02
            ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, label='y=x')
            ax.set_xlabel('Normal cat acc')
            ax.set_ylabel('Sem-only cat acc')
            ax.set_title(f'{emb}: sem_only vs normal\n(above diagonal = sem_only wins)')
            ax.legend(fontsize=7)
    fig3.tight_layout()
    imgC = fig_to_base64(fig3)

    return f"""
    <h2>2. Semantic Residualization</h2>
    <p><b>Question:</b> Does phoneme decoding survive after surgically removing
    semantic neural dimensions?</p>
    <p><b>Method:</b> Fit semantic PLS to find "semantic directions" in neural space,
    project them out (Gram-Schmidt), then fit phoneme PLS on the residual. Three conditions:
    <em>normal</em> (unmodified X), <em>residualized</em> (semantic projected out),
    <em>sem_only</em> (only semantic subspace, sanity check).</p>
    <p><b>Interpretation:</b> A large drop (red, Fig B) after residualization means the
    phoneme model was exploiting semantic dimensions. If <em>sem_only</em> outperforms
    <em>normal</em> (above diagonal in Fig C), semantic information alone drives category
    accuracy — the strongest evidence of co-representation.</p>
    <img src="data:image/png;base64,{imgA}" style="max-width:100%">
    <b>Fig B — Accuracy drop after removing semantic subspace (negative = model was using semantics):</b>
    <img src="data:image/png;base64,{imgB}" style="max-width:100%">
    <b>Fig C — sem_only vs normal category accuracy (above diagonal = semantic alone outperforms phoneme model):</b>
    <img src="data:image/png;base64,{imgC}" style="max-width:100%">
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

    BONF_ALPHA = 0.05 / 58   # 58 time bins

    for patient in patients:
        pat = df[df['patient'] == patient]
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(f'{patient} — Partial RSA', fontsize=14)

        for j, phon in enumerate(PHONEME_EMBEDDINGS):
            sub = pat[pat['phon_emb'] == phon].sort_values('bin_index')
            if len(sub) == 0:
                continue

            t = sub['time_ms'].values
            # Bonferroni-significant bins (using raw p-values as proxy for partial)
            sig_phon = sub['p_pred_phon'].values < BONF_ALPHA
            sig_sem  = sub['p_pred_sem'].values  < BONF_ALPHA

            # Full correlations
            axes[0, j].plot(t, sub['r_pred_phon'], color=EMB_COLORS[phon],
                           label='r(pred, phoneme)')
            axes[0, j].plot(t, sub['r_pred_sem'], color='#E91E63', ls='--',
                           label='r(pred, semantic)')
            # Significance ticks at bottom
            axes[0, j].scatter(t[sig_phon], np.full(sig_phon.sum(), sub['r_pred_phon'].min() - 0.005),
                               marker='|', color=EMB_COLORS[phon], s=30, label='_Bonf. sig (phon)')
            axes[0, j].scatter(t[sig_sem], np.full(sig_sem.sum(), sub['r_pred_phon'].min() - 0.010),
                               marker='|', color='#E91E63', s=30, label='_Bonf. sig (sem)')
            axes[0, j].axhline(0, color='grey', lw=0.5)
            axes[0, j].axvline(0, color='grey', lw=0.5, ls=':')
            axes[0, j].set_title(f'{phon} — Full correlations\n(ticks = Bonferroni p<{BONF_ALPHA:.4f})')
            axes[0, j].set_ylabel('Spearman r')
            axes[0, j].legend(fontsize=7)

            # Partial correlations
            axes[1, j].plot(t, sub['r_partial_phon'], color=EMB_COLORS[phon],
                           label='r_partial(pred, phon | sem)')
            axes[1, j].plot(t, sub['r_partial_sem'], color='#E91E63', ls='--',
                           label='r_partial(pred, sem | phon)')
            # Shade significant windows
            for k in range(len(t)):
                if sig_phon[k] and k < len(t):
                    axes[1, j].axvspan(t[k] - 50, t[k] + 50, alpha=0.08,
                                       color=EMB_COLORS[phon])
                if sig_sem[k] and k < len(t):
                    axes[1, j].axvspan(t[k] - 50, t[k] + 50, alpha=0.08,
                                       color='#E91E63')
            axes[1, j].axhline(0, color='grey', lw=0.5)
            axes[1, j].axvline(0, color='grey', lw=0.5, ls=':')
            axes[1, j].set_title(f'{phon} — Partial correlations\n(shaded = Bonferroni sig on raw r)')
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

    # ── Group-average time-course with SEM ───────────────────────────────
    group_figs = ""
    for phon in PHONEME_EMBEDDINGS:
        sub_all = df[df['phon_emb'] == phon].sort_values(['patient', 'bin_index'])
        times = sub_all['time_ms'].unique()
        times = np.sort(times)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.suptitle(f'Group average — {phon}', fontsize=12)

        for ax, rcol, label, color in [
            (axes[0], 'r_pred_phon',    'r(pred, phoneme)',          EMB_COLORS[phon]),
            (axes[0], 'r_pred_sem',     'r(pred, semantic)',         '#E91E63'),
            (axes[1], 'r_partial_phon', 'r_partial(pred, phon|sem)', EMB_COLORS[phon]),
            (axes[1], 'r_partial_sem',  'r_partial(pred, sem|phon)', '#E91E63'),
        ]:
            vals_by_time = []
            for t in times:
                pat_vals = sub_all[sub_all['time_ms'] == t][rcol].dropna().values
                vals_by_time.append(pat_vals)
            means = np.array([v.mean() if len(v) > 0 else np.nan for v in vals_by_time])
            sems  = np.array([v.std() / np.sqrt(len(v)) if len(v) > 1 else 0 for v in vals_by_time])
            ls = '--' if 'sem' in rcol and 'partial' not in rcol else ('--' if rcol == 'r_partial_sem' else '-')
            ax.plot(times, means, color=color, ls=ls, lw=2, label=label)
            ax.fill_between(times, means - sems, means + sems, color=color, alpha=0.15)

        for ax in axes:
            ax.axhline(0, color='grey', lw=0.5)
            ax.axvline(0, color='grey', lw=0.5, ls=':')
            ax.set_xlabel('Time from onset (ms)')
            ax.set_ylabel('Spearman r')
            ax.legend(fontsize=8)
        axes[0].set_title('Full correlations (mean ± SEM across patients)')
        axes[1].set_title('Partial correlations (mean ± SEM across patients)')
        fig.tight_layout()
        group_figs += f'<img src="data:image/png;base64,{fig_to_base64(fig)}" style="max-width:100%">'

    # ── Peak timing heatmap ───────────────────────────────────────────────
    fig_ht, axes_ht = plt.subplots(1, 2, figsize=(12, max(4, len(patients) * 0.45 + 1.5)))
    for ax, rcol, title in [
        (axes_ht[0], 'r_partial_phon', 'Peak time: r_partial_phon (ms)'),
        (axes_ht[1], 'r_partial_sem',  'Peak time: r_partial_sem (ms)'),
    ]:
        data_mat = np.full((len(patients), len(PHONEME_EMBEDDINGS)), np.nan)
        for pi, p in enumerate(patients):
            for ei, emb in enumerate(PHONEME_EMBEDDINGS):
                sub = df[(df['patient'] == p) & (df['phon_emb'] == emb)].dropna(subset=[rcol])
                if len(sub) > 0:
                    data_mat[pi, ei] = sub.loc[sub[rcol].idxmax(), 'time_ms']
        im = ax.imshow(data_mat, aspect='auto', cmap='RdBu_r', vmin=-500, vmax=2500)
        ax.set_xticks(range(len(PHONEME_EMBEDDINGS)))
        ax.set_xticklabels(PHONEME_EMBEDDINGS)
        ax.set_yticks(range(len(patients)))
        ax.set_yticklabels(patients)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, label='ms')
        for pi in range(len(patients)):
            for ei in range(len(PHONEME_EMBEDDINGS)):
                if not np.isnan(data_mat[pi, ei]):
                    ax.text(ei, pi, f'{data_mat[pi, ei]:.0f}', ha='center', va='center',
                            fontsize=7, color='k')
    fig_ht.suptitle('Peak timing of partial correlations per patient × embedding', fontsize=11)
    fig_ht.tight_layout()
    img_ht = fig_to_base64(fig_ht)

    return f"""
    <h2>3. Partial RSA</h2>
    <p><b>Question:</b> What fraction of neural prediction geometry is uniquely
    phonological vs uniquely semantic?</p>
    <p><b>Method:</b> Spearman correlation between neural prediction RDM and
    phoneme/semantic ground-truth RDMs, with partial correlation to control for
    the other modality. Word-stratified train/test split guarantees every word
    appears in the RDM.</p>
    <p><b>Note on significance:</b> Shading and tick marks indicate Bonferroni-corrected
    bins (α = 0.05/58 = {BONF_ALPHA:.5f}) based on <em>raw</em> p-values (p_pred_phon,
    p_pred_sem). Partial correlations do not have analytic p-values — significance is
    inferred from the raw correlations as a conservative proxy.</p>
    <p><b>Interpretation:</b> Temporal dissociation between semantic peak (early, 400–800 ms)
    and phonological peak (late, 1500–2500 ms) confirms genuine co-representation in
    partially separable neural dimensions.</p>

    <b>Group-average time-courses (shaded = ±SEM across patients):</b>
    {group_figs}

    <b>Peak timing heatmap (when does each patient peak for each modality?):</b>
    <img src="data:image/png;base64,{img_ht}" style="max-width:100%">

    <table class="data">
    <tr><th>Patient</th><th>Phon Emb</th>
        <th>Peak r_partial_phon</th><th>@ ms</th>
        <th>Peak r_partial_sem</th><th>@ ms</th>
        <th>r(phon,sem)</th></tr>
    {summary_rows}
    </table>

    <b>Per-patient time-courses:</b>
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
                        help='Input directory with CSV files (default: '
                             'results/phoneme_semantic_dissociation, resolved through '
                             'utils.paths.results_dir). The previous help text named a '
                             'root the 2026-07 reorganisation deleted; the code already '
                             'resolved correctly, only the text was wrong.')
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
