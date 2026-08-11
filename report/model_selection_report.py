# -*- coding: utf-8 -*-
"""
tests.model_selection_report — Summarise model comparison and n_components sweep results.

Reads model_comparison_*.csv and pls_learning_curve_*.csv from a results directory
and answers: which model (Ridge / KRR / PLS / Kernel PLS) should you use,
and what is the optimal n_components for PLS?

Outputs a standalone HTML with bar charts per metric, per-patient breakdowns,
Wilcoxon significance tables, and PLS learning curves.

Usage:
    python -m analysis.model_selection_report --results_dir <path> --out <path>
"""
import argparse, glob, os, json, warnings
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis.helpers._phoneme_semantic_helpers import get_out_dir
from report.helper.html_utils import fig_to_base64
from report.render import stylesheet

warnings.filterwarnings('ignore')

# ── colour palette ────────────────────────────────────────────────────────────
COLOURS = {
    'linear_ridge':  '#4e79a7',
    'krr':           '#f28e2b',
    'pls':           '#59a14f',
    'kernel_pls':    '#e15759',
}
MODEL_ORDER  = ['linear_ridge', 'krr', 'pls', 'kernel_pls']
MODEL_LABELS = {
    'linear_ridge': 'Linear Ridge',
    'krr':          'Kernel Ridge (KRR)',
    'pls':          'PLS',
    'kernel_pls':   'Kernel PLS',
}
METRIC_LABELS = {
    'test_r2':       'Test R²',
    'delta_r2':      'ΔR² (vs chance)',
    'cat_bal_acc':   'Cat. Bal. Acc.',
    'word_bal_acc':  'Word Bal. Acc.',
    'test_cosine':   'Cosine Similarity',
    'pred_entropy_norm': 'Pred. Entropy (bias↓)',
}

# ── Chart helpers ─────────────────────────────────────────────────────────────
# These were ~200 lines of hand-composed SVG: manual axis transforms, tick loops,
# and error-bar caps drawn as three <line> elements each. Rewritten on matplotlib
# 2026-08-11 (Alec). `width`/`height` stay in PIXELS so the call sites are unchanged;
# _figsize converts. Charts render as inline base64 PNG, like every other report.

CMAP = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2', '#edc948']
_DPI = 130


def _figsize(width, height):
    """Pixel dimensions -> matplotlib inches at _DPI."""
    return (width / _DPI * 1.3, height / _DPI * 1.3)


def _img(fig):
    return '<img alt="" src="data:image/png;base64,{}" />'.format(
        fig_to_base64(fig, dpi=_DPI))


def _style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888')
    ax.spines['bottom'].set_color('#888')
    ax.tick_params(labelsize=8, colors='#333')


def _bar_chart(series_dict, title, ylabel, width=520, height=280,
               colour_map=None, ymin=None, ymax=None, baseline=None,
               baseline_label='chance'):
    """series_dict: {label: value} or {label: (mean, se)}. Returns an <img> tag."""
    labels = list(series_dict.keys())
    if not labels:
        return '<p class="subtle">(no data)</p>'
    raw   = list(series_dict.values())
    means = [r[0] if isinstance(r, tuple) else r for r in raw]
    ses   = [r[1] if isinstance(r, tuple) else 0.0 for r in raw]

    colours = [(colour_map or {}).get(l, CMAP[0]) for l in labels]
    short = [l.replace('linear_ridge', 'Ridge').replace('kernel_pls', 'KPLS')
              .replace('krr', 'KRR').replace('pls', 'PLS') for l in labels]

    fig, ax = plt.subplots(figsize=_figsize(width, height))
    x = np.arange(len(labels))
    ax.bar(x, means, width=0.55, color=colours, alpha=0.85,
           yerr=[s if s > 0 else 0 for s in ses],
           error_kw=dict(ecolor='#333', capsize=4, lw=1.3))

    # Label above the ERROR BAR, not the bar top -- at the bar top it collides with
    # the upper cap whenever se > 0. Negative bars get their label below.
    for xi, m, s in zip(x, means, ses):
        top = m + s if m >= 0 else m - s
        ax.annotate('{:.3f}'.format(m), (xi, top), textcoords='offset points',
                    xytext=(0, 4 if m >= 0 else -11), ha='center', fontsize=7.5)

    if baseline is not None:
        ax.axhline(baseline, color='#aaa', lw=1.2, ls=(0, (5, 3)))
        ax.annotate(baseline_label, (0.99, baseline), xycoords=('axes fraction', 'data'),
                    textcoords='offset points', xytext=(-2, 3),
                    ha='right', va='bottom', fontsize=7.5, color='#999',
                    annotation_clip=False)

    lo = min(means + [m - s for m, s in zip(means, ses)] + ([baseline] if baseline is not None else []))
    hi = max(means + [m + s for m, s in zip(means, ses)] + ([baseline] if baseline is not None else []))
    # Pad by a fraction of the RANGE, not of each endpoint: scaling `lo` by a factor
    # gives almost no room when lo is small and negative, and the below-bar value
    # label then lands on top of the x tick labels.
    span = (hi - lo) or abs(hi) or 1.0
    ax.set_ylim(ymin if ymin is not None else (lo - 0.16 * span if lo < 0 else 0),
                ymax if ymax is not None else hi + 0.14 * span)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axhline(0, color='#888', lw=1)
    _style(ax)
    fig.tight_layout()
    return _img(fig)


def _line_chart(curves_dict, title, xlabel, ylabel,
                width=540, height=280, se_dict=None,
                baseline=None, baseline_label='chance',
                highlight_x=None, highlight_label=''):
    """curves_dict: {label: {x: y}}; se_dict: {label: {x: se}} for ±1 SE shading."""
    if not curves_dict:
        return '<p class="subtle">(no data)</p>'
    all_xs = sorted({x for d in curves_dict.values() for x in d})

    fig, ax = plt.subplots(figsize=_figsize(width, height))
    for idx, (lbl, curve) in enumerate(curves_dict.items()):
        col = CMAP[idx % len(CMAP)]
        xs = sorted(curve)
        ys = [curve[x] for x in xs]
        if se_dict and lbl in se_dict:
            sd = se_dict[lbl]
            ax.fill_between(xs,
                            [curve[x] - sd.get(x, 0) for x in xs],
                            [curve[x] + sd.get(x, 0) for x in xs],
                            color=col, alpha=0.15, lw=0)
        ax.plot(xs, ys, color=col, lw=2.2, marker='o', ms=4, label=lbl)

    if baseline is not None:
        ax.axhline(baseline, color='#aaa', lw=1, ls=(0, (5, 3)))
        ax.annotate(baseline_label, (0.99, baseline), xycoords=('axes fraction', 'data'),
                    textcoords='offset points', xytext=(-2, 3),
                    ha='right', va='bottom', fontsize=7.5, color='#999',
                    annotation_clip=False)
    if highlight_x is not None:
        ax.axvline(highlight_x, color='#e15759', lw=1.5, ls=(0, (5, 3)), alpha=0.7)
        ax.annotate(highlight_label, (highlight_x, 1.0), xycoords=('data', 'axes fraction'),
                    textcoords='offset points', xytext=(3, -12),
                    fontsize=7.5, color='#e15759')

    ax.set_xticks(all_xs)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=7.5, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
    _style(ax)
    fig.tight_layout()
    return _img(fig)


# ── Analysis helpers ──────────────────────────────────────────────────────────

def load_model_comparison(results_dir):
    dfs = []
    for f in glob.glob(os.path.join(results_dir, 'model_comparison_*.csv')):
        dfs.append(pd.read_csv(f))
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    # add cosine if missing (AA/VB didn't have it)
    if 'test_cosine' not in df.columns:
        df['test_cosine'] = np.nan
    return df


def load_pls_learning(results_dir):
    dfs = []
    for f in glob.glob(os.path.join(results_dir, 'pls_learning_curve_*.csv')):
        dfs.append(pd.read_csv(f))
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def wilcoxon_p(a, b):
    """One-sided Wilcoxon: a > b?  Returns p-value."""
    diff = np.array(a) - np.array(b)
    diff = diff[diff != 0]
    if len(diff) < 3:
        return np.nan
    try:
        _, p = stats.wilcoxon(diff, alternative='greater')
        return p
    except Exception:
        return np.nan


# ── Section 1: Model comparison ───────────────────────────────────────────────

def analyze_model_comparison(df):
    patients = sorted(df['patient'].unique())
    embeddings = sorted(df['embedding'].unique())
    metrics = ['test_r2', 'delta_r2', 'cat_bal_acc', 'word_bal_acc', 'test_cosine', 'pred_entropy_norm']

    # ---- aggregate: mean ± SE across patients × embeddings per model
    agg = df.groupby('model')[metrics].agg(['mean','sem']).reset_index()

    # ---- nonlinearity effect: Ridge vs KRR, PLS vs KernelPLS
    nl_pairs = [('linear_ridge', 'krr'), ('pls', 'kernel_pls')]
    # ---- PLS effect: Ridge vs PLS, KRR vs KernelPLS
    pls_pairs = [('linear_ridge', 'pls'), ('krr', 'kernel_pls')]

    return agg, nl_pairs, pls_pairs, patients, embeddings


def _model_means_with_se(df, metric):
    """Returns {model: (mean, se)} for a given metric."""
    g = df.groupby('model')[metric]
    return {m: (g.mean()[m], g.sem()[m]) for m in MODEL_ORDER if m in g.mean().index}


def _effect_table(df, pairs, metric_cols):
    rows = []
    for (a, b) in pairs:
        da = df[df.model == a]
        db = df[df.model == b]
        row = {'comparison': f'{MODEL_LABELS[a]}  →  {MODEL_LABELS[b]}'}
        for m in metric_cols:
            av = da[m].dropna().values
            bv = db[m].dropna().values
            delta = np.mean(bv) - np.mean(av)
            p = wilcoxon_p(bv, av)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            row[m] = f'{delta:+.3f} ({sig})'
        rows.append(row)
    return pd.DataFrame(rows)


# ── Section 2: PLS learning curve ─────────────────────────────────────────────

def analyze_pls_learning(df):
    """Returns per-patient per-embedding curves: {patient→embed→{n_comp: (mean_test, se_test, mean_train, se_train)}}"""
    result = {}
    for (pat, emb, model), gdf in df.groupby(['patient','embedding','model']):
        per_nc = gdf.groupby('n_components').agg(
            test_r2_mean=('test_r2','mean'), test_r2_se=('test_r2','sem'),
            train_r2_mean=('train_r2','mean'), train_r2_se=('train_r2','sem'),
            test_cos_mean=('test_cosine','mean'), test_cos_se=('test_cosine','sem'),
            train_cos_mean=('train_cosine','mean'), train_cos_se=('train_cosine','sem'),
            word_acc_mean=('word_bal_acc','mean'), word_acc_se=('word_bal_acc','sem'),
            cat_acc_mean=('cat_bal_acc','mean'), cat_acc_se=('cat_bal_acc','sem'),
        ).reset_index()
        key = (pat, emb, model)
        result[key] = per_nc
    return result


def best_n_components(per_nc_df):
    """Find n_components maximising each metric without overfitting (gap < 0.10)."""
    df = per_nc_df.copy()
    df['gap'] = df['train_r2_mean'] - df['test_r2_mean']
    ok = df[df['gap'] < 0.10]
    if ok.empty:
        ok = df
    best_cos  = ok.loc[ok['test_cos_mean'].idxmax()]
    best_r2   = ok.loc[ok['test_r2_mean'].idxmax()]
    best_word = ok.loc[ok['word_acc_mean'].idxmax()]
    best_cat  = ok.loc[ok['cat_acc_mean'].idxmax()]
    return (int(best_cos['n_components']),  float(best_cos['test_cos_mean']),
            int(best_r2['n_components']),   float(best_r2['test_r2_mean']),
            int(best_word['n_components']), float(best_word['word_acc_mean']),
            int(best_cat['n_components']),  float(best_cat['cat_acc_mean']))


# ── HTML generation ───────────────────────────────────────────────────────────

def build_html(mc_df, lc_df, results_dir):
    mc_agg, nl_pairs, pls_pairs, patients, embeddings = analyze_model_comparison(mc_df)
    lc_curves = analyze_pls_learning(lc_df) if lc_df is not None else {}

    has_cosine = mc_df['test_cosine'].notna().any()
    mc_metrics = ['delta_r2', 'cat_bal_acc', 'word_bal_acc']
    if has_cosine:
        mc_metrics.append('test_cosine')

    # ── build SVG figures ────────────────────────────────────────────────────
    figs_model = {}
    for metric in mc_metrics:
        series = {m: (_model_means_with_se(mc_df, metric)[m])
                  for m in MODEL_ORDER if m in mc_df['model'].unique()}
        col_map = {m: COLOURS[m] for m in MODEL_ORDER}
        baseline = 0.0 if metric in ('delta_r2','test_r2','test_cosine') else None
        bl_lbl = 'zero' if baseline == 0.0 else None
        figs_model[metric] = _bar_chart(
            series, title=METRIC_LABELS.get(metric, metric),
            ylabel=METRIC_LABELS.get(metric, metric),
            colour_map={k: COLOURS[k] for k in series},
            baseline=baseline, baseline_label=bl_lbl or ''
        )

    # ── per-patient model bar charts ─────────────────────────────────────────
    figs_patient = {}
    for pat in patients:
        pdf = mc_df[mc_df.patient == pat]
        for metric in mc_metrics:
            series = {}
            for m in MODEL_ORDER:
                vals = pdf[pdf.model == m][metric].dropna()
                if len(vals):
                    series[m] = (vals.mean(), vals.sem())
            figs_patient[(pat, metric)] = _bar_chart(
                series, title=f'{pat}', ylabel='',
                colour_map={k: COLOURS[k] for k in series},
                baseline=0.0 if metric in ('delta_r2','test_r2','test_cosine') else None,
                baseline_label='zero'
            )

    # ── learning curve figures ───────────────────────────────────────────────
    figs_lc = {}
    best_nc_summary = []
    lc_patients = sorted(set(k[0] for k in lc_curves))
    lc_embeddings = sorted(set(k[1] for k in lc_curves))

    for emb in lc_embeddings:
        # R² and cosine: show train + test per patient
        for metric_pair in [('test_r2_mean','train_r2_mean','test_r2_se','train_r2_se','R²'),
                             ('test_cos_mean','train_cos_mean','test_cos_se','train_cos_se','Cosine Sim.')]:
            test_key, train_key, test_se_key, train_se_key, metric_name = metric_pair
            curves = {}
            ses    = {}
            for pat in lc_patients:
                key = (pat, emb, 'pls')
                if key not in lc_curves:
                    continue
                d = lc_curves[key]
                curves[f'{pat} test']  = dict(zip(d['n_components'], d[test_key]))
                curves[f'{pat} train'] = dict(zip(d['n_components'], d[train_key]))
                ses[f'{pat} test']     = dict(zip(d['n_components'], d[test_se_key]))
                ses[f'{pat} train']    = dict(zip(d['n_components'], d[train_se_key]))
            if curves:
                figs_lc[(emb, metric_name)] = _line_chart(
                    curves, title=f'{emb} — {metric_name} vs n_components',
                    xlabel='n_components', ylabel=metric_name,
                    se_dict=ses,
                    baseline=0.0 if metric_name == 'R²' else None,
                    baseline_label='zero',
                    highlight_x=4, highlight_label='n=4'
                )

        # Word and category accuracy: test only (no separate train tracked)
        for val_key, se_key, metric_name in [
                ('word_acc_mean', 'word_acc_se', 'Word Acc. (test)'),
                ('cat_acc_mean',  'cat_acc_se',  'Cat. Acc. (test)')]:
            curves = {}
            ses    = {}
            for pat in lc_patients:
                key = (pat, emb, 'pls')
                if key not in lc_curves:
                    continue
                d = lc_curves[key]
                curves[pat] = dict(zip(d['n_components'], d[val_key]))
                ses[pat]    = dict(zip(d['n_components'], d[se_key]))
            if curves:
                figs_lc[(emb, metric_name)] = _line_chart(
                    curves, title=f'{emb} — {metric_name} vs n_components',
                    xlabel='n_components', ylabel=metric_name,
                    se_dict=ses,
                    baseline=None,
                    highlight_x=4, highlight_label='n=4'
                )

    # best n_components summary
    for (pat, emb, model), d in lc_curves.items():
        if model != 'pls':
            continue
        nc_cos, cos_val, nc_r2, r2_val, nc_word, word_val, nc_cat, cat_val = best_n_components(d)
        best_nc_summary.append({
            'patient': pat, 'embedding': emb,
            'best_nc_cosine': nc_cos,  'best_cosine': round(cos_val, 3),
            'best_nc_word':   nc_word, 'best_word_acc': round(word_val, 3),
            'best_nc_cat':    nc_cat,  'best_cat_acc': round(cat_val, 3),
            'best_nc_r2':     nc_r2,   'best_r2': round(r2_val, 3),
        })
    best_nc_df = pd.DataFrame(best_nc_summary) if best_nc_summary else pd.DataFrame()

    # ── effect tables ─────────────────────────────────────────────────────────
    nl_table  = _effect_table(mc_df, nl_pairs, mc_metrics)
    pls_table = _effect_table(mc_df, pls_pairs, mc_metrics)

    # ── statistical summary ───────────────────────────────────────────────────
    summary_rows = []
    for metric in mc_metrics:
        for (base_m, cmp_m) in [('linear_ridge','krr'), ('linear_ridge','pls'), ('krr','kernel_pls'), ('pls','kernel_pls')]:
            a_vals = mc_df[mc_df.model == base_m][metric].dropna().values
            b_vals = mc_df[mc_df.model == cmp_m][metric].dropna().values
            p = wilcoxon_p(b_vals, a_vals)
            delta = np.mean(b_vals) - np.mean(a_vals)
            summary_rows.append({
                'Metric': METRIC_LABELS.get(metric, metric),
                'Comparison': f'{MODEL_LABELS[base_m]} → {MODEL_LABELS[cmp_m]}',
                'Δ (mean)': f'{delta:+.4f}',
                'p (Wilcoxon)': f'{p:.4f}' if not np.isnan(p) else 'n/a',
                'Sig.': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns',
            })
    stat_df = pd.DataFrame(summary_rows)

    # ── HTML assembly ─────────────────────────────────────────────────────────
    # Shared rules come from report.render; .key-finding is aliased there onto the
    # canonical "finding" callout. Kept here: the full-width table and this report's
    # own significance-star colour.
    style = stylesheet("""
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th { background: #4e79a7; color: #fff; padding: 7px 10px; text-align: left; }
td { padding: 6px 10px; border-bottom: 1px solid #eee; }
tr:nth-child(even) td { background: #f5f7fa; }
.key-finding.good { border-color: #59a14f; background: #edf7eb; }
.key-finding.warn { border-color: #e15759; background: #fdeaea; }
.sig-star { color: #e15759; font-weight: bold; }
.fig-grid > div { flex: 1 1 500px; }
""")

    def df_to_html(df, highlight_col=None):
        html = '<table><tr>'
        for c in df.columns:
            html += f'<th>{c}</th>'
        html += '</tr>'
        for _, row in df.iterrows():
            html += '<tr>'
            for c in df.columns:
                v = row[c]
                cls = ' class="sig-star"' if highlight_col == c and str(v).endswith(('*','**','***')) else ''
                html += f'<td{cls}>{v}</td>'
            html += '</tr>'
        html += '</table>'
        return html

    # ── key findings ─────────────────────────────────────────────────────────
    # does KRR beat Ridge on cosine?
    if has_cosine:
        krr_cos = mc_df[mc_df.model=='krr']['test_cosine'].mean()
        ridge_cos = mc_df[mc_df.model=='linear_ridge']['test_cosine'].mean()
        pls_cos   = mc_df[mc_df.model=='pls']['test_cosine'].mean()
        kpls_cos  = mc_df[mc_df.model=='kernel_pls']['test_cosine'].mean()
    else:
        krr_cos = mc_df[mc_df.model=='krr']['word_bal_acc'].mean()
        ridge_cos = mc_df[mc_df.model=='linear_ridge']['word_bal_acc'].mean()
        pls_cos   = mc_df[mc_df.model=='pls']['word_bal_acc'].mean()
        kpls_cos  = mc_df[mc_df.model=='kernel_pls']['word_bal_acc'].mean()

    krr_r2    = mc_df[mc_df.model=='krr']['delta_r2'].mean()
    ridge_r2  = mc_df[mc_df.model=='linear_ridge']['delta_r2'].mean()
    pls_r2    = mc_df[mc_df.model=='pls']['delta_r2'].mean()

    best_nc_cos_mode  = best_nc_df['best_nc_cosine'].mode()[0] if not best_nc_df.empty else '?'
    best_nc_word_mode = best_nc_df['best_nc_word'].mode()[0]   if not best_nc_df.empty else '?'
    best_nc_cat_mode  = best_nc_df['best_nc_cat'].mode()[0]    if not best_nc_df.empty else '?'
    best_nc_r2_mode   = best_nc_df['best_nc_r2'].mode()[0]     if not best_nc_df.empty else '?'

    html_parts = [f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Model Analysis Summary</title>
{style}</head><body>
<h1>Model Analysis Summary</h1>
<p style="color:#666">Patients: {', '.join(patients)} &nbsp;|&nbsp; Embeddings: {', '.join(embeddings)}</p>
"""]

    # ── Key findings box ─────────────────────────────────────────────────────
    html_parts.append('<div class="box"><h2 style="margin-top:0">Key Findings</h2>')

    # nonlinearity
    nl_effect = krr_r2 - ridge_r2
    nl_cls = 'good' if nl_effect > 0.02 else 'warn'
    html_parts.append(f'<div class="key-finding {nl_cls}">📊 <b>Nonlinearity (Kernel):</b> '
                      f'KRR beats Ridge by ΔR²={nl_effect:+.3f} on average. '
                      f'{"Kernel substantially helps R²." if nl_effect > 0.02 else "Minimal R² gain from kernel."}</div>')

    # PLS vs ridge - retrieval
    pls_effect_ret = pls_cos - ridge_cos
    pls_cls = 'good' if pls_effect_ret > 0 else 'warn'
    metric_name_ret = 'cosine similarity' if has_cosine else 'word acc'
    html_parts.append(f'<div class="key-finding {pls_cls}">🎯 <b>PLS vs Linear Ridge ({metric_name_ret}):</b> '
                      f'Δ={pls_effect_ret:+.3f}. '
                      f'{"PLS clearly improves retrieval." if pls_effect_ret > 0.01 else "Marginal PLS retrieval improvement."}</div>')

    # best n_components
    if not best_nc_df.empty:
        html_parts.append(f'<div class="key-finding good">🔧 <b>Optimal n_components (PLS, gap &lt; 0.10):</b> '
                          f'Cosine: <b>n={best_nc_cos_mode}</b> &nbsp;|&nbsp; '
                          f'Word acc: <b>n={best_nc_word_mode}</b> &nbsp;|&nbsp; '
                          f'Cat acc: <b>n={best_nc_cat_mode}</b> &nbsp;|&nbsp; '
                          f'R²: <b>n={best_nc_r2_mode}</b></div>')

    html_parts.append('</div>')

    # ── Section 1: overall model comparison ─────────────────────────────────
    html_parts.append('<div class="box"><h2>1. Model Comparison (all patients × embeddings)</h2>')
    html_parts.append('<div class="fig-grid">')
    for metric in mc_metrics:
        html_parts.append(f'<div>{figs_model[metric]}</div>')
    html_parts.append('</div></div>')

    # ── Section 2: per-patient ───────────────────────────────────────────────
    html_parts.append('<div class="box"><h2>2. Per-Patient Breakdown</h2>')
    for metric in mc_metrics:
        html_parts.append(f'<h3>{METRIC_LABELS.get(metric, metric)}</h3>')
        html_parts.append('<div class="fig-grid">')
        for pat in patients:
            if (pat, metric) in figs_patient:
                html_parts.append(f'<div>{figs_patient[(pat, metric)]}</div>')
        html_parts.append('</div>')
    html_parts.append('</div>')

    # ── Section 3: statistical tests ─────────────────────────────────────────
    html_parts.append('<div class="box"><h2>3. Statistical Tests (Wilcoxon one-sided, across patients × embeddings)</h2>')
    html_parts.append('<h3>Effect of Nonlinearity</h3>')
    html_parts.append(df_to_html(nl_table, 'Sig.'))
    html_parts.append('<h3>Effect of PLS</h3>')
    html_parts.append(df_to_html(pls_table, 'Sig.'))
    html_parts.append('<h3>All pairwise comparisons</h3>')
    html_parts.append(df_to_html(stat_df, 'Sig.'))
    html_parts.append('</div>')

    # ── Section 4: PLS learning curves ───────────────────────────────────────
    if lc_curves:
        html_parts.append('<div class="box"><h2>4. PLS n_components Sweep (learning curves)</h2>')
        html_parts.append('<p>Vertical dashed line = n=4 (your confirmed optimum). '
                          'Shading = ±1 SE across epochs.</p>')
        for emb in lc_embeddings:
            html_parts.append(f'<h3>Embedding: {emb}</h3>')
            html_parts.append('<div class="fig-grid">')
            for metric_name in ['R²', 'Cosine Sim.', 'Word Acc. (test)', 'Cat. Acc. (test)']:
                key = (emb, metric_name)
                if key in figs_lc:
                    html_parts.append(f'<div>{figs_lc[key]}</div>')
            html_parts.append('</div>')
        # best n summary table
        if not best_nc_df.empty:
            html_parts.append('<h3>Best n_components per patient × embedding (gap &lt; 0.10)</h3>')
            html_parts.append(df_to_html(best_nc_df))
        html_parts.append('</div>')

    html_parts.append('</body></html>')
    return '\n'.join(html_parts)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', '--results_dir', default=None,
                    help='Directory containing model_comparison_*.csv and pls_learning_curve_*.csv')
    ap.add_argument('--out', default=None,
                    help='Output HTML path')
    args = ap.parse_args()

    results_dir = get_out_dir(args.results_dir)
    out_path = args.out or os.path.join(results_dir, 'report_model_selection.html')

    mc_df = load_model_comparison(results_dir)
    if mc_df is None:
        print('No model_comparison_*.csv files found.')
        return
    lc_df = load_pls_learning(results_dir)
    if lc_df is None:
        print('No pls_learning_curve_*.csv files found — skipping learning curves.')

    html = build_html(mc_df, lc_df, results_dir)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
