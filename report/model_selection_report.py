# -*- coding: utf-8 -*-
"""
tests.model_selection_report — Summarise model comparison and n_components sweep results.

Reads model_comparison_*.csv and pls_learning_curve_*.csv from a results directory
and answers: which model (Ridge / KRR / PLS / Kernel PLS) should you use,
and what is the optimal n_components for PLS?

Outputs a standalone HTML with bar charts per metric, per-patient breakdowns,
Wilcoxon significance tables, and PLS learning curves.

Usage:
    python -m tests.model_selection_report --results_dir <path> --out <path>
"""
import argparse, glob, os, json, warnings
import pandas as pd
import numpy as np
from scipy import stats
from tests.helpers._phoneme_semantic_helpers import get_out_dir

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

# ── SVG helpers ───────────────────────────────────────────────────────────────
def _svg_bar_chart(series_dict, title, ylabel, width=520, height=280,
                   colour_map=None, ymin=None, ymax=None, baseline=None,
                   baseline_label='chance'):
    """
    series_dict: {label: value}  or  {label: (mean, se)}
    Returns SVG string.
    """
    labels = list(series_dict.keys())
    raw    = list(series_dict.values())
    means  = [r[0] if isinstance(r, tuple) else r for r in raw]
    ses    = [r[1] if isinstance(r, tuple) else 0.0 for r in raw]

    pad_l, pad_r, pad_t, pad_b = 70, 20, 40, 55
    w = width - pad_l - pad_r
    h = height - pad_t - pad_b

    all_vals = means + [m - s for m, s in zip(means, ses)] + [m + s for m, s in zip(means, ses)]
    if baseline is not None:
        all_vals.append(baseline)
    vmin = ymin if ymin is not None else min(all_vals) * 1.05 if min(all_vals) < 0 else 0
    vmax = ymax if ymax is not None else max(all_vals) * 1.1
    vrange = vmax - vmin or 1

    def fy(v):
        return pad_t + h - h * (v - vmin) / vrange

    if not labels:
        return '<svg width="1" height="1"></svg>'
    bar_w  = w / len(labels) * 0.55
    bar_gap = w / len(labels)

    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
           f'style="font-family:sans-serif">']
    # title
    svg.append(f'<text x="{width/2}" y="18" text-anchor="middle" '
               f'font-size="13" font-weight="bold">{title}</text>')
    # axes
    y0 = fy(max(0, vmin))
    svg.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+h}" '
               f'stroke="#888" stroke-width="1.5"/>')
    svg.append(f'<line x1="{pad_l}" y1="{y0}" x2="{pad_l+w}" y2="{y0}" '
               f'stroke="#888" stroke-width="1"/>')
    # y-axis ticks
    n_ticks = 5
    for i in range(n_ticks + 1):
        tv = vmin + vrange * i / n_ticks
        ty = fy(tv)
        svg.append(f'<line x1="{pad_l-4}" y1="{ty}" x2="{pad_l}" y2="{ty}" stroke="#888"/>')
        svg.append(f'<text x="{pad_l-7}" y="{ty+4}" text-anchor="end" font-size="10">'
                   f'{tv:.2f}</text>')
    # baseline
    if baseline is not None:
        by = fy(baseline)
        svg.append(f'<line x1="{pad_l}" y1="{by}" x2="{pad_l+w}" y2="{by}" '
                   f'stroke="#aaa" stroke-width="1.2" stroke-dasharray="5,3"/>')
        svg.append(f'<text x="{pad_l+w-2}" y="{by-3}" text-anchor="end" '
                   f'font-size="9" fill="#999">{baseline_label}</text>')
    # bars
    for i, (lbl, mean, se) in enumerate(zip(labels, means, ses)):
        cx = pad_l + bar_gap * (i + 0.5)
        x  = cx - bar_w / 2
        bar_top = fy(mean)
        bar_bot = fy(max(0, vmin))
        bh = abs(bar_bot - bar_top)
        col = colour_map.get(lbl, '#4e79a7') if colour_map else '#4e79a7'
        if mean < 0:
            svg.append(f'<rect x="{x:.1f}" y="{bar_top:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" '
                       f'fill="{col}" opacity="0.85" rx="2"/>')
        else:
            svg.append(f'<rect x="{x:.1f}" y="{bar_top:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" '
                       f'fill="{col}" opacity="0.85" rx="2"/>')
        # error bar
        if se > 0:
            ey_top = fy(mean + se)
            ey_bot = fy(mean - se)
            svg.append(f'<line x1="{cx}" y1="{ey_top:.1f}" x2="{cx}" y2="{ey_bot:.1f}" '
                       f'stroke="#333" stroke-width="1.3"/>')
            svg.append(f'<line x1="{cx-4}" y1="{ey_top:.1f}" x2="{cx+4}" y2="{ey_top:.1f}" '
                       f'stroke="#333" stroke-width="1.3"/>')
            svg.append(f'<line x1="{cx-4}" y1="{ey_bot:.1f}" x2="{cx+4}" y2="{ey_bot:.1f}" '
                       f'stroke="#333" stroke-width="1.3"/>')
        # value label
        svg.append(f'<text x="{cx:.1f}" y="{bar_top - 4:.1f}" text-anchor="middle" '
                   f'font-size="9">{mean:.3f}</text>')
        # x label
        short = lbl.replace('linear_ridge','Ridge').replace('kernel_pls','KPLS')\
                   .replace('krr','KRR').replace('pls','PLS')
        svg.append(f'<text x="{cx:.1f}" y="{pad_t+h+14}" text-anchor="middle" '
                   f'font-size="10">{short}</text>')
    # y-label
    svg.append(f'<text transform="rotate(-90)" x="-{(pad_t+h/2):.0f}" y="15" '
               f'text-anchor="middle" font-size="11">{ylabel}</text>')
    svg.append('</svg>')
    return '\n'.join(svg)


def _svg_line_chart(curves_dict, title, xlabel, ylabel,
                    width=540, height=280, se_dict=None,
                    baseline=None, baseline_label='chance',
                    highlight_x=None, highlight_label=''):
    """
    curves_dict: {label: {x: y}}
    se_dict:     {label: {x: se}}   (optional ±1 SE shading)
    """
    all_labels = list(curves_dict.keys())
    all_xs  = sorted(set(x for d in curves_dict.values() for x in d))
    all_ys  = [y for d in curves_dict.values() for y in d.values()]
    if baseline is not None:
        all_ys.append(baseline)
    if se_dict:
        for d, sd in zip(curves_dict.values(), se_dict.values()):
            for x, y in d.items():
                se = sd.get(x, 0)
                all_ys += [y - se, y + se]

    pad_l, pad_r, pad_t, pad_b = 65, 20, 40, 55
    w = width - pad_l - pad_r
    h = height - pad_t - pad_b

    vmin = min(all_ys)
    vmax = max(all_ys)
    vrange = vmax - vmin or 1

    xmin, xmax = all_xs[0], all_xs[-1]
    xrange_ = xmax - xmin or 1

    def fx(v):
        return pad_l + w * (v - xmin) / xrange_
    def fy(v):
        return pad_t + h - h * (v - vmin) / vrange

    CMAP = ['#4e79a7','#f28e2b','#59a14f','#e15759','#76b7b2','#edc948']

    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
           f'style="font-family:sans-serif">']
    svg.append(f'<text x="{width/2}" y="18" text-anchor="middle" '
               f'font-size="13" font-weight="bold">{title}</text>')
    # axes
    svg.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+h}" stroke="#888" stroke-width="1.5"/>')
    svg.append(f'<line x1="{pad_l}" y1="{pad_t+h}" x2="{pad_l+w}" y2="{pad_t+h}" stroke="#888" stroke-width="1.5"/>')
    # y ticks
    for i in range(6):
        tv = vmin + vrange * i / 5
        ty = fy(tv)
        svg.append(f'<line x1="{pad_l-4}" y1="{ty}" x2="{pad_l}" y2="{ty}" stroke="#888"/>')
        svg.append(f'<text x="{pad_l-7}" y="{ty+4}" text-anchor="end" font-size="10">{tv:.2f}</text>')
    # x ticks
    for x in all_xs:
        tx = fx(x)
        svg.append(f'<line x1="{tx}" y1="{pad_t+h}" x2="{tx}" y2="{pad_t+h+4}" stroke="#888"/>')
        svg.append(f'<text x="{tx}" y="{pad_t+h+15}" text-anchor="middle" font-size="10">{x}</text>')
    # baseline
    if baseline is not None:
        by = fy(baseline)
        svg.append(f'<line x1="{pad_l}" y1="{by}" x2="{pad_l+w}" y2="{by}" '
                   f'stroke="#aaa" stroke-width="1" stroke-dasharray="5,3"/>')
        svg.append(f'<text x="{pad_l+w}" y="{by-3}" text-anchor="end" font-size="9" fill="#999">{baseline_label}</text>')
    # highlight x
    if highlight_x is not None:
        hx = fx(highlight_x)
        svg.append(f'<line x1="{hx}" y1="{pad_t}" x2="{hx}" y2="{pad_t+h}" '
                   f'stroke="#e15759" stroke-width="1.5" stroke-dasharray="5,3" opacity="0.7"/>')
        svg.append(f'<text x="{hx+3}" y="{pad_t+12}" font-size="9" fill="#e15759">{highlight_label}</text>')
    # shading + lines
    for idx, (lbl, curve) in enumerate(curves_dict.items()):
        col = CMAP[idx % len(CMAP)]
        xs = sorted(curve.keys())
        if se_dict and lbl in se_dict:
            sd = se_dict[lbl]
            # upper path then lower reversed
            upper = [(fx(x), fy(curve[x] + sd.get(x, 0))) for x in xs]
            lower = [(fx(x), fy(curve[x] - sd.get(x, 0))) for x in reversed(xs)]
            pts = upper + lower
            path = ' '.join(f'{"M" if i==0 else "L"}{px:.1f},{py:.1f}' for i,(px,py) in enumerate(pts))
            svg.append(f'<path d="{path} Z" fill="{col}" opacity="0.15"/>')
        pts = ' '.join(f'{"M" if i==0 else "L"}{fx(x):.1f},{fy(curve[x]):.1f}'
                       for i, x in enumerate(xs))
        svg.append(f'<path d="{pts}" fill="none" stroke="{col}" stroke-width="2.2"/>')
        # dots
        for x in xs:
            svg.append(f'<circle cx="{fx(x):.1f}" cy="{fy(curve[x]):.1f}" r="4" fill="{col}"/>')
        # legend entry
        leg_x = pad_l + w + 5
        leg_y = pad_t + 18 + idx * 18
        svg.append(f'<line x1="{leg_x}" y1="{leg_y}" x2="{leg_x+18}" y2="{leg_y}" '
                   f'stroke="{col}" stroke-width="2.5"/>')
        svg.append(f'<text x="{leg_x+22}" y="{leg_y+4}" font-size="10">{lbl}</text>')
    # axis labels
    svg.append(f'<text x="{pad_l+w/2}" y="{height-5}" text-anchor="middle" font-size="11">{xlabel}</text>')
    svg.append(f'<text transform="rotate(-90)" x="-{(pad_t+h/2):.0f}" y="15" '
               f'text-anchor="middle" font-size="11">{ylabel}</text>')
    svg.append('</svg>')
    return '\n'.join(svg)


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
        figs_model[metric] = _svg_bar_chart(
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
            figs_patient[(pat, metric)] = _svg_bar_chart(
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
                figs_lc[(emb, metric_name)] = _svg_line_chart(
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
                figs_lc[(emb, metric_name)] = _svg_line_chart(
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
    style = """
    body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; color: #222; background: #fafafa; }
    h1 { border-bottom: 3px solid #4e79a7; padding-bottom: 8px; }
    h2 { margin-top: 40px; color: #333; border-left: 5px solid #4e79a7; padding-left: 10px; }
    h3 { color: #555; margin-top: 24px; }
    .box { background: #fff; border-radius: 8px; box-shadow: 0 1px 6px #0001; padding: 20px; margin-bottom: 24px; }
    .key-finding { background: #eaf3fb; border-left: 4px solid #4e79a7; padding: 10px 16px; border-radius: 4px; margin-bottom: 10px; }
    .key-finding.good { border-color: #59a14f; background: #edf7eb; }
    .key-finding.warn { border-color: #e15759; background: #fdeaea; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th { background: #4e79a7; color: #fff; padding: 7px 10px; text-align: left; }
    td { padding: 6px 10px; border-bottom: 1px solid #eee; }
    tr:nth-child(even) td { background: #f5f7fa; }
    .sig-star { color: #e15759; font-weight: bold; }
    .fig-grid { display: flex; flex-wrap: wrap; gap: 20px; }
    .fig-grid > div { flex: 1 1 500px; }
    """

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
<style>{style}</style></head><body>
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
