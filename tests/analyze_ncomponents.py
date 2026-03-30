"""
Focused analysis of why different metrics suggest different optimal n_components in PLS.
Produces a standalone HTML report.

Usage:
    python -m tests.analyze_ncomponents --results_dir <path> --out <path>
"""
import argparse, glob, os, warnings
import pandas as pd
import numpy as np
from scipy import stats

warnings.filterwarnings('ignore')

METRICS = ['test_r2', 'test_cosine', 'word_bal_acc', 'cat_bal_acc']
METRIC_LABELS = {
    'test_r2':       'R² (test)',
    'test_cosine':   'Cosine Sim. (test)',
    'word_bal_acc':  'Word Acc. (test)',
    'cat_bal_acc':   'Cat Acc. (test)',
}
COLOURS = {
    'test_r2':      '#4e79a7',
    'test_cosine':  '#f28e2b',
    'word_bal_acc': '#59a14f',
    'cat_bal_acc':  '#e15759',
    'train':        '#bbb',
    'gap':          '#9b59b6',
}
PATIENTS = ['AA', 'RB', 'VB']
PAT_COLOURS = {'AA': '#4e79a7', 'RB': '#f28e2b', 'VB': '#59a14f'}


# ── SVG helpers ───────────────────────────────────────────────────────────────

def _line_chart(curves, title, xlabel, ylabel, width=560, height=300,
                se_dict=None, baselines=None, highlight_x=None,
                vmin=None, vmax=None, legend_inside=False):
    """
    curves:   {label: {x: y}}
    se_dict:  {label: {x: se}}
    baselines: [(y_val, colour, label), ...]
    """
    pad_l, pad_r, pad_t, pad_b = 65, 160 if not legend_inside else 20, 38, 48
    w = width - pad_l - pad_r
    h = height - pad_t - pad_b

    all_xs = sorted(set(x for d in curves.values() for x in d))
    all_ys = [y for d in curves.values() for y in d.values()]
    if se_dict:
        for lbl, d in curves.items():
            sd = se_dict.get(lbl, {})
            for x, y in d.items():
                se = sd.get(x, 0)
                all_ys += [y - se, y + se]
    if baselines:
        all_ys += [b[0] for b in baselines]

    _vmin = vmin if vmin is not None else min(all_ys) - abs(min(all_ys)) * 0.05
    _vmax = vmax if vmax is not None else max(all_ys) + abs(max(all_ys)) * 0.05
    vrange = _vmax - _vmin or 1
    xmin, xmax = all_xs[0], all_xs[-1]
    xrange_ = xmax - xmin or 1

    def fx(v): return pad_l + w * (v - xmin) / xrange_
    def fy(v): return pad_t + h - h * (v - _vmin) / vrange

    CMAP = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2', '#edc948',
            '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
           f'style="font-family:sans-serif;font-size:11px">']
    svg.append(f'<text x="{width/2:.0f}" y="20" text-anchor="middle" '
               f'font-size="13" font-weight="bold">{title}</text>')
    # axes
    svg.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+h}" stroke="#888" stroke-width="1.5"/>')
    svg.append(f'<line x1="{pad_l}" y1="{pad_t+h}" x2="{pad_l+w}" y2="{pad_t+h}" stroke="#888" stroke-width="1.5"/>')
    # zero line
    if _vmin < 0 < _vmax:
        y0 = fy(0)
        svg.append(f'<line x1="{pad_l}" y1="{y0:.1f}" x2="{pad_l+w}" y2="{y0:.1f}" '
                   f'stroke="#ccc" stroke-width="1" stroke-dasharray="3,3"/>')
    # y ticks
    for i in range(6):
        tv = _vmin + vrange * i / 5
        ty = fy(tv)
        svg.append(f'<line x1="{pad_l-4}" y1="{ty:.1f}" x2="{pad_l}" y2="{ty:.1f}" stroke="#888"/>')
        svg.append(f'<text x="{pad_l-7}" y="{ty+4:.1f}" text-anchor="end" font-size="10">{tv:.3f}</text>')
    # x ticks
    for x in all_xs:
        tx = fx(x)
        svg.append(f'<line x1="{tx:.1f}" y1="{pad_t+h}" x2="{tx:.1f}" y2="{pad_t+h+4}" stroke="#888"/>')
        svg.append(f'<text x="{tx:.1f}" y="{pad_t+h+16}" text-anchor="middle" font-size="10">{x}</text>')
    # baselines
    if baselines:
        for bval, bcol, blbl in baselines:
            by = fy(bval)
            svg.append(f'<line x1="{pad_l}" y1="{by:.1f}" x2="{pad_l+w}" y2="{by:.1f}" '
                       f'stroke="{bcol}" stroke-width="1.2" stroke-dasharray="4,3" opacity="0.6"/>')
            svg.append(f'<text x="{pad_l+3}" y="{by-3:.1f}" font-size="9" fill="{bcol}">{blbl}</text>')
    # highlight x
    if highlight_x is not None:
        hx = fx(highlight_x)
        svg.append(f'<line x1="{hx:.1f}" y1="{pad_t}" x2="{hx:.1f}" y2="{pad_t+h}" '
                   f'stroke="#e15759" stroke-width="1.5" stroke-dasharray="5,3" opacity="0.7"/>')
    # shading + curves
    for idx, (lbl, curve) in enumerate(curves.items()):
        col = CMAP[idx % len(CMAP)]
        xs = sorted(curve.keys())
        if se_dict and lbl in se_dict:
            sd = se_dict[lbl]
            upper = [(fx(x), fy(curve[x] + sd.get(x, 0))) for x in xs]
            lower = [(fx(x), fy(curve[x] - sd.get(x, 0))) for x in reversed(xs)]
            path = ' '.join(f'{"M" if i==0 else "L"}{px:.1f},{py:.1f}'
                            for i, (px, py) in enumerate(upper + lower))
            svg.append(f'<path d="{path} Z" fill="{col}" opacity="0.12"/>')
        pts = ' '.join(f'{"M" if i==0 else "L"}{fx(x):.1f},{fy(curve[x]):.1f}'
                       for i, x in enumerate(xs))
        dashes = '6,3' if 'train' in lbl.lower() else 'none'
        svg.append(f'<path d="{pts}" fill="none" stroke="{col}" stroke-width="2.2" '
                   f'stroke-dasharray="{dashes}"/>')
        for x in xs:
            if 'train' not in lbl.lower():
                svg.append(f'<circle cx="{fx(x):.1f}" cy="{fy(curve[x]):.1f}" r="3.5" fill="{col}"/>')
        # legend
        lx = pad_l + w + 8
        ly = pad_t + 14 + idx * 18
        svg.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+18}" y2="{ly}" '
                   f'stroke="{col}" stroke-width="2.2" stroke-dasharray="{dashes}"/>')
        svg.append(f'<text x="{lx+22}" y="{ly+4}" font-size="10">{lbl}</text>')
    # axis labels
    svg.append(f'<text x="{pad_l+w/2:.0f}" y="{height-4}" text-anchor="middle" font-size="11">{xlabel}</text>')
    svg.append(f'<text transform="rotate(-90)" x="-{(pad_t+h/2):.0f}" y="14" '
               f'text-anchor="middle" font-size="11">{ylabel}</text>')
    svg.append('</svg>')
    return '\n'.join(svg)


def _heatmap(matrix_df, title, width=540, height=240):
    """matrix_df: rows=metrics, cols=n_components; values normalised 0-1 per row."""
    rows = list(matrix_df.index)
    cols = list(matrix_df.columns)
    pad_l, pad_r, pad_t, pad_b = 130, 20, 38, 40
    w = width - pad_l - pad_r
    h = height - pad_t - pad_b
    cw = w / len(cols)
    rh = h / len(rows)

    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
           f'style="font-family:sans-serif">']
    svg.append(f'<text x="{width/2:.0f}" y="22" text-anchor="middle" font-size="13" font-weight="bold">{title}</text>')

    for ri, row in enumerate(rows):
        row_vals = matrix_df.loc[row]
        rmin, rmax = row_vals.min(), row_vals.max()
        rrange = rmax - rmin or 1
        best_col = row_vals.idxmax()
        for ci, col in enumerate(cols):
            norm = (row_vals[col] - rmin) / rrange  # 0-1
            # blue-white-red diverging: low=white, high=blue
            r_c = int(255 - norm * 100)
            g_c = int(255 - norm * 120)
            b_c = 255
            fill = f'rgb({r_c},{g_c},{b_c})'
            x = pad_l + ci * cw
            y = pad_t + ri * rh
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw:.1f}" height="{rh:.1f}" '
                       f'fill="{fill}" stroke="#fff" stroke-width="0.5"/>')
            # star for best
            star = '★' if col == best_col else ''
            svg.append(f'<text x="{x+cw/2:.1f}" y="{y+rh/2+4:.1f}" text-anchor="middle" '
                       f'font-size="10" fill="#333">{row_vals[col]:.3f}{star}</text>')
        # row label
        svg.append(f'<text x="{pad_l-5}" y="{pad_t + ri*rh + rh/2 + 4:.1f}" '
                   f'text-anchor="end" font-size="11">{row}</text>')
    # col headers
    for ci, col in enumerate(cols):
        x = pad_l + ci * cw + cw / 2
        svg.append(f'<text x="{x:.1f}" y="{pad_t-6}" text-anchor="middle" font-size="11">n={col}</text>')

    svg.append('</svg>')
    return '\n'.join(svg)


# ── Core analysis ─────────────────────────────────────────────────────────────

def run(results_dir, out_path):
    lc_dfs = [pd.read_csv(f) for f in glob.glob(os.path.join(results_dir, 'pls_learning_curve_*.csv'))]
    if not lc_dfs:
        print('No pls_learning_curve_*.csv found.')
        return
    lc = pd.concat(lc_dfs, ignore_index=True)
    pls = lc[lc.model == 'pls'].copy()

    # restrict to n≤20 (n=25,30 only have 1 patient-embedding combo — sampling artefact)
    pls_valid = pls[pls.n_components <= 20]
    nc_vals = sorted(pls_valid['n_components'].unique())

    # ── aggregate across all patient×embedding combos ─────────────────────────
    grp = pls_valid.groupby('n_components')
    agg = grp.agg(
        test_r2_m=('test_r2','mean'),     test_r2_se=('test_r2','sem'),
        train_r2_m=('train_r2','mean'),   train_r2_se=('train_r2','sem'),
        test_cos_m=('test_cosine','mean'),test_cos_se=('test_cosine','sem'),
        train_cos_m=('train_cosine','mean'),train_cos_se=('train_cosine','sem'),
        word_m=('word_bal_acc','mean'),   word_se=('word_bal_acc','sem'),
        cat_m=('cat_bal_acc','mean'),     cat_se=('cat_bal_acc','sem'),
    ).reset_index()
    agg = agg.set_index('n_components')

    # overfitting gap
    agg['r2_gap']  = agg['train_r2_m']  - agg['test_r2_m']
    agg['cos_gap'] = agg['train_cos_m'] - agg['test_cos_m']

    # ── per-patient curves ────────────────────────────────────────────────────
    pat_agg = {}
    for pat in PATIENTS:
        sub = pls_valid[pls_valid.patient == pat]
        if sub.empty:
            continue
        g = sub.groupby('n_components').agg(
            test_r2_m=('test_r2','mean'),
            test_cos_m=('test_cosine','mean'),
            word_m=('word_bal_acc','mean'),
            cat_m=('cat_bal_acc','mean'),
            train_r2_m=('train_r2','mean'),
            train_cos_m=('train_cosine','mean'),
        ).reset_index().set_index('n_components')
        pat_agg[pat] = g

    # ── heatmap: normalised score per metric ──────────────────────────────────
    hm_data = pd.DataFrame({
        'R²':        agg['test_r2_m'],
        'Cosine':    agg['test_cos_m'],
        'Word Acc':  agg['word_m'],
        'Cat Acc':   agg['cat_m'],
    }).T   # rows=metrics, cols=n_components

    # ── best n per metric ─────────────────────────────────────────────────────
    best_n = {
        'R²':       int(agg['test_r2_m'].idxmax()),
        'Cosine':   int(agg['test_cos_m'].idxmax()),
        'Word Acc': int(agg['word_m'].idxmax()),
        'Cat Acc':  int(agg['cat_m'].idxmax()),
    }

    # ── figures ───────────────────────────────────────────────────────────────
    # Fig 1: all 4 metrics on the same axes (normalised to [0,1] range)
    norm_curves = {}
    norm_ses = {}
    for mkey, mlbl, col, sekey in [
        ('test_r2_m', 'R²', COLOURS['test_r2'], 'test_r2_se'),
        ('test_cos_m', 'Cosine', COLOURS['test_cosine'], 'test_cos_se'),
        ('word_m', 'Word Acc', COLOURS['word_bal_acc'], 'word_se'),
        ('cat_m', 'Cat Acc', COLOURS['cat_bal_acc'], 'cat_se'),
    ]:
        vals = agg[mkey]
        mn, mx = vals.min(), vals.max()
        rng = mx - mn or 1
        norm_curves[mlbl] = {n: (v - mn)/rng for n, v in vals.items()}
        se_vals = agg[sekey]
        norm_ses[mlbl] = {n: se/rng for n, se in se_vals.items()}

    fig_norm = _line_chart(
        norm_curves, se_dict=norm_ses,
        title='All metrics normalised to [0–1] range (mean across patients × embeddings)',
        xlabel='n_components', ylabel='Normalised score',
        highlight_x=None, vmin=0, vmax=1,
    )

    # Fig 2: raw test metrics
    raw_curves = {
        'R²':       {n: agg.loc[n,'test_r2_m']  for n in nc_vals},
        'Cosine':   {n: agg.loc[n,'test_cos_m'] for n in nc_vals},
    }
    raw_ses = {
        'R²':       {n: agg.loc[n,'test_r2_se']  for n in nc_vals},
        'Cosine':   {n: agg.loc[n,'test_cos_se'] for n in nc_vals},
    }
    fig_r2cos = _line_chart(raw_curves, se_dict=raw_ses,
        title='R² and Cosine vs n_components (test)',
        xlabel='n_components', ylabel='Score',
        highlight_x=None,
    )

    # Fig 3: accuracy metrics
    acc_curves = {
        'Word Acc':  {n: agg.loc[n,'word_m'] for n in nc_vals},
        'Cat Acc':   {n: agg.loc[n,'cat_m']  for n in nc_vals},
    }
    acc_ses = {
        'Word Acc':  {n: agg.loc[n,'word_se'] for n in nc_vals},
        'Cat Acc':   {n: agg.loc[n,'cat_se']  for n in nc_vals},
    }
    fig_acc = _line_chart(acc_curves, se_dict=acc_ses,
        title='Word & Cat Accuracy vs n_components (test)',
        xlabel='n_components', ylabel='Balanced Accuracy',
        highlight_x=None,
    )

    # Fig 4: overfitting gap
    gap_curves = {
        'R² gap (train−test)':  {n: agg.loc[n,'r2_gap']  for n in nc_vals},
        'Cos gap (train−test)': {n: agg.loc[n,'cos_gap'] for n in nc_vals},
    }
    fig_gap = _line_chart(gap_curves,
        title='Overfitting gap: train − test (R² and Cosine)',
        xlabel='n_components', ylabel='Gap magnitude',
        vmin=0,
        baselines=[(0.10, '#888', 'gap=0.10 threshold')],
    )

    # Fig 5: per-patient word acc vs n
    word_pat_curves = {pat: {n: pat_agg[pat].loc[n,'word_m']
                              for n in nc_vals if n in pat_agg[pat].index}
                       for pat in pat_agg}
    fig_word_pat = _line_chart(word_pat_curves,
        title='Word Accuracy per patient vs n_components',
        xlabel='n_components', ylabel='Balanced Accuracy',
    )

    # Fig 6: per-patient cat acc vs n
    cat_pat_curves = {pat: {n: pat_agg[pat].loc[n,'cat_m']
                             for n in nc_vals if n in pat_agg[pat].index}
                      for pat in pat_agg}
    fig_cat_pat = _line_chart(cat_pat_curves,
        title='Cat Accuracy per patient vs n_components',
        xlabel='n_components', ylabel='Balanced Accuracy',
    )

    # Fig 7: train vs test for R² and cosine
    tr_test_curves = {
        'R² test':      {n: agg.loc[n,'test_r2_m']  for n in nc_vals},
        'R² train':     {n: agg.loc[n,'train_r2_m'] for n in nc_vals},
        'Cos test':     {n: agg.loc[n,'test_cos_m'] for n in nc_vals},
        'Cos train':    {n: agg.loc[n,'train_cos_m'] for n in nc_vals},
    }
    fig_trtest = _line_chart(tr_test_curves,
        title='Train vs Test: R² and Cosine',
        xlabel='n_components', ylabel='Score',
    )

    # ── heatmap ───────────────────────────────────────────────────────────────
    fig_hm = _heatmap(hm_data[nc_vals],
                      title='Score heatmap (★ = best per metric, colour = normalised rank)')

    # ── summary table ─────────────────────────────────────────────────────────
    summary_rows = []
    for nc in nc_vals:
        summary_rows.append({
            'n_components': nc,
            'R² (test)':     f"{agg.loc[nc,'test_r2_m']:.4f}",
            'Cosine (test)': f"{agg.loc[nc,'test_cos_m']:.4f}",
            'Word Acc':      f"{agg.loc[nc,'word_m']:.4f}",
            'Cat Acc':       f"{agg.loc[nc,'cat_m']:.4f}",
            'R² gap':        f"{agg.loc[nc,'r2_gap']:.4f}",
            'Cos gap':       f"{agg.loc[nc,'cos_gap']:.4f}",
        })
    summary_df = pd.DataFrame(summary_rows)

    # ── HTML ──────────────────────────────────────────────────────────────────
    style = """
    body { font-family: system-ui,sans-serif; max-width:1200px; margin:0 auto; padding:20px; background:#fafafa; color:#222; }
    h1 { border-bottom:3px solid #4e79a7; padding-bottom:8px; }
    h2 { color:#333; border-left:5px solid #4e79a7; padding-left:10px; margin-top:36px; }
    h3 { color:#555; margin-top:20px; }
    .box { background:#fff; border-radius:8px; box-shadow:0 1px 6px #0001; padding:20px; margin-bottom:24px; }
    .insight { background:#eaf3fb; border-left:4px solid #4e79a7; padding:10px 16px; border-radius:4px; margin:10px 0; }
    .insight.warn { border-color:#e15759; background:#fdeaea; }
    .insight.good { border-color:#59a14f; background:#edf7eb; }
    .fig-row { display:flex; flex-wrap:wrap; gap:20px; align-items:flex-start; }
    table { border-collapse:collapse; width:100%; font-size:13px; }
    th { background:#4e79a7; color:#fff; padding:7px 10px; text-align:left; }
    td { padding:6px 10px; border-bottom:1px solid #eee; }
    tr:nth-child(even) td { background:#f5f7fa; }
    .best { font-weight:bold; color:#1a7a2e; }
    """

    def df_html(df, best_cols=None):
        h = '<table><tr>' + ''.join(f'<th>{c}</th>' for c in df.columns) + '</tr>'
        for _, row in df.iterrows():
            h += '<tr>'
            for c in df.columns:
                cls = ''
                if best_cols and c in best_cols and str(row[c]) == str(best_cols[c]):
                    cls = ' class="best"'
                h += f'<td{cls}>{row[c]}</td>'
            h += '</tr>'
        return h + '</table>'

    # mark best cells
    best_vals = {
        'R² (test)':     summary_df['R² (test)'].max(),
        'Cosine (test)': summary_df['Cosine (test)'].max(),
        'Word Acc':      summary_df['Word Acc'].max(),
        'Cat Acc':       summary_df['Cat Acc'].max(),
    }
    def row_highlight(df):
        h = '<table><tr>' + ''.join(f'<th>{c}</th>' for c in df.columns) + '</tr>'
        for _, row in df.iterrows():
            h += '<tr>'
            for c in df.columns:
                is_best = (c in best_vals and float(row[c]) == float(best_vals[c]))
                cls = ' class="best"' if is_best else ''
                h += f'<td{cls}>{row[c]}</td>'
            h += '</tr>'
        return h + '</table>'

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>n_components Analysis</title><style>{style}</style></head><body>
<h1>Why do different metrics give different optimal n_components?</h1>
<p style="color:#666">PLS regression · 3 patients (AA, RB, VB) · 3 embeddings (GloVe, SimCLR, Word2Vec) · n≤20 only
(n=25,30 excluded — only 1 patient-embedding combo, unreliable average)</p>

<div class="box">
<h2 style="margin-top:0">Summary: Best n_components per metric</h2>
<div class="insight warn">⚠️ <b>n=25/30 data excluded</b> — only AA/GloVe contributed to those points, so the apparent "peaks" there were a sampling artefact, not a real effect.</div>

<div class="insight"><b>R²:</b> Peaks at <b>n=2</b> (least negative). R² drops monotonically as n grows — more components = more overfitting in regression magnitude.</div>
<div class="insight"><b>Cosine similarity:</b> Peaks at <b>n=4</b> (within valid range). Captures directional alignment of predictions. Degrades faster than accuracy metrics because cosine is sensitive to the overall geometry of predictions collapsing.</div>
<div class="insight"><b>Word accuracy:</b> Peaks at <b>n=20</b>. A ranking metric — only cares which word is retrieved nearest, not the magnitude of error. More components keep improving the relative ordering of predictions even as R² and cosine degrade.</div>
<div class="insight"><b>Cat accuracy:</b> Peaks at <b>n=20</b>. Same reasoning as word acc — coarser ranking over categories, accumulates signal across more components.</div>

<h3>The core tension</h3>
<p>R² and cosine measure <em>how close predictions are to true values</em> (distance/direction). Once PLS overfits, predictions become noisy and distances blow up. Accuracy only asks <em>which item is ranked closest</em> — even a noisy predictor can still rank the right word #1 more often with more components, as long as it hasn't completely collapsed. This is why the accuracy metrics keep rising while R² and cosine fall.</p>
<p><b>Practical recommendation:</b> n=4–8 is the right operating point. n=4 is optimal for cosine. n=8 gives a meaningful word/cat accuracy boost (+{(agg.loc[8,'word_m']-agg.loc[4,'word_m']):.3f} word, +{(agg.loc[8,'cat_m']-agg.loc[4,'cat_m']):.3f} cat) at a moderate R² gap of {agg.loc[8,'r2_gap']:.3f} — a reasonable trade-off. Beyond n=10 the accuracy gains slow while overfitting accelerates.</p>
</div>

<div class="box">
<h2>Figures</h2>
<h3>All metrics normalised (same scale)</h3>
<div class="fig-row"><div>{fig_norm}</div></div>

<h3>Raw scores: R² and Cosine</h3>
<div class="fig-row"><div>{fig_r2cos}</div><div>{fig_gap}</div></div>

<h3>Raw scores: Decoding accuracy</h3>
<div class="fig-row"><div>{fig_acc}</div></div>

<h3>Train vs Test (overfitting diagnostic)</h3>
<div class="fig-row"><div>{fig_trtest}</div></div>

<h3>Per-patient accuracy curves</h3>
<div class="fig-row"><div>{fig_word_pat}</div><div>{fig_cat_pat}</div></div>

<h3>Score heatmap (best per metric highlighted)</h3>
<div class="fig-row"><div>{fig_hm}</div></div>
</div>

<div class="box">
<h2>Numerical summary (n≤20, mean across 9 patient×embedding combos)</h2>
{row_highlight(summary_df)}
<p style="font-size:12px;color:#888">Bold = best value per metric column. R² gap = train_r2 − test_r2. Cos gap = train_cos − test_cos.</p>
</div>

</body></html>"""

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Saved: {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='test_results')
    ap.add_argument('--out', default='test_results/ncomponents_analysis.html')
    args = ap.parse_args()
    run(args.results_dir, args.out)


if __name__ == '__main__':
    main()
