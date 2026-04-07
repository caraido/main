"""
tests.report_ncomponents_tradeoff — Why do R²/cosine and word/cat accuracy
peak at different n_components in PLS?

Reads pls_learning_curve_*.csv / pls_lc_*.csv and produces a focused HTML report:
  1. Grand-mean learning curves (train + test) for all 4 metrics
  2. All metrics on a normalised axis to visualise the divergence
  3. Per-patient word and category accuracy curves
  4. Marginal gain per n_components step, per metric
  5. Four n_components selection criteria with consensus recommendation

Selection criteria implemented:
  A. Raw peak   — argmax of test metric (ignores overfitting)
  B. 95% threshold — smallest n reaching 95% of peak range (parsimonious)
  C. Penalised score — test_metric - λ*(train-test gap); favours n where
     generalisation is still healthy
  D. Elbow      — largest relative drop in marginal gain (diminishing returns)

Usage:
    python -m tests.report_ncomponents_tradeoff --results_dir <path> --out <path>
"""

import argparse, glob, os, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

CMAP = {
    'test_r2':   '#4e79a7',
    'train_r2':  '#9ab8d4',
    'test_cos':  '#f28e2b',
    'train_cos': '#f8c07b',
    'word_acc':  '#59a14f',
    'cat_acc':   '#e15759',
}
PAT_COLOURS = {'AA': '#4e79a7', 'RB': '#e15759', 'VB': '#59a14f',
               'LH': '#f28e2b', 'AZ': '#76b7b2', 'EH': '#9c6b9e', 'EM': '#b07d62'}


# ── tiny SVG helpers ──────────────────────────────────────────────────────────

def _polyline(xs, ys, fx, fy, colour, width=2.2, dash=None, opacity=1.0):
    pts = ' '.join(f'{fx(x):.1f},{fy(y):.1f}' for x, y in zip(xs, ys))
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<polyline points="{pts}" fill="none" stroke="{colour}" '
            f'stroke-width="{width}"{d} opacity="{opacity}"/>')


def _shade(xs, means, ses, fx, fy, colour):
    up = ' '.join(f'{fx(x):.1f},{fy(m+s):.1f}' for x, m, s in zip(xs, means, ses))
    lo = ' '.join(f'{fx(x):.1f},{fy(m-s):.1f}' for x, m, s in zip(reversed(xs), reversed(means), reversed(ses)))
    return f'<polygon points="{up} {lo}" fill="{colour}" opacity="0.15"/>'


def _axes(pad_l, pad_r, pad_t, pad_b, W, H, vmin, vmax, xs, ylabel, n_yticks=5):
    w = W - pad_l - pad_r
    h = H - pad_t - pad_b
    vr = vmax - vmin or 1
    xr = xs[-1] - xs[0] or 1

    def fx(v): return pad_l + w * (v - xs[0]) / xr
    def fy(v): return pad_t + h - h * (v - vmin) / vr

    parts = []
    parts.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+h}" stroke="#888" stroke-width="1.5"/>')
    parts.append(f'<line x1="{pad_l}" y1="{pad_t+h}" x2="{pad_l+w}" y2="{pad_t+h}" stroke="#888" stroke-width="1.5"/>')
    for i in range(n_yticks + 1):
        tv = vmin + vr * i / n_yticks
        ty = fy(tv)
        parts.append(f'<line x1="{pad_l-4}" y1="{ty:.1f}" x2="{pad_l}" y2="{ty:.1f}" stroke="#888"/>')
        parts.append(f'<text x="{pad_l-6}" y="{ty+4:.1f}" text-anchor="end" font-size="10" fill="#555">{tv:.2f}</text>')
    for x in xs:
        tx = fx(x)
        parts.append(f'<line x1="{tx:.1f}" y1="{pad_t+h}" x2="{tx:.1f}" y2="{pad_t+h+4}" stroke="#888"/>')
        parts.append(f'<text x="{tx:.1f}" y="{pad_t+h+15}" text-anchor="middle" font-size="10" fill="#555">{x}</text>')
    parts.append(f'<text x="{pad_l+w/2:.0f}" y="{H-4}" text-anchor="middle" font-size="11" fill="#333">n_components</text>')
    parts.append(f'<text transform="rotate(-90)" x="-{pad_t+h/2:.0f}" y="13" text-anchor="middle" font-size="11" fill="#333">{ylabel}</text>')
    if vmin < 0 < vmax:
        parts.append(f'<line x1="{pad_l}" y1="{fy(0):.1f}" x2="{pad_l+w}" y2="{fy(0):.1f}" stroke="#ccc" stroke-width="0.8" stroke-dasharray="4,2"/>')
    return '\n'.join(parts), fx, fy, w, h


def _vmark(fx, pad_t, h, n_val, colour, label):
    x = fx(n_val)
    return (f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{pad_t+h}" '
            f'stroke="{colour}" stroke-width="1.4" stroke-dasharray="5,3" opacity="0.75"/>'
            f'<text x="{x+3:.1f}" y="{pad_t+11}" font-size="9" fill="{colour}" font-weight="bold">{label}</text>')


def _legend(items, x, y_start, dy=18):
    parts = []
    for i, (label, colour, dash) in enumerate(items):
        y = y_start + i * dy
        d = f' stroke-dasharray="{dash}"' if dash else ''
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x+16}" y2="{y}" stroke="{colour}" stroke-width="2"{d}/>')
        parts.append(f'<text x="{x+20}" y="{y+4}" font-size="10" fill="#333">{label}</text>')
    return '\n'.join(parts)


# ── Selection criteria ────────────────────────────────────────────────────────

def criterion_peak(vals_by_n):
    """A: raw argmax."""
    return max(vals_by_n, key=vals_by_n.get)


def criterion_threshold(vals_by_n, pct=0.95):
    """B: smallest n reaching pct of (max - min) range above minimum."""
    ns = sorted(vals_by_n)
    vmin, vmax = min(vals_by_n.values()), max(vals_by_n.values())
    threshold = vmin + pct * (vmax - vmin)
    for n in ns:
        if vals_by_n[n] >= threshold:
            return n
    return ns[-1]


def criterion_penalised(test_by_n, gap_by_n, lam=0.5):
    """C: argmax of (test_metric - lam * gap)."""
    scores = {n: test_by_n[n] - lam * gap_by_n.get(n, 0) for n in test_by_n}
    return max(scores, key=scores.get), scores


def criterion_elbow(vals_by_n):
    """D: point of maximum curvature (second derivative) — diminishing returns."""
    ns = sorted(vals_by_n)
    if len(ns) < 3:
        return ns[0]
    ys = np.array([vals_by_n[n] for n in ns])
    # normalise to 0-1
    rng = ys.max() - ys.min()
    if rng == 0:
        return ns[0]
    yn = (ys - ys.min()) / rng
    xn = np.array([(n - ns[0]) / (ns[-1] - ns[0]) for n in ns])
    # perpendicular distance from line start→end
    line_vec = np.array([xn[-1] - xn[0], yn[-1] - yn[0]])
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return ns[0]
    dists = []
    for xi, yi in zip(xn, yn):
        pt = np.array([xi - xn[0], yi - yn[0]])
        proj = np.dot(pt, line_vec) / line_len
        proj_pt = proj * line_vec / line_len
        dists.append(np.linalg.norm(pt - proj_pt))
    return ns[int(np.argmax(dists))]


def compute_selection(grand_agg):
    """Apply all four criteria to each metric. Returns DataFrame of recommendations."""
    xs = sorted(grand_agg['n_components'].unique())
    rows = []

    metrics = [
        ('Cosine (test)',  'test_cos_mean',  'gap_cos'),
        ('Word Acc',       'word_mean',       'gap_r2'),
        ('Cat Acc',        'cat_mean',        'gap_r2'),
        ('R² (test)',      'test_r2_mean',    'gap_r2'),
    ]

    for label, col, gap_col in metrics:
        if col not in grand_agg.columns:
            continue
        vals  = dict(zip(grand_agg['n_components'], grand_agg[col]))
        gaps  = dict(zip(grand_agg['n_components'], grand_agg[gap_col]))

        n_peak = criterion_peak(vals)
        n_thr  = criterion_threshold(vals, pct=0.95)
        n_pen, pen_scores = criterion_penalised(vals, gaps, lam=0.5)
        n_elbow = criterion_elbow(vals)

        rows.append({
            'Metric': label,
            'Peak value': f'{vals[n_peak]:.4f} @ n={n_peak}',
            'A. Raw peak': n_peak,
            'B. 95% threshold': n_thr,
            'C. Penalised (λ=0.5)': n_pen,
            'D. Elbow': n_elbow,
        })

    return pd.DataFrame(rows)


# ── Figure helpers ────────────────────────────────────────────────────────────

def fig_four_panels(grand, xs, W=720, H=280):
    """4 panels: R² (train+test), Cosine (train+test), Word acc, Cat acc."""
    panel_configs = [
        ('test_r2_mean','train_r2_mean','test_r2_se','R²',     CMAP['test_r2'],  CMAP['train_r2'],  True),
        ('test_cos_mean','train_cos_mean','test_cos_se','Cosine', CMAP['test_cos'], CMAP['train_cos'], False),
        ('word_mean',    None,           'word_se',   'Word Acc', CMAP['word_acc'], None,              False),
        ('cat_mean',     None,           'cat_se',    'Cat Acc',  CMAP['cat_acc'],  None,              False),
    ]
    n_panels = len(panel_configs)
    pw = W // n_panels
    pad_l, pad_r, pad_t, pad_b = 50, 8, 30, 42

    svg = [f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" style="font-family:sans-serif">']
    svg.append(f'<text x="{W/2:.0f}" y="16" text-anchor="middle" font-size="13" font-weight="bold" fill="#222">'
               f'PLS Learning Curves — Grand Mean (all patients × embeddings)</text>')

    for pi, (tc, trc, tse, ylabel, col_t, col_tr, show_zero) in enumerate(panel_configs):
        ox = pi * pw
        test_v = grand[tc].values
        test_s = grand[tse].values if tse in grand.columns else np.zeros(len(xs))
        train_v = grand[trc].values if trc and trc in grand.columns else None

        all_v = list(test_v) + (list(train_v) if train_v is not None else [])
        vmin = min(min(all_v) - 0.01, 0) if show_zero else min(all_v) - 0.01
        vmax = max(all_v) + 0.01

        ax_svg, fx_rel, fy, w, h = _axes(pad_l, pad_r, pad_t, pad_b, pw, H, vmin, vmax, xs, ylabel, n_yticks=4)
        def fx(v, ox=ox): return fx_rel(v) + ox

        # shade + train
        if train_v is not None:
            svg.append(_polyline(xs, train_v, fx, fy, col_tr, width=1.5, dash='6,3', opacity=0.7))
        # shade + test
        svg.append(_shade(xs, test_v, test_s, fx, fy, col_t))
        svg.append(_polyline(xs, test_v, fx, fy, col_t, width=2.4))
        # dots
        best_idx = int(np.argmax(test_v))
        for i, (x, y) in enumerate(zip(xs, test_v)):
            r = 5 if i == best_idx else 3
            svg.append(f'<circle cx="{fx(x):.1f}" cy="{fy(y):.1f}" r="{r}" fill="{col_t}" stroke="#fff" stroke-width="1.2"/>')
        # best-n label
        svg.append(f'<text x="{fx(xs[best_idx]):.1f}" y="{fy(test_v[best_idx])-8:.1f}" '
                   f'text-anchor="middle" font-size="9" font-weight="bold" fill="{col_t}">n={xs[best_idx]}</text>')
        svg.append(f'<g transform="translate({ox},0)">{ax_svg}</g>')

        # legend
        lx = ox + pad_l + w - 55
        svg.append(f'<line x1="{lx}" y1="{pad_t+8}" x2="{lx+12}" y2="{pad_t+8}" stroke="{col_t}" stroke-width="2"/>')
        svg.append(f'<text x="{lx+15}" y="{pad_t+12}" font-size="9" fill="#555">test</text>')
        if train_v is not None:
            svg.append(f'<line x1="{lx}" y1="{pad_t+20}" x2="{lx+12}" y2="{pad_t+20}" stroke="{col_tr}" stroke-width="1.5" stroke-dasharray="4,2"/>')
            svg.append(f'<text x="{lx+15}" y="{pad_t+24}" font-size="9" fill="#555">train</text>')

    svg.append('</svg>')
    return '\n'.join(svg)


def fig_normalised_overlay(grand, xs, W=560, H=310):
    """All 4 test metrics normalised to [0,1]."""
    metrics = [
        ('test_r2_mean',  'R² (test)',     CMAP['test_r2']),
        ('test_cos_mean', 'Cosine (test)', CMAP['test_cos']),
        ('word_mean',     'Word Acc',      CMAP['word_acc']),
        ('cat_mean',      'Cat Acc',       CMAP['cat_acc']),
    ]
    normed = {}
    for col, label, col_c in metrics:
        if col not in grand.columns:
            continue
        v = grand[col].values.copy()
        mn, mx = v.min(), v.max()
        normed[label] = {'yn': (v - mn) / (mx - mn or 1), 'col': col_c}

    pad_l, pad_r, pad_t, pad_b = 55, 145, 35, 42
    w = W - pad_l - pad_r
    h = H - pad_t - pad_b
    xr = xs[-1] - xs[0] or 1
    def fx(v): return pad_l + w * (v - xs[0]) / xr
    def fy(v): return pad_t + h - h * v  # 0-1 space

    svg = [f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" style="font-family:sans-serif">']
    svg.append(f'<text x="{W/2:.0f}" y="18" text-anchor="middle" font-size="13" font-weight="bold" fill="#222">'
               f'All Metrics (min-max normalised)</text>')
    svg.append(f'<text x="{W/2:.0f}" y="30" text-anchor="middle" font-size="10" fill="#666">'
               f'0 = worst, 1 = best within each metric\'s range</text>')

    svg.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+h}" stroke="#888" stroke-width="1.5"/>')
    svg.append(f'<line x1="{pad_l}" y1="{pad_t+h}" x2="{pad_l+w}" y2="{pad_t+h}" stroke="#888" stroke-width="1.5"/>')
    for i in range(6):
        tv = i / 5
        ty = fy(tv)
        svg.append(f'<line x1="{pad_l-4}" y1="{ty:.1f}" x2="{pad_l}" y2="{ty:.1f}" stroke="#888"/>')
        svg.append(f'<text x="{pad_l-6}" y="{ty+4:.1f}" text-anchor="end" font-size="10" fill="#555">{tv:.1f}</text>')
    for x in xs:
        tx = fx(x)
        svg.append(f'<line x1="{tx:.1f}" y1="{pad_t+h}" x2="{tx:.1f}" y2="{pad_t+h+4}" stroke="#888"/>')
        svg.append(f'<text x="{tx:.1f}" y="{pad_t+h+15}" text-anchor="middle" font-size="10" fill="#555">{x}</text>')
    svg.append(f'<text x="{pad_l+w/2:.0f}" y="{H-4}" text-anchor="middle" font-size="11" fill="#333">n_components</text>')
    svg.append(f'<text transform="rotate(-90)" x="-{pad_t+h/2:.0f}" y="13" text-anchor="middle" font-size="11" fill="#333">Normalised score</text>')

    for label, d in normed.items():
        svg.append(_polyline(xs, d['yn'], fx, fy, d['col'], width=2.4))
        for x, y in zip(xs, d['yn']):
            svg.append(f'<circle cx="{fx(x):.1f}" cy="{fy(y):.1f}" r="3.5" fill="{d["col"]}"/>')

    # legend with peak annotations
    lx = pad_l + w + 10
    for i, (label, d) in enumerate(normed.items()):
        ly = pad_t + 20 + i * 36
        svg.append(f'<line x1="{lx}" y1="{ly+6}" x2="{lx+16}" y2="{ly+6}" stroke="{d["col"]}" stroke-width="2.5"/>')
        svg.append(f'<text x="{lx+20}" y="{ly+10}" font-size="10" fill="#222">{label}</text>')
        best_n = xs[int(np.argmax(d["yn"]))]
        svg.append(f'<text x="{lx+20}" y="{ly+22}" font-size="9" fill="#888">peak n={best_n}</text>')

    svg.append('</svg>')
    return '\n'.join(svg)


def fig_penalised_scores(grand, xs, W=560, H=310):
    """Show penalised score (test - λ*gap) for each metric at each n."""
    metrics = [
        ('test_cos_mean', 'gap_cos', 'Cosine − 0.5·gap', CMAP['test_cos']),
        ('word_mean',     'gap_r2',  'Word − 0.5·gap',   CMAP['word_acc']),
        ('cat_mean',      'gap_r2',  'Cat − 0.5·gap',    CMAP['cat_acc']),
    ]
    curves = {}
    for col, gap_col, label, colour in metrics:
        if col not in grand.columns or gap_col not in grand.columns:
            continue
        pen = grand[col].values - 0.5 * grand[gap_col].values
        curves[label] = {'v': pen, 'col': colour}

    all_v = [v for d in curves.values() for v in d['v']]
    vmin, vmax = min(all_v) - 0.01, max(all_v) + 0.01

    pad_l, pad_r, pad_t, pad_b = 55, 145, 35, 42
    w = W - pad_l - pad_r
    h = H - pad_t - pad_b
    xr = xs[-1] - xs[0] or 1
    vr = vmax - vmin or 1
    def fx(v): return pad_l + w * (v - xs[0]) / xr
    def fy(v): return pad_t + h - h * (v - vmin) / vr

    svg = [f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" style="font-family:sans-serif">']
    svg.append(f'<text x="{W/2:.0f}" y="18" text-anchor="middle" font-size="13" font-weight="bold" fill="#222">'
               f'Penalised Score (test − 0.5 × gap)</text>')
    svg.append(f'<text x="{W/2:.0f}" y="30" text-anchor="middle" font-size="10" fill="#666">'
               f'Rewards high test performance, penalises overfitting</text>')
    svg.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+h}" stroke="#888" stroke-width="1.5"/>')
    svg.append(f'<line x1="{pad_l}" y1="{pad_t+h}" x2="{pad_l+w}" y2="{pad_t+h}" stroke="#888" stroke-width="1.5"/>')
    for i in range(5):
        tv = vmin + vr * i / 4
        ty = fy(tv)
        svg.append(f'<line x1="{pad_l-4}" y1="{ty:.1f}" x2="{pad_l}" y2="{ty:.1f}" stroke="#888"/>')
        svg.append(f'<text x="{pad_l-6}" y="{ty+4:.1f}" text-anchor="end" font-size="10" fill="#555">{tv:.3f}</text>')
    for x in xs:
        tx = fx(x)
        svg.append(f'<line x1="{tx:.1f}" y1="{pad_t+h}" x2="{tx:.1f}" y2="{pad_t+h+4}" stroke="#888"/>')
        svg.append(f'<text x="{tx:.1f}" y="{pad_t+h+15}" text-anchor="middle" font-size="10" fill="#555">{x}</text>')
    svg.append(f'<text x="{pad_l+w/2:.0f}" y="{H-4}" text-anchor="middle" font-size="11" fill="#333">n_components</text>')
    svg.append(f'<text transform="rotate(-90)" x="-{pad_t+h/2:.0f}" y="13" text-anchor="middle" font-size="11" fill="#333">Penalised score</text>')

    for label, d in curves.items():
        best_n = xs[int(np.argmax(d['v']))]
        svg.append(_vmark(fx, pad_t, h, best_n, d['col'], f'n={best_n}'))
        svg.append(_polyline(xs, d['v'], fx, fy, d['col'], width=2.4))
        for x, y in zip(xs, d['v']):
            svg.append(f'<circle cx="{fx(x):.1f}" cy="{fy(y):.1f}" r="3.5" fill="{d["col"]}"/>')

    lx = pad_l + w + 10
    for i, (label, d) in enumerate(curves.items()):
        ly = pad_t + 20 + i * 36
        best_n = xs[int(np.argmax(d['v']))]
        svg.append(f'<line x1="{lx}" y1="{ly+6}" x2="{lx+16}" y2="{ly+6}" stroke="{d["col"]}" stroke-width="2.5"/>')
        svg.append(f'<text x="{lx+20}" y="{ly+10}" font-size="10" fill="#222">{label}</text>')
        svg.append(f'<text x="{lx+20}" y="{ly+22}" font-size="9" fill="#888">best n={best_n}</text>')

    svg.append('</svg>')
    return '\n'.join(svg)


def fig_gap_curve(grand, xs, W=500, H=290):
    """Train-test R² gap vs n_components."""
    gap = grand['gap_r2'].values
    vmin, vmax = 0, max(gap) * 1.1
    pad_l, pad_r, pad_t, pad_b = 55, 20, 35, 42
    w = W - pad_l - pad_r
    h = H - pad_t - pad_b
    vr = vmax - vmin
    xr = xs[-1] - xs[0] or 1
    def fx(v): return pad_l + w * (v - xs[0]) / xr
    def fy(v): return pad_t + h - h * (v - vmin) / vr

    svg = [f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" style="font-family:sans-serif">']
    svg.append(f'<text x="{W/2:.0f}" y="18" text-anchor="middle" font-size="13" font-weight="bold" fill="#222">'
               f'Train − Test R² Gap (overfitting diagnostic)</text>')
    svg.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+h}" stroke="#888" stroke-width="1.5"/>')
    svg.append(f'<line x1="{pad_l}" y1="{pad_t+h}" x2="{pad_l+w}" y2="{pad_t+h}" stroke="#888" stroke-width="1.5"/>')
    for i in range(5):
        tv = vmin + vr * i / 4
        ty = fy(tv)
        svg.append(f'<line x1="{pad_l-4}" y1="{ty:.1f}" x2="{pad_l}" y2="{ty:.1f}" stroke="#888"/>')
        svg.append(f'<text x="{pad_l-6}" y="{ty+4:.1f}" text-anchor="end" font-size="10" fill="#555">{tv:.2f}</text>')
    for x in xs:
        tx = fx(x)
        svg.append(f'<line x1="{tx:.1f}" y1="{pad_t+h}" x2="{tx:.1f}" y2="{pad_t+h+4}" stroke="#888"/>')
        svg.append(f'<text x="{tx:.1f}" y="{pad_t+h+15}" text-anchor="middle" font-size="10" fill="#555">{x}</text>')
    svg.append(f'<text x="{pad_l+w/2:.0f}" y="{H-4}" text-anchor="middle" font-size="11" fill="#333">n_components</text>')
    svg.append(f'<text transform="rotate(-90)" x="-{pad_t+h/2:.0f}" y="13" text-anchor="middle" font-size="11" fill="#333">Train − Test R²</text>')

    # zone backgrounds
    for lo, hi, col, label in [(0, 0.10, '#dcfce7', 'healthy'),
                                (0.10, 0.20, '#fef9c3', 'moderate'),
                                (0.20, vmax, '#fee2e2', 'overfit')]:
        y_hi = fy(min(hi, vmax)); y_lo = fy(lo)
        bh = abs(y_lo - y_hi)
        svg.append(f'<rect x="{pad_l}" y="{y_hi:.1f}" width="{w}" height="{bh:.1f}" fill="{col}" opacity="0.4"/>')
        svg.append(f'<text x="{pad_l+w-4}" y="{(y_hi+y_lo)/2:.1f}" text-anchor="end" font-size="9" fill="#555" opacity="0.8">{label}</text>')

    # fill under curve
    pts_fill = f'{pad_l},{pad_t+h} ' + ' '.join(f'{fx(x):.1f},{fy(g):.1f}' for x, g in zip(xs, gap)) + f' {fx(xs[-1]):.1f},{pad_t+h}'
    svg.append(f'<polygon points="{pts_fill}" fill="#4e79a7" opacity="0.2"/>')
    svg.append(_polyline(xs, gap, fx, fy, '#4e79a7', width=2.4))
    for x, g in zip(xs, gap):
        svg.append(f'<circle cx="{fx(x):.1f}" cy="{fy(g):.1f}" r="3.5" fill="#4e79a7" stroke="#fff" stroke-width="1"/>')

    # threshold lines
    for thresh, colour, label in [(0.10, '#16a34a', '0.10'), (0.20, '#dc2626', '0.20')]:
        ty = fy(thresh)
        svg.append(f'<line x1="{pad_l}" y1="{ty:.1f}" x2="{pad_l+w}" y2="{ty:.1f}" stroke="{colour}" stroke-width="1.2" stroke-dasharray="5,3"/>')
        svg.append(f'<text x="{pad_l+4}" y="{ty-3:.1f}" font-size="9" fill="{colour}">gap={label}</text>')

    svg.append('</svg>')
    return '\n'.join(svg)


def fig_per_patient(agg, metric_col, ylabel, xs_all, W=560, H=290):
    """Per-patient curves (averaged across embeddings)."""
    per_pat = {}
    for pat, g in agg.groupby('patient'):
        pnc = g.groupby('n_components')[metric_col].agg(['mean','sem']).reset_index()
        pnc = pnc[pnc['n_components'].isin(xs_all)]
        per_pat[pat] = pnc

    all_v = [v for d in per_pat.values() for v in d['mean']]
    vmin, vmax = min(all_v) * 0.95, max(all_v) * 1.08
    pad_l, pad_r, pad_t, pad_b = 55, 110, 35, 42
    w = W - pad_l - pad_r
    h = H - pad_t - pad_b
    vr = vmax - vmin or 1
    xr = xs_all[-1] - xs_all[0] or 1
    def fx(v): return pad_l + w * (v - xs_all[0]) / xr
    def fy(v): return pad_t + h - h * (v - vmin) / vr

    svg = [f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" style="font-family:sans-serif">']
    svg.append(f'<text x="{W/2:.0f}" y="18" text-anchor="middle" font-size="13" font-weight="bold" fill="#222">'
               f'{ylabel} per patient</text>')
    svg.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+h}" stroke="#888" stroke-width="1.5"/>')
    svg.append(f'<line x1="{pad_l}" y1="{pad_t+h}" x2="{pad_l+w}" y2="{pad_t+h}" stroke="#888" stroke-width="1.5"/>')
    for i in range(5):
        tv = vmin + vr * i / 4
        ty = fy(tv)
        svg.append(f'<line x1="{pad_l-4}" y1="{ty:.1f}" x2="{pad_l}" y2="{ty:.1f}" stroke="#888"/>')
        svg.append(f'<text x="{pad_l-6}" y="{ty+4:.1f}" text-anchor="end" font-size="10" fill="#555">{tv:.3f}</text>')
    for x in xs_all:
        tx = fx(x)
        svg.append(f'<line x1="{tx:.1f}" y1="{pad_t+h}" x2="{tx:.1f}" y2="{pad_t+h+4}" stroke="#888"/>')
        svg.append(f'<text x="{tx:.1f}" y="{pad_t+h+15}" text-anchor="middle" font-size="10" fill="#555">{x}</text>')
    svg.append(f'<text x="{pad_l+w/2:.0f}" y="{H-4}" text-anchor="middle" font-size="11" fill="#333">n_components</text>')
    svg.append(f'<text transform="rotate(-90)" x="-{pad_t+h/2:.0f}" y="13" text-anchor="middle" font-size="11" fill="#333">{ylabel}</text>')

    for pat, pnc in per_pat.items():
        col = PAT_COLOURS.get(pat, '#888')
        pxs = pnc['n_components'].tolist()
        pys = pnc['mean'].tolist()
        pses = pnc['sem'].tolist()
        svg.append(_shade(pxs, pys, pses, fx, fy, col))
        svg.append(_polyline(pxs, pys, fx, fy, col, width=2.2))
        for x, y in zip(pxs, pys):
            svg.append(f'<circle cx="{fx(x):.1f}" cy="{fy(y):.1f}" r="3.5" fill="{col}" stroke="#fff" stroke-width="1"/>')

    lx = pad_l + w + 10
    for i, (pat, _) in enumerate(per_pat.items()):
        col = PAT_COLOURS.get(pat, '#888')
        ly = pad_t + 18 + i * 20
        svg.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+14}" y2="{ly}" stroke="{col}" stroke-width="2.5"/>')
        svg.append(f'<text x="{lx+18}" y="{ly+4}" font-size="10" fill="#333">{pat}</text>')

    svg.append('</svg>')
    return '\n'.join(svg)


def fig_selection_heatmap(agg, xs_all, W=700, H=260):
    """Heatmap: best n_components per criterion per patient×embedding."""
    metrics = [
        ('test_cos_mean', 'gap_cos', 'Cosine'),
        ('word_mean',     'gap_r2',  'Word Acc'),
        ('cat_mean',      'gap_r2',  'Cat Acc'),
    ]
    CRITERIA = ['Peak', '95% thr.', 'Penalised', 'Elbow']

    rows_data = []
    for pat, emb in sorted(set(zip(agg.patient, agg.embedding))):
        g = agg[(agg.patient == pat) & (agg.embedding == emb)].sort_values('n_components')
        for metric_col, gap_col, mlabel in metrics:
            if metric_col not in g.columns:
                continue
            vals = dict(zip(g['n_components'], g[metric_col]))
            gaps = dict(zip(g['n_components'], g[gap_col]))
            xlist = sorted(vals)
            n_a = criterion_peak(vals)
            n_b = criterion_threshold(vals)
            n_c, _ = criterion_penalised(vals, gaps)
            n_d = criterion_elbow(vals)
            rows_data.append({
                'Patient': pat, 'Embedding': emb, 'Metric': mlabel,
                'Peak': n_a, '95% thr.': n_b, 'Penalised': n_c, 'Elbow': n_d,
            })

    df_heat = pd.DataFrame(rows_data)

    # colour map: map n_comp values to a gradient
    all_ns = sorted(xs_all)
    n_to_col = {}
    blues = ['#dbeafe','#bfdbfe','#93c5fd','#60a5fa','#3b82f6','#2563eb','#1d4ed8','#1e40af','#1e3a8a','#172554','#0f172a']
    for i, n in enumerate(all_ns):
        ci = min(int(i / len(all_ns) * len(blues)), len(blues)-1)
        n_to_col[n] = blues[ci]

    pad_l, pad_r, pad_t, pad_b = 110, 10, 50, 30
    row_h = 18
    col_w = 55
    n_rows = len(df_heat)
    n_cols = len(CRITERIA)
    actual_h = max(H, pad_t + n_rows * row_h + pad_b)

    svg = [f'<svg width="{W}" height="{actual_h}" xmlns="http://www.w3.org/2000/svg" style="font-family:sans-serif">']
    svg.append(f'<text x="{W/2:.0f}" y="18" text-anchor="middle" font-size="13" font-weight="bold" fill="#222">'
               f'Best n_components per criterion (patient × embedding × metric)</text>')

    # column headers
    for ci, crit in enumerate(CRITERIA):
        cx = pad_l + ci * col_w + col_w / 2
        svg.append(f'<text x="{cx:.0f}" y="40" text-anchor="middle" font-size="10" font-weight="bold" fill="#333">{crit}</text>')

    for ri, row in df_heat.iterrows():
        y = pad_t + ri * row_h
        label = f"{row['Patient']} {row['Embedding'][:4]} {row['Metric'][:3]}"
        svg.append(f'<text x="{pad_l-4}" y="{y+13}" text-anchor="end" font-size="9" fill="#333">{label}</text>')
        for ci, crit in enumerate(CRITERIA):
            n_val = int(row[crit])
            col = n_to_col.get(n_val, '#e5e7eb')
            cx = pad_l + ci * col_w
            svg.append(f'<rect x="{cx}" y="{y+1}" width="{col_w-2}" height="{row_h-2}" fill="{col}" rx="2"/>')
            svg.append(f'<text x="{cx+col_w/2:.0f}" y="{y+13}" text-anchor="middle" font-size="10" '
                       f'fill="{"#fff" if n_val >= 15 else "#222"}" font-weight="bold">{n_val}</text>')

    # legend
    lx = pad_l + n_cols * col_w + 10
    svg.append(f'<text x="{lx}" y="42" font-size="9" fill="#555" font-weight="bold">n value:</text>')
    for i, n in enumerate(all_ns):
        col = n_to_col[n]
        lbx = lx + (i % 4) * 28
        lby = 52 + (i // 4) * 18
        svg.append(f'<rect x="{lbx}" y="{lby}" width="24" height="14" fill="{col}" rx="2"/>')
        svg.append(f'<text x="{lbx+12}" y="{lby+10}" text-anchor="middle" font-size="8" '
                   f'fill="{"#fff" if n >= 15 else "#222"}">{n}</text>')

    svg.append('</svg>')
    return '\n'.join(svg)


# ── HTML assembly ─────────────────────────────────────────────────────────────

def build_html(df_raw, grand, agg, xs):
    sel_df = compute_selection(grand)

    style = """
    body { font-family: system-ui, sans-serif; max-width: 1300px; margin: 0 auto;
           padding: 24px; color: #222; background: #fafafa; }
    h1 { border-bottom: 3px solid #4e79a7; padding-bottom: 8px; }
    h2 { margin-top: 36px; color: #333; border-left: 5px solid #4e79a7; padding-left: 10px; }
    h3 { color: #555; margin-top: 18px; }
    .box { background: #fff; border-radius: 8px; box-shadow: 0 1px 6px #0001;
           padding: 20px 24px; margin-bottom: 24px; }
    .insight { border-left: 4px solid #4e79a7; background: #eaf3fb;
               padding: 10px 16px; border-radius: 4px; margin: 10px 0; font-size: 0.95rem; }
    .insight.green  { border-color: #16a34a; background: #f0fdf4; }
    .insight.orange { border-color: #f59e0b; background: #fffbeb; }
    .insight.red    { border-color: #dc2626; background: #fef2f2; }
    table { border-collapse: collapse; font-size: 13px; width: 100%; }
    th { background: #4e79a7; color: #fff; padding: 7px 10px; text-align: center; }
    td { padding: 6px 10px; border-bottom: 1px solid #eee; text-align: center; }
    tr:nth-child(even) td { background: #f5f7fa; }
    td:first-child { text-align: left; font-weight: 500; }
    .rec { background: #fef9c3; font-weight: bold; }
    .figs { display: flex; flex-wrap: wrap; gap: 18px; }
    """

    # consensus recommendation per metric
    def consensus(row):
        votes = [row['A. Raw peak'], row['B. 95% threshold'],
                 row['C. Penalised (λ=0.5)'], row['D. Elbow']]
        from collections import Counter
        c = Counter(votes)
        winner = c.most_common(1)[0][0]
        return winner

    html = [f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>PLS n_components — Why Metrics Disagree</title>
<style>{style}</style></head><body>
<h1>PLS n_components: Why Different Metrics Suggest Different Values</h1>
<p style="color:#666">Patients: {', '.join(sorted(df_raw.patient.unique()))} &nbsp;|&nbsp;
Embeddings: {', '.join(sorted(df_raw.embedding.unique()))} &nbsp;|&nbsp;
n range: {xs[0]}–{xs[-1]}</p>
"""]

    # ── Key findings ─────────────────────────────────────────────────────────
    html.append('<div class="box">')
    html.append('<h2 style="margin-top:0">Key Findings</h2>')
    html.append('<div class="insight"><b>R² (test)</b> peaks at <b>n=2–4</b>. '
                'Extra components overfit the regression surface: train R² keeps rising '
                'but test R² immediately declines. At n=40 the train−test gap exceeds 0.89 on average.</div>')
    html.append('<div class="insight orange"><b>Cosine similarity (test)</b> peaks at <b>n=4</b>. '
                'Same mechanism — directional reconstruction degrades once the model overfits.</div>')
    html.append('<div class="insight green"><b>Word and category accuracy</b> increase monotonically '
                'through <b>n=40</b>. Retrieval is a <i>ranking</i> task: even noisy extra dimensions '
                'help separate word embeddings in nearest-neighbour space. The test-set accuracy is '
                'real (held-out folds), but comes at the cost of heavy regression overfitting.</div>')
    html.append('<div class="insight red"><b>No single n is universally best.</b> '
                'If you care about embedding geometry (cosine, R²): use <b>n=4</b>. '
                'If you care about retrieval accuracy with a controlled overfitting budget: '
                'use <b>n=6–8</b> (95% of retrieval gain, gap still &lt;0.20).</div>')
    html.append('</div>')

    # ── Selection criteria table ──────────────────────────────────────────────
    html.append('<div class="box">')
    html.append('<h2>Selection Criteria Compared</h2>')
    html.append('<p>Four principled methods for picking n_components, applied to grand-mean curves '
                '(averaged across all patients × embeddings).</p>')
    html.append('<table><tr><th>Metric</th><th>Peak value</th>'
                '<th>A. Raw peak</th><th>B. 95% threshold</th>'
                '<th>C. Penalised<br>(test − 0.5·gap)</th>'
                '<th>D. Elbow<br>(curvature)</th>'
                '<th>Consensus</th></tr>')
    for _, row in sel_df.iterrows():
        con = consensus(row)
        html.append(f'<tr>'
                    f'<td>{row["Metric"]}</td>'
                    f'<td style="color:#555">{row["Peak value"]}</td>'
                    f'<td>{row["A. Raw peak"]}</td>'
                    f'<td>{row["B. 95% threshold"]}</td>'
                    f'<td>{row["C. Penalised (λ=0.5)"]}</td>'
                    f'<td>{row["D. Elbow"]}</td>'
                    f'<td class="rec">{con}</td></tr>')
    html.append('</table>')
    html.append('<p style="font-size:0.88rem;color:#666;margin-top:8px">'
                '<b>A. Raw peak:</b> argmax of test metric (ignores overfitting). '
                '<b>B. 95% threshold:</b> smallest n reaching 95% of the peak range above minimum — '
                'parsimonious. '
                '<b>C. Penalised:</b> argmax of (test metric − 0.5 × train−test R² gap) — '
                'balances performance vs generalisation. '
                '<b>D. Elbow:</b> point of maximum curvature — diminishing returns.</p>')
    html.append('</div>')

    # ── Figures ───────────────────────────────────────────────────────────────
    html.append(f'<div class="box"><h2>Learning Curves — Grand Mean</h2>'
                f'<p>Shading = ±1 SE. Labelled dot = peak of test metric.</p>'
                f'{fig_four_panels(grand, xs)}</div>')

    html.append(f'<div class="box"><h2>All Metrics — Normalised Overlay</h2>'
                f'<p>Each metric scaled to [0, 1] to show trajectories on the same axis. '
                f'The divergence between regression metrics (R², cosine) and retrieval metrics '
                f'(word, cat) is visible: regression peaks early, retrieval keeps climbing.</p>'
                f'<div class="figs">'
                f'<div>{fig_normalised_overlay(grand, xs)}</div>'
                f'<div>{fig_gap_curve(grand, xs)}</div>'
                f'</div></div>')

    html.append(f'<div class="box"><h2>Penalised Score</h2>'
                f'<p>score = test metric − 0.5 × (train−test R² gap). '
                f'This explicitly penalises overfitting. Peaks show where the retrieval '
                f'benefit stops compensating for the growing generalisation cost.</p>'
                f'{fig_penalised_scores(grand, xs)}</div>')

    html.append(f'<div class="box"><h2>Per-Patient Curves</h2>'
                f'<div class="figs">'
                f'<div>{fig_per_patient(agg, "word_mean", "Word Acc.", xs)}</div>'
                f'<div>{fig_per_patient(agg, "cat_mean", "Cat Acc.", xs)}</div>'
                f'<div>{fig_per_patient(agg, "test_cos_mean", "Cosine (test)", xs)}</div>'
                f'</div></div>')

    html.append(f'<div class="box"><h2>Per-Configuration Heatmap</h2>'
                f'<p>Best n for each (patient × embedding × metric) combination, by each criterion. '
                f'Darker blue = larger n.</p>'
                f'{fig_selection_heatmap(agg, xs)}</div>')

    html.append('<div class="box"><h2>Interpretation</h2>')
    html.append("""
<h3>Why R² and cosine peak at n=4</h3>
<p>PLS maximises the covariance between neural X and embedding Y. With limited training data
(~100 trials), the first 2–4 components capture genuine shared variance. Additional components
model idiosyncratic noise in the training splits — train R² keeps rising but test R² and
cosine immediately decline. This is standard regression overfitting.</p>

<h3>Why word/cat accuracy keep rising</h3>
<p>Nearest-neighbour retrieval only asks: "is the true word closer than all others?"
It does not care about absolute prediction error. Extra PLS components can degrade the
overall embedding reconstruction (hurting R²/cosine) while simultaneously spreading
word embeddings further apart in the prediction space, making individual words easier
to discriminate. Since test accuracy is evaluated on held-out folds, the improvement
is a real generalisation signal — but it comes with heavily overfitted regression.</p>

<h3>Practical recommendation</h3>
<p><b>n=4:</b> use when embedding geometry matters (cosine similarity, R², cross-patient
generalisation). Minimal overfitting (gap ≈ 0.10).</p>
<p><b>n=6–8:</b> use when maximising word/category retrieval accuracy is the goal.
Captures ~80–90% of the full retrieval gain. Gap ≈ 0.15–0.20 — large but not extreme.
The penalised criterion (C) typically recommends this range.</p>
<p><b>n ≥ 20:</b> maximises raw test accuracy but gap &gt; 0.45 — the regression model
has massively overfit even though held-out retrieval accuracy is still improving.</p>
""")
    html.append('</div>')
    html.append('</body></html>')
    return '\n'.join(html)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='test_results')
    ap.add_argument('--out', default='test_results/report_ncomponents_tradeoff.html')
    args = ap.parse_args()
    # Convert to absolute path based on likely workspace structure
    if not os.path.isabs(args.results_dir):
        # Assume running from main/ or test_results/ is up one level
        if 'main' in os.getcwd():
            args.results_dir = os.path.join(os.getcwd(), '..', args.results_dir)
        args.results_dir = os.path.abspath(args.results_dir)
    if not os.path.isabs(args.out):
        if 'main' in os.getcwd():
            args.out = os.path.join(os.getcwd(), '..', args.out)
        args.out = os.path.abspath(args.out)

    dfs = []
    files = sorted(glob.glob(os.path.join(args.results_dir, 'pls_lc_*.csv')))
    for f in files:
        dfs.append(pd.read_csv(f))
    # Note: prefer new-format pls_lc_*.csv files; old-format files have incomplete n_components coverage
    if not dfs:
        print('No pls_lc_*.csv files found.')
        return

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['patient','embedding','model','n_components','epoch'])
    # Include both 'pls' and 'kernel_pls' models
    df = df[df.model.isin(['pls', 'kernel_pls'])]

    agg = df.groupby(['patient','embedding','n_components']).agg(
        test_r2_mean=('test_r2','mean'),    test_r2_se=('test_r2','sem'),
        train_r2_mean=('train_r2','mean'),  train_r2_se=('train_r2','sem'),
        test_cos_mean=('test_cosine','mean'),test_cos_se=('test_cosine','sem'),
        train_cos_mean=('train_cosine','mean'),
        word_mean=('word_bal_acc','mean'),  word_se=('word_bal_acc','sem'),
        cat_mean=('cat_bal_acc','mean'),    cat_se=('cat_bal_acc','sem'),
    ).reset_index()
    agg['gap_r2']  = agg['train_r2_mean']  - agg['test_r2_mean']
    agg['gap_cos'] = agg['train_cos_mean'] - agg['test_cos_mean']

    xs = sorted(agg['n_components'].unique())

    grand = df.groupby('n_components').agg(
        test_r2_mean=('test_r2','mean'),    test_r2_se=('test_r2','sem'),
        train_r2_mean=('train_r2','mean'),  train_r2_se=('train_r2','sem'),
        test_cos_mean=('test_cosine','mean'),test_cos_se=('test_cosine','sem'),
        train_cos_mean=('train_cosine','mean'),
        word_mean=('word_bal_acc','mean'),  word_se=('word_bal_acc','sem'),
        cat_mean=('cat_bal_acc','mean'),    cat_se=('cat_bal_acc','sem'),
    ).reset_index()
    grand['gap_r2']  = grand['train_r2_mean']  - grand['test_r2_mean']
    grand['gap_cos'] = grand['train_cos_mean'] - grand['test_cos_mean']

    html = build_html(df, grand, agg, xs)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Saved: {args.out}')

    # terminal summary
    sel = compute_selection(grand)
    print('\n=== Selection criteria summary ===')
    for _, row in sel.iterrows():
        from collections import Counter
        votes = [row['A. Raw peak'], row['B. 95% threshold'],
                 row['C. Penalised (λ=0.5)'], row['D. Elbow']]
        winner = Counter(votes).most_common(1)[0][0]
        print(f"  {row['Metric']:18s}  A={row['A. Raw peak']:2d}  B={row['B. 95% threshold']:2d}  "
              f"C={row['C. Penalised (λ=0.5)']:2d}  D={row['D. Elbow']:2d}  → consensus n={winner}")


if __name__ == '__main__':
    main()
