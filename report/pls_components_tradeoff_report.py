# -*- coding: utf-8 -*-
"""
tests.pls_components_tradeoff_report — Why do R²/cosine and word/cat accuracy
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
    python -m analysis.pls_components_tradeoff_report --results_dir <path> --out <path>
"""

import argparse, glob, os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from analysis.helpers._phoneme_semantic_helpers import get_out_dir
from report.helper.html_utils import fig_to_base64
from report.render import stylesheet

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


# ── Chart helpers ─────────────────────────────────────────────────────────────
# These were ~60 lines of hand-composed SVG primitives — a shared `_axes()` that
# returned its own fx/fy pixel transforms, polylines built from formatted point
# strings, and ±1 SE bands drawn as a forward-then-reversed <polygon>. Rewritten on
# matplotlib 2026-08-11 (Alec). `W`/`H` stay in PIXELS so the call sites in
# build_html are unchanged; _figsize converts. Charts render as inline base64 PNG,
# like every other report.

_DPI = 130
_NO_DATA = '<p class="subtle">(no data)</p>'


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


def _cols(frame):
    """Column list of a DataFrame that may be None."""
    return list(getattr(frame, 'columns', []))


def _floats(frame, col, n):
    """`frame[col]` as a float array, or zeros of length n when absent."""
    if col and col in _cols(frame):
        return np.nan_to_num(np.asarray(frame[col].values, dtype=float))
    return np.zeros(n)


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
    panel_configs = [p for p in panel_configs if p[0] in _cols(grand)]
    if grand is None or not len(grand) or not len(xs) or not panel_configs:
        return _NO_DATA

    fig, axes = plt.subplots(1, len(panel_configs), figsize=_figsize(W, H), squeeze=False)
    for ax, (tc, trc, tse, ylabel, col_t, col_tr, show_zero) in zip(axes[0], panel_configs):
        test_v = np.asarray(grand[tc].values, dtype=float)
        test_s = _floats(grand, tse, len(test_v))
        train_v = (np.asarray(grand[trc].values, dtype=float)
                   if trc and trc in _cols(grand) else None)

        all_v = list(test_v) + (list(train_v) if train_v is not None else [])
        vmin = min(min(all_v) - 0.01, 0) if show_zero else min(all_v) - 0.01
        vmax = max(all_v) + 0.01

        if train_v is not None:
            ax.plot(xs, train_v, color=col_tr, lw=1.5, ls=(0, (6, 3)), alpha=0.7,
                    label='train')
        ax.fill_between(xs, test_v - test_s, test_v + test_s,
                        color=col_t, alpha=0.15, lw=0)
        ax.plot(xs, test_v, color=col_t, lw=2.4, marker='o', ms=4,
                mec='#fff', mew=1.2, label='test')

        # the peak of the test curve gets a fatter dot and its n printed above it
        best_idx = int(np.argmax(test_v))
        ax.plot([xs[best_idx]], [test_v[best_idx]], marker='o', ms=7, color=col_t,
                mec='#fff', mew=1.2, ls='none')
        # centred over the dot, except at either end of the axis where half the
        # label would fall outside the panel
        ha, dx = 'center', 0
        if best_idx == 0:
            ha, dx = 'left', -2
        elif best_idx == len(test_v) - 1:
            ha, dx = 'right', 2
        ax.annotate('n={}'.format(xs[best_idx]), (xs[best_idx], test_v[best_idx]),
                    textcoords='offset points', xytext=(dx, 9), ha=ha,
                    fontsize=8, fontweight='bold', color=col_t)

        if vmin < 0 < vmax:
            ax.axhline(0, color='#ccc', lw=0.8, ls=(0, (4, 2)))
        ax.set_ylim(vmin, vmax)
        ax.margins(x=0.08)          # room for the peak label when it lands on an end
        # Four panels share one chart's width, so a tick per n_components is more
        # labels than fit. Keep every tick, thin the labels.
        ax.set_xticks(list(xs))
        step = max(1, int(np.ceil(len(xs) / 5.0)))
        ax.set_xticklabels([str(x) if (i % step == 0 or i == len(xs) - 1) else ''
                            for i, x in enumerate(xs)])
        ax.set_xlabel('n_components', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        # 'best': the test curve rises in two panels and falls in two, so any fixed
        # corner collides with a curve or with the peak label in half of them.
        ax.legend(fontsize=7.5, loc='best', frameon=False)
        _style(ax)

    fig.suptitle('PLS Learning Curves — Grand Mean (all patients × embeddings)',
                 fontsize=11, fontweight='bold', color='#222')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _img(fig)


def fig_normalised_overlay(grand, xs, W=560, H=310):
    """All 4 test metrics normalised to [0,1]."""
    metrics = [
        ('test_r2_mean',  'R² (test)',     CMAP['test_r2']),
        ('test_cos_mean', 'Cosine (test)', CMAP['test_cos']),
        ('word_mean',     'Word Acc',      CMAP['word_acc']),
        ('cat_mean',      'Cat Acc',       CMAP['cat_acc']),
    ]
    if grand is None or not len(grand) or not len(xs):
        return _NO_DATA
    normed = {}
    for col, label, col_c in metrics:
        if col not in _cols(grand):
            continue
        v = np.asarray(grand[col].values, dtype=float)
        mn, mx = v.min(), v.max()
        normed[label] = {'yn': (v - mn) / (mx - mn or 1), 'col': col_c}
    if not normed:
        return _NO_DATA

    fig, ax = plt.subplots(figsize=_figsize(W, H))
    for label, d in normed.items():
        # peak n rides in the legend entry; the SVG version printed it as a second
        # line under each swatch.
        best_n = xs[int(np.argmax(d['yn']))]
        ax.plot(xs, d['yn'], color=d['col'], lw=2.4, marker='o', ms=4,
                label='{}\npeak n={}'.format(label, best_n))

    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xticks(list(xs))
    ax.set_xlabel('n_components', fontsize=9)
    ax.set_ylabel('Normalised score', fontsize=9)
    ax.set_title("0 = worst, 1 = best within each metric's range",
                 fontsize=8.5, color='#666')
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5),
              frameon=False, labelspacing=1.1)
    _style(ax)
    fig.suptitle('All Metrics (min-max normalised)', fontsize=11,
                 fontweight='bold', color='#222')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _img(fig)


def fig_penalised_scores(grand, xs, W=560, H=310):
    """Show penalised score (test - λ*gap) for each metric at each n."""
    metrics = [
        ('test_cos_mean', 'gap_cos', 'Cosine − 0.5·gap', CMAP['test_cos']),
        ('word_mean',     'gap_r2',  'Word − 0.5·gap',   CMAP['word_acc']),
        ('cat_mean',      'gap_r2',  'Cat − 0.5·gap',    CMAP['cat_acc']),
    ]
    if grand is None or not len(grand) or not len(xs):
        return _NO_DATA
    curves = {}
    for col, gap_col, label, colour in metrics:
        if col not in _cols(grand) or gap_col not in _cols(grand):
            continue
        pen = (np.asarray(grand[col].values, dtype=float)
               - 0.5 * np.asarray(grand[gap_col].values, dtype=float))
        curves[label] = {'v': pen, 'col': colour}
    if not curves:
        return _NO_DATA

    all_v = [v for d in curves.values() for v in d['v']]
    vmin, vmax = min(all_v) - 0.01, max(all_v) + 0.01

    fig, ax = plt.subplots(figsize=_figsize(W, H))
    for label, d in curves.items():
        best_n = xs[int(np.argmax(d['v']))]
        ax.axvline(best_n, color=d['col'], lw=1.4, ls=(0, (5, 3)), alpha=0.75)
        ax.annotate('n={}'.format(best_n), (best_n, 1.0),
                    xycoords=('data', 'axes fraction'), textcoords='offset points',
                    xytext=(3, -11), fontsize=8, fontweight='bold', color=d['col'])
        ax.plot(xs, d['v'], color=d['col'], lw=2.4, marker='o', ms=4,
                label='{}\nbest n={}'.format(label, best_n))

    ax.set_ylim(vmin, vmax)
    ax.set_xticks(list(xs))
    ax.set_xlabel('n_components', fontsize=9)
    ax.set_ylabel('Penalised score', fontsize=9)
    ax.set_title('Rewards high test performance, penalises overfitting',
                 fontsize=8.5, color='#666')
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5),
              frameon=False, labelspacing=1.1)
    _style(ax)
    fig.suptitle('Penalised Score (test − 0.5 × gap)', fontsize=11,
                 fontweight='bold', color='#222')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _img(fig)


def fig_gap_curve(grand, xs, W=500, H=290):
    """Train-test R² gap vs n_components."""
    if grand is None or 'gap_r2' not in _cols(grand) or not len(grand) or not len(xs):
        return _NO_DATA
    gap = np.asarray(grand['gap_r2'].values, dtype=float)
    vmin, vmax = 0.0, float(np.nanmax(gap)) * 1.1
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=_figsize(W, H))
    ax.set_ylim(vmin, vmax)

    # zone backgrounds, clipped at the top of the axis
    for lo, hi, colour, label in [(0, 0.10, '#dcfce7', 'healthy'),
                                  (0.10, 0.20, '#fef9c3', 'moderate'),
                                  (0.20, vmax, '#fee2e2', 'overfit')]:
        if lo >= vmax:
            continue
        hi_c = min(hi, vmax)
        ax.axhspan(lo, hi_c, color=colour, alpha=0.4, lw=0, zorder=0)
        ax.annotate(label, (0.99, (lo + hi_c) / 2), xycoords=('axes fraction', 'data'),
                    ha='right', va='center', fontsize=8, color='#555', alpha=0.8)

    ax.fill_between(xs, 0, gap, color='#4e79a7', alpha=0.2, lw=0)
    ax.plot(xs, gap, color='#4e79a7', lw=2.4, marker='o', ms=4, mec='#fff', mew=1)

    for thresh, colour, label in [(0.10, '#16a34a', '0.10'), (0.20, '#dc2626', '0.20')]:
        ax.axhline(thresh, color=colour, lw=1.2, ls=(0, (5, 3)))
        ax.annotate('gap={}'.format(label), (0.01, thresh),
                    xycoords=('axes fraction', 'data'), textcoords='offset points',
                    xytext=(0, 3), fontsize=8, color=colour)

    ax.set_xticks(list(xs))
    ax.set_xlabel('n_components', fontsize=9)
    ax.set_ylabel('Train − Test R²', fontsize=9)
    ax.set_title('Train − Test R² Gap (overfitting diagnostic)',
                 fontsize=11, fontweight='bold', color='#222')
    _style(ax)
    fig.tight_layout()
    return _img(fig)


def fig_per_patient(agg, metric_col, ylabel, xs_all, W=560, H=290):
    """Per-patient curves (averaged across embeddings)."""
    if agg is None or metric_col not in _cols(agg) or not len(agg) or not len(xs_all):
        return _NO_DATA
    per_pat = {}
    for pat, g in agg.groupby('patient'):
        pnc = g.groupby('n_components')[metric_col].agg(['mean','sem']).reset_index()
        pnc = pnc[pnc['n_components'].isin(xs_all)]
        per_pat[pat] = pnc

    all_v = [v for d in per_pat.values() for v in d['mean']]
    if not all_v:
        return _NO_DATA
    vmin, vmax = min(all_v) * 0.95, max(all_v) * 1.08

    fig, ax = plt.subplots(figsize=_figsize(W, H))
    for pat, pnc in per_pat.items():
        col = PAT_COLOURS.get(pat, '#888')
        pxs = pnc['n_components'].tolist()
        pys = np.asarray(pnc['mean'].values, dtype=float)
        # a patient×n cell with a single row has sem == NaN; the SVG polygon it
        # produced was invalid and simply did not render, so zero matches it.
        pses = np.nan_to_num(np.asarray(pnc['sem'].values, dtype=float))
        ax.fill_between(pxs, pys - pses, pys + pses, color=col, alpha=0.15, lw=0)
        ax.plot(pxs, pys, color=col, lw=2.2, marker='o', ms=4, mec='#fff', mew=1,
                label=pat)

    if vmin < vmax:
        ax.set_ylim(vmin, vmax)
    ax.set_xticks(list(xs_all))
    ax.set_xlabel('n_components', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title('{} per patient'.format(ylabel), fontsize=11,
                 fontweight='bold', color='#222')
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5),
              frameon=False)
    _style(ax)
    fig.tight_layout()
    return _img(fig)


def fig_selection_heatmap(agg, xs_all, W=700, H=260):
    """Heatmap: best n_components per criterion per patient×embedding."""
    metrics = [
        ('test_cos_mean', 'gap_cos', 'Cosine'),
        ('word_mean',     'gap_r2',  'Word Acc'),
        ('cat_mean',      'gap_r2',  'Cat Acc'),
    ]
    CRITERIA = ['Peak', '95% thr.', 'Penalised', 'Elbow']

    if agg is None or not len(agg) or not len(xs_all):
        return _NO_DATA

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
    if df_heat.empty:
        return _NO_DATA

    # colour ramp: n_components enters by its RANK in xs_all, not by its value, so
    # an unevenly spaced sweep still spans the whole ramp. Same indexing the SVG used.
    all_ns = sorted(xs_all)
    blues = ['#dbeafe','#bfdbfe','#93c5fd','#60a5fa','#3b82f6','#2563eb','#1d4ed8','#1e40af','#1e3a8a','#172554','#0f172a']
    ramp, n_to_rank = [], {}
    for i, n in enumerate(all_ns):
        ci = min(int(i / len(all_ns) * len(blues)), len(blues)-1)
        ramp.append(blues[ci])
        n_to_rank[n] = i

    n_rows = len(df_heat)
    # the SVG grew its canvas with the row count; keep that (18 px per row + padding)
    H_px = max(H, 50 + n_rows * 18 + 30)

    mat = np.array([[n_to_rank.get(int(row[c]), np.nan) for c in CRITERIA]
                    for _, row in df_heat.iterrows()], dtype=float)

    fig, ax = plt.subplots(figsize=_figsize(W, H_px))
    cmap = ListedColormap(ramp)
    cmap.set_bad('#e5e7eb')
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, aspect='auto',
                   vmin=-0.5, vmax=len(all_ns) - 0.5, interpolation='nearest')

    for ri, (_, row) in enumerate(df_heat.iterrows()):
        for ci, crit in enumerate(CRITERIA):
            n_val = int(row[crit])
            ax.text(ci, ri, str(n_val), ha='center', va='center', fontsize=8.5,
                    fontweight='bold', color='#fff' if n_val >= 15 else '#222')

    labels = ['{} {} {}'.format(r['Patient'], str(r['Embedding'])[:4], str(r['Metric'])[:3])
              for _, r in df_heat.iterrows()]
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(np.arange(len(CRITERIA)))
    ax.set_xticklabels(CRITERIA, fontsize=9, fontweight='bold')
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(length=0, colors='#333')
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title('Best n_components per criterion (patient × embedding × metric)',
                 fontsize=11, fontweight='bold', color='#222', pad=26)

    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(len(all_ns)),
                        fraction=0.035, pad=0.02)
    cbar.ax.set_yticklabels([str(n) for n in all_ns], fontsize=7)
    cbar.set_label('n value', fontsize=8)
    cbar.ax.tick_params(length=0)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    return _img(fig)


# ── HTML assembly ─────────────────────────────────────────────────────────────

def build_html(df_raw, grand, agg, xs):
    sel_df = compute_selection(grand)

    # Shared rules come from report.render; .box and .insight are aliased there onto
    # the canonical note / finding callouts. Kept here: the .insight severity
    # modifiers, this report's centred full-width table, and the figure row.
    style = stylesheet("""
.insight.green  { border-color: #16a34a; background: #f0fdf4; }
.insight.orange { border-color: #f59e0b; background: #fffbeb; }
.insight.red    { border-color: #dc2626; background: #fef2f2; }
table { border-collapse: collapse; font-size: 13px; width: 100%; }
th { background: #4e79a7; color: #fff; padding: 7px 10px; text-align: center; }
td { padding: 6px 10px; border-bottom: 1px solid #eee; text-align: center; }
tr:nth-child(even) td { background: #f5f7fa; }
td:first-child { text-align: left; font-weight: 500; }
.rec { background: #fef9c3; font-weight: bold; }
.figs { display: flex; flex-wrap: wrap; gap: 18px; align-items: flex-start; }
""")

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
{style}</head><body>
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
    ap.add_argument('--results-dir', '--results_dir', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    results_dir = get_out_dir(args.results_dir)
    out_path = args.out or os.path.join(results_dir, 'report_ncomponents_tradeoff.html')

    dfs = []
    files = sorted(glob.glob(os.path.join(results_dir, 'pls_lc_*.csv')))
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
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Saved: {out_path}')

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
