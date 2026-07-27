# -*- coding: utf-8 -*-
"""
figures_for_paper/extendability_co_trained — Open-vocabulary extendability of the
CO-TRAINED decoder, evaluated on picture- and auditory-naming trials separately.

Co-trained analogue of ``figures_for_paper/extendability/extendability_panels.py``.
A single kernel-PLS is co-trained on pooled picture + auditory trials; the per-trial
predicted GloVe vector is ranked against an OPEN gallery (stimulus words + thousands
of POS/frequency-matched distractors).  Because the decoder outputs a point in a
linguistic space, new words can be added to the gallery without retraining, and words
never seen in training can still be retrieved — here from ONE decoder that generalises
across both speech modalities.

Renders, for EACH task (picture_naming, auditory_naming), the six-panel extendability
figure a–f + supplements, PLUS a top-level picture-vs-auditory comparison figure.

Six panels per task (one topic — "extendability"):
  a  median percentile rank vs gallery size N (200-5000)               (evidence 1)
  b  top-k retrieval accuracy vs k at N=5000                           (evidence 1)
  c  in-vocab vs held-out (zero-shot) median percentile rank          (evidence 2)
  d  Wu-Palmer similarity of top-10 retrieved neighbours vs matched null (evidence 3)
  e  nDCG@100 of the neural ranking vs matched null                    (evidence 3)
  f  2D MDS (cosine) semantic-neighbourhood showcase (best participant) (illustration)

Comparison figure (00_extendability_combined_comparison): panels a–e with picture-test
and auditory-test juxtaposed per position (picture = filled ●, solid box; auditory =
open □, hatched box), sharing per-participant colours.

Inputs (all from figures_for_paper/extendability_co_trained/source_data/, written by
run_co_trained_retrieval.py + compute_extendability_data.py):
  per_patient_metrics_{task}.csv, sweep_{task}.csv, group_inference_{task}.json,
  cache_heldout_trial_percentile_by_N_{task}.csv, cache_panelf_mds_{task}.csv,
  cache_panelf_{pat}_{task}.csv, cache_qualitative_bestcases_{task}.csv,
  panelf_showcase_{task}.json

Reproduce (any env with numpy/pandas/matplotlib/scipy; reads CSVs, not project pkls):
  # (once, in the Speech env) produce co-trained predictions + metrics:
  python figures_for_paper/extendability_co_trained/run_co_trained_retrieval.py
  # (once, in the Speech env) build the panel-f/S1/S2 caches:
  python figures_for_paper/extendability_co_trained/compute_extendability_data.py
  # then render (any env):
  python figures_for_paper/extendability_co_trained/extendability_panels.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.ticker import NullLocator

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS_ROOT = os.path.dirname(HERE)                          # …/figures_for_paper
MAIN_DIR = os.path.dirname(FIGS_ROOT)                      # …/main
sys.path.insert(0, FIGS_ROOT)                              # shared figure conventions
from paper_common import display_id, assign_colors, apply_paper_style  # noqa: E402
from utils.config import ALPHA, p_stars                    # noqa: E402  (repo-wide cutoff)

FIG_DIR = HERE
SRC_DIR = os.path.join(HERE, 'source_data')

TASKS = ['picture_naming', 'auditory_naming']
TASK_LABEL = {'picture_naming': 'Picture naming', 'auditory_naming': 'Auditory naming'}
TASK_SHORT = {'picture_naming': 'picture', 'auditory_naming': 'auditory'}
HEADLINE_N = 5000
HEADLINE_VARIANT = 'matched'
KS = [1, 5, 10, 50, 100]
NS = [200, 500, 1000, 2000, 5000]
CHANCE_PCT = 0.5
SIG_ALPHA = ALPHA         # repo-wide cutoff (utils/config.py)

BLUE = 'tab:blue'
GREY = '#888888'
BOX_FACE = '#e8e8e8'
AUD_EDGE = '#c0651a'        # auditory box edge / accent in the comparison figure

apply_paper_style()


# ── Significance helpers ──────────────────────────────────────────────────────

def _stars(p):
    """p-value -> significance string (utils.config.p_stars; n.s. spelled out)."""
    return p_stars(p)


def _wilcoxon(values, chance, alternative):
    from scipy.stats import wilcoxon
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    diffs = v - chance
    if len(v) < 1 or np.allclose(diffs, 0):
        return np.nan, len(v)
    try:
        _, p = wilcoxon(diffs, alternative=alternative)
    except ValueError:
        return np.nan, len(v)
    return float(p), len(v)


def _wilcoxon_paired(a, b, alternative):
    from scipy.stats import wilcoxon
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) < 1 or np.allclose(a - b, 0):
        return np.nan, len(a)
    try:
        _, p = wilcoxon(a, b, alternative=alternative)
    except ValueError:
        return np.nan, len(a)
    return float(p), len(a)


def _sig_bracket(ax, x0, x1, y, text, color='#222222', fs=8):
    if ax.get_yscale() == 'log':
        y2 = y * 1.10
    else:
        yl = ax.get_ylim(); y2 = y + 0.03 * (yl[1] - yl[0])
    ax.plot([x0, x0, x1, x1], [y, y2, y2, y], lw=1.0, color=color, clip_on=False)
    ax.text((x0 + x1) / 2, y2, text, ha='center', va='bottom', fontsize=fs,
            color=color, clip_on=False)


def _box_points(ax, positions, data_by_pos, patients, colors, width=0.55, seed=0):
    """Boxplot (IQR + median) per position with jittered per-participant points
    (fixed colour per participant) and a black across-participant mean line."""
    color_of = {p: colors[i] for i, p in enumerate(patients)}
    ax.boxplot(data_by_pos, positions=positions, widths=width, showfliers=False,
               patch_artist=True, zorder=2,
               medianprops=dict(color='#333333', lw=1.3),
               boxprops=dict(facecolor=BOX_FACE, edgecolor='#999999', lw=0.8),
               whiskerprops=dict(color='#999999', lw=0.8),
               capprops=dict(color='#999999', lw=0.8))
    rng = _rng(seed)
    for xi, arr in zip(positions, data_by_pos):
        arr = np.asarray(arr, dtype=float)
        for pi, p in enumerate(patients):
            jx = xi + (rng.random() - 0.5) * 0.28
            ax.plot(jx, arr[pi], 'o', ms=3.0, color=color_of[p], alpha=0.85, zorder=3, mew=0)
    means = [float(np.nanmean(np.asarray(a, dtype=float))) for a in data_by_pos]
    ax.plot(positions, means, color='black', lw=2.0, marker='o', ms=4, zorder=4)
    return means


def _rng(seed=0):
    return np.random.default_rng(seed)


def _sem(a):
    a = np.asarray(a, dtype=float)
    return float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else np.nan


# ── Data loading ────────────────────────────────────────────────────────────────

def load_inputs(task, patients_order=None):
    perp = pd.read_csv(os.path.join(SRC_DIR, f'per_patient_metrics_{task}.csv'))
    sweep = pd.read_csv(os.path.join(SRC_DIR, f'sweep_{task}.csv'))
    with open(os.path.join(SRC_DIR, f'group_inference_{task}.json'), encoding='utf-8') as f:
        ginf = json.load(f)
    if patients_order is not None:
        perp = perp.set_index('patient').reindex(patients_order).reset_index()
        patients = list(patients_order)
    else:
        patients = list(perp['patient'])
    return perp, sweep, ginf, patients


def _cache(name):
    path = os.path.join(SRC_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def _showcase(task):
    path = os.path.join(SRC_DIR, f'panelf_showcase_{task}.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        return d.get('best'), d.get('extras', [])
    return None, []


def canonical_patients():
    """Consistent participant order + colours shared across both tasks and the
    comparison figure (picture-task order, any auditory-only appended)."""
    pic = pd.read_csv(os.path.join(SRC_DIR, 'per_patient_metrics_picture_naming.csv'))
    order = list(dict.fromkeys(pic['patient']))
    aud = pd.read_csv(os.path.join(SRC_DIR, 'per_patient_metrics_auditory_naming.csv'))
    order += [p for p in dict.fromkeys(aud['patient']) if p not in order]
    return order


# ── Panel a: median percentile rank vs N (box + points) ──────────────────────────

def draw_scaling(ax, sweep, patients, colors, panel_letter=None):
    sub = sweep[sweep['variant'] == HEADLINE_VARIANT]
    piv = sub.pivot(index='patient', columns='N', values='median_percentile').reindex(patients)
    xpos = np.arange(len(NS))
    data = [piv[n].to_numpy(dtype=float) for n in NS]
    _box_points(ax, xpos, data, patients, colors, seed=0)
    ax.axhline(CHANCE_PCT, ls='--', color=GREY, lw=1.0, zorder=1)
    ax.text(xpos[-1], CHANCE_PCT, ' chance', color=GREY, fontsize=6.5, va='center', ha='left')
    for xi, d in enumerate(data):
        p, _ = _wilcoxon(d, CHANCE_PCT, 'less')
        ax.text(xi, 0.4, _stars(p), ha='center', va='center', fontsize=8, color='#222222')
    ax.set_xticks(xpos); ax.set_xticklabels([str(n) for n in NS])
    ax.set_xlim(-0.6, len(NS) - 0.4)
    ax.set_ylim(0, 0.55)
    ax.set_xlabel('Gallery size $N$ (words)')
    ax.set_ylabel('Median percentile rank')
    _letter(ax, panel_letter)


# ── Panel b: top-k accuracy vs k (box + points) ─────────────────────────────────

def draw_cmc(ax, perp, patients, colors, panel_letter=None):
    xpos = np.arange(len(KS))
    data = [perp[f'top{k}_all'].to_numpy(dtype=float) for k in KS]
    ax.set_yscale('log')
    _box_points(ax, xpos, data, patients, colors, seed=1)
    chance = [k / float(HEADLINE_N) for k in KS]
    ax.plot(xpos, chance, ls='--', color=GREY, lw=1.0, marker='.', zorder=1)
    ax.text(xpos[-1], chance[-1], ' chance\n ($k/N$)', color=GREY, fontsize=6.5,
            va='bottom', ha='left')
    for xi, (d, k) in enumerate(zip(data, KS)):
        p, _ = _wilcoxon(d, k / float(HEADLINE_N), 'greater')
        top = np.nanmax(d)
        if not np.isfinite(top) or top <= 0:
            top = k / float(HEADLINE_N)
        ax.text(xi, top * 1.35, _stars(p), ha='center', va='bottom', fontsize=8, color='#222222')
    ax.set_xticks(xpos); ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlim(-0.6, len(KS) - 0.4)
    ax.set_xlabel('Rank $k$ (gallery $N$=5000)')
    ax.set_ylabel('Top-$k$ retrieval accuracy')
    _letter(ax, panel_letter)


# ── Panel c: in-vocab vs held-out ───────────────────────────────────────────────

def draw_zeroshot(ax, perp, ginf, patients, colors, panel_letter=None):
    inv = perp['median_percentile_invocab'].to_numpy(dtype=float)
    hld = perp['median_percentile_heldout'].to_numpy(dtype=float)
    ax.set_yscale('log')
    _box_points(ax, [0, 1], [inv, hld], patients, colors, width=0.5, seed=3)
    ax.axhline(CHANCE_PCT, ls='--', color=GREY, lw=1.0, zorder=1)
    ax.text(1.35, CHANCE_PCT, ' chance', color=GREY, fontsize=6.5, va='bottom', ha='left')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['In-vocab', 'Held-out\n(zero-shot)'])
    ax.set_xlim(-0.5, 1.5)
    lo = np.nanmin([np.nanmin(inv), np.nanmin(hld), 0.01])
    ax.set_ylim(max(0.004, lo * 0.6), 0.9)
    ax.set_yticks([0.01, 0.03, 0.1, 0.5])
    ax.set_yticklabels(['0.01', '0.03', '0.1', '0.5'])
    ax.get_yaxis().set_minor_locator(NullLocator())
    ax.set_ylabel('Median percentile rank')
    p_pair, _ = _wilcoxon_paired(inv, hld, 'less')
    _sig_bracket(ax, 0, 1, np.nanmax([np.nanmax(inv), np.nanmax(hld)]) * 1.3, _stars(p_pair))
    _letter(ax, panel_letter)


# ── Panels d/e: matched null vs neural (near-miss / nDCG) ────────────────────────

def _draw_null_vs_neural(ax, perp, ginf, patients, colors, obs_col, null_col,
                         group_key, ylabel, panel_letter=None):
    obs = perp[obs_col].to_numpy(dtype=float)
    null = perp[null_col].to_numpy(dtype=float)
    _box_points(ax, [0, 1], [null, obs], patients, colors, width=0.5, seed=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Matched\nnull', 'Neural\nretrieval'])
    ax.set_xlim(-0.5, 1.5)
    lo = np.nanmin([np.nanmin(null), np.nanmin(obs)]); hi = np.nanmax([np.nanmax(null), np.nanmax(obs)])
    pad = 0.06 * (hi - lo + 1e-9)
    ax.set_ylim(lo - pad, hi + 4.0 * pad)
    ax.set_ylabel(ylabel)
    gp = ginf.get(group_key, {}).get('p_value', np.nan)
    _sig_bracket(ax, 0, 1, hi + 1.4 * pad, _stars(gp))
    _letter(ax, panel_letter)


def draw_neighbours(ax, perp, ginf, patients, colors, panel_letter=None):
    _draw_null_vs_neural(ax, perp, ginf, patients, colors,
                         obs_col='graded_near_miss_sim_mean', null_col='near_miss_null_mean',
                         group_key='near_miss_vs_null',
                         ylabel='Neighbour similarity\n(Wu–Palmer)', panel_letter=panel_letter)


def draw_ndcg(ax, perp, ginf, patients, colors, panel_letter=None):
    _draw_null_vs_neural(ax, perp, ginf, patients, colors,
                         obs_col='graded_ndcg_mean', null_col='ndcg_null_mean',
                         group_key='ndcg_vs_null',
                         ylabel='nDCG@100\n(independent WordNet grade)', panel_letter=panel_letter)


# ── Panel f: 2D MDS (cosine) semantic-neighbourhood showcase ─────────────────────

def draw_mds(ax, mds, panel_letter=None):
    if mds is None:
        ax.text(0.5, 0.5, 'panel f cache missing\n(run compute_extendability_data.py)',
                ha='center', va='center', fontsize=7, color='#999999', transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([]); _letter(ax, panel_letter)
        return
    did = mds['display_id'].iloc[0]
    for grp, g in mds.groupby('trial_group'):
        pr = g[g['role'] == 'predicted']; tr = g[g['role'] == 'truth']
        if len(pr) and len(tr):
            ax.plot([pr['x'].iloc[0], tr['x'].iloc[0]], [pr['y'].iloc[0], tr['y'].iloc[0]],
                    color='#cccccc', lw=0.8, zorder=1)
    for _, r in mds[mds['role'] == 'neighbor'].iterrows():
        ax.text(r['x'], r['y'], r['label'], fontsize=5.2, color='#9a9a9a',
                ha='center', va='center', zorder=2)
    for _, r in mds[mds['role'] == 'truth'].iterrows():
        ax.plot(r['x'], r['y'], 'o', ms=3.5, color='black', zorder=4)
        ax.annotate(r['label'], (r['x'], r['y']), textcoords='offset points',
                    xytext=(0, 5), fontsize=6.8, color='black', fontweight='bold',
                    ha='center', va='bottom', zorder=6)
    for _, r in mds[mds['role'] == 'predicted'].iterrows():
        ax.plot(r['x'], r['y'], 'o', ms=3.5, color=BLUE, zorder=5)
        ax.annotate(r['label'], (r['x'], r['y']), textcoords='offset points',
                    xytext=(0, -5), fontsize=6.8, color=BLUE, fontweight='bold',
                    ha='center', va='top', zorder=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('MDS dim 1'); ax.set_ylabel('MDS dim 2')
    ax.set_title(f'Semantic neighbourhood ({did})', fontsize=8, fontweight='bold')
    xr = mds['x'].max() - mds['x'].min(); yr = mds['y'].max() - mds['y'].min()
    ax.set_xlim(mds['x'].min() - 0.08 * xr, mds['x'].max() + 0.08 * xr)
    ax.set_ylim(mds['y'].min() - 0.08 * yr, mds['y'].max() + 0.08 * yr)
    _letter(ax, panel_letter)


def _letter(ax, letter):
    if letter is None:
        return
    ax.annotate(letter, xy=(0, 1), xycoords='axes fraction',
                xytext=(-40, 14), textcoords='offset points',
                fontsize=12, fontweight='bold', va='bottom', ha='left')


def _legend_handles(patients, colors):
    h = [mlines.Line2D([], [], color=colors[i], marker='o', ls='', ms=5, label=display_id(p))
         for i, p in enumerate(patients)]
    h.append(mlines.Line2D([], [], color='black', lw=2.5, marker='o', ms=5, label='mean'))
    h.append(mlines.Line2D([], [], color=GREY, lw=1.2, ls='--', label='chance'))
    return h


# ══════════════════════════════════════════════════════════════════════════
# Comparison figure — picture vs auditory juxtaposed (panels a–e)
# ══════════════════════════════════════════════════════════════════════════

def _sweep_pivot(sweep, patients, value='median_percentile'):
    sub = sweep[sweep['variant'] == HEADLINE_VARIANT]
    piv = sub.pivot(index='patient', columns='N', values=value).reindex(patients)
    return [piv[n].to_numpy(dtype=float) for n in NS]


def _grouped_boxes(ax, base, data_pic, data_aud, patients, colors,
                   width=0.32, off=0.20, seed=0):
    """Two offset box groups per base position (picture / auditory), per-participant
    points coloured by participant (picture = filled ●, auditory = open □), and a
    black mean line per task (picture solid, auditory dashed)."""
    color_of = {p: colors[i] for i, p in enumerate(patients)}
    base = np.asarray(base, dtype=float)

    def draw(datas, positions, face, edge, hatch, marker, open_face):
        ax.boxplot(datas, positions=positions, widths=width, showfliers=False,
                   patch_artist=True, zorder=2,
                   medianprops=dict(color='#333333', lw=1.1),
                   boxprops=dict(facecolor=face, edgecolor=edge, lw=0.8, hatch=hatch),
                   whiskerprops=dict(color=edge, lw=0.8),
                   capprops=dict(color=edge, lw=0.8))
        rng = _rng(seed)
        for xi, arr in zip(positions, datas):
            arr = np.asarray(arr, dtype=float)
            for pi, p in enumerate(patients):
                jx = xi + (rng.random() - 0.5) * 0.16
                if open_face:
                    ax.plot(jx, arr[pi], marker=marker, ms=3.0, ls='',
                            mec=color_of[p], mfc='white', mew=0.9, zorder=3)
                else:
                    ax.plot(jx, arr[pi], marker=marker, ms=3.0, ls='',
                            color=color_of[p], alpha=0.9, mew=0, zorder=3)
        return [float(np.nanmean(np.asarray(a, dtype=float))) for a in datas]

    mp = draw(data_pic, base - off, BOX_FACE, '#8a8a8a', '', 'o', False)
    ma = draw(data_aud, base + off, 'white', AUD_EDGE, '////', 's', True)
    ax.plot(base - off, mp, color='black', lw=1.7, marker='o', ms=4, zorder=4)
    ax.plot(base + off, ma, color='black', lw=1.7, ls='--', marker='s', ms=4, zorder=4)
    return mp, ma


def cmp_scaling(ax, sweep_pic, sweep_aud, patients, colors, letter=None):
    base = np.arange(len(NS))
    dp = _sweep_pivot(sweep_pic, patients); da = _sweep_pivot(sweep_aud, patients)
    _grouped_boxes(ax, base, dp, da, patients, colors, seed=0)
    ax.axhline(CHANCE_PCT, ls='--', color=GREY, lw=1.0, zorder=1)
    ax.text(base[-1] + 0.35, CHANCE_PCT, ' chance', color=GREY, fontsize=6.5, va='center', ha='left')
    top = 0.72
    for xi in base:
        p, _ = _wilcoxon_paired(dp[int(xi)], da[int(xi)], 'less')
        ax.text(xi, top, _stars(p), ha='center', va='top', fontsize=7.5, color='#222222')
    ax.set_xticks(base); ax.set_xticklabels([str(n) for n in NS])
    ax.set_xlim(-0.6, len(NS) - 0.4); ax.set_ylim(0, 0.78)
    ax.set_xlabel('Gallery size $N$ (words)')
    ax.set_ylabel('Median percentile rank')
    _letter(ax, letter)


def cmp_cmc(ax, perp_pic, perp_aud, patients, colors, letter=None):
    base = np.arange(len(KS))
    dp = [perp_pic[f'top{k}_all'].to_numpy(dtype=float) for k in KS]
    da = [perp_aud[f'top{k}_all'].to_numpy(dtype=float) for k in KS]
    ax.set_yscale('log')
    _grouped_boxes(ax, base, dp, da, patients, colors, seed=1)
    chance = [k / float(HEADLINE_N) for k in KS]
    ax.plot(base, chance, ls='--', color=GREY, lw=1.0, marker='.', zorder=1)
    ax.text(base[-1], chance[-1], ' chance\n ($k/N$)', color=GREY, fontsize=6.5, va='bottom', ha='left')
    for xi, k in zip(base, KS):
        p, _ = _wilcoxon_paired(dp[int(xi)], da[int(xi)], 'greater')
        top = np.nanmax(np.concatenate([dp[int(xi)], da[int(xi)]]))
        if not np.isfinite(top) or top <= 0:
            top = k / float(HEADLINE_N)
        ax.text(xi, top * 1.6, _stars(p), ha='center', va='bottom', fontsize=7.5, color='#222222')
    ax.set_xticks(base); ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlim(-0.6, len(KS) - 0.4)
    ax.set_xlabel('Rank $k$ (gallery $N$=5000)')
    ax.set_ylabel('Top-$k$ retrieval accuracy')
    _letter(ax, letter)


def cmp_zeroshot(ax, perp_pic, perp_aud, patients, colors, letter=None):
    base = np.arange(2)
    dp = [perp_pic['median_percentile_invocab'].to_numpy(dtype=float),
          perp_pic['median_percentile_heldout'].to_numpy(dtype=float)]
    da = [perp_aud['median_percentile_invocab'].to_numpy(dtype=float),
          perp_aud['median_percentile_heldout'].to_numpy(dtype=float)]
    ax.set_yscale('log')
    _grouped_boxes(ax, base, dp, da, patients, colors, width=0.30, off=0.20, seed=3)
    ax.axhline(CHANCE_PCT, ls='--', color=GREY, lw=1.0, zorder=1)
    ax.text(1.4, CHANCE_PCT, ' chance', color=GREY, fontsize=6.5, va='bottom', ha='left')
    ax.set_xticks(base); ax.set_xticklabels(['In-vocab', 'Held-out\n(zero-shot)'])
    ax.set_xlim(-0.6, 1.6)
    allv = np.concatenate([v for v in dp + da])
    lo = np.nanmin(allv[allv > 0]) if np.any(allv > 0) else 0.01
    ax.set_ylim(max(0.004, lo * 0.6), 0.95)
    ax.set_yticks([0.01, 0.03, 0.1, 0.5])
    ax.set_yticklabels(['0.01', '0.03', '0.1', '0.5'])
    ax.get_yaxis().set_minor_locator(NullLocator())
    ax.set_ylabel('Median percentile rank')
    _letter(ax, letter)


def _cmp_null_vs_neural(ax, perp_pic, perp_aud, patients, colors, obs_col, null_col,
                        ylabel, letter=None):
    base = np.arange(2)
    dp = [perp_pic[null_col].to_numpy(dtype=float), perp_pic[obs_col].to_numpy(dtype=float)]
    da = [perp_aud[null_col].to_numpy(dtype=float), perp_aud[obs_col].to_numpy(dtype=float)]
    _grouped_boxes(ax, base, dp, da, patients, colors, width=0.30, off=0.20, seed=4)
    ax.set_xticks(base); ax.set_xticklabels(['Matched\nnull', 'Neural\nretrieval'])
    ax.set_xlim(-0.6, 1.6)
    allv = np.concatenate([v for v in dp + da])
    lo = np.nanmin(allv); hi = np.nanmax(allv); pad = 0.08 * (hi - lo + 1e-9)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel(ylabel)
    _letter(ax, letter)


def cmp_neighbours(ax, perp_pic, perp_aud, patients, colors, letter=None):
    _cmp_null_vs_neural(ax, perp_pic, perp_aud, patients, colors,
                        obs_col='graded_near_miss_sim_mean', null_col='near_miss_null_mean',
                        ylabel='Neighbour similarity\n(Wu–Palmer)', letter=letter)


def cmp_ndcg(ax, perp_pic, perp_aud, patients, colors, letter=None):
    _cmp_null_vs_neural(ax, perp_pic, perp_aud, patients, colors,
                        obs_col='graded_ndcg_mean', null_col='ndcg_null_mean',
                        ylabel='nDCG@100\n(independent WordNet grade)', letter=letter)


def _cmp_legend_handles(patients, colors):
    h = [mlines.Line2D([], [], color=colors[i], marker='o', ls='', ms=5, label=display_id(p))
         for i, p in enumerate(patients)]
    h.append(mpatches.Patch(facecolor=BOX_FACE, edgecolor='#8a8a8a', label='picture (●)'))
    h.append(mpatches.Patch(facecolor='white', edgecolor=AUD_EDGE, hatch='////', label='auditory (□)'))
    h.append(mlines.Line2D([], [], color='black', lw=2.0, label='mean'))
    h.append(mlines.Line2D([], [], color=GREY, lw=1.2, ls='--', label='chance'))
    return h


def generate_comparison(data, patients, colors):
    pic = data['picture_naming']; aud = data['auditory_naming']
    fig = plt.figure(figsize=(12.0, 7.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.36)
    top = outer[0].subgridspec(1, 2, wspace=0.24)
    bot = outer[1].subgridspec(1, 3, wspace=0.42)
    cmp_scaling(fig.add_subplot(top[0, 0]), pic['sweep'], aud['sweep'], patients, colors, letter='a')
    cmp_cmc(fig.add_subplot(top[0, 1]), pic['perp'], aud['perp'], patients, colors, letter='b')
    cmp_zeroshot(fig.add_subplot(bot[0, 0]), pic['perp'], aud['perp'], patients, colors, letter='c')
    cmp_neighbours(fig.add_subplot(bot[0, 1]), pic['perp'], aud['perp'], patients, colors, letter='d')
    cmp_ndcg(fig.add_subplot(bot[0, 2]), pic['perp'], aud['perp'], patients, colors, letter='e')
    fig.suptitle('Co-trained decoder — open-vocabulary retrieval: picture vs auditory naming',
                 fontsize=11, fontweight='bold')
    fig.legend(handles=_cmp_legend_handles(patients, colors), ncol=10, loc='lower center',
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    _save(fig, os.path.join(FIG_DIR, '00_extendability_combined_comparison'), dpi=300)


# ── Supplement 1: per-participant held-out trial distributions across N ──────────

def supp_heldout_distributions(heldout, patients, colors, task):
    if heldout is None:
        print(f"[extendability] S1 ({task}) skipped — cache missing")
        return
    color_of = {p: colors[i] for i, p in enumerate(patients)}
    ncol = 3
    nrow = int(np.ceil(len(patients) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9, 2.6 * nrow), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    xpos = np.arange(len(NS))
    rng = _rng(2)
    for ai, p in enumerate(patients):
        ax = axes[ai]
        sub = heldout[heldout['patient'] == p]
        data = [sub[sub['N'] == n]['percentile'].to_numpy(dtype=float) for n in NS]
        if any(len(d) for d in data):
            ax.boxplot([d if len(d) else [np.nan] for d in data], positions=xpos, widths=0.6,
                       showfliers=False, patch_artist=True,
                       medianprops=dict(color='#333333', lw=1.2),
                       boxprops=dict(facecolor=BOX_FACE, edgecolor='#aaaaaa', lw=0.7),
                       whiskerprops=dict(color='#aaaaaa', lw=0.7),
                       capprops=dict(color='#aaaaaa', lw=0.7))
            for xi, d in enumerate(data):
                jx = xi + (rng.random(len(d)) - 0.5) * 0.3
                ax.plot(jx, d, 'o', ms=2.2, color=color_of[p], alpha=0.55, mew=0, zorder=3)
        ax.axhline(CHANCE_PCT, ls='--', color=GREY, lw=0.8)
        ax.set_title(display_id(p), fontsize=8, fontweight='bold')
        ax.set_xticks(xpos); ax.set_xticklabels([str(n) for n in NS], fontsize=6, rotation=45)
        ax.set_ylim(-0.02, 0.72)
    for ai in range(len(patients), len(axes)):
        axes[ai].axis('off')
    fig.text(0.5, 0.015, 'Gallery size $N$ (words)', ha='center', fontsize=9)
    fig.text(0.005, 0.5, 'Per-trial percentile rank (held-out trials)', va='center',
             rotation='vertical', fontsize=9)
    fig.suptitle(f'Held-out (zero-shot) per-trial retrieval by participant — {TASK_LABEL[task]}',
                 fontsize=10, fontweight='bold')
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.97))
    _save(fig, os.path.join(FIG_DIR, f'S1_heldout_trial_distributions_{TASK_SHORT[task]}'))


# ── Supplement 2: qualitative best-case table (HTML + CSV) ───────────────────────

def supp_bestcase_table(best, task):
    if best is None:
        print(f"[extendability] S2 ({task}) skipped — cache missing")
        return
    b = best.copy()
    b.to_csv(os.path.join(SRC_DIR, f'S2_qualitative_bestcases_{TASK_SHORT[task]}.csv'), index=False)

    def _row_html(r):
        tops = []
        grades = str(r['grades']).split(';')
        for j, col in enumerate(['top1', 'top2', 'top3', 'top4', 'top5']):
            w = r[col]
            g = grades[j] if j < len(grades) else ''
            hit = (str(w).lower() == str(r['true_word']).lower())
            style = 'font-weight:bold;color:#1f6f1f;' if hit else 'color:#333;'
            tops.append(f'<td style="{style}">{w}<br><span style="color:#999;font-size:11px">{g}</span></td>')
        return (f"<tr><td>{r['display_id']}</td>"
                f"<td style='font-weight:bold'>{r['true_word']}</td>"
                f"<td style='color:#666'>{r['category']}</td>"
                f"<td style='text-align:right'>{r['rank']}</td>"
                f"<td style='text-align:right'>{r['near_miss_sim']:.3f}</td>"
                f"<td style='text-align:right'>{r['ndcg']:.3f}</td>"
                + ''.join(tops) + "</tr>")

    head = ("<th>ID</th><th>true word</th><th>category</th><th>rank</th>"
            "<th>near-miss<br>sim</th><th>nDCG@100</th>"
            "<th>top-1</th><th>top-2</th><th>top-3</th><th>top-4</th><th>top-5</th>")
    rows = '\n'.join(_row_html(r) for _, r in b.iterrows())
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Extendability (co-trained, {TASK_SHORT[task]}) — qualitative best cases</title>
<style>
 body{{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#222}}
 h2{{font-size:18px}} p{{color:#555;max-width:820px;font-size:13px}}
 table{{border-collapse:collapse;font-size:13px}}
 th,td{{border:1px solid #ddd;padding:5px 8px;vertical-align:top}}
 th{{background:#f3f3f3;text-align:left}}
</style></head><body>
<h2>Qualitative best-case retrievals — co-trained decoder ({TASK_LABEL[task]})</h2>
<p>Per participant, the words whose mean predicted embedding retrieved the most
semantically related neighbourhood (highest top-10 Wu–Palmer near-miss similarity),
one per semantic category, from a single decoder co-trained on picture + auditory
trials. Each cell shows a retrieved word and its independent WordNet Wu–Palmer
similarity to the true word; green bold marks the exact true word.</p>
<table><thead><tr>{head}</tr></thead><tbody>
{rows}
</tbody></table></body></html>"""
    with open(os.path.join(FIG_DIR, f'S2_qualitative_bestcases_{TASK_SHORT[task]}.html'), 'w',
              encoding='utf-8') as f:
        f.write(html)
    print(f"[extendability] S2 ({task}) -> {len(b)} best-case rows")


# ── Source data ─────────────────────────────────────────────────────────────────

def write_source_data(perp, sweep, ginf, patients, task):
    os.makedirs(SRC_DIR, exist_ok=True)
    did = {p: display_id(p) for p in patients}
    sfx = TASK_SHORT[task]

    s = sweep.copy(); s.insert(0, 'display_id', s['patient'].map(did))
    keep = ['display_id', 'patient', 'task', 'variant', 'N',
            'median_percentile', 'top1', 'top5', 'top10', 'top50', 'top100',
            'median_rank', 'chance_median_percentile']
    s[[c for c in keep if c in s.columns]].to_csv(
        os.path.join(SRC_DIR, f'panel_a_sweep_per_participant_{sfx}.csv'), index=False)

    rows = []
    sub = sweep[sweep['variant'] == HEADLINE_VARIANT]
    for n in NS:
        vals = sub[sub['N'] == n]['median_percentile'].values.astype(float)
        pv, _ = _wilcoxon(vals, CHANCE_PCT, 'less')
        rows.append(dict(N=n, variant=HEADLINE_VARIANT,
                         median_percentile_mean=float(np.nanmean(vals)),
                         median_percentile_sem=_sem(vals),
                         wilcoxon_p_vs_chance=pv, sig=_stars(pv),
                         chance_median_percentile=CHANCE_PCT))
    pd.DataFrame(rows).to_csv(os.path.join(SRC_DIR, f'panel_a_group_mean_sem_{sfx}.csv'), index=False)

    rows = []
    for p in patients:
        r = perp[perp['patient'] == p].iloc[0]
        d = dict(display_id=did[p], patient=p, N=HEADLINE_N, n_trials=int(r['n_trials']),
                 median_rank=float(r['median_rank_all']))
        for k in KS:
            d[f'top{k}'] = float(r[f'top{k}_all']); d[f'chance_top{k}'] = k / float(HEADLINE_N)
        rows.append(d)
    dfb = pd.DataFrame(rows)
    dfb.to_csv(os.path.join(SRC_DIR, f'panel_b_cmc_N5000_{sfx}.csv'), index=False)
    sig_rows = []
    for k in KS:
        pv, _ = _wilcoxon(dfb[f'top{k}'].values, k / float(HEADLINE_N), 'greater')
        sig_rows.append(dict(k=k, chance=k / float(HEADLINE_N),
                             topk_mean=float(dfb[f'top{k}'].mean()), wilcoxon_p=pv, sig=_stars(pv)))
    pd.DataFrame(sig_rows).to_csv(os.path.join(SRC_DIR, f'panel_b_significance_{sfx}.csv'), index=False)

    c = perp[['patient', 'n_trials', 'n_held_out',
              'median_percentile_all', 'median_percentile_invocab', 'median_percentile_heldout',
              'perm_p_median_percentile_all']].copy()
    c.insert(0, 'display_id', c['patient'].map(did))
    c['chance_median_percentile'] = CHANCE_PCT
    c.rename(columns={'perm_p_median_percentile_all': 'perm_p_all'}, inplace=True)
    c.to_csv(os.path.join(SRC_DIR, f'panel_c_zeroshot_{sfx}.csv'), index=False)

    d = perp[['patient', 'graded_near_miss_sim_mean', 'near_miss_null_mean',
              'perm_p_near_miss', 'category_hit_at_k']].copy()
    d.insert(0, 'display_id', d['patient'].map(did))
    d['near_miss_delta'] = d['graded_near_miss_sim_mean'] - d['near_miss_null_mean']
    d.rename(columns={'graded_near_miss_sim_mean': 'near_miss_obs',
                      'near_miss_null_mean': 'near_miss_null'}, inplace=True)
    d.to_csv(os.path.join(SRC_DIR, f'panel_d_semantic_neighbours_{sfx}.csv'), index=False)

    e = perp[['patient', 'graded_ndcg_mean', 'ndcg_null_mean', 'perm_p_ndcg']].copy()
    e.insert(0, 'display_id', e['patient'].map(did))
    e['ndcg_delta'] = e['graded_ndcg_mean'] - e['ndcg_null_mean']
    e.rename(columns={'graded_ndcg_mean': 'ndcg_obs', 'ndcg_null_mean': 'ndcg_null'}, inplace=True)
    e.to_csv(os.path.join(SRC_DIR, f'panel_e_ndcg_{sfx}.csv'), index=False)

    grows = []
    for key in ['median_percentile_all', 'median_percentile_invocab', 'median_percentile_heldout']:
        if key not in ginf:
            continue
        g = ginf[key]
        grows.append(dict(metric=key, n=g['n'], median=g['median'], chance=g['chance'],
                          ci_mean=g['ci_mean'], ci_lo=g['ci_lo'], ci_hi=g['ci_hi'],
                          wilcoxon_p=g['p_value'], test='Wilcoxon signed-rank vs chance (one-sided)'))
    for key, lab in [('near_miss_vs_null', 'near_miss_obs_minus_null'),
                     ('ndcg_vs_null', 'ndcg_obs_minus_null')]:
        if key in ginf:
            nm = ginf[key]
            grows.append(dict(metric=lab, n=nm['n'], median=nm['median'], chance=nm['chance'],
                              ci_mean=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                              wilcoxon_p=nm['p_value'], test='Wilcoxon signed-rank vs 0 (one-sided)'))
    if 'ndcg' in ginf:
        nd = ginf['ndcg']
        grows.append(dict(metric='ndcg_at_100', n=nd['n'], median=np.nan, chance=np.nan,
                          ci_mean=nd['mean'], ci_lo=nd['lo'], ci_hi=nd['hi'],
                          wilcoxon_p=np.nan, test='bootstrap 95% CI (descriptive)'))
    pd.DataFrame(grows).to_csv(os.path.join(SRC_DIR, f'group_inference_{sfx}.csv'), index=False)


def write_comparison_source_data(data, patients):
    did = {p: display_id(p) for p in patients}
    rows = []
    for task in TASKS:
        perp = data[task]['perp']; sweep = data[task]['sweep']
        sub = sweep[sweep['variant'] == HEADLINE_VARIANT]
        for p in patients:
            r = perp[perp['patient'] == p].iloc[0]
            piv = sub[sub['patient'] == p].set_index('N')['median_percentile']
            row = dict(display_id=did[p], patient=p, task=task,
                       median_percentile_all=float(r['median_percentile_all']),
                       median_percentile_invocab=float(r['median_percentile_invocab']),
                       median_percentile_heldout=float(r['median_percentile_heldout']),
                       near_miss_obs=float(r['graded_near_miss_sim_mean']),
                       near_miss_null=float(r['near_miss_null_mean']),
                       ndcg_obs=float(r['graded_ndcg_mean']),
                       ndcg_null=float(r['ndcg_null_mean']))
            for k in KS:
                row[f'top{k}'] = float(r[f'top{k}_all'])
            for n in NS:
                row[f'median_percentile_N{n}'] = float(piv.get(n, np.nan))
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(SRC_DIR, 'comparison_picture_vs_auditory.csv'), index=False)

    # paired picture-vs-auditory tests per N (median %rank) and per k (top-k)
    trows = []
    pic = data['picture_naming']; aud = data['auditory_naming']
    ps = pic['sweep'][pic['sweep']['variant'] == HEADLINE_VARIANT]
    as_ = aud['sweep'][aud['sweep']['variant'] == HEADLINE_VARIANT]
    for n in NS:
        vp = ps[ps['N'] == n].set_index('patient')['median_percentile'].reindex(patients).values.astype(float)
        va = as_[as_['N'] == n].set_index('patient')['median_percentile'].reindex(patients).values.astype(float)
        pv, _ = _wilcoxon_paired(vp, va, 'less')
        trows.append(dict(metric='median_percentile', at=f'N={n}', picture_mean=float(np.nanmean(vp)),
                          auditory_mean=float(np.nanmean(va)), wilcoxon_p_pic_lt_aud=pv, sig=_stars(pv)))
    for k in KS:
        vp = pic['perp'][f'top{k}_all'].to_numpy(dtype=float)
        va = aud['perp'][f'top{k}_all'].to_numpy(dtype=float)
        pv, _ = _wilcoxon_paired(vp, va, 'greater')
        trows.append(dict(metric='topk', at=f'k={k}', picture_mean=float(np.nanmean(vp)),
                          auditory_mean=float(np.nanmean(va)), wilcoxon_p_pic_gt_aud=pv, sig=_stars(pv)))
    pd.DataFrame(trows).to_csv(os.path.join(SRC_DIR, 'comparison_paired_tests.csv'), index=False)


# ── Caption ─────────────────────────────────────────────────────────────────────

CAPTION = """# Figure caption — Extendability of the CO-TRAINED regression-and-retrieval decoder

Extendability of a single decoder co-trained on pooled picture- and auditory-naming trials,
evaluated on {TASK} trials ({N} participants with both tasks: NUEx041, NUEx044, NUEx045, NUEx038,
NUEx031, NUEx036). The kernel-PLS decoder (Nystroem RBF kernel followed by PLS regression onto
GloVe word-embedding targets) is fit on the intersection of the two tasks' electrodes and predicts
an embedding per trial; the predicted vector is ranked by cosine similarity against an open word
gallery of {HN} words (the stimulus words plus POS- and frequency-matched distractors never
presented), and the rank of the true word is the score. Predictions are out-of-fold; a fraction
of the unique words across BOTH tasks is held entirely out of training in either modality
(zero-shot), so an in-vocab word may have been seen only cross-modally. Chance: median percentile
rank 0.5; top-k accuracy k/N. **a** Median percentile rank (rank/N; lower is better) versus gallery
size N (200–5000 words); box, interquartile range and median across participants; coloured points,
participants; bold black, mean; stars, Wilcoxon versus chance per N. **b** Top-k retrieval accuracy
versus rank k at N=5000 (log y; dashed, chance k/N; stars, Wilcoxon versus chance per k).
**c** Median percentile rank at N=5000 for words seen in training (in-vocab) versus held entirely
out (zero-shot); bracket, paired Wilcoxon. **d** Wu–Palmer similarity between the true word and its
top-10 retrieved neighbours, matched null versus neural (WordNet grade independent of the GloVe
decode target). **e** nDCG@100 of the neural ranking versus the matched permutation null. In
**d**,**e**: bracket, group Wilcoxon of the observed-minus-null difference. **f** 2D MDS (cosine)
of a best-participant semantic-neighbourhood showcase: the predicted word (blue, bold; top-retrieved
gallery word at its own embedding) beside the ground-truth word (black, bold) and their nearest
neighbours (grey). In **a**–**e**: coloured points, participants (fixed colour each); bold black,
mean; dashed grey, chance. Group tests are Wilcoxon signed-rank. Participants identified by display
ID (NUEx###). N={N}. Auditory naming has few trials and few repeated words, so auditory panels are
noisier and closer to chance — expected for the weaker modality. A companion comparison figure
juxtaposes picture- versus auditory-test performance for panels a–e. Supplements: S1, per-participant
held-out per-trial percentile distributions across N; S2, qualitative best-case retrievals; S3–S4,
semantic-neighbourhood showcases for further participants.
"""


def write_caption(patients, task):
    txt = CAPTION.format(N=len(patients), HN=HEADLINE_N, TASK=TASK_LABEL[task].lower())
    with open(os.path.join(FIG_DIR, f'caption_{TASK_SHORT[task]}.md'), 'w',
              encoding='utf-8', newline='\n') as f:
        f.write(txt)


# ── Orchestration ───────────────────────────────────────────────────────────────

def _save(fig, stem, dpi=200):
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    fig.savefig(stem + '.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def generate_task(task, patients, colors):
    perp, sweep, ginf, _ = load_inputs(task, patients_order=patients)
    sfx = TASK_SHORT[task]
    mds = _cache(f'cache_panelf_mds_{task}.csv')
    heldout = _cache(f'cache_heldout_trial_percentile_by_N_{task}.csv')
    best = _cache(f'cache_qualitative_bestcases_{task}.csv')

    specs = [
        (f'01_scaling_median_percentile_{sfx}', (4.3, 3.4), lambda ax: draw_scaling(ax, sweep, patients, colors)),
        (f'02_cmc_N5000_{sfx}', (4.3, 3.4), lambda ax: draw_cmc(ax, perp, patients, colors)),
        (f'03_zeroshot_invocab_vs_heldout_{sfx}', (4.0, 3.4), lambda ax: draw_zeroshot(ax, perp, ginf, patients, colors)),
        (f'04_semantic_neighbours_{sfx}', (4.0, 3.4), lambda ax: draw_neighbours(ax, perp, ginf, patients, colors)),
        (f'05_ndcg_vs_null_{sfx}', (4.0, 3.4), lambda ax: draw_ndcg(ax, perp, ginf, patients, colors)),
        (f'06_mds_neighbourhood_{sfx}', (5.2, 4.6), lambda ax: draw_mds(ax, mds)),
    ]
    for stem, size, fn in specs:
        fig, ax = plt.subplots(figsize=size); fn(ax)
        fig.tight_layout(); _save(fig, os.path.join(FIG_DIR, stem))

    fig = plt.figure(figsize=(12.0, 7.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.36)
    top = outer[0].subgridspec(1, 2, wspace=0.24)
    bot = outer[1].subgridspec(1, 4, width_ratios=[1, 1, 1, 1.9], wspace=0.42)
    draw_scaling(fig.add_subplot(top[0, 0]), sweep, patients, colors, panel_letter='a')
    draw_cmc(fig.add_subplot(top[0, 1]), perp, patients, colors, panel_letter='b')
    draw_zeroshot(fig.add_subplot(bot[0, 0]), perp, ginf, patients, colors, panel_letter='c')
    draw_neighbours(fig.add_subplot(bot[0, 1]), perp, ginf, patients, colors, panel_letter='d')
    draw_ndcg(fig.add_subplot(bot[0, 2]), perp, ginf, patients, colors, panel_letter='e')
    draw_mds(fig.add_subplot(bot[0, 3]), mds, panel_letter='f')
    fig.suptitle(f'Co-trained decoder — open-vocabulary retrieval ({TASK_LABEL[task]})',
                 fontsize=11, fontweight='bold')
    fig.legend(handles=_legend_handles(patients, colors), ncol=8, loc='lower center',
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    _save(fig, os.path.join(FIG_DIR, f'00_extendability_combined_{sfx}'), dpi=300)

    supp_heldout_distributions(heldout, patients, colors, task)
    supp_bestcase_table(best, task)
    _, extras = _showcase(task)
    supp_labels = ['S3', 'S4', 'S5']
    for i, pat in enumerate(extras):
        df = _cache(f'cache_panelf_{pat}_{task}.csv')
        if df is None:
            print(f"[extendability] panel-f supp {pat} ({task}) skipped — cache missing")
            continue
        slabel = supp_labels[i] if i < len(supp_labels) else f'S{5 + i}'
        figx, axx = plt.subplots(figsize=(5.2, 4.6)); draw_mds(axx, df)
        figx.tight_layout()
        _save(figx, os.path.join(FIG_DIR, f'{slabel}_mds_neighbourhood_{pat}_{sfx}'))

    write_source_data(perp, sweep, ginf, patients, task)
    write_caption(patients, task)

    print(f"[{task}] participants: {[display_id(p) for p in patients]}")
    if 'median_percentile_all' in ginf:
        g = ginf['median_percentile_all']
        print(f"  median %rank all median={g['median']:.4f} Wilcoxon p={g['p_value']:.2e}")
    return dict(perp=perp, sweep=sweep, ginf=ginf)


def generate():
    os.makedirs(SRC_DIR, exist_ok=True)
    patients = canonical_patients()
    colors = assign_colors(patients)
    data = {}
    for task in TASKS:
        print(f"\n===== rendering {task} =====")
        data[task] = generate_task(task, patients, colors)

    print("\n===== rendering comparison =====")
    generate_comparison(data, patients, colors)
    write_comparison_source_data(data, patients)

    print("\n[extendability_co_trained] figures + captions ->", FIG_DIR)
    print("[extendability_co_trained] source data        ->", SRC_DIR)


if __name__ == '__main__':
    generate()
