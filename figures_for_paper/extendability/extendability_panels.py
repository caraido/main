# -*- coding: utf-8 -*-
"""
figures_for_paper/extendability — Open-vocabulary / zero-shot extendability panels.

Paper-figure generator for the *extendability* of the regression-and-retrieval
decoder. The kernel-PLS decoder predicts a GloVe embedding per trial; the
predicted vector is ranked by cosine similarity against an OPEN word gallery (the
stimulus words plus thousands of never-presented, POS/frequency-matched
distractors). Because the decoder outputs a point in a linguistic space rather
than a class label, new words can be added to the retrieval gallery without any
retraining, and words never seen in training can still be retrieved.

Six panels (one topic — "extendability"):
  a  median percentile rank vs gallery size N (200-5000), distribution over
     participants (box + points) with the mean trend                (evidence 1)
  b  top-k retrieval accuracy vs k at N=5000, distribution over participants
     (box + points)                                                 (evidence 1)
  c  in-vocab vs held-out (zero-shot) median percentile rank         (evidence 2)
  d  Wu-Palmer similarity of the top-10 retrieved neighbours vs a matched null
     (predictions land on semantically related words)               (evidence 3)
  e  nDCG@100 of the neural ranking vs the matched null (whole-list semantic
     organisation, independent WordNet grade)                       (evidence 3)
  f  2D MDS (cosine) of a best-participant showcase: predicted words land among the
     ground-truth word and its near-synonyms                        (illustration)

Supplements (NOT in the combined main figure):
  S1  per-participant held-out trial percentile distributions across N (12 panels)
  S2  qualitative best-case retrievals per participant (HTML + CSV)

Inputs (already computed; this script only re-plots — it does NOT re-run the heavy
permutation pipeline):
  main/figures/open_vocab_retrieval/source_data/
    per_patient_metrics_picture_naming.csv   (N=5000 headline metrics per patient)
    sweep_picture_naming.csv                 (per-patient N x variant sweep)
    group_inference_picture_naming.json      (group Wilcoxon vs chance + CIs)
  figures_for_paper/extendability/source_data/   (from compute_extendability_data.py)
    cache_heldout_trial_percentile_by_N.csv  (supp S1)
    cache_panelf_mds.csv                     (panel f)
    cache_qualitative_bestcases.csv          (supp S2)

Reproduce (any env with numpy/pandas/matplotlib/scipy; reads CSVs, not project pkls):
  # (once, in the Speech env) build the caches for panels e/f + supplements:
  python figures_for_paper/extendability/compute_extendability_data.py
  # then render:
  python figures_for_paper/extendability/extendability_panels.py
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
from matplotlib.ticker import NullLocator

# Editable-text vector output (house rule)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS_ROOT = os.path.dirname(HERE)                          # …/figures_for_paper
MAIN_DIR = os.path.dirname(FIGS_ROOT)                      # …/main
sys.path.insert(0, FIGS_ROOT)                              # shared figure conventions
from paper_common import display_id, assign_colors, apply_paper_style  # noqa: E402

OPENVOCAB_SRC = os.path.join(MAIN_DIR, 'figures', 'open_vocab_retrieval', 'source_data')
FIG_DIR = HERE
SRC_DIR = os.path.join(HERE, 'source_data')

TASK = 'picture_naming'
HEADLINE_N = 5000
HEADLINE_VARIANT = 'matched'
KS = [1, 5, 10, 50, 100]
NS = [200, 500, 1000, 2000, 5000]
CHANCE_PCT = 0.5
SIG_ALPHA = 0.05          # threshold for a per-participant effect drawn as "significant"
# Supplementary panel-f showcases (not in the combined figure): (patient, supp label).
# RB/AA are S3 (the two next-best after VB); WBH is added as S4.
PANELF_SUPP = [('RB', 'S3'), ('AA', 'S3'), ('WBH', 'S4')]

BLUE = 'tab:blue'
GREY = '#888888'
BOX_FACE = '#e8e8e8'

apply_paper_style()


# ── Significance helpers ──────────────────────────────────────────────────────

def _stars(p):
    """p-value -> significance string (house convention, n.s. spelled out)."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return 'n.s.'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


def _wilcoxon(values, chance, alternative):
    """One-sided Wilcoxon signed-rank of `values` vs a chance constant.
    Returns (p_value, n). Mirrors stats.wilcoxon_vs_chance without importing the
    heavy pipeline package."""
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
    """One-sided paired Wilcoxon signed-rank between two per-participant vectors."""
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
    """Draw a significance bracket spanning [x0,x1] with a centred label, working on
    both linear and log y-axes (tick height is multiplicative on a log axis)."""
    if ax.get_yscale() == 'log':
        y2 = y * 1.10
    else:
        yl = ax.get_ylim(); y2 = y + 0.03 * (yl[1] - yl[0])
    ax.plot([x0, x0, x1, x1], [y, y2, y2, y], lw=1.0, color=color, clip_on=False)
    ax.text((x0 + x1) / 2, y2, text, ha='center', va='bottom', fontsize=fs,
            color=color, clip_on=False)


def _box_points(ax, positions, data_by_pos, patients, colors, width=0.55, seed=0):
    """Boxplot (IQR + median) per position with jittered per-participant points
    (fixed colour per participant) and a black across-participant mean line.
    Shared by panels a, b, c, d, e. ``data_by_pos[i]`` is aligned to ``patients``."""
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


# ── Data loading ────────────────────────────────────────────────────────────────

def load_inputs():
    perp = pd.read_csv(os.path.join(OPENVOCAB_SRC, f'per_patient_metrics_{TASK}.csv'))
    sweep = pd.read_csv(os.path.join(OPENVOCAB_SRC, f'sweep_{TASK}.csv'))
    with open(os.path.join(OPENVOCAB_SRC, f'group_inference_{TASK}.json'), encoding='utf-8') as f:
        ginf = json.load(f)
    patients = list(perp['patient'])
    return perp, sweep, ginf, patients


def _cache(name):
    path = os.path.join(SRC_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def _sem(a):
    a = np.asarray(a, dtype=float)
    return float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else np.nan


def _rng(seed=0):
    return np.random.default_rng(seed)


# ── Panel a: median percentile rank vs N (box + points) ──────────────────────────

def draw_scaling(ax, sweep, patients, colors, panel_letter=None):
    """a — distribution of median percentile rank (rank/N; lower=better) over
    participants at each gallery size N, with the across-participant mean trend."""
    sub = sweep[sweep['variant'] == HEADLINE_VARIANT]
    piv = sub.pivot(index='patient', columns='N', values='median_percentile').reindex(patients)
    xpos = np.arange(len(NS))
    data = [piv[n].to_numpy(dtype=float) for n in NS]
    _box_points(ax, xpos, data, patients, colors, seed=0)
    ax.axhline(CHANCE_PCT, ls='--', color=GREY, lw=1.0, zorder=1)
    ax.text(xpos[-1], CHANCE_PCT, ' chance', color=GREY, fontsize=6.5, va='center', ha='left')
    # significance vs chance (Wilcoxon, one-sided less) at y≈0.4
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
    """b — distribution of top-k retrieval accuracy over participants at each k
    (N=5000), matching panel a's box+points fashion."""
    xpos = np.arange(len(KS))
    data = [perp[f'top{k}_all'].to_numpy(dtype=float) for k in KS]
    ax.set_yscale('log')
    _box_points(ax, xpos, data, patients, colors, seed=1)
    # chance = k / N
    chance = [k / float(HEADLINE_N) for k in KS]
    ax.plot(xpos, chance, ls='--', color=GREY, lw=1.0, marker='.', zorder=1)
    ax.text(xpos[-1], chance[-1], ' chance\n ($k/N$)', color=GREY, fontsize=6.5,
            va='bottom', ha='left')
    # significance vs chance k/N (Wilcoxon greater), above each box
    for xi, (d, k) in enumerate(zip(data, KS)):
        p, _ = _wilcoxon(d, k / float(HEADLINE_N), 'greater')
        ax.text(xi, np.nanmax(d) * 1.35, _stars(p), ha='center', va='bottom', fontsize=8, color='#222222')
    ax.set_xticks(xpos); ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlim(-0.6, len(KS) - 0.4)
    ax.set_xlabel('Rank $k$ (gallery $N$=5000)')
    ax.set_ylabel('Top-$k$ retrieval accuracy')
    _letter(ax, panel_letter)


# ── Panel c: in-vocab vs held-out ───────────────────────────────────────────────

def draw_zeroshot(ax, perp, ginf, patients, colors, panel_letter=None):
    """c — in-vocab vs held-out (zero-shot) median percentile rank; box + points."""
    inv = perp['median_percentile_invocab'].to_numpy(dtype=float)
    hld = perp['median_percentile_heldout'].to_numpy(dtype=float)
    ax.set_yscale('log')
    _box_points(ax, [0, 1], [inv, hld], patients, colors, width=0.5, seed=3)
    ax.axhline(CHANCE_PCT, ls='--', color=GREY, lw=1.0, zorder=1)
    ax.text(1.35, CHANCE_PCT, ' chance', color=GREY, fontsize=6.5, va='bottom', ha='left')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['In-vocab', 'Held-out\n(zero-shot)'])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0.005, 0.9)
    ax.set_yticks([0.01, 0.03, 0.1, 0.5])
    ax.set_yticklabels(['0.01', '0.03', '0.1', '0.5'])
    ax.get_yaxis().set_minor_locator(NullLocator())
    ax.set_ylabel('Median percentile rank')
    # bracket = in-vocab vs held-out (paired Wilcoxon, in-vocab < held-out)
    p_pair, _ = _wilcoxon_paired(inv, hld, 'less')
    _sig_bracket(ax, 0, 1, max(inv.max(), hld.max()) * 1.3, _stars(p_pair))
    _letter(ax, panel_letter)


# ── Panels d/e: matched null vs neural (near-miss / nDCG) ────────────────────────

def _draw_null_vs_neural(ax, perp, ginf, patients, colors, obs_col, null_col,
                         group_key, ylabel, panel_letter=None):
    """Shared layout for panel d (near-miss sim) and panel e (nDCG@100): two boxes
    matched-null vs neural retrieval with per-participant points, black mean line,
    and a group-level bracket + stars (Wilcoxon of observed-minus-null)."""
    obs = perp[obs_col].to_numpy(dtype=float)
    null = perp[null_col].to_numpy(dtype=float)
    _box_points(ax, [0, 1], [null, obs], patients, colors, width=0.5, seed=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Matched\nnull', 'Neural\nretrieval'])
    ax.set_xlim(-0.5, 1.5)
    lo = min(null.min(), obs.min()); hi = max(null.max(), obs.max())
    pad = 0.06 * (hi - lo + 1e-9)
    ax.set_ylim(lo - pad, hi + 4.0 * pad)
    ax.set_ylabel(ylabel)
    gp = ginf.get(group_key, {}).get('p_value', np.nan)
    _sig_bracket(ax, 0, 1, hi + 1.4 * pad, _stars(gp))
    _letter(ax, panel_letter)


def draw_neighbours(ax, perp, ginf, patients, colors, panel_letter=None):
    """d — top-10 neighbour Wu-Palmer similarity: matched null vs neural retrieval."""
    _draw_null_vs_neural(ax, perp, ginf, patients, colors,
                         obs_col='graded_near_miss_sim_mean', null_col='near_miss_null_mean',
                         group_key='near_miss_vs_null',
                         ylabel='Neighbour similarity\n(Wu–Palmer)', panel_letter=panel_letter)


def draw_ndcg(ax, perp, ginf, patients, colors, panel_letter=None):
    """e — nDCG@100 of the neural ranking vs its matched permutation null."""
    _draw_null_vs_neural(ax, perp, ginf, patients, colors,
                         obs_col='graded_ndcg_mean', null_col='ndcg_null_mean',
                         group_key='ndcg_vs_null',
                         ylabel='nDCG@100\n(independent WordNet grade)', panel_letter=panel_letter)


# ── Panel f: 2D MDS (cosine) semantic-neighbourhood showcase ─────────────────────

def draw_mds(ax, mds, panel_letter=None):
    """f — MDS (cosine) of a best-participant showcase: each predicted word (blue,
    bold, placed at its own GloVe vector) sits beside the ground-truth word (black,
    bold) and their shared near-synonym neighbours (grey)."""
    if mds is None:
        ax.text(0.5, 0.5, 'panel f cache missing\n(run compute_extendability_data.py)',
                ha='center', va='center', fontsize=7, color='#999999', transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([]); _letter(ax, panel_letter)
        return
    did = mds['display_id'].iloc[0]
    # faint predicted -> truth connector per showcase group
    for grp, g in mds.groupby('trial_group'):
        pr = g[g['role'] == 'predicted']; tr = g[g['role'] == 'truth']
        if len(pr) and len(tr):
            ax.plot([pr['x'].iloc[0], tr['x'].iloc[0]], [pr['y'].iloc[0], tr['y'].iloc[0]],
                    color='#cccccc', lw=0.8, zorder=1)
    # peripheral neighbours (grey text)
    for _, r in mds[mds['role'] == 'neighbor'].iterrows():
        ax.text(r['x'], r['y'], r['label'], fontsize=5.2, color='#9a9a9a',
                ha='center', va='center', zorder=2)
    # ground-truth words (black, bold) — marker + label offset ABOVE the point
    for _, r in mds[mds['role'] == 'truth'].iterrows():
        ax.plot(r['x'], r['y'], 'o', ms=3.5, color='black', zorder=4)
        ax.annotate(r['label'], (r['x'], r['y']), textcoords='offset points',
                    xytext=(0, 5), fontsize=6.8, color='black', fontweight='bold',
                    ha='center', va='bottom', zorder=6)
    # predicted words (blue, bold) — marker + label offset BELOW the point, so a
    # near-identical prediction (e.g. spring/fall) stays legible instead of overlapping
    for _, r in mds[mds['role'] == 'predicted'].iterrows():
        ax.plot(r['x'], r['y'], 'o', ms=3.5, color=BLUE, zorder=5)
        ax.annotate(r['label'], (r['x'], r['y']), textcoords='offset points',
                    xytext=(0, -5), fontsize=6.8, color=BLUE, fontweight='bold',
                    ha='center', va='top', zorder=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('MDS dim 1'); ax.set_ylabel('MDS dim 2')
    ax.set_title(f'Semantic neighbourhood ({did})', fontsize=8, fontweight='bold')
    # pad limits so edge labels are not clipped
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


# ── Supplement 1: per-participant held-out trial distributions across N ──────────

def supp_heldout_distributions(heldout, patients, colors):
    if heldout is None:
        print("[extendability] S1 skipped — cache_heldout_trial_percentile_by_N.csv missing")
        return
    color_of = {p: colors[i] for i, p in enumerate(patients)}
    ncol = 4
    nrow = int(np.ceil(len(patients) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.6 * nrow), sharex=True, sharey=True)
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
        ax.set_ylim(-0.02, 0.62)
    for ai in range(len(patients), len(axes)):
        axes[ai].axis('off')
    fig.text(0.5, 0.015, 'Gallery size $N$ (words)', ha='center', fontsize=9)
    fig.text(0.005, 0.5, 'Per-trial percentile rank (held-out trials)', va='center',
             rotation='vertical', fontsize=9)
    fig.suptitle('Held-out (zero-shot) per-trial retrieval distributions by participant',
                 fontsize=10, fontweight='bold')
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.97))
    _save(fig, os.path.join(FIG_DIR, 'S1_heldout_trial_distributions'))


# ── Supplement 2: qualitative best-case table (HTML + CSV) ───────────────────────

def supp_bestcase_table(best):
    if best is None:
        print("[extendability] S2 skipped — cache_qualitative_bestcases.csv missing")
        return
    b = best.copy()
    b.to_csv(os.path.join(SRC_DIR, 'S2_qualitative_bestcases.csv'), index=False)

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
<title>Extendability — qualitative best cases</title>
<style>
 body{{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#222}}
 h2{{font-size:18px}} p{{color:#555;max-width:820px;font-size:13px}}
 table{{border-collapse:collapse;font-size:13px}}
 th,td{{border:1px solid #ddd;padding:5px 8px;vertical-align:top}}
 th{{background:#f3f3f3;text-align:left}}
</style></head><body>
<h2>Qualitative best-case retrievals (picture naming)</h2>
<p>Per participant, the words whose mean predicted embedding retrieved the most
semantically related neighbourhood (highest top-10 Wu–Palmer near-miss similarity),
one per semantic category. Each cell shows a retrieved word and its independent
WordNet Wu–Palmer similarity to the true word; green bold marks the exact true word.</p>
<table><thead><tr>{head}</tr></thead><tbody>
{rows}
</tbody></table></body></html>"""
    with open(os.path.join(FIG_DIR, 'S2_qualitative_bestcases.html'), 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[extendability] S2 -> {len(b)} best-case rows")


# ── Source data ─────────────────────────────────────────────────────────────────

def write_source_data(perp, sweep, ginf, patients):
    os.makedirs(SRC_DIR, exist_ok=True)
    did = {p: display_id(p) for p in patients}

    # Panel a + robustness: full N x variant sweep, per participant.
    s = sweep.copy(); s.insert(0, 'display_id', s['patient'].map(did))
    keep = ['display_id', 'patient', 'task', 'variant', 'N',
            'median_percentile', 'top1', 'top5', 'top10', 'top50', 'top100',
            'median_rank', 'chance_median_percentile']
    s[[c for c in keep if c in s.columns]].to_csv(
        os.path.join(SRC_DIR, 'panel_a_sweep_per_participant.csv'), index=False)

    # Panel a group mean±sem + per-N Wilcoxon vs chance.
    rows = []
    sub = sweep[sweep['variant'] == HEADLINE_VARIANT]
    for n in NS:
        vals = sub[sub['N'] == n]['median_percentile'].values.astype(float)
        pv, _ = _wilcoxon(vals, CHANCE_PCT, 'less')
        rows.append(dict(N=n, variant=HEADLINE_VARIANT,
                         median_percentile_mean=float(np.mean(vals)),
                         median_percentile_sem=_sem(vals),
                         wilcoxon_p_vs_chance=pv, sig=_stars(pv),
                         chance_median_percentile=CHANCE_PCT))
    pd.DataFrame(rows).to_csv(os.path.join(SRC_DIR, 'panel_a_group_mean_sem.csv'), index=False)

    # Panel b: CMC at N=5000 — per-participant top-k + chance + per-k Wilcoxon.
    rows = []
    for p in patients:
        r = perp[perp['patient'] == p].iloc[0]
        d = dict(display_id=did[p], patient=p, N=HEADLINE_N, n_trials=int(r['n_trials']),
                 median_rank=float(r['median_rank_all']))
        for k in KS:
            d[f'top{k}'] = float(r[f'top{k}_all']); d[f'chance_top{k}'] = k / float(HEADLINE_N)
        rows.append(d)
    dfb = pd.DataFrame(rows)
    dfb.to_csv(os.path.join(SRC_DIR, 'panel_b_cmc_N5000.csv'), index=False)
    sig_rows = []
    for k in KS:
        pv, _ = _wilcoxon(dfb[f'top{k}'].values, k / float(HEADLINE_N), 'greater')
        sig_rows.append(dict(k=k, chance=k / float(HEADLINE_N),
                             topk_mean=float(dfb[f'top{k}'].mean()), wilcoxon_p=pv, sig=_stars(pv)))
    pd.DataFrame(sig_rows).to_csv(os.path.join(SRC_DIR, 'panel_b_significance.csv'), index=False)

    # Panel c: zero-shot — in-vocab vs held-out + counts + perm p + paired test.
    c = perp[['patient', 'n_trials', 'n_held_out',
              'median_percentile_all', 'median_percentile_invocab', 'median_percentile_heldout',
              'perm_p_median_percentile_all']].copy()
    c.insert(0, 'display_id', c['patient'].map(did))
    c['chance_median_percentile'] = CHANCE_PCT
    c.rename(columns={'perm_p_median_percentile_all': 'perm_p_all'}, inplace=True)
    c.to_csv(os.path.join(SRC_DIR, 'panel_c_zeroshot.csv'), index=False)

    # Panel d: neighbour similarity — obs vs null, within-participant perm p.
    d = perp[['patient', 'graded_near_miss_sim_mean', 'near_miss_null_mean',
              'perm_p_near_miss', 'category_hit_at_k']].copy()
    d.insert(0, 'display_id', d['patient'].map(did))
    d['near_miss_delta'] = d['graded_near_miss_sim_mean'] - d['near_miss_null_mean']
    d.rename(columns={'graded_near_miss_sim_mean': 'near_miss_obs',
                      'near_miss_null_mean': 'near_miss_null'}, inplace=True)
    d.to_csv(os.path.join(SRC_DIR, 'panel_d_semantic_neighbours.csv'), index=False)

    # Panel e: nDCG@100 — obs vs null, within-participant perm p.
    e = perp[['patient', 'graded_ndcg_mean', 'ndcg_null_mean', 'perm_p_ndcg']].copy()
    e.insert(0, 'display_id', e['patient'].map(did))
    e['ndcg_delta'] = e['graded_ndcg_mean'] - e['ndcg_null_mean']
    e.rename(columns={'graded_ndcg_mean': 'ndcg_obs', 'ndcg_null_mean': 'ndcg_null'}, inplace=True)
    e.to_csv(os.path.join(SRC_DIR, 'panel_e_ndcg.csv'), index=False)

    # Group-level inference (tidy from the pipeline JSON) — Results text.
    grows = []
    for key in ['median_percentile_all', 'median_percentile_invocab', 'median_percentile_heldout']:
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
    nd = ginf['ndcg']
    grows.append(dict(metric='ndcg_at_100', n=nd['n'], median=np.nan, chance=np.nan,
                      ci_mean=nd['mean'], ci_lo=nd['lo'], ci_hi=nd['hi'],
                      wilcoxon_p=np.nan, test='bootstrap 95% CI (descriptive)'))
    pd.DataFrame(grows).to_csv(os.path.join(SRC_DIR, 'group_inference.csv'), index=False)


# ── Caption ─────────────────────────────────────────────────────────────────────

CAPTION = """# Figure caption — Extendability of the regression-and-retrieval decoder

Extendability of the regression-and-retrieval decoder (picture naming; {N} participants).
The kernel-PLS decoder (Nystroem RBF kernel followed by PLS regression onto GloVe word-embedding
targets) predicts an embedding per trial; the predicted vector is ranked by cosine similarity
against an open word gallery of {HN} words (the stimulus words plus POS- and frequency-matched
distractors never presented to any participant), and the rank of the true word is the score.
Chance: median percentile rank 0.5; top-k accuracy k/N. **a** Median percentile rank (rank/N;
lower is better) versus gallery size N (200–5000 words); box, interquartile range and median across
participants; coloured points, individual participants; bold black, across-participant mean; stars,
Wilcoxon signed-rank versus chance per N. **b** Top-k retrieval accuracy versus rank k at N=5000
(same box/points/mean convention; log y; dashed, chance k/N; stars, Wilcoxon versus chance per k).
**c** Median percentile rank at N=5000 for words seen in training (in-vocab) versus words held
entirely out of training (held-out, zero-shot; 30% of unique words held out per cross-validation
split); box + points across participants; bracket, paired Wilcoxon (in-vocab versus held-out).
**d** Wu–Palmer WordNet similarity between the true word and its top-10 retrieved neighbours, for a
matched random-draw null versus the neural retrieval (WordNet grade is independent of the GloVe
decode target). **e** nDCG@100 of the neural ranking versus the same matched permutation null
(whole-list semantic organisation under the independent grade). In **d**,**e**: box + points across
participants; bracket, group Wilcoxon of the observed-minus-null difference. **f** Two-dimensional
MDS (cosine) of a best-participant semantic-neighbourhood showcase: for several stimulus words of
diverse semantic category, the predicted word (blue, bold; the top-retrieved gallery word at its own
embedding) is shown beside the ground-truth word (black, bold) and their nearest gallery neighbours
(grey); predictions land on the true word and its near-synonyms. In **a**–**e**: box, interquartile
range and median across participants; coloured points, individual participants (one fixed colour per
participant); bold black, across-participant mean; dashed grey, chance. Group tests are Wilcoxon
signed-rank (see Results). Participants identified by display ID (NUEx###). **a**–**f** N={N}.
Supplements: S1, per-participant held-out per-trial percentile distributions across N; S2,
qualitative best-case retrievals; S3–S4, semantic-neighbourhood showcases for three further
participants (S3: NUEx031, NUEx041; S4: NUEx036).
"""


def write_caption(patients):
    txt = CAPTION.format(N=len(patients), HN=HEADLINE_N, sig=SIG_ALPHA)
    with open(os.path.join(FIG_DIR, 'caption.md'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(txt)


# ── Orchestration ───────────────────────────────────────────────────────────────

def _save(fig, stem, dpi=200):
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    fig.savefig(stem + '.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def generate():
    os.makedirs(SRC_DIR, exist_ok=True)
    perp, sweep, ginf, patients = load_inputs()
    colors = assign_colors(patients)
    mds = _cache('cache_panelf_mds.csv')
    heldout = _cache('cache_heldout_trial_percentile_by_N.csv')
    best = _cache('cache_qualitative_bestcases.csv')

    # Standalone panels
    specs = [
        ('01_scaling_median_percentile', (4.3, 3.4), lambda ax: draw_scaling(ax, sweep, patients, colors)),
        ('02_cmc_N5000', (4.3, 3.4), lambda ax: draw_cmc(ax, perp, patients, colors)),
        ('03_zeroshot_invocab_vs_heldout', (4.0, 3.4), lambda ax: draw_zeroshot(ax, perp, ginf, patients, colors)),
        ('04_semantic_neighbours', (4.0, 3.4), lambda ax: draw_neighbours(ax, perp, ginf, patients, colors)),
        ('05_ndcg_vs_null', (4.0, 3.4), lambda ax: draw_ndcg(ax, perp, ginf, patients, colors)),
        ('06_mds_neighbourhood', (5.2, 4.6), lambda ax: draw_mds(ax, mds)),
    ]
    for stem, size, fn in specs:
        fig, ax = plt.subplots(figsize=size); fn(ax)
        fig.tight_layout(); _save(fig, os.path.join(FIG_DIR, stem))

    # Combined layout: row 1 = a, b (wide); row 2 = c, d, e (narrow) + f (wider).
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
    fig.legend(handles=_legend_handles(patients, colors), ncol=8, loc='lower center',
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, os.path.join(FIG_DIR, '00_extendability_combined'), dpi=300)

    # Supplements (not in the combined figure)
    supp_heldout_distributions(heldout, patients, colors)
    supp_bestcase_table(best)
    # Supplementary panel-f showcases for the next-best participants
    for pat, slabel in PANELF_SUPP:
        df = _cache(f'cache_panelf_{pat}.csv')
        if df is None:
            print(f"[extendability] {slabel} {pat} skipped — cache_panelf_{pat}.csv missing")
            continue
        figx, axx = plt.subplots(figsize=(5.2, 4.6)); draw_mds(axx, df)
        figx.tight_layout(); _save(figx, os.path.join(FIG_DIR, f'{slabel}_mds_neighbourhood_{pat}'))

    write_source_data(perp, sweep, ginf, patients)
    write_caption(patients)

    # Results-text numbers
    def gmean(col):
        v = perp[col].values.astype(float); return float(np.mean(v)), _sem(v)
    print("[extendability] figures + caption ->", FIG_DIR)
    print("[extendability] source data       ->", SRC_DIR)
    print(f"  participants: {[display_id(p) for p in patients]}")
    print(f"  median %rank  all   median={ginf['median_percentile_all']['median']:.4f} "
          f"Wilcoxon p={ginf['median_percentile_all']['p_value']:.2e}")
    print(f"  median %rank  invoc median={ginf['median_percentile_invocab']['median']:.4f} "
          f"heldout median={ginf['median_percentile_heldout']['median']:.4f} "
          f"(p={ginf['median_percentile_heldout']['p_value']:.2e})")
    m, s = gmean('top10_all'); print(f"  top-10 (N=5000) mean={m:.3f} +/- {s:.3f}")
    m, s = gmean('top100_all'); print(f"  top-100(N=5000) mean={m:.3f} +/- {s:.3f}")
    print(f"  nDCG@100 mean={ginf['ndcg']['mean']:.3f} CI[{ginf['ndcg']['lo']:.3f},{ginf['ndcg']['hi']:.3f}]")
    print(f"  near-miss delta Wilcoxon p={ginf['near_miss_vs_null']['p_value']:.2e}")
    if 'ndcg_vs_null' in ginf:
        print(f"  nDCG delta Wilcoxon p={ginf['ndcg_vs_null']['p_value']:.2e}")


if __name__ == '__main__':
    generate()
