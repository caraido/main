# -*- coding: utf-8 -*-
"""Throwaway diagnostic: six per-bin decision rules on the same (n_epochs, n_bins)
obs/null arrays from the pinned picture-naming and auditory-naming runs, broken down
per participant and per time bin as a significance raster -- the same idiom as
figures_for_paper/semantic_regression (participant traces on top, one raster row per
participant below, rows ordered by peak accuracy, participant colours from
participants.json via paper_common.assign_colors).

Every rule is ONE-SIDED, testing obs > null: the two permutation rules take an upper-tail
count, and the three sample-based rules pass alternative='greater'. Rules (3)-(5) are
additionally Bonferroni corrected over the time bins within each participant, and judged at
their own cutoff (--alpha-sample, default 0.005 -- the one-sided equivalent of a two-sided
0.01) rather than at utils.config.ALPHA.

This does NOT propose changing the shipped test. docs/agent-context/scientific-integrity.md
records that the test family is settled: every t-test variant was tried and failed the same
way (a reliable ~0.01 obs-over-null offset at baseline passes any t-test at n=100 epochs,
putting 30-45 % of PRE-ONSET bins over threshold). Pre-onset bins are therefore kept and
drawn, unlike the shipped panels, which mask t<0 out of the raster (panels.py:540).

Metrics: r2, category_indep, word_top1/3/5. `cosine` is NOT here -- no shuffled cosine is
stored anywhere in the repo (models/model.py never scores one), so no test of any kind can
be run on it. r2 stands in as the continuous-fit metric that does carry a matched null.

Run from main/:   python -m tests.significance_test_comparison.perbin_test_comparison
                  python -m tests.significance_test_comparison.perbin_test_comparison \
                      --alpha-sample 0.001
Outputs, all under results/significance_test_comparison/:
                  perbin_test_comparison.html
                  source_data/perbin_test_comparison.csv            (pooled totals)
                  source_data/perbin_test_comparison_bypatient.csv  (per participant)
                  source_data/perbin_test_comparison_perbin.csv     (per participant per bin)
"""

import os
import sys
import json
import base64
import argparse
import warnings
from io import BytesIO

import numpy as np
import pandas as pd
from scipy import stats

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGS_ROOT = os.path.join(MAIN_DIR, 'figures_for_paper')
for _p in (MAIN_DIR, FIGS_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.config import PIC_RUN, AUD_RUN, ALPHA, PCTILE, run_dir   # noqa: E402
from utils.paths import results_dir                                  # noqa: E402
from paper_common import display_id, assign_colors                   # noqa: E402
from tests.significance_test_comparison import r2_cache_build as R2   # noqa: E402

#: This pilot's one output destination, and the only place it may write. `source_data/`
#: mirrors the convention a run directory uses -- the numbers actually plotted, beside the
#: thing that plots them.
ANALYSIS = 'significance_test_comparison'
OUT_DIR = str(results_dir(ANALYSIS))
SRC_DIR = str(results_dir(ANALYSIS, 'source_data'))

import matplotlib                                                    # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                      # noqa: E402
import matplotlib.lines as mlines                                    # noqa: E402
from matplotlib.patches import Patch                                 # noqa: E402
from matplotlib.gridspec import GridSpec                             # noqa: E402

PANELS_DIR = os.path.join(MAIN_DIR, 'figures_for_paper', 'semantic_regression')
EMBEDDING = 'GloVe'
TASKS = {'picture': PIC_RUN, 'auditory': AUD_RUN}
TASK_LABEL = {'picture': 'Picture naming', 'auditory': 'Auditory naming'}
ALIGN_LABEL = {'picture': 'Time from trial onset (s)',
               'auditory': 'Time from auditory stimulus onset (s)'}

METRICS = [
    ('r2', 'R²  (cosine stand-in)'),
    ('category_indep', 'Category accuracy'),
    ('word_top1', 'Word top-1'),
    ('word_top3', 'Word top-3'),
    ('word_top5', 'Word top-5'),
]

# (id, base label, bonferroni?) in fixed order -> fixed colour slot. Never cycled.
# Every rule is one-sided, obs > null. The permutation rules are NOT Bonferroni corrected:
# their p floors at 1/(n_epochs+1) = 0.0099, so any correction with m >= 2 would zero them
# out mechanically rather than informatively, and the shipped rule has to stay a faithful
# reference. Cutoffs come from the CLI (see build_alphas), never from a literal here --
# AGENTS.md forbids a module-level p-value cutoff.
TESTS = [
    ('pctile',    f'shipped rule: mean(obs) > pct{PCTILE:g}(null)', False),
    ('perm_fwd',  '(1) perm: P(null ≥ mean obs)', False),
    ('perm_rev',  '(2) perm reversed: P(obs ≤ mean null)', False),
    ('ttest_ind', '(3) unpaired t-test', True),
    ('ttest_rel', '(4) paired t-test', True),
    ('wilcoxon',  '(5) Wilcoxon signed-rank', True),
]
TEST_IDS = [t for t, _l, _b in TESTS]
TEST_BASE = {t: l for t, l, _b in TESTS}
TEST_BONF = {t: b for t, _l, b in TESTS}
_SHORT = {'pctile': 'shipped pctile', 'perm_fwd': '(1) perm', 'perm_rev': '(2) perm rev',
          'ttest_ind': '(3) t unpaired', 'ttest_rel': '(4) t paired',
          'wilcoxon': '(5) Wilcoxon'}
_STACK = {'pctile': 'shipped\npctile rule', 'perm_fwd': '(1) perm',
          'perm_rev': '(2) perm\nreversed', 'ttest_ind': '(3) unpaired t',
          'ttest_rel': '(4) paired t', 'wilcoxon': '(5) Wilcoxon'}


def build_alphas(alpha_perm, alpha_sample):
    """test id -> cutoff. The two permutation rules and the shipped rule keep the
    repo-wide cutoff; the three sample-based rules get their own, stricter one."""
    return {t: (alpha_sample if TEST_BONF[t] else alpha_perm) for t in TEST_IDS}


def test_label(t, alphas):
    """Full legend label, carrying the correction and the cutoff actually applied."""
    if t == 'pctile':
        return TEST_BASE[t]
    tail = f'Bonferroni, α={alphas[t]:g}' if TEST_BONF[t] else f'α={alphas[t]:g}'
    return f'{TEST_BASE[t]} · {tail}'


def test_short(t, alphas):
    return _SHORT[t] if t == 'pctile' else f'{_SHORT[t]} α={alphas[t]:g}'


def test_stack(t, alphas):
    """Two-line form for the raster block labels, which sit in the left margin."""
    if t == 'pctile':
        return _STACK[t]
    tail = f'+ Bonf, α={alphas[t]:g}' if TEST_BONF[t] else f'α={alphas[t]:g}'
    return f'{_STACK[t]}\n{tail}'

# dataviz reference palette, categorical slots 1-6 (light mode).
# Validated: node scripts/validate_palette.js "<these>" --mode light -> ALL CHECKS PASS
# (contrast WARN on aqua/yellow/magenta -> relief rule: every bar carries a visible value
#  label and the full count table is rendered below the figure).
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
INK_MUTED = '#8a8880'


# ── Loading ───────────────────────────────────────────────────────────────────

def load_task(task):
    """{'patients', 'meta', 'arrays': {(pat, metric): (obs, null)}, 'run'}.

    Retrieval metrics come from the panels cache (already built, keyed on the same
    pinned run); r2 comes from this pilot's own r2_cache_build.py.
    """
    rdir = run_dir(TASKS[task])
    panels_npz = os.path.join(PANELS_DIR, f'panels_cache_{task}_{EMBEDDING}.npz')
    side = json.load(open(panels_npz + '.json'))
    if os.path.abspath(side['run_dir']) != os.path.abspath(rdir):
        raise RuntimeError(
            f"{os.path.basename(panels_npz)} was built from {side['run_dir']}, not the "
            f"pinned {TASKS[task]} -- rebuild it before trusting this comparison")
    panels = dict(np.load(panels_npz))

    r2 = R2.get(task)
    if os.path.abspath(r2['side']['run_dir']) != os.path.abspath(rdir):
        raise RuntimeError(f"r2 cache for {task} is from a different run")

    patients = list(side['patients'])
    if list(r2['side']['patients']) != patients:
        raise RuntimeError(f"{task}: r2 cache patients {r2['side']['patients']} != "
                           f"panels cache patients {patients}")

    arrays = {}
    for p in patients:
        for key, _lab in METRICS:
            src = r2['arrays'] if key == 'r2' else panels
            arrays[(p, key)] = (np.asarray(src[f'{p}__{key}__obs'], dtype=np.float64),
                                np.asarray(src[f'{p}__{key}__null'], dtype=np.float64))

    meta = json.load(open(os.path.join(rdir, 'meta.json')))
    return {'patients': patients, 'meta': meta, 'arrays': arrays, 'run': TASKS[task]}


def time_axis(n_bins, n_bins_history, bin_size_ms):
    """Same formula as semantic_regression_panels._time_axis (:345-346)."""
    return np.array([(b - n_bins_history) * bin_size_ms / 1000.0 for b in range(n_bins)])


# ── The six decision rules ────────────────────────────────────────────────────

def perbin_pvalues(obs, null):
    """obs/null: (n_epochs, n_bins). Returns (pvals, obs_mean, null_mean).

    pvals[test] is (n_bins,), one-sided 'obs > null', RAW (uncorrected). `pctile` is a
    threshold rule rather than a p-value and is returned as 0/1. NaN wherever a test is
    undefined (all-NaN bin, or all-zero paired differences) -- never silently 0 or 1.
    """
    n_bins = obs.shape[1]
    out = {t: np.full(n_bins, np.nan) for t in TEST_IDS}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN bins are expected
        obs_mean = np.nanmean(obs, axis=0)
        null_mean = np.nanmean(null, axis=0)
        thr = np.nanpercentile(null, PCTILE, axis=0)

    out['pctile'] = np.where(
        np.isfinite(obs_mean) & np.isfinite(thr), (obs_mean > thr).astype(float), np.nan)

    for b in range(n_bins):
        o = obs[:, b][np.isfinite(obs[:, b])]
        n = null[:, b][np.isfinite(null[:, b])]
        if o.size == 0 or n.size == 0:
            continue
        # (1) current: where the observed epoch-mean sits in the null distribution
        out['perm_fwd'][b] = (np.sum(n >= obs_mean[b]) + 1) / (n.size + 1)
        # (2) reverse: where the null mean sits in the observed distribution
        out['perm_rev'][b] = (np.sum(o <= null_mean[b]) + 1) / (o.size + 1)
        # (3)/(4)/(5) treat the epochs as samples -- see the caveat in the header
        if o.size >= 2 and n.size >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                out['ttest_ind'][b] = float(stats.ttest_ind(o, n, alternative='greater').pvalue)
        # ttest_1samp on the epoch-wise differences IS ttest_rel(obs, null), and it drops
        # NaN pairs cleanly; wilcoxon takes the same differences.
        pair = obs[:, b] - null[:, b]
        pair = pair[np.isfinite(pair)]
        if pair.size >= 2 and not np.allclose(pair, 0):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                out['ttest_rel'][b] = float(stats.ttest_1samp(pair, 0.0,
                                                              alternative='greater').pvalue)
                try:
                    out['wilcoxon'][b] = float(stats.wilcoxon(pair, alternative='greater').pvalue)
                except ValueError:
                    pass
    return out, obs_mean, null_mean


def adjust_and_decide(pvals, alphas):
    """Raw p -> (p_adj, sig, defined) per test, each judged at its own cutoff.

    Bonferroni family for rules (3)-(5) = the time bins of THIS participant for THIS
    metric on which the test is defined (m below). Correcting within the participant
    is what the repo previously tried (scientific-integrity.md:47-55); it does not
    correct across participants or across metrics, and the HTML says so.
    """
    padj, sig, defined = {}, {}, {}
    for t in TEST_IDS:
        v = np.asarray(pvals[t], dtype=float)
        d = np.isfinite(v)
        if t == 'pctile':
            padj[t] = v                       # a decision, not a p-value
            s = np.zeros_like(d)
            s[d] = v[d] > 0.5
        else:
            a = v.copy()
            if TEST_BONF[t]:
                m = int(d.sum())               # number of bins actually tested
                a = np.where(d, np.minimum(1.0, v * max(m, 1)), np.nan)
            padj[t] = a
            s = np.zeros_like(d)
            s[d] = a[d] < alphas[t]
        sig[t], defined[t] = s, d
    return padj, sig, defined


# ── Per-participant computation ───────────────────────────────────────────────

def compute(task_data, task, alphas):
    """{(metric, patient): dict(time_s, obs_mean, null_mean, p_raw, p_adj, sig, defined, m)}"""
    m = task_data['meta']
    nbh, bsz = m['n_bins_history'], m['bin_size_ms']
    out = {}
    for key, _lab in METRICS:
        for p in task_data['patients']:
            obs, null = task_data['arrays'][(p, key)]
            praw, om, nm = perbin_pvalues(obs, null)
            padj, sig, defined = adjust_and_decide(praw, alphas)
            out[(key, p)] = dict(
                task=task, metric=key, patient=p,
                time_s=time_axis(obs.shape[1], nbh, bsz),
                obs_mean=om, null_mean=nm,
                p_raw=praw, p_adj=padj, sig=sig, defined=defined,
                m={t: int(defined[t].sum()) for t in TEST_IDS})
    return out


def tally(per, task, alphas):
    """Pooled and per-participant significant-bin counts, split pre/post onset.

    `n_sig_uncorrected` applies the same cutoff to the RAW p, so the column isolates what
    the Bonferroni step did rather than confounding it with the change of cutoff.
    """
    prows = []
    for (key, p), d in per.items():
        ts = d['time_s']
        for t in TEST_IDS:
            for win, mask in (('pre', ts < 0), ('post', ts >= 0)):
                raw_sig = (np.isfinite(d['p_raw'][t]) & (d['p_raw'][t] < alphas[t])
                           if t != 'pctile' else d['sig'][t])
                prows.append(dict(
                    task=task, patient=p, metric=key, test=t, window=win,
                    alpha=alphas[t],
                    n_sig=int(np.sum(d['sig'][t] & mask)),
                    n_sig_uncorrected=int(np.sum(raw_sig & mask)),
                    n_total=int(np.sum(mask)),
                    n_undefined=int(np.sum(mask & ~d['defined'][t])),
                    bonferroni_m=d['m'][t] if TEST_BONF[t] else 0))
    dfp = pd.DataFrame(prows)
    cols = ['n_sig', 'n_sig_uncorrected', 'n_total', 'n_undefined']
    df = dfp.groupby(['task', 'metric', 'test', 'window', 'alpha'], as_index=False)[cols].sum()
    df['pct_sig'] = np.where(df['n_total'] > 0, 100.0 * df['n_sig'] / df['n_total'], np.nan)
    return df, dfp


def perbin_frame(per):
    """Long form: one row per participant x bin, with every rule's p and decision."""
    rows = []
    for (_key, _p), d in per.items():
        for b in range(len(d['time_s'])):
            r = dict(task=d['task'], metric=d['metric'], patient=d['patient'],
                     display_id=display_id(d['patient']), bin_index=b,
                     time_s=round(float(d['time_s'][b]), 4),
                     obs_mean=d['obs_mean'][b], null_mean=d['null_mean'][b])
            for t in TEST_IDS:
                if t != 'pctile':
                    r[f'p_raw__{t}'] = d['p_raw'][t][b]
                    if TEST_BONF[t]:
                        r[f'p_bonf__{t}'] = d['p_adj'][t][b]
                r[f'sig__{t}'] = bool(d['sig'][t][b])
            rows.append(r)
    return pd.DataFrame(rows)


# ── Figures ───────────────────────────────────────────────────────────────────

def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color('#d5d4cf')
    ax.tick_params(colors=INK_2, labelsize=7, length=3)


def raster_figure(per, patients, task, key, label, color_of, alphas):
    """Participant traces + one significance raster block per rule, on a shared time axis.

    Raster rows are ordered by peak observed accuracy (highest at the top) and painted in
    the participant's fixed colour, matching semantic_regression_panels._draw_panel:380-390.
    Unlike the shipped panels, pre-onset bins are NOT masked -- everything left of the
    dotted line at t=0 is the false-positive control.
    """
    n_pat = len(patients)
    bin_s = float(np.diff(per[(key, patients[0])]['time_s'])[0])
    xmin = min(per[(key, p)]['time_s'][0] for p in patients) - bin_s
    xmax = max(per[(key, p)]['time_s'][-1] for p in patients) + bin_s
    order = sorted(patients, key=lambda p: np.nanmax(per[(key, p)]['obs_mean']), reverse=True)

    block_h = max(0.9, 0.105 * n_pat)
    fig = plt.figure(figsize=(11.0, 2.5 + len(TESTS) * block_h + 0.5), facecolor=SURFACE)
    gs = GridSpec(1 + len(TESTS), 1, figure=fig,
                  height_ratios=[2.5] + [block_h] * len(TESTS), hspace=0.16)

    # ── traces
    ax = fig.add_subplot(gs[0])
    _style_axis(ax)
    for p in patients:
        d = per[(key, p)]
        ax.plot(d['time_s'], d['obs_mean'], color=color_of[p], lw=1.1, alpha=0.9, zorder=3)
        ax.plot(d['time_s'], d['null_mean'], color=color_of[p], lw=0.7, alpha=0.35,
                ls='--', zorder=2)
    ax.axvline(0, color=INK, lw=0.9, ls=':', zorder=1)
    ax.set_xlim(xmin, xmax)
    ax.set_ylabel(label, fontsize=8.5, color=INK)
    ax.set_title(f'{TASK_LABEL[task]} (N={n_pat}) — {label} — per participant, per time bin',
                 fontsize=11, color=INK, pad=8)
    ax.text(0.0, 1.012, 'solid = observed   dashed = shuffled null',
            transform=ax.transAxes, fontsize=6.5, color=INK_MUTED, va='bottom')

    # ── one raster block per rule
    for i, t in enumerate(TEST_IDS):
        axr = fig.add_subplot(gs[i + 1])
        _style_axis(axr)
        axr.set_xlim(xmin, xmax)
        axr.set_ylim(0, n_pat)
        axr.axvspan(xmin, 0, color='#efeee9', lw=0, zorder=0)     # pre-onset shading
        for j, p in enumerate(order):
            d = per[(key, p)]
            y0 = n_pat - j - 1
            segs = [(d['time_s'][b] - bin_s / 2, bin_s)
                    for b in range(len(d['time_s'])) if d['sig'][t][b]]
            if segs:
                axr.broken_barh(segs, (y0 + 0.08, 0.84), facecolors=color_of[p],
                                edgecolors='none', zorder=2)
        axr.axvline(0, color=INK, lw=0.9, ls=':', zorder=3)
        axr.set_yticks(np.arange(n_pat) + 0.5)
        axr.set_yticklabels([display_id(p) for p in order[::-1]], fontsize=5.6, color=INK_2)
        axr.set_ylabel(test_stack(t, alphas), fontsize=7.5, color=SERIES[i], rotation=0,
                       ha='right', va='center', labelpad=44, fontweight='bold',
                       linespacing=1.4)
        ns_post = sum(int(np.sum(per[(key, p)]['sig'][t] & (per[(key, p)]['time_s'] >= 0)))
                      for p in patients)
        nt_post = sum(int(np.sum(per[(key, p)]['time_s'] >= 0)) for p in patients)
        ns_pre = sum(int(np.sum(per[(key, p)]['sig'][t] & (per[(key, p)]['time_s'] < 0)))
                     for p in patients)
        nt_pre = sum(int(np.sum(per[(key, p)]['time_s'] < 0)) for p in patients)
        axr.text(1.004, 0.5, f'post {ns_post}/{nt_post} ({100*ns_post/nt_post:.0f}%)\n'
                             f'pre  {ns_pre}/{nt_pre} ({100*ns_pre/nt_pre:.0f}%)',
                 transform=axr.transAxes, fontsize=6.3, color=INK_2, va='center',
                 ha='left', linespacing=1.5)
        if i < len(TESTS) - 1:
            axr.set_xticklabels([])
        else:
            axr.set_xlabel(ALIGN_LABEL[task], fontsize=8.5, color=INK)

    fig.subplots_adjust(left=0.205, right=0.885, top=1 - 0.42 / fig.get_figheight(),
                        bottom=0.42 / fig.get_figheight())
    return _png(fig)


def summary_figure(df, n_pat, alphas):
    """Pooled counts, kept as the at-a-glance view above the per-participant rasters."""
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.4), facecolor=SURFACE)
    width = 1.0 / (len(TESTS) + 1.5)
    xpos = np.arange(len(METRICS))
    wt = {'post': 'POST-onset (t ≥ 0) — sensitivity',
          'pre': 'PRE-onset (t < 0) — false-positive control'}

    for r, task in enumerate(['picture', 'auditory']):
        for c, win in enumerate(['post', 'pre']):
            ax = axes[r][c]
            _style_axis(ax)
            sub = df[(df['task'] == task) & (df['window'] == win)]
            ntot = int(sub['n_total'].max())
            for i, t in enumerate(TEST_IDS):
                s = sub[sub['test'] == t].set_index('metric')
                vals = [int(s.loc[k, 'n_sig']) for k, _ in METRICS]
                off = (i - (len(TESTS) - 1) / 2) * width
                ax.bar(xpos + off, vals, width * 0.9, color=SERIES[i],
                       edgecolor=SURFACE, linewidth=1.0)
                for x, v in zip(xpos + off, vals):
                    ax.text(x, v + ntot * 0.012, str(v), ha='center', va='bottom',
                            fontsize=6.0, color=INK_2, rotation=90)
            ax.set_xticks(xpos)
            ax.set_xticklabels([lab.replace('  ', '\n') for _k, lab in METRICS],
                               fontsize=8.5, color=INK)
            ax.set_ylabel('significant participant×bin cells', fontsize=8, color=INK_2)
            ax.set_title(f'{task} naming (N={n_pat[task]})  —  {wt[win]}\n'
                         f'{ntot} cells per metric', fontsize=9.5, color=INK, pad=7)
            ax.set_ylim(0, ntot * 1.22)
            ax.grid(axis='y', color='#e6e5e0', linewidth=0.7)
            ax.set_axisbelow(True)
            ax.axhline(ntot, color=INK_MUTED, linewidth=0.8, linestyle=':')
            ax.text(ax.get_xlim()[1], ntot, ' all', fontsize=6.5, color=INK_MUTED,
                    va='center', ha='left', clip_on=False)

    fig.legend([Patch(facecolor=SERIES[i]) for i in range(len(TESTS))],
               [test_label(t, alphas) for t in TEST_IDS], loc='upper center',
               bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False,
               fontsize=9, labelcolor=INK_2, handlelength=1.4, columnspacing=2.2)
    fig.subplots_adjust(left=0.055, right=0.982, top=0.845, bottom=0.065,
                        wspace=0.17, hspace=0.42)
    return _png(fig)


def legend_figure(patients_by_task, color_of):
    """Participant colour key — the raster rows are painted by participant, so identity
    must be readable without counting rows."""
    seen, handles = set(), []
    for pats in patients_by_task.values():
        for p in pats:
            if p not in seen:
                seen.add(p)
                handles.append(mlines.Line2D([], [], color=color_of[p], lw=3,
                                             label=f'{display_id(p)} ({p})'))
    fig = plt.figure(figsize=(11.0, 0.85), facecolor=SURFACE)
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.legend(handles=handles, ncol=8, loc='center', fontsize=7.5, frameon=False,
              labelcolor=INK_2)
    return _png(fig)


def _png(fig, dpi=140):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


# ── HTML ──────────────────────────────────────────────────────────────────────

def html_table(df, alphas):
    piv = df.pivot_table(index=['task', 'window', 'metric'], columns='test',
                         values='n_sig', aggfunc='sum').reindex(columns=TEST_IDS)
    raw = df.pivot_table(index=['task', 'window', 'metric'], columns='test',
                         values='n_sig_uncorrected', aggfunc='sum').reindex(columns=TEST_IDS)
    tot = df.pivot_table(index=['task', 'window', 'metric'], values='n_total', aggfunc='max')
    und = df.pivot_table(index=['task', 'window', 'metric'], values='n_undefined', aggfunc='max')
    order = [(tk, w, m) for tk in ('picture', 'auditory') for w in ('post', 'pre')
             for m, _l in METRICS]
    head = ('<tr><th>task</th><th>window</th><th>metric</th><th>cells</th>'
            + ''.join(f'<th>{test_short(t, alphas)}</th>' for t in TEST_IDS)
            + '<th>undefined</th></tr>')
    body = []
    for k in order:
        n = int(tot.loc[k, 'n_total'])
        cells = ''
        for t in TEST_IDS:
            v = int(piv.loc[k, t])
            extra = (f'<span class="pc">{100.0 * v / n:.0f}%</span>'
                     if not TEST_BONF[t] else
                     f'<span class="pc">{100.0 * v / n:.0f}% · raw {int(raw.loc[k, t])}</span>')
            cells += f'<td>{v}{extra}</td>'
        body.append(f'<tr><td>{k[0]}</td><td>{k[1]}</td><td>{k[2]}</td>'
                    f'<td class="mut">{n}</td>{cells}'
                    f'<td class="mut">{int(und.loc[k, "n_undefined"])}</td></tr>')
    return f'<table>{head}{"".join(body)}</table>'


def write_html(summary_png, legend_png, rasters, df, n_pat, check_msg, pre_bins, alphas):
    nav = ' · '.join(
        f'<a href="#{task}-{k}">{task[:3]} {lab.split("  ")[0]}</a>'
        for task in ('picture', 'auditory') for k, lab in METRICS)
    blocks = []
    for task in ('picture', 'auditory'):
        for k, lab in METRICS:
            blocks.append(
                f'<h3 id="{task}-{k}">{TASK_LABEL[task]} — {lab}</h3>'
                f'<img src="data:image/png;base64,{rasters[(task, k)]}" '
                f'alt="{task} {k} per-participant significance raster">')

    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>Per-bin significance test comparison — per participant</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ background:{SURFACE}; color:{INK}; margin:0; padding:32px 40px 80px;
         font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif; max-width:1500px; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  h2 {{ font-size:15px; margin:38px 0 8px; }}
  h3 {{ font-size:13px; margin:30px 0 4px; color:{INK_2}; font-weight:600; }}
  p, li {{ color:{INK_2}; max-width:95ch; }}
  code {{ background:#f0efea; padding:1px 4px; border-radius:3px; font-size:12.5px; }}
  a {{ color:#2a78d6; }}
  .warn {{ border-left:3px solid #eb6834; background:#fdf3ee; padding:12px 16px; margin:18px 0; }}
  .ok {{ border-left:3px solid #1baf7a; background:#eefaf5; padding:12px 16px; margin:18px 0; }}
  .nav {{ font-size:12px; background:#f4f3ee; padding:10px 14px; border-radius:4px;
          margin:18px 0; line-height:2.1; }}
  img {{ width:100%; height:auto; margin:6px 0 4px; }}
  table {{ border-collapse:collapse; font-size:12px; margin-top:6px; }}
  th, td {{ border-bottom:1px solid #e6e5e0; padding:4px 10px; text-align:right; }}
  th:nth-child(-n+3), td:nth-child(-n+3) {{ text-align:left; }}
  th {{ color:{INK_2}; font-weight:600; border-bottom:1px solid #cfcec9; }}
  .pc {{ color:{INK_MUTED}; font-size:10.5px; margin-left:5px; }}
  .mut {{ color:{INK_MUTED}; }}
</style>
<h1>Per-bin significance: six decision rules, per participant and per time bin</h1>
<p>picture naming <code>{PIC_RUN}</code> (N={n_pat['picture']})<br>
auditory naming <code>{AUD_RUN}</code> (N={n_pat['auditory']})<br>
GloVe, 100 epochs. Shipped rule and permutation rules at
<code>utils.config.ALPHA = {alphas['perm_fwd']:g}</code>; sample-based rules (3)–(5) at
<code>α = {alphas['ttest_ind']:g}</code>. Diagnostic only — nothing under
<code>figures_for_paper/</code>, <code>results/</code> or <code>figures/</code> was touched.</p>

<div class="ok">
<b>Every rule is one-sided, testing obs &gt; null.</b> The two permutation rules take an
upper-tail count — (1) P(null ≥ mean obs), (2) P(obs ≤ mean null) — and the three
sample-based rules all pass <code>alternative='greater'</code>
(<code>scipy.stats.ttest_ind</code>, <code>ttest_1samp</code> on the epoch-wise
obs−null differences, which is <code>ttest_rel</code>, and <code>wilcoxon</code> on the
same differences). No two-sided test appears anywhere in this comparison.
</div>

<div class="warn">
<b>Rules (3)–(5) are Bonferroni corrected and judged at α = {alphas['ttest_ind']:g};
rules (1)–(2) and the shipped rule are uncorrected at α = {alphas['perm_fwd']:g}.</b>
The correction family is the time bins of one participant for one metric
(m = number of bins where the test is defined, 46–98 depending on participant), so
p<sub>bonf</sub> = min(1, p·m) and a bin counts as significant at p<sub>bonf</sub> &lt;
{alphas['ttest_ind']:g} — equivalently p<sub>raw</sub> &lt; {alphas['ttest_ind']:g}/m. It
does <i>not</i> correct across participants or across the five metrics; a whole-experiment
correction would be a much larger m. The permutation rules are left uncorrected on purpose:
their p floors at 1/(100+1) = 0.0099, so any Bonferroni step with m ≥ 2 would zero them out
mechanically rather than informatively, and the shipped rule has to stay a faithful
reference. The <code>raw</code> figure beside each corrected count in the table is that
rule's count at the <i>same</i> cutoff without the Bonferroni step, so it isolates what the
correction did.
</div>

<div class="warn">
<b>cosine is absent, and cannot be added from what is on disk.</b> No shuffled cosine
exists anywhere in the repo: <code>models/model.py</code> stores <code>all_cosine_sim</code>
but never scores a cosine on the shuffled fit, and
<code>tests/auditory_alignment/config.py:55</code> sets its null attribute to
<code>None</code> for that reason. With no null, none of these six rules is defined for it.
<b>R&sup2;</b> (<code>all_test_score</code> vs <code>all_chance</code>) is shown in its place
as the one continuous-fit metric that does carry a matched null — a substitution, not the
requested metric. (The <code>chance_mean</code> column of <code>per_time_scores.csv</code>
is this R&sup2; null, despite being drawn as a cosine chance line elsewhere; it is not a
cosine null.)
</div>

<div class="warn">
<b>Rules (3)–(5) treat the 100 epochs as independent samples, and they are not.</b> They are
repeated random train/test splits over the same trials, so their spread underestimates
sampling variance by an unknown factor.
<code>docs/agent-context/scientific-integrity.md:47-55</code> records that this was already
settled: scalar-mean Wilcoxon+BH → paired Wilcoxon+BH → paired t+Bonferroni → one-sample
t+Bonferroni → the current percentile permutation. Every t-test variant failed the same way —
observed accuracy sits a reliable ~0.01 above null even at baseline, and at n=100 epochs that
constant offset passes any t-test. <b>Read the shaded pre-onset region of each raster first.</b>
A rule with filled cells left of the dotted line is buying post-onset bins with false positives.
</div>

<div class="warn">
<b>The pre-onset window is only {pre_bins} bins per participant.</b> Both runs set
<code>back_sec = null</code>, so the only bins at <code>t &lt; 0</code> are the
<code>n_bins_history</code> history-fill bins. A real baseline, but a short one — the
percentages under it rest on few cells, and it is coarser than the 30–45 % figure quoted in
the integrity doc, which came from a longer pre-onset window.
</div>

<div class="ok"><b>Cross-check vs the shipped figure.</b> {check_msg}</div>

<h2>Pooled counts</h2>
<img src="data:image/png;base64,{summary_png}" alt="Six decision rules, pooled counts">
<p class="mut">Bar height is the raw count of significant participant×bin cells. y-axes are
per-panel — the pre and post windows have very different cell counts, so compare within a
panel and use the table for cross-panel percentages. Dotted line = every bin significant.</p>

{html_table(df, alphas)}
<p class="mut"><code>cells</code> = participant×bin cells in that window;
<code>undefined</code> = cells where the test could not be computed (all-NaN bin, or
all-zero paired differences), counted as not-significant. Permutation p floors at
1/(100+1) = 0.0099, so rules (1) and (2) can never fall below it — with
α = {alphas['perm_fwd']:g} that floor is comfortably under the cutoff, but it would make
α ≤ 0.0099 unreachable for them.</p>

<h2>Per participant, per time bin</h2>
<p>Traces: each participant's observed (solid) and shuffled-null (dashed) epoch-mean.
Below them, one significance raster per rule — one row per participant, ordered by peak
observed accuracy (highest at the top), painted in that participant's fixed colour from
<code>participants.json</code>. The shaded region left of the dotted line is pre-onset.
Counts at the right of each block are post / pre.</p>
<img src="data:image/png;base64,{legend_png}" alt="participant colour key">
<div class="nav">{nav}</div>
{''.join(blocks)}

<p class="mut" style="margin-top:40px">Every number here also exists as text, under
<code>results/significance_test_comparison/source_data/</code>:
<code>perbin_test_comparison.csv</code> (pooled),
<code>perbin_test_comparison_bypatient.csv</code> (per participant, with the
Bonferroni m), <code>perbin_test_comparison_perbin.csv</code> (per participant per bin,
with each rule's raw and corrected p).</p>
"""
    out = os.path.join(OUT_DIR, 'perbin_test_comparison.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(doc)
    return out


# ── Cross-check against the shipped source_data.csv ───────────────────────────

def crosscheck(per_by_task):
    """The `pctile` rule on post-onset bins must reproduce the `significant` column of
    figures_for_paper/semantic_regression/source_data/source_data.csv exactly."""
    sd_path = os.path.join(PANELS_DIR, 'source_data', 'source_data.csv')
    if not os.path.exists(sd_path):
        return "source_data.csv not found -- not checked."
    sd = pd.read_csv(sd_path)
    mine = []
    for task, per in per_by_task.items():
        for (key, p), d in per.items():
            if key == 'r2':
                continue                     # r2 is not in the shipped figure
            s = d['sig']['pctile'] & (d['time_s'] >= 0)   # panels.py:540 masks pre-onset
            for b in range(len(s)):
                mine.append((task, key, p, b, bool(s[b])))
    md = pd.DataFrame(mine, columns=['task', 'metric', 'patient', 'bin_index', 'mine'])
    j = sd.merge(md, on=['task', 'metric', 'patient', 'bin_index'], how='inner')
    if not len(j):
        return "no overlapping rows with source_data.csv -- not checked."
    bad = j[j['significant'].astype(bool) != j['mine']]
    n, agree = len(j), len(j) - len(bad)
    msg = (f"<b>{agree}/{n}</b> rows agree between the <code>pctile</code> rule computed "
           f"here and the <code>significant</code> column of the shipped "
           f"<code>source_data.csv</code>, over the four retrieval metrics "
           f"(r&sup2; is not in the shipped figure). ")
    if len(bad) == 0:
        return msg + "Exact match, so this script is reading the same data as the figure."
    det = []
    for _i, r in bad.iterrows():
        d = per_by_task[r['task']][(r['metric'], r['patient'])]
        b = int(r['bin_index'])
        gap = float(d['obs_mean'][b] - r['null_p'])
        det.append(f"{r['task']}/{r['metric']}/{r['patient']} bin {b}: "
                   f"mean(obs) &minus; threshold = {gap:.3g}")
    return (msg + f"The {len(bad)} exception(s) are exact ties, not method differences — "
            f"the panels script means the cached <code>float32</code> arrays directly while "
            f"this script casts to <code>float64</code> first: " + "; ".join(det) + ".")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--alpha', type=float, default=ALPHA,
                    help='cutoff for the shipped rule and the two permutation rules '
                         f'(default: utils.config.ALPHA = {ALPHA})')
    ap.add_argument('--alpha-sample', type=float, default=0.005,
                    help='cutoff for the Bonferroni-corrected sample-based rules '
                         '(3)-(5) (default: 0.005 = one-sided equivalent of a '
                         'two-sided 0.01)')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    alphas = build_alphas(args.alpha, args.alpha_sample)
    print(f"alpha: perm/shipped {args.alpha:g}, sample-based (3)-(5) {args.alpha_sample:g} "
          f"(one-sided obs > null throughout)", flush=True)

    data = {t: load_task(t) for t in TASKS}
    n_pat = {t: len(d['patients']) for t, d in data.items()}
    print(f"patients: {n_pat}", flush=True)

    per_by_task, dfs, pdfs, bins = {}, [], [], []
    for task, td in data.items():
        print(f"[{task}] computing per participant ...", flush=True)
        per = compute(td, task, alphas)
        per_by_task[task] = per
        a, b = tally(per, task, alphas)
        dfs.append(a)
        pdfs.append(b)
        bins.append(perbin_frame(per))
    df = pd.concat(dfs, ignore_index=True)
    dfp = pd.concat(pdfs, ignore_index=True)
    dfb = pd.concat(bins, ignore_index=True)

    df.to_csv(os.path.join(SRC_DIR, 'perbin_test_comparison.csv'), index=False)
    dfp.to_csv(os.path.join(SRC_DIR, 'perbin_test_comparison_bypatient.csv'), index=False)
    dfb.to_csv(os.path.join(SRC_DIR, 'perbin_test_comparison_perbin.csv'), index=False)
    print(f"  per-bin rows: {len(dfb)}", flush=True)

    print("[check] cross-checking pctile rule vs source_data.csv ...", flush=True)
    msg = crosscheck(per_by_task)
    print("  " + msg.replace('<b>', '').replace('</b>', '')
                    .replace('<code>', '').replace('</code>', ''), flush=True)

    all_pat = sorted({p for d in data.values() for p in d['patients']})
    color_of = dict(zip(all_pat, assign_colors(all_pat)))

    print("[fig] pooled summary ...", flush=True)
    summary_png = summary_figure(df, n_pat, alphas)
    legend_png = legend_figure({t: d['patients'] for t, d in data.items()}, color_of)
    rasters = {}
    for task, td in data.items():
        for key, lab in METRICS:
            print(f"[fig] raster {task}/{key} ...", flush=True)
            rasters[(task, key)] = raster_figure(per_by_task[task], td['patients'],
                                                 task, key, lab, color_of, alphas)

    pre_bins = int(data['picture']['meta']['n_bins_history'])
    out = write_html(summary_png, legend_png, rasters, df, n_pat, msg, pre_bins, alphas)
    print(f"\nwrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB)", flush=True)

    piv = df.pivot_table(index=['task', 'window', 'metric'], columns='test',
                         values='pct_sig').reindex(columns=TEST_IDS)
    print(f"\n% of bins significant  [(1),(2),shipped @ α={args.alpha:g} uncorrected; "
          f"(3)-(5) @ α={args.alpha_sample:g} Bonferroni within participant]:\n")
    print(piv.round(1).to_string())


if __name__ == '__main__':
    main()
