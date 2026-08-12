# -*- coding: utf-8 -*-
"""
figures_for_paper/semantic_regression — Cross-patient decoding time-course panels.

Paper-figure generator. Covers BOTH naming tasks (picture naming, auditory naming),
each from its own results run. Produces one panel per decoding metric per task (GloVe
only), each overlaying every participant of that task in a distinct colour, with:

  * a per-participant significance raster *below the chance line* — bins where the
    observed mean accuracy exceeds the PCTILE-th percentile of the shuffled-null
    distribution at that bin (per-bin one-sided permutation test at the repo-wide
    ALPHA = 0.05, i.e. the 95th percentile; both live in utils/config.py);
  * cue markers as a vertical line at the across-participant mean time with a shaded
    band = ± 1 s.d. across participants; a cue identical across participants (the
    group-warped stimulus offset) has no band and is drawn as a single crisp line.
    The run's own alignment cue, and cues falling outside the panel's time window,
    are skipped;
  * an x-axis in seconds with 0 at that task's alignment cue (trial onset for picture
    naming, auditory stimulus onset for auditory naming);
  * a y-axis scale shared within a metric family (the three word top-k panels share one
    scale; the category panel has its own) and set from THAT TASK'S own data. Picture
    naming (main figure) and auditory naming (supplementary) became separate figures on
    2026-08-11, so each fills its own axes and neither caption claims cross-task
    comparability; only the uncaptioned cross-task grid keeps one scale across tasks,
    which is the whole point of that view.

Metrics (from the per-epoch PKL arrays, cached to panels_cache_{task}_{emb}.npz):
  1. category_indep — independent-centroid balanced category accuracy
  2. word_top1 / word_top3 / word_top5 — raw top-k word-retrieval accuracy

Outputs (this folder):
  00_legend.pdf/.png
  01_picture_category_indep … 04_picture_word_top5          (per-metric, picture)
  05_auditory_category_indep … 08_auditory_word_top5        (per-metric, auditory)
  09_combined_picture     — MAIN paper figure (2×2, a–d);  caption: caption.md
  10_combined_auditory    — SUPPLEMENTARY figure (2×2, a–d); caption: caption_auditory.md
  11_combined_both_tasks  — 4 metrics × 2 tasks (a–h). Kept as an internal cross-task
                            view; it is NOT a paper figure and is deliberately uncaptioned
  caption.md, caption_auditory.md                           (PDFs: pdf.fonttype 42)
  source_data/source_data.csv     — per task × metric × patient × bin: obs, chance,
                                    null threshold, permutation p, significant
  source_data/cue_timing.csv      — per task: aggregated cue mean ± s.d.
  source_data/peak_rise_stats.csv — per task × metric: peak acc ± s.e.m., empirical
                                    chance, rise/peak latency mean ± s.d. (Results text)

Reproduce:
  # fast path — uses the cached arrays in this folder:
  python figures_for_paper/semantic_regression/semantic_regression_panels.py
  # rebuild one task's cache from the (large) result PKLs (both tasks still rendered):
  python figures_for_paper/semantic_regression/semantic_regression_panels.py \
      --rebuild-cache auditory
(run with cwd = main/, in the Speech conda env — the result PKLs need dill)
"""

import os
import sys
import json
import argparse
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch

# Progress lines carry ± / → ; the Windows console defaults to cp1252 and would raise.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')

# Editable-text vector output
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS_ROOT = os.path.dirname(HERE)                          # …/figures_for_paper
MAIN_DIR = os.path.dirname(FIGS_ROOT)                      # …/main
sys.path.insert(0, FIGS_ROOT)                              # shared figure conventions
from paper_common import (display_id, assign_colors,       # noqa: E402
                          load_cue_style)                  # participant/cue style from config
from utils.config import PIC_RUN, AUD_RUN_FIGURE, ALPHA, PCTILE   # noqa: E402  (pinned runs + cutoff)
RESULTS_DIR = os.path.join(MAIN_DIR, 'results', 'semantic_regression')
FIG_DIR = HERE
SRC_DIR = os.path.join(HERE, 'source_data')

# ── Tasks ─────────────────────────────────────────────────────────────────────
# key → (figure label, results run). Each run supplies its own patients, alignment cue
# and time base (read from its meta.json), so the two tasks need not match in either.
#
# `role` + `caption` pin the figure↔caption pairing in one place. The two tasks ship as TWO
# figures as of 2026-08-11 (Alec's call): picture naming is the main figure, auditory naming
# a supplementary one, each with its own caption. Before that both were one a–h grid under a
# single caption.md. A caption is generated per task, so adding a task here adds its caption
# rather than silently leaving a panel undescribed.
TASKS = OrderedDict([
    ('picture', dict(
        label='Picture naming',
        role='main', caption='caption.md',
        # `fig_label` + `fig_title` open the caption as Nature legends do: "Figure | <title
        # sentence>." The title is a noun phrase with no verb and no result in it, and it is
        # editorial — write it here rather than deriving it from a metric name. The number is
        # left out because it is assigned when the manuscript is assembled, not here.
        fig_label='Figure',
        fig_title='Cross-patient semantic-decoding using a regression-retrieval based decoder',
        run_dir=os.path.join(RESULTS_DIR, PIC_RUN))),
    # AUD_RUN_FIGURE, not AUD_RUN. The two tasks are NOT matched: picture is 13 regions at
    # 5 bins, auditory is 23 regions (tpfm: + frontal + medial) at 10. They differ on BOTH
    # the channel set and the feature window, and the caption says so -- "temporal-parietal
    # cortex" does not describe the auditory arm. Rationale and the two unseparated
    # explanations are in utils/config.py beside the constant, and in
    # docs/experiments/001-history-and-scope-diagnostic.md. Every OTHER auditory analysis
    # in the repo still reads AUD_RUN at tp/5 bins.
    ('auditory', dict(
        label='Auditory naming',
        role='supplementary', caption='caption_auditory.md',
        fig_label='Supplementary Figure',
        fig_title='Cross-patient semantic-decoding during auditory naming',
        run_dir=os.path.join(RESULTS_DIR, AUD_RUN_FIGURE))),
])

EMBEDDING = 'GloVe'
# Significance cutoff: a bin is significant iff obs mean > this percentile of the
# null. PCTILE = 100*(1-ALPHA) is derived from the repo-wide ALPHA in
# utils/config.py — never retype it, and never annotate a p-value that was not
# computed from it (figures_for_paper/README.md §5).

# ── Metric definitions ────────────────────────────────────────────────────────
# key → (pretty label, obs attr, null attr, family)  — panels in a family share y.
# The caption is generated from this list, so every panel is always described; add a
# matching PANEL_CAPTION entry when adding a metric.
METRICS = [
    ('category_indep', 'Category accuracy',
     'all_retrieval_category_indep_balanced_acc',
     'all_retrieval_category_indep_chance_balanced_acc', 'category'),
    ('word_top1', 'Word top-1 accuracy',
     'all_retrieval_top1', 'all_retrieval_chance_top1', 'word'),
    ('word_top3', 'Word top-3 accuracy',
     'all_retrieval_top3', 'all_retrieval_chance_top3', 'word'),
    ('word_top5', 'Word top-5 accuracy',
     'all_retrieval_top5', 'all_retrieval_chance_top5', 'word'),
]

# Per-panel caption phrase (key → sentence describing that panel); falls back to the
# pretty label if a key is missing, so the caption always covers every panel.
PANEL_CAPTION = {
    'category_indep': 'Independent balanced category accuracy',
    'word_top1': 'Top-1 word-retrieval accuracy',
    'word_top3': 'Top-3 word-retrieval accuracy',
    'word_top5': 'Top-5 word-retrieval accuracy',
}

# roi_scope token (meta.json) → the phrase a caption uses for it. "temporal-parietal" stops
# being true of a run gated on a wider scope, so a caption never says it unless the run did.
_SCOPE_TEXT = {
    'tp': 'the 13-region temporal-parietal whitelist',
    'tpm': 'an 18-region set extending the temporal-parietal whitelist with medial/deep '
           'regions',
    'tpfm': 'a 23-region set extending the temporal-parietal whitelist with frontal '
            'and medial/deep regions',
}

# Cue marker colours/labels — from figures_for_paper/cue_style.json (shared config).
CUE_STYLE = load_cue_style()


# ── Cache construction (extract small arrays from big PKLs) ────────────────────

def _patient_dirs(run_dir):
    return sorted(
        d for d in os.listdir(run_dir)
        if os.path.isdir(os.path.join(run_dir, d))
        and not d.endswith('.json') and d not in ('report', '__pycache__')
    )


def cache_path_for(task, embedding=EMBEDDING):
    return os.path.join(FIG_DIR, f'panels_cache_{task}_{embedding}.npz')


def build_cache(run_dir, cache_path, embedding=EMBEDDING):
    """Load each patient PKL once, extract per-epoch obs/null arrays for the
    requested embedding + all metrics, and per-patient cue means. Save to npz."""
    # Lazy import: only needed when rebuilding (normal runs use the cache).
    sys.path.insert(0, MAIN_DIR)
    from report.helper.results_loader import load_pkl_raw

    patients = _patient_dirs(run_dir)
    arrays, cues, kept = {}, {}, []
    for p in patients:
        pkl_path = os.path.join(run_dir, p, 'semantic_regression_results.pkl')
        if not os.path.exists(pkl_path):
            continue
        print(f"  [cache] loading {p} ...", flush=True)
        try:
            data = load_pkl_raw(pkl_path)
        except Exception as e:
            print(f"  [cache] {p}: FAILED ({e})", flush=True)
            continue
        if data is None or embedding not in data.get('regressors', {}):
            print(f"  [cache] {p}: no '{embedding}' regressor — skipped", flush=True)
            continue
        br = data['regressors'][embedding]
        ok = True
        for key, _l, obs_attr, null_attr, _f in METRICS:
            if not (hasattr(br, obs_attr) and hasattr(br, null_attr)):
                print(f"  [cache] {p}: missing {obs_attr}/{null_attr} — skipped", flush=True)
                ok = False
                break
            arrays[f'{p}__{key}__obs'] = np.asarray(getattr(br, obs_attr), dtype=np.float32)
            arrays[f'{p}__{key}__null'] = np.asarray(getattr(br, null_attr), dtype=np.float32)
        if not ok:
            continue
        cs = {}
        cs_path = os.path.join(run_dir, p, 'cue_stats.json')
        if os.path.exists(cs_path):
            for cue, v in json.load(open(cs_path)).get('rel_cues', {}).items():
                m = v.get('mean')
                if m is not None and np.isfinite(m):
                    cs[cue] = float(m)
        cues[p] = cs
        kept.append(p)
        del data, br

    if not kept:
        raise RuntimeError(
            f"no patients cached from {run_dir} (embedding {embedding!r}) — "
            "nothing to plot; check the run and the embedding name")

    side = {'patients': kept, 'cues': cues, 'embedding': embedding,
            'run_dir': os.path.abspath(run_dir)}
    np.savez_compressed(cache_path, **arrays)
    with open(cache_path + '.json', 'w') as f:
        json.dump(side, f)
    print(f"  [cache] saved {len(kept)} patients → {os.path.basename(cache_path)}", flush=True)
    return {'arrays': dict(np.load(cache_path)), 'side': side}


def load_cache(cache_path, run_dir=None):
    """Return the cached arrays, or None (→ rebuild) if the cache is absent or was
    built from a different run. The cache is keyed by task, not by run, so without
    this check repointing TASKS at a new run silently re-plots the old one."""
    if not (os.path.exists(cache_path) and os.path.exists(cache_path + '.json')):
        return None
    side = json.load(open(cache_path + '.json'))
    if run_dir is not None:
        cached = side.get('run_dir')
        if cached is None or os.path.abspath(cached) != os.path.abspath(run_dir):
            print(f"  [cache] {os.path.basename(cache_path)} was built from "
                  f"{cached or 'an unrecorded run'} — stale, rebuilding", flush=True)
            return None
    return {'arrays': dict(np.load(cache_path)), 'side': side}


# ── Statistics ────────────────────────────────────────────────────────────────

def perbin_significance(obs, null, pctile=PCTILE):
    """Per-bin permutation test: a bin is significant iff the observed mean
    accuracy exceeds the `pctile`-th percentile of the shuffled-null distribution
    at that bin (one-sided; the default pctile=100*(1-ALPHA)=95 ⇒ p<0.05). This
    compares the data directly against the full shuffled distribution — it does not
    inflate with epoch count the way a t-test on a tiny reliable offset does, which
    is why the test family is settled (docs/agent-context/scientific-integrity.md).
    obs/null are (n_epochs, n_bins).
    Returns (sig_mask, p_perm, null_thresh, obs_mean, null_mean), where p_perm is
    the empirical one-sided permutation p-value and null_thresh is the percentile."""
    n_epochs = null.shape[0]
    null_mean = null.mean(0)
    obs_mean = obs.mean(0)
    thr = np.percentile(null, pctile, axis=0)
    sig = obs_mean > thr
    # empirical one-sided permutation p-value: P(null >= observed mean)
    p_perm = np.array([(np.sum(null[:, b] >= obs_mean[b]) + 1) / (n_epochs + 1)
                       for b in range(null.shape[1])])
    return sig, p_perm, thr, obs_mean, null_mean


def _group_mean_curve(per_patient, patients, attr):
    """Across-participant mean of `attr` on the union time grid (participants may
    have different bin counts). Returns (times, mean_curve, n_patients_per_bin)
    sorted by time — the count matters because the tail of the union grid is carried
    by the one or two participants with the longest trials."""
    from collections import defaultdict
    acc = defaultdict(list)
    for p in patients:
        for tv, yv in zip(per_patient[p]['time_s'], per_patient[p][attr]):
            acc[round(float(tv), 6)].append(float(yv))
    times = np.array(sorted(acc))
    mean = np.array([float(np.mean(acc[t])) for t in times])
    counts = np.array([len(acc[t]) for t in times])
    return times, mean, counts


def compute_peak_rise_stats(results, patients):
    """Per-metric summary numbers used in the Results text — recomputed from
    whatever participants are present, so they stay correct as the cohort grows.

    Per metric:
      * peak accuracy = across-participant mean at the group peak bin (t* = argmax of
        the across-participant mean curve over t>=0), with s.e.m.; `emp_chance` is the
        mean permuted-null at t*. t* is searched only over bins ALL participants cover:
        participants have different trial lengths (auditory: 83–91 bins), and the tail of
        the union grid is a single long-trial participant, whose accuracy would otherwise
        be reported as the group peak.
      * peak/rise latencies are per-participant, averaged over participants that show
        ANY significant bin (rise is only defined there); reported as mean ± s.d.
        Rise = onset of the first significant bin (t>=0); peak = argmax over t>=0.
    Returns a tidy DataFrame (one row per metric, in METRICS order)."""
    rows = []
    for key, label, *_rest in METRICS:
        pp = results[key]['per_patient']
        # group peak bin (t>=0) on the across-participant mean curve, over bins the
        # whole cohort covers (see docstring)
        gt, gm, gn = _group_mean_curve(pp, patients, 'obs_mean')
        pos = (gt >= 0) & (gn == len(patients))
        t_star = float(gt[pos][np.argmax(gm[pos])])
        # per-participant obs / chance at t* (skip participants lacking that bin)
        at_star = [(p, pp[p]) for p in patients]
        obs_star = [float(d['obs_mean'][np.isclose(d['time_s'], t_star)][0])
                    for _p, d in at_star if np.any(np.isclose(d['time_s'], t_star))]
        chance_star = [float(d['null_mean'][np.isclose(d['time_s'], t_star)][0])
                       for _p, d in at_star if np.any(np.isclose(d['time_s'], t_star))]
        obs_star = np.array(obs_star)
        peak_acc = obs_star.mean()
        peak_sem = obs_star.std(ddof=1) / np.sqrt(len(obs_star)) if len(obs_star) > 1 else np.nan
        emp_chance = float(np.mean(chance_star))
        # per-participant peak / rise latencies over significant participants
        peak_ts, rise_ts = [], []
        for p in patients:
            d = pp[p]
            t = d['time_s']
            m = t >= 0
            if not np.any(d['sig']):
                continue  # no significant decoding → latency undefined
            peak_ts.append(float(t[m][np.argmax(d['obs_mean'][m])]))
            rise_ts.append(float(t[d['sig']].min()))
        peak_ts, rise_ts = np.array(peak_ts), np.array(rise_ts)

        def _ms(a):
            return (a.mean(), a.std(ddof=1)) if len(a) > 1 else (
                (a[0], np.nan) if len(a) == 1 else (np.nan, np.nan))
        pk_m, pk_sd = _ms(peak_ts)
        rs_m, rs_sd = _ms(rise_ts)
        rows.append(dict(
            metric=key, label=label, n_total=len(patients), n_sig=len(peak_ts),
            peak_acc_mean=peak_acc, peak_acc_sem=peak_sem, emp_chance=emp_chance,
            peak_bin_time_s=t_star,
            peak_time_mean_s=pk_m, peak_time_sd_s=pk_sd,
            rise_time_mean_s=rs_m, rise_time_sd_s=rs_sd,
        ))
    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _aggregate_cues(cues, patients, xlim=None, align_cue=None):
    """cue → (mean_time, std_time) across patients. The run's own alignment cue is
    skipped (it sits at 0 for every participant by construction). Cues whose mean
    falls outside `xlim` are skipped too — the auditory window ends before the go cue
    / voice onset, and an off-panel cue would otherwise appear in the legend with
    nothing to show.

    A cue with zero spread is KEPT and drawn as a single crisp line. Under group time
    warping the stimulus interval is identical across participants, so zero spread is
    a real, maximally precise measurement — not a degenerate one to be dropped."""
    out = {}
    for cue in CUE_STYLE:
        if align_cue is not None and cue == align_cue:
            continue
        vals = [cues[p][cue] for p in patients if cue in cues.get(p, {})]
        vals = [v for v in vals if np.isfinite(v)]
        if len(vals) >= 2:
            s = float(np.std(vals))
            m = float(np.mean(vals))
            if xlim is not None and not (xlim[0] <= m <= xlim[1]):
                continue
            out[cue] = (m, s)
    return out


def _time_axis(n_bins, n_bins_history, bin_size_ms):
    return np.array([(b - n_bins_history) * bin_size_ms / 1000.0 for b in range(n_bins)])


def _draw_panel(ax, label, per_patient, patients, color_of, cue_agg, bin_size_s,
                y_top, align_label, chance_t, chance_mean, pctile=PCTILE,
                panel_letter=None):
    """Draw one metric panel onto `ax`. per_patient[p] = dict(obs_mean, null_mean,
    sig, time_s). Patients may have different bin counts → each uses its own axis.
    Traces keep each participant's fixed colour; the significance raster rows are
    ordered by peak accuracy (highest at the top, lowest at the bottom)."""
    raster_top = -0.03 * y_top
    raster_bottom = -0.34 * y_top
    row_h = (raster_top - raster_bottom) / max(len(patients), 1)
    xmin = min(per_patient[p]['time_s'][0] for p in patients)
    xmax = max(per_patient[p]['time_s'][-1] for p in patients)

    for cue, (mu, sd) in cue_agg.items():
        st = CUE_STYLE[cue]
        if sd > 0:
            ax.axvspan(mu - sd, mu + sd, color=st['color'], alpha=0.08, lw=0, zorder=0)
            ax.axvline(mu, color=st['color'], lw=1.0, ls='-', alpha=0.55, zorder=1)
        else:
            # identical across participants (group-warped stimulus interval) — the
            # cue time is exact, so draw one crisp line and no uncertainty band
            ax.axvline(mu, color=st['color'], lw=1.3, ls='-', alpha=0.85, zorder=1)

    ax.axvline(0, color='black', lw=0.9, ls=':', zorder=1)
    ax.axhline(0, color='#999999', lw=0.6, zorder=1)

    # decoding traces — fixed per-participant colour, order irrelevant
    for p in patients:
        t = per_patient[p]['time_s']
        ax.plot(t, per_patient[p]['obs_mean'], color=color_of[p], lw=1.2, alpha=0.9, zorder=3)

    # significance raster — sort rows by peak accuracy, highest at the top
    raster_order = sorted(patients, key=lambda p: np.nanmax(per_patient[p]['obs_mean']),
                          reverse=True)
    for i, p in enumerate(raster_order):
        t = per_patient[p]['time_s']
        sig = per_patient[p]['sig']
        y0 = raster_top - (i + 1) * row_h
        segs = [(t[b] - bin_size_s / 2, bin_size_s) for b in range(len(sig)) if sig[b]]
        if segs:
            ax.broken_barh(segs, (y0, row_h * 0.9), facecolors=color_of[p],
                           edgecolors='none', zorder=2)

    ax.plot(chance_t, chance_mean, color='#444444', lw=1.1, ls='--', alpha=0.8,
            zorder=4, label='mean shuffled chance')

    ax.set_xlabel(align_label)
    ax.set_ylabel(label)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(raster_bottom - 0.02 * y_top, y_top)
    yt = [t for t in ax.get_yticks() if t >= -1e-9]
    ax.set_yticks(yt)
    ax.set_yticklabels([f'{t:.2f}' for t in yt])
    ax.text(xmin, (raster_top + raster_bottom) / 2, f'sig.\n(p<{1 - pctile/100:.2g})',
            fontsize=6.5, color='#555555', ha='right', va='center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if panel_letter is not None:
        ax.annotate(panel_letter, xy=(0, 1), xycoords='axes fraction',
                    xytext=(-44, 16), textcoords='offset points',
                    fontsize=12, fontweight='bold', va='bottom', ha='left')


def plot_panel(label, per_patient, patients, color_of, cue_agg, bin_size_s,
               y_top, align_label, chance_t, chance_mean, pctile=PCTILE):
    """Render one metric panel as a standalone figure."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    _draw_panel(ax, label, per_patient, patients, color_of, cue_agg, bin_size_s,
                y_top, align_label, chance_t, chance_mean, pctile=pctile)
    fig.tight_layout()
    return fig


def _merge_cue_spread(*cue_aggs):
    """cue → True if it has non-zero across-participant spread in any task where it is
    drawn. Decides whether the legend entry reads '±1 s.d.' (a band) or 'aligned'
    (a crisp line, e.g. the group-warped stimulus offset)."""
    out = OrderedDict()
    for agg in cue_aggs:
        for cue, (_m, s) in agg.items():
            out[cue] = out.get(cue, False) or (s > 0)
    return out


def _legend_handles(patients, color_of, cue_spread):
    handles = []
    for p in patients:
        handles.append(mlines.Line2D([], [], color=color_of[p], lw=2, label=display_id(p)))
    handles.append(mlines.Line2D([], [], color='#444444', lw=1.5, ls='--', label='mean chance'))
    for cue, has_spread in cue_spread.items():
        st = CUE_STYLE[cue]
        if has_spread:
            handles.append(Patch(facecolor=st['color'], alpha=0.3,
                                 label=f"{st['label']} (±1 s.d.)"))
        else:
            handles.append(mlines.Line2D([], [], color=st['color'], lw=1.3,
                                         label=f"{st['label']} (aligned)"))
    return handles


def legend_figure(patients, color_of, cue_spread):
    fig, ax = plt.subplots(figsize=(6.4, 1.1))
    ax.axis('off')
    ax.legend(handles=_legend_handles(patients, color_of, cue_spread),
              ncol=6, loc='center', fontsize=7.5, frameon=False)
    fig.tight_layout()
    return fig


def _layout_above_legend(fig, leg, pad=0.025):
    """Lay the axes out in whatever height the legend leaves, measured rather than guessed.

    A fixed `tight_layout(rect=...)` bottom cannot do this: the legend gains a row every time
    the entry count crosses a multiple of `ncol`, and at N=15 (20 entries over 6 columns, so
    four rows) it ran straight through the bottom row's x-axis labels. The legend has no
    extent until the figure is drawn, hence the draw before the measurement."""
    fig.canvas.draw()
    h_px = leg.get_window_extent(fig.canvas.get_renderer()).height
    frac = h_px / (fig.get_size_inches()[1] * fig.dpi)
    fig.tight_layout(rect=(0, min(frac + pad, 0.45), 1, 1))


def plot_combined(task, color_of, pctile=PCTILE):
    """Nature-style 2×2 combined figure of the four metric panels (a–d) for one task,
    with a shared participant/cue legend below."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    for i, (ax, (key, _l, _oa, _na, _f)) in enumerate(zip(axes.ravel(), METRICS)):
        d = task['results'][key]
        ct, cm = task['chance_curves'][key]
        _draw_panel(ax, d['label'], d['per_patient'], task['patients'], color_of,
                    task['cue_agg'], task['bin_size_s'], task['fam_top'][d['family']],
                    task['align_label'], ct, cm, pctile=pctile,
                    panel_letter=_panel_letter(i))
    leg = fig.legend(handles=_legend_handles(task['patients'], color_of,
                                             _merge_cue_spread(task['cue_agg'])),
                     ncol=6, loc='lower center', fontsize=7, frameon=False,
                     bbox_to_anchor=(0.5, 0.0))
    _layout_above_legend(fig, leg)
    return fig


def plot_combined_tasks(tasks, all_patients, color_of, fam_top, pctile=PCTILE):
    """Cross-task combined figure: rows = metrics, columns = tasks. Panel letters run
    column-major (a–d = first task, e–h = second), so each column reads top-to-bottom.
    Columns share the per-family y-scale `fam_top` — taken across tasks, NOT each task's
    own, which is what makes the accuracy difference between tasks readable and is the
    only reason this view exists now that the two tasks ship as separate figures. Each
    column keeps its own alignment cue, time base and cues."""
    n_rows, n_cols = len(METRICS), len(tasks)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 2.75 * n_rows),
                             squeeze=False)
    for c, (tkey, task) in enumerate(tasks.items()):
        for r, (key, _l, _oa, _na, _f) in enumerate(METRICS):
            ax = axes[r][c]
            d = task['results'][key]
            ct, cm = task['chance_curves'][key]
            _draw_panel(ax, d['label'], d['per_patient'], task['patients'], color_of,
                        task['cue_agg'], task['bin_size_s'], fam_top[d['family']],
                        task['align_label'], ct, cm, pctile=pctile,
                        panel_letter=_panel_letter(c * n_rows + r))
            if r == 0:
                ax.set_title(f"{task['label']} (N={len(task['patients'])})",
                             fontsize=10, fontweight='bold', pad=10)
    cue_spread = _merge_cue_spread(*(t['cue_agg'] for t in tasks.values()))
    leg = fig.legend(handles=_legend_handles(all_patients, color_of, cue_spread),
                     ncol=6, loc='lower center', fontsize=7, frameon=False,
                     bbox_to_anchor=(0.5, 0.0))
    _layout_above_legend(fig, leg)
    return fig


# ── Orchestration ─────────────────────────────────────────────────────────────

def _load_task(task_key, run_dir, embedding=EMBEDDING, rebuild_cache=False, pctile=PCTILE):
    """Load one task's run: cache → per-patient stats → chance curves → cue timing.
    Returns everything the panel drawers need (family y-tops are filled in later, once
    both tasks are known, so the two share one scale)."""
    meta_path = os.path.join(run_dir, 'meta.json')
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    n_bins_history = meta.get('n_bins_history', 10)
    bin_size_ms = meta.get('bin_size_ms', 100)
    align_cue = meta.get('align_cue') or 'trial_onset'
    if align_cue in (None, 'none', ''):
        align_cue = 'trial_onset'
    # human cue name from the shared cue_style config (falls back to the raw key)
    align_name = CUE_STYLE.get(align_cue, {}).get('label', align_cue.replace('_', ' ')).lower()
    align_label = f"Time from {align_name} (s)"

    cache_path = cache_path_for(task_key, embedding)
    cache = None if rebuild_cache else load_cache(cache_path, run_dir)
    if cache is None:
        print(f"[panels] {task_key}: building cache from PKLs (one-time, slow)...", flush=True)
        cache = build_cache(run_dir, cache_path, embedding=embedding)
    arrays, side = cache['arrays'], cache['side']
    patients = side['patients']

    results = {}
    for key, label, _oa, _na, fam in METRICS:
        per_patient = {}
        for p in patients:
            obs = arrays[f'{p}__{key}__obs']
            null = arrays[f'{p}__{key}__null']
            sig, pv, thr, obs_m, null_m = perbin_significance(obs, null, pctile=pctile)
            t = _time_axis(obs.shape[1], n_bins_history, bin_size_ms)
            # We make no decoding claim before the alignment cue (t=0), so pre-onset bins
            # are never marked significant — in the raster or the source data.
            sig = sig & (t >= 0)
            per_patient[p] = dict(obs_mean=obs_m, null_mean=null_m, sig=sig, time_s=t,
                                  p_perm=pv, null_thresh=thr)
        results[key] = dict(per_patient=per_patient, family=fam, label=label)

    def _chance_curve(per_patient):
        max_bins = max(len(per_patient[p]['null_mean']) for p in patients)
        grid_t = _time_axis(max_bins, n_bins_history, bin_size_ms)
        chance = np.full(max_bins, np.nan)
        for b in range(max_bins):
            vals = [per_patient[p]['null_mean'][b] for p in patients
                    if b < len(per_patient[p]['null_mean'])]
            if vals:
                chance[b] = float(np.mean(vals))
        return grid_t, chance

    chance_curves = {key: _chance_curve(results[key]['per_patient'])
                     for key, *_r in METRICS}

    # x-range of this task's panels — cues outside it are dropped (see _aggregate_cues)
    any_pp = results[METRICS[0][0]]['per_patient']
    xlim = (min(any_pp[p]['time_s'][0] for p in patients),
            max(any_pp[p]['time_s'][-1] for p in patients))
    cue_agg = _aggregate_cues(side['cues'], patients, xlim=xlim, align_cue=align_cue)

    return dict(key=task_key, label=TASKS[task_key]['label'],
                role=TASKS[task_key]['role'], caption=TASKS[task_key]['caption'],
                fig_label=TASKS[task_key]['fig_label'],
                fig_title=TASKS[task_key]['fig_title'],
                run_dir=run_dir, meta=meta,
                patients=patients, results=results, chance_curves=chance_curves,
                cue_agg=cue_agg, bin_size_s=bin_size_ms / 1000.0,
                # Resolved, not raw meta, so a run predating a key still captions correctly.
                # A run with no `roi_scope` predates the scope axis (added 2026-08-11) and is
                # `tp` by definition — the same rule the run-id token follows. PIC_RUN is such
                # a run, and without this default the caption reads "the None region set".
                n_bins_history=n_bins_history, bin_size_ms=bin_size_ms,
                roi_scope=meta.get('roi_scope') or 'tp',
                align_cue=align_cue, align_name=align_name, align_label=align_label,
                fam_top={}, fig_stem=None)


def generate_panels(rebuild_cache=(), run_dirs=None, embedding=EMBEDDING, pctile=PCTILE):
    """Build every figure + source-data table. Always renders ALL tasks in TASKS (the
    figures, the panel numbering and the source-data tables are cross-task, so a partial
    render would leave them inconsistent); `rebuild_cache` is the set of task keys whose
    per-epoch arrays are re-extracted from the result PKLs instead of read from the
    cached npz."""
    os.makedirs(SRC_DIR, exist_ok=True)
    run_dirs = run_dirs or {}

    tasks = OrderedDict(
        (k, _load_task(k, run_dirs.get(k, TASKS[k]['run_dir']), embedding=embedding,
                       rebuild_cache=(k in rebuild_cache), pctile=pctile))
        for k in TASKS
    )

    # y-scale is shared within a metric family but NOT across tasks. The two tasks ship as
    # two figures (picture = main, auditory = supplementary) as of 2026-08-11, so each is
    # scaled to its own data and neither caption claims the other's scale. The cross-task
    # grid still needs the across-task scale — comparability is the only thing that view
    # adds — so both are computed: per-task onto the task, across-task as shared_fam_top.
    def _fam_top(task):
        out = {}
        for d in task['results'].values():
            top = max(np.nanmax(d['per_patient'][p]['obs_mean']) for p in task['patients'])
            out[d['family']] = max(out.get(d['family'], 0.0), top)
        return {f: v * 1.10 for f, v in out.items()}

    for task in tasks.values():
        task['fam_top'] = _fam_top(task)
    shared_fam_top = {}
    for task in tasks.values():
        for f, v in task['fam_top'].items():
            shared_fam_top[f] = max(shared_fam_top.get(f, 0.0), v)

    # one colour per participant, over the union of tasks → same colour in every column
    all_patients = list(OrderedDict((p, None) for t in tasks.values() for p in t['patients']))
    color_of = dict(zip(all_patients, assign_colors(all_patients)))

    # ── per-metric panels, numbered task-major: 01–04 picture, 05–08 auditory ──
    n_metrics = len(METRICS)
    for ti, (tkey, task) in enumerate(tasks.items()):
        for mi, (key, label, _oa, _na, fam) in enumerate(METRICS):
            d = task['results'][key]
            ct, cm = task['chance_curves'][key]
            fig = plot_panel(label, d['per_patient'], task['patients'], color_of,
                             task['cue_agg'], task['bin_size_s'], task['fam_top'][fam],
                             task['align_label'], ct, cm, pctile=pctile)
            stem = os.path.join(FIG_DIR, f"{ti * n_metrics + mi + 1:02d}_{tkey}_{key}")
            fig.savefig(stem + '.pdf', bbox_inches='tight')
            fig.savefig(stem + '.png', dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"  [panels] {tkey}/{key}: saved {os.path.basename(stem)}.pdf/.png", flush=True)

    # ── 2×2 combined per task (09 = main figure, 10 = supplementary), then the
    #    uncaptioned cross-task grid ──
    n_single = len(tasks) * n_metrics
    for ti, (tkey, task) in enumerate(tasks.items()):
        fig = plot_combined(task, color_of, pctile=pctile)
        stem = os.path.join(FIG_DIR, f"{n_single + ti + 1:02d}_combined_{tkey}")
        # the caption names the file it captions, and takes the name from the render
        task['fig_stem'] = os.path.basename(stem)
        fig.savefig(stem + '.pdf', bbox_inches='tight')
        fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [panels] combined: saved {os.path.basename(stem)}.pdf/.png", flush=True)

    grid_stem = None
    if len(tasks) > 1:
        fig = plot_combined_tasks(tasks, all_patients, color_of, shared_fam_top,
                                  pctile=pctile)
        both_stem = os.path.join(FIG_DIR, f"{n_single + len(tasks) + 1:02d}_combined_both_tasks")
        grid_stem = os.path.basename(both_stem)
        fig.savefig(both_stem + '.pdf', bbox_inches='tight')
        fig.savefig(both_stem + '.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [panels] combined: saved {os.path.basename(both_stem)}.pdf/.png", flush=True)

    leg = legend_figure(all_patients, color_of,
                        _merge_cue_spread(*(t['cue_agg'] for t in tasks.values())))
    leg.savefig(os.path.join(FIG_DIR, '00_legend.pdf'), bbox_inches='tight')
    leg.savefig(os.path.join(FIG_DIR, '00_legend.png'), dpi=200, bbox_inches='tight')
    plt.close(leg)

    # ── source data (one table per file, `task` column distinguishes the runs) ──
    src_rows, cue_rows, stat_frames = [], [], []
    for tkey, task in tasks.items():
        for key, *_r in METRICS:
            pp = task['results'][key]['per_patient']
            for p in task['patients']:
                d = pp[p]
                did = display_id(p)
                for b in range(len(d['time_s'])):
                    src_rows.append(dict(
                        task=tkey, metric=key, display_id=did, patient=p,
                        bin_index=b, time_s=d['time_s'][b],
                        obs_mean=d['obs_mean'][b], chance_mean=d['null_mean'][b],
                        null_p=d['null_thresh'][b], p_perm=d['p_perm'][b],
                        significant=bool(d['sig'][b])))
        for c, (m, s) in task['cue_agg'].items():
            cue_rows.append(dict(task=tkey, cue=c, mean_s=m, std_s=s))
        st = compute_peak_rise_stats(task['results'], task['patients'])
        st.insert(0, 'task', tkey)
        stat_frames.append(st)

    pd.DataFrame(src_rows).to_csv(os.path.join(SRC_DIR, 'source_data.csv'), index=False)
    pd.DataFrame(cue_rows).to_csv(os.path.join(SRC_DIR, 'cue_timing.csv'), index=False)

    # Results-text summary numbers (peak accuracy, empirical chance, rise/peak
    # latencies) — recomputed from the current cohort so the paragraph stays correct
    # as participants are added.
    stats = pd.concat(stat_frames, ignore_index=True)
    stats.to_csv(os.path.join(SRC_DIR, 'peak_rise_stats.csv'), index=False)
    print("  [panels] peak/rise stats (Results text):")
    for r in stats.itertuples(index=False):
        print(f"    {r.task:9s} {r.metric:14s} peak {r.peak_acc_mean:.3f}±{r.peak_acc_sem:.3f} "
              f"vs chance {r.emp_chance:.3f} | rise {r.rise_time_mean_s:.2f}±{r.rise_time_sd_s:.2f}s "
              f"peak {r.peak_time_mean_s:.2f}±{r.peak_time_sd_s:.2f}s (n_sig={r.n_sig}/{r.n_total})",
              flush=True)

    _write_captions(tasks, embedding, pctile, grid_stem=grid_stem)
    print(f"[panels] figures + captions in: {FIG_DIR}")
    print(f"[panels] source data in:       {SRC_DIR}")


def _panel_letter(i):
    """0-based panel index → letter(s): a, b, …, z, aa, ab, …"""
    s = ''
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(97 + r) + s
    return s


def _common_tail_words(caps, min_words=2):
    """Longest word-aligned common suffix of `caps`, or '' if shorter than `min_words`.
    Lets 'Top-1/3/5 word-retrieval accuracy' collapse into one lettered range without the
    shared noun phrase being typed into the module a second time."""
    words = [c.split() for c in caps]
    n = 0
    while n < min(len(w) for w in words) and len({w[-(n + 1)] for w in words}) == 1:
        n += 1
    return ' '.join(words[0][len(words[0]) - n:]) if n >= min_words else ''


def _metric_panels_text():
    """Panel descriptions for one task's 2×2 figure, Nature style: bold letter, no period
    after the letter, and consecutive panels of one metric family collapsed into a lettered
    range ending in 'respectively' ("**b**–**d** Top-1, top-3 and top-5 word-retrieval
    accuracy, respectively."). Generated from METRICS in the order `plot_combined` lays them
    out, so a metric cannot be added without its caption sentence appearing. Both shipped
    figures run a–d; only the uncaptioned cross-task grid runs a–h."""
    groups = []                       # consecutive runs of one family, in panel order
    for i, (key, label, _oa, _na, fam) in enumerate(METRICS):
        item = (_panel_letter(i), PANEL_CAPTION.get(key, label))
        if groups and groups[-1][0] == fam:
            groups[-1][1].append(item)
        else:
            groups.append((fam, [item]))

    out = []
    for _fam, items in groups:
        shared = _common_tail_words([c for _l, c in items]) if len(items) > 1 else ''
        if not shared:
            # single panel, or a family whose labels share no phrase to factor out
            out.extend('**{}** {}.'.format(letter, cap) for letter, cap in items)
            continue
        heads = [cap[:len(cap) - len(shared)].strip() for _l, cap in items]
        heads = heads[:1] + [h[:1].lower() + h[1:] for h in heads[1:]]
        out.append('**{}**–**{}** {} and {} {}, respectively.'.format(
            items[0][0], items[-1][0], ', '.join(heads[:-1]), heads[-1], shared))
    return ' '.join(out)


def _config_phrase(task, with_bins=False):
    """'<region set> and <window> ms of preceding activity per time point' — every number
    read from that run's own meta.json. `with_bins` appends the bin decomposition, which
    belongs in the notes below the caption rather than in it (Nature legends minimise
    methodological detail)."""
    h, b = task['n_bins_history'], task['bin_size_ms']
    txt = '{} and {:,} ms of preceding activity per time point'.format(
        _SCOPE_TEXT.get(task['roi_scope'], 'the {} region set'.format(task['roi_scope'])),
        int(h * b))
    return txt + ' ({} bins × {} ms)'.format(h, b) if with_bins else txt


def _sibling(task, tasks):
    """The other task, or None when only one is rendered."""
    others = [t for k, t in tasks.items() if k != task['key']]
    return others[0] if len(others) == 1 else None


def _matched(task, sib):
    """True when the two runs share channel scope AND history window — i.e. when a
    picture-vs-auditory comparison would be a controlled contrast."""
    return ((task['roi_scope'], task['n_bins_history'])
            == (sib['roi_scope'], sib['n_bins_history']))


def _scale_sentence(task, tasks):
    """Two things a reader cannot see: which panels share a y axis, and that the scale is
    set per figure — so the supplementary figure must not be compared with the main one by
    eye. Says the opposite of what it said while both tasks lived in one a–h grid."""
    groups = OrderedDict()
    for key, label, _oa, _na, fam in METRICS:
        groups.setdefault(fam, []).append(PANEL_CAPTION.get(key, label))
    parts = []
    for caps in groups.values():
        if len(caps) > 1:
            parts.append('The {} panels share one y-axis scale.'.format(
                _common_tail_words(caps) or 'grouped'))
            break
    sib = _sibling(task, tasks)
    if sib is not None:
        parts.append('Axis scales are set from each figure\'s own data and are not '
                     'comparable with the {} figure.'.format(sib['label'].lower()))
    return ' '.join(parts)


def _config_sentence(task, tasks):
    """Channel set and integration window: invisible in the panels, and the two tasks are
    matched on neither, so each caption states its own and the supplementary one adds the
    confound in a single sentence — that is the caption a reader has open when comparing the
    two figures. Derived from each run's own meta.json, so if the arms are ever matched
    again the warning disappears rather than going quietly stale."""
    own = 'Decoding used {}.'.format(_config_phrase(task))
    sib = _sibling(task, tasks)
    if sib is None or _matched(task, sib):
        return own
    if task['role'] == 'main':
        return own + (' The {} figure used a different channel set and a longer window, so '
                      'the two figures are not a controlled contrast.'.format(
                          sib['label'].lower()))
    return own + (' The {} figure used {}; each configuration was selected on decoding '
                  'performance rather than matched across tasks, so a difference between the '
                  'two figures confounds task with channel set and integration window, and '
                  'neither selection is supported by a corrected significance test.'.format(
                      sib['label'].lower(), _config_phrase(sib)))


def _cohort_sentence(task):
    """The auditory cohort is not homogeneous and a reader cannot see that from the panels:
    two participants ran an earlier stimulus set with longer spoken prompts and a different
    category inventory, so chance differs between participants. Display IDs only — initials
    must never reach a caption."""
    older = [display_id(p) for p in ('CP', 'RB') if p in task['patients']]
    if task['key'] != 'auditory' or not older:
        return ''
    return ('The auditory cohort spans two stimulus sets — {} heard an earlier set with '
            'longer spoken prompts and a different category inventory — so the number of '
            'semantic categories differs between participants and the chance line is the '
            'mean of the per-participant shuffled nulls rather than a single '
            '1/n_categories.'.format(' and '.join(older)))


def _cue_sentence(task):
    """Cue markers. The "identical across participants" clause is emitted only when this task
    actually has a zero-spread cue (a group-warped stimulus interval); in a task where every
    cue varies it would describe nothing that is drawn."""
    txt = ('Dotted vertical line at 0 s: alignment cue ({}). Vertical coloured lines and '
           'shaded areas: mean cue time across participants ± 1 s.d.'.format(task['align_name']))
    if any(not (s > 0) for _m, s in task['cue_agg'].values()):
        txt += (' Cues identical across participants (a group-warped stimulus interval) are '
                'drawn as a single line without a shaded area.')
    return txt


def _notes_section(task, tasks, embedding, pctile, grid_stem=None):
    """Everything that is provenance rather than caption: which files this figure is, where
    the sibling captions live, and the methodological detail a Nature legend keeps out. Kept
    below the caption under a heading, never behind a `---` rule — `_write_caption` treats
    the first `\\n---\\n` as the start of a hand-written tail to preserve, so a generated rule
    would make this section duplicate on every render."""
    sib = _sibling(task, tasks)
    lines = [
        '## Notes — not part of the caption',
        '',
        '- Figure: `{}.{{png,pdf}}`; plotted values: `source_data/source_data.csv` '
        '(rows with `task = {}`).'.format(task['fig_stem'] or '(not rendered)', task['key']),
        '- Run: `{}`; {}; {} embeddings.'.format(
            os.path.basename(os.path.normpath(task['run_dir'])),
            _config_phrase(task, with_bins=True), embedding),
        '- Significance: a bin is significant when the observed mean exceeds the {}th '
        'percentile of that bin\'s shuffled null (`utils.config.ALPHA` = {:g}); bins before '
        'the alignment cue are masked out of both the raster and the source data.'.format(
            pctile, 1 - pctile / 100),
    ]
    if sib is not None:
        lines.append('- {} is a separate {} figure: `{}`, captioned in `{}`.'.format(
            sib['label'], sib['role'], sib['fig_stem'] or '(not rendered)', sib['caption']))
    lines.append('- The within-category-null supplementary figure has its own caption, '
                 '`S5_within_category_null_caption.md`, beside this file.')
    if grid_stem:
        lines.append('- `{}` is an internal cross-task view — not a paper figure, and '
                     'deliberately uncaptioned. It is the only figure here whose y-axis '
                     'scale is shared across tasks.'.format(grid_stem))
    lines.append('- Generated by `semantic_regression_panels.py`. Edit the generator, not '
                 'this file; a re-render overwrites everything above a `---` rule.')
    return '\n'.join(lines)


def _caption_text(task, tasks, embedding, pctile, grid_stem=None):
    """One figure's caption, in Nature legend style (figures_for_paper/README.md §4): bold
    title sentence, bold panel letters with no period after the letter, N in the opening
    sentence, every visual element defined, the test and its threshold named, and no result
    or trend anywhere. One paragraph, so it can be pasted into the manuscript as is."""
    sentences = [
        '**{} | {}.**'.format(task['fig_label'], task['fig_title']),
        'Held-out decoding accuracy as a function of time for {} (N = {}).'.format(
            task['label'].lower(), len(task['patients'])),
        'High-gamma activity was mapped onto {} word embeddings by kernel partial-least-'
        'squares regression and scored by nearest-neighbour retrieval.'.format(embedding),
        _metric_panels_text(),
        'Each participant is shown in a distinct colour, the same in every panel, and is '
        'identified by display ID.',
        'Horizontal coloured bars below the chance line are a per-participant significance '
        'raster, rows ordered by peak accuracy.',
        'Significance is calculated for each time bin separately (one-sided permutation test '
        "against that bin's shuffled null, p < {:g}; pre-onset bins are not tested).".format(
            1 - pctile / 100),
        'Dashed horizontal line: mean shuffled chance across participants.',
        _cue_sentence(task),
        _scale_sentence(task, tasks),
        _config_sentence(task, tasks),
        _cohort_sentence(task),
    ]
    caption = ' '.join(s for s in sentences if s)
    return """# {} caption — semantic-decoding time courses, {}

The paragraph below is the caption as it should appear in the manuscript — copy it whole.
Everything under "Notes" is provenance for this repository and is not part of the caption.

{}

{}
""".format(task['fig_label'], task['label'].lower(), caption,
           _notes_section(task, tasks, embedding, pctile, grid_stem))


def _write_captions(tasks, embedding, pctile, grid_stem=None):
    """One caption file per task — the two figures are captioned separately (2026-08-11)."""
    for task in tasks.values():
        path = os.path.join(FIG_DIR, task['caption'])
        _write_caption(path, _caption_text(task, tasks, embedding, pctile, grid_stem))
        print(f"  [panels] caption: wrote {task['caption']} — {task['role']} figure "
              f"({task['fig_stem']})", flush=True)


def _write_caption(path, txt):
    # Preserve anything appended below a horizontal rule. This file is regenerated on every
    # render, and on 2026-08-11 that silently deleted the supplementary S5 caption a
    # different pipeline (within_category_null_panels.py) had appended here -- 30 lines of
    # hand-written text, gone with nothing in the render output saying so. Only the
    # generated head is rewritten; everything from the first `\n---\n` onward is carried
    # through untouched.
    tail = ''
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            existing = f.read()
        marker = existing.find('\n---\n')
        if marker != -1:
            tail = existing[marker:]
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(txt.rstrip('\n') + '\n' + tail if tail else txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--picture-run-dir', default=TASKS['picture']['run_dir'])
    ap.add_argument('--auditory-run-dir', default=TASKS['auditory']['run_dir'])
    ap.add_argument('--embedding', default=EMBEDDING)
    ap.add_argument('--pctile', type=float, default=PCTILE,
                    help=f'null percentile threshold for significance '
                         f'(default {PCTILE:g} = 100*(1-ALPHA) ⇒ p<{ALPHA:g})')
    ap.add_argument('--rebuild-cache', nargs='?', const='both',
                    choices=list(TASKS) + ['both'], default=None,
                    help="re-extract the per-epoch arrays from the result PKLs for this "
                         "task (or 'both'); omit to use the cached npz. All tasks are "
                         "rendered either way.")
    args = ap.parse_args()
    rebuild = (list(TASKS) if args.rebuild_cache == 'both'
               else [args.rebuild_cache] if args.rebuild_cache else [])

    def _resolve_run_dir(rd):
        # Accept a full path, a bare run name, or a relative path — every run lives
        # directly under RESULTS_DIR, so fall back to matching by basename there.
        if os.path.isdir(rd):
            return rd
        for cand in (os.path.join(RESULTS_DIR, rd),
                     os.path.join(RESULTS_DIR, os.path.basename(os.path.normpath(rd)))):
            if os.path.isdir(cand):
                return cand
        return rd

    run_dirs = {'picture': _resolve_run_dir(args.picture_run_dir),
                'auditory': _resolve_run_dir(args.auditory_run_dir)}
    generate_panels(rebuild_cache=rebuild, run_dirs=run_dirs,
                    embedding=args.embedding, pctile=args.pctile)


if __name__ == '__main__':
    main()
