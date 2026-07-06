# -*- coding: utf-8 -*-
"""
figures_for_paper/semantic_regression — Cross-patient decoding time-course panels.

Paper-figure generator. Produces one panel per decoding metric (GloVe only),
each overlaying every participant in a distinct colour, with:

  * a per-participant significance raster *below the chance line* — bins where the
    observed mean accuracy exceeds the 99th percentile of the shuffled-null
    distribution at that bin (per-bin one-sided permutation test, ≈ p<0.01);
  * cue markers (go cue, voice onset, voice offset) as a vertical line at the
    across-participant mean time with a shaded band = ± 1 s.d. across participants
    (cues with zero spread, e.g. the alignment cue, are skipped);
  * an x-axis in seconds with 0 at the alignment cue (trial onset here);
  * a shared y-axis scale within a metric family (the three word top-k panels
    share one scale; the category panel has its own).

Metrics (from the per-epoch PKL arrays, cached to panels_cache_{emb}.npz):
  1. category_indep — independent-centroid balanced category accuracy
  2. word_top1 / word_top3 / word_top5 — raw top-k word-retrieval accuracy

Outputs (this folder):
  00_legend.pdf/.png, 01_category_indep.pdf/.png, 02_word_top1.pdf/.png,
  03_word_top3.pdf/.png, 04_word_top5.pdf/.png       (PDFs: pdf.fonttype 42)
  caption.md
  source_data/source_data.csv     — per patient × bin: obs, chance, p_raw, q_bh, sig
  source_data/cue_timing.csv      — aggregated cue mean ± s.d.
  source_data/peak_rise_stats.csv — per metric: peak acc ± s.e.m., empirical chance,
                                    rise/peak latency mean ± s.d. (Results-text numbers)

Reproduce:
  # fast path — uses the cached arrays in this folder:
  python figures_for_paper/semantic_regression/semantic_regression_panels.py
  # rebuild the cache from the (large) result PKLs:
  python figures_for_paper/semantic_regression/semantic_regression_panels.py --rebuild-cache
(run with cwd = main/)
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch
from scipy import stats as _stats

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
RUN_DIR = os.path.join(
    MAIN_DIR, 'results', 'semantic_regression',
    '2026-06-02_17-25-11_picture_naming_kernel_pls_cosine_100ep')
FIG_DIR = HERE
SRC_DIR = os.path.join(HERE, 'source_data')

EMBEDDING = 'GloVe'
PCTILE = 99          # a bin is significant iff obs mean > this percentile of the null (~p<0.01)

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
    'category_indep': 'Category accuracy',
    'word_top1': 'Top-1 word-retrieval accuracy',
    'word_top3': 'Top-3 word-retrieval accuracy',
    'word_top5': 'Top-5 word-retrieval accuracy',
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
            continue
        br = data['regressors'][embedding]
        ok = True
        for key, _l, obs_attr, null_attr, _f in METRICS:
            if not (hasattr(br, obs_attr) and hasattr(br, null_attr)):
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

    np.savez_compressed(cache_path, **arrays)
    with open(cache_path + '.json', 'w') as f:
        json.dump({'patients': kept, 'cues': cues, 'embedding': embedding}, f)
    print(f"  [cache] saved {len(kept)} patients → {cache_path}", flush=True)
    return {'arrays': dict(np.load(cache_path)),
            'side': {'patients': kept, 'cues': cues, 'embedding': embedding}}


def load_cache(cache_path):
    if not (os.path.exists(cache_path) and os.path.exists(cache_path + '.json')):
        return None
    return {'arrays': dict(np.load(cache_path)),
            'side': json.load(open(cache_path + '.json'))}


# ── Statistics ────────────────────────────────────────────────────────────────

def perbin_significance(obs, null, pctile=PCTILE):
    """Per-bin permutation test: a bin is significant iff the observed mean
    accuracy exceeds the `pctile`-th percentile of the shuffled-null distribution
    at that bin (one-sided; pctile=99 ≈ p<0.01). This compares the data directly
    against the full shuffled distribution and is naturally strict — it does not
    inflate with epoch count the way a t-test on a tiny reliable offset does.
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
    have different bin counts). Returns (times, mean_curve) sorted by time."""
    from collections import defaultdict
    acc = defaultdict(list)
    for p in patients:
        for tv, yv in zip(per_patient[p]['time_s'], per_patient[p][attr]):
            acc[round(float(tv), 6)].append(float(yv))
    times = np.array(sorted(acc))
    mean = np.array([float(np.mean(acc[t])) for t in times])
    return times, mean


def compute_peak_rise_stats(results, patients):
    """Per-metric summary numbers used in the Results text — recomputed from
    whatever participants are present, so they stay correct as the cohort grows.

    Per metric:
      * peak accuracy = across-participant mean at the group peak bin (t* = argmax of
        the across-participant mean curve over t>=0), with s.e.m.; `emp_chance` is the
        mean permuted-null at t*.
      * peak/rise latencies are per-participant, averaged over participants that show
        ANY significant bin (rise is only defined there); reported as mean ± s.d.
        Rise = onset of the first significant bin (t>=0); peak = argmax over t>=0.
    Returns a tidy DataFrame (one row per metric, in METRICS order)."""
    rows = []
    for key, label, *_rest in METRICS:
        pp = results[key]['per_patient']
        # group peak bin (t>=0) on the across-participant mean curve
        gt, gm = _group_mean_curve(pp, patients, 'obs_mean')
        pos = gt >= 0
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

def _aggregate_cues(cues, patients):
    """cue → (mean_time, std_time) across patients (skip zero-spread cues)."""
    out = {}
    for cue in CUE_STYLE:
        vals = [cues[p][cue] for p in patients if cue in cues.get(p, {})]
        vals = [v for v in vals if np.isfinite(v)]
        if len(vals) >= 2:
            s = float(np.std(vals))
            if s > 0:
                out[cue] = (float(np.mean(vals)), s)
    return out


def _time_axis(n_bins, n_bins_history, bin_size_ms):
    return np.array([(b - n_bins_history) * bin_size_ms / 1000.0 for b in range(n_bins)])


def _draw_panel(ax, label, per_patient, patients, colors, cue_agg, bin_size_s,
                y_top, align_label, chance_t, chance_mean, pctile=PCTILE,
                panel_letter=None):
    """Draw one metric panel onto `ax`. per_patient[p] = dict(obs_mean, null_mean,
    sig, time_s). Patients may have different bin counts → each uses its own axis.
    Traces keep each participant's fixed colour; the significance raster rows are
    ordered by peak accuracy (highest at the top, lowest at the bottom)."""
    color_of = {p: colors[i] for i, p in enumerate(patients)}

    raster_top = -0.03 * y_top
    raster_bottom = -0.34 * y_top
    row_h = (raster_top - raster_bottom) / max(len(patients), 1)
    xmin = min(per_patient[p]['time_s'][0] for p in patients)
    xmax = max(per_patient[p]['time_s'][-1] for p in patients)

    for cue, (mu, sd) in cue_agg.items():
        st = CUE_STYLE[cue]
        ax.axvspan(mu - sd, mu + sd, color=st['color'], alpha=0.08, lw=0, zorder=0)
        ax.axvline(mu, color=st['color'], lw=1.0, ls='-', alpha=0.55, zorder=1)

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


def plot_panel(label, per_patient, patients, colors, cue_agg, bin_size_s,
               y_top, align_label, chance_t, chance_mean, pctile=PCTILE):
    """Render one metric panel as a standalone figure."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    _draw_panel(ax, label, per_patient, patients, colors, cue_agg, bin_size_s,
                y_top, align_label, chance_t, chance_mean, pctile=pctile)
    fig.tight_layout()
    return fig


def _legend_handles(patients, colors, cue_agg):
    handles = [mlines.Line2D([], [], color=colors[i], lw=2, label=display_id(p))
               for i, p in enumerate(patients)]
    handles.append(mlines.Line2D([], [], color='#444444', lw=1.5, ls='--', label='mean chance'))
    for cue in cue_agg:
        handles.append(Patch(facecolor=CUE_STYLE[cue]['color'], alpha=0.3,
                             label=f"{CUE_STYLE[cue]['label']} (±1 s.d.)"))
    return handles


def legend_figure(patients, colors, cue_agg):
    fig, ax = plt.subplots(figsize=(5.2, 0.9))
    ax.axis('off')
    ax.legend(handles=_legend_handles(patients, colors, cue_agg),
              ncol=6, loc='center', fontsize=7.5, frameon=False)
    fig.tight_layout()
    return fig


def plot_combined(metric_order, results, chance_curves, patients, colors, cue_agg,
                  bin_size_s, fam_top, align_label, pctile=PCTILE):
    """Nature-style 2×2 combined figure of the four metric panels (a–d),
    with a shared participant/cue legend below."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    letters = ['a', 'b', 'c', 'd']
    for ax, letter, key in zip(axes.ravel(), letters, metric_order):
        d = results[key]
        ct, cm = chance_curves[key]
        _draw_panel(ax, d['label'], d['per_patient'], patients, colors, cue_agg,
                    bin_size_s, fam_top[d['family']], align_label, ct, cm,
                    pctile=pctile, panel_letter=letter)
    fig.legend(handles=_legend_handles(patients, colors, cue_agg),
               ncol=6, loc='lower center', fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


# ── Orchestration ─────────────────────────────────────────────────────────────

def generate_panels(run_dir=RUN_DIR, rebuild_cache=False, embedding=EMBEDDING, pctile=PCTILE):
    os.makedirs(SRC_DIR, exist_ok=True)
    meta_path = os.path.join(run_dir, 'meta.json')
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    n_bins_history = meta.get('n_bins_history', 10)
    bin_size_ms = meta.get('bin_size_ms', 100)
    align_cue = meta.get('align_cue') or 'trial_onset'
    if align_cue in (None, 'none', ''):
        align_cue = 'trial_onset'
    align_label = f"Time from {align_cue.replace('_', ' ')} (s)"

    cache_path = os.path.join(FIG_DIR, f'panels_cache_{embedding}.npz')
    cache = None if rebuild_cache else load_cache(cache_path)
    if cache is None:
        print("[panels] building cache from PKLs (one-time, slow)...", flush=True)
        cache = build_cache(run_dir, cache_path, embedding=embedding)

    arrays, side = cache['arrays'], cache['side']
    patients = side['patients']
    colors = assign_colors(patients)                       # fixed per-participant colour (config)
    cue_agg = _aggregate_cues(side['cues'], patients)

    # per-patient stats for every metric
    results, src_rows = {}, []
    for key, label, _oa, _na, fam in METRICS:
        per_patient = {}
        for p in patients:
            obs = arrays[f'{p}__{key}__obs']
            null = arrays[f'{p}__{key}__null']
            sig, pv, thr, obs_m, null_m = perbin_significance(obs, null, pctile=pctile)
            t = _time_axis(obs.shape[1], n_bins_history, bin_size_ms)
            # We make no decoding claim before trial onset (t=0), so pre-onset bins
            # are never marked significant — in the raster or the source data.
            sig = sig & (t >= 0)
            per_patient[p] = dict(obs_mean=obs_m, null_mean=null_m, sig=sig, time_s=t)
            did = display_id(p)
            for b in range(obs.shape[1]):
                src_rows.append(dict(metric=key, display_id=did, patient=p,
                                     bin_index=b, time_s=t[b],
                                     obs_mean=obs_m[b], chance_mean=null_m[b],
                                     null_p=thr[b], p_perm=pv[b], significant=bool(sig[b])))
        results[key] = dict(per_patient=per_patient, family=fam, label=label)

    # shared y-scale within a metric family
    fam_top = {}
    for d in results.values():
        top = max(np.nanmax(d['per_patient'][p]['obs_mean']) for p in patients)
        fam_top[d['family']] = max(fam_top.get(d['family'], 0.0), top)
    for fam in fam_top:
        fam_top[fam] *= 1.10

    bin_size_s = bin_size_ms / 1000.0

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

    order = {'category_indep': '01', 'word_top1': '02', 'word_top3': '03', 'word_top5': '04'}
    chance_curves = {}
    for key, label, _oa, _na, fam in METRICS:
        d = results[key]
        ct, cm = _chance_curve(d['per_patient'])
        chance_curves[key] = (ct, cm)
        fig = plot_panel(label, d['per_patient'], patients, colors, cue_agg,
                         bin_size_s, fam_top[fam], align_label, ct, cm, pctile=pctile)
        stem = os.path.join(FIG_DIR, f"{order[key]}_{key}")
        fig.savefig(stem + '.pdf', bbox_inches='tight')
        fig.savefig(stem + '.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  [panels] {key}: saved {order[key]}_{key}.pdf/.png", flush=True)

    # Nature-style 2×2 combined figure (panels a–d)
    metric_order = [m[0] for m in METRICS]
    comb = plot_combined(metric_order, results, chance_curves, patients, colors,
                         cue_agg, bin_size_s, fam_top, align_label, pctile=pctile)
    comb_stem = os.path.join(FIG_DIR, '05_combined')
    comb.savefig(comb_stem + '.pdf', bbox_inches='tight')
    comb.savefig(comb_stem + '.png', dpi=300, bbox_inches='tight')
    plt.close(comb)
    print("  [panels] combined: saved 05_combined.pdf/.png", flush=True)

    leg = legend_figure(patients, colors, cue_agg)
    leg.savefig(os.path.join(FIG_DIR, '00_legend.pdf'), bbox_inches='tight')
    leg.savefig(os.path.join(FIG_DIR, '00_legend.png'), dpi=200, bbox_inches='tight')
    plt.close(leg)

    pd.DataFrame(src_rows).to_csv(os.path.join(SRC_DIR, 'source_data.csv'), index=False)
    pd.DataFrame([dict(cue=c, mean_s=m, std_s=s) for c, (m, s) in cue_agg.items()]
                 ).to_csv(os.path.join(SRC_DIR, 'cue_timing.csv'), index=False)

    # Results-text summary numbers (peak accuracy, empirical chance, rise/peak
    # latencies) — recomputed from the current cohort so the paragraph stays correct
    # as participants are added.
    stats = compute_peak_rise_stats(results, patients)
    stats.to_csv(os.path.join(SRC_DIR, 'peak_rise_stats.csv'), index=False)
    print("  [panels] peak/rise stats (Results text):")
    for r in stats.itertuples(index=False):
        print(f"    {r.metric:14s} peak {r.peak_acc_mean:.3f}±{r.peak_acc_sem:.3f} "
              f"vs chance {r.emp_chance:.3f} | rise {r.rise_time_mean_s:.2f}±{r.rise_time_sd_s:.2f}s "
              f"peak {r.peak_time_mean_s:.2f}±{r.peak_time_sd_s:.2f}s (n_sig={r.n_sig}/{r.n_total})",
              flush=True)

    _write_caption(os.path.join(FIG_DIR, 'caption.md'), patients, embedding, pctile, align_cue)
    print(f"[panels] figures + caption in: {FIG_DIR}")
    print(f"[panels] source data in:       {SRC_DIR}")


def _panel_letter(i):
    """0-based panel index → letter(s): a, b, …, z, aa, ab, …"""
    s = ''
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(97 + r) + s
    return s


def _caption_panels_text():
    """Per-panel descriptions + shared-y-scale note, generated from METRICS so the
    caption always covers every panel (in order) — panels a, b, c, …"""
    from collections import OrderedDict
    letters = [_panel_letter(i) for i in range(len(METRICS))]
    per_panel = ' '.join(
        f"**{letters[i]}** {PANEL_CAPTION.get(key, label)}."
        for i, (key, label, *_rest) in enumerate(METRICS)
    )
    # families with >1 panel share one y-scale
    fam = OrderedDict()
    for i, m in enumerate(METRICS):
        fam.setdefault(m[4], []).append(letters[i])
    shared = [', '.join(f'**{l}**' for l in ls) for ls in fam.values() if len(ls) > 1]
    share_note = (' ' + '; '.join(f"{s} share one y-scale" for s in shared) + '.') if shared else ''
    all_letters = ', '.join(f'**{l}**' for l in letters)
    return per_panel, share_note, all_letters


def _write_caption(path, patients, embedding, pctile, align_cue):
    per_panel, share_note, all_letters = _caption_panels_text()
    txt = f"""# Figure caption — Cross-patient semantic-decoding time courses

Cross-patient semantic-decoding time courses ({embedding}). Held-out decoding accuracy as a
function of time for picture naming ({len(patients)} participants; kernel-PLS: Nystroem RBF kernel
followed by PLS regression onto {embedding} word-embedding targets), each participant in a distinct
colour. {per_panel}{share_note} Coloured bars below the chance
line are a per-participant significance raster (rows ordered by peak accuracy, highest at top): time
bins after trial onset where the observed mean accuracy exceeds the {pctile}th percentile of the
shuffled-null distribution at that bin (per-bin one-sided permutation test, p < {1 - pctile/100:.2g};
pre-onset bins are not tested). Dashed line: mean shuffled chance across participants.
Dotted vertical line at 0 s: alignment cue ({align_cue.replace('_', ' ')}). Shaded vertical bands: mean
cue time across participants ± 1 s.d. (cues with zero spread excluded). x-axis in seconds. Participants
are identified by display ID (NUEx###). {all_letters} N={len(patients)}.
"""
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', default=RUN_DIR)
    ap.add_argument('--embedding', default=EMBEDDING)
    ap.add_argument('--pctile', type=float, default=PCTILE,
                    help='null percentile threshold for significance (default 99 ≈ p<0.01)')
    ap.add_argument('--rebuild-cache', action='store_true')
    args = ap.parse_args()
    generate_panels(run_dir=args.run_dir, rebuild_cache=args.rebuild_cache,
                    embedding=args.embedding, pctile=args.pctile)


if __name__ == '__main__':
    main()
