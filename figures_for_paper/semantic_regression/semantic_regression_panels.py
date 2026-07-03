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
  source_data/source_data.csv   — per patient × bin: obs, chance, p_raw, q_bh, sig
  source_data/cue_timing.csv    — aggregated cue mean ± s.d.

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
MAIN_DIR = os.path.dirname(os.path.dirname(HERE))          # …/main
RUN_DIR = os.path.join(
    MAIN_DIR, 'results', 'semantic_regression',
    '2026-06-02_17-25-11_picture_naming_kernel_pls_cosine_100ep')
FIG_DIR = HERE
SRC_DIR = os.path.join(HERE, 'source_data')

EMBEDDING = 'GloVe'
PCTILE = 99          # a bin is significant iff obs mean > this percentile of the null (~p<0.01)

# ── Metric definitions ────────────────────────────────────────────────────────
# key → (pretty label, obs attr, null attr, family)  — panels in a family share y.
METRICS = [
    ('category_indep', 'Independent category accuracy',
     'all_retrieval_category_indep_balanced_acc',
     'all_retrieval_category_indep_chance_balanced_acc', 'category'),
    ('word_top1', 'Word top-1 accuracy',
     'all_retrieval_top1', 'all_retrieval_chance_top1', 'word'),
    ('word_top3', 'Word top-3 accuracy',
     'all_retrieval_top3', 'all_retrieval_chance_top3', 'word'),
    ('word_top5', 'Word top-5 accuracy',
     'all_retrieval_top5', 'all_retrieval_chance_top5', 'word'),
]

CUE_STYLE = {
    'go_cue':       dict(color='#003388', label='Go cue'),
    'voice_onset':  dict(color='#006600', label='Voice onset'),
    'voice_offset': dict(color='#8a5a00', label='Voice offset'),
}

_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#393b79', '#e7298a']


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


def plot_panel(label, per_patient, patients, colors, cue_agg, bin_size_s,
               y_top, align_label, chance_t, chance_mean):
    """Render one metric panel. per_patient[p] = dict(obs_mean, null_mean, sig,
    time_s). Patients may have different bin counts → each uses its own axis."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))

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

    for i, p in enumerate(patients):
        c = colors[i]
        t = per_patient[p]['time_s']
        ax.plot(t, per_patient[p]['obs_mean'], color=c, lw=1.2, alpha=0.9, zorder=3)
        sig = per_patient[p]['sig']
        y0 = raster_top - (i + 1) * row_h
        segs = [(t[b] - bin_size_s / 2, bin_size_s) for b in range(len(sig)) if sig[b]]
        if segs:
            ax.broken_barh(segs, (y0, row_h * 0.9), facecolors=c, edgecolors='none', zorder=2)

    ax.plot(chance_t, chance_mean, color='#444444', lw=1.1, ls='--', alpha=0.8,
            zorder=4, label='mean shuffled chance')

    ax.set_xlabel(align_label)
    ax.set_ylabel(label)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(raster_bottom - 0.02 * y_top, y_top)
    yt = [t for t in ax.get_yticks() if t >= -1e-9]
    ax.set_yticks(yt)
    ax.set_yticklabels([f'{t:.2f}' for t in yt])
    ax.text(xmin, (raster_top + raster_bottom) / 2, 'sig.\n(p<.01)',
            fontsize=6.5, color='#555555', ha='right', va='center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


def legend_figure(patients, colors, cue_agg):
    fig, ax = plt.subplots(figsize=(5.2, 0.9))
    ax.axis('off')
    handles = [mlines.Line2D([], [], color=colors[i], lw=2, label=p)
               for i, p in enumerate(patients)]
    handles.append(mlines.Line2D([], [], color='#444444', lw=1.5, ls='--', label='mean chance'))
    for cue in cue_agg:
        handles.append(Patch(facecolor=CUE_STYLE[cue]['color'], alpha=0.3,
                             label=f"{CUE_STYLE[cue]['label']} (±1 s.d.)"))
    ax.legend(handles=handles, ncol=6, loc='center', fontsize=7.5, frameon=False)
    fig.tight_layout()
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
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(patients))]
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
            per_patient[p] = dict(obs_mean=obs_m, null_mean=null_m, sig=sig, time_s=t)
            for b in range(obs.shape[1]):
                src_rows.append(dict(metric=key, patient=p, bin_index=b, time_s=t[b],
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
    for key, label, _oa, _na, fam in METRICS:
        d = results[key]
        ct, cm = _chance_curve(d['per_patient'])
        fig = plot_panel(label, d['per_patient'], patients, colors, cue_agg,
                         bin_size_s, fam_top[fam], align_label, ct, cm)
        stem = os.path.join(FIG_DIR, f"{order[key]}_{key}")
        fig.savefig(stem + '.pdf', bbox_inches='tight')
        fig.savefig(stem + '.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  [panels] {key}: saved {order[key]}_{key}.pdf/.png", flush=True)

    leg = legend_figure(patients, colors, cue_agg)
    leg.savefig(os.path.join(FIG_DIR, '00_legend.pdf'), bbox_inches='tight')
    leg.savefig(os.path.join(FIG_DIR, '00_legend.png'), dpi=200, bbox_inches='tight')
    plt.close(leg)

    pd.DataFrame(src_rows).to_csv(os.path.join(SRC_DIR, 'source_data.csv'), index=False)
    pd.DataFrame([dict(cue=c, mean_s=m, std_s=s) for c, (m, s) in cue_agg.items()]
                 ).to_csv(os.path.join(SRC_DIR, 'cue_timing.csv'), index=False)
    _write_caption(os.path.join(FIG_DIR, 'caption.md'), patients, embedding, pctile, align_cue)
    print(f"[panels] figures + caption in: {FIG_DIR}")
    print(f"[panels] source data in:       {SRC_DIR}")


def _write_caption(path, patients, embedding, pctile, align_cue):
    txt = f"""# Figure caption — Cross-patient semantic-decoding time courses

Cross-patient semantic-decoding time courses ({embedding}). Held-out decoding accuracy as a
function of time for picture naming ({len(patients)} participants; kernel-PLS: Nystroem RBF kernel
followed by PLS regression onto {embedding} word-embedding targets), each participant in a distinct
colour. **a** Independent-centroid balanced category accuracy. **b**, **c**, **d** Raw word-retrieval
top-1, top-3 and top-5 accuracy; **b**, **c**, **d** share one y-scale. Coloured bars below the chance
line are a per-participant significance raster: time bins where the observed mean accuracy exceeds the
{pctile}th percentile of the shuffled-null distribution at that bin (per-bin one-sided permutation
test, p < {1 - pctile/100:.2g}). Dashed line: mean shuffled chance across participants.
Dotted vertical line at 0 s: alignment cue ({align_cue.replace('_', ' ')}). Shaded vertical bands: mean
cue time across participants ± 1 s.d. (cues with zero spread excluded). x-axis in seconds. **a, b, c, d**
N={len(patients)}.
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
