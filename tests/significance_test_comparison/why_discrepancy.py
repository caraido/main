# -*- coding: utf-8 -*-
"""Why do the two test families disagree? Measure it, don't assert it.

Family A (shipped rule, and rules (1)-(2)): compares ONE number -- the epoch-mean of the
observed score -- against the DISTRIBUTION of individual shuffled draws.
Family B (rules (3)-(5)): compares the two distributions' MEANS, with a standard error that
shrinks like 1/sqrt(n_epochs).

Four diagnostics, all computed from the cached obs/null arrays of the pinned runs:

  D1  baseline offset      mean(obs) - mean(null) on PRE-onset bins, where there is nothing
                           to decode. Anything non-zero here is a property of how the null
                           is built, not of the signal.
  D2  yardstick ratio      SD of individual null draws  /  SE of the mean difference.
                           This is the entire size of the disagreement, and it is ~sqrt(E).
  D3  epoch correlation    r(obs, null) across epochs within a bin -- how much of the
                           epoch-to-epoch wobble is shared (i.e. what the pairing removes).
  D4  n_epochs dependence  re-run both families on subsets of the 100 epochs. A calibrated
                           test's false-positive rate must NOT depend on how many resamples
                           were drawn. This is the decisive one.

Run from main/:   python -m tests.significance_test_comparison.why_discrepancy
Outputs:          results/significance_test_comparison/why_discrepancy.html
                  results/significance_test_comparison/source_data/why_discrepancy_*.csv
"""

import os
import sys
import base64
import warnings
from io import BytesIO

import numpy as np
import pandas as pd
from scipy import stats

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (MAIN_DIR, os.path.join(MAIN_DIR, 'figures_for_paper')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.config import ALPHA, PCTILE                              # noqa: E402
from utils.paths import results_dir                                 # noqa: E402
from tests.significance_test_comparison import perbin_test_comparison as C  # noqa: E402

ANALYSIS = 'significance_test_comparison'
OUT_DIR = str(results_dir(ANALYSIS))
SRC_DIR = str(results_dir(ANALYSIS, 'source_data'))

import matplotlib                                                    # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                      # noqa: E402

ALPHA_SAMPLE = 0.005          # the cutoff Alec settled on for rules (3)-(5)
EPOCH_GRID = [10, 20, 30, 50, 75, 100]
N_REPEATS = 5                 # random epoch subsets per grid point
SEED = 0

SURFACE, INK, INK_2, INK_MUTED = C.SURFACE, C.INK, C.INK_2, C.INK_MUTED
SERIES = C.SERIES


def _finite_stats(obs, null):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return (np.nanmean(obs, axis=0), np.nanmean(null, axis=0),
                np.nanstd(null, axis=0, ddof=1), np.nanstd(obs - null, axis=0, ddof=1))


# ── D1-D3 ─────────────────────────────────────────────────────────────────────

def descriptives(data):
    """Per task x metric x patient x bin: offset, null spread, paired SE, correlation."""
    rows = []
    for task, td in data.items():
        nbh = td['meta']['n_bins_history']
        bsz = td['meta']['bin_size_ms']
        for key, _lab in C.METRICS:
            for p in td['patients']:
                obs, null = td['arrays'][(p, key)]
                E = obs.shape[0]
                ts = C.time_axis(obs.shape[1], nbh, bsz)
                om, nm, sd_null, sd_diff = _finite_stats(obs, null)
                se_diff = sd_diff / np.sqrt(E)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    r = np.array([
                        (np.corrcoef(obs[:, b], null[:, b])[0, 1]
                         if np.all(np.isfinite(obs[:, b])) and np.all(np.isfinite(null[:, b]))
                         and np.std(obs[:, b]) > 0 and np.std(null[:, b]) > 0 else np.nan)
                        for b in range(obs.shape[1])])
                for b in range(obs.shape[1]):
                    rows.append(dict(
                        task=task, metric=key, patient=p, bin_index=b,
                        time_s=round(float(ts[b]), 4), window='pre' if ts[b] < 0 else 'post',
                        obs_mean=om[b], null_mean=nm[b], offset=om[b] - nm[b],
                        sd_null_draws=sd_null[b], se_mean_diff=se_diff[b],
                        yardstick_ratio=(sd_null[b] / se_diff[b]
                                         if np.isfinite(se_diff[b]) and se_diff[b] > 0 else np.nan),
                        cohens_d=((om[b] - nm[b]) / sd_null[b]
                                  if np.isfinite(sd_null[b]) and sd_null[b] > 0 else np.nan),
                        r_obs_null=r[b]))
    return pd.DataFrame(rows)


# ── D4: does the answer depend on how many epochs were drawn? ─────────────────

def epoch_sweep(data, rng):
    """% of bins called significant vs number of epochs used, for one rule from each family.

    Family A = the shipped percentile rule (no p floor, so it is comparable across E).
    Family B = the paired t-test, Bonferroni within participant, at ALPHA_SAMPLE.
    """
    rows = []
    for task, td in data.items():
        nbh, bsz = td['meta']['n_bins_history'], td['meta']['bin_size_ms']
        for key, _lab in C.METRICS:
            for E in EPOCH_GRID:
                acc = {(f, w): [0, 0] for f in ('A_pctile', 'B_ttest_rel')
                       for w in ('pre', 'post')}
                for _rep in range(N_REPEATS):
                    for p in td['patients']:
                        obs, null = td['arrays'][(p, key)]
                        idx = rng.choice(obs.shape[0], size=E, replace=False)
                        o, n = obs[idx], null[idx]
                        ts = C.time_axis(obs.shape[1], nbh, bsz)
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            om = np.nanmean(o, axis=0)
                            thr = np.nanpercentile(n, PCTILE, axis=0)
                            sigA = np.isfinite(om) & np.isfinite(thr) & (om > thr)
                            d = o - n
                            res = stats.ttest_1samp(d, 0.0, axis=0, alternative='greater',
                                                    nan_policy='omit')
                            praw = np.asarray(res.pvalue, dtype=float)
                        m = int(np.isfinite(praw).sum())
                        padj = np.where(np.isfinite(praw),
                                        np.minimum(1.0, praw * max(m, 1)), np.nan)
                        sigB = np.isfinite(padj) & (padj < ALPHA_SAMPLE)
                        for w, mask in (('pre', ts < 0), ('post', ts >= 0)):
                            acc[('A_pctile', w)][0] += int(np.sum(sigA & mask))
                            acc[('A_pctile', w)][1] += int(np.sum(mask))
                            acc[('B_ttest_rel', w)][0] += int(np.sum(sigB & mask))
                            acc[('B_ttest_rel', w)][1] += int(np.sum(mask))
                for (fam, w), (ns, nt) in acc.items():
                    rows.append(dict(task=task, metric=key, n_epochs=E, family=fam,
                                     window=w, n_sig=ns, n_total=nt,
                                     pct_sig=100.0 * ns / nt if nt else np.nan))
    return pd.DataFrame(rows)


# ── Figure ────────────────────────────────────────────────────────────────────

def sweep_figure(sw):
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), facecolor=SURFACE)
    fam_style = {'A_pctile': ('Family A — shipped percentile rule', SERIES[0], 'o-'),
                 'B_ttest_rel': ('Family B — paired t · Bonferroni, α=0.005', SERIES[4], 's-')}
    for r, task in enumerate(['picture', 'auditory']):
        for c, win in enumerate(['post', 'pre']):
            ax = axes[r][c]
            C._style_axis(ax)
            for fam, (lab, col, mk) in fam_style.items():
                sub = (sw[(sw.task == task) & (sw.window == win) & (sw.family == fam)]
                       .groupby('n_epochs', as_index=False)['pct_sig'].mean())
                ax.plot(sub['n_epochs'].to_numpy(), sub['pct_sig'].to_numpy(), mk,
                        color=col, lw=1.8, ms=5,
                        label=lab if (r == 0 and c == 0) else None)
            ax.set_xlabel('epochs used (of 100)', fontsize=8.5, color=INK_2)
            ax.set_ylabel('% of bins called significant', fontsize=8.5, color=INK_2)
            ax.set_title(f'{task} naming — {"POST" if win == "post" else "PRE"}-onset'
                         f'{"  (false-positive control)" if win == "pre" else ""}',
                         fontsize=10, color=INK, pad=6)
            ax.set_ylim(0, max(2, sw[(sw.task == task) & (sw.window == win)]['pct_sig'].max() * 1.15))
            ax.grid(color='#e6e5e0', linewidth=0.7)
            ax.set_axisbelow(True)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.998), ncol=2, frameon=False,
               fontsize=9.5, labelcolor=INK_2)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.075,
                        wspace=0.20, hspace=0.42)
    return C._png(fig)


def _tbl(df, cols, fmt):
    head = ''.join(f'<th>{c}</th>' for c in cols)
    body = ''.join('<tr>' + ''.join(
        f'<td>{fmt(c, r[c])}</td>' for c in cols) + '</tr>' for _i, r in df.iterrows())
    return f'<table><tr>{head}</tr>{body}</table>'


def main():
    rng = np.random.default_rng(SEED)
    data = {t: C.load_task(t) for t in C.TASKS}
    print(f"patients: { {t: len(d['patients']) for t, d in data.items()} }", flush=True)

    print("[D1-D3] descriptives ...", flush=True)
    d = descriptives(data)
    d.to_csv(os.path.join(SRC_DIR, 'why_discrepancy_perbin.csv'), index=False)

    pre = d[d.window == 'pre']
    s1 = (pre.groupby(['task', 'metric'], as_index=False)
             .agg(offset=('offset', 'median'), sd_null=('sd_null_draws', 'median'),
                  se_diff=('se_mean_diff', 'median'), d=('cohens_d', 'median'),
                  ratio=('yardstick_ratio', 'median'), r=('r_obs_null', 'median')))
    s1['t_equiv'] = s1['offset'] / s1['se_diff']
    s1.to_csv(os.path.join(SRC_DIR, 'why_discrepancy_summary.csv'), index=False)
    print("\nPRE-onset medians (nothing to decode here):\n")
    print(s1.round(4).to_string(index=False))

    print(f"\n[D4] epoch sweep ({N_REPEATS} random subsets per point) ...", flush=True)
    sw = epoch_sweep(data, rng)
    sw.to_csv(os.path.join(SRC_DIR, 'why_discrepancy_epoch_sweep.csv'), index=False)
    piv = (sw[sw.window == 'pre'].groupby(['task', 'family', 'n_epochs'])['pct_sig']
             .mean().unstack('n_epochs'))
    print("\nPRE-onset % significant vs epochs used:\n")
    print(piv.round(1).to_string())

    png = sweep_figure(sw)

    def f1(c, v):
        return f'{v:.4f}' if isinstance(v, float) and c not in ('ratio', 't_equiv') else (
            f'{v:.1f}' if isinstance(v, float) else str(v))

    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>Why the two test families disagree</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ background:{SURFACE}; color:{INK}; margin:0; padding:32px 40px 72px;
         font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif; max-width:1400px; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  h2 {{ font-size:15px; margin:34px 0 8px; }}
  p, li {{ color:{INK_2}; max-width:95ch; }}
  code {{ background:#f0efea; padding:1px 4px; border-radius:3px; font-size:12.5px; }}
  .warn {{ border-left:3px solid #eb6834; background:#fdf3ee; padding:12px 16px; margin:16px 0; }}
  .ok {{ border-left:3px solid #1baf7a; background:#eefaf5; padding:12px 16px; margin:16px 0; }}
  img {{ width:100%; height:auto; margin:6px 0; }}
  table {{ border-collapse:collapse; font-size:12px; margin:8px 0; }}
  th, td {{ border-bottom:1px solid #e6e5e0; padding:4px 10px; text-align:right; }}
  th:nth-child(-n+2), td:nth-child(-n+2) {{ text-align:left; }}
  th {{ color:{INK_2}; font-weight:600; border-bottom:1px solid #cfcec9; }}
</style>
<h1>Why the two test families disagree</h1>
<p>Same cached obs/null arrays as <code>perbin_test_comparison.html</code> beside this file;
pinned picture (N=15) and auditory (N=10) runs, 100 epochs, GloVe.</p>

<h2>D1–D3 — pre-onset medians, where there is nothing to decode</h2>
{_tbl(s1, ['task', 'metric', 'offset', 'sd_null', 'se_diff', 'd', 't_equiv', 'ratio', 'r'], f1)}
<p><code>offset</code> = median(mean obs − mean null) over pre-onset bins.
<code>sd_null</code> = SD of individual shuffled draws (family A's yardstick).
<code>se_diff</code> = SD(obs−null)/√100 (family B's yardstick).
<code>d</code> = offset / sd_null. <code>t_equiv</code> = offset / se_diff.
<code>ratio</code> = sd_null / se_diff. <code>r</code> = median correlation between the
observed and shuffled score across epochs within a bin.</p>

<h2>D4 — does the answer depend on how many epochs were drawn?</h2>
<img src="data:image/png;base64,{png}" alt="significant-bin rate vs number of epochs">
<p>A calibrated test's false-positive rate must be flat in the number of resamples. The
right-hand panels are the pre-onset baseline: family A is flat, family B climbs.</p>
<p style="color:{INK_MUTED}">Numbers, under <code>source_data/</code> beside this file:
<code>why_discrepancy_summary.csv</code>,
<code>why_discrepancy_epoch_sweep.csv</code>,
<code>why_discrepancy_perbin.csv</code>.</p>
"""
    out = os.path.join(OUT_DIR, 'why_discrepancy.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(doc)
    print(f"\nwrote {out}", flush=True)


if __name__ == '__main__':
    main()
