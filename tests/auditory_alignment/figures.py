# -*- coding: utf-8 -*-
"""Figures for the auditory_alignment report (figures_for_paper style).

Imports the paper's single-source-of-truth style (participant colours, display IDs,
cue colours, rcParams) from figures_for_paper/paper_common.py. The small box+points /
significance-bracket helpers are re-implemented here (adapted from
figures_for_paper/extendability/extendability_panels.py) so this pilot stays self-contained
and does not import that module's heavy top-level code.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# ── Path bootstrap + paper style ──────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAIN_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)
_FIGS_FOR_PAPER = os.path.join(_MAIN_DIR, "figures_for_paper")
if _FIGS_FOR_PAPER not in sys.path:
    sys.path.insert(0, _FIGS_FOR_PAPER)

from paper_common import (apply_paper_style, display_id, assign_colors,   # noqa: E402
                          load_cue_style)

from tests.auditory_alignment import config          # noqa: E402
from tests.auditory_alignment import aggregate as A   # noqa: E402
from tests.auditory_alignment import stats as S       # noqa: E402

apply_paper_style()

BOX_FACE = "#e8e8e8"
CUE_STYLE = load_cue_style()   # sr_cue_name -> {'color','label'}


# ── small reusable helpers (adapted from extendability_panels) ────────────────

def _rng(seed):
    return np.random.default_rng(seed)


def _cue_color(cue_key):
    return CUE_STYLE.get(config.CUES[cue_key], {}).get("color", "#555555")


def _cue_long_label(sr_name):
    return CUE_STYLE.get(sr_name, {}).get("label", sr_name)


def _box_points(ax, positions, data_by_pos, patients, colors, width=0.55, seed=0, ms=3.2):
    """Box (IQR+median) per position + jittered per-participant points (fixed colour) +
    a black across-participant mean line. data_by_pos[i] aligned to `patients`; NaNs
    (missing cells / few-trial patients) are dropped from the box and skipped as points."""
    color_of = {p: colors[i] for i, p in enumerate(patients)}
    box_data, box_pos = [], []
    for xi, arr in zip(positions, data_by_pos):
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            box_data.append(a)
            box_pos.append(xi)
    if box_data:
        ax.boxplot(box_data, positions=box_pos, widths=width, showfliers=False,
                   patch_artist=True, zorder=2,
                   medianprops=dict(color="#333333", lw=1.3),
                   boxprops=dict(facecolor=BOX_FACE, edgecolor="#999999", lw=0.8),
                   whiskerprops=dict(color="#999999", lw=0.8),
                   capprops=dict(color="#999999", lw=0.8))
    rng = _rng(seed)
    for xi, arr in zip(positions, data_by_pos):
        arr = np.asarray(arr, dtype=float)
        for pi, p in enumerate(patients):
            if pi < len(arr) and np.isfinite(arr[pi]):
                jx = xi + (rng.random() - 0.5) * 0.28
                ax.plot(jx, arr[pi], "o", ms=ms, color=color_of[p], alpha=0.85, zorder=3, mew=0)
    means = [float(np.nanmean(np.asarray(a, dtype=float))) if np.any(np.isfinite(a)) else np.nan
             for a in data_by_pos]
    fin = [(x, m) for x, m in zip(positions, means) if np.isfinite(m)]
    if fin:
        xs, ms_ = zip(*fin)
        ax.plot(xs, ms_, color="black", lw=2.0, marker="o", ms=4, zorder=4)
    return means


def _sig_bracket(ax, x0, x1, y, text, color="#222222", fs=8, h=None):
    yl = ax.get_ylim()
    if h is None:
        h = 0.03 * (yl[1] - yl[0])
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], lw=1.0, color=color, clip_on=False)
    ax.text((x0 + x1) / 2, y + h, text, ha="center", va="bottom", fontsize=fs,
            color=color, clip_on=False)


def _patient_colors(patients):
    cols = assign_colors(patients)
    return cols, {p: cols[i] for i, p in enumerate(patients)}


def _bin_s(records):
    for rec in records.values():
        return rec["meta"]["bin_size_ms"] / 1000.0
    return 0.1


# ── 1. Headline: peak height × temporal locking ───────────────────────────────

def fig_locking_scatter(peak_df, patients, cues, metrics=None):
    """Per metric: x = cross-patient SD of peak latency (locking), y = group peak height
    (mean±sem). One marker per alignment (cue-coloured). The trigger cue is high & left."""
    if metrics is None:
        metrics = config.METRIC_KEYS
    summ = A.peak_summary(peak_df, patients)
    n = len(metrics)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.2 * nrow), squeeze=False)
    for i, metric in enumerate(metrics):
        ax = axes[i // ncol][i % ncol]
        sub = summ[summ["metric"] == metric]
        for cue_key in cues:
            r = sub[sub["cue_key"] == cue_key]
            if len(r) == 0:
                continue
            x = float(r["latency_sd_s"].iloc[0])
            y = float(r["peak_mean"].iloc[0])
            ye = float(r["peak_sem"].iloc[0])
            c = _cue_color(cue_key)
            ax.errorbar(x, y, yerr=(0 if np.isnan(ye) else ye), fmt="o", ms=9,
                        color=c, ecolor=c, elinewidth=1.2, capsize=2, zorder=3)
            ax.annotate(config.CUE_LABELS[cue_key], (x, y), textcoords="offset points",
                        xytext=(7, 3), fontsize=7, color=c)
        ax.set_title(config.METRIC_LABEL[metric])
        ax.set_xlabel("Peak-latency SD across patients (s)  →  jitter")
        ax.set_ylabel("Group peak (mean ± s.e.m.)")
        ax.margins(0.25)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Which cue triggers semantic info? — high & LEFT = strong and time-locked",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# ── 2. Per-metric 4-cue time-course grid ──────────────────────────────────────

def fig_timecourse_grid(records, cues, metric, patients, alpha=0.05, show_patients=True):
    """One row of 4 panels (one per cue). Group mean±sem vs cue-relative time, chance
    line, group significance raster (Fisher+FDR) below the chance line, and other-cue
    mean±std bands. x=0 at the aligned cue. Shared y within the metric."""
    bin_s = _bin_s(records)
    # collect group curves + group significance per cue
    gts, gsig, bands, pres = {}, {}, {}, {}
    y_hi, y_lo = [], []
    for cue_key in cues:
        pp = A.present_patients(records, cue_key, metric, patients)
        pres[cue_key] = pp
        if not pp:
            continue
        gt = A.group_timecourse(records, cue_key, metric, pp)
        gs = S.group_perbin(records, cue_key, metric, pp, alpha=alpha)
        gts[cue_key], gsig[cue_key] = gt, gs
        bands[cue_key] = A.cue_bands(records, cue_key, pp)
        if len(gt):
            y_hi.append(np.nanmax((gt["mean"] + gt["sem"].fillna(0)).values))
            y_lo.append(np.nanmin(gt["mean"].values))
            if np.any(np.isfinite(gt["null"].values)):
                y_lo.append(np.nanmin(gt["null"].values))
    if not y_hi:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, f"no data for {metric}", ha="center"); ax.axis("off")
        return fig
    y_top = max(y_hi) * 1.12
    y_base = min(0.0, min(y_lo))
    raster_top = y_base - 0.04 * (y_top - y_base)
    raster_bot = y_base - 0.16 * (y_top - y_base)

    ncol = len(cues)
    fig, axes = plt.subplots(1, ncol, figsize=(3.4 * ncol, 3.4), squeeze=False, sharey=True)
    cols, color_of = _patient_colors(patients)
    for j, cue_key in enumerate(cues):
        ax = axes[0][j]
        gt = gts.get(cue_key)
        # other-cue bands
        for sr_name, (mu, sd) in bands.get(cue_key, {}).items():
            c = CUE_STYLE.get(sr_name, {}).get("color", "#999999")
            if sd > 0:
                ax.axvspan(mu - sd, mu + sd, color=c, alpha=0.08, lw=0, zorder=0)
            ax.axvline(mu, color=c, lw=1.0, ls="-", alpha=0.55, zorder=1)
        ax.axvline(0, color="black", lw=1.0, ls=":", zorder=2)
        ax.axhline(0, color="#bbbbbb", lw=0.6, zorder=1)
        if gt is None or len(gt) == 0:
            ax.set_title(f"{config.CUE_LABELS[cue_key]}\n(no data)")
            continue
        t = gt["t_s"].values
        # faint per-patient traces
        if show_patients:
            for p in pres[cue_key]:
                md = records[(cue_key, p)]["metrics"][metric]
                ax.plot(md["t_s"], md["obs_mean"], color=color_of[p], lw=0.7, alpha=0.35, zorder=2)
        # group mean ± sem
        ax.plot(t, gt["mean"].values, color="#111111", lw=1.8, zorder=4)
        ax.fill_between(t, gt["mean"] - gt["sem"].fillna(0), gt["mean"] + gt["sem"].fillna(0),
                        color="#111111", alpha=0.15, lw=0, zorder=3)
        # chance line
        if np.any(np.isfinite(gt["null"].values)):
            ax.plot(t, gt["null"].values, color="#444444", lw=1.0, ls="--", alpha=0.8, zorder=4)
        # significance raster (group Fisher + BH-FDR)
        gs = gsig.get(cue_key)
        if gs is not None and len(gs) and "sig_fdr" in gs:
            segs = [(row.t_s - bin_s / 2, bin_s) for row in gs.itertuples() if bool(row.sig_fdr)]
            if segs:
                ax.broken_barh(segs, (raster_bot, raster_top - raster_bot),
                               facecolors="#c62828", edgecolors="none", zorder=3)
        ax.set_title(config.CUE_LABELS[cue_key])
        ax.set_xlabel(f"Time from {config.CUE_LABELS[cue_key].lower()} (s)")
        if j == 0:
            ax.set_ylabel(config.METRIC_LABEL[metric])
        ax.set_ylim(raster_bot - 0.02 * (y_top - y_base), y_top)
    axes[0][0].text(axes[0][0].get_xlim()[0], (raster_top + raster_bot) / 2,
                    "sig.", fontsize=6.5, color="#c62828", ha="right", va="center")
    testable = config.METRIC_HAS_NULL.get(metric, False)
    note = "" if testable else "  (cosine has no per-bin null → descriptive; no significance test)"
    fig.suptitle(f"{config.METRIC_LABEL[metric]} — group mean±s.e.m. by alignment"
                 f"   [red = Fisher+FDR sig; bands = other cues mean±s.d.]{note}",
                 fontsize=9.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# ── 3. Peak-value box + points across alignments ──────────────────────────────

def fig_peak_box(peak_df, metric, patients, cues):
    """Box+points of per-patient peak height across the 4 alignments; chance line;
    brackets comparing the top-ranked alignment vs each other (paired Wilcoxon, greater)."""
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    cols, _ = _patient_colors(patients)
    positions = list(range(len(cues)))
    data_by_pos, base_by_pos = [], []
    for cue_key in cues:
        g = peak_df[(peak_df["metric"] == metric) & (peak_df["cue_key"] == cue_key)]
        vals = g.set_index("patient")["peak_val"].reindex(patients).values.astype(float)
        base = g.set_index("patient")["baseline"].reindex(patients).values.astype(float)
        data_by_pos.append(vals)
        base_by_pos.append(base)
    means = _box_points(ax, positions, data_by_pos, patients, cols)
    # chance line (mean baseline across all cues/patients)
    allbase = np.concatenate(base_by_pos)
    if np.any(np.isfinite(allbase)):
        ax.axhline(float(np.nanmean(allbase)), color="#888888", lw=1.0, ls="--", zorder=1,
                   label="chance")
    ax.set_xticks(positions)
    ax.set_xticklabels([config.CUE_LABELS[c] for c in cues])
    ax.set_ylabel(f"{config.METRIC_LABEL[metric]} — peak (per patient)")
    ax.set_title(f"Peak {config.METRIC_LABEL[metric]} by alignment")
    # brackets: best cue vs the others
    finite_all = np.concatenate([np.asarray(d, float)[np.isfinite(np.asarray(d, float))]
                                 for d in data_by_pos]) if data_by_pos else np.array([])
    if not np.any(np.isfinite(means)) or finite_all.size == 0:
        ax.legend(loc="upper right", fontsize=6.5)
        fig.tight_layout()
        return fig
    order = np.argsort([-(m if np.isfinite(m) else -np.inf) for m in means])
    best = int(order[0])
    y0 = float(np.nanmax(finite_all))
    step = 0.10 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    slot = 0
    for oi in order[1:]:
        p, n = S.paired_wilcoxon_peaks(peak_df, metric, cues[best], cues[int(oi)], "greater")
        if n >= 3:
            _sig_bracket(ax, positions[best], positions[int(oi)], y0 + step * (1 + slot),
                         S.stars(p), h=0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]))
            slot += 1
    ax.legend(loc="upper right", fontsize=6.5)
    fig.tight_layout()
    return fig


# ── 4. Peak-latency box + points ──────────────────────────────────────────────

def fig_latency_box(peak_df, metric, patients, cues):
    """Per-patient peak latency (s from cue) by alignment. Cross-patient spread = the
    temporal-locking signal (tight = the decoder found a consistent bin)."""
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    cols, _ = _patient_colors(patients)
    positions = list(range(len(cues)))
    data_by_pos = []
    for cue_key in cues:
        g = peak_df[(peak_df["metric"] == metric) & (peak_df["cue_key"] == cue_key)]
        vals = g.set_index("patient")["peak_t_s"].reindex(patients).values.astype(float)
        data_by_pos.append(vals)
    _box_points(ax, positions, data_by_pos, patients, cols)
    ax.axhline(0, color="black", lw=0.9, ls=":", zorder=1)
    # headroom band above all data for the s.d. annotations (avoid collisions)
    allv = (np.concatenate([np.asarray(d, float)[np.isfinite(np.asarray(d, float))]
                            for d in data_by_pos]) if data_by_pos else np.array([]))
    if allv.size:
        lo, hi = float(np.min(allv)), float(np.max(allv))
        rng = max(hi - lo, 1e-6)
        ax.set_ylim(lo - 0.08 * rng, hi + 0.22 * rng)
    ytxt = ax.get_ylim()[1] - 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for i, cue_key in enumerate(cues):
        v = data_by_pos[i][np.isfinite(data_by_pos[i])]
        sd = np.std(v, ddof=1) if v.size > 1 else np.nan
        ax.text(i, ytxt, f"s.d. {sd:.2f}s" if np.isfinite(sd) else "s.d. —",
                ha="center", va="top", fontsize=6.5, color="#555555")
    ax.set_xticks(positions)
    ax.set_xticklabels([config.CUE_LABELS[c] for c in cues])
    ax.set_ylabel(f"Peak latency of {config.METRIC_LABEL[metric]} (s from cue)")
    ax.set_title(f"Peak-latency locking — {config.METRIC_LABEL[metric]}")
    fig.tight_layout()
    return fig


# ── 5. Within-patient argmax-cue vote ─────────────────────────────────────────

def fig_vote(tally, cues, metrics=None):
    """Stacked bar per metric of which alignment wins (highest per-patient peak)."""
    if metrics is None:
        metrics = list(tally.index)
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ind = np.arange(len(metrics))
    bottom = np.zeros(len(metrics))
    for cue_key in cues:
        vals = np.array([tally.loc[m, cue_key] if (m in tally.index and cue_key in tally.columns)
                         else 0 for m in metrics], dtype=float)
        ax.bar(ind, vals, bottom=bottom, color=_cue_color(cue_key),
               label=config.CUE_LABELS[cue_key], width=0.7, edgecolor="white", lw=0.5)
        bottom += vals
    ax.set_xticks(ind)
    ax.set_xticklabels([config.METRIC_LABEL[m] for m in metrics], rotation=20, ha="right")
    ax.set_ylabel("# patients (winning alignment)")
    ax.set_title("Within-patient vote: which alignment peaks highest")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=len(cues), fontsize=7)
    fig.tight_layout()
    return fig


# ── 6. cue × metric peak heatmap ──────────────────────────────────────────────

def fig_heatmap(peak_df, patients, cues, metrics=None):
    """Group-mean peak per (metric, cue), z-scored within metric (across cues) so metrics
    on different scales are comparable; annotate raw group-mean peak."""
    if metrics is None:
        metrics = config.METRIC_KEYS
    summ = A.peak_summary(peak_df, patients)
    raw = np.full((len(metrics), len(cues)), np.nan)
    for i, m in enumerate(metrics):
        for j, c in enumerate(cues):
            r = summ[(summ["metric"] == m) & (summ["cue_key"] == c)]
            if len(r):
                raw[i, j] = float(r["peak_mean"].iloc[0])
    z = np.full_like(raw, np.nan)
    for i in range(len(metrics)):
        row = raw[i]
        mu, sd = np.nanmean(row), np.nanstd(row)
        z[i] = (row - mu) / sd if (sd and np.isfinite(sd)) else 0.0
    fig, ax = plt.subplots(figsize=(1.0 + 1.1 * len(cues), 0.7 + 0.55 * len(metrics)))
    im = ax.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-1.6, vmax=1.6)
    ax.set_xticks(range(len(cues)))
    ax.set_xticklabels([config.CUE_LABELS[c] for c in cues])
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([config.METRIC_LABEL[m] for m in metrics])
    for i in range(len(metrics)):
        for j in range(len(cues)):
            if np.isfinite(raw[i, j]):
                ax.text(j, i, f"{raw[i, j]:.3f}", ha="center", va="center", fontsize=6.5,
                        color="#111111")
    ax.set_title("Group-mean peak (annot.) — colour = z within metric")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="z (across cues)")
    fig.tight_layout()
    return fig


# ── 7. Per-patient detail (one metric, 4 cues overlaid) ───────────────────────

def fig_patient_detail(records, cues, metric, patients):
    """Small multiples: one subplot per patient, the 4 cue-aligned curves overlaid
    (cue-coloured), x=0 at each curve's own cue. Essential for spotting RB/AA/DR."""
    n = len(patients)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.8 * nrow), squeeze=False)
    for i, p in enumerate(patients):
        ax = axes[i // ncol][i % ncol]
        ax.axvline(0, color="black", lw=0.8, ls=":", zorder=1)
        ax.axhline(0, color="#cccccc", lw=0.6, zorder=1)
        any_data = False
        for cue_key in cues:
            rec = records.get((cue_key, p))
            if rec is None or metric not in rec["metrics"]:
                continue
            md = rec["metrics"][metric]
            ax.plot(md["t_s"], md["obs_mean"], color=_cue_color(cue_key), lw=1.2,
                    label=config.CUE_LABELS[cue_key], zorder=3)
            any_data = True
        ax.set_title(display_id(p) + (f"  ({p})" if display_id(p) != p else ""))
        if not any_data:
            ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
        if i % ncol == 0:
            ax.set_ylabel(config.METRIC_LABEL[metric])
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", ncol=len(cues), fontsize=7,
                   frameon=False)
    fig.suptitle(f"Per-patient {config.METRIC_LABEL[metric]} by alignment (x=0 at each cue)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
