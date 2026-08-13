# -*- coding: utf-8 -*-
"""
figures_for_paper/cross_task/cross_task_panels.py
=================================================
Render the cross-task co-training figure from ``source_data/`` CSVs only (no
project pkls, any env with numpy/pandas/matplotlib).  Run
``compute_cross_task_data.py`` first.

Main figure (00_cross_task_combined + individual 01_-03_), following the storyline:
  a  a single decoder co-trained on both tasks works — within / cross / pooled
  b  which regions does it lean on?      — Jacobian sensitivity, cross-participant ranking
  c  what can a region do alone?         — ROI-only decoder accuracy, picture vs auditory

``04_region_knockout`` is still rendered but is **not part of the shipped figure** and is
uncaptioned: knockout left the combined figure and the manuscript on 2026-08-13.

**Rebuilt 2026-08-13.** The semantic-organization MDS panel, its S1/S2 (2D + 3D) MDS/PCA
supplements, the S3 all-participant knockout supplement, the S7 RSA supplement and the
single-participant ROI bar panel were all retired here; the figure is these four panels and
nothing else.  ``panel_a_mds_points.csv``, ``panel_a_mds_alignment.csv`` and
``panel_s7_rsa.csv`` are still shipped in ``source_data/`` — they are inputs to pending work
(docs/experiments/018), NOT sources for anything drawn here.

Display conventions for panels c and d, all requested by Alec 2026-08-13 and all deliberate:
no panel titles (a title is caption text), no region legend (every region is labelled in
place), no per-participant markers, semi-transparent markers sized by participant count,
and region labels placed radially around the cloud with BLACK leader lines
(``paper_common.place_labels``).

Reproduce:
    python figures_for_paper/cross_task/cross_task_panels.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS_ROOT = os.path.dirname(HERE)
sys.path.insert(0, FIGS_ROOT)
from paper_common import (apply_paper_style, place_labels,  # noqa: E402
                          DPI_PANEL, DPI_COMBINED, AXES_LABEL_SIZE, TICK_SIZE,
                          LEGEND_SIZE)

SRC = os.path.join(HERE, "source_data")
apply_paper_style()

#: Panel letter size. Not in utils/config.py — matched to
#: semantic_regression_panels.py, the reference implementation, which uses 12.
LETTER_SIZE = 12
#: Region names inside the scatters. Deliberately below AXES_LABEL_SIZE: 17 of them share one
#: panel, and at the house label size they cannot be de-collided without leader lines so long
#: the association is lost. This is the one size in this file that is a considered exception
#: rather than a house value.
REGION_LABEL_SIZE = 7

# ── palette / labels ───────────────────────────────────────────────────────
SRC_ORDER = ["within", "cross", "pooled"]
SRC_COLOR = {"within": "#7f7f7f", "cross": "#d62728", "pooled": "#1f77b4"}
SRC_LABEL = {"within": "Within", "cross": "Cross", "pooled": "Pooled"}
TARGET_LABEL = {"pic": "Picture-naming test", "aud": "Auditory-naming test"}
METRIC_ORDER = ["cat_indep_bal_acc", "word_bal_acc", "cosine_mean"]
METRIC_LABEL = {
    "cat_indep_bal_acc": "Category-independent\nbalanced accuracy",
    "word_bal_acc": "Word balanced accuracy",
    "cosine_mean": "Cosine similarity",
}
METRIC_CHANCE = {"cosine_mean": 0.0}
TASK_LABEL = {"pic": "Picture", "aud": "Auditory"}

#: Only these two contrasts are drawn: grey-vs-red and blue-vs-red, i.e. both arms against
#: the CROSS (transfer) baseline. within-vs-pooled (grey-vs-blue) is not tested
#: (Alec, 2026-08-13), so the retention ratio in the caption carries no p-value.
#: compute_cross_task_data.py does not compute it either, so a stats row for it cannot
#: silently reappear here.
BRACKETS = ["within-cross", "cross-pooled"]

#: comparison name -> the two x positions its bracket spans, from SRC_ORDER's index.
_BRACKET_X = {"within-cross": (0, 1), "cross-pooled": (1, 2), "within-pooled": (0, 2)}


def _csv(name):
    return pd.read_csv(os.path.join(SRC, name))


def _chance_by_task():
    """``{'pic'|'aud': (mean, lo, hi)}`` chance for cat_indep_bal_acc.

    Chance is 1 / n_categories and the cohort does not share one taxonomy (the
    older auditory stimulus set adds abstract/action and drops vehicle), so this
    was a hard-coded 1/6 that was already wrong for one participant.  Read the
    per-participant table instead and draw the spread, so a heterogeneous cohort
    is never represented by a single tidy line.
    """
    f = os.path.join(SRC, "chance_by_participant.csv")
    if not os.path.exists(f):
        return {}
    d = pd.read_csv(f)
    out = {}
    for task, key in (("picture", "pic"), ("auditory", "aud")):
        t = d[d["task"] == task]
        if not t.empty:
            out[key] = (float(t["chance"].mean()),
                        float(t["chance"].min()), float(t["chance"].max()))
    return out


def _roi_colors():
    """region -> colour, resolved upstream into ``roi_style.csv``.

    Not computed here: the colours come from the vendored ``utils.roi_palette`` (so a region
    is the same colour as in the electrode_labeling brain figures) plus report-only colours
    for the regions the `tpm` scope adds, which that palette renders in one grey. Resolving
    it upstream is what keeps this script CSV-only.
    """
    d = _csv("roi_style.csv")
    return dict(zip(d["region"].astype(str), d["color"]))


def _save(fig, stem, dpi=DPI_PANEL):
    fig.savefig(os.path.join(HERE, stem + ".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(os.path.join(HERE, stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def _letter(ax, s, dx=-0.02, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=LETTER_SIZE, fontweight="bold",
            va="bottom", ha="right")


def _marker_size(n):
    """Aggregate marker area from the number of contributing participants.

    Size is the ONLY reliability cue left on panels c and d — the ``(n)`` suffix that used
    to follow each region name was removed as clutter — and four of the 17 regions come
    from one or two participants. Regions are never dropped for low n; they are drawn small.
    """
    return 60 + 34 * np.asarray(n, dtype=float)


# ══════════════════════════════════════════════════════════════════════════
# a — co-training generalization (within / cross / pooled)
# ══════════════════════════════════════════════════════════════════════════

def _bracket(ax, x0, x1, y, p_stars, h):
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], lw=0.8, color="0.35")
    ax.text((x0 + x1) / 2, y + h, p_stars, ha="center", va="bottom",
            fontsize=AXES_LABEL_SIZE if p_stars != "n.s." else TICK_SIZE - 0.5,
            color="0.2")


def draw_generalization(axes):
    """axes: 2x3 array (rows = pic/aud, cols = 3 metrics)."""
    per = _csv("panel_b_generalization.csv")
    grp = _csv("panel_b_generalization_group.csv")
    sta = _csv("panel_b_generalization_stats.csv")
    chance_by_task = _chance_by_task()
    xpos = {s: i for i, s in enumerate(SRC_ORDER)}

    for r, target in enumerate(["pic", "aud"]):
        for c, metric in enumerate(METRIC_ORDER):
            ax = axes[r, c]
            g = grp[(grp.target == target) & (grp.metric == metric)]
            for s in SRC_ORDER:
                row = g[g.source == s].iloc[0]
                ax.bar(xpos[s], row["mean"], yerr=row["sem"], width=0.7,
                       color=SRC_COLOR[s], alpha=0.9, capsize=2,
                       error_kw=dict(lw=0.8))
            # per-patient points + faint pairing lines
            pv = per[(per.target == target) & (per.metric == metric)]
            for pid, gg in pv.groupby("display_id"):
                ys = [gg[gg.source == s]["value"].iloc[0] for s in SRC_ORDER]
                ax.plot([xpos[s] for s in SRC_ORDER], ys, color="0.6",
                        lw=0.4, alpha=0.5, zorder=3)
                ax.scatter([xpos[s] for s in SRC_ORDER], ys, s=9, color="0.25",
                           zorder=4, alpha=0.8)
            if metric == "cat_indep_bal_acc" and target in chance_by_task:
                cmean, clo, chi = chance_by_task[target]
                if chi > clo:      # heterogeneous cohort: show the spread
                    ax.axhspan(clo, chi, color="0.5", alpha=0.15,
                               lw=0, zorder=0)
                ax.axhline(cmean, ls="--", lw=0.8, color="0.5", zorder=1)
            elif metric in METRIC_CHANCE:
                ax.axhline(METRIC_CHANCE[metric], ls="--", lw=0.8,
                           color="0.5", zorder=1)
            # significance brackets — the two tested contrasts only
            st = sta[(sta.target == target) & (sta.metric == metric)]
            top = max(ax.get_ylim()[1], pv["value"].max() * 1.05)
            step = 0.08 * top
            order = {p: i for i, p in enumerate(BRACKETS)}
            for _, s in st.iterrows():
                pair = s["comparison"]
                if pair not in order:
                    continue
                i0, i1 = _BRACKET_X[pair]
                _bracket(ax, i0, i1, top + step * (order[pair] + 0.4),
                         s["stars"], step * 0.3)
            ax.set_ylim(top=top + step * 2.6)
            ax.set_xticks(range(3))
            ax.set_xticklabels([SRC_LABEL[s] for s in SRC_ORDER], fontsize=TICK_SIZE)
            if r == 0:
                ax.set_title(METRIC_LABEL[metric], fontsize=AXES_LABEL_SIZE + 0.5)
            if c == 0:
                ax.set_ylabel(TARGET_LABEL[target], fontsize=AXES_LABEL_SIZE)
            ax.margins(x=0.15)


def fig_generalization():
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.0))
    draw_generalization(axes)
    _letter(axes[0, 0], "a", dx=-0.12)
    fig.tight_layout()
    _save(fig, "01_generalization")


# ══════════════════════════════════════════════════════════════════════════
# b — Jacobian sensitivity, cross-participant ROI ranking
# ══════════════════════════════════════════════════════════════════════════

def draw_jacobian(ax):
    """Regions ranked by per-electrode Jacobian enrichment, aggregated across participants.

    A RANKING, not a picture-vs-auditory plane, and that is not a stylistic choice: one
    co-trained model scores both tasks through a single shared map, so it ranks regions
    near-identically for the two tasks (rho = +0.99 per electrode) whatever the anatomy is.
    There is no interpretable off-diagonal to draw, so the tasks are averaged and the one
    thing the measure supports — a cross-participant ordering — is what is shown.

    y = the region's per-electrode ‖∂ŷ/∂x‖ divided by that participant's own whole-brain
    per-electrode average for the same task, so 1.0 (dashed) is that participant's average
    electrode. Faded markers are individual participants and are KEPT here, unlike on panels
    c and d: x is a rank, so they spread out along it instead of piling onto the aggregate.
    """
    d = _csv("panel_c_roi.csv")
    agg = _csv("panel_roi_aggregate.csv").sort_values("jac_enrich", ascending=False)
    rcol = _roi_colors()
    order = list(agg["region"].astype(str))
    xpos = {r: i for i, r in enumerate(order)}
    d = d.copy()
    d["jac_enrich"] = d[["jac_sens_pic_std", "jac_sens_aud_std"]].mean(axis=1)
    d = d[np.isfinite(d["jac_enrich"])]
    ax.axhline(1.0, ls="--", color="#666", lw=1.0, zorder=1)
    # Jitter is centred WITHIN EACH REGION, not by the participant's index in the cohort.
    # Indexing globally puts every point of a region contributed only by late participants
    # off to one side of its own tick, where it reads as belonging to the next region.
    for reg, g in d.groupby("region"):
        reg = str(reg)
        if reg not in xpos:
            continue
        g = g.sort_values("display_id")
        m = len(g)
        offs = (np.arange(m) - (m - 1) / 2.0) * 0.09
        ax.scatter(xpos[reg] + offs, g["jac_enrich"].to_numpy(), s=26,
                   color=rcol.get(reg, "#777"), edgecolors="none", alpha=0.40,
                   zorder=2)
    ax.scatter(range(len(order)), agg["jac_enrich"],
               s=_marker_size(agg["n_participants"]),
               color=[rcol.get(r, "#777") for r in order],
               edgecolors="#111", linewidths=1.4, alpha=0.80, zorder=6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=LEGEND_SIZE)
    ax.set_xlim(-0.6, len(order) - 0.4)
    # Bare label (Alec, 2026-08-13). What the normalization IS -- per electrode, divided by
    # that participant's whole-brain per-electrode average for the same task, averaged over
    # tasks -- lives in Methods, not on the axis. The caption keeps only the one clause the
    # figure convention requires: what the dashed line marks.
    ax.set_ylabel("Normalized Jacobian sensitivity")
    ax.grid(axis="y", alpha=0.3)


def fig_jacobian():
    n = len(_csv("panel_roi_aggregate.csv"))
    fig, ax = plt.subplots(figsize=(max(7.0, 0.42 * n + 2.6), 4.2))
    draw_jacobian(ax)
    _letter(ax, "b", dx=-0.06)
    fig.tight_layout()
    _save(fig, "02_jacobian_ranking")


# ══════════════════════════════════════════════════════════════════════════
# c, d — the two picture-vs-auditory region scatters
# ══════════════════════════════════════════════════════════════════════════

def _offscale(v, k=6.0):
    """Boolean mask of values so far out they would flatten everything else.

    ``v > median + k*IQR``. Used only to choose the AXIS RANGE — nothing is dropped, and an
    off-scale region is still drawn, clamped to the boundary and labelled with its true
    value. On the knockout panel pFus is 0.0133 per electrode against 0.0035 for the next
    region, so a range that contains it puts the other sixteen in one corner; this is the
    standard way to show an outlier without either hiding it or losing the rest.

    The rule is deterministic and computed from the data, not a hand-picked cutoff: at this
    arm it flags exactly pFus on the picture axis and nothing on the auditory axis.
    """
    v = np.asarray(v, dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size < 4:
        return np.zeros(v.shape, dtype=bool)
    q25, q75 = np.percentile(finite, [25, 75])
    return v > float(np.median(finite)) + k * float(q75 - q25)


def draw_region_scatter(ax, xcol, ycol, xlabel, ylabel, band=None, anchor=0.0,
                        label_fontsize=REGION_LABEL_SIZE, zoom=False,
                        label_margin=(0.16, 0.01),
                        label_gap=2.0):
    """One region per marker, at its cross-participant MEAN on each task.

    Shared by panels c and d because they differ only in which column they read and whether
    a chance band is drawn. Everything else is the same set of choices, all made 2026-08-13:

      * no title and no legend — both are caption text, and with every region labelled in
        place a colour legend is a key to something the reader is already reading;
      * no per-participant markers — at 17 regions x 7 participants the faded cloud buried
        the aggregates, which are the actual readout. The values stay in panel_c_roi.csv;
      * ``alpha=0.80`` so overlapping markers stay separable;
      * region names placed radially around the cloud with black leaders, not all leaning
        one way (``paper_common.place_labels``);
      * equal aspect and one shared range on both axes, so the dotted identity line really
        is 45 degrees and distance from it is readable as task asymmetry.

    ``anchor`` is the value the range is forced to include: 0 for a Δ measure, where zero
    means "no effect"; chance for a raw accuracy, where dragging the floor to 0 spends most
    of the axis on empty space.
    """
    d = _csv("panel_c_roi.csv")
    rcol = _roi_colors()
    dd = d[np.isfinite(d[xcol]) & np.isfinite(d[ycol])]
    agg = (dd.groupby("region")
             .agg(x=(xcol, "mean"), y=(ycol, "mean"), n=("patient", "nunique"))
             .reset_index())

    # Axis range. Under `zoom`, off-scale regions are excluded from the range calculation
    # and clamped to the boundary instead, so the bulk of the regions fill the panel rather
    # than collapsing into a corner behind a single extreme value.
    off = (_offscale(agg["x"]) | _offscale(agg["y"])) if zoom else np.zeros(len(agg), bool)
    inr = agg[~off]
    vals = np.concatenate([inr["x"].to_numpy(), inr["y"].to_numpy()])
    lo = min(float(vals.min()), anchor)
    hi = max(float(vals.max()), anchor)
    pad = ((hi - lo) or 0.02) * (0.16 if off.any() else 0.10)
    lo, hi = lo - pad, hi + pad

    if band is not None:
        # Centre lines only. The +/-1 SEM shading was removed 2026-08-13 (Alec): the band is
        # narrower than a marker on the picture axis and reads as a drawing artifact, and it
        # invited being read as a significance interval, which it is not. The band values
        # are still in source_data/roi_chance_band.csv and in the caption.
        (_cx, _, _), (_cy, _, _) = band
        ax.axvline(_cx, color="#444", lw=0.9, ls="-.", zorder=1)
        ax.axhline(_cy, color="#444", lw=0.9, ls="-.", zorder=1)
    else:
        # Dash-dot to match the chance reference in panel c: both mark "no effect / chance",
        # and drawing one solid and one dashed implied a difference in kind that is not there.
        ax.axhline(0, color="#444", lw=0.9, ls="-.", zorder=1)
        ax.axvline(0, color="#444", lw=0.9, ls="-.", zorder=1)
    ax.plot([lo, hi], [lo, hi], ls=":", color="#999", lw=0.8, zorder=1)

    # Off-scale markers are drawn AT the boundary, keeping their in-range coordinate, and
    # their label carries the true value so nothing is silently misread off the axis.
    px = agg["x"].clip(lo + pad * 0.35, hi - pad * 0.35)
    py = agg["y"].clip(lo + pad * 0.35, hi - pad * 0.35)
    ax.scatter(px, py, s=_marker_size(agg["n"]),
               color=[rcol.get(str(r), "#777") for r in agg["region"]],
               edgecolors="#111", linewidths=1.4, alpha=0.80, zorder=6)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    # Force the aspect adjustment NOW. `adjustable="box"` shrinks the axes rectangle to make
    # the plot square, but only at draw time -- until then `transData` and `transAxes` still
    # describe the un-squared cell. place_labels() below works in display space, so without
    # this it lays labels out against geometry the figure will not have, and the error scales
    # with how non-square the cell is: barely visible in the standalone panel, but enough in
    # the combined figure's wide row to deadlock `temporal pole` on top of `cingulate` no
    # matter how the margins or the minimum gap were tuned.
    ax.apply_aspect()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    labels = []
    for i, r in enumerate(agg.itertuples()):
        name = str(r.region)
        if off[i]:
            name += "\n({:.4f}, {:.4f})".format(r.x, r.y)
        labels.append((float(px.iloc[i]), float(py.iloc[i]), name))
    # Labels AFTER the limits: placement is computed in display space, so it needs the
    # final data->pixel transform.
    place_labels(ax, labels, fontsize=label_fontsize, margin_frac=label_margin,
                 gap_pt=label_gap)
    return agg


def _band_from_csv():
    """((pic_centre, lo, hi), (aud_centre, lo, hi)) from ``roi_chance_band.csv``."""
    b = _csv("roi_chance_band.csv").set_index("task")
    return tuple((float(b.loc[t, "centre"]), float(b.loc[t, "lo"]),
                  float(b.loc[t, "hi"])) for t in ("picture", "auditory"))


def draw_sufficiency(ax, label_fontsize=REGION_LABEL_SIZE, label_margin=(0.02, 0.01),
                     label_gap=2.0):
    band = _band_from_csv()
    # Labels kept essentially inside the axes at full size: panel c's markers are well
    # spread, so the relaxation has room, and a generous horizontal margin would only put
    # `angular` and `aSTG` on top of the y tick labels. Inside the COMBINED figure the panel
    # is physically smaller while the text is not, so the caller widens the margin there --
    # without it `temporal pole` and `cingulate` have nowhere to go and overlap.
    return draw_region_scatter(
        ax, "suff_pooled_pic", "suff_pooled_aud",
        "Category accuracy — Picture", "Category accuracy — Auditory",
        band=band, anchor=min(band[0][0], band[1][0]),
        label_fontsize=label_fontsize, label_margin=label_margin, label_gap=label_gap)


def draw_knockout(ax, label_fontsize=REGION_LABEL_SIZE):
    return draw_region_scatter(
        ax, "perm_imp_pic_pc", "perm_imp_aud_pc",
        "Normalized change in category accuracy — Picture",
        "Normalized change in category accuracy — Auditory",
        band=None, anchor=0.0, label_fontsize=label_fontsize, zoom=True)


def fig_sufficiency():
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    draw_sufficiency(ax)
    _letter(ax, "c", dx=-0.08)
    fig.tight_layout()
    _save(fig, "03_roi_sufficiency")


def fig_knockout():
    """INTERNAL working figure, not part of the shipped figure (Alec, 2026-08-13).

    Region knockout was dropped from the combined figure and from the Results and Methods.
    This still renders so the analysis is one command away if it comes back, and it keeps
    its ``d`` so it drops straight into the old layout. It is deliberately **uncaptioned** --
    ``figures_for_paper/README.md`` asks for one caption per *shipped* figure, and nothing in
    the manuscript references this one.
    """
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    draw_knockout(ax)
    _letter(ax, "d", dx=-0.08)
    fig.tight_layout()
    _save(fig, "04_region_knockout")


# ══════════════════════════════════════════════════════════════════════════
# 00 — combined main figure
# ══════════════════════════════════════════════════════════════════════════

def fig_combined():
    """The shipped main figure: a, b, c.

    **Region knockout is NOT in it** (Alec, 2026-08-13). It is still rendered standalone as
    ``04_region_knockout`` -- keeping its ``d`` -- but that file is an internal working
    figure: it is uncaptioned, nothing in the manuscript points at it, and the Results and
    Methods no longer make a knockout claim at all. Do not re-add it here without also
    restoring the text, or the figure will carry a panel the paper does not discuss.
    """
    fig = plt.figure(figsize=(11.8, 12.6))
    # NESTED grids, because a single GridSpec has ONE hspace and this figure needs two very
    # different ones: the a-to-b gap should be tight (a is a block of small bar panels and
    # was leaving a band of pure white beneath it), while the b-to-c gap has to clear b's
    # rotated region names, which hang a long way below its axes. With a uniform hspace,
    # closing the first gap put panel c's letter on top of "superior parietal".
    #
    # Row heights are uneven for a second reason: `c` is equal-aspect, so its drawn size is
    # min(cell width, cell height) and it only grows if BOTH grow — hence the large bottom
    # share and the 6-of-10 column span below.
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.00, 1.15], hspace=0.26)
    gsab = outer[0].subgridspec(2, 1, height_ratios=[1.05, 0.82], hspace=0.30)

    # a — generalization (2x3, spanning the full width)
    # a's own hspace has to clear its rotated row labels ("Picture-naming test" /
    # "Auditory-naming test"), which run the full height of their panels and touch at 0.46.
    gsa = gsab[0].subgridspec(2, 3, hspace=0.58, wspace=0.36)
    axes_a = np.array([[fig.add_subplot(gsa[r, c]) for c in range(3)]
                       for r in range(2)])
    draw_generalization(axes_a)
    _letter(axes_a[0, 0], "a", dx=-0.20)

    # b — Jacobian ranking (full width)
    axb = fig.add_subplot(gsab[1])
    draw_jacobian(axb)
    _letter(axb, "b", dx=-0.05)

    # row 3 — c alone, centred. Ten columns and take the middle six, so the scatter keeps its
    # square proportions (its aspect is equal, so the identity diagonal is only at 45 degrees
    # while it does) rather than being stretched to fill the row.
    #
    # The panel is at least as large as its standalone version on purpose. Shrinking it while
    # holding the text size fixed is what made the label relaxation deadlock: the dense
    # lower-left cluster had nowhere to go inside the clamp, and `temporal pole` printed
    # across `cingulate`. (The root cause was `apply_aspect` — see draw_region_scatter — but
    # a cramped panel is what made it visible, so the room is kept.)
    gsc = outer[1].subgridspec(1, 10, wspace=0.0)
    axc = fig.add_subplot(gsc[0, 2:8])
    # Margins near zero: the panel is large enough here that the labels fit inside, and a
    # horizontal margin only pushes `angular` and `aSTG` onto the y tick labels.
    draw_sufficiency(axc, label_margin=(0.01, 0.02))
    _letter(axc, "c", dx=-0.09)

    _save(fig, "00_cross_task_combined", dpi=DPI_COMBINED)


def main():
    fig_generalization()
    fig_jacobian()
    fig_sufficiency()
    fig_knockout()
    fig_combined()
    print("[cross_task_panels] wrote figures to", HERE)


if __name__ == "__main__":
    main()
