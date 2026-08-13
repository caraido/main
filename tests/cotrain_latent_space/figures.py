# -*- coding: utf-8 -*-
"""One figure per participant: the three candidate views, at the WORD level.

Deliberately QC-grade, not paper-grade. The pilot exists to decide WHICH view to build
properly; whichever wins gets rebuilt in ``figures_for_paper/`` under that folder's
conventions.

Two design decisions, both forced by the first pass:

**Points are words, not trials.** The single-trial clouds overlapped almost completely in
all three views, which is a finding rather than a plotting failure -- but it made the panels
unreadable and it weighted each word by however many times it happened to be presented.

**Every centroid carries a bootstrap ellipse, and every category carries its null.** The
question "is this cross-task shift meaningful?" cannot be answered by looking at two dots and
a line, which is exactly what the first pass asked a reader to do. The ellipse is the
sampling error of the centroid itself (resampled over words), so two overlapping ellipses
mean the shift is not resolvable at this n. The right-hand panel gives the same question a
number: the observed cross-task cosine per category against its category-shuffle null.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

# Same category hues as the shipped cross-task figure, so a pilot panel and a paper panel
# never disagree about which colour is "animal". Imported rather than restated.
from figures_for_paper.cross_task.compute_cross_task_data import CATEGORY_COLORS

_SPARE = ["#a65628", "#bcbd22", "#000000", "#777777"]


def _palette(cats):
    spare = iter(_SPARE)
    return {c: CATEGORY_COLORS.get(c) or next(spare, "#777777")
            for c in sorted(set(map(str, cats)))}


def _centroid_and_ellipse(P, n_boot=400, seed=0):
    """(centroid, 95 % bootstrap ellipse of the centroid) for a set of word-level points.

    Resamples WORDS with replacement, so the ellipse answers "how much would this centroid
    move if I had drawn a different sample of words?" -- which is the uncertainty that a
    cross-task displacement has to be read against. Returns None for the ellipse when there
    are too few words to resample meaningfully.
    """
    c = P.mean(axis=0)
    if len(P) < 3:
        return c, None
    rng = np.random.default_rng(seed)
    boots = np.array([P[rng.integers(0, len(P), len(P))].mean(axis=0)
                      for _ in range(n_boot)])
    cov = np.cov(boots.T)
    if not np.all(np.isfinite(cov)) or np.linalg.det(cov) <= 0:
        return c, None
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ang = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    # 2 * sqrt(5.991 * eigenvalue) = full width of the 95 % ellipse in 2 dof
    w, h = 2 * np.sqrt(5.991 * np.maximum(vals, 0))
    return c, (w, h, ang)


def draw_view(ax, E, cats, task, title, pal, seed=0, show_words=True):
    """One 2-D view at the word level: word points faint, centroids with 95 % ellipses,
    matched categories joined across tasks."""
    is_pic = task == "picture"
    if show_words:
        for mask, marker in ((is_pic, "o"), (~is_pic, "^")):
            ax.scatter(E[mask, 0], E[mask, 1], s=16, marker=marker, alpha=0.30,
                       c=[pal.get(str(c), "#999") for c in cats[mask]],
                       edgecolors="none", zorder=2)
    for c in sorted(set(cats[is_pic]) & set(cats[~is_pic])):
        col = pal.get(str(c), "#999")
        cp, ep = _centroid_and_ellipse(E[is_pic & (cats == c)], seed=seed)
        ca, ea = _centroid_and_ellipse(E[~is_pic & (cats == c)], seed=seed + 1)
        for cen, ell in ((cp, ep), (ca, ea)):
            if ell is not None:
                ax.add_patch(Ellipse(cen, ell[0], ell[1], angle=ell[2], facecolor=col,
                                     alpha=0.16, edgecolor=col, lw=0.8, zorder=4))
        ax.plot([cp[0], ca[0]], [cp[1], ca[1]], color=col, lw=1.6, alpha=0.9, zorder=5)
        ax.scatter(*cp, s=130, marker="o", color=col, edgecolors="#111",
                   linewidths=1.4, zorder=6)
        ax.scatter(*ca, s=130, marker="^", color=col, edgecolors="#111",
                   linewidths=1.4, zorder=6)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=6)
    ax.axhline(0, color="0.92", lw=0.6, zorder=0)
    ax.axvline(0, color="0.92", lw=0.6, zorder=0)


def draw_percat(ax, pc, pal, title):
    """Per-category cross-task cosine against its own category-shuffle null.

    This is the panel that makes "meaningful" answerable without geometry: a bar above its
    null marker is a category whose two tasks agree more than label-shuffling would produce.
    """
    if pc is None or pc.empty:
        ax.axis("off")
        return
    pc = pc.sort_values("cosine", ascending=True)
    y = np.arange(len(pc))
    ax.barh(y, pc["cosine"], color=[pal.get(str(c), "#999") for c in pc["category"]],
            edgecolor="#111", linewidth=0.8, alpha=0.85, zorder=3)
    ax.scatter(pc["null_p95"], y, marker="|", s=180, color="#111", zorder=5,
               label="shuffle null, 95th pct")
    ax.axvline(0, color="#444", lw=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(["{}  (p={:.3g})".format(c, p)
                        for c, p in zip(pc["category"], pc["p"])], fontsize=7)
    ax.set_xlabel("cross-task centroid cosine", fontsize=8)
    ax.set_xlim(-1.05, 1.05)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6.5, loc="lower right", frameon=False)


def patient_figure(views, cats, task, patient, summary, out_png, percat=None, seed=0):
    """``views`` = [(key, title, E or None)]; one column each, plus a per-category column."""
    pal = _palette(cats)
    n = len(views)
    fig, axes = plt.subplots(1, n + 1, figsize=(4.3 * (n + 1), 4.6))
    for ax, (key, title, E) in zip(axes, views):
        if E is None:
            ax.text(0.5, 0.5, "not computable\n(see summary.csv)", ha="center",
                    va="center", transform=ax.transAxes, fontsize=8, color="#b00")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        r = summary.get(key, {})
        sub = title
        if np.isfinite(r.get("alignment", np.nan)):
            sub += "\nalignment r={:.3f}, p={:.3g}".format(r["alignment"], r["alignment_p"])
        draw_view(ax, E, cats, task, sub, pal, seed=seed)
    draw_percat(axes[-1], percat, pal, "per category, best view\n(bar > marker = agrees)")

    handles = [Line2D([0], [0], marker="o", ls="", markersize=7, markerfacecolor=v,
                      markeredgecolor="none", label=k) for k, v in pal.items()]
    handles += [Line2D([0], [0], marker="o", ls="", markersize=8, markerfacecolor="#bbb",
                       markeredgecolor="#111", label="picture centroid"),
                Line2D([0], [0], marker="^", ls="", markersize=8, markerfacecolor="#bbb",
                       markeredgecolor="#111", label="auditory centroid")]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 8),
               fontsize=7, frameon=False)
    fig.suptitle("{} — co-trained decoder, three candidate 2-D views. One point per WORD; "
                 "shaded = 95 % bootstrap ellipse of the centroid.".format(patient),
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def diagnostics_figure(diag, comps, patient, out_png):
    """Which latent components carry category, and which are task axes.

    Drawn per participant rather than summarised: a cohort mean would hide a participant
    whose leading components are pure task. (Measured: none are — max task AUC is 0.60–0.81,
    so the co-trained space barely encodes task at all.)
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    k = diag["component"].to_numpy()
    chosen = np.isin(k, comps)
    axes[0].bar(k, diag["cat_f_min"], color=np.where(chosen, "#2ca02c", "#c7c7c7"))
    axes[0].set_xlabel("PLS component"); axes[0].set_ylabel("category F (min over tasks)")
    axes[0].set_title("carries category in BOTH tasks", fontsize=9)
    axes[1].bar(k, diag["task_auc"], color=np.where(chosen, "#2ca02c", "#c7c7c7"))
    axes[1].axhline(0.5, ls="--", lw=0.9, color="#666")
    axes[1].set_ylim(0.45, 1.0)
    axes[1].set_xlabel("PLS component"); axes[1].set_ylabel("task AUC (0.5 = no task info)")
    axes[1].set_title("separates picture from auditory", fontsize=9)
    for ax in axes:
        ax.set_xticks(k)
    fig.suptitle("{} — latent-component diagnostics (green = plotted)".format(patient),
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
