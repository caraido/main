# -*- coding: utf-8 -*-
"""
figures_for_paper/cross_task/cross_task_panels.py
=================================================
Render the cross-task co-training figure from ``source_data/`` CSVs only (no
project pkls, any env with numpy/pandas/matplotlib/scipy).  Run
``compute_cross_task_data.py`` first.

Main figure (00_cross_task_combined + individual):
  a  semantic organization of the two separate per-task decoders (cosine-MDS)
  b  co-training generalization: within / cross / pooled, 3 metrics x 2 tasks
  c  task-general electrodes (VIP) — representative participant
Supplements: S1 MDS all-6, S3 VIP all-6, S4 permutation scatters all-6,
  S5 ROI knockout (partial coverage + placeholder), S7 cross-task RSA.

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
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS_ROOT = os.path.dirname(HERE)
sys.path.insert(0, FIGS_ROOT)
from paper_common import apply_paper_style  # noqa: E402

SRC = os.path.join(HERE, "source_data")
apply_paper_style()

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
METRIC_CHANCE = {"cat_indep_bal_acc": 1.0 / 6, "cosine_mean": 0.0}
TASK_PANEL = {"picture": "Picture decoder", "auditory": "Auditory decoder"}


def _cat_palette():
    d = pd.read_csv(os.path.join(SRC, "category_style.csv"))
    return dict(zip(d["category"].astype(str), d["color"]))


def _save(fig, stem, dpi=200):
    fig.savefig(os.path.join(HERE, stem + ".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(os.path.join(HERE, stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def _letter(ax, s, dx=-0.02, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="bottom", ha="right")


# ══════════════════════════════════════════════════════════════════════════
# a — semantic-organization MDS of the two separate decoders
# ══════════════════════════════════════════════════════════════════════════

def _legend_handles(cats, pal):
    return [Line2D([0], [0], marker="o", linestyle="", markersize=7,
                   markerfacecolor=pal.get(str(c), "#999999"),
                   markeredgecolor="none", label=c) for c in cats]


def draw_embedding(ax_pic, ax_aud, xcol="mds1", ycol="mds2", axis_label="MDS",
                   patient_id=None, max_points=280, seed=0, show_legend=True):
    """One shared 2D embedding (fit on both tasks jointly), split into the two
    task subplots.  Both subplots use the SAME x- and y-limits (computed from the
    combined 2nd–98th percentile) so positions are directly comparable."""
    pts = pd.read_csv(os.path.join(SRC, "panel_a_mds_points.csv"))
    align = pd.read_csv(os.path.join(SRC, "panel_a_mds_alignment.csv"))
    if patient_id is None:
        patient_id = align[align.is_representative]["display_id"].iloc[0]
    pal = _cat_palette()
    d = pts[pts.display_id == patient_id]
    a = align[align.display_id == patient_id].iloc[0]

    xall, yall = d[xcol].to_numpy(), d[ycol].to_numpy()
    xr = tuple(np.percentile(xall, [2, 98])); yr = tuple(np.percentile(yall, [2, 98]))
    xpad = 0.12 * (xr[1] - xr[0] + 1e-9); ypad = 0.12 * (yr[1] - yr[0] + 1e-9)
    rng = np.random.default_rng(seed)
    cats_present = sorted(d["category"].astype(str).unique())

    for ax, task in ((ax_pic, "picture"), (ax_aud, "auditory")):
        sub = d[d.task == task]
        n_full = len(sub)
        if n_full > max_points:                     # cap dense tasks for legibility
            sub = sub.iloc[rng.choice(n_full, max_points, replace=False)]
        order = rng.permutation(len(sub))           # random z-order (opaque dots)
        cols = [pal.get(str(c), "#999999")
                for c in sub["category"].to_numpy()[order]]
        ax.scatter(sub[xcol].to_numpy()[order], sub[ycol].to_numpy()[order],
                   s=15, alpha=1.0, c=cols, edgecolor="none", zorder=2)
        ax.set_title(f"{TASK_PANEL[task]}  (n={n_full})", fontsize=9)
        ax.set_xlabel(f"{axis_label} 1"); ax.set_xlim(xr[0] - xpad, xr[1] + xpad)
        ax.set_ylim(yr[0] - ypad, yr[1] + ypad)
        ax.axhline(0, color="0.9", lw=0.6, zorder=0)
        ax.axvline(0, color="0.9", lw=0.6, zorder=0)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(labelsize=6, length=3)
    ax_pic.set_ylabel(f"{axis_label} 2")
    if show_legend:
        ax_aud.legend(handles=_legend_handles(cats_present, pal),
                      loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6.5,
                      frameon=False, title="category", handletextpad=0.2,
                      borderpad=0.2, labelspacing=0.25)
    star = "*" if a.cat_centroid_alignment_p < 0.05 else ""
    ax_pic.text(0.02, 0.98,
                f"cross-task alignment r={a.cat_centroid_alignment:.2f}{star}",
                transform=ax_pic.transAxes, fontsize=7.5, va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.5))
    return patient_id, cats_present, pal


def fig_mds():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.5))
    rep, _, _ = draw_embedding(ax1, ax2, "mds1", "mds2", "MDS")
    _letter(ax1, "a", dx=-0.08)
    fig.suptitle(f"Separate per-task decoders — predicted GloVe (cosine-MDS), "
                 f"{rep}", fontsize=9, y=1.02)
    fig.tight_layout()
    _save(fig, "01_semantic_organization_mds")


def _fig_embedding_all(xcol, ycol, axis_label, method_name, stem):
    align = pd.read_csv(os.path.join(SRC, "panel_a_mds_alignment.csv"))
    ids = list(align["display_id"])
    fig, axes = plt.subplots(len(ids), 2, figsize=(6.6, 2.7 * len(ids)))
    for i, pid in enumerate(ids):
        draw_embedding(axes[i, 0], axes[i, 1], xcol, ycol, axis_label,
                       patient_id=pid)
        axes[i, 0].set_ylabel(f"{pid}\n{axis_label} 2", fontsize=8)
    fig.suptitle(f"Semantic organization of the separate per-task decoders "
                 f"({method_name}) — all participants", fontsize=10, y=1.005)
    fig.tight_layout()
    _save(fig, stem, dpi=170)


def fig_mds_all():
    _fig_embedding_all("mds1", "mds2", "MDS", "cosine-MDS",
                       "S1_semantic_organization_mds_all")


def fig_pca_all():
    _fig_embedding_all("pc1", "pc2", "PC", "PCA, fit on both tasks jointly",
                       "S2_semantic_organization_pca_all")


def draw_embedding_3d(ax_pic, ax_aud, cols, axis_label, patient_id,
                      max_points=280, seed=0, show_legend=True):
    """One shared 3D embedding (fit on both tasks jointly), split into the two
    task subplots; both use the SAME x/y/z-limits (combined 2nd–98th percentile)."""
    xcol, ycol, zcol = cols
    pts = pd.read_csv(os.path.join(SRC, "panel_a_mds_points.csv"))
    align = pd.read_csv(os.path.join(SRC, "panel_a_mds_alignment.csv"))
    pal = _cat_palette()
    d = pts[pts.display_id == patient_id]
    a = align[align.display_id == patient_id].iloc[0]
    lims = {}
    for c in cols:
        v = d[c].to_numpy(); lo, hi = np.percentile(v, [2, 98])
        pad = 0.08 * (hi - lo + 1e-9); lims[c] = (lo - pad, hi + pad)
    rng = np.random.default_rng(seed)
    cats_present = sorted(d["category"].astype(str).unique())

    for ax, task in ((ax_pic, "picture"), (ax_aud, "auditory")):
        sub = d[d.task == task]; n_full = len(sub)
        if n_full > max_points:
            sub = sub.iloc[rng.choice(n_full, max_points, replace=False)]
        order = rng.permutation(len(sub))
        cc = [pal.get(str(c), "#999999")
              for c in sub["category"].to_numpy()[order]]
        ax.scatter(sub[xcol].to_numpy()[order], sub[ycol].to_numpy()[order],
                   sub[zcol].to_numpy()[order], c=cc, s=11, alpha=1.0,
                   edgecolor="none", depthshade=True)
        ax.set_title(f"{TASK_PANEL[task]}  (n={n_full})", fontsize=8)
        ax.set_xlim(*lims[xcol]); ax.set_ylim(*lims[ycol]); ax.set_zlim(*lims[zcol])
        ax.set_xlabel(f"{axis_label} 1", fontsize=6, labelpad=-9)
        ax.set_ylabel(f"{axis_label} 2", fontsize=6, labelpad=-9)
        ax.set_zlabel(f"{axis_label} 3", fontsize=6, labelpad=-9)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(3))
        ax.tick_params(labelsize=5, pad=-2)
        ax.view_init(elev=18, azim=-60)
    if show_legend:
        ax_aud.legend(handles=_legend_handles(cats_present, pal),
                      loc="center left", bbox_to_anchor=(1.08, 0.5), fontsize=6,
                      frameon=False, title="category", handletextpad=0.2,
                      labelspacing=0.25)
    star = "*" if a.cat_centroid_alignment_p < 0.05 else ""
    ax_pic.text2D(0.0, 0.96, f"{patient_id}  ·  cross-task r="
                  f"{a.cat_centroid_alignment:.2f}{star}",
                  transform=ax_pic.transAxes, fontsize=7.5, fontweight="bold")
    return cats_present, pal


def _fig_embedding_all_3d(cols, axis_label, method_name, stem):
    align = pd.read_csv(os.path.join(SRC, "panel_a_mds_alignment.csv"))
    ids = list(align["display_id"])
    fig = plt.figure(figsize=(8.2, 3.4 * len(ids)))
    for i, pid in enumerate(ids):
        ax1 = fig.add_subplot(len(ids), 2, 2 * i + 1, projection="3d")
        ax2 = fig.add_subplot(len(ids), 2, 2 * i + 2, projection="3d")
        draw_embedding_3d(ax1, ax2, cols, axis_label, pid)
    fig.suptitle(f"Semantic organization of the separate per-task decoders "
                 f"({method_name}) — all participants", fontsize=11, y=0.998)
    fig.subplots_adjust(left=0.03, right=0.9, top=0.97, bottom=0.02,
                        hspace=0.15, wspace=0.02)
    _save(fig, stem, dpi=160)


def fig_mds_all_3d():
    _fig_embedding_all_3d(("mds3d_1", "mds3d_2", "mds3d_3"), "MDS",
                          "cosine-MDS, 3D", "S1_semantic_organization_mds_all_3d")


def fig_pca_all_3d():
    _fig_embedding_all_3d(("pc1", "pc2", "pc3"), "PC",
                          "PCA, 3D, fit on both tasks jointly",
                          "S2_semantic_organization_pca_all_3d")


# ══════════════════════════════════════════════════════════════════════════
# b — co-training generalization (within / cross / pooled)
# ══════════════════════════════════════════════════════════════════════════

def _bracket(ax, x0, x1, y, p_stars, h):
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], lw=0.8, color="0.35")
    ax.text((x0 + x1) / 2, y + h, p_stars, ha="center", va="bottom",
            fontsize=8 if p_stars != "n.s." else 6.5,
            color="0.2")


def draw_generalization(axes):
    """axes: 2x3 array (rows = pic/aud, cols = 3 metrics)."""
    per = pd.read_csv(os.path.join(SRC, "panel_b_generalization.csv"))
    grp = pd.read_csv(os.path.join(SRC, "panel_b_generalization_group.csv"))
    sta = pd.read_csv(os.path.join(SRC, "panel_b_generalization_stats.csv"))
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
            if metric in METRIC_CHANCE:
                ax.axhline(METRIC_CHANCE[metric], ls="--", lw=0.8,
                           color="0.5", zorder=1)
            # significance brackets
            st = sta[(sta.target == target) & (sta.metric == metric)]
            top = max(ax.get_ylim()[1], pv["value"].max() * 1.05)
            step = 0.08 * top
            order = {"within-cross": 0, "cross-pooled": 1, "within-pooled": 2}
            for _, s in st.iterrows():
                pair = s["comparison"]
                if pair not in order:
                    continue
                i0, i1 = (0, 1) if pair == "within-cross" else \
                         (1, 2) if pair == "cross-pooled" else (0, 2)
                _bracket(ax, i0, i1, top + step * (order[pair] + 0.4),
                         s["stars"], step * 0.3)
            ax.set_ylim(top=top + step * 3.4)
            ax.set_xticks(range(3))
            ax.set_xticklabels([SRC_LABEL[s] for s in SRC_ORDER], fontsize=7)
            if r == 0:
                ax.set_title(METRIC_LABEL[metric], fontsize=8.5)
            if c == 0:
                ax.set_ylabel(TARGET_LABEL[target], fontsize=8)
            ax.margins(x=0.15)


def fig_generalization():
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.0))
    draw_generalization(axes)
    _letter(axes[0, 0], "b", dx=-0.12)
    fig.suptitle("Co-training generalization — within / cross / pooled "
                 "(bars: mean±SEM, dots: participants; stars: paired "
                 "Wilcoxon, n=6)", fontsize=9, y=1.01)
    fig.tight_layout()
    _save(fig, "02_generalization")


# ══════════════════════════════════════════════════════════════════════════
# c — task-general electrodes (VIP)
# ══════════════════════════════════════════════════════════════════════════

def draw_vip(ax, patient_id=None, top_label=4, compact=False):
    vip = pd.read_csv(os.path.join(SRC, "panel_c_vip.csv"))
    fr = pd.read_csv(os.path.join(SRC, "panel_c_vip_fractions.csv"))
    if patient_id is None:
        patient_id = fr[fr.is_representative]["display_id"].iloc[0]
    d = vip[vip.display_id == patient_id].sort_values("vip", ascending=False)
    frow = fr[fr.display_id == patient_id].iloc[0]
    vals = d["vip"].to_numpy()
    colors = ["#c0392b" if v > 1 else "#c7c7c7" for v in vals]
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors, width=1.0, linewidth=0)
    ax.axhline(1.0, ls="--", lw=0.9, color="0.35")
    for i in range(min(top_label, len(d))):
        ax.text(x[i], vals[i] + 0.03, str(d["channel"].iloc[i]),
                ha="center", va="bottom", fontsize=6.5, rotation=0)
    ax.set_xlim(-0.5, len(vals) - 0.5)
    ax.set_ylabel("VIP")
    ax.set_xlabel("electrode (VIP rank)")
    ax.set_xticks([])
    frac_txt = (f"{patient_id}: {int(frow.n_vip_gt1)}/{int(frow.n_channels)} "
                f"electrodes VIP>1 ({frow.frac_vip_gt1*100:.0f}%)")
    ax.text(0.98, 0.95, frac_txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5 if compact else 7.5)
    return patient_id


def fig_vip():
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    rep = draw_vip(ax)
    fr = pd.read_csv(os.path.join(SRC, "panel_c_vip_fractions.csv"))
    ax.set_title(f"Task-general electrodes (VIP), {rep}", fontsize=9, pad=20)
    ax.text(0.5, 1.03, f"{fr.frac_vip_gt1.mean()*100:.0f}% VIP>1 on average "
            f"(range {fr.frac_vip_gt1.min()*100:.0f}"
            f"–{fr.frac_vip_gt1.max()*100:.0f}%)", transform=ax.transAxes,
            ha="center", fontsize=7.5, color="0.3")
    _letter(ax, "c", dx=-0.08, dy=1.12)
    fig.tight_layout()
    _save(fig, "03_vip_electrodes")


def fig_vip_all():
    fr = pd.read_csv(os.path.join(SRC, "panel_c_vip_fractions.csv"))
    ids = list(fr["display_id"])
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.0))
    for ax, pid in zip(axes.ravel(), ids):
        draw_vip(ax, patient_id=pid, top_label=3, compact=True)
    for ax in axes.ravel()[len(ids):]:
        ax.axis("off")
    fig.suptitle("VIP electrode importance from the co-trained model — "
                 "all participants (dashed = VIP 1; red = above average)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    _save(fig, "S3_vip_electrodes_all")


# ══════════════════════════════════════════════════════════════════════════
# S4 — permutation-importance scatters
# ══════════════════════════════════════════════════════════════════════════

_GROUP_COLOR = {"both": "#2ca02c", "picture_only": "#1f77b4",
                "auditory_only": "#d62728", "neither": "#c7c7c7"}


def fig_permutation_all():
    d = pd.read_csv(os.path.join(SRC, "panel_s4_permutation.csv"))
    ids = list(pd.unique(d["display_id"]))
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.2))
    for ax, pid in zip(axes.ravel(), ids):
        g = d[d.display_id == pid]
        for grp_name, gg in g.groupby("group"):
            ax.scatter(gg["perm_imp_pic"], gg["perm_imp_aud"], s=14,
                       color=_GROUP_COLOR.get(grp_name, "#c7c7c7"),
                       alpha=0.8, edgecolor="none", label=grp_name)
        # label the strongest few by max(|pic|,|aud|)
        g2 = g.assign(m=g[["perm_imp_pic", "perm_imp_aud"]].abs().max(axis=1))
        for _, r in g2.sort_values("m", ascending=False).head(4).iterrows():
            ax.annotate(str(r["channel"]), (r["perm_imp_pic"], r["perm_imp_aud"]),
                        fontsize=6, xytext=(2, 2), textcoords="offset points")
        ax.axhline(0, color="0.8", lw=0.6); ax.axvline(0, color="0.8", lw=0.6)
        ax.set_title(pid, fontsize=8.5)
        ax.set_xlabel("Δ cat-indep (picture)", fontsize=7)
        ax.set_ylabel("Δ cat-indep (auditory)", fontsize=7)
    fig.suptitle("Per-channel permutation importance from the co-trained model "
                 "— all participants", fontsize=10, y=1.01)
    fig.tight_layout()
    _save(fig, "S4_permutation_importance_all")


# ══════════════════════════════════════════════════════════════════════════
# S5 — ROI / region knockout (partial coverage)
# ══════════════════════════════════════════════════════════════════════════

def fig_roi():
    roi = pd.read_csv(os.path.join(SRC, "panel_s5_roi.csv"))
    cov = pd.read_csv(os.path.join(SRC, "panel_s5_roi_coverage.csv"))
    ids = list(cov["display_id"])
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    for ax, pid in zip(axes.ravel(), ids):
        crow = cov[cov.display_id == pid].iloc[0]
        if not bool(crow.has_roi):
            ax.axis("off")
            ax.text(0.5, 0.5, f"{pid}\n\nNo ROI atlas available\nfor this "
                    "participant", ha="center", va="center", fontsize=8.5,
                    color="0.4",
                    bbox=dict(boxstyle="round,pad=0.6", fc="#f5f5f5", ec="0.8"))
            continue
        g = roi[roi.display_id == pid].sort_values("perm_imp_pic",
                                                   ascending=True)
        y = np.arange(len(g))
        ax.barh(y - 0.2, g["perm_imp_pic"], height=0.4, color="#1f77b4",
                label="picture")
        ax.barh(y + 0.2, g["perm_imp_aud"], height=0.4, color="#d62728",
                label="auditory")
        ax.axvline(0, color="0.8", lw=0.6)
        ax.set_yticks(y); ax.set_yticklabels(g["region"], fontsize=6.5)
        ax.set_title(pid, fontsize=8.5)
        ax.set_xlabel("Δ cat-indep (region knockout)", fontsize=7)
        if pid == ids[0]:
            ax.legend(fontsize=6.5, loc="lower right")
    for ax in axes.ravel()[len(ids):]:
        ax.axis("off")
    fig.suptitle("Region (ROI) knockout importance from the co-trained model "
                 "— available participants", fontsize=10, y=1.01)
    fig.tight_layout()
    _save(fig, "S5_roi_knockout")


# ══════════════════════════════════════════════════════════════════════════
# S7 — cross-task RSA
# ══════════════════════════════════════════════════════════════════════════

def fig_rsa():
    r = pd.read_csv(os.path.join(SRC, "panel_s7_rsa.csv"))
    cols = ["rdm_pic_vs_aud", "rdm_pic_vs_glove", "rdm_aud_vs_glove"]
    labels = ["pic ↔ aud", "pic ↔ GloVe", "aud ↔ GloVe"]
    colors = ["#6a51a3", "#1f77b4", "#d62728"]
    ids = list(r["display_id"])
    x = np.arange(len(ids)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    for k, (c, lab, col) in enumerate(zip(cols, labels, colors)):
        ax.bar(x + (k - 1) * w, r[c], width=w, color=col, label=lab)
    ax.axhline(0, color="0.7", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(ids, fontsize=7)
    ax.set_ylabel("RDM correlation (Spearman)")
    ax.set_title("Cross-task representational similarity of the per-word neural "
                 "geometry", fontsize=9)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    fig.tight_layout()
    _save(fig, "S7_cross_task_rsa")


# ══════════════════════════════════════════════════════════════════════════
# 00 — combined main figure
# ══════════════════════════════════════════════════════════════════════════

def fig_combined():
    fig = plt.figure(figsize=(11.5, 12.5))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.05, 1.5, 0.9],
                  hspace=0.5, wspace=0.42)

    # row a — MDS (2 cols) + a small per-participant alignment col
    axa1 = fig.add_subplot(gs[0, 0]); axa2 = fig.add_subplot(gs[0, 1])
    rep, cats_present, pal = draw_embedding(axa1, axa2, "mds1", "mds2", "MDS",
                                            show_legend=False)
    _letter(axa1, "a", dx=-0.14)
    axa1.set_title("Picture decoder", fontsize=8)
    axa2.set_title("Auditory decoder", fontsize=8)
    fig.text(0.34, 0.965, f"Separate per-task decoders — predicted GloVe "
             f"(cosine-MDS), {rep}", ha="center", fontsize=9, fontweight="bold")
    # shared category legend in the gap below row a
    fig.legend(handles=_legend_handles(cats_present, pal), loc="center",
               bbox_to_anchor=(0.34, 0.665), ncol=6, fontsize=7.5, frameon=False,
               title="semantic category", columnspacing=1.1, handletextpad=0.3)
    # alignment summary across participants (small)
    axa3 = fig.add_subplot(gs[0, 2])
    al = pd.read_csv(os.path.join(SRC, "panel_a_mds_alignment.csv"))
    yy = np.arange(len(al))
    bc = ["#c0392b" if p < 0.05 else "#c7c7c7"
          for p in al["cat_centroid_alignment_p"]]
    axa3.barh(yy, al["cat_centroid_alignment"], color=bc)
    axa3.set_yticks(yy); axa3.set_yticklabels(al["display_id"], fontsize=6.5)
    axa3.set_xlabel("cross-task\ncat. alignment", fontsize=7)
    axa3.set_title("all participants (red p<.05)", fontsize=7)
    axa3.invert_yaxis()

    # row b — generalization 2x3
    gsb = gs[1, :].subgridspec(2, 3, hspace=0.55, wspace=0.4)
    axesb = np.array([[fig.add_subplot(gsb[r, c]) for c in range(3)]
                      for r in range(2)])
    draw_generalization(axesb)
    _letter(axesb[0, 0], "b", dx=-0.18)

    # row c — VIP representative (left 2/3) + fraction bars (right)
    axc = fig.add_subplot(gs[2, :2])
    repc = draw_vip(axc)
    _letter(axc, "c", dx=-0.05)
    axc.set_title(f"Task-general electrodes (VIP), {repc}", fontsize=8)
    axc2 = fig.add_subplot(gs[2, 2])
    fr = pd.read_csv(os.path.join(SRC, "panel_c_vip_fractions.csv"))
    yy = np.arange(len(fr))
    axc2.barh(yy, fr["frac_vip_gt1"] * 100, color="#c0392b")
    axc2.axvline(100 / 6, ls=":", color="0.5", lw=0.8)
    axc2.set_yticks(yy); axc2.set_yticklabels(fr["display_id"], fontsize=6.5)
    axc2.set_xlabel("% electrodes VIP>1", fontsize=7)
    axc2.set_title("all participants", fontsize=7)
    axc2.invert_yaxis()

    fig.suptitle("A single decoder co-trained on both speech tasks generalizes "
                 "across modalities", fontsize=12, fontweight="bold", y=0.995)
    _save(fig, "00_cross_task_combined", dpi=200)


def main():
    fig_mds()
    fig_generalization()
    fig_vip()
    fig_mds_all()
    fig_pca_all()
    fig_mds_all_3d()
    fig_pca_all_3d()
    fig_vip_all()
    fig_permutation_all()
    fig_roi()
    fig_rsa()
    fig_combined()
    print("[cross_task_panels] wrote figures to", HERE)


if __name__ == "__main__":
    main()
