# cross_task_region_importance_report.py
# HTML report from cross_task_region_importance.py output (region_importance_all.csv).
#
# Region-only successor to the retired per-channel report
# (_archive/cross_task_reports/cross_task_channel_importance_report.py).
#
# Structure: ONE PART PER ATLAS — NMM and DK, peers not variants — each carrying the
# same six sections, so the same question can be read under both parcellations.
# (Was fine vs coarse ROIs; the coarse anterior/posterior merge was retired 2026-08-08
# along with primary_roi.) An arm with no CSV is skipped, so this renders from one atlas:
#   1. Δ category accuracy   (region knockout), per electrode   — pic-vs-aud scatter
#   2. Δ cosine to GloVe     (region knockout), per electrode   — pic-vs-aud scatter
#   3. Jacobian sensitivity, per electrode                      — cross-participant ROI ranking
#   4. Neural–GloVe covariance, per electrode                   — pic-vs-aud scatter
#   5. Co-trained vs single-modality decoders (3 panels + ROI × decoder heatmap)
#   6. Region SUFFICIENCY: ROI-only decoder vs matched-N null   — pic-vs-aud scatter
#      (sections 1-5 all measure NECESSITY — what breaks when a region is removed;
#      section 6 is the complement, what a region can do alone. Optional: present
#      only when --roi-sufficiency produced the suff_* columns.)
#
# Only section 3 is a RANKING rather than a scatter. The Jacobian reads one co-trained
# model that scores both tasks through a shared map, so its pic-vs-aud plane has no
# interpretable off-diagonal (ρ = +0.99 per electrode, structural) — see
# `_roi_ranked_strip`. Covariance (4) keeps its scatter: it involves no model at all,
# computed separately on each task's own trials, so an asymmetry there is a property
# of the data. Its raw region-total diagonal (ρ = +0.96) was an electrode-count
# artifact though — per electrode it falls to −0.09, which is why it is normalized.
#
# Region TOTALS are never plotted (ρ 0.96-0.99 with ROI channel count); everything is
# per electrode or per-electrode enrichment. Plain-PLS VIP and the retrieval-aligned
# Jacobian were retired 2026-07-23 — pre-existing CSVs still carry those columns and
# this report ignores them.
#
# Inputs (from --in-dir, default: main/results/cross_task_cotrain/):
#   region_importance_nmm_all.csv      (permutation Δacc/Δcosine + Jacobian + covariance
#                                        + whole-brain ceiling, grouped by nmm_roi)
#   region_importance_dk_all.csv       (the same, grouped by dk_roi. NOT a relabelling of
#                                        the NMM file: each atlas gates channel selection
#                                        too, so the two are different channel sets.)
# At least one is required; both is the intended state.
#
# Output (default): <in-dir>/region_importance_report.html
#
# Usage:
#   python -m analysis.cross_task.cross_task_region_importance_report
#   python -m analysis.cross_task.cross_task_region_importance_report --metric cat_indep_bal_acc
#   python -m analysis.cross_task.cross_task_region_importance_report --in-dir <dir> --out <out.html>

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAIN_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)
DEFAULT_IN_DIR = Path(_MAIN_DIR) / "results" / "cross_task_cotrain"

from utils.roi_palette import (color_of, ordered as roi_ordered,  # noqa: E402
                               OTHER, OTHER_COLOR)
from utils.config import (ROI_ATLAS_DEFAULT,                     # noqa: E402
                          OLD_STIMULUS_SET_PATIENTS)
from report.helper.html_utils import fig_to_base64               # noqa: E402
from report.render import Document                               # noqa: E402

METRIC_SLUG = {
    "cat_indep_bal_acc": "catindep",
    "word_bal_acc": "word",
    "cosine_mean": "cosine",
}

_GCOL = {"both": "#2ca02c", "picture_only": "#1f77b4",
         "auditory_only": "#d62728", "neither": "#bbbbbb"}

# Patient marker glyphs for the aggregated scatter (one per participant).
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "p", "h"]



# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

import re as _re

# The fold/table-of-contents machinery that used to live here (a module-level _TOC
# list plus _fold/_toc_html/_TOC_SCRIPT) is now report.render.Document. The list
# being module-level was a latent bug: two reports built in one process appended to
# the same contents.


def _fig_to_img(fig, alt: str) -> str:
    b64 = fig_to_base64(fig, dpi=140)
    return '<img alt="{}" src="data:image/png;base64,{}" />'.format(alt, b64)


def _region_colors(regions) -> dict:
    """Region -> colour, from the one shared palette (``utils.roi_palette``).

    Replaced a 20-entry list indexed by ALPHABETICAL RANK on 2026-08-08.  That
    assignment had two failure modes, and the second is fatal for a two-atlas report:

    1. Adding or removing one region shifted the colour of every region after it, so no
       two runs of the report agreed on what colour anything was.
    2. It was called separately per part, on that part's own region set.  NMM and DK do
       not have the same regions -- DK has ``pSTS``, NMM does not -- so a region present
       in both would be drawn in DIFFERENT colours in the two panels, which is exactly
       the comparison the reader is being asked to make.

    The shared palette is keyed on region name, so a region is the same colour in both
    panels by construction, and the same colour as in the electrode_labeling brain
    figures.

    Regions outside the vendored 13
    -------------------------------
    ``utils.roi_palette`` covers the 13 and is vendored + drift-checked, so it cannot be
    extended from this repository -- ``color_of`` returns ``OTHER_COLOR`` for anything else,
    and that is the SAME grey as the ``other``/unknown sentinel.  Under a wider
    ``--roi-scope`` (``tpfm`` admits 10 more regions) that made eleven different things one
    indistinguishable colour across all six sections, with as many identical grey swatches
    in the legend.

    So the vendored 13 keep their exact vendored colours, ``#9a9a9a`` stays reserved for the
    sentinel alone, and anything else is given a distinguishable colour here.  These are
    REPORT colours, not palette colours: they are deliberately desaturated so they never read
    as vendored, they are assigned by sorted name so a region keeps its colour across both
    atlas parts and across runs (the failure mode 1. above), and no vendored file is touched,
    which is what keeps ``scripts/check_roi_vocabulary.py`` green.
    """
    # Two sentinels, not one. `other` is utils.roi_palette's; `unknown` is what
    # _build_region_labels writes for a channel the atlas could not label. Neither is a
    # region, so both keep the reserved grey -- giving "unlabelled" a cheerful colour of
    # its own would read as a finding.
    sentinels = {OTHER, "unknown", ""}
    names = sorted(set(map(str, regions)))
    out = {}
    extra = [n for n in names if n not in sentinels and color_of(n) == OTHER_COLOR]
    # Muted, evenly spaced hues — visibly a different family from the vendored palette.
    ramp = plt.get_cmap("tab20b")
    extra_colors = {n: mcolors.to_hex(ramp(i / max(len(extra) - 1, 1) * 0.95))
                    for i, n in enumerate(extra)}
    for n in names:
        out[n] = OTHER_COLOR if n in sentinels else extra_colors.get(n) or color_of(n)
    return out


def _palette_note(rcol) -> str:
    """One line naming the regions drawn in report-only colours, or '' when there are none."""
    off = sorted(r for r, c in rcol.items() if c != color_of(r) and c != OTHER_COLOR)
    if not off:
        return ""
    return (
        "<p class='note'><b>Colour note.</b> "
        f"{len(off)} region(s) are outside the vendored 13-region palette and are drawn in "
        "report-only colours so they can be told apart: <i>" + ", ".join(off) + "</i>. "
        "These colours are assigned by this report and do not match the "
        "<code>electrode_labeling</code> brain figures; the 13 vendored regions do. "
        "Grey <span style='color:#9a9a9a'>&#9632;</span> remains reserved for "
        "<code>other</code>/unrecognised.</p>")


# ── Sections 1-2: the two KNOCKOUT measures, per channel ──────────────────────
# Picture-vs-auditory scatters. Only the knockouts get this treatment: they are
# the only measures whose off-diagonal is interpretable, because they are the only
# ones that re-score the model per task rather than reading a shared map (per
# electrode, pic-vs-aud is +0.07 for Δcat-acc and −0.01 for Δcosine — i.e. genuinely
# task-specific, unlike the Jacobian's structural +0.99). Region TOTALS are never
# plotted: within participant they correlate 0.96-0.99 with ROI channel count.
MEASURES_KNOCKOUT_PC = [
    dict(key="catacc_pc", xcol="perm_imp_pic_pc", ycol="perm_imp_aud_pc",
         name="Δ category-independent accuracy (region knockout)  · per channel",
         axis="Δ cat-indep accuracy / channel",
         blurb="Drop in retrieval category accuracy when the whole region is knocked "
               "out, divided by the region's electrode count. The end-task measure — "
               "furthest from what the PLS model optimises, and the only measure "
               "carrying a significance test."),
    dict(key="cosine_pc", xcol="cos_imp_pic_pc", ycol="cos_imp_aud_pc",
         name="Δ cosine to GloVe (region knockout)  · per channel",
         axis="Δ cosine(ŷ, GloVe) / channel",
         blurb="Drop in cosine between predicted and true GloVe when the region is "
               "knocked out — the knockout closest to the decoder's own objective."),
]

# Columns needing a `_pc` form: the two knockouts (sections 1-2), jac_sens (the
# solo section's middle panel), and every single-modality column (section 5).
_PC_COLS = ["perm_imp_pic", "perm_imp_aud", "cos_imp_pic", "cos_imp_aud",
            "jac_sens_pic", "jac_sens_aud"]


def _add_per_channel(df):
    """Add `<col>_pc = <col> / n_channels` for the columns the report plots.
    Safe no-op for missing columns."""
    if df is None or df.empty or "n_channels" not in df.columns:
        return df
    n = df["n_channels"].replace(0, np.nan)
    cols = set(_PC_COLS)
    cols.update(c for c in df.columns if c.endswith("_solo"))   # single-modality
    for c in cols:
        if c in df.columns:
            df[c + "_pc"] = df[c] / n
    return df


# ── Sections 3-4: the two SCALE-BEARING magnitude measures, cross-participant ──
# Both are read as a PER-ELECTRODE ENRICHMENT (`<col>_std`): the region's
# per-electrode value ÷ that participant's whole-brain per-electrode average for the
# same task. To pool an ROI across people you must remove BOTH (a) the
# per-participant magnitude scale (γ/‖A‖/HGA amplitude for the Jacobian; the
# 1/√n_trials floor for covariance) AND (b) the ROI's channel count.
# ≈1 = as informative per electrode as the participant's average; >1 = enriched.
#
# They render DIFFERENTLY. The Jacobian gets `_roi_ranked_strip` because it is
# task-blind by construction — one co-trained model scores both tasks through one
# shared map, so its pic-vs-aud plane has no interpretable off-diagonal at all
# (ρ = +0.99 per electrode). Covariance keeps the pic-vs-aud scatter: it is computed
# separately per task with no model in the loop, so a task asymmetry there is a
# property of the data and worth being able to see.
_STD_COLS = [("jac_sens_pic", "jac_sens_aud"), ("cov_nc_pic", "cov_nc_aud")]

# section 3 — ranked strip (Jacobian only)
_STD_SPECS = [
    ("jac", "jac_sens_pic", "jac_sens_aud",
     "Jacobian sensitivity · cross-participant",
     "per-electrode ‖∂ŷ/∂x‖ ÷ whole-brain avg"),
]

# section 4 — pic-vs-aud scatter (covariance)
MEASURES_COV = [
    dict(key="cov_std", xcol="cov_nc_pic_std", ycol="cov_nc_aud_std",
         name="Neural–GloVe covariance (null-corrected) · per electrode",
         axis="per-electrode covariance ÷ whole-brain avg",
         blurb="Region-total standardized neural↔GloVe cross-covariance — the rawest form "
               "of the PLS objective and the only <b>model-free</b> measure here: no fit, no "
               "split, computed separately on each task's own trials. Null-corrected (the "
               "1/√n_trials floor subtracted) and shown as a per-electrode enrichment, ÷ that "
               "participant's whole-brain average <i>for the same task</i>."),
]

# ROI sufficiency (--roi-sufficiency). Delta first: it is the size-controlled
# quantity. Raw accuracy is shown second and must not be ranked across regions.
MEASURES_SUFF = [
    dict(key="suff_delta", xcol="suff_delta_pic", ycol="suff_delta_aud",
         name="ROI-only decoder vs matched-N null (Δ accuracy)",
         axis="Δ cat-indep accuracy vs same-size random channels",
         blurb="A co-trained decoder trained on <b>only this region's channels</b>, minus the "
               "mean of K decoders trained on random channel sets of the <b>same size</b> drawn "
               "from the whole brain (same splits, same seed, same &gamma;). Positive = this "
               "region decodes better than any N electrodes would. This is the "
               "<b>size-controlled</b> quantity and the one to rank on."),
    dict(key="suff_pooled", xcol="suff_pooled_pic", ycol="suff_pooled_aud",
         name="ROI-only decoder · raw accuracy (not size-controlled)",
         axis="cat-indep balanced accuracy",
         # Accuracy, not a Δ: frame from chance, not from 0. Forcing 0 into range put
         # every region in one corner of an axis that was mostly empty.
         anchor="chance",
         blurb="The same decoder's raw held-out accuracy. Shown for reference only: it rises "
               "with electrode count, so a cross-region ranking here is substantially an "
               "implant-coverage ranking. Rank on the Δ panel above where it is present, or "
               "on the size-detrended panel below."
               "<br><br><b>The grey band is the shuffled-null chance level</b>, one per "
               "task: each participant's label-shuffled category-independent accuracy is "
               "averaged, and the band is the <b>mean &plusmn; 1 SEM across "
               "participants</b> &mdash; the precision of the cohort's chance estimate, "
               "which is the quantity on the same footing as the markers (each of which is "
               "itself a cross-participant aggregate). "
               "A <b>&plusmn;1 SD of the pooled shuffles</b> was tried first and is far too "
               "wide to be informative: 0.029 picture / 0.071 auditory, roughly 70&times; "
               "and 14&times; the between-participant spread, because it measures how much "
               "a <i>single shuffle</i> moves rather than how well chance is pinned down. "
               "Note the SEM does narrow with participant count. "
               "<b>It marks where chance sits; it is not a test.</b> The nulls come from the "
               "whole-brain semantic-regression decoder "
               "(<code>figures_for_paper/semantic_regression/panels_cache_*.npz</code>; "
               "picture <code>tp</code>/h5, auditory <code>tpfm</code>/h10) while these "
               "markers are ROI-only decoders with far fewer channels under a different "
               "split scheme, so a small region's own null would be wider and the band is "
               "anti-conservative for small ROIs."),
    dict(key="suff_resid", xcol="suff_resid_pic", ycol="suff_resid_aud",
         name="ROI-only decoder · size-detrended accuracy",
         axis="accuracy residual vs log₂(region channel count)",
         blurb="Raw accuracy with its electrode-count trend removed <b>empirically</b>: within "
               "each participant, that participant's regions are fitted with "
               "acc ~ a + b·log&#8322;(n_channels) and the residual is plotted. Zero = the "
               "accuracy this region's size would predict for this participant; positive = it "
               "beats its own size. <b>Do not divide accuracy by channel count instead.</b> "
               "Knockout &Delta;acc has a zero floor and is roughly additive over electrodes, "
               "so a per-electrode rate is meaningful there. An accuracy has a <b>chance</b> "
               "floor and saturates, so acc/n hands every 1&ndash;2 channel region a huge score "
               "for carrying no information at all: measured on this arm, acc/n correlates "
               "&rho;&nbsp;=&nbsp;&minus;0.97 with channel count &mdash; it inverts the size "
               "ranking rather than removing it &mdash; against &rho;&nbsp;=&nbsp;&minus;0.11 "
               "for this residual and +0.27 for the raw accuracy. "
               "<b>This is a de-trending, not a test.</b> It cannot say whether a region beats "
               "random channels of its own size; only the matched-N null does that, and it "
               "carries the p-value. Use this to rank when the null has not been run."),
]

# Section 5: co-trained vs single-modality. Covariance is excluded (model-free —
# there is no "decoder" to train one-per-task). Columns are per-channel (_pc).
MEASURES_SOLO = [
    dict(key="catacc", name="Δ category accuracy", axis="Δ cat-acc / channel",
         solo_pic="perm_imp_pic_solo_pc", cotr_pic="perm_imp_pic_pc",
         solo_aud="perm_imp_aud_solo_pc", cotr_aud="perm_imp_aud_pc",
         mid_x="perm_imp_pic_pc", mid_y="perm_imp_aud_pc"),
    dict(key="cosine", name="Δ cosine to GloVe", axis="Δ cosine / channel",
         solo_pic="cos_imp_pic_solo_pc", cotr_pic="cos_imp_pic_pc",
         solo_aud="cos_imp_aud_solo_pc", cotr_aud="cos_imp_aud_pc",
         mid_x="cos_imp_pic_pc", mid_y="cos_imp_aud_pc"),
    dict(key="jac", name="Jacobian sensitivity", axis="‖∂ŷ/∂x‖ / channel",
         solo_pic="jac_sens_pic_solo_pc", cotr_pic="jac_sens_pic_pc",
         solo_aud="jac_sens_aud_solo_pc", cotr_aud="jac_sens_aud_pc",
         mid_x="jac_sens_pic_pc", mid_y="jac_sens_aud_pc"),
]

_MEASURES_SOLO_NOTE = (
    "<div class='box'><b>This is the section that can speak to task specificity.</b>&nbsp; "
    "Everywhere else on this page a single co-trained model scores both tasks, so a "
    "picture&ndash;auditory agreement can be structural rather than anatomical (the Jacobian's "
    "&rho;&nbsp;= +0.99 per electrode is exactly that). Here the <code>_solo</code> columns come "
    "from <b>two independently trained decoders</b> &mdash; a picture-only and an auditory-only "
    "kernel-PLS on the same splits &mdash; so their agreement is an empirical result. It is much "
    "weaker: per electrode, solo picture vs solo auditory is &rho;&nbsp;= +0.08 "
    "(&#916;cat-acc), +0.02 (&#916;cosine), +0.43 (Jacobian) &mdash; against +0.99 for the "
    "co-trained Jacobian. The Jacobian's near-perfect co-trained diagonal is therefore an "
    "artifact of the shared map, and this is the direct evidence for that. "
    "<b>The second result here is asymmetric:</b> co-trained-vs-solo agreement is "
    "&rho;&nbsp;&asymp;&nbsp;0.94&ndash;0.99 for picture but only 0.53&ndash;0.78 for auditory "
    "&mdash; co-training largely <i>preserves</i> which ROIs the picture decoder used and "
    "<i>reorganizes</i> the auditory ones.<br><br>"
    "Does co-training rely on the "
    "same ROIs a single-task decoder would? For each measure, the <b>left</b> scatter plots the "
    "<b>picture-only</b> decoder's per-electrode ROI importance (x) against the <b>co-trained</b> model's "
    "picture importance (y); the <b>right</b> does the same for <b>auditory-only</b> vs co-trained "
    "auditory; the <b>middle</b> is the co-trained model itself (picture vs auditory). Points on the "
    "diagonal in left/right = co-training preserved that ROI's reliance; off-diagonal = it reorganized. "
    "Ringed markers are the cross-participant aggregate (size &prop; n). "
    "<b>Covariance is omitted</b> (model-free &mdash; there is no decoder to train one per task). "
    "<b>Caveat:</b> "
    "the auditory-only decoder is underpowered where the auditory task has few trials/repeats (AA, DR ~1 "
    "trial/word) &mdash; read its points as noisy.</div>")


def _heatmap(df, measure, out_title):
    """ROI × condition heatmap for one measure: rows = ROIs (aggregated across
    participants), columns = [picture-only, co-trained·pic, co-trained·aud,
    auditory-only], cells min–max scaled within the measure."""
    pairs = list(zip(["picture-only", "co-trained·pic", "co-trained·aud", "auditory-only"],
                     [measure["solo_pic"], measure["cotr_pic"],
                      measure["cotr_aud"], measure["solo_aud"]]))
    seen, use = set(), []                      # keep present columns, dedupe
    for lab, c in pairs:
        if c in df.columns and c not in seen:
            seen.add(c); use.append((lab, c))
    if len(use) < 2:
        return ""
    labels = [l for l, _ in use]; cols = [c for _, c in use]
    med = df.groupby("region")[cols].agg(AGG)
    sort_col = measure["cotr_pic"] if measure["cotr_pic"] in med.columns else cols[0]
    med = med.reindex(med[sort_col].sort_values(ascending=False).index)
    M = med.to_numpy(dtype=float)
    finite = M[np.isfinite(M)]
    if finite.size:
        lo, hi = float(np.nanmin(M)), float(np.nanmax(M))
        Mn = (M - lo) / (hi - lo) if hi > lo else np.zeros_like(M)
    else:
        Mn = M
    fig, ax = plt.subplots(figsize=(3.6, max(2.6, 0.34 * len(med) + 0.8)))
    im = ax.imshow(Mn, cmap="magma", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels[:len(cols)], rotation=35, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(med))); ax.set_yticklabels(med.index, fontsize=7.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, "{:.2g}".format(v), ha="center", va="center", fontsize=6,
                        color="white" if Mn[i, j] < 0.6 else "black")
    ax.set_title(out_title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="{} (min–max scaled)".format(AGG))
    fig.tight_layout()
    return _fig_to_img(fig, out_title)


def section_solo(df, rcol, dfs_for_lims):
    """Co-trained vs single-modality comparison: per measure, a 3-scatter agreement
    row (solo-pic vs co-trained-pic | co-trained pic-vs-aud | solo-aud vs
    co-trained-aud) + an ROI × condition heatmap."""
    if not any(c.endswith("_solo_pc") for c in df.columns):
        return ""
    blocks = [_MEASURES_SOLO_NOTE]
    for m in MEASURES_SOLO:
        need = [m["solo_pic"], m["cotr_pic"], m["solo_aud"], m["cotr_aud"], m["mid_x"], m["mid_y"]]
        if not all(c in df.columns for c in need):
            continue
        lim_pic = _shared_limits(dfs_for_lims, m["solo_pic"], m["cotr_pic"])
        lim_mid = _shared_limits(dfs_for_lims, m["mid_x"], m["mid_y"])
        lim_aud = _shared_limits(dfs_for_lims, m["solo_aud"], m["cotr_aud"])
        left = _aggregated_scatter(df, rcol, lim_pic, xcol=m["solo_pic"], ycol=m["cotr_pic"],
                                   title="picture: single-modality vs co-trained",
                                   xlabel="picture-only decoder", ylabel="co-trained · picture")
        mid = _aggregated_scatter(df, rcol, lim_mid, xcol=m["mid_x"], ycol=m["mid_y"],
                                  axis=m["axis"], title="co-trained (picture vs auditory)")
        right = _aggregated_scatter(df, rcol, lim_aud, xcol=m["solo_aud"], ycol=m["cotr_aud"],
                                    title="auditory: single-modality vs co-trained",
                                    xlabel="auditory-only decoder", ylabel="co-trained · auditory")
        heat = _heatmap(df, m, "{} — ROI × decoder".format(m["name"]))
        blocks.append(
            "<details class='meas' open><summary>{}</summary>".format(m["name"]) +
            "<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px'>"
            "<div>{}</div><div>{}</div><div>{}</div></div>".format(left, mid, right) +
            "<div style='max-width:26rem'>{}</div></details>".format(heat))
    return "<h2>Co-trained vs single-modality decoders</h2>" + "".join(blocks)


def _add_standardized(df):
    """Add `<col>_std` = per-channel value ÷ the participant's whole-brain per-channel
    average FOR THE SAME TASK — removing the per-participant scale, the ROI channel
    count, and the picture-vs-auditory scale offset, so an ROI is comparable when
    pooled across participants. Covariance uses its null-corrected columns.

    Reference is PER TASK (changed 2026-07-23). It used to be a single scalar per
    participant, joint over pic+aud, on the reasoning that a joint reference
    preserves the pic-vs-aud asymmetry the scatter exists to show. It does not: the
    two tasks have very different trial counts, so their raw magnitudes sit on
    different scales and a joint reference imports that offset wholesale. Under the
    old joint reference, raw covariance put 100 % of auditory ROIs above 1 and 94 %
    of picture ROIs below it — every point on one side of the diagonal by
    construction, with distribution skew of −0.04 (i.e. symmetric, so skew could not
    explain it). A per-task reference centres each task's own distribution on its own
    whole-brain average. The cost, stated in the gallery note: distance from the
    diagonal is now RELATIVE ROI rank between tasks, not an absolute magnitude
    difference — which is the only one of the two that was ever interpretable."""
    if df is None or df.empty or "patient" not in df.columns or "n_channels" not in df.columns:
        return df
    g = df.groupby("patient")
    totch = g["n_channels"].transform("sum")               # participant's total channels
    nch = df["n_channels"].replace(0, np.nan)
    # driven by _STD_COLS, not _STD_SPECS — the two measures render differently
    # (Jacobian ranked, covariance scattered) but both need the same `_std` form
    for xp, ya in _STD_COLS:
        if xp not in df.columns or ya not in df.columns:
            continue
        for col in (xp, ya):                               # one reference per task
            ref_pc = (g[col].transform("sum") / totch).replace(0, np.nan)
            df[col + "_std"] = (df[col] / nch) / ref_pc
    return df


#: Cross-participant aggregator for the ringed markers, the heatmap and the ROI ranking.
#: "median" (default) or "mean", set once from --aggregate. A module global rather than a
#: threaded parameter to match ``_FALLBACK_TEST_FRAC`` below; every aggregation site reads
#: it so the page cannot end up mixing the two.
AGG = "median"


def _add_size_detrended(df) -> None:
    """Add ``suff_resid_{pic,aud}``: ROI-only accuracy minus its own size trend.

    Fitted WITHIN each participant, ``acc ~ a + b*log2(n_channels)`` over that
    participant's regions, residual kept. Within-participant because implant coverage is a
    property of the participant: pooling would let one densely-implanted participant set
    the trend for everyone.

    log2 rather than n: accuracy saturates with electrodes, so it is roughly linear in the
    log. On this arm the residual sits at rho = -0.11 with channel count against +0.27 for
    the raw accuracy -- most of the trend removed, not all.

    Needs >= 3 regions with >= 2 distinct sizes to fit; participants below that keep NaN
    rather than a fit through two points. No-op when the sufficiency columns are absent.
    """
    for tag in ("pic", "aud"):
        src, dst = "suff_pooled_{}".format(tag), "suff_resid_{}".format(tag)
        if src not in df.columns:
            continue
        df[dst] = np.nan
        for pat, g in df.groupby("patient"):
            ok = g[np.isfinite(g[src]) & np.isfinite(g["n_channels"]) & (g["n_channels"] > 0)]
            if len(ok) < 3 or ok["n_channels"].nunique() < 2:
                continue
            x = np.log2(ok["n_channels"].to_numpy(dtype=float))
            y = ok[src].to_numpy(dtype=float)
            b, a = np.polyfit(x, y, 1)
            df.loc[ok.index, dst] = y - (a + b * x)


#: Where the participant filter reads per-bin significance from. This is the
#: semantic_regression figure's shipped table, chosen 2026-08-13. **It is a different
#: configuration from any cross-task arm** -- its picture arm is `tp`/h5 and its auditory
#: arm `tpfm`/h10 -- so the filter is "participants whose semantic decoding was significant
#: in the shipped time-course figure", not "...in this report's own runs". That is a real
#: caveat and the page states it; the alternative needs the 92 MB per-patient pkls, which a
#: CSV-only report script has no business loading.
_SIG_SOURCE = Path(_MAIN_DIR) / "figures_for_paper" / "semantic_regression" / \
    "source_data" / "source_data.csv"


def significant_participants(metric="category_indep", rule="both"):
    """Participants with >= 1 significant time bin for *metric*, or None if unavailable.

    ``rule='both'`` requires significance in picture AND auditory. That is the only rule
    that filters anything at this cohort: measured 2026-08-13, 'either' selects all 9 and
    'picture' selects all 9, so both are no-ops; 'both' and 'auditory' both give 7,
    dropping AZ and DR.
    """
    if not _SIG_SOURCE.exists():
        return None
    d = pd.read_csv(_SIG_SOURCE)
    d = d[(d["metric"] == metric) & d["significant"].astype(bool)]
    if d.empty:
        return None
    by_task = {t: set(g["patient"]) for t, g in d.groupby("task")}
    pic, aud = by_task.get("picture", set()), by_task.get("auditory", set())
    return sorted(pic & aud) if rule == "both" else sorted(pic | aud)


#: Shuffled-null caches written by figures_for_paper/semantic_regression. Each holds
#: ``{patient}__category_indep__null`` of shape (n_shuffles, n_bins) -- the label-shuffled
#: chance accuracy, which is what makes a chance *band* possible at all. Small (< 1 MB), so
#: the report reads them directly rather than touching the 92 MB per-patient pkls.
_NULL_CACHE = {
    "pic": Path(_MAIN_DIR) / "figures_for_paper" / "semantic_regression"
           / "panels_cache_picture_GloVe.npz",
    "aud": Path(_MAIN_DIR) / "figures_for_paper" / "semantic_regression"
           / "panels_cache_auditory_GloVe.npz",
}


def _chance_band(df):
    """((pic_centre, pic_lo, pic_hi), (aud_centre, aud_lo, aud_hi)) or None.

    Built from the **shuffled nulls**. Centre: the mean of every kept participant's pooled
    label-shuffled category-independent accuracy. Half-width: the **SEM across
    participants** -- SD of the per-participant null means divided by sqrt(n_participants).

    Three forms were tried, in this order (Alec, 2026-08-13):

    * percentile across participants -- rejected, a CI-like form whose width tracks n;
    * mean +/- pooled SD -- rejected as too wide to be informative, and rightly: the pooled
      SD is the spread of a *single shuffle's* accuracy (0.029 picture / 0.071 auditory),
      while every marker is a cross-participant aggregate, so the band was ~70x/14x wider
      than the uncertainty that actually belongs to a marker and swallowed the panel;
    * mean +/- SEM across participants -- what this returns. It is the precision of the
      cohort's chance estimate, which is the quantity on the same footing as the markers.

    NB the SEM is n-dependent (SD/sqrt(n)), the property that ruled out the percentile form.
    It is used because it is matched in scale to what the markers are, not because it
    escapes that dependence.

    One thing it is emphatically NOT: the spread of ``1/n_categories``. That is a
    deterministic constant per participant -- exactly 1/6 for everyone on the current
    stimulus set -- so its "range" collapses to zero width the moment the odd-category
    participant is set aside. A constant has no distribution; this does.

    Averaged over ALL bins rather than the cross-task peak bin: chance does not depend on
    time (< 0.004 difference, measured), and a peak bin would need the upstream run ids,
    which this report has no manifest to tell it.

    **The old-stimulus-set participants are excluded** (Alec, 2026-08-13), via
    ``utils.config.OLD_STIMULUS_SET_PATIENTS`` intersected with the run's cohort -- never
    hard-coded, since that membership already changed once when CP was retired. Their
    category inventory differs (RB: 7 picture / 5 auditory against 6/6), which put RB's
    measured auditory null at 0.199 against ~0.167 for everyone else and made it the sole
    author of the band's upper edge. They still contribute their accuracy to the region
    markers; only the chance reference leaves them out, and the panel says so.
    """
    keep = [p for p in sorted(set(df["patient"].astype(str)))
            if p not in set(OLD_STIMULUS_SET_PATIENTS)]
    if not keep:                        # everyone ran the old set: no basis to exclude
        keep = sorted(set(df["patient"].astype(str)))
    out = []
    for tag in ("pic", "aud"):
        path = _NULL_CACHE[tag]
        if not path.exists():
            return None
        z = np.load(path)
        per_pat = []
        for pat in keep:
            key = "{}__category_indep__null".format(pat)
            if key in z:
                v = np.asarray(z[key], dtype=float).ravel()
                v = v[np.isfinite(v)]
                if v.size:
                    per_pat.append(float(v.mean()))
        if len(per_pat) < 2:
            return None
        per_pat = np.asarray(per_pat)
        mu = float(per_pat.mean())
        sem = float(per_pat.std(ddof=1) / np.sqrt(per_pat.size))
        out.append((mu, mu - sem, mu + sem))
    return tuple(out)


def _excluded_from_chance(df):
    """Which participants the chance band leaves out, for the on-panel note."""
    return sorted(set(df["patient"].astype(str)) & set(OLD_STIMULUS_SET_PATIENTS))


def _shared_limits(dfs, xcol, ycol, margin=0.08, anchor=0.0):
    """One (lo, hi) range shared by BOTH axes across the scatters of a measure,
    framed to the per-ROI aggregate across participants (the emphasized markers), so
    the robust view reads clearly and a single participant's extreme faded point may
    fall outside rather than squishing everything. Equal range → 45° diagonal.

    ``anchor`` is the value the range is forced to include. 0.0 is right for a Δ measure,
    where zero means "no effect". It is wrong for a raw accuracy: ROI-only accuracies sit
    at 0.15-0.30, so dragging the floor to 0 spends most of the axis on empty space and
    clusters every region into one corner. Sufficiency passes chance instead.
    """
    if not dfs:
        return -0.01, 0.01
    meds = []
    for d in dfs:
        if xcol in d and ycol in d:
            dd = d[np.isfinite(d[xcol]) & np.isfinite(d[ycol])]
            m = dd.groupby("region").agg(x=(xcol, AGG), y=(ycol, AGG))
            meds.append(m["x"].values); meds.append(m["y"].values)
    vals = np.concatenate(meds) if meds else np.array([])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -0.01, 0.01
    lo = min(float(vals.min()), anchor)
    hi = max(float(vals.max()), anchor)
    span = (hi - lo) or 0.02
    pad = span * margin
    return lo - pad, hi + pad


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def _place_labels(ax, labels, min_gap_pt=9.0, dx_pt=6.0):
    """Annotate `labels` = [(x, y, text)] with vertical de-collision.

    Region markers cluster tightly on these panels -- several ROIs routinely share a y to
    within a point or two -- and matplotlib will happily stack the text on top of itself.
    Labels are pushed apart in DISPLAY space (points, not data units, because the axes are
    equal-aspect but the two variables are not on the same numeric scale as the figure),
    then anchored back to their marker with a hairline leader when they have moved far
    enough that the association would otherwise be ambiguous.

    Labels also flip to the OUTSIDE: a marker in the left half is labelled leftwards, one in
    the right half rightwards. Everything ran rightward before, so the dense middle cluster
    pushed its text straight over its neighbours' markers.

    Greedy single pass over y within each side: cheap, stable, and good enough for <= ~25
    labels. It does not attempt horizontal packing, so a dense cluster becomes a vertical
    column, which is the readable failure mode.
    """
    if not labels:
        return
    inv = ax.transData.inverted()
    x0, x1 = ax.get_xlim()
    mid = 0.5 * (x0 + x1)
    dpi_scale = ax.figure.dpi / 72.0                     # points -> pixels
    min_gap = min_gap_pt * dpi_scale

    for side in ("left", "right"):
        sel = [(x, y, t) for x, y, t in labels
               if (x < mid) == (side == "left")]
        if not sel:
            continue
        pts = [(ax.transData.transform((x, y)), (x, y), t) for x, y, t in sel]
        pts.sort(key=lambda p: p[0][1])                  # by display y, bottom-up
        placed, last_y = [], None
        for (px, py), data_xy, text in pts:
            ty = py if last_y is None else max(py, last_y + min_gap)
            placed.append((px, ty, py, data_xy, text))
            last_y = ty
        sign = -1.0 if side == "left" else 1.0
        ha = "right" if side == "left" else "left"
        for px, ty, py, (dx, dy), text in placed:
            tx_data, ty_data = inv.transform((px + sign * dx_pt * dpi_scale, ty))
            moved = abs(ty - py) > 2.0 * dpi_scale
            ax.annotate(
                text, xy=(dx, dy), xytext=(tx_data, ty_data), textcoords="data",
                fontsize=7, fontweight="bold", va="center", ha=ha, zorder=7,
                arrowprops=(dict(arrowstyle="-", color="#999", lw=0.5,
                                 shrinkA=0, shrinkB=2) if moved else None))


def _aggregated_scatter(df, rcol, lims,
                        xcol="perm_imp_pic", ycol="perm_imp_aud",
                        axis="Δ cat-indep accuracy",
                        title="All regions, all participants — colour = region",
                        xlabel=None, ylabel=None, chance=None, band=None) -> str:
    """All patients' regions on one (x, y) plane. Colour = region (shared across
    subjects), marker = patient. Two legends (region colour, patient marker).
    `lims=(lo,hi)` is the shared equal-scale range applied to both axes."""
    lo, hi = lims
    patients = sorted(df["patient"].unique())
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    # Per-participant points are NOT drawn (Alec, 2026-08-13): with 17 regions x 9
    # participants the faded cloud dominated the panel and the aggregate markers -- the
    # actual readout -- had to be picked out of it. The per-participant values remain in
    # region_importance_<atlas>_all.csv, and the ROI-ranked strip still shows them.
    med = (df[np.isfinite(df[xcol]) & np.isfinite(df[ycol])]
           .groupby("region").agg(x=(xcol, AGG), y=(ycol, AGG),
                                   n=("patient", "nunique")).reset_index())
    # Uniform markers: colour = region, size = participant count. No per-marker encoding
    # of a significance test -- the panel shows where the regions sit and where chance is,
    # and leaves testing to the measures that actually carry a null.
    labels = []
    for _, r in med.iterrows():
        reg = str(r["region"])
        ax.scatter(r["x"], r["y"], s=70 + 34 * r["n"], color=rcol.get(reg, "#777"),
                   edgecolors="#111", linewidths=1.8, alpha=0.98, zorder=6)
        labels.append((float(r["x"]), float(r["y"]), reg))
    ax.plot([lo, hi], [lo, hi], ls=":", color="#999", lw=0.8, zorder=1,
            label="_pic = aud")
    if chance is None:
        ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    else:
        # Two lines and two bands, not one of each: picture and auditory chance are
        # separate distributions measured on separate tasks, and a single reference would
        # misstate one of the axes.
        (cx, xlo, xhi), (cy, ylo, yhi) = band
        ax.axvspan(xlo, xhi, color="#444", alpha=0.10, zorder=0)
        ax.axhspan(ylo, yhi, color="#444", alpha=0.10, zorder=0)
        ax.axvline(cx, color="#444", lw=0.9, ls="-.", zorder=1)
        ax.axhline(cy, color="#444", lw=0.9, ls="-.", zorder=1)
        txt = ("shuffled-null chance, mean across participants "
               "(pic {:.4f} / aud {:.4f})\n"
               "shaded = ±1 SEM across participants "
               "({:.4f}–{:.4f} / {:.4f}–{:.4f})".format(
                   cx, cy, xlo, xhi, ylo, yhi))
        excl = _excluded_from_chance(df)
        if excl:
            txt += "\nexcludes {} (earlier stimulus set)".format(", ".join(excl))
        ax.annotate(txt, xy=(0.015, 0.985), xycoords="axes fraction", va="top",
                    fontsize=7, color="#444")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel if xlabel else "{} — picture".format(axis))
    ax.set_ylabel(ylabel if ylabel else "{} — auditory".format(axis))
    ax.set_title(title + "\n(marker = {} across participants, size ∝ n)".format(AGG),
                 fontsize=9)
    # Labels are placed AFTER the limits are set: de-collision works in display space, so
    # it needs the final data->pixel transform.
    _place_labels(ax, labels)
    # region colour legend (only regions actually present)
    from matplotlib.lines import Line2D
    # roi_ordered, not sorted(): the vendored order groups by family, anterior/ventral
    # first, which is the order every other ROI axis in the project uses. It was imported
    # here from the start and never called, so this legend has been alphabetical -- which
    # scatters the anterior/posterior pairs and gets worse the more regions there are.
    regs_present = roi_ordered(set(str(r) for r in df["region"]))
    reg_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=rcol[r],
                          markeredgecolor="#333", markersize=8, label=r)
                   for r in regs_present]
    # No participant legend: individual markers are no longer drawn, so it would be a key
    # to symbols that do not appear on the panel.
    ax.legend(handles=reg_handles, title="region", fontsize=7,
              loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, ncol=1)
    fig.tight_layout()
    return _fig_to_img(fig, "aggregated region scatter")


def _roi_ranked_strip(df, rcol, base, axis, title) -> str:
    """ROI-ranked strip plot: x = ROI (ranked by descending cross-participant
    median), y = per-electrode enrichment, one faded point per participant plus a
    ringed median.

    This is the plot style for the JACOBIAN, which is task-blind by construction:
    the co-trained model scores both tasks through one shared map, so it ranks ROIs
    near-identically for the two tasks (rho = +0.99 per electrode) whatever the
    anatomy is. A pic-vs-aud plane for it has no interpretable off-diagonal, so the
    tasks are collapsed (mean of the two enrichment columns) and the ONE thing the
    measure can support - a cross-participant ROI ranking - is what is drawn.

    Covariance was briefly drawn this way too and is back on a pic-vs-aud scatter
    (`section_cov`): it is model-free, computed separately on each task's own trials,
    so its task asymmetry is a property of the data rather than of a shared map.

    Unlike `_aggregated_scatter` this panel DOES keep the per-participant markers: its x is
    a rank, so the individuals spread out along it instead of piling onto the aggregate.
    Colour = region, marker = participant,
    ringed median sized by the number of contributing participants.

    No ROI is dropped for low participant count. Instead every ROI carries an
    `n=` / `ch=` annotation, because the ranking is substantially a SIZE ranking:
    ROI-level rho(median enrichment, mean channel count) was -0.71 (fine) / -0.75
    (coarse) under the retired primary_roi parcellation; ROI size and ROI identity
    are collinear by implant design. The
    reader needs the channel count in the same glance as the rank."""
    xc, yc = base + "_pic_std", base + "_aud_std"
    if xc not in df.columns or yc not in df.columns:
        return ""
    d = df[["patient", "region", "n_channels"]].copy()
    d["_v"] = df[[xc, yc]].mean(axis=1)             # joint pic+aud enrichment
    d = d[np.isfinite(d["_v"])]
    if d.empty:
        return ""

    agg = (d.groupby("region")
            .agg(med=("_v", AGG), n=("patient", "nunique"),
                 nch=("n_channels", "mean"))
            .sort_values("med", ascending=False))
    order = list(agg.index)
    xpos = {r: i for i, r in enumerate(order)}
    patients = sorted(d["patient"].unique())
    pmark = {p: _MARKERS[i % len(_MARKERS)] for i, p in enumerate(patients)}

    fig, ax = plt.subplots(figsize=(max(7.5, 0.7 * len(order) + 3.5), 5.4))
    ax.axhline(1.0, ls="--", color="#666", lw=1.0, zorder=1)
    # faded individual participant points, jittered on x so ties stay readable
    for _, r in d.iterrows():
        reg = str(r["region"])
        k = xpos[reg]
        off = (patients.index(r["patient"]) - (len(patients) - 1) / 2.0) * 0.10
        ax.scatter(k + off, r["_v"], s=40, color=rcol.get(reg, "#777"),
                   marker=pmark[r["patient"]], edgecolors="none", alpha=0.45, zorder=2)
    # ringed cross-participant aggregate — the robust readout
    for reg, row in agg.iterrows():
        ax.scatter(xpos[reg], row["med"], s=70 + 34 * row["n"],
                   color=rcol.get(str(reg), "#777"), edgecolors="#111",
                   linewidths=1.8, alpha=0.98, zorder=6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        ["{}\nn={:d}  ch={:.0f}".format(r, int(agg.loc[r, "n"]), agg.loc[r, "nch"])
         for r in order], rotation=45, ha="right", fontsize=7.5)
    ax.set_xlim(-0.6, len(order) - 0.4)
    # reference meaning goes in the label, not an inline annotation — at 10-15 ROIs
    # an in-axes note at y=1.0 lands on top of the densest part of the data
    ax.set_ylabel(axis + "\n(dashed 1.0 = participant's average electrode)")
    ax.set_title(title + "\n(ringed = {a} across participants, size ∝ n; "
                         "faded = individual; ROIs ranked by {a})".format(a=AGG),
                 fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker=pmark[p], color="w",
                              markerfacecolor="#888", markeredgecolor="#333",
                              markersize=8, label=p) for p in patients],
              title="participant", fontsize=7, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    return _fig_to_img(fig, title)


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def section_overview(df: pd.DataFrame) -> str:
    wb = df.groupby("patient")[["wb_imp_pic", "wb_imp_aud"]].first()
    counts = df.groupby("patient").size()
    sig = (df[df.group != "neither"].groupby("patient").size()
           .reindex(counts.index).fillna(0).astype(int))
    head = ("<table class='results'><tr><th>participant</th><th># regions</th>"
            "<th># sig regions</th><th>WB ceiling pic</th><th>WB ceiling aud</th></tr>")
    body = "".join(
        "<tr><td class='text'>{p}</td><td>{n}</td><td>{s}</td>"
        "<td>{wp:+.4f}</td><td>{wa:+.4f}</td></tr>".format(
            p=p, n=int(counts[p]), s=int(sig[p]),
            wp=wb.loc[p, "wb_imp_pic"], wa=wb.loc[p, "wb_imp_aud"])
        for p in counts.index)
    tbl = head + body + "</table>"
    return "<h2>Cross-participant overview</h2>" + tbl


_KNOCKOUT_NOTE = (
    "<div class='box'><b>Region knockout, per electrode.</b>&nbsp; Each scatter places every "
    "participant's region at its <b>picture</b> (x) and <b>auditory</b> (y) importance, divided by "
    "the region's electrode count. Colour = region (shared across participants), marker = "
    "participant; ringed markers are the per-ROI cross-participant aggregate (size &prop; n) and are "
    "the robust readout &mdash; one participant's outlier cannot dominate them. "
    "<b>These two measures get a picture-vs-auditory plane because they are the only ones whose "
    "off-diagonal means anything.</b> They re-score the model separately on each task's held-out "
    "trials, so a region can genuinely matter for one task and not the other (per electrode, "
    "pic-vs-aud is &rho;&nbsp;= +0.07 for &#916;cat-acc and &minus;0.01 for &#916;cosine). The "
    "Jacobian below cannot do this &mdash; see its note. "
    "<b>Region totals are never plotted</b> (removed 2026-07-23): within participant they correlate "
    "with ROI channel count at &rho;&nbsp;= 0.99 (Jacobian) and 0.96 (covariance), so as "
    "cross-participant quantities they measure the implant, not the brain. The knockouts are the "
    "size-robust exception (&rho;&nbsp;= 0.19), but are shown per electrode for consistency. "
    "Caveats: the per-electrode mean is noisier for small ROIs (2&ndash;4 channels), and "
    "&ldquo;per channel&rdquo; is a rough heuristic here &mdash; a joint region knockout is not the "
    "sum of its per-channel effects. Only &#916;cat-acc carries a significance test; under the "
    "Nystroem-RBF dilution 52/53 regions land in <span class='ns'>neither</span>, so read the "
    "<i>ranking</i>, not per-region certification.</div>")

_RANKED_NOTE = (
    "<div class='box'><b>Cross-participant ROI ranking, per electrode.</b>&nbsp; "
    "x&nbsp;= ROI ranked by descending cross-participant aggregate; y&nbsp;= <b>per-electrode "
    "enrichment</b>, i.e. the region's value &divide; its channel count, divided by that "
    "participant's whole-brain per-electrode average <b>for the same task</b>, then averaged over "
    "the two tasks. Faded markers are individual participants, ringed markers the median "
    "(size &prop; n). "
    "<b>Why this one is not plotted picture-vs-auditory.</b> The Jacobian reads a single "
    "co-trained model that scores both tasks through one shared map, so it ranks ROIs "
    "near-identically for picture and auditory (&rho;&nbsp;= <b>+0.99 even per electrode</b>) "
    "whatever the anatomy is &mdash; that diagonal is structural and is <b>not</b> evidence of "
    "amodal coding. There is no interpretable off-diagonal to draw, so the tasks are collapsed and "
    "the one thing the measure supports &mdash; a cross-participant ROI ranking &mdash; is what is "
    "shown. For task specificity use the two knockouts above or the single-modality section below. "
    "<b>Reading the y-axis.</b> 1.0 = that participant's own whole-brain average electrode. But "
    "the reference is a channel-weighted <i>mean</i> of a right-skewed quantity, so the median ROI "
    "sits slightly below it (1.01 for the Jacobian) &mdash; an ROI just under 1 is the modal case, "
    "not a depleted region. And the reference is the "
    "participant's own <i>implant</i>, so 1.0 is implant-relative, not brain-relative: two people "
    "with identical physiology but different coverage get different enrichment for the same ROI. "
    "<b>The ranking is substantially a SIZE ranking.</b> ROI-level &rho;(median enrichment, mean "
    "channel count) was <b>&minus;0.71</b> / <b>&minus;0.75</b> under the retired "
    "<code>primary_roi</code> parcellation (not yet recomputed for NMM/DK), because ROI size "
    "and ROI identity are collinear by implant design &mdash; depth shanks and MTG strips carry "
    "~20 contacts, ventral gyral ROIs 3&ndash;6. That is why every ROI is labelled with its mean "
    "channel count (<code>ch=</code>) as well as its participant count (<code>n=</code>): read them "
    "together with the rank. Normalization cannot remove this; even within participant, enrichment "
    "correlates &asymp;&minus;0.33 with channel count. "
    "<b>No ROI is dropped for low n</b>, so rows with <code>n=1</code> or <code>n=2</code> are "
    "single-participant observations, not group results. <b>No ROI clears a BH-corrected "
    "group-level test</b> of enrichment against 1 across participants. Treat this as a "
    "descriptive ranking, not a finding. "
    "<b>(The specific p/q and &rho; values in these notes are HARD-CODED from the 2026-07-23 "
    "audit and are NOT recomputed when this report is regenerated &mdash; they describe the "
    "7-participant, 50-epoch-picture analysis. Recompute from region_importance_all.csv before "
    "quoting any of them. The former claim that MTG is the largest ROI in every participant was "
    "checked on 2026-07-30 and is false at n=8: aMTG/pMTG rank 2nd&ndash;5th by channel count "
    "and largest in none.)</b></div>")

_COV_NOTE = (
    "<div class='box'><b>Neural&ndash;GloVe covariance, per electrode.</b>&nbsp; "
    "x&nbsp;= picture, y&nbsp;= auditory, both as <b>per-electrode enrichment</b>: the region's "
    "value &divide; its channel count, divided by that participant's whole-brain per-electrode "
    "average <b>for the same task</b>. Colour = region, marker = participant; ringed markers are "
    "the per-ROI cross-participant aggregate (size &prop; n). "
    "<b>Why this one keeps a picture-vs-auditory plane while the Jacobian does not.</b> Covariance "
    "involves <i>no model</i> &mdash; it is computed separately on each task's own trials &mdash; "
    "so a task asymmetry here is a property of the data rather than of a shared decoder map. The "
    "Jacobian's near-perfect diagonal is structural; this one is not, so it is worth being able to "
    "see. <b>But do not read agreement here as amodality either:</b> as region <i>totals</i> "
    "covariance's pic-vs-aud agreement was &rho;&nbsp;= +0.96, which was almost entirely the "
    "electrode-count artifact &mdash; per electrode it falls to &minus;0.09. Whatever structure "
    "survives normalization is what this panel shows. "
    "<b>Reading the axes.</b> 1.0 on each axis = that participant's own whole-brain average "
    "electrode <i>for that task</i>. Because the two references are separate, distance from the "
    "diagonal is <b>relative ROI rank between tasks</b>, not an absolute magnitude difference "
    "&mdash; a joint reference was tried and imported the tasks' trial-count scale offset wholesale "
    "(it put 100&nbsp;% of auditory ROIs on one side by construction). Null-corrected values are "
    "used throughout (<code>cov_nc</code>: the 1/&radic;n_trials floor subtracted, clipped at 0), "
    "since the raw floor otherwise sorts participants by trial count. Same size caveat as the "
    "Jacobian ranking: ROI size and identity are collinear by implant design, so enrichment retains "
    "&rho;&nbsp;&asymp;&nbsp;&minus;0.33 with channel count within participant.</div>")


_SUFF_NOTE = (
    "<div class='box'><b>Region sufficiency &mdash; can this region decode on its own?</b>&nbsp; "
    "Every other section on this page measures <b>necessity</b>: one decoder is trained on all "
    "channels and a region is destroyed at test time, so a region scores highly only if the rest "
    "of the brain cannot compensate. Two regions carrying the same information therefore both "
    "look unimportant. This section is the complement &mdash; the co-trained decoder is "
    "<b>trained on only that region's channels</b> and tested on it. Read the two together: "
    "high knockout + high sufficiency = dominant and unique; <b>low knockout + high sufficiency "
    "= redundant</b>, which knockout alone cannot see; high knockout + low sufficiency = "
    "necessary but not sufficient alone. "
    "<b>The size control is the &Delta;, not a per-electrode divide.</b> Dividing an accuracy by "
    "electrode count is not the correction it is for the knockout: knockout &Delta;acc is roughly "
    "additive over electrodes, an accuracy saturates, so a per-channel accuracy would rank the "
    "smallest regions highest. Instead each region is compared against K decoders trained on "
    "<b>random channel sets of its own size</b>, drawn from the whole brain including its own "
    "channels (excluding them would give every region a different reference population; the "
    "overlap is conservative, deflating &Delta; most for the largest regions). "
    "<b>Kernel width is fixed across regions.</b> sklearn's default &gamma;&nbsp;=&nbsp;1/n_features "
    "would make the RBF bandwidth a function of region size &mdash; a 97&times; spread between a "
    "1-channel region and the whole brain &mdash; so &gamma; is pinned to the whole-brain value "
    "everywhere. For the whole brain that <i>is</i> the default, so the knockout model is "
    "unchanged. "
    "<b>Caveats.</b> Regions inherit the <b>whole-brain</b> per-task peak bin (re-peaking per "
    "region would select on the same data used to score it), so a region peaking elsewhere is "
    "evaluated off-peak and understated. Below ~5 channels the kernel is numerically degenerate "
    "and absolute <code>suff_*</code> values are uninterpretable &mdash; only the &Delta; is, "
    "because the matched-N null shares the same dimensionality. The whole-brain decoder is "
    "<b>not</b> an upper bound (more features under a fixed &gamma; can hurt), so there is no "
    "share-of-ceiling reading here and these values must never be placed on the same axis as the "
    "knockout &Delta;acc &mdash; one is an accuracy, the other a change in accuracy. The "
    "permutation p floors at 1/(K+1).</div>")


#: Replaces the matched-N paragraphs when the null was not computed. Without it the page
#: kept its "vs matched-N null" heading and a note explaining a size control that never ran,
#: while `section_measures` silently dropped the all-NaN delta panel -- the report would
#: have advertised a control it did not have.
_SUFF_NOTE_NO_NULL = (
    "<div class='box'><b>Region sufficiency &mdash; can this region decode on its own?</b>&nbsp; "
    "Every other section on this page measures <b>necessity</b>: one decoder is trained on all "
    "channels and a region is destroyed at test time, so a region scores highly only if the rest "
    "of the brain cannot compensate. This section is the complement &mdash; the co-trained "
    "decoder is <b>trained on only that region's channels</b> and tested on it. "
    "<b style='color:#b00'>This pass ran with --suff-null-draws 0, so there is NO matched-N "
    "null and no size control.</b> Raw ROI-only accuracy rises with electrode count, so a "
    "cross-region ranking on this page is substantially an implant-coverage ranking. What it "
    "does support is the <i>same</i> region compared across configurations (history length, "
    "ROI scope, balance), where the channel set is identical on both sides and the size "
    "confound cancels. <code>suff_delta_*</code>, <code>suff_null_*</code> and "
    "<code>suff_p_*</code> are NaN by construction, so the &Delta; panel and the only "
    "significance test in this section are both absent. "
    "<b>Kernel width is still fixed across regions</b> at the whole-brain "
    "&gamma;&nbsp;=&nbsp;1/n_features, so the knockout model is unchanged. "
    "<b>Caveats.</b> Regions inherit the <b>whole-brain</b> per-task peak bin, so a region "
    "peaking elsewhere is evaluated off-peak and understated. Below ~5 channels the kernel is "
    "numerically degenerate and absolute <code>suff_*</code> values are uninterpretable "
    "&mdash; and here there is no &Delta; to fall back on. These values must never be placed "
    "on the same axis as the knockout &Delta;acc: one is an accuracy, the other a change in "
    "accuracy.</div>")


def section_suff(df, rcol, dfs_for_lims) -> str:
    """ROI-sufficiency panels. Absent unless --roi-sufficiency produced the columns.

    The columns exist but are all-NaN under ``--suff-null-draws 0``; the heading and note
    switch with them so the page never claims a matched-N control it did not compute."""
    if "suff_delta_pic" not in df.columns:
        return ""
    has_null = bool(np.isfinite(pd.to_numeric(df["suff_delta_pic"],
                                              errors="coerce")).any())
    heading = ("Region sufficiency &middot; ROI-only decoder vs matched-N null" if has_null
               else "Region sufficiency &middot; ROI-only decoder, raw accuracy "
                    "(no size control)")
    return section_measures(df, rcol, dfs_for_lims, MEASURES_SUFF, heading,
                            _SUFF_NOTE if has_null else _SUFF_NOTE_NO_NULL)


def section_measures(df, rcol, dfs_for_lims, measures=MEASURES_KNOCKOUT_PC,
                     heading="Region knockout &middot; per electrode",
                     note=_KNOCKOUT_NOTE) -> str:
    """Gallery of aggregated pic-vs-aud scatters, one per knockout measure, each
    with its own shared equal-scale range (pooled over `dfs_for_lims` so the atlas
    arms share a scale per measure). Skips measures whose columns are absent or
    all-NaN."""
    blocks = [note]
    for m in measures:
        if m["xcol"] not in df.columns or m["ycol"] not in df.columns:
            continue
        if not (np.isfinite(df[m["xcol"]]).any() and np.isfinite(df[m["ycol"]]).any()):
            continue
        # A measure may ask to be framed from chance rather than from 0 (raw accuracies).
        # If the shuffled-null caches are missing, fall back to the zero anchor rather than
        # inventing a chance value.
        band = _chance_band(df) if m.get("anchor") == "chance" else None
        chance = (band[0][0], band[1][0]) if band else None
        # Anchor on the chance LINE, not the band's lower edge. A +/-1 SD pooled-null band
        # is far wider than the regions it frames (auditory spans 0.093-0.235 against data
        # in 0.17-0.22), so anchoring on its edge stretched the axis and squashed every
        # marker into the middle. The band still draws, clipped by the axis.
        lims = _shared_limits(dfs_for_lims, m["xcol"], m["ycol"],
                              anchor=min(chance) if chance else 0.0)
        agg = _aggregated_scatter(df, rcol, lims, xcol=m["xcol"], ycol=m["ycol"],
                                  axis=m["axis"], chance=chance, band=band,
                                  title="{} — colour = region".format(m["name"]))
        blocks.append("<details class='meas' open><summary>{}</summary>"
                      "<p class='subtle'>{}</p>{}</details>".format(
                          m["name"], m["blurb"], agg))
    return "<h2>{}</h2>".format(heading) + "".join(blocks)


def section_ranked(df, rcol) -> str:
    """The Jacobian as a cross-participant ROI ranking. Covariance used to live here
    too but is back on a pic-vs-aud scatter (`section_cov`) — it is model-free, so
    unlike the Jacobian its task asymmetry is a property of the data, not of a shared
    decoder map, and is worth being able to see."""
    blocks = [_RANKED_NOTE]
    for (_key, xp, _ya, name, axis) in _STD_SPECS:
        base = xp.rsplit("_", 1)[0]                     # "<base>_pic" -> "<base>"
        img = _roi_ranked_strip(df, rcol, base, axis, name)
        if not img:
            continue
        blocks.append("<details class='meas' open><summary>{}</summary>{}</details>".format(
            name, img))
    if len(blocks) == 1:
        return ""
    return "<h2>Jacobian sensitivity &middot; cross-participant ROI ranking</h2>" + "".join(blocks)


def section_cov(df, rcol, dfs_for_lims) -> str:
    """Neural-GloVe covariance as a picture-vs-auditory scatter (the original style)."""
    return section_measures(df, rcol, dfs_for_lims, MEASURES_COV,
                            "Neural&ndash;GloVe covariance &middot; per electrode",
                            _COV_NOTE)


def section_part(doc, df, rcol, heading, subtitle, slug, dfs_for_lims) -> str:
    """One atlas arm: overview + the measure sections, in order, each individually
    foldable and TOC'd under this part.

    The arms are the same five views on two parcellations, so they are one code path
    rather than a duplicated assembly. `slug` (the atlas name) namespaces the child
    section ids so the parts do not collide.

    The part's own TOC entry is recorded HERE, before its children are built —
    `doc.fold` registers on return, so folding this part in the caller would file
    the parent after its own children."""
    doc.add_toc_entry("s-" + slug, heading)
    sid = lambda s: "s-{}-{}".format(slug, s)
    return (
        "<h1>{}</h1><p class='subtle'>{}</p>".format(heading, subtitle)
        + doc.fold(section_overview(df), sid("overview"), open=True, sub=True)
        + doc.fold(section_measures(df, rcol, dfs_for_lims), sid("knockout"), sub=True)
        + doc.fold(section_ranked(df, rcol), sid("ranked"), sub=True)
        + doc.fold(section_cov(df, rcol, dfs_for_lims), sid("cov"), sub=True)
        + doc.fold(section_solo(df, rcol, dfs_for_lims), sid("solo"), sub=True)
        + doc.fold(section_suff(df, rcol, dfs_for_lims), sid("suff"), sub=True)
    )


def section_caveats() -> str:
    return (
        "<h2>Interpretation &amp; caveats</h2>"
        "<div class='qbox'>"
        "<b>Scores in the CSV are region totals</b> (summed over the region's channels) and "
        "are size-confounded for both magnitude measures (&rho; with channel count 0.99 "
        "Jacobian / 0.96 covariance; only the knockouts are size-robust at 0.19). "
        "The report therefore plots only normalized views; if you go back to the CSV, "
        "normalize first. "
        "<b>The pic&nbsp;=&nbsp;aud diagonal is not amodality.</b> One co-trained model scores "
        "both tasks, so the Jacobian ranks ROIs near-identically by construction "
        "(&rho;&nbsp;= +0.99 per electrode). Covariance <i>is</i> a separate per-task quantity, "
        "but its raw agreement was the size artifact &mdash; per electrode it is &minus;0.09. "
        "That is why neither gets a picture-vs-auditory plane. Use the two knockouts or the "
        "<code>_solo</code> single-modality decoders for any task-specificity claim. "
        "<b>ROI size and ROI identity are collinear</b> by implant design, so the "
        "cross-participant ranking retains &rho;&nbsp;&asymp;&nbsp;&minus;0.75 with ROI channel "
        "count and no ROI clears a BH-corrected group test. Read the rankings as descriptive. "
        "<b>Read auditory against its ceiling</b>: the pooled model decodes auditory only "
        "slightly above chance, so the whole-brain auditory ceiling is small — a region can "
        "hold a large <i>share</i> (frac WB aud) while its absolute &#916;acc looks like "
        "noise. <b>The auditory ceiling is not significant in any participant</b> "
        "(<code>wb_p_aud</code> 0.23&ndash;0.42), so <code>frac_wb_aud</code> divides by a "
        "denominator indistinguishable from zero &mdash; which is why it returns values like "
        "0.94 and &minus;0.57. Do not quote it without quoting <code>wb_p_aud</code>. "
        "<b><code>frac_wb</code> and the size-fair gallery are different normalizations</b>: "
        "<code>frac_wb</code> divides by the <i>subadditive</i> whole-brain knockout, the "
        "<code>_std</code> enrichment by an <i>additive</i> whole-brain sum. They will not agree "
        "and are not interchangeable. "
        "<b>Significance is conservative</b>: under the Nystroem-RBF dilution even whole-region "
        "knockout rarely clears BH-FDR (52/53 regions land in <span class='ns'>neither</span>), "
        "so the region <i>ranking</i> and <i>ceiling share</i> carry the signal rather than "
        "per-region certification. "
        "<b>Peak-bin selection is not nested</b>: the peak bin is the argmax of a CV accuracy "
        "curve over all of that patient's trials, which are then re-split into the importance "
        "bootstraps. Selection is over <i>time</i>, not channels, so the ROI ranking is largely "
        "protected, but <code>wb_imp</code> (and hence <code>frac_wb</code>) is optimistically "
        "biased. The peak sits on a plateau (AA picture bins 16&ndash;21 span 0.384&ndash;0.399; "
        "bin 20 beats bin 18 by 0.0003), so the argmax is arbitrary within it. "
        "Single-channel attribution is deliberately not reported (retired 2026-07-20); the "
        "retrieval-aligned Jacobian and plain-PLS VIP were retired 2026-07-23. CSVs written "
        "before that date still carry dead <code>jac_dir_*</code> and <code>vip*</code> columns; "
        "this report ignores them."
        "</div>")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _load_atlas(in_dir: Path, atlas: str, metric: str):
    """The region table for *atlas* and *metric*, or None if that arm has not been run."""
    csv = in_dir / "region_importance_{}_all.csv".format(atlas)
    if not csv.exists():
        return None
    m = pd.read_csv(csv)
    m = m[m["metric"] == metric].copy()
    return m if not m.empty else None


_ATLAS_SUBTITLE = {
    "nmm": (
        "Neuromorphometrics parcellation, volumetric and in each participant's native "
        "space, with anterior/posterior halves split at that participant's own parcel "
        "centroid. Labels every contact including subcortical ones, so a contact outside "
        "the temporal-parietal whitelist is NAMED rather than snapped to the nearest "
        "cortex. Right-hemisphere regions carry a <code>Right&nbsp;</code> prefix and are "
        "therefore outside the vocabulary by construction. "
        "Source <code>region_importance_nmm_all.csv</code>."),
    "dk": (
        "Desikan-Killiany parcellation on the fsaverage surface, with anterior/posterior "
        "halves split at ONE cohort-wide plane rather than per participant. Surface-only: "
        "it has no subcortical parcels at all, so a contact NMM calls hippocampus gets "
        "whatever cortex is nearest &mdash; and it carries no hemisphere prefix, which is "
        "why right-hemisphere contacts must be excluded by name upstream. Has "
        "<code>pSTS</code>, which NMM cannot express. "
        "Source <code>region_importance_dk_all.csv</code>."),
}

_ATLAS_CAVEAT = (
    "<div class='box'><b>Reading the two parts against each other.</b>&nbsp; "
    "The parts are <b>not</b> two labellings of one analysis. Each atlas gates channel "
    "selection as well as grouping, so the two arms are trained on <b>different channel "
    "sets</b> &mdash; 643 vs 702 contacts cohort-wide before artifact rejection, agreeing "
    "on only 627. A region present in both parts therefore does not contain the same "
    "electrodes in both. The axis limits are shared so the panels can be laid side by "
    "side, which means one arm's scale is partly set by the other's data; that is a "
    "presentation choice, not a claim that the numbers are paired. "
    "Treat a result that holds under both as robust to the parcellation, and a result "
    "that holds under only one as a finding about that parcellation.</div>")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="HTML report for cross_task_region_importance results")
    ap.add_argument("--balance", default="none",
                    choices=["none", "downsample", "upsample"],
                    help="Which resampling setting to report on. Resolves --in-dir to "
                         "<results>/cross_task_cotrain/balance_<BALANCE>/, matching the "
                         "analysis module's output layout (default: none).")
    ap.add_argument("--aggregate", default="median", choices=["median", "mean"],
                    help="How the ringed cross-participant markers, the heatmap and the "
                         "ROI ranking aggregate over participants. The aggregate is "
                         "UNWEIGHTED either way -- a participant with 3 contacts in an "
                         "ROI counts as much as one with 20; marker size encodes the "
                         "number of contributing participants, not their electrodes. "
                         "'median' (default) is robust to one participant's outlier; "
                         "'mean' is not, and on a right-skewed quantity over 9 "
                         "participants a single extreme value moves it visibly -- which "
                         "is the reason to look at both. Writes "
                         "region_importance_report_mean.html so the median page survives.")
    ap.add_argument("--participants", default="all", choices=["all", "significant"],
                    help="Which participants the page covers. 'significant' keeps only "
                         "those with at least one significant category-independent time "
                         "bin in BOTH tasks, read from the semantic_regression figure's "
                         "shipped source_data.csv -- which is a DIFFERENT configuration "
                         "from any cross-task arm (tp/h5 picture, tpfm/h10 auditory), and "
                         "the page says so. At the 2026-08-13 cohort it drops AZ and DR "
                         "(9 -> 7); 'either task' would drop nobody, which is why the rule "
                         "is 'both'.")
    ap.add_argument("--batch", action="store_true",
                    help="Write all four standard pages into --in-dir: "
                         "{all, significant} x {median, mean}. This is the normal way to "
                         "regenerate an arm; --aggregate/--participants are ignored and "
                         "--out is ignored (each page needs its own filename).")
    ap.add_argument("--in-dir", default=None,
                    help="Directory containing region_importance_all.csv. Overrides "
                         "--balance.")
    ap.add_argument("--out", default=None,
                    help="Output HTML path (default: <in-dir>/region_importance_report.html)")
    ap.add_argument("--metric", default="cat_indep_bal_acc", choices=list(METRIC_SLUG),
                    help="Metric to report (default: cat_indep_bal_acc)")
    ap.add_argument("--atlas", nargs="+", default=[ROI_ATLAS_DEFAULT, "dk"],
                    choices=["nmm", "dk"],
                    help="Atlas arm(s) to include, in order; each becomes one Part. An "
                         "arm with no CSV is skipped with a note, so the report renders "
                         "from whichever arms have actually been run (default: nmm dk).")
    args = ap.parse_args()
    # dict.fromkeys: preserve the order given, drop a repeat.
    args.atlas = list(dict.fromkeys(args.atlas))

    in_dir = (Path(args.in_dir) if args.in_dir
              else DEFAULT_IN_DIR / "balance_{}".format(args.balance))

    # The four pages are the standard set (2026-08-13): {all, significant} x {median, mean}.
    # Each combination gets its own filename so none can overwrite another -- they are the
    # same numbers under different readings and are meant to be compared, not replaced.
    combos = ([(a, c) for c in ("all", "significant") for a in ("median", "mean")]
              if args.batch else [(args.aggregate, args.participants)])
    if args.batch and args.out:
        print("NOTE: --out is ignored under --batch; four files are written into", in_dir)

    rc = 0
    for aggregate, cohort in combos:
        out_path = (Path(args.out) if (args.out and not args.batch)
                    else in_dir / _report_name(aggregate, cohort))
        rc |= _one_report(args, in_dir, out_path, aggregate, cohort)
    return rc


def _report_name(aggregate, cohort) -> str:
    """Filename for one (aggregator, cohort) combination.

    ``median`` + ``all`` keeps the historical bare name so existing links and the
    figure pipeline's expectations do not break; every other combination is suffixed.
    """
    stem = "region_importance_report"
    if cohort != "all":
        stem += "_significant"
    if aggregate != "median":
        stem += "_" + aggregate
    return stem + ".html"


def _one_report(args, in_dir, out_path, aggregate, cohort) -> int:
    """Render one page. `aggregate` is median|mean, `cohort` is all|significant."""
    # Set before any figure is built: every aggregation site reads the module global, so
    # assigning it here is what keeps one page from mixing median and mean markers.
    global AGG
    AGG = aggregate

    # One arm per atlas. Either may be absent: the NMM pass runs first, and the report
    # must render from it alone rather than half-filling a two-part layout.
    arms = [(a, _load_atlas(in_dir, a, args.metric)) for a in args.atlas]
    arms = [(a, d) for a, d in arms if d is not None and not d.empty]
    if not arms:
        wanted = ", ".join("region_importance_{}_all.csv".format(a) for a in args.atlas)
        print("ERROR: none of these exist (or none has rows for metric '{}'): {}"
              .format(args.metric, wanted))
        print("       in:", in_dir)
        if args.in_dir is None:
            print("       (resolved from --balance {}; pass --in-dir to override)"
                  .format(args.balance))
        return 1

    # Participant filter BEFORE any derived column: _add_standardized builds a
    # per-participant reference and _shared_limits pools across arms, so filtering
    # afterwards would leave both computed over participants the page does not show.
    sig_note = ""
    if cohort == "significant":
        keep = significant_participants()
        if keep is None:
            print("ERROR: cannot build the significant-participant page --",
                  _SIG_SOURCE, "is missing.")
            return 1
        dropped = sorted(set(arms[0][1]["patient"]) - set(keep))
        arms = [(a, d[d["patient"].isin(keep)].copy()) for a, d in arms]
        arms = [(a, d) for a, d in arms if not d.empty]
        if not arms:
            print("ERROR: no participants left after the significance filter.")
            return 1
        sig_note = (
            "<div class='box'><b>Restricted cohort.</b>&nbsp; This page shows only "
            "participants with at least one <b>significant time bin</b> for "
            "category-independent accuracy in <b>both</b> tasks"
            "{dropped}. The 'both' rule is used because it is the only one that filters "
            "anything at this cohort &mdash; 'either task' and 'picture alone' both select "
            "every participant. "
            "<b>The significance comes from a different configuration than this page.</b> "
            "It is read from <code>figures_for_paper/semantic_regression/source_data/"
            "source_data.csv</code>, whose picture arm is <code>tp</code>/h5 and auditory arm "
            "<code>tpfm</code>/h10, while this report is built on its own run pair. Treat it "
            "as \"participants whose semantic decoding was significant in the shipped "
            "time-course figure\", not as a statement about this arm's runs.</div>")
        sig_note = sig_note.format(
            dropped=(", dropping " + ", ".join(dropped)) if dropped else "")

    df = arms[0][1]
    patients = sorted(df["patient"].unique())
    for _, d in arms:
        _add_per_channel(d)       # <col>_pc    (sections 1, 2, 5)
        _add_standardized(d)      # <col>_std   (sections 3, 4)
        _add_size_detrended(d)    # suff_resid_* (section 6)
    # Knockout scatters share one equal-scale range across the arms (computed inside
    # section_measures) so the panels can be read side by side.
    dfs_for_lims = [d for _, d in arms]
    print("Atlas arms: {} | patients: {} | metric: {}".format(
        " | ".join("{}={} regions".format(a, d["region"].nunique()) for a, d in arms),
        ", ".join(patients), args.metric))
    missing = [a for a in args.atlas if a not in {x for x, _ in arms}]
    if missing:
        print("NOTE: no data for atlas arm(s) {} -- rendering a single-atlas report."
              .format(", ".join(missing)))

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    method = (
        "<div class='box'><b>Method.</b>&nbsp; A single <b>kernel-PLS</b> model "
        "(Nystroem-RBF + PLSRegression &rarr; GloVe) is trained on pooled picture- and "
        "auditory-naming trials per participant (same model as "
        "<code>cross_task_cotrain.py</code>). Importance is assessed at the level of brain "
        "<b>regions</b> (<code>nmm_roi</code> / <code>dk_roi</code>), on held-out test trials over bootstraps, "
        "by <b>four necessity measures</b>, everywhere <b>per electrode</b>: "
        "<b>(1) &#916;category-accuracy knockout</b> (with a per-bootstrap label-shuffle null "
        "&rarr; BH-FDR groups <span class='sig'>both / picture_only / auditory_only</span> / "
        "<span class='ns'>neither</span>); <b>(2) &#916;cosine-to-GloVe knockout</b>; "
        "<b>(3) Jacobian sensitivity</b> &#8214;&#8706;&#375;/&#8706;x&#8214;; and "
        "<b>(4) neural&ndash;GloVe covariance</b> (model-free). They run from the end task "
        "toward the decoder's own covariance objective. Each region is read against the "
        "<b>whole-brain ceiling</b> (&#916;acc when all channels are knocked out); "
        "<code>frac_wb_*</code> is its share. A <b>fifth</b> section compares the co-trained "
        "model against picture-only and auditory-only decoders. A <b>sixth</b> section, present "
        "only when the run was given <code>--roi-sufficiency</code>, inverts the question "
        "entirely: measures 1&ndash;5 all ask what <b>breaks when a region is removed</b> "
        "(necessity), while it asks what a region <b>can do alone</b> (sufficiency), by training "
        "the decoder on that region's channels only and comparing it against same-size random "
        "channel sets. A region redundant with another scores ~0 on knockout yet can decode well "
        "by itself, so the two readings are complementary."
        "<br><br><b>How this page is organised.</b> One part per <b>atlas</b> &mdash; NMM "
        "(volumetric, native-space) and DK (surface, fsaverage) &mdash; each carrying the same "
        "sections, so you can read the same question under both parcellations. They are peers: "
        "neither is the reference the other is scored against. Measures 1, 2 and 4 are drawn "
        "picture-vs-auditory. Measure 3 (the "
        "<b>Jacobian</b>) is drawn as a cross-participant ROI <i>ranking</i> instead: it reads a "
        "single co-trained model that scores both tasks through one shared map, so its pic-vs-aud "
        "plane has no interpretable off-diagonal (&rho;&nbsp;= +0.99 per electrode, structural). "
        "Covariance keeps its scatter because it involves no model at all &mdash; it is computed "
        "separately on each task's own trials, so an asymmetry there is a property of the data."
        "<br><br><b>Changed after an external audit (2026-07-23).</b> "
        "(a) <b>Region totals are no longer shown</b> &mdash; within participant they correlate "
        "with ROI channel count at &rho;&nbsp;=&nbsp;0.96&ndash;0.99 for both magnitude measures, "
        "so they read the implant rather than the brain. Everything below is normalized. "
        "(b) The <b>retrieval-aligned Jacobian</b> |&#8706;(&#375;&middot;&#251;)/&#8706;x| was "
        "<b>retired</b>: a constant rescaling of measure&nbsp;3 (ratio CV 0.8&ndash;6.7 % within "
        "participant, &rho;&nbsp;=&nbsp;0.99), because every per-feature gradient factors through "
        "the same rank-&le;&nbsp;10 PLS map, leaving the projection onto the correct-answer "
        "direction a per-trial constant with no channel index. "
        "(c) <b>Plain-PLS VIP was retired</b> &mdash; it attributed a linear surrogate the paper "
        "does not report (there is no well-defined input-space VIP under the Nystroem map), and "
        "as a region total it was an electrode-count proxy (&rho;&nbsp;=&nbsp;0.98). "
        "<b>The pic&nbsp;=&nbsp;aud diagonal is never amodality evidence.</b> Task specificity "
        "lives in the two knockouts and in the <b>single-modality</b> section, the only place two "
        "independently trained decoders are compared."
        "</div>").format(metric=args.metric)

    # the balance setting goes in the <title> and the header: the two settings produce
    # otherwise-identical-looking reports, and confusing them is easy
    bal = in_dir.name if in_dir.name.startswith("balance_") else args.balance
    # Cohort and aggregator go in the <title> as well as the header: four pages are
    # generated per arm and they look identical at a glance, which is exactly how one
    # gets quoted for another.
    cohort_label = "all participants" if cohort == "all" else "significant participants"
    doc = Document(
        "ROI importance ({}, {}, {}) — cross-task".format(bal, cohort_label, AGG),
        "{npat} participants ({cohort}) &bull; metric <code>{metric}</code> &bull; trial "
        "resampling <b><code>{bal}</code></b> &bull; cross-participant marker "
        "<b>{agg}</b> &bull; atlas <b>{atlases}</b> &bull; source "
        "<code>{src}/</code>".format(
            npat=len(patients), cohort=cohort_label, metric=args.metric, bal=bal,
            agg=AGG, atlases=" + ".join(a.upper() for a, _ in arms), src=in_dir.name))

    # One part per atlas arm, same sections in each. The colour map is built ONCE, over
    # the union of both arms' regions, so a region is the same colour in both panels --
    # colouring per part is what made the same region change colour between them.
    all_regions = set()
    for _, d in arms:
        all_regions |= set(d["region"].astype(str))
    rcol = _region_colors(all_regions)
    # Emitted only when the run's scope reaches past the vendored 13, so a `tp` report is
    # byte-for-byte what it was before this existed.
    palette_note = _palette_note(rcol)

    parts = ""
    for i, (atlas, d) in enumerate(arms):
        parts += doc.fold(
            section_part(doc, d, rcol,
                         "Part {} &mdash; {} ({} regions)".format(
                             i + 1, atlas.upper(), d["region"].nunique()),
                         _ATLAS_SUBTITLE.get(atlas, ""), atlas, dfs_for_lims),
            "s-{}".format(atlas), open=(i == 0), in_toc=False)
    if len(arms) > 1:
        parts = _ATLAS_CAVEAT + parts
    doc.add_html(method + sig_note + palette_note + parts)
    doc.add_section(section_caveats(), "s-caveats")

    out_path.write_text(doc.render(generated=generated), encoding="utf-8")
    print("Wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
