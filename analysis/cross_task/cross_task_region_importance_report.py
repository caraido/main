# cross_task_region_importance_report.py
# HTML report from cross_task_region_importance.py output (region_importance_all.csv).
#
# Region-only successor to the retired per-channel report
# (_archive/cross_task_reports/cross_task_channel_importance_report.py).
#
# Structure: TWO PARTS — fine ROIs and coarse (merged) ROIs — each carrying the same
# five sections, so the same question can be read at two granularities:
#   1. Δ category accuracy   (region knockout), per electrode   — pic-vs-aud scatter
#   2. Δ cosine to GloVe     (region knockout), per electrode   — pic-vs-aud scatter
#   3. Jacobian sensitivity, per electrode                      — cross-participant ROI ranking
#   4. Neural–GloVe covariance, per electrode                   — pic-vs-aud scatter
#   5. Co-trained vs single-modality decoders (3 panels + ROI × decoder heatmap)
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
#   region_importance_all.csv          (required: fine ROIs — permutation Δacc/Δcosine +
#                                        Jacobian + covariance + whole-brain ceiling)
#   region_importance_merged_all.csv   (optional: from --merge-regions; supplies Part 2.
#                                        A full recompute on the coarser grouping, not a
#                                        sum of the fine one — knockout Δ is not additive)
#
# Output (default): <in-dir>/region_importance_report.html
#
# Usage:
#   python -m analysis.cross_task.cross_task_region_importance_report
#   python -m analysis.cross_task.cross_task_region_importance_report --metric cat_indep_bal_acc
#   python -m analysis.cross_task.cross_task_region_importance_report --in-dir <dir> --out <out.html>

from __future__ import annotations

import argparse
import base64
import io
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAIN_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
DEFAULT_IN_DIR = Path(_MAIN_DIR) / "results" / "cross_task_cotrain"

METRIC_SLUG = {
    "cat_indep_bal_acc": "catindep",
    "word_bal_acc": "word",
    "cosine_mean": "cosine",
}

_GCOL = {"both": "#2ca02c", "picture_only": "#1f77b4",
         "auditory_only": "#d62728", "neither": "#bbbbbb"}

# Patient marker glyphs for the aggregated scatter (one per participant).
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "p", "h"]

CSS = """<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif;
       max-width: 1280px; margin: 28px auto; padding: 0 20px; color: #1a1a1a; line-height: 1.45; }
h1 { color: #1565C0; border-bottom: 2px solid #1565C0; padding-bottom: 8px; }
h2 { color: #0D47A1; margin-top: 36px; border-bottom: 1px solid #BBDEFB; padding-bottom: 4px; }
h3 { color: #424242; margin-top: 22px; }
table.results { border-collapse: collapse; margin: 10px 0; font-size: 12px; width: auto; }
table.results th, table.results td { border: 1px solid #ccc; padding: 5px 9px; text-align: right; }
table.results th { background: #ECEFF1; font-weight: 600; text-align: center; }
table.results td.text { text-align: left; }
table.results td.top1 { background: #E8F5E9; font-weight: 700; }
table.results td.top2 { background: #F1F8E9; }
table.results td.neg  { color: #9E9E9E; }
.subtle { color: #757575; font-size: 12px; }
img { max-width: 100%; border: 1px solid #e0e0e0; padding: 4px; background: white; margin: 6px 0; }
.box  { background: #F5F7FA; padding: 10px 14px; border-left: 3px solid #1565C0; margin: 12px 0; font-size: 13px; }
.qbox { background: #FFF8E1; padding: 10px 14px; border-left: 3px solid #F9A825; margin: 12px 0; font-size: 13px; }
.sig  { color: #2E7D32; font-weight: 600; }
.ns   { color: #9E9E9E; }
.pat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 12px 0; }
@media (max-width: 900px) { .pat-grid { grid-template-columns: 1fr; } }
/* collapsible sections */
details.sec { border: 1px solid #D6E2F0; border-radius: 8px; margin: 14px 0; background: #FBFDFF; }
details.sec > summary { cursor: pointer; list-style: none; padding: 10px 16px; font-size: 1.15rem;
    font-weight: 600; color: #0D47A1; user-select: none; }
details.sec > summary::-webkit-details-marker { display: none; }
details.sec > summary::before { content: "▸ "; color: #5C9BD6; font-size: .9em; }
details.sec[open] > summary::before { content: "▾ "; }
details.sec[open] > summary { border-bottom: 1px solid #E3EDF7; }
details.sec > :not(summary) { margin-left: 16px; margin-right: 16px; }
details.sec > summary + * { margin-top: 12px; }
details.meas { border-left: 2px solid #E3EDF7; margin: 10px 0 10px 4px; padding-left: 10px; }
details.meas > summary { cursor: pointer; list-style: none; color: #424242; font-weight: 600;
    font-size: 1rem; padding: 4px 0; user-select: none; }
details.meas > summary::-webkit-details-marker { display: none; }
details.meas > summary::before { content: "▸ "; color: #9E9E9E; }
details.meas[open] > summary::before { content: "▾ "; }
details.sec > summary h2, details.sec > summary h1 { display: inline; border: none; margin: 0; padding: 0; }
/* table of contents + toolbar */
nav.toc { background: #F5F7FA; border: 1px solid #D6E2F0; border-radius: 8px; padding: 12px 18px;
    margin: 18px 0 26px; }
nav.toc .toc-title { font-weight: 700; color: #0D47A1; font-size: .95rem; letter-spacing: .02em; }
nav.toc ul { columns: 2; column-gap: 28px; margin: 8px 0 0; padding-left: 18px; }
@media (max-width: 700px) { nav.toc ul { columns: 1; } }
nav.toc li { margin: 3px 0; }
nav.toc li.sub { margin-left: 14px; list-style: circle; }
nav.toc li.sub a { font-size: 12px; color: #3D6FB5; }
nav.toc a { color: #1565C0; text-decoration: none; font-size: 13px; }
nav.toc a:hover { text-decoration: underline; }
.toolbar { margin: 8px 0 2px; display: flex; gap: 8px; }
.toolbar button { font: inherit; font-size: 12px; padding: 4px 12px; border: 1px solid #BBD3EC;
    background: #fff; color: #1565C0; border-radius: 6px; cursor: pointer; }
.toolbar button:hover { background: #EAF2FB; }
</style>"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

import re as _re

_TOC = []   # (id, title) collected as sections are folded, for the table of contents


def _fold(html, sid, open=False, add_toc=True, sub=False):
    """Wrap a section (whose HTML starts with <h1>/<h2>Title</...>) in a collapsible
    <details>, using the heading as the summary. Records (id, title, sub) for the
    TOC; `sub=True` marks it as nested under the preceding part (indented in the
    TOC). Folds nest fine — the TOC click handler opens every ancestor <details>."""
    m = _re.match(r"\s*<(h[12])>(.*?)</\1>(.*)", html, _re.S)
    if not m:
        return html
    _tag, title, rest = m.group(1), m.group(2), m.group(3)
    if add_toc:
        _TOC.append((sid, _re.sub(r"<[^>]+>", "", title), sub))
    return '<details id="{}" class="sec"{}><summary>{}</summary>{}</details>'.format(
        sid, " open" if open else "", title, rest)


def _toc_html():
    items = "".join(
        '<li{cls}><a href="#{sid}">{t}</a></li>'.format(
            cls=' class="sub"' if sub else "", sid=sid, t=title)
        for sid, title, sub in _TOC)
    return ('<nav class="toc"><div class="toc-title">Contents</div>'
            '<div class="toolbar"><button type="button" onclick="setAll(true)">Expand all</button>'
            '<button type="button" onclick="setAll(false)">Collapse all</button></div>'
            '<ul>{}</ul></nav>'.format(items))


_TOC_SCRIPT = (
    "<script>"
    "function setAll(o){document.querySelectorAll('details').forEach(function(d){d.open=o;});}"
    "document.querySelectorAll('nav.toc a').forEach(function(a){a.addEventListener('click',function(){"
    "var el=document.querySelector(a.getAttribute('href'));var p=el;"
    "while(p){if(p.tagName==='DETAILS')p.open=true;p=p.parentElement;}});});"
    "</script>")


def _fig_to_img(fig, alt: str) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return '<img alt="{}" src="data:image/png;base64,{}" />'.format(alt, b64)


# High-contrast qualitative palette (saturated, well-separated hues), front-loaded
# with the most distinct colours so different ROIs read as clearly different.
_DISTINCT = [
    "#e6194B",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#f032e6",  # magenta
    "#008080",  # teal
    "#9A6324",  # brown
    "#42d4f4",  # cyan
    "#bfef45",  # lime
    "#800000",  # maroon
    "#808000",  # olive
    "#000075",  # navy
    "#ff69b4",  # hot pink
    "#000000",  # black
    "#ffe119",  # yellow
    "#a9a9a9",  # grey
    "#dcbeff",  # lavender
    "#aaffc3",  # mint
    "#00ced1",  # dark turquoise
]


def _region_colors(regions) -> dict:
    """Deterministic, dataset-wide region -> color map (curated high-contrast
    palette) so a region reads as the same distinct colour in every patient's
    scatter and in the aggregated scatter."""
    regs = sorted(set(str(r) for r in regions))
    return {r: _DISTINCT[i % len(_DISTINCT)] for i, r in enumerate(regs)}


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
    "Ringed markers are the median across participants (size &prop; n). "
    "<b>Covariance is omitted</b> (model-free &mdash; there is no decoder to train one per task). "
    "<b>Caveat:</b> "
    "the auditory-only decoder is underpowered where the auditory task has few trials/repeats (AA, DR ~1 "
    "trial/word) &mdash; read its points as noisy.</div>")


def _heatmap(df, measure, out_title):
    """ROI × condition heatmap for one measure: rows = ROIs (median across
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
    med = df.groupby("region")[cols].median()
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
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="median (min–max scaled)")
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


def _shared_limits(dfs, xcol, ycol, margin=0.08):
    """One (lo, hi) range shared by BOTH axes across the scatters of a measure,
    framed to the per-ROI MEDIANS across participants (the emphasized markers), so
    the robust view reads clearly and a single participant's extreme faded point may
    fall outside rather than squishing everything. Equal range → 45° diagonal."""
    if not dfs:
        return -0.01, 0.01
    meds = []
    for d in dfs:
        if xcol in d and ycol in d:
            dd = d[np.isfinite(d[xcol]) & np.isfinite(d[ycol])]
            m = dd.groupby("region").agg(x=(xcol, "median"), y=(ycol, "median"))
            meds.append(m["x"].values); meds.append(m["y"].values)
    vals = np.concatenate(meds) if meds else np.array([])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -0.01, 0.01
    lo = min(float(vals.min()), 0.0)
    hi = max(float(vals.max()), 0.0)
    span = (hi - lo) or 0.02
    pad = span * margin
    return lo - pad, hi + pad


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def _aggregated_scatter(df, rcol, lims,
                        xcol="perm_imp_pic", ycol="perm_imp_aud",
                        axis="Δ cat-indep accuracy",
                        title="All regions, all participants — colour = region, marker = participant",
                        xlabel=None, ylabel=None) -> str:
    """All patients' regions on one (x, y) plane. Colour = region (shared across
    subjects), marker = patient. Two legends (region colour, patient marker).
    `lims=(lo,hi)` is the shared equal-scale range applied to both axes."""
    lo, hi = lims
    patients = sorted(df["patient"].unique())
    pmark = {p: _MARKERS[i % len(_MARKERS)] for i, p in enumerate(patients)}
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    # individual participant × region points — kept but faded (context, not headline)
    for _, r in df.iterrows():
        if not (np.isfinite(r[xcol]) and np.isfinite(r[ycol])):
            continue
        reg, pat = str(r["region"]), r["patient"]
        ax.scatter(r[xcol], r[ycol], s=40,
                   color=rcol.get(reg, "#777"), marker=pmark[pat],
                   edgecolors="none", alpha=0.4, zorder=2)
    # per-ROI MEDIAN across participants — robust to any one patient's outliers;
    # marker size ∝ number of contributing participants (bigger = better sampled).
    med = (df[np.isfinite(df[xcol]) & np.isfinite(df[ycol])]
           .groupby("region").agg(x=(xcol, "median"), y=(ycol, "median"),
                                   n=("patient", "nunique")).reset_index())
    for _, r in med.iterrows():
        reg = str(r["region"])
        ax.scatter(r["x"], r["y"], s=70 + 34 * r["n"], color=rcol.get(reg, "#777"),
                   edgecolors="#111", linewidths=1.8, alpha=0.98, zorder=6)
        ax.annotate(reg, (r["x"], r["y"]), fontsize=7, fontweight="bold",
                    xytext=(5, 4), textcoords="offset points", zorder=7)
    ax.plot([lo, hi], [lo, hi], ls=":", color="#999", lw=0.8, zorder=1,
            label="_pic = aud")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel if xlabel else "{} — picture".format(axis))
    ax.set_ylabel(ylabel if ylabel else "{} — auditory".format(axis))
    ax.set_title(title + "\n(ringed = median across participants, size ∝ n; faded = individual)",
                 fontsize=9)
    # region colour legend (only regions actually present)
    from matplotlib.lines import Line2D
    regs_present = sorted(set(str(r) for r in df["region"]))
    reg_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=rcol[r],
                          markeredgecolor="#333", markersize=8, label=r)
                   for r in regs_present]
    pat_handles = [Line2D([0], [0], marker=pmark[p], color="w", markerfacecolor="#888",
                          markeredgecolor="#333", markersize=8, label=p)
                   for p in patients]
    leg1 = ax.legend(handles=reg_handles, title="region", fontsize=7,
                     loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
                     ncol=1)
    ax.add_artist(leg1)
    ax.legend(handles=pat_handles, title="participant", fontsize=7,
              loc="lower left", bbox_to_anchor=(1.01, 0.0), frameon=False)
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

    Conventions match `_aggregated_scatter`: colour = region, marker = participant,
    ringed median sized by the number of contributing participants.

    No ROI is dropped for low participant count. Instead every ROI carries an
    `n=` / `ch=` annotation, because the ranking is substantially a SIZE ranking:
    ROI-level rho(median enrichment, mean channel count) is -0.71 (fine) / -0.75
    (coarse), ROI size and ROI identity being collinear by implant design. The
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
            .agg(med=("_v", "median"), n=("patient", "nunique"),
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
    # ringed median across participants — the robust readout
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
    ax.set_title(title + "\n(ringed = median across participants, size ∝ n; "
                         "faded = individual; ROIs ranked by median)", fontsize=9)
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
    "participant; ringed markers are the per-ROI median across participants (size &prop; n) and are "
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
    "x&nbsp;= ROI ranked by descending median across participants; y&nbsp;= <b>per-electrode "
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
    "channel count) is <b>&minus;0.71</b> (fine) / <b>&minus;0.75</b> (coarse), because ROI size "
    "and ROI identity are collinear by implant design &mdash; depth shanks and MTG strips carry "
    "~20 contacts, ventral gyral ROIs 3&ndash;6. That is why every ROI is labelled with its mean "
    "channel count (<code>ch=</code>) as well as its participant count (<code>n=</code>): read them "
    "together with the rank. Normalization cannot remove this; even within participant, enrichment "
    "correlates &asymp;&minus;0.33 with channel count. "
    "<b>No ROI is dropped for low n</b>, so rows with <code>n=1</code> or <code>n=2</code> are "
    "single-participant observations, not group results. <b>No ROI clears a BH-corrected "
    "group-level test</b> of enrichment against 1 across participants (the strongest, MTG, is "
    "p&nbsp;= 0.031 &rarr; q&nbsp;= 0.28 over 9 ROIs &mdash; and MTG is the largest ROI in every "
    "participant, exactly what the size artifact predicts). Treat this as a descriptive ranking, "
    "not a finding.</div>")

_COV_NOTE = (
    "<div class='box'><b>Neural&ndash;GloVe covariance, per electrode.</b>&nbsp; "
    "x&nbsp;= picture, y&nbsp;= auditory, both as <b>per-electrode enrichment</b>: the region's "
    "value &divide; its channel count, divided by that participant's whole-brain per-electrode "
    "average <b>for the same task</b>. Colour = region, marker = participant; ringed markers are "
    "the per-ROI median across participants (size &prop; n). "
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


def section_measures(df, rcol, dfs_for_lims, measures=MEASURES_KNOCKOUT_PC,
                     heading="Region knockout &middot; per electrode",
                     note=_KNOCKOUT_NOTE) -> str:
    """Gallery of aggregated pic-vs-aud scatters, one per knockout measure, each
    with its own shared equal-scale range (pooled over `dfs_for_lims` so the fine
    and coarse groupings share a scale per measure). Skips measures whose columns
    are absent or all-NaN."""
    blocks = [note]
    for m in measures:
        if m["xcol"] not in df.columns or m["ycol"] not in df.columns:
            continue
        if not (np.isfinite(df[m["xcol"]]).any() and np.isfinite(df[m["ycol"]]).any()):
            continue
        lims = _shared_limits(dfs_for_lims, m["xcol"], m["ycol"])
        agg = _aggregated_scatter(df, rcol, lims, xcol=m["xcol"], ycol=m["ycol"],
                                  axis=m["axis"],
                                  title="{} — colour = region, marker = participant".format(m["name"]))
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


def section_part(df, rcol, heading, subtitle, slug, dfs_for_lims) -> str:
    """One granularity (fine or coarse): overview + the five sections, in order,
    each individually foldable and TOC'd under this part.

    Fine and coarse are the same five views on two ROI parcellations, so they are
    one code path rather than the duplicated assembly this replaces. `slug`
    namespaces the child section ids so the two parts do not collide.

    The part's own TOC entry is recorded HERE, before its children are built —
    `_fold` appends on return, so folding this part in the caller would file the
    parent after its own children."""
    _TOC.append(("s-" + slug, _re.sub(r"<[^>]+>", "", heading), False))
    sid = lambda s: "s-{}-{}".format(slug, s)
    return (
        "<h1>{}</h1><p class='subtle'>{}</p>".format(heading, subtitle)
        + _fold(section_overview(df), sid("overview"), open=True, sub=True)
        + _fold(section_measures(df, rcol, dfs_for_lims), sid("knockout"), sub=True)
        + _fold(section_ranked(df, rcol), sid("ranked"), sub=True)
        + _fold(section_cov(df, rcol, dfs_for_lims), sid("cov"), sub=True)
        + _fold(section_solo(df, rcol, dfs_for_lims), sid("solo"), sub=True)
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

def _load_merged(in_dir: Path, metric: str):
    """Return the merged (--merge-regions) dataframe for `metric`, or None."""
    csv = in_dir / "region_importance_merged_all.csv"
    if not csv.exists():
        return None
    m = pd.read_csv(csv)
    m = m[m["metric"] == metric].copy()
    return m if not m.empty else None


_FINE_SUBTITLE = (
    "Fine <code>primary_roi</code> parcels as they come from each participant's atlas "
    "(<code>{PAT}_*channels.pkl</code>) &mdash; anterior and posterior banks kept separate "
    "(aFus / pFus, aMTG / pMTG, &hellip;). The finest available anatomy, at the cost of ragged "
    "label sets: several parcels exist in only one or two participants, so their medians are "
    "single-participant observations. Source <code>region_importance_all.csv</code>.")

_COARSE_SUBTITLE = (
    "Anterior/posterior gyral pairs merged into one parcel (aFus+pFus &rarr; Fus, "
    "aMTG+pMTG &rarr; MTG, &hellip;) and atlas naming variants normalised "
    "(temporo-occipital &rarr; temporooccipital); <code>ant depth</code> / "
    "<code>post depth</code> kept separate. Coarser anatomy but better sampled &mdash; more "
    "participants contribute to each parcel, so the cross-participant medians are firmer. "
    "All measures are <b>recomputed</b> on the coarser grouping, not summed from the fine one "
    "(a joint knockout &#916; is not additive across sub-regions). Source "
    "<code>region_importance_merged_all.csv</code>.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="HTML report for cross_task_region_importance results")
    ap.add_argument("--balance", default="none",
                    choices=["none", "downsample", "upsample"],
                    help="Which resampling setting to report on. Resolves --in-dir to "
                         "<results>/cross_task_cotrain/balance_<BALANCE>/, matching the "
                         "analysis module's output layout (default: none).")
    ap.add_argument("--in-dir", default=None,
                    help="Directory containing region_importance_all.csv. Overrides "
                         "--balance.")
    ap.add_argument("--out", default=None,
                    help="Output HTML path (default: <in-dir>/region_importance_report.html)")
    ap.add_argument("--metric", default="cat_indep_bal_acc", choices=list(METRIC_SLUG),
                    help="Metric to report (default: cat_indep_bal_acc)")
    args = ap.parse_args()

    in_dir = (Path(args.in_dir) if args.in_dir
              else DEFAULT_IN_DIR / "balance_{}".format(args.balance))
    out_path = Path(args.out) if args.out else (in_dir / "region_importance_report.html")
    all_csv = in_dir / "region_importance_all.csv"
    if not all_csv.exists():
        print("ERROR: not found:", all_csv)
        if args.in_dir is None:
            print("       (resolved from --balance {}; pass --in-dir to override)"
                  .format(args.balance))
        return 1

    df = pd.read_csv(all_csv)
    df = df[df["metric"] == args.metric].copy()
    if df.empty:
        print("ERROR: no rows for metric '{}'".format(args.metric))
        return 1

    patients = sorted(df["patient"].unique())
    rcol = _region_colors(df["region"])
    # Knockout scatters share one equal-scale range across the fine + coarse
    # groupings (computed inside section_measures) so the two parts are comparable.
    merged_df = _load_merged(in_dir, args.metric)
    _add_per_channel(df); _add_per_channel(merged_df)      # <col>_pc  (sections 1, 2, 5)
    _add_standardized(df); _add_standardized(merged_df)    # <col>_std (sections 3, 4)
    dfs_for_lims = [df] + ([merged_df] if merged_df is not None else [])
    print("Fine ROIs: {} regions | coarse ROIs: {} | patients: {} | metric: {}".format(
        df["region"].nunique(),
        merged_df["region"].nunique() if merged_df is not None else 0,
        ", ".join(patients), args.metric))

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    method = (
        "<div class='box'><b>Method.</b>&nbsp; A single <b>kernel-PLS</b> model "
        "(Nystroem-RBF + PLSRegression &rarr; GloVe) is trained on pooled picture- and "
        "auditory-naming trials per participant (same model as "
        "<code>cross_task_cotrain.py</code>). Importance is assessed at the level of brain "
        "<b>regions</b> (<code>primary_roi</code>), on held-out test trials over bootstraps, "
        "by <b>four measures</b>, everywhere <b>per electrode</b>: "
        "<b>(1) &#916;category-accuracy knockout</b> (with a per-bootstrap label-shuffle null "
        "&rarr; BH-FDR groups <span class='sig'>both / picture_only / auditory_only</span> / "
        "<span class='ns'>neither</span>); <b>(2) &#916;cosine-to-GloVe knockout</b>; "
        "<b>(3) Jacobian sensitivity</b> &#8214;&#8706;&#375;/&#8706;x&#8214;; and "
        "<b>(4) neural&ndash;GloVe covariance</b> (model-free). They run from the end task "
        "toward the decoder's own covariance objective. Each region is read against the "
        "<b>whole-brain ceiling</b> (&#916;acc when all channels are knocked out); "
        "<code>frac_wb_*</code> is its share. A <b>fifth</b> section compares the co-trained "
        "model against picture-only and auditory-only decoders."
        "<br><br><b>How this page is organised.</b> Two parts &mdash; <b>fine ROIs</b> (the "
        "atlas parcels as given) and <b>coarse ROIs</b> (anterior/posterior banks merged) "
        "&mdash; each carrying the same five sections, so you can read the same question at two "
        "granularities. Measures 1, 2 and 4 are drawn picture-vs-auditory. Measure 3 (the "
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

    _TOC.clear()
    # the balance setting goes in the <title> and the header: the two settings produce
    # otherwise-identical-looking reports, and confusing them is easy
    bal = in_dir.name if in_dir.name.startswith("balance_") else args.balance
    header = (
        "<h1>Cross-task region (ROI) importance: picture &amp; auditory naming</h1>\n"
        "<p class='subtle'>Generated {gen} &bull; {npat} participants &bull; metric "
        "<code>{metric}</code> &bull; trial resampling <b><code>{bal}</code></b> &bull; source "
        "<code>{src}/region_importance_all.csv</code></p>\n"
    ).format(gen=generated, npat=len(patients), metric=args.metric,
             bal=bal, src=in_dir.name)
    # Two parts, same five sections each: fine parcels, then coarse (merged) parcels.
    parts = _fold(
        section_part(df, rcol,
                     "Part 1 &mdash; Fine ROIs ({} parcels)".format(df["region"].nunique()),
                     _FINE_SUBTITLE, "fine", dfs_for_lims),
        "s-fine", open=True, add_toc=False)          # TOC entry filed by section_part
    if merged_df is not None and not merged_df.empty:
        parts += _fold(
            section_part(merged_df, _region_colors(merged_df["region"]),
                         "Part 2 &mdash; Coarse ROIs ({} parcels)".format(
                             merged_df["region"].nunique()),
                         _COARSE_SUBTITLE, "coarse", dfs_for_lims),
            "s-coarse", add_toc=False)
    sections = method + parts + _fold(section_caveats(), "s-caveats")
    body = header + _toc_html() + sections   # TOC built after _fold populated _TOC

    html = ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>ROI importance ({}) — cross-task</title>".format(bal)
            + CSS + "</head><body>" + body + _TOC_SCRIPT + "</body></html>")
    out_path.write_text(html, encoding="utf-8")
    print("Wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
