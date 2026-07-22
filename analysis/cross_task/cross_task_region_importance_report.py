# cross_task_region_importance_report.py
# HTML report from cross_task_region_importance.py output (region_importance_all.csv).
#
# Region-only successor to the retired per-channel report
# (_archive/cross_task_reports/cross_task_channel_importance_report.py). Synthesizes
# the three region-organized methods — permutation region-knockout Δacc + Jacobian
# sensitivity (kernel PLS) and plain-PLS VIP — into one HTML report with, per patient
# and aggregated across all patients, a scatter of each region on the
# (Δcat-acc picture, Δcat-acc auditory) plane.
#
# Inputs (from --in-dir, default: main/results/cross_task_cotrain/):
#   region_importance_all.csv          (required: permutation Δacc + Jacobian + VIP + wb ceiling)
#   region_importance_merged_all.csv   (optional: from --merge-regions; adds a "Merged ROIs"
#                                        section with the same scatters/tables on coarser regions)
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


def _fold(html, sid, open=False, add_toc=True):
    """Wrap a section (whose HTML starts with <h1>/<h2>Title</...>) in a collapsible
    <details>, using the heading as the summary. Records (id, title) for the TOC."""
    m = _re.match(r"\s*<(h[12])>(.*?)</\1>(.*)", html, _re.S)
    if not m:
        return html
    _tag, title, rest = m.group(1), m.group(2), m.group(3)
    if add_toc:
        _TOC.append((sid, _re.sub(r"<[^>]+>", "", title)))
    return '<details id="{}" class="sec"{}><summary>{}</summary>{}</details>'.format(
        sid, " open" if open else "", title, rest)


def _toc_html():
    items = "".join('<li><a href="#{}">{}</a></li>'.format(sid, title) for sid, title in _TOC)
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


# ── per-task scatter measures (x = picture, y = auditory; one point per region) ──
# Each renders as its own aggregated scatter with a shared equal-scale range.
MEASURES = [
    dict(key="catacc", xcol="perm_imp_pic", ycol="perm_imp_aud",
         name="Δ category-independent accuracy (region knockout)",
         axis="Δ cat-indep accuracy",
         blurb="Drop in retrieval category accuracy when the whole region is knocked "
               "out. The end-task measure — furthest from what the PLS model optimises."),
    dict(key="cosine", xcol="cos_imp_pic", ycol="cos_imp_aud",
         name="Δ cosine to GloVe (region knockout)",
         axis="Δ cosine(ŷ, GloVe)",
         blurb="Drop in cosine between predicted and true GloVe when the region is "
               "knocked out — the knockout closest to the decoder's own objective."),
    dict(key="jac", xcol="jac_sens_pic", ycol="jac_sens_aud",
         name="Jacobian sensitivity  ‖∂ŷ/∂x‖",
         axis="Σ ‖∂ŷ/∂x‖ over region",
         blurb="Region-summed magnitude of the model-output gradient — how much the "
               "predicted embedding moves with the region. Model-intrinsic, per task."),
    dict(key="jacdir", xcol="jac_dir_pic", ycol="jac_dir_aud",
         name="Retrieval-aligned Jacobian  |∂(ŷ·û)/∂x|",
         axis="Σ |∂(ŷ·û)/∂x| over region",
         blurb="Gradient projected onto the correct-answer direction — sensitivity of "
               "the right GloVe alignment, not just gradient magnitude."),
    dict(key="vip", xcol="vip_pic", ycol="vip_aud",
         name="Per-task PLS VIP",
         axis="Σ VIP over region",
         blurb="Region-total VIP from separate picture-only and auditory-only PLS fits "
               "— what each task's own linear decoder leans on."),
    dict(key="cov", xcol="cov_pic", ycol="cov_aud",
         name="Neural–GloVe covariance",
         axis="Σ ‖covariance‖ over region",
         blurb="Region-total standardized neural↔GloVe cross-covariance — the rawest "
               "form of the PLS objective, purely data-driven per task."),
]

# Per-channel (÷ n_channels) variants of the same 6 measures. Region totals scale
# with how many electrodes landed in the ROI (an implant artifact), so the
# per-channel mean controls for size and is more comparable across participants.
MEASURES_PC = [
    dict(key=m["key"] + "_pc", xcol=m["xcol"] + "_pc", ycol=m["ycol"] + "_pc",
         name=m["name"] + "  · per channel",
         axis=m["axis"].replace("Σ ", "").replace(" over region", "") + " / channel",
         blurb=m["blurb"])
    for m in MEASURES
]


def _add_per_channel(df):
    """Add `<col>_pc = <col> / n_channels` for every measure's pic/aud columns, so
    the per-channel gallery can plot them. Safe no-op for missing columns."""
    if df is None or df.empty or "n_channels" not in df.columns:
        return df
    n = df["n_channels"].replace(0, np.nan)
    cols = set()
    for m in MEASURES:
        cols.update((m["xcol"], m["ycol"]))
    cols.update(c for c in df.columns if c.endswith("_solo"))   # single-modality
    if "vip" in df.columns:
        cols.add("vip")                                          # pooled VIP
    for c in cols:
        if c in df.columns:
            df[c + "_pc"] = df[c] / n
    return df


# Within-patient, size-fair standardized variants for the three SCALE-BEARING
# magnitude measures (Jacobian, aligned Jacobian, covariance). For aggregating an
# ROI across participants you must remove BOTH (a) the per-participant magnitude
# scale (γ/‖A‖/amplitude for the Jacobians; the 1/√n_trials floor for covariance)
# AND (b) the ROI's channel count — the same ROI has different electrode counts in
# different people, so a total would be dominated by whoever had the most contacts.
# The quantity is a PER-CHANNEL ENRICHMENT: the region's per-electrode value divided
# by that participant's whole-brain per-electrode average, JOINTLY over pic+aud.
# ≈1 = as informative per electrode as the participant's average; >1 = enriched.
_STD_SPECS = [
    ("jac",    "jac_sens_pic", "jac_sens_aud", "Jacobian sensitivity  · within-patient, size-fair",
     "per-channel ‖∂ŷ/∂x‖ ÷ whole-brain avg"),
    ("jacdir", "jac_dir_pic",  "jac_dir_aud",  "Retrieval-aligned Jacobian  · within-patient, size-fair",
     "per-channel |∂(ŷ·û)/∂x| ÷ whole-brain avg"),
    ("cov",    "cov_nc_pic",   "cov_nc_aud",   "Neural–GloVe covariance (null-corrected) · within-patient, size-fair",
     "per-channel covariance ÷ whole-brain avg"),
]
MEASURES_STD = [
    dict(key=k + "_std", xcol=xp + "_std", ycol=ya + "_std", name=name, axis=axis,
         blurb="Per-electrode enrichment: the region's value ÷ its channel count, "
               "divided by the participant's whole-brain per-electrode average (one "
               "reference for both tasks). Removes both the per-participant scale and "
               "the ROI's channel count, so ROIs are comparable across people. ≈1 = "
               "an average electrode; >1 = above-average per electrode.")
    for (k, xp, ya, name, axis) in _STD_SPECS
]

# Co-trained vs single-modality comparison. For each measure (all but covariance —
# covariance is model-free), the single-modality decoder's per-channel importance vs
# the co-trained model's, per task. VIP uses its existing triple (pic-only / pooled /
# aud-only fits). Columns are the per-channel (_pc) forms.
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
    dict(key="jacdir", name="Retrieval-aligned Jacobian", axis="|∂(ŷ·û)/∂x| / channel",
         solo_pic="jac_dir_pic_solo_pc", cotr_pic="jac_dir_pic_pc",
         solo_aud="jac_dir_aud_solo_pc", cotr_aud="jac_dir_aud_pc",
         mid_x="jac_dir_pic_pc", mid_y="jac_dir_aud_pc"),
    dict(key="vip", name="Per-task PLS VIP", axis="VIP / channel",
         solo_pic="vip_pic_pc", cotr_pic="vip_pc",
         solo_aud="vip_aud_pc", cotr_aud="vip_pc",
         mid_x="vip_pic_pc", mid_y="vip_aud_pc"),
]

_MEASURES_SOLO_NOTE = (
    "<div class='box'><b>Co-trained vs single-modality decoders.</b>&nbsp; Does co-training rely on the "
    "same ROIs a single-task decoder would? For each measure, the <b>left</b> scatter plots the "
    "<b>picture-only</b> decoder's per-electrode ROI importance (x) against the <b>co-trained</b> model's "
    "picture importance (y); the <b>right</b> does the same for <b>auditory-only</b> vs co-trained "
    "auditory; the <b>middle</b> is the co-trained model itself (picture vs auditory). Points on the "
    "diagonal in left/right = co-training preserved that ROI's reliance; off-diagonal = it reorganized. "
    "Ringed markers are the median across participants (size &prop; n). VIP uses its three fits "
    "(picture-only / pooled / auditory-only). <b>Covariance is omitted</b> (model-free). <b>Caveat:</b> "
    "the auditory-only decoder is underpowered where the auditory task has few trials/repeats (AA, DR ~1 "
    "trial/word) &mdash; read its points as noisy.</div>")


def _heatmap(df, measure, out_title):
    """ROI × condition heatmap for one measure: rows = ROIs (median across
    participants), columns = [picture-only, co-trained·pic, co-trained·aud,
    auditory-only], cells min–max scaled within the measure."""
    pairs = list(zip(["picture-only", "co-trained·pic", "co-trained·aud", "auditory-only"],
                     [measure["solo_pic"], measure["cotr_pic"],
                      measure["cotr_aud"], measure["solo_aud"]]))
    seen, use = set(), []          # keep present columns, dedupe (VIP shares a pooled col)
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
    if not any(c.endswith("_solo_pc") for c in df.columns) and "vip_pc" not in df.columns:
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
    average — removing BOTH the per-participant scale and the ROI channel count, so an
    ROI is comparable when pooled across participants. Reference is a SINGLE scalar per
    participant, joint over pic+aud. Covariance uses its null-corrected columns."""
    if df is None or df.empty or "patient" not in df.columns or "n_channels" not in df.columns:
        return df
    g = df.groupby("patient")
    totch = g["n_channels"].transform("sum")               # participant's total channels
    nch = df["n_channels"].replace(0, np.nan)
    for _, xp, ya, _, _ in _STD_SPECS:
        if xp not in df.columns or ya not in df.columns:
            continue
        wb = (g[xp].transform("sum") + g[ya].transform("sum")) / 2.0   # whole-brain total (joint)
        ref_pc = (wb / totch).replace(0, np.nan)           # whole-brain per-channel average
        df[xp + "_std"] = (df[xp] / nch) / ref_pc
        df[ya + "_std"] = (df[ya] / nch) / ref_pc
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


def _full_limits(dfs, xcol, ycol, margin=0.08):
    """Full-range (all individual values) equal limits — for the per-participant
    detail scatters, which show one participant's every region and must not clip."""
    if not dfs:
        return -0.01, 0.01
    xs = np.concatenate([np.asarray(d[xcol], float) for d in dfs if xcol in d])
    ys = np.concatenate([np.asarray(d[ycol], float) for d in dfs if ycol in d])
    xs = xs[np.isfinite(xs)]; ys = ys[np.isfinite(ys)]
    if xs.size == 0 or ys.size == 0:
        return -0.01, 0.01
    lo = min(float(xs.min()), float(ys.min()))
    hi = max(float(xs.max()), float(ys.max()))
    span = (hi - lo) or 0.02
    pad = span * margin
    return lo - pad, hi + pad


# ---------------------------------------------------------------------------
# scatter plots  (Δcat-acc picture  vs  Δcat-acc auditory, one point per region)
# ---------------------------------------------------------------------------

def _patient_scatter(df_pat, rcol, title, lims,
                     xcol="perm_imp_pic", ycol="perm_imp_aud",
                     xlabel="picture", ylabel="auditory") -> str:
    """Per-patient scatter: each region a point at (xcol, ycol), coloured by region
    (dataset-wide map), every region labelled. `lims=(lo,hi)` is the shared
    equal-scale range applied to both axes."""
    lo, hi = lims
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    for _, r in df_pat.iterrows():
        if not (np.isfinite(r[xcol]) and np.isfinite(r[ycol])):
            continue
        reg = str(r["region"])
        ax.scatter(r[xcol], r[ycol], s=85, color=rcol.get(reg, "#777"),
                   edgecolors="#222", linewidths=0.7, zorder=3)
        ax.annotate(reg, (r[xcol], r[ycol]), fontsize=7.5,
                    xytext=(4, 3), textcoords="offset points")
    ax.plot([lo, hi], [lo, hi], ls=":", color="#999", lw=0.8, zorder=1)
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("{} — picture".format(xlabel))
    ax.set_ylabel("{} — auditory".format(ylabel))
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return _fig_to_img(fig, title)


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


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def _delta_cell(v, rank=0) -> str:
    if not np.isfinite(v):
        return "<td>&mdash;</td>"
    cls = "top1" if rank == 1 else "top2" if rank == 2 else ("neg" if v < 0 else "")
    return "<td class='{}'>{:+.4f}</td>".format(cls, v)


def _region_table(df_pat: pd.DataFrame) -> str:
    d = df_pat.sort_values("perm_imp_pic", ascending=False)
    rank_pic = d["perm_imp_pic"].rank(ascending=False, method="min")
    head = ("<table class='results'><tr>"
            "<th>region</th><th>n_ch</th>"
            "<th>Δacc pic</th><th>Δacc aud</th>"
            "<th>frac WB pic</th><th>frac WB aud</th>"
            "<th>Jac pic</th><th>Jac aud</th><th>VIP</th>"
            "<th>group</th></tr>")
    rows = []
    for _, r in d.iterrows():
        rp = int(rank_pic.loc[r.name])
        vip = "{:.1f}".format(r["vip"]) if "vip" in r and np.isfinite(r.get("vip", np.nan)) else "&mdash;"
        fwp = "{:.0%}".format(r["frac_wb_pic"]) if np.isfinite(r.get("frac_wb_pic", np.nan)) else "&mdash;"
        fwa = "{:.0%}".format(r["frac_wb_aud"]) if np.isfinite(r.get("frac_wb_aud", np.nan)) else "&mdash;"
        g = str(r["group"])
        gcls = "sig" if g != "neither" else "ns"
        rows.append(
            "<tr><td class='text'>{reg}</td><td>{nch}</td>{dpic}{daud}"
            "<td>{fwp}</td><td>{fwa}</td><td>{jp:.2f}</td><td>{ja:.2f}</td>"
            "<td>{vip}</td><td class='text {gcls}'>{g}</td></tr>".format(
                reg=r["region"], nch=int(r["n_channels"]),
                dpic=_delta_cell(r["perm_imp_pic"], 1 if rp == 1 else 2 if rp == 2 else 0),
                daud=_delta_cell(r["perm_imp_aud"]),
                fwp=fwp, fwa=fwa, jp=r["jac_sens_pic"], ja=r["jac_sens_aud"],
                vip=vip, g=g, gcls=gcls))
    return head + "".join(rows) + "</table>"


def _consensus(df: pd.DataFrame) -> str:
    """Cross-patient region consensus: within each patient rank regions by VIP,
    permutation max(pic,aud) and Jacobian mean(pic,aud) as percentiles, average the
    three, then average that within-patient score across patients per region label."""
    rows = []
    for pat, g in df.groupby("patient"):
        g = g.copy()
        g["_perm"] = g[["perm_imp_pic", "perm_imp_aud"]].max(axis=1)
        g["_jac"] = g[["jac_sens_pic", "jac_sens_aud"]].mean(axis=1)
        for col, dst in [("vip", "p_vip"), ("_perm", "p_perm"), ("_jac", "p_jac")]:
            if col in g and g[col].notna().any():
                g[dst] = g[col].rank(pct=True)
            else:
                g[dst] = np.nan
        g["score"] = g[["p_vip", "p_perm", "p_jac"]].mean(axis=1)
        rows.append(g[["patient", "region", "score", "group"]])
    allg = pd.concat(rows, ignore_index=True)
    agg = (allg.groupby("region")
           .agg(mean_score=("score", "mean"), n_pat=("patient", "nunique"),
                n_sig=("group", lambda s: int((s != "neither").sum())))
           .sort_values("mean_score", ascending=False))
    head = ("<table class='results'><tr><th>region</th>"
            "<th># participants</th><th>mean consensus %ile</th>"
            "<th># sig (any task)</th></tr>")
    body = "".join(
        "<tr><td class='text'>{r}</td><td>{n}</td><td>{s:.0%}</td><td>{k}</td></tr>".format(
            r=reg, n=int(row.n_pat), s=row.mean_score, k=int(row.n_sig))
        for reg, row in agg.iterrows())
    return head + body + "</table>"


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


_MEASURES_NOTE = (
    "<div class='box'><b>Per-task importance measures.</b>&nbsp; Each scatter places every "
    "participant's region at its <b>picture</b> (x) and <b>auditory</b> (y) importance under one "
    "measure. Colour = region (shared across participants), marker = participant. The measures run "
    "from the end-task (&#916;category accuracy) toward the decoder's own objective (covariance / "
    "VIP); points near the dotted <i>pic&nbsp;=&nbsp;aud</i> diagonal are amodal, off-axis points "
    "are task-biased. Region-total magnitudes (Jacobian / VIP / covariance) scale with region "
    "size.</div>")

_MEASURES_PC_NOTE = (
    "<div class='box'><b>Per-channel (&divide; n_channels).</b>&nbsp; The same six measures divided "
    "by the number of electrodes in the ROI, i.e. a per-electrode intensity rather than a region "
    "total. This controls for how many contacts happened to land in a region (an implant artifact) "
    "and is more comparable across participants. Two caveats: the mean is noisier for small ROIs "
    "(2&ndash;4 channels), and for the two knockout measures &ldquo;per channel&rdquo; is a rough "
    "heuristic (a joint knockout is not a sum of per-channel effects). <b>Note:</b> the magnitude "
    "measures (Jacobian, aligned Jacobian, covariance) <i>cluster by participant</i> here &mdash; "
    "they carry a per-participant scale (&gamma;/&#8214;A&#8214;/amplitude; the 1/&radic;n sampling "
    "floor for covariance) that size-normalization does not remove. The next gallery fixes that.</div>")

_MEASURES_STD_NOTE = (
    "<div class='box'><b>Within-participant, size-fair (for cross-participant aggregation).</b>&nbsp; "
    "This is the view for pooling an ROI across people. Each region is a <b>per-electrode "
    "enrichment</b>: its value &divide; its channel count, divided by the participant's whole-brain "
    "per-electrode average, with one reference for both tasks. This removes <i>both</i> the "
    "per-participant magnitude scale <i>and</i> the ROI's channel count &mdash; the latter matters "
    "because the same ROI has different electrode counts in different people, so a total would be "
    "dominated by whoever had the most contacts there. <b>&asymp;1</b> = as informative per electrode "
    "as that participant's average; <b>&gt;1</b> = above-average. Covariance is <b>null-corrected "
    "first</b> (its 1/&radic;n_trials floor subtracted, which also removes the trial-count offset that "
    "pushed raw covariance above the diagonal). Only measures 3/4/6 are shown &mdash; the knockouts "
    "(1/2) already have <code>frac_wb</code> and VIP (5) is normalized by construction. Caveat: the "
    "per-electrode mean is noisier for small ROIs (2&ndash;4 channels).</div>")


def section_measures(df, rcol, dfs_for_lims, measures=MEASURES,
                     heading="Task-importance measures", note=_MEASURES_NOTE) -> str:
    """The gallery: one aggregated pic-vs-aud scatter per measure, each with its own
    shared equal-scale range (pooled over `dfs_for_lims` so fine and coarse groupings
    share a scale per measure). Skips measures whose columns are absent or all-NaN."""
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


def section_patient(pat: str, df_pat: pd.DataFrame, rcol: dict, lims) -> str:
    """Per-patient detail for the PRIMARY measure (Δcat-acc knockout) + full table."""
    n_reg = len(df_pat)
    scatter = _patient_scatter(df_pat, rcol,
                               "{} — regions on the Δcat-acc pic / aud plane".format(pat),
                               lims)
    tbl = _region_table(df_pat)
    return ("<details class='meas'><summary>Participant {p} &mdash; {n} regions</summary>"
            "<div class='pat-grid'><div>{s}</div><div>{t}</div></div></details>".format(
                p=pat, n=n_reg, s=scatter, t=tbl))


def section_consensus(df: pd.DataFrame) -> str:
    intro = ("<p class='subtle'>Per participant, regions are ranked by VIP, by "
             "permutation max(pic,&nbsp;aud) and by Jacobian mean(pic,&nbsp;aud) as "
             "percentiles; the three are averaged, then averaged across participants "
             "per region label. Higher = consistently important across methods and "
             "people. Region label sets are ragged across participants, so "
             "'# participants' shows how many contribute to each row.</p>")
    return "<h2>Cross-participant region consensus</h2>" + intro + _consensus(df)


def section_caveats() -> str:
    return (
        "<h2>Interpretation &amp; caveats</h2>"
        "<div class='qbox'>"
        "<b>Region score is a total</b> (summed over the region's channels), so larger "
        "regions can score higher partly by size; the per-channel-normalised columns "
        "(<code>perm_imp_*_per_ch</code>) in the CSV separate 'matters because big' from "
        "'matters per electrode'. "
        "<b>Read auditory against its ceiling</b>: the pooled model decodes auditory only "
        "slightly above chance, so the whole-brain auditory ceiling is small — a region can "
        "hold a large <i>share</i> (frac WB aud) while its absolute &#916;acc looks like "
        "noise. "
        "<b>Significance is conservative</b>: under the Nystroem-RBF dilution even whole-region "
        "knockout rarely clears BH-FDR, so the region <i>ranking</i> and <i>ceiling share</i> "
        "carry the signal rather than per-region certification. "
        "Single-channel attribution is deliberately not reported (retired 2026-07-20)."
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


def _merged_section(merged_df, dfs_for_lims, catacc_lims) -> str:
    """Optional 'Merged ROIs' section (only if merged_df is present). Reuses the
    same dataframe-generic builders on the coarser merged data, with its own region
    colour map (fewer labels -> even more distinct); measure scatters share a scale
    with the fine grouping (per-measure lims from `dfs_for_lims`)."""
    if merged_df is None or merged_df.empty:
        return ""
    m = merged_df
    rcol_m = _region_colors(m["region"])
    pats = sorted(m["patient"].unique())
    print("Merged ROIs: {} regions across {} participants".format(
        m["region"].nunique(), len(pats)))
    return (
        "<h1>Merged ROIs (anterior + posterior combined)</h1>"
        "<p class='subtle'>Anterior/posterior gyral pairs merged into one region "
        "(aFus+pFus &rarr; Fus, aMTG+pMTG &rarr; MTG, &hellip;) and atlas naming "
        "variants normalised; <code>ant depth</code> / <code>post depth</code> kept "
        "separate. All measures are <b>recomputed</b> on the coarser grouping "
        "(knockout &#916; is not additive across sub-regions). Source "
        "<code>region_importance_merged_all.csv</code>.</p>"
        + section_overview(m)
        + section_measures(m, rcol_m, dfs_for_lims)
        + section_measures(m, rcol_m, dfs_for_lims, MEASURES_PC,
                           "Task-importance measures · per channel", _MEASURES_PC_NOTE)
        + section_measures(m, rcol_m, dfs_for_lims, MEASURES_STD,
                           "Magnitude measures · within-participant, size-fair (cross-patient)", _MEASURES_STD_NOTE)
        + section_consensus(m)
        + "<h2>Per-participant merged regions (Δcat-acc)</h2>"
        + "\n".join(section_patient(p, m[m.patient == p].copy(), rcol_m, catacc_lims)
                    for p in pats)
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="HTML report for cross_task_region_importance results")
    ap.add_argument("--in-dir", default=str(DEFAULT_IN_DIR),
                    help="Directory containing region_importance_all.csv")
    ap.add_argument("--out", default=None,
                    help="Output HTML path (default: <in-dir>/region_importance_report.html)")
    ap.add_argument("--metric", default="cat_indep_bal_acc", choices=list(METRIC_SLUG),
                    help="Metric to report (default: cat_indep_bal_acc)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out) if args.out else (in_dir / "region_importance_report.html")
    all_csv = in_dir / "region_importance_all.csv"
    if not all_csv.exists():
        print("ERROR: not found:", all_csv)
        return 1

    df = pd.read_csv(all_csv)
    df = df[df["metric"] == args.metric].copy()
    if df.empty:
        print("ERROR: no rows for metric '{}'".format(args.metric))
        return 1

    patients = sorted(df["patient"].unique())
    rcol = _region_colors(df["region"])
    # Per-measure scatters share one equal-scale range across the fine + merged
    # groupings (computed inside section_measures); the per-patient Δcat-acc detail
    # uses the catacc range.
    merged_df = _load_merged(in_dir, args.metric)
    _add_per_channel(df); _add_per_channel(merged_df)      # <col>_pc for the per-channel gallery
    _add_standardized(df); _add_standardized(merged_df)    # <col>_std for the within-patient gallery
    dfs_for_lims = [df] + ([merged_df] if merged_df is not None else [])
    catacc_lims = _full_limits(dfs_for_lims, "perm_imp_pic", "perm_imp_aud")
    print("Patients: {} | regions: {} | metric: {} | measures: {}".format(
        ", ".join(patients), df["region"].nunique(), args.metric,
        ", ".join(m["key"] for m in MEASURES)))

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    method = (
        "<div class='box'><b>Method.</b>&nbsp; A single <b>kernel-PLS</b> model "
        "(Nystroem-RBF + PLSRegression &rarr; GloVe) is trained on pooled picture- and "
        "auditory-naming trials per participant (same model as "
        "<code>cross_task_cotrain.py</code>). Importance is assessed at the level of brain "
        "<b>regions</b> (<code>primary_roi</code>), each score a <b>total</b> over the "
        "region's electrodes, on held-out test trials over bootstraps, by <b>six per-task "
        "measures</b> (each shown pic-vs-aud in the gallery below): "
        "<b>(1) &#916;category-accuracy knockout</b> (with a per-bootstrap label-shuffle null "
        "&rarr; BH-FDR groups <span class='sig'>both / picture_only / auditory_only</span> / "
        "<span class='ns'>neither</span>); <b>(2) &#916;cosine-to-GloVe knockout</b>; "
        "<b>(3) Jacobian sensitivity</b> &#8214;&#8706;&#375;/&#8706;x&#8214;; "
        "<b>(4) retrieval-aligned Jacobian</b> |&#8706;(&#375;&middot;&#251;)/&#8706;x|; "
        "<b>(5) per-task PLS VIP</b> (separate picture-only / auditory-only fits); and "
        "<b>(6) neural&ndash;GloVe covariance</b>. They run from the end task toward the "
        "decoder's own covariance objective. Each region is read against the <b>whole-brain "
        "ceiling</b> (&#916;acc when all channels are knocked out); <code>frac_wb_*</code> is "
        "its share."
        "</div>").format(metric=args.metric)

    _TOC.clear()
    header = (
        "<h1>Cross-task region (ROI) importance: picture &amp; auditory naming</h1>\n"
        "<p class='subtle'>Generated {gen} &bull; {npat} participants &bull; metric "
        "<code>{metric}</code> &bull; source <code>region_importance_all.csv</code></p>\n"
    ).format(gen=generated, npat=len(patients), metric=args.metric)
    perpat = ("<h2>Per-participant regions (Δcat-acc)</h2>"
              + "\n".join(section_patient(p, df[df.patient == p].copy(), rcol, catacc_lims)
                          for p in patients))
    sections = (
        method
        + _fold(section_overview(df), "s-overview", open=True)
        + _fold(section_measures(df, rcol, dfs_for_lims), "s-totals")
        + _fold(section_measures(df, rcol, dfs_for_lims, MEASURES_PC,
                                 "Task-importance measures · per channel", _MEASURES_PC_NOTE), "s-perch")
        + _fold(section_measures(df, rcol, dfs_for_lims, MEASURES_STD,
                                 "Magnitude measures · within-participant, size-fair (cross-patient)",
                                 _MEASURES_STD_NOTE), "s-sizefair")
        + _fold(section_solo(df, rcol, dfs_for_lims), "s-solo")
        + _fold(section_consensus(df), "s-consensus")
        + _fold(perpat, "s-perpatient")
        + _fold(_merged_section(merged_df, dfs_for_lims, catacc_lims), "s-merged")
        + _fold(section_caveats(), "s-caveats")
    )
    body = header + _toc_html() + sections   # TOC built after _fold populated _TOC

    html = ("<!DOCTYPE html><html><head><meta charset='utf-8'>" + CSS + "</head><body>"
            + body + _TOC_SCRIPT + "</body></html>")
    out_path.write_text(html, encoding="utf-8")
    print("Wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
