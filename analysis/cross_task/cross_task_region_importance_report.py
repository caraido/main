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
</style>"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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


def _shared_limits(dfs, xcol, ycol, margin=0.08):
    """One (lo, hi) range shared by BOTH axes across ALL scatters of a measure,
    pooled over every row of every dataframe in `dfs`, so plots are on the same
    scale and the pic=aud diagonal is a true 45°.
        lo = min(min x, min y) - margin
        hi = max(max x, max y) + margin   (full range — nothing clipped)."""
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
                        title="All regions, all participants — colour = region, marker = participant") -> str:
    """All patients' regions on one (x, y) plane. Colour = region (shared across
    subjects), marker = patient. Two legends (region colour, patient marker).
    `lims=(lo,hi)` is the shared equal-scale range applied to both axes."""
    lo, hi = lims
    patients = sorted(df["patient"].unique())
    pmark = {p: _MARKERS[i % len(_MARKERS)] for i, p in enumerate(patients)}
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    for _, r in df.iterrows():
        if not (np.isfinite(r[xcol]) and np.isfinite(r[ycol])):
            continue
        reg, pat = str(r["region"]), r["patient"]
        ax.scatter(r[xcol], r[ycol], s=80,
                   color=rcol.get(reg, "#777"), marker=pmark[pat],
                   edgecolors="#222", linewidths=0.6, alpha=0.95, zorder=3)
    ax.plot([lo, hi], [lo, hi], ls=":", color="#999", lw=0.8, zorder=1,
            label="_pic = aud")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("{} — picture".format(axis))
    ax.set_ylabel("{} — auditory".format(axis))
    ax.set_title(title, fontsize=10)
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


def section_measures(df: pd.DataFrame, rcol: dict, dfs_for_lims) -> str:
    """The gallery: one aggregated pic-vs-aud scatter per measure in MEASURES, each
    with its own shared equal-scale range (pooled over `dfs_for_lims` so fine and
    coarse groupings share a scale per measure). Skips measures whose columns are
    absent or all-NaN."""
    note = ("<div class='box'><b>Per-task importance measures.</b>&nbsp; Each scatter "
            "places every participant's region at its <b>picture</b> (x) and "
            "<b>auditory</b> (y) importance under one measure. Colour = region (shared "
            "across participants), marker = participant. The measures run from the "
            "end-task (&#916;category accuracy) toward the decoder's own objective "
            "(covariance / VIP); points near the dotted <i>pic&nbsp;=&nbsp;aud</i> "
            "diagonal are amodal, off-axis points are task-biased. Region-total "
            "magnitudes (Jacobian / VIP / covariance) scale with region size.</div>")
    blocks = [note]
    for m in MEASURES:
        if m["xcol"] not in df.columns or m["ycol"] not in df.columns:
            continue
        if not (np.isfinite(df[m["xcol"]]).any() and np.isfinite(df[m["ycol"]]).any()):
            continue
        lims = _shared_limits(dfs_for_lims, m["xcol"], m["ycol"])
        agg = _aggregated_scatter(df, rcol, lims, xcol=m["xcol"], ycol=m["ycol"],
                                  axis=m["axis"],
                                  title="{} — colour = region, marker = participant".format(m["name"]))
        blocks.append("<h3>{}</h3><p class='subtle'>{}</p>{}".format(
            m["name"], m["blurb"], agg))
    return "<h2>Task-importance measures</h2>" + "".join(blocks)


def section_patient(pat: str, df_pat: pd.DataFrame, rcol: dict, lims) -> str:
    """Per-patient detail for the PRIMARY measure (Δcat-acc knockout) + full table."""
    n_reg = len(df_pat)
    scatter = _patient_scatter(df_pat, rcol,
                               "{} — regions on the Δcat-acc pic / aud plane".format(pat),
                               lims)
    tbl = _region_table(df_pat)
    return ("<h2>Participant {p} &mdash; {n} regions</h2>"
            "<div class='pat-grid'><div>{s}</div><div>{t}</div></div>".format(
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
    dfs_for_lims = [df] + ([merged_df] if merged_df is not None else [])
    catacc_lims = _shared_limits(dfs_for_lims, "perm_imp_pic", "perm_imp_aud")
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

    body = (
        "<h1>Cross-task region (ROI) importance: picture &amp; auditory naming</h1>\n"
        "<p class='subtle'>Generated {gen} &bull; {npat} participants &bull; metric "
        "<code>{metric}</code> &bull; source <code>region_importance_all.csv</code></p>\n"
        + method
        + section_overview(df)
        + section_measures(df, rcol, dfs_for_lims)
        + section_consensus(df)
        + "<h2>Per-participant regions (Δcat-acc)</h2>"
        + "\n".join(section_patient(p, df[df.patient == p].copy(), rcol, catacc_lims)
                    for p in patients)
        + _merged_section(merged_df, dfs_for_lims, catacc_lims)
        + section_caveats()
    ).format(gen=generated, npat=len(patients), metric=args.metric)

    html = "<!DOCTYPE html><html><head><meta charset='utf-8'>" + CSS + "</head><body>" \
           + body + "</body></html>"
    out_path.write_text(html, encoding="utf-8")
    print("Wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
