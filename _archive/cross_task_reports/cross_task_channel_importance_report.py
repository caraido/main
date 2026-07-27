# cross_task_channel_importance_report.py
# HTML report from cross_task_channel_importance.py CSV outputs.
#
# RETIRED 2026-07-20: the per-channel analysis this wrapped is gone (superseded by
# ROI-only analysis/cross_task/cross_task_region_importance.py). Its input CSVs are
# archived at _archive/cross_task_channel_importance_results/. Kept for reference only.
#
# Synthesizes the three channel-importance methods into one report:
#   - permutation Δacc + Jacobian sensitivity  (kernel PLS, --analysis permutation)
#   - VIP                                       (plain linear PLS, --analysis pls)
# with a per-patient method-agreement table and a cross-patient consensus ranking.
#
# Inputs (from --in-dir, default: main/tests/results/cross_task_cotrain/):
#   channel_importance_all.csv                  (required: permutation + Jacobian)
#   channel_pls_vip_all.csv                     (optional: VIP; degrades gracefully)
#   <PAT>/channel_importance_{PAT}_{metric_slug}.png
#   <PAT>/channel_jacobian_{PAT}_{metric_slug}.png
#
# Output (default):
#   <in-dir>/channel_importance_report.html
#
# Usage:
#   python -m main.analysis.cross_task.cross_task_channel_importance_report
#   python -m main.analysis.cross_task.cross_task_channel_importance_report --metric cat_indep_bal_acc --top 5
#   python -m main.analysis.cross_task.cross_task_channel_importance_report --in-dir <dir> --out <out.html>

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_THIS_DIR)
_MAIN_DIR = os.path.dirname(_TESTS_DIR)
DEFAULT_IN_DIR = Path(_MAIN_DIR) / "tests" / "results" / "cross_task_cotrain"
DEFAULT_DATA_DIR = Path(_MAIN_DIR) / "data"

METRIC_SLUG = {
    "cat_indep_bal_acc": "catindep",
    "word_bal_acc": "word",
    "cosine_mean": "cosine",
}

# ---------------------------------------------------------------------------
# CSS  (same visual language as cross_task_cotrain_report.py)
# ---------------------------------------------------------------------------

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
details { margin: 8px 0; }
summary { cursor: pointer; color: #1565C0; font-size: 13px; font-weight: 500; }
summary:hover { text-decoration: underline; }
.box  { background: #F5F7FA; padding: 10px 14px; border-left: 3px solid #1565C0; margin: 12px 0; font-size: 13px; }
.qbox { background: #FFF8E1; padding: 10px 14px; border-left: 3px solid #F9A825; margin: 12px 0; font-size: 13px; }
.rbox { background: #FBE9E7; padding: 10px 14px; border-left: 3px solid #E64A19; margin: 12px 0; font-size: 13px; }
.sig  { color: #2E7D32; font-weight: 600; }
.ns   { color: #9E9E9E; }
.pat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 12px 0; }
@media (max-width: 900px) { .pat-grid { grid-template-columns: 1fr; } }
</style>"""


# ---------------------------------------------------------------------------
# Channel name resolution
# ---------------------------------------------------------------------------

# Mirror semantic_regression.py's patient-specific shank exclusions so positional
# ch{N} labels resolve to the channels the model actually kept (see the same map and
# rationale in cross_task_channel_importance.py). Applied only to the anatomical-name
# branch; RB resolves by integer position into its (V-inclusive) dataframe and is left
# intact.
_PATIENT_EXCLUDE_PREFIXES = {
    "LH": ("O", "V", "P", "Q", "R"),
    "RB": ("V",),
}


def _build_channel_map(pat: str, data_dir: Path) -> dict:
    """Return {csv_label: electrode_name} for a patient. Returns {} on failure.

    Resolution rules (from CLAUDE.md):
      AZ / LH / WBH  — ch{N}  → clean_channel_names[N] from *_channels.pkl
                       (post SR shank-exclusion, so ch{N} matches the model's data)
      DR             — int N  → channel_names[N] from DR_picture_naming_df.pkl  (dill)
      RB             — int N  → channel_names[N] from RB_picture_naming_combined_df.pkl (dill)
      AA             — names already correct, no mapping needed
    """
    try:
        if pat in ("AZ", "LH", "WBH"):
            pkls = sorted((data_dir / pat).glob("{}_*channels*.pkl".format(pat)))
            if not pkls:
                return {}
            ch_df = pd.read_pickle(pkls[0])
            clean = ch_df[ch_df["clean"]]["channel_name"].astype(str).tolist()
            prefixes = _PATIENT_EXCLUDE_PREFIXES.get(pat)
            if prefixes:                       # drop the same shanks SR deleted
                clean = [c for c in clean if not c.startswith(prefixes)]
            return {"ch{}".format(n): name for n, name in enumerate(clean)}
        elif pat == "DR":
            import dill
            with open(data_dir / "DR" / "DR_picture_naming_df.pkl", "rb") as fh:
                df = dill.load(fh)
            cnames = df.iloc[0]["channel_names"]
            return {str(n): str(cnames[n]) for n in range(len(cnames))}
        elif pat == "RB":
            import dill
            with open(data_dir / "RB" / "RB_picture_naming_combined_df.pkl", "rb") as fh:
                df = dill.load(fh)
            cnames = df.iloc[0]["channel_names"]
            return {str(n): str(cnames[n]) for n in range(len(cnames))}
    except Exception as e:
        print("WARNING: could not resolve channel names for {}: {}".format(pat, e))
    return {}


def _apply_channel_maps(df, data_dir: Path) -> pd.DataFrame:
    """Resolve raw channel labels to electrode names in-place (copy returned)."""
    df = df.copy()
    for pat in df["patient"].unique():
        chan_map = _build_channel_map(pat, data_dir)
        if chan_map:
            mask = df["patient"] == pat
            df.loc[mask, "channel"] = df.loc[mask, "channel"].map(
                lambda x, m=chan_map: m.get(str(x), str(x))
            )
    return df


# ---------------------------------------------------------------------------
# Significance basis (FDR-corrected q vs. uncorrected raw p)
# ---------------------------------------------------------------------------

def _sig_label(basis: str, alpha: float) -> str:
    """Human-readable significance criterion for the report wording."""
    if basis == "raw":
        return "raw p &lt; {:g} (uncorrected)".format(alpha)
    return "BH-FDR q &lt; {:g}".format(alpha)


def _regroup(df: pd.DataFrame, basis: str, alpha: float) -> pd.DataFrame:
    """Recompute the both / picture_only / auditory_only / neither grouping from
    the chosen significance basis. ``fdr`` uses the BH-corrected q columns (as the
    source CSV was built); ``raw`` uses the uncorrected per-bootstrap p columns.
    A channel is significant for a task only if its p/q < alpha AND Δacc > 0,
    matching analyze_patient in cross_task_channel_importance.py."""
    pic_col, aud_col = ("p_pic", "p_aud") if basis == "raw" else ("q_pic", "q_aud")
    df = df.copy()
    sig_pic = (df[pic_col] < alpha) & (df["perm_imp_pic"] > 0)
    sig_aud = (df[aud_col] < alpha) & (df["perm_imp_aud"] > 0)
    df["group"] = np.select(
        [sig_pic & sig_aud, sig_pic, sig_aud],
        ["both", "picture_only", "auditory_only"], default="neither")
    return df


# ---------------------------------------------------------------------------
# VIP (plain-PLS) merge  — the third importance method
# ---------------------------------------------------------------------------

def _load_vip(in_dir: Path, data_dir: Path):
    """Load channel_pls_vip_all.csv (plain-PLS VIP importance) if present, with
    channel names resolved to electrode names. Returns None when absent."""
    vip_csv = in_dir / "channel_pls_vip_all.csv"
    if not vip_csv.exists():
        return None
    v = pd.read_csv(vip_csv)
    if "vip" not in v.columns or v.empty:
        return None
    keep = [c for c in ("patient", "channel", "vip", "vip_std") if c in v.columns]
    return _apply_channel_maps(v[keep], data_dir)


def _merge_vip(df_m: pd.DataFrame, vip) -> pd.DataFrame:
    """Left-merge VIP onto the permutation/Jacobian frame on (patient, channel).
    Always returns a frame with a 'vip' column (NaN where VIP is unavailable)."""
    df_m = df_m.copy()
    df_m["channel"] = df_m["channel"].astype(str)
    if vip is None:
        df_m["vip"] = float("nan")
        return df_m
    vip = vip.copy()
    vip["channel"] = vip["channel"].astype(str)
    return df_m.merge(vip[["patient", "channel", "vip"]],
                      on=["patient", "channel"], how="left")


# ---------------------------------------------------------------------------
# Region-level permutation importance  (region_importance_all.csv)
# ---------------------------------------------------------------------------

def _load_region(in_dir: Path, metric: str, basis: str, alpha: float):
    """Load region_importance_all.csv for *metric*, regrouped on the chosen
    significance basis (same both/pic/aud/neither logic as channels). Region
    names are already resolved in the CSV, so no channel-map step is needed.
    Returns None when the file is absent (older runs) or has no rows."""
    csv = in_dir / "region_importance_all.csv"
    if not csv.exists():
        return None
    r = pd.read_csv(csv)
    r = r[r["metric"] == metric].copy()
    if r.empty:
        return None
    return _regroup(r, basis, alpha)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

_GCOL = {"both": "#2ca02c", "picture_only": "#1f77b4",
         "auditory_only": "#d62728", "neither": "#bbbbbb"}


def _make_scatter(df_pat, xcol: str, ycol: str, title: str,
                  xlab: str, ylab: str, annotate_top: int = 5) -> str:
    """Render a scatter plot from the resolved DataFrame and return an <img> tag."""
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    for g, sub in df_pat.groupby("group"):
        ax.scatter(sub[xcol], sub[ycol], s=26, c=_GCOL.get(g, "#777"),
                   label="{} (n={})".format(g, len(sub)), alpha=0.8, edgecolors="none")
    top = df_pat.assign(_m=df_pat[[xcol, ycol]].min(axis=1)).nlargest(annotate_top, "_m")
    for _, r in top.iterrows():
        ax.annotate(str(r["channel"]), (r[xcol], r[ycol]), fontsize=8,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return '<img alt="{}" src="data:image/png;base64,{}" />'.format(title, b64)


def _delta_cell(v: float, rank: int = 0) -> str:
    """Format a Δacc value as an HTML <td>. rank=1/2 adds green highlight."""
    parts = []
    if rank == 1:
        parts.append("top1")
    elif rank == 2:
        parts.append("top2")
    if v < 0:
        parts.append("neg")
    cls = " class='{}'".format(" ".join(parts)) if parts else ""
    fmt = "+{:.4f}".format(v) if v >= 0 else "&minus;{:.4f}".format(abs(v))
    return "<td{}>{}</td>".format(cls, fmt)


def _group_counts(df_pat: pd.DataFrame) -> dict:
    counts = {"both": 0, "picture_only": 0, "auditory_only": 0, "neither": 0}
    for g in df_pat["group"]:
        key = str(g)
        if key in counts:
            counts[key] += 1
    return counts


def _vip_cell(v) -> str:
    """Format a VIP value as a <td>; green highlight when VIP >= 1 (above avg)."""
    if v is None or pd.isna(v):
        return "<td class='neg'>&mdash;</td>"
    cls = " class='top1'" if v >= 1.0 else ""
    return "<td{}>{:.2f}</td>".format(cls, v)


_GRP_TXT = {
    "both": "<span class='sig'>both</span>",
    "picture_only": "picture",
    "auditory_only": "<span class='sig'>auditory</span>",
    "neither": "<span class='ns'>neither</span>",
}


def _consensus_frame(df_pat: pd.DataFrame) -> pd.DataFrame:
    """Add a per-patient consensus across the three methods.

    perm_overall = max(Δacc pic, Δacc aud)   — important in *either* task
    jac_overall  = mean(Jac pic, Jac aud)
    *_pct        = within-patient percentile rank (higher = more important)
    consensus    = mean of the available method percentiles (VIP / perm / Jac)
    n_agree      = #methods placing the channel in its top quartile (pct >= .75)
    """
    out = df_pat.copy()
    out["perm_overall"] = out[["perm_imp_pic", "perm_imp_aud"]].max(axis=1)
    out["jac_overall"] = out[["jac_sens_pic", "jac_sens_aud"]].mean(axis=1)
    pct_cols = []
    for raw in ("vip", "perm_overall", "jac_overall"):
        col = raw + "_pct"
        out[col] = out[raw].rank(pct=True)
        pct_cols.append(col)
    out["consensus"] = out[pct_cols].mean(axis=1)
    out["n_agree"] = (out[pct_cols] >= 0.75).sum(axis=1)
    return out


def _agree_badge(n_agree: int, sig: bool) -> str:
    """Filled/empty dots for how many methods agree, ★ when all 3 + significant."""
    n = int(n_agree)
    dots = "&#9679;" * n + "&#9675;" * (3 - n)
    star = "&nbsp;&#9733;" if (n == 3 and sig) else ""
    cls = "sig" if n >= 2 else "ns"
    return "<span class='{}'>{}{}</span>".format(cls, dots, star)


# ---------------------------------------------------------------------------
# Per-patient channel tables
# ---------------------------------------------------------------------------

def _channel_table_pic(df_pat: pd.DataFrame, top_n: int) -> str:
    top = df_pat.sort_values("perm_imp_pic", ascending=False).head(top_n)
    thead = (
        "<thead><tr>"
        "<th>Rank</th><th>Channel</th>"
        "<th>&#916;acc&nbsp;(pic)</th><th>&#916;acc&nbsp;(aud)</th>"
        "<th>VIP</th>"
        "<th>p_pic&nbsp;(raw)</th><th>q_pic&nbsp;(BH)</th>"
        "<th>Jac&nbsp;(pic)</th><th>Jac&nbsp;(aud)</th>"
        "</tr></thead>"
    )
    rows = []
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        rows.append(
            "<tr><td>{rank}</td>"
            "<td class='text{rcls}'>{ch}</td>"
            "{pv}{av}{vip}"
            "<td>{pp:.4f}</td><td>{qp:.4f}</td>"
            "<td>{jp:.3f}</td><td>{ja:.3f}</td></tr>".format(
                rank=rank,
                rcls=" top1" if rank == 1 else (" top2" if rank == 2 else ""),
                ch=r["channel"],
                pv=_delta_cell(r["perm_imp_pic"], rank),
                av=_delta_cell(r["perm_imp_aud"]),
                vip=_vip_cell(r.get("vip")),
                pp=r["p_pic"], qp=r["q_pic"],
                jp=r["jac_sens_pic"], ja=r["jac_sens_aud"],
            )
        )
    return "<table class='results'>{}<tbody>{}</tbody></table>".format(thead, "".join(rows))


def _channel_table_aud(df_pat: pd.DataFrame, top_n: int) -> tuple[str, str]:
    """Returns (label_suffix, table_html). Shows only positive-importance channels."""
    df_pos = df_pat[df_pat["perm_imp_aud"] > 0].sort_values("perm_imp_aud", ascending=False)
    top = df_pos.head(top_n)
    if top.empty:
        return "", "<p class='subtle'>(no positive-importance auditory channels)</p>"
    label_suffix = " (positive &#916;acc only)" if len(df_pos) < len(df_pat) else ""
    thead = (
        "<thead><tr>"
        "<th>Rank</th><th>Channel</th>"
        "<th>&#916;acc&nbsp;(aud)</th><th>&#916;acc&nbsp;(pic)</th>"
        "<th>VIP</th>"
        "<th>p_aud&nbsp;(raw)</th><th>q_aud&nbsp;(BH)</th>"
        "<th>Jac&nbsp;(pic)</th><th>Jac&nbsp;(aud)</th>"
        "</tr></thead>"
    )
    rows = []
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        rows.append(
            "<tr><td>{rank}</td>"
            "<td class='text{rcls}'>{ch}</td>"
            "{av}{pv}{vip}"
            "<td>{pa:.4f}</td><td>{qa:.4f}</td>"
            "<td>{jp:.3f}</td><td>{ja:.3f}</td></tr>".format(
                rank=rank,
                rcls=" top1" if rank == 1 else (" top2" if rank == 2 else ""),
                ch=r["channel"],
                av=_delta_cell(r["perm_imp_aud"], rank),
                pv=_delta_cell(r["perm_imp_pic"]),
                vip=_vip_cell(r.get("vip")),
                pa=r["p_aud"], qa=r["q_aud"],
                jp=r["jac_sens_pic"], ja=r["jac_sens_aud"],
            )
        )
    return label_suffix, "<table class='results'>{}<tbody>{}</tbody></table>".format(thead, "".join(rows))


# ---------------------------------------------------------------------------
# Cross-patient overview table
# ---------------------------------------------------------------------------

def section_overview(df: pd.DataFrame, sig_label: str) -> str:
    rows = []
    for pat in sorted(df["patient"].unique()):
        dp = df[df["patient"] == pat]
        best_pic = dp.loc[dp["perm_imp_pic"].idxmax()]
        best_aud = dp.loc[dp["perm_imp_aud"].idxmax()]
        if dp["vip"].notna().any():
            best_vip = dp.loc[dp["vip"].idxmax()]
            bvc, bvv = str(best_vip["channel"]), "{:.2f}".format(best_vip["vip"])
        else:
            bvc, bvv = "&mdash;", "&mdash;"
        gc = _group_counts(dp)
        rows.append(
            "<tr>"
            "<td class='text'>{pat}</td><td>{n}</td>"
            "<td class='text top1'>{bpc}</td>{bpv}"
            "<td class='text top1'>{bac}</td>{bav}"
            "<td class='text top1'>{bvc}</td><td>{bvv}</td>"
            "<td>{pp:.4f}</td>"
            "<td>{jp:.3f}</td><td>{ja:.3f}</td>"
            "<td>{both}</td><td>{po}</td><td>{ao}</td><td>{ne}</td>"
            "</tr>".format(
                pat=pat, n=len(dp),
                bpc=best_pic["channel"],
                bpv=_delta_cell(best_pic["perm_imp_pic"]),
                bac=best_aud["channel"],
                bav=_delta_cell(best_aud["perm_imp_aud"]),
                bvc=bvc, bvv=bvv,
                pp=best_pic["p_pic"],
                jp=best_pic["jac_sens_pic"],
                ja=best_pic["jac_sens_aud"],
                both=gc["both"], po=gc["picture_only"],
                ao=gc["auditory_only"], ne=gc["neither"],
            )
        )
    thead = (
        "<thead><tr>"
        "<th>Patient</th><th>N&nbsp;chan</th>"
        "<th>Best&nbsp;pic&nbsp;channel</th><th>&#916;acc&nbsp;(pic)</th>"
        "<th>Best&nbsp;aud&nbsp;channel</th><th>&#916;acc&nbsp;(aud)</th>"
        "<th>Top&nbsp;VIP&nbsp;channel</th><th>VIP</th>"
        "<th>p_pic&nbsp;(raw)</th>"
        "<th>Jac&nbsp;(pic)</th><th>Jac&nbsp;(aud)</th>"
        "<th>both</th><th>pic_only</th><th>aud_only</th><th>neither</th>"
        "</tr></thead>"
    )
    note = (
        "<div class='box'>"
        "<b>How to read.</b>&nbsp;"
        "Best picture / auditory channel = highest mean &#916;acc on that task's test set "
        "across bootstraps; Top VIP channel = highest plain-PLS VIP (linear, all pooled "
        "trials). p_pic (raw) is the averaged per-bootstrap p-value for the best "
        "picture channel. Group counts (both / pic_only / aud_only / neither) reflect "
        "significance at " + sig_label + "."
        "</div>"
    )
    return (
        "<h2>Cross-patient overview</h2>"
        + note
        + "<table class='results'>{}<tbody>{}</tbody></table>".format(thead, "".join(rows))
    )


# ---------------------------------------------------------------------------
# Brain-region permutation importance
# ---------------------------------------------------------------------------

def _make_region_bar(df_pat: pd.DataFrame, metric_tag: str) -> str:
    """Grouped horizontal bars of Δacc per region (picture vs auditory), sorted
    by picture importance, with the whole-brain ceiling drawn as dashed reference
    lines. Returns an inline <img> tag."""
    sub = df_pat.sort_values("perm_imp_pic", ascending=True)
    y = np.arange(len(sub))
    h = 0.4
    fig, ax = plt.subplots(figsize=(5.8, max(2.6, 0.46 * len(sub))))
    ax.barh(y + h / 2, sub["perm_imp_pic"].values, height=h,
            color=_GCOL["picture_only"], alpha=0.9, label="picture")
    ax.barh(y - h / 2, sub["perm_imp_aud"].values, height=h,
            color=_GCOL["auditory_only"], alpha=0.9, label="auditory")
    ax.axvline(0, color="k", lw=0.6)
    if "wb_imp_pic" in sub.columns:
        wbp, wba = float(sub["wb_imp_pic"].iloc[0]), float(sub["wb_imp_aud"].iloc[0])
        ax.axvline(wbp, color=_GCOL["picture_only"], lw=1.1, ls="--",
                   label="whole-brain pic ({:+.3f})".format(wbp))
        ax.axvline(wba, color=_GCOL["auditory_only"], lw=1.1, ls="--",
                   label="whole-brain aud ({:+.3f})".format(wba))
    ax.set_yticks(y)
    ax.set_yticklabels(["{}  (n={})".format(r, n) for r, n
                        in zip(sub["region"], sub["n_channels"])], fontsize=8)
    ax.set_xlabel("Δ{}  (region shuffled; dashed = whole-brain ceiling)".format(metric_tag))
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return '<img alt="region importance" src="data:image/png;base64,{}" />'.format(b64)


def _pct_cell(v) -> str:
    """Format a whole-brain share (fraction) as a percentage <td>."""
    if v is None or pd.isna(v):
        return "<td class='neg'>&mdash;</td>"
    cls = " class='neg'" if v < 0 else ""
    return "<td{}>{:.0f}%</td>".format(cls, v * 100.0)


def _region_table(df_pat: pd.DataFrame, top_n: int) -> str:
    """Per-patient region table, ranked by picture Δacc, showing each region's
    Δacc and its share of the whole-brain ceiling for both tasks."""
    has_wb = "frac_wb_pic" in df_pat.columns
    top = df_pat.sort_values("perm_imp_pic", ascending=False).head(top_n)
    thead = (
        "<thead><tr>"
        "<th>Rank</th><th>Region</th><th>N&nbsp;ch</th>"
        "<th>&#916;acc&nbsp;(pic)</th>" + ("<th>share&nbsp;(pic)</th>" if has_wb else "")
        + "<th>&#916;acc&nbsp;(aud)</th>" + ("<th>share&nbsp;(aud)</th>" if has_wb else "")
        + "<th>p_pic</th><th>q_pic</th><th>group</th></tr></thead>"
    )
    rows = []
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        share_pic = _pct_cell(r["frac_wb_pic"]) if has_wb else ""
        share_aud = _pct_cell(r["frac_wb_aud"]) if has_wb else ""
        rows.append(
            "<tr><td>{rank}</td>"
            "<td class='text{rcls}'>{reg}</td><td>{nch}</td>"
            "{pv}{spic}{av}{saud}"
            "<td>{pp:.4f}</td><td>{qp:.4f}</td>"
            "<td class='text'>{grp}</td></tr>".format(
                rank=rank,
                rcls=" top1" if rank == 1 else (" top2" if rank == 2 else ""),
                reg=r["region"], nch=int(r["n_channels"]),
                pv=_delta_cell(r["perm_imp_pic"], rank), spic=share_pic,
                av=_delta_cell(r["perm_imp_aud"]), saud=share_aud,
                pp=r["p_pic"], qp=r["q_pic"],
                grp=_GRP_TXT.get(str(r["group"]), str(r["group"])),
            )
        )
    return "<table class='results'>{}<tbody>{}</tbody></table>".format(thead, "".join(rows))


def _region_overview(region_df: pd.DataFrame) -> str:
    """Cross-patient summary: strongest region knockout and its share of the
    whole-brain ceiling (the total accuracy the model attributes to the neural
    data), for each patient and both tasks."""
    has_wb = "wb_imp_pic" in region_df.columns
    rows = []
    for pat in sorted(region_df["patient"].unique()):
        rp = region_df[region_df["patient"] == pat]
        best = rp.loc[rp["perm_imp_pic"].idxmax()]
        n_sig = int((rp["group"] != "neither").sum())
        if has_wb:
            wbp = float(rp["wb_imp_pic"].iloc[0])
            wba = float(rp["wb_imp_aud"].iloc[0])
            share = best["perm_imp_pic"] / wbp if abs(wbp) > 1e-9 else float("nan")
            extra = "{sh}{wp}{wa}".format(
                sh=_pct_cell(share), wp=_delta_cell(wbp), wa=_delta_cell(wba))
        else:
            extra = ""
        rows.append(
            "<tr><td class='text'>{pat}</td><td>{nr}</td>"
            "<td class='text top1'>{reg}</td><td>{nch}</td>{rv}{extra}"
            "<td>{ns}</td></tr>".format(
                pat=pat, nr=len(rp), reg=best["region"],
                nch=int(best["n_channels"]),
                rv=_delta_cell(best["perm_imp_pic"]), extra=extra, ns=n_sig,
            )
        )
    thead = (
        "<thead><tr><th>Patient</th><th>N&nbsp;regions</th>"
        "<th>Top&nbsp;region&nbsp;(pic)</th><th>N&nbsp;ch</th>"
        "<th>&#916;acc&nbsp;region</th>"
        + ("<th>share&nbsp;of&nbsp;ceiling</th>"
           "<th>whole-brain&nbsp;pic</th><th>whole-brain&nbsp;aud</th>" if has_wb else "")
        + "<th>sig&nbsp;regions</th></tr></thead>"
    )
    return "<table class='results'>{}<tbody>{}</tbody></table>".format(thead, "".join(rows))


def section_regions(region_df: pd.DataFrame, metric_tag: str,
                    sig_label: str, top_n: int = 12) -> str:
    """Region-level permutation importance: knock out whole anatomical regions
    (primary_roi) rather than single channels, read against the whole-brain
    ceiling (total accuracy the model attributes to the neural data)."""
    has_wb = "wb_imp_pic" in region_df.columns
    intro = (
        "<div class='box'>"
        "<b>Why regions.</b>&nbsp;"
        "ECoG semantic information is encoded at the <i>population</i> level and is "
        "redundant across nearby electrodes, so shuffling any one channel barely dents "
        "accuracy (hence almost every channel lands in <i>neither</i> above). Here the "
        "<b>same permutation test</b> is applied to whole brain regions: every channel "
        "assigned to a region (<code>primary_roi</code> in <code>{PAT}_*channels.pkl</code>) "
        "has its full history block shuffled <i>together</i>, so &#916;acc measures the drop "
        "when that entire region is removed from the pooled kernel-PLS model. Regions are "
        "knocked out on each task's held-out test set, bootstrapped, with the same "
        "per-bootstrap label-shuffle null and BH-FDR correction as the channel analysis "
        "(the region null is pooled across regions)."
        "</div>"
    )
    contrast = (
        "<div class='qbox'>"
        "<b>Reading it against the whole-brain ceiling.</b>&nbsp;"
        "The <b>whole-brain</b> &#916;acc (all channels shuffled) is the <i>total</i> accuracy "
        "the model attributes to the neural data &mdash; the ceiling on what any region can "
        "contribute. Each region's <b>share</b> is its &#916;acc as a fraction of that ceiling. "
        "This matters most for <b>auditory</b>: its ceiling is small (the pooled model decodes "
        "auditory only a little above chance, on far fewer trials), so a region can hold a large "
        "<i>share</i> of the auditory signal while its absolute &#916;acc still looks tiny. "
        "Shares need not sum to 100% &mdash; with redundant or synergistic coding, separate "
        "region knockouts can over- or under-count the whole. <code>sig regions</code> counts "
        "regions significant at " + sig_label + "."
        "</div>"
    )
    if not has_wb:
        contrast = (
            "<div class='qbox'><b>Note.</b> This region CSV predates the whole-brain "
            "ceiling; re-run <code>cross_task_channel_importance.py</code> to add it. "
            "<code>sig regions</code> counts regions significant at " + sig_label + ".</div>"
        )
    overview = _region_overview(region_df)

    per_pat = []
    for pat in sorted(region_df["patient"].unique()):
        rp = region_df[region_df["patient"] == pat].copy()
        gc = _group_counts(rp)
        group_str = ("both={both}, picture_only={po}, auditory_only={ao}, neither={ne}"
                     .format(both=gc["both"], po=gc["picture_only"],
                             ao=gc["auditory_only"], ne=gc["neither"]))
        ceil = ""
        if has_wb:
            ceil = ("<p class='subtle'>Whole-brain ceiling (total attributable "
                    "&#916;acc): picture {wp:+.3f} (p={pp:.3f}), auditory {wa:+.3f} "
                    "(p={pa:.3f}). Shares above are relative to this.</p>".format(
                        wp=float(rp["wb_imp_pic"].iloc[0]), pp=float(rp["wb_p_pic"].iloc[0]),
                        wa=float(rp["wb_imp_aud"].iloc[0]), pa=float(rp["wb_p_aud"].iloc[0])))
        per_pat.append(
            "<h3>Patient {pat} &mdash; {n} regions</h3>"
            "<div class='pat-grid'><div>{tbl}{ceil}"
            "<p class='subtle'>Groups ({sig}): {gs}</p></div>"
            "<div>{img}</div></div>".format(
                pat=pat, n=len(rp),
                tbl=_region_table(rp, top_n), ceil=ceil,
                sig=sig_label, gs=group_str,
                img=_make_region_bar(rp, metric_tag),
            )
        )

    return (
        "<h2>Brain-region permutation importance</h2>"
        + intro + overview + contrast + "\n".join(per_pat)
    )


# ---------------------------------------------------------------------------
# Cross-patient ranking table
# ---------------------------------------------------------------------------

def _candidate_type(pic: float, aud: float) -> str:
    if pic >= 0.02 and aud >= 0.02:
        return "<span class='sig'>amodal candidate</span>"
    if aud > 0 and aud > pic * 2:
        return "<span class='sig'>auditory candidate</span>"
    if pic > 0 and aud <= 0:
        return "picture-only"
    if pic > 0:
        return "picture-dominant"
    return "&mdash;"


def section_ranking(df: pd.DataFrame, top_n: int = 8) -> str:
    top = df.sort_values("perm_imp_pic", ascending=False).head(top_n)
    rows = []
    for _, r in top.iterrows():
        rows.append(
            "<tr>"
            "<td class='text'>{pat}</td>"
            "<td class='text'>{ch}</td>"
            "{pv}{av}"
            "<td>{pp:.4f}</td><td>{pa:.4f}</td>"
            "<td>{jp:.3f}</td><td>{ja:.3f}</td>"
            "<td class='text'>{ct}</td>"
            "</tr>".format(
                pat=r["patient"], ch=r["channel"],
                pv=_delta_cell(r["perm_imp_pic"]),
                av=_delta_cell(r["perm_imp_aud"]),
                pp=r["p_pic"], pa=r["p_aud"],
                jp=r["jac_sens_pic"], ja=r["jac_sens_aud"],
                ct=_candidate_type(r["perm_imp_pic"], r["perm_imp_aud"]),
            )
        )
    thead = (
        "<thead><tr>"
        "<th>Patient</th><th>Channel</th>"
        "<th>&#916;acc&nbsp;(pic)</th><th>&#916;acc&nbsp;(aud)</th>"
        "<th>p_pic&nbsp;(raw)</th><th>p_aud&nbsp;(raw)</th>"
        "<th>Jac&nbsp;(pic)</th><th>Jac&nbsp;(aud)</th>"
        "<th>Candidate&nbsp;type</th>"
        "</tr></thead>"
    )
    note = (
        "<div class='box'>"
        "Top channels across all patients ranked by picture permutation importance. "
        "Channels with substantial &#916;acc in both tasks are amodal candidates; "
        "those with aud&nbsp;&gt;&nbsp;2&times;pic are auditory candidates."
        "</div>"
    )
    return (
        "<h2>Cross-patient channel ranking</h2>"
        + note
        + "<table class='results'>{}<tbody>{}</tbody></table>".format(thead, "".join(rows))
    )


# ---------------------------------------------------------------------------
# Jacobian sensitivity ranking  (top-N per patient, one consolidated table)
# ---------------------------------------------------------------------------

def section_jacobian_ranking(df_m: pd.DataFrame, top_n: int = 10) -> str:
    """Compact grid: one column per patient, rows = rank 1..top_n, each cell the
    channel name ranked by Jacobian sensitivity (mean of picture & auditory)
    within that patient. Ranked *within* patient because Jacobian magnitude
    scales with each model/dataset and is not comparable across patients."""
    note = (
        "<div class='box'>"
        "Top {n} channels per patient ranked by <b>Jacobian sensitivity</b> "
        "(&#8214;&#8706;&#375;/&#8706;x&#8214;, mean of picture &amp; auditory) &mdash; the "
        "local responsiveness of the predicted GloVe embedding to each channel. Ranked "
        "<i>within</i> each patient (Jacobian magnitude is not comparable across patients)."
        "</div>"
    ).format(n=top_n)
    patients = sorted(df_m["patient"].unique())
    ranked = {}
    for pat in patients:
        dp = df_m[df_m["patient"] == pat].copy()
        dp["jac_overall"] = dp[["jac_sens_pic", "jac_sens_aud"]].mean(axis=1)
        ranked[pat] = (dp.sort_values("jac_overall", ascending=False)["channel"]
                       .astype(str).head(top_n).tolist())
    thead = ("<thead><tr><th>Rank</th>"
             + "".join("<th>{}</th>".format(p) for p in patients)
             + "</tr></thead>")
    rows = []
    for i in range(top_n):
        rcls = " top1" if i == 0 else (" top2" if i == 1 else "")
        cells = "".join(
            "<td class='text{}'>{}</td>".format(
                rcls, ranked[p][i] if i < len(ranked[p]) else "&mdash;")
            for p in patients
        )
        rows.append("<tr><td>{}</td>{}</tr>".format(i + 1, cells))
    return (
        "<h2>Jacobian sensitivity ranking (top {n} per patient)</h2>".format(n=top_n)
        + note
        + "<table class='results'>{}<tbody>{}</tbody></table>".format(thead, "".join(rows))
    )


# ---------------------------------------------------------------------------
# Method synthesis  (VIP + permutation Δacc + Jacobian)
# ---------------------------------------------------------------------------

def section_synthesis(df_m: pd.DataFrame, top_n: int = 15) -> str:
    """Synthesize the three importance methods: per-patient rank agreement plus
    a cross-patient consensus ranking."""
    has_vip = df_m["vip"].notna().any()
    cons = pd.concat(
        [_consensus_frame(df_m[df_m["patient"] == p].copy())
         for p in sorted(df_m["patient"].unique())],
        ignore_index=True,
    )

    intro = (
        "<div class='box'>"
        "<b>How the three methods relate.</b> They measure different things, so "
        "agreement between them is the strongest evidence a channel is real:"
        "<ul style='margin:6px 0;padding-left:18px'>"
        "<li><b>VIP</b> (plain linear PLS, all pooled trials) &mdash; global, "
        "accuracy-aligned: which channels the linear model leans on to predict GloVe. "
        "VIP&nbsp;&gt;&nbsp;1 = above average.</li>"
        "<li><b>Permutation &#916;acc</b> (kernel PLS, per task) &mdash; accuracy "
        "contribution with a significance test (BH-FDR).</li>"
        "<li><b>Jacobian</b> (kernel PLS, per task) &mdash; local sensitivity of the "
        "predicted embedding; responsiveness, not accuracy.</li>"
        "</ul>"
        "<b>Consensus</b> = mean of each channel's within-patient percentile rank on "
        "VIP, permutation (max of pic/aud) and Jacobian (mean of pic/aud). "
        "<b>Agreement</b> dots = number of methods placing the channel in its top "
        "quartile (&#9733; = all three <i>and</i> permutation-significant)."
        "</div>"
    )
    if not has_vip:
        intro += (
            "<div class='qbox'><b>Note.</b> <code>channel_pls_vip_all.csv</code> "
            "was not found, so the consensus uses permutation + Jacobian only. Run "
            "<code>cross_task_channel_importance.py --analysis pls</code> to add VIP."
            "</div>"
        )

    # (a) per-patient rank agreement
    agr_rows = []
    for pat in sorted(cons["patient"].unique()):
        dp = cons[cons["patient"] == pat]

        def _rho(a, b, d=dp):
            r = d[a].corr(d[b], method="spearman")
            return "&mdash;" if pd.isna(r) else "{:.2f}".format(r)

        n_vip1 = int((dp["vip"] >= 1).sum()) if has_vip else 0
        n_sig = int((dp["group"] != "neither").sum())
        agr_rows.append(
            "<tr><td class='text'>{pat}</td><td>{n}</td>"
            "<td>{nv}</td><td>{ns}</td>"
            "<td>{r_vp}</td><td>{r_vj}</td><td>{r_pj}</td></tr>".format(
                pat=pat, n=len(dp),
                nv=(n_vip1 if has_vip else "&mdash;"), ns=n_sig,
                r_vp=_rho("vip", "perm_overall"),
                r_vj=_rho("vip", "jac_overall"),
                r_pj=_rho("perm_overall", "jac_overall"),
            )
        )
    agr_thead = (
        "<thead><tr><th>Patient</th><th>N&nbsp;chan</th>"
        "<th>VIP&nbsp;&gt;&nbsp;1</th><th>perm&nbsp;sig.</th>"
        "<th>&#961;(VIP,&nbsp;perm)</th><th>&#961;(VIP,&nbsp;Jac)</th>"
        "<th>&#961;(perm,&nbsp;Jac)</th></tr></thead>"
    )
    agr_note = (
        "<div class='box'>Spearman rank correlation (&#961;) between methods, per "
        "patient. High &#961; means the methods broadly agree on the channel ordering; "
        "low or negative &#961; means they emphasise different channels (expected, since "
        "VIP is linear/global and the kernel methods are local/per-task).</div>"
    )
    agr_tbl = (
        "<h3>Per-patient method agreement</h3>" + agr_note
        + "<table class='results'>{}<tbody>{}</tbody></table>".format(
            agr_thead, "".join(agr_rows))
    )

    # (b) cross-patient consensus ranking
    top = (cons.dropna(subset=["consensus"])
              .sort_values("consensus", ascending=False).head(top_n))
    cons_rows = []
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        sig = str(r["group"]) != "neither"
        cons_rows.append(
            "<tr><td>{rank}</td><td class='text'>{pat}</td>"
            "<td class='text{rcls}'>{ch}</td>"
            "{vip}{pv}{av}"
            "<td class='text'>{grp}</td>"
            "<td>{jac:.3f}</td><td>{cons:.0f}%</td>"
            "<td class='text'>{badge}</td></tr>".format(
                rank=rank,
                rcls=" top1" if rank == 1 else (" top2" if rank == 2 else ""),
                pat=r["patient"], ch=r["channel"],
                vip=_vip_cell(r.get("vip")),
                pv=_delta_cell(r["perm_imp_pic"]),
                av=_delta_cell(r["perm_imp_aud"]),
                grp=_GRP_TXT.get(str(r["group"]), str(r["group"])),
                jac=r["jac_overall"], cons=r["consensus"] * 100.0,
                badge=_agree_badge(r["n_agree"], sig),
            )
        )
    cons_thead = (
        "<thead><tr><th>Rank</th><th>Patient</th><th>Channel</th>"
        "<th>VIP</th><th>&#916;acc&nbsp;(pic)</th><th>&#916;acc&nbsp;(aud)</th>"
        "<th>perm&nbsp;group</th><th>Jac</th><th>consensus</th>"
        "<th>agreement</th></tr></thead>"
    )
    cons_tbl = (
        "<h3>Cross-patient consensus ranking (top {n})</h3>"
        "<table class='results'>{thead}<tbody>{body}</tbody></table>"
    ).format(n=top_n, thead=cons_thead, body="".join(cons_rows))

    return "<h2>Method synthesis</h2>" + intro + agr_tbl + cons_tbl


# ---------------------------------------------------------------------------
# Per-patient section
# ---------------------------------------------------------------------------

def section_patient(pat: str, df_pat, slug: str, top_n: int, sig_label: str,
                    scatter_top: int = 10) -> str:
    pic_tbl = _channel_table_pic(df_pat, top_n)
    aud_suffix, aud_tbl = _channel_table_aud(df_pat, top_n)
    gc = _group_counts(df_pat)
    group_str = "both={both}, picture_only={po}, auditory_only={ao}, neither={ne}".format(
        both=gc["both"], po=gc["picture_only"], ao=gc["auditory_only"], ne=gc["neither"])
    metric_tag = slug  # e.g. "catindep"
    imp_img = _make_scatter(
        df_pat, "perm_imp_pic", "perm_imp_aud",
        "{} · permutation importance (Δ {})".format(pat, metric_tag),
        "Δ{} picture".format(metric_tag), "Δ{} auditory".format(metric_tag),
        annotate_top=scatter_top,
    )
    return (
        "<h2>Patient {pat} &mdash; {n} channels</h2>"
        "<div class='pat-grid'>"
        "<div>"
        "<h3>Top channels &mdash; picture test set</h3>{pic_tbl}"
        "<h3>Top channels &mdash; auditory test set{aud_suffix}</h3>{aud_tbl}"
        "<p class='subtle'>Groups ({sig_label}): {group_str}</p>"
        "</div>"
        "<div>"
        "<h3>Permutation importance scatter</h3>{imp_img}"
        "</div>"
        "</div>"
    ).format(
        pat=pat, n=len(df_pat),
        pic_tbl=pic_tbl,
        aud_suffix=aud_suffix,
        aud_tbl=aud_tbl,
        group_str=group_str,
        sig_label=sig_label,
        imp_img=imp_img,
    )


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

def section_interpretation() -> str:
    return (
        "<h2>Interpretation and caveats</h2>"
        "<div class='box'>"
        "<b>Picture vs auditory asymmetry.</b>&nbsp;"
        "Permutation importance is consistently larger for the picture test set than the "
        "auditory test set across all patients. Likely causes: (a) more picture-naming trials "
        "bias the pooled model toward picture variance; (b) auditory epochs are shorter/noisier "
        "after time-warping; (c) the retrieval database for auditory naming may have fewer unique "
        "words, compressing the metric range."
        "</div>"
        "<div class='box'>"
        "<b>Shank-level clustering.</b>&nbsp;"
        "In several patients the top picture channels are adjacent electrodes on the same shank. "
        "This spatial clustering strengthens the evidence for genuine cortical signals (vs. "
        "chance one-off channels) and should inform region-of-interest analysis."
        "</div>"
        "<div class='qbox'>"
        "<b>Kernel-PLS limitation &amp; why VIP is included.</b>&nbsp;"
        "The Nystroem-RBF map distributes information across 100 landmark projections, diluting "
        "single-channel permutation effects (small &#916;acc even for real channels). The "
        "<b>plain-PLS VIP</b> column sidesteps this: it is an exact, sampling-free analytic "
        "importance from a linear model trained on all pooled trials. Treat the three methods as "
        "a triangulation &mdash; VIP (linear, global, accuracy-aligned), permutation &#916;acc "
        "(kernel, accuracy, significance-tested) and Jacobian (kernel, local responsiveness)."
        "</div>"
        "<div class='box'>"
        "<b>Reading the consensus.</b>&nbsp;"
        "A high consensus score with a 3-dot (&#9733;) agreement badge means a channel ranks in "
        "the top quartile by <i>all three</i> independent methods and is permutation-significant "
        "&mdash; the highest-confidence drivers. A channel high on VIP/Jacobian but with "
        "non-significant &#916;acc is one the model responds to but whose response does not "
        "translate into measurable retrieval gain (redundant, off-manifold, or noise). "
        "VIP&nbsp;&gt;&nbsp;1 alone is only an <i>above-average</i> flag, not significance."
        "</div>"
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="HTML report for cross_task_channel_importance results")
    ap.add_argument("--in-dir", default=str(DEFAULT_IN_DIR),
                    help="Directory containing channel_importance_all.csv and per-patient subdirs")
    ap.add_argument("--out", default=None,
                    help="Output HTML path (default: <in-dir>/channel_importance_report.html)")
    ap.add_argument("--metric", default="cat_indep_bal_acc", choices=list(METRIC_SLUG),
                    help="Importance metric to report (default: cat_indep_bal_acc)")
    ap.add_argument("--top", type=int, default=5,
                    help="Top N channels to show per task per patient (default: 5)")
    ap.add_argument("--jac-top", type=int, default=10,
                    help="Top N channels per patient in the consolidated Jacobian "
                         "sensitivity ranking table (default: 10)")
    ap.add_argument("--scatter-top", type=int, default=10,
                    help="Channels to label in each permutation-importance scatter "
                         "(default: 10)")
    ap.add_argument("--significance", choices=["fdr", "raw"], default="fdr",
                    help="Significance basis for the both/pic/aud/neither grouping: "
                         "'fdr' (BH-corrected q, as the source CSV was built; default) "
                         "or 'raw' (uncorrected per-bootstrap p).")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Significance threshold applied to the chosen p/q basis (default: 0.05)")
    ap.add_argument("--data-dir", default=None,
                    help="Patient data directory containing *_channels.pkl files (default: main/data/)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    metric = args.metric
    slug = METRIC_SLUG.get(metric) or metric.replace("_", "")
    sig_label = _sig_label(args.significance, args.alpha)
    default_name = ("channel_importance_report.html" if args.significance == "fdr"
                    else "channel_importance_report_rawp.html")
    out_path = Path(args.out) if args.out else (in_dir / default_name)

    all_csv = in_dir / "channel_importance_all.csv"
    if not all_csv.exists():
        print("ERROR: not found:", all_csv)
        return 1

    df = pd.read_csv(all_csv)
    df_m = df[df["metric"] == metric].copy()
    if df_m.empty:
        available = df["metric"].unique().tolist()
        print("ERROR: no rows for metric '{}'. Available: {}".format(metric, available))
        return 1

    df_m = _regroup(df_m, args.significance, args.alpha)
    print("Significance basis: {} (alpha={}). Grouping: {}".format(
        args.significance, args.alpha,
        df_m["group"].value_counts().to_dict()))

    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
    df_m = _apply_channel_maps(df_m, data_dir)

    vip = _load_vip(in_dir, data_dir)
    df_m = _merge_vip(df_m, vip)

    region_df = _load_region(in_dir, metric, args.significance, args.alpha)
    region_html = ""
    if region_df is not None:
        region_html = section_regions(region_df, slug, sig_label)

    patients = sorted(df_m["patient"].unique())
    print("Patients: {} | metric: {} | VIP: {} | regions: {}".format(
        ", ".join(patients), metric,
        "loaded" if vip is not None else "not found (run --analysis pls)",
        "{} patients".format(region_df["patient"].nunique())
        if region_df is not None else "not found (region_importance_all.csv)"))

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    method = (
        "<div class='box'>"
        "<b>Method.</b>&nbsp;"
        "A single <b>kernel-PLS</b> model (Nystroem-RBF + PLSRegression, predicting GloVe "
        "embeddings) is trained on pooled picture- and auditory-naming trials for each patient "
        "(same model as <code>cross_task_cotrain.py</code>). Channel importance is assessed on "
        "held-out test trials using two complementary attributions over <b>N&nbsp;=&nbsp;20 "
        "bootstraps</b>:"
        "<ol style='margin:6px 0;padding-left:18px'>"
        "<li><b>Permutation importance</b> &mdash; drop in <code>{metric}</code> "
        "(&#916;acc&nbsp;=&nbsp;baseline&nbsp;&minus;&nbsp;shuffled) when a channel's entire "
        "history block is randomly permuted across trials, evaluated separately on picture and "
        "auditory test sets and averaged across bootstraps.</li>"
        "<li><b>Jacobian sensitivity</b> &mdash; mean &#8214;&#8706;&#375;/&#8706;x&#8214; "
        "back-propagated analytically through the Nystroem-RBF map and the PLS affine map, "
        "aggregated per channel. Measures sensitivity of the predicted GloVe embedding "
        "(not accuracy), reported as a cross-check.</li>"
        "</ol>"
        "A third, independent method &mdash; <b>plain-PLS VIP</b> (Variable Importance in "
        "Projection) &mdash; is computed by a linear <code>PLSRegression</code> fit on "
        "<i>all</i> pooled trials (no split), aggregated per channel over history bins. "
        "VIP&nbsp;&gt;&nbsp;1 marks above-average channels. See the <b>Method synthesis</b> "
        "section for how the three combine. "
        "Significance (permutation only): per-bootstrap p-values (each bootstrap's &#916;acc vs. "
        "that bootstrap's label-shuffle null), averaged across bootstraps, then thresholded at "
        + sig_label + ". Electrode names are pre-resolved in the source CSVs."
        "</div>"
    ).format(metric=metric)

    per_patient_html = "\n".join(
        section_patient(pat, df_m[df_m["patient"] == pat].copy(), slug, args.top,
                        sig_label, args.scatter_top)
        for pat in patients
    )

    body = (
        "<h1>Cross-task channel importance: picture &amp; auditory naming</h1>\n"
        "<p class='subtle'>Generated {gen} &middot; "
        "source: <code>{src}</code> &middot; "
        "metric: <code>{metric}</code> &middot; "
        "script: <code>cross_task_channel_importance.py</code></p>\n"
        "{method}\n"
        "{overview}\n"
        "{regions}\n"
        "{synthesis}\n"
        "{per_pat}\n"
        "{ranking}\n"
        "{jac_ranking}\n"
        "{interp}\n"
        "<p class='subtle' style='margin-top:32px'>"
        "CSV source: <code>{all_csv}</code>. "
        "DR/RB integer channel indices resolved from picture-naming dataframe pkls; "
        "AZ/LH/WBH <code>ch{{N}}</code> labels resolved from <code>semantic_regression_results.pkl</code>."
        "</p>\n"
    ).format(
        gen=generated, src=str(in_dir), metric=metric,
        method=method,
        overview=section_overview(df_m, sig_label),
        regions=region_html,
        synthesis=section_synthesis(df_m, top_n=15),
        per_pat=per_patient_html,
        ranking=section_ranking(df_m),
        jac_ranking=section_jacobian_ranking(df_m, top_n=args.jac_top),
        interp=section_interpretation(),
        all_csv=all_csv,
    )

    html = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>\n"
        + CSS + "\n"
        + "<title>Channel importance &mdash; cross-task co-training</title>"
        + "</head><body>\n"
        + body
        + "</body></html>\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print("Wrote:", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
