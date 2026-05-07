# -*- coding: utf-8 -*-
"""
report/cross_task_regression_report.py
======================================
Aggregate per-patient outputs from `tests/cross_task_regression.py` into a
single self-contained HTML report.

Inputs (default):
    semantic_regression_figures/cross_task_regression/<patient>/{
        peaks.csv, alignment_metrics.csv, principal_angles.csv,
        cca_canonical_correlations.csv, cross_task_accuracy.csv,
        projection_2d_trials.csv,
        peak_traces.png, principal_angles.png, quiver_align.png,
        scatter_2d.png, cross_task_bars.png
    }
    semantic_regression_figures/cross_task_regression/cross_patient_summary.csv

Output:
    semantic_regression_figures/cross_task_regression/cross_task_regression_report.html

Usage:
    python -m main.report.cross_task_regression_report
    python -m main.report.cross_task_regression_report --in-dir <dir> --out report.html
"""

from __future__ import annotations
import argparse
import base64
import io
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN_DIR = PROJECT_ROOT / "semantic_regression_figures" / "cross_task_regression"


def img_to_base64(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def fig_to_base64(fig, dpi=140) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def collect_summary(in_dir: Path) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
    """Load cross_patient_summary.csv plus per-patient CSVs."""
    summary_path = in_dir / "cross_patient_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    per_pat: dict[str, dict[str, pd.DataFrame]] = {}
    for pat_dir in sorted([p for p in in_dir.iterdir() if p.is_dir()]):
        pat = pat_dir.name
        per_pat[pat] = {}
        for f in ["peaks.csv", "alignment_metrics.csv", "principal_angles.csv",
                  "cca_canonical_correlations.csv", "cross_task_accuracy.csv"]:
            p = pat_dir / f
            if p.exists():
                per_pat[pat][f] = pd.read_csv(p)
    return summary, per_pat


# ── Cross-patient summary figures ────────────────────────────────────────
def fig_alignment_overview(summary: pd.DataFrame) -> str:
    if summary.empty: return ""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    pats = summary["patient"].tolist()
    x = np.arange(len(pats))
    axes[0].bar(x - 0.2, summary["alignment_index"], 0.4, color="#1f77b4", label="alignment idx")
    axes[0].bar(x + 0.2, summary["first_canon_corr"], 0.4, color="#ff7f0e", label="first canon corr")
    axes[0].set_xticks(x); axes[0].set_xticklabels(pats)
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("score")
    axes[0].set_title("Subspace alignment summary"); axes[0].legend(fontsize=8)

    axes[1].bar(x - 0.2, summary["pic_peak_acc"], 0.4, color="#1f77b4", label="picture peak")
    axes[1].bar(x + 0.2, summary["aud_peak_acc"], 0.4, color="#d62728", label="auditory peak")
    axes[1].set_xticks(x); axes[1].set_xticklabels(pats)
    axes[1].set_ylabel("category_balanced_acc"); axes[1].set_title("Per-task peak loose-category accuracy")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig_to_base64(fig)


def fig_cross_task(summary: pd.DataFrame) -> str:
    if summary.empty: return ""
    fig, ax = plt.subplots(figsize=(9, 4))
    pats = summary["patient"].tolist()
    x = np.arange(len(pats))
    ax.bar(x - 0.30, summary["within_pic_holdout"], 0.18, color="#9ecae1", label="pic holdout")
    ax.bar(x - 0.10, summary["cross_aud_to_pic"], 0.18, color="#1f77b4", label="aud→pic")
    ax.bar(x + 0.10, summary["within_aud_holdout"], 0.18, color="#fdae6b", label="aud holdout")
    ax.bar(x + 0.30, summary["cross_pic_to_aud"], 0.18, color="#d62728", label="pic→aud")
    ax.set_xticks(x); ax.set_xticklabels(pats)
    ax.set_ylabel("category_balanced_acc")
    ax.set_title("Within- (holdout) vs cross-task category retrieval at peak bins")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig_to_base64(fig)


def fig_principal_angles_stack(per_pat: dict) -> str:
    rows = []
    for pat, dfs in per_pat.items():
        if "principal_angles.csv" in dfs:
            d = dfs["principal_angles.csv"].copy()
            d["patient"] = pat
            rows.append(d)
    if not rows: return ""
    df = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    pats = sorted(df["patient"].unique())
    width = 0.8 / max(1, len(pats))
    for i, pat in enumerate(pats):
        sub = df[df["patient"] == pat]
        x = sub["dim"].values + (i - len(pats)/2 + 0.5) * width
        ax.bar(x, sub["principal_angle_deg"].values, width=width, label=pat, alpha=0.85)
    ax.axhline(45, ls=":", color="grey", lw=1)
    ax.set_xlabel("PLS dimension"); ax.set_ylabel("Principal angle (deg)")
    ax.set_title("Principal angles between picture & auditory PLS subspaces")
    ax.set_ylim(0, 95); ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    return fig_to_base64(fig)


def fig_canon_corr_stack(per_pat: dict) -> str:
    rows = []
    for pat, dfs in per_pat.items():
        if "cca_canonical_correlations.csv" in dfs:
            d = dfs["cca_canonical_correlations.csv"].copy()
            d["patient"] = pat
            rows.append(d)
    if not rows: return ""
    df = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for pat in sorted(df["patient"].unique()):
        sub = df[df["patient"] == pat]
        ax.plot(sub["dim"].values, sub["canon_corr"].values, marker="o", label=pat)
    ax.set_xlabel("Canonical dimension"); ax.set_ylabel("Canonical correlation")
    ax.set_title("CCA canonical correlations (matched-word averaged)")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    return fig_to_base64(fig)


# ── HTML rendering ───────────────────────────────────────────────────────
CSS = """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif;
       max-width: 1200px; margin: 28px auto; padding: 0 20px; color: #1a1a1a; line-height: 1.45; }
h1 { color: #1565C0; border-bottom: 2px solid #1565C0; padding-bottom: 8px; }
h2 { color: #0D47A1; margin-top: 36px; border-bottom: 1px solid #BBDEFB; padding-bottom: 4px; }
h3 { color: #424242; margin-top: 22px; }
table { border-collapse: collapse; margin: 10px 0; font-size: 13px; }
th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: right; }
th { background: #ECEFF1; font-weight: 600; }
td.text { text-align: left; }
.subtle { color: #757575; font-size: 12px; }
img { max-width: 100%; border: 1px solid #e0e0e0; padding: 4px; background: white; margin: 6px 0; }
.figrow { display: flex; flex-wrap: wrap; gap: 12px; }
.figrow > div { flex: 1 1 320px; min-width: 320px; }
.box { background: #F5F7FA; padding: 10px 14px; border-left: 3px solid #1565C0; margin: 12px 0; }
.delta-pos { color: #2E7D32; font-weight: 600; }
.delta-neg { color: #C62828; font-weight: 600; }
</style>
"""


def df_to_html(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    if df is None or df.empty: return "<p class='subtle'>(no data)</p>"
    fmts = {c: float_fmt.format for c in df.select_dtypes(include="float").columns}
    return df.to_html(index=False, formatters=fmts, classes="results")


def section_cross_patient(summary: pd.DataFrame, per_pat: dict) -> str:
    align_b64 = fig_alignment_overview(summary)
    cross_b64 = fig_cross_task(summary)
    angles_b64 = fig_principal_angles_stack(per_pat)
    cc_b64 = fig_canon_corr_stack(per_pat)

    desc_table = ""
    if not summary.empty:
        cols = ["patient", "embedding", "pic_peak_bin", "pic_peak_acc",
                "aud_peak_bin", "aud_peak_acc", "n_shared_words",
                "alignment_index", "grassmann_distance",
                "first_canon_corr", "mean_canon_corr",
                "within_pic_holdout", "within_aud_holdout",
                "cross_pic_to_aud", "cross_aud_to_pic"]
        sub = summary[[c for c in cols if c in summary.columns]].copy()
        desc_table = df_to_html(sub)

    return f"""
    <h2>Cross-patient summary</h2>
    <div class="box">
      <b>What this section answers:</b> across the 6 shared patients, how aligned
      are the per-task PLS subspaces, and how well does cross-task decoding hold up?
      Higher alignment index (toward 1) and higher first canonical correlation indicate
      that picture-naming and auditory-naming PLS geometries share a common semantic axis.
      Cross-task accuracy near within-task accuracy indicates a shared representation.
    </div>
    <h3>Summary table</h3>
    {desc_table}
    <h3>Alignment overview & per-task peak accuracy</h3>
    <img src="data:image/png;base64,{align_b64}" alt="alignment overview">
    <h3>Cross-task category retrieval</h3>
    <img src="data:image/png;base64,{cross_b64}" alt="cross-task bars">
    <h3>Principal angles per patient</h3>
    <img src="data:image/png;base64,{angles_b64}" alt="principal angles">
    <h3>CCA canonical correlations per patient</h3>
    <img src="data:image/png;base64,{cc_b64}" alt="canonical correlations">
    """


def section_per_patient(in_dir: Path, summary: pd.DataFrame, per_pat: dict) -> str:
    blocks = []
    for pat in sorted(per_pat.keys()):
        pdir = in_dir / pat
        peaks = per_pat[pat].get("peaks.csv")
        align = per_pat[pat].get("alignment_metrics.csv")
        cross = per_pat[pat].get("cross_task_accuracy.csv")

        # Load images
        peak_b = img_to_base64(pdir / "peak_traces.png")
        ang_b = img_to_base64(pdir / "principal_angles.png")
        quiv_b = img_to_base64(pdir / "quiver_align.png")
        scat_b = img_to_base64(pdir / "scatter_2d.png")
        bars_b = img_to_base64(pdir / "cross_task_bars.png")

        blocks.append(f"""
        <h2>Patient {pat}</h2>
        <h3>Peaks &amp; alignment</h3>
        <div class="figrow">
          <div>{df_to_html(peaks)}</div>
          <div>{df_to_html(align)}</div>
        </div>
        <h3>Loose-category retrieval traces</h3>
        <img src="data:image/png;base64,{peak_b}" alt="peak traces">
        <div class="figrow">
          <div>
            <h3>Principal angles</h3>
            <img src="data:image/png;base64,{ang_b}" alt="principal angles">
          </div>
          <div>
            <h3>PLS axes mapped through CCA (quiver)</h3>
            <img src="data:image/png;base64,{quiv_b}" alt="quiver">
          </div>
        </div>
        <h3>Trial-level co-projection (pre vs post CCA)</h3>
        <img src="data:image/png;base64,{scat_b}" alt="scatter 2d">
        <h3>Cross-task category retrieval</h3>
        <div class="figrow">
          <div>{df_to_html(cross)}</div>
          <div><img src="data:image/png;base64,{bars_b}" alt="cross-task bars"></div>
        </div>
        """)
    return "\n".join(blocks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default=str(DEFAULT_IN_DIR))
    p.add_argument("--out", default=None)
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out) if args.out else (in_dir / "cross_task_regression_report.html")
    summary, per_pat = collect_summary(in_dir)

    body = []
    body.append(f"<h1>Cross-task semantic regression: picture vs auditory naming</h1>")
    body.append(f"<p class='subtle'>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} "
                f"&middot; source: <code>{in_dir}</code></p>")
    body.append("""
    <div class='box'>
    <b>Method.</b> For each patient we (1) located the peak loose-semantic-category retrieval
    bin (<code>category_balanced_acc</code>) per task, (2) trained a fresh kernel-PLS regressor
    (Nystroem-RBF → PLS, 10 components) at each task's peak bin on ALL trials, (3) compared
    the resulting 10D projection geometries via principal-angles &amp; CCA on word-averaged
    score matrices restricted to words shared across tasks, (4) co-projected trial-level PLS
    scores into a 2D CCA space (with a pre-CCA PCA panel for reference), and (5) measured
    cross-task category decoding by applying each task's PLS pipeline to the other task's
    trials at its own peak bin.
    </div>
    """)

    body.append(section_cross_patient(summary, per_pat))
    body.append(section_per_patient(in_dir, summary, per_pat))

    html = "<!DOCTYPE html><html><head><meta charset='utf-8'>" + CSS + \
           "<title>Cross-task semantic regression</title></head><body>" + \
           "\n".join(body) + "</body></html>"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
