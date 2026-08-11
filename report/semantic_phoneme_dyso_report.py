# -*- coding: utf-8 -*-
"""
report/semantic_phoneme_dyso_report.py
======================================
Aggregate per-patient outputs from tests/semantic_phoneme_dyso.py into a
single HTML report.

Inputs (default):
    tests/results/semantic_phoneme_dyso/<patient>/{
        per_bin_metrics.csv,
        projections_peak.pkl,
        figures/dyso_traces.png,
        figures/scatter_3d.png,
        figures/quiver.png,
    }
    tests/results/semantic_phoneme_dyso/cross_patient_metrics.csv

Output: tests/results/semantic_phoneme_dyso/semantic_phoneme_dyso_report.html
"""

from __future__ import annotations
import argparse
import base64
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.paths import results_dir

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIN_DIR     = Path(__file__).resolve().parents[1]   # …/main/
# Was MAIN_DIR/"tests"/"results"/"semantic_phoneme_dyso" -- a root the 2026-07
# reorganisation deleted, so this default could never have resolved.
# NB the correct destination, results/dyso_dissociation/semantic_phoneme/, was itself
# pruned on 2026-08-10 (approved, unreferenced, 13.4 MB). This report therefore has no
# input on disk until that suite is re-run. It will now fail with a clear missing-path
# error instead of silently pointing at a directory that never existed.
DEFAULT_IN   = results_dir("dyso_dissociation", "semantic_phoneme", create=False)

CSS = """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif;
       max-width: 1200px; margin: 24px auto; padding: 0 18px; color: #1a1a1a; }
h1 { color: #1565C0; border-bottom: 2px solid #1565C0; padding-bottom: 8px; }
h2 { color: #0D47A1; margin-top: 32px; border-bottom: 1px solid #BBDEFB; padding-bottom: 4px; }
h3 { color: #424242; margin-top: 18px; }
table { border-collapse: collapse; margin: 8px 0; font-size: 13px; }
th, td { border: 1px solid #ccc; padding: 6px 9px; text-align: right; }
th { background: #ECEFF1; }
.box { background: #F5F7FA; padding: 10px 14px; border-left: 3px solid #1565C0; margin: 12px 0; }
.subtle { color: #757575; font-size: 12px; }
img { max-width: 100%; border: 1px solid #e0e0e0; padding: 4px; background: white; margin: 6px 0; }
.figrow { display: flex; flex-wrap: wrap; gap: 12px; }
.figrow > div { flex: 1 1 360px; min-width: 360px; }
.diag { background: #E8F5E9; }      /* diagonal cells (good) */
.offdiag { background: #FFF3E0; }   /* off-diagonal (leakage) */
</style>
"""


def img_b64(path: Path) -> str:
    if not path.exists(): return ""
    return base64.b64encode(path.read_bytes()).decode()


def fig_b64(fig, dpi=140) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def load_all(in_dir: Path):
    summary = pd.DataFrame()
    p = in_dir / "cross_patient_metrics.csv"
    if p.exists():
        summary = pd.read_csv(p)
    per_pat = {}
    for d in sorted([x for x in in_dir.iterdir() if x.is_dir()]):
        pat = d.name
        bins_csv = d / "per_bin_metrics.csv"
        per_pat[pat] = {}
        if bins_csv.exists():
            per_pat[pat]["bins"] = pd.read_csv(bins_csv)
        for fname in ("figures/dyso_traces.png", "figures/scatter_3d.png", "figures/quiver.png"):
            f = d / fname
            if f.exists():
                per_pat[pat][fname] = f
    return summary, per_pat


# ── Cross-patient figures ────────────────────────────────────────────────
def fig_peak_r2_bar(per_pat: dict) -> str:
    """For each patient, plot the peak R² on diagonal and off-diagonal cells."""
    rows = []
    for pat, info in per_pat.items():
        df = info.get("bins")
        if df is None or df.empty: continue
        joint = (df["R2_S_on_sem"].values + df["R2_P_on_phon"].values) / 2
        peak = int(np.argmax(joint))
        rows.append({
            "patient": pat,
            "R2_S|sem": df["R2_S_on_sem"].iloc[peak],
            "R2_P|phon": df["R2_P_on_phon"].iloc[peak],
            "R2_S|phon (leak)": df["R2_S_on_phon"].iloc[peak],
            "R2_P|sem (leak)": df["R2_P_on_sem"].iloc[peak],
            "R2_S|shared": df["R2_S_on_shared"].iloc[peak],
            "R2_P|shared": df["R2_P_on_shared"].iloc[peak],
        })
    if not rows: return ""
    sdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    n_pats = len(sdf)
    x = np.arange(n_pats)
    cols = ["R2_S|sem","R2_P|phon","R2_S|phon (leak)","R2_P|sem (leak)",
            "R2_S|shared","R2_P|shared"]
    colors = ["#1f77b4","#d62728","#9ecae1","#fdae6b","#7f7f7f","#bdbdbd"]
    w = 0.13
    for i, (c, col) in enumerate(zip(cols, colors)):
        ax.bar(x + (i - len(cols)/2 + 0.5) * w, sdf[c].values, width=w,
               color=col, label=c)
    ax.set_xticks(x); ax.set_xticklabels(sdf["patient"].values)
    ax.set_ylabel("R² (CV at peak bin)")
    ax.set_title("Per-patient peak-bin R²: diagonal (private) vs off-diagonal (leakage)")
    ax.axhline(0, color="grey", lw=0.5)
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    fig.tight_layout(); return fig_b64(fig)


def fig_traces_stack(per_pat: dict) -> str:
    """Per-patient time courses, stacked rows. R²(S|sem) and R²(P|phon) only."""
    pats = [p for p, info in per_pat.items() if "bins" in info]
    if not pats: return ""
    fig, axes = plt.subplots(len(pats), 1, figsize=(11, 1.7 * len(pats)),
                              sharex=True, squeeze=False)
    for ax, pat in zip(axes[:, 0], pats):
        df = per_pat[pat]["bins"]
        t = df["bin_index"].values
        ax.plot(t, df["R2_S_on_sem"].values, color="#1f77b4", label="R²(S|sem)")
        ax.plot(t, df["R2_P_on_phon"].values, color="#d62728", label="R²(P|phon)")
        ax.plot(t, df["R2_S_on_phon"].values, color="#9ecae1", lw=0.8, ls="--", label="R²(S|phon) leak")
        ax.plot(t, df["R2_P_on_sem"].values, color="#fdae6b", lw=0.8, ls="--", label="R²(P|sem) leak")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel(f"{pat}\nR²", fontsize=9)
        if pat == pats[0]: ax.legend(fontsize=7, ncol=4, loc="upper right")
    axes[-1, 0].set_xlabel("bin_index")
    fig.suptitle("Time-resolved subspace R² per patient", y=1.0, fontsize=11)
    fig.tight_layout(); return fig_b64(fig)


def fig_timing_dissoc(per_pat: dict) -> str:
    """Peak-time of semantic R² vs phoneme R² per patient."""
    rows = []
    for pat, info in per_pat.items():
        df = info.get("bins")
        if df is None or df.empty: continue
        b_sem  = int(df["bin_index"].values[np.argmax(df["R2_S_on_sem"].values)])
        b_phon = int(df["bin_index"].values[np.argmax(df["R2_P_on_phon"].values)])
        rows.append({"patient": pat, "peak_sem_bin": b_sem, "peak_phon_bin": b_phon,
                     "delta": b_phon - b_sem})
    if not rows: return ""
    sdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(sdf))
    ax.bar(x - 0.18, sdf["peak_sem_bin"], 0.36, color="#1f77b4", label="peak R²(S|sem)")
    ax.bar(x + 0.18, sdf["peak_phon_bin"], 0.36, color="#d62728", label="peak R²(P|phon)")
    ax.set_xticks(x); ax.set_xticklabels(sdf["patient"].values)
    ax.set_ylabel("bin index (peak)")
    ax.set_title("Peak-bin dissociation: semantic vs phoneme subspace")
    ax.legend(fontsize=8)
    for i, dv in enumerate(sdf["delta"].values):
        ax.text(i, max(sdf["peak_sem_bin"].iloc[i], sdf["peak_phon_bin"].iloc[i]) + 1,
                f"Δ={dv:+d}", ha="center", fontsize=8)
    fig.tight_layout(); return fig_b64(fig)


# ── HTML rendering ───────────────────────────────────────────────────────
def render_per_patient(in_dir: Path, per_pat: dict) -> str:
    blocks = []
    for pat in sorted(per_pat.keys()):
        info = per_pat[pat]
        traces_b = img_b64(info.get("figures/dyso_traces.png", Path("/")))
        scat_b   = img_b64(info.get("figures/scatter_3d.png",   Path("/")))
        quiv_b   = img_b64(info.get("figures/quiver.png",       Path("/")))
        df = info.get("bins")
        peak_html = ""
        if df is not None and len(df):
            joint = (df["R2_S_on_sem"].values + df["R2_P_on_phon"].values) / 2
            pi = int(np.argmax(joint))
            row = df.iloc[pi]
            peak_html = (
                f"<p class='subtle'>Peak bin: {int(row['bin_index'])} &nbsp;|&nbsp; "
                f"R²(S|sem)={row['R2_S_on_sem']:.3f}, "
                f"R²(P|phon)={row['R2_P_on_phon']:.3f} &nbsp;|&nbsp; "
                f"leak R²(S|phon)={row['R2_S_on_phon']:+.3f}, "
                f"R²(P|sem)={row['R2_P_on_sem']:+.3f}</p>"
            )
        blocks.append(f"""
        <h2>Patient {pat}</h2>
        {peak_html}
        <h3>Time-resolved subspace R²</h3>
        <img src='data:image/png;base64,{traces_b}'>
        <div class='figrow'>
          <div><h3>Trials in orthogonal neural space (peak bin)</h3>
               <img src='data:image/png;base64,{scat_b}'></div>
          <div><h3>Word trajectories</h3>
               <img src='data:image/png;base64,{quiv_b}'></div>
        </div>
        """)
    return "\n".join(blocks)


def render_summary_table(per_pat: dict) -> str:
    """A diagonal-vs-off-diagonal R² table to read leakage at a glance."""
    rows_html = []
    for pat in sorted(per_pat.keys()):
        df = per_pat[pat].get("bins")
        if df is None or df.empty: continue
        joint = (df["R2_S_on_sem"].values + df["R2_P_on_phon"].values) / 2
        pi = int(np.argmax(joint))
        r = df.iloc[pi]
        def cell(v, cls=""): return f"<td class='{cls}'>{v:+.3f}</td>"
        rows_html.append(
            f"<tr><td><b>{pat}</b></td>"
            f"<td>{int(r['bin_index'])}</td>"
            + cell(r['R2_S_on_sem'], "diag")
            + cell(r['R2_P_on_sem'], "offdiag")
            + cell(r['R2_S_on_shared'])
            + cell(r['R2_S_on_phon'], "offdiag")
            + cell(r['R2_P_on_phon'], "diag")
            + cell(r['R2_P_on_shared'])
            + "</tr>"
        )
    return f"""
    <table>
      <tr><th rowspan=2>Patient</th><th rowspan=2>Peak bin</th>
          <th colspan=3>Semantic embedding (S)</th>
          <th colspan=3>Phoneme embedding (P)</th></tr>
      <tr><th>via U_sem</th><th>via U_phon (leak)</th><th>via U_shared</th>
          <th>via U_phon</th><th>via U_sem (leak)</th><th>via U_shared</th></tr>
      {''.join(rows_html)}
    </table>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default=str(DEFAULT_IN))
    p.add_argument("--out", default=None)
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out) if args.out else (in_dir / "semantic_phoneme_dyso_report.html")
    summary, per_pat = load_all(in_dir)

    body = []
    body.append("<h1>Semantic vs Phoneme — DySO decomposition</h1>")
    body.append(f"<p class='subtle'>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} "
                f"&middot; source: <code>{in_dir}</code></p>")
    body.append("""
    <div class='box'>
    <b>Method.</b> Per patient and time bin, we (i) PCA both GloVe (semantic, S)
    and panphon (phoneme, P) embeddings to a common dim, (ii) run DySO on the pair to obtain
    orthonormal embedding-side bases for the semantic-private, phoneme-private,
    and shared sub-subspaces, (iii) ridge-regress neural HGA onto each
    subspace's reconstructed target and QR-orthonormalize the regression
    coefficients to get neural-axis bases U_sem, U_phon, U_shared,
    (iv) project trials onto each basis and report 5-fold cross-validated R²
    of each target against each subspace. A clean dissociation shows the
    diagonal (R²(S|U_sem), R²(P|U_phon)) high and off-diagonal terms near zero,
    with the shared subspace capturing whatever joint variance remains.
    </div>
    """)
    body.append("<h2>Cross-patient summary</h2>")
    body.append(render_summary_table(per_pat))
    body.append("<h3>Peak-bin R²: diagonal (private) vs off-diagonal (leakage)</h3>")
    body.append(f"<img src='data:image/png;base64,{fig_peak_r2_bar(per_pat)}'>")
    body.append("<h3>Time-resolved R² (per patient)</h3>")
    body.append(f"<img src='data:image/png;base64,{fig_traces_stack(per_pat)}'>")
    body.append("<h3>Peak-bin timing dissociation</h3>")
    body.append(f"<img src='data:image/png;base64,{fig_timing_dissoc(per_pat)}'>")

    body.append(render_per_patient(in_dir, per_pat))

    html = "<!DOCTYPE html><html><head><meta charset='utf-8'>" + CSS + \
           "<title>Semantic vs Phoneme DySO</title></head><body>" + \
           "\n".join(body) + "</body></html>"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
