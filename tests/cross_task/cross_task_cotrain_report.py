# cross_task_cotrain_report.py
# HTML report from cross_task_cotrain.py CSV outputs.  Same visual style as
# cross_task_transfer_report.py.
#
# Sections:
#   1. Cross-patient overview -- word_bal_acc / cosine per (eval-target x train-source)
#      grouped by patient, framed around the three questions.
#   2. Q1 (shared representation): cross-vs-within retention + RSA of per-word
#      neural geometry across tasks.
#   3. Q3 (one decoder for both): pooled-vs-within gain.
#   4. Q2 (amodal electrodes): per-electrode RSA(pic) vs RSA(aud), top channels.
#   5. Per-patient detail -- all metrics x (target,source), seen/unseen split,
#      paired Wilcoxon, and the static figures saved by the pipeline.
#
# Inputs (default: main/tests/results/cross_task_cotrain/):
#   cotrain_conditions_summary.csv, cotrain_rsa_summary.csv
#   <patient>/cotrain_conditions_<patient>.csv  (per-bootstrap rows)
#   <patient>/cotrain_electrodes_<patient>.csv
#   <patient>/cotrain_<patient>_bars.png, cotrain_<patient>_electrodes.png
#
# Output:
#   main/tests/results/cross_task_cotrain/cross_task_cotrain_report.html
#
# Usage:
#   python -m main.tests.cross_task.cross_task_cotrain_report
#   python -m main.tests.cross_task.cross_task_cotrain_report --in-dir <dir> --model kernel_pls

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_THIS_DIR)
_MAIN_DIR = os.path.dirname(_TESTS_DIR)
DEFAULT_IN_DIR = Path(_MAIN_DIR) / "tests" / "results" / "cross_task_cotrain"

# ---------------------------------------------------------------------------
# Conditions organised by EVALUATION TARGET x TRAIN SOURCE
# ---------------------------------------------------------------------------
# For each target task we compare three training sources:
#   within  = trained on the same task           (per-task ceiling)
#   cross   = trained on the OTHER task           (Q1: shared representation?)
#   pooled  = trained on BOTH tasks pooled        (Q3: one decoder for both?)
TARGETS = ["pic", "aud"]
TARGET_LABELS = {"pic": "Picture-naming test trials",
                 "aud": "Auditory-naming test trials"}
SRC_ORDER = ["within", "cross", "pooled"]
SRC_LABELS = {"within": "Within (same task)",
              "cross": "Cross (other task)",
              "pooled": "Pooled (both tasks)"}
SRC_COLORS = {"within": "#7f7f7f", "cross": "#d62728", "pooled": "#1f77b4"}
# (target, source) -> condition name in the CSVs
COND = {
    ("pic", "within"): "within_pic", ("pic", "cross"): "cross_a2p", ("pic", "pooled"): "pooled_pic",
    ("aud", "within"): "within_aud", ("aud", "cross"): "cross_p2a", ("aud", "pooled"): "pooled_aud",
}

METRICS = [
    ("word_bal_acc",      "Word balanced accuracy"),
    ("cat_indep_bal_acc", "Category-independent balanced accuracy"),
    ("cosine_mean",       "Cosine similarity"),
]


# ---------------------------------------------------------------------------
# Figure / HTML helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig, dpi: int = 130) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img_tag(fig, alt: str = "", dpi: int = 130) -> str:
    return '<img alt="{}" src="data:image/png;base64,{}" />'.format(alt, _fig_to_b64(fig, dpi))


def _file_img_tag(path: Path, alt: str = "") -> str:
    if not path.exists():
        return "<p class='subtle'>(figure not found: {})</p>".format(path.name)
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return '<img alt="{}" src="data:image/png;base64,{}" />'.format(alt, b64)


CSS = """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif;
       max-width: 1280px; margin: 28px auto; padding: 0 20px; color: #1a1a1a; line-height: 1.45; }
h1 { color: #1565C0; border-bottom: 2px solid #1565C0; padding-bottom: 8px; }
h2 { color: #0D47A1; margin-top: 36px; border-bottom: 1px solid #BBDEFB; padding-bottom: 4px; }
h3 { color: #424242; margin-top: 22px; }
table.results { border-collapse: collapse; margin: 10px 0; font-size: 12px; width: auto; }
table.results th, table.results td { border: 1px solid #ccc; padding: 5px 9px; text-align: right; }
table.results th { background: #ECEFF1; font-weight: 600; text-align: center; }
table.results td.text { text-align: left; }
.subtle { color: #757575; font-size: 12px; }
img { max-width: 100%; border: 1px solid #e0e0e0; padding: 4px; background: white; margin: 6px 0; }
details { margin: 8px 0; }
summary { cursor: pointer; color: #1565C0; font-size: 13px; font-weight: 500; }
summary:hover { text-decoration: underline; }
.box { background: #F5F7FA; padding: 10px 14px; border-left: 3px solid #1565C0; margin: 12px 0; font-size: 13px; }
.qbox { background: #FFF8E1; padding: 10px 14px; border-left: 3px solid #F9A825; margin: 12px 0; font-size: 13px; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 8px 0 14px 0; font-size: 13px; }
.swatch { display: inline-block; width: 14px; height: 14px; border-radius: 3px; vertical-align: middle; margin-right: 4px; }
.figrow { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
.figrow > div { flex: 1 1 480px; }
.sig { color: #2E7D32; font-weight: 600; }
.ns  { color: #9E9E9E; }
</style>
"""


def _legend_html() -> str:
    items = "".join(
        "<span><span class='swatch' style='background:{}'></span>{}</span>".format(
            SRC_COLORS[s], SRC_LABELS[s]) for s in SRC_ORDER)
    return "<div class='legend'>{}</div>".format(items)


def _df_to_html(df, float_fmt: str = "{:.3f}") -> str:
    if df is None or df.empty:
        return "<p class='subtle'>(no data)</p>"
    fmts = {c: float_fmt.format for c in df.select_dtypes(include="float").columns}
    return df.to_html(index=False, formatters=fmts, classes="results")


def _highlight_p(val) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return str(val)
    return "<b class='sig'>{:.4f}</b>".format(v) if v < 0.05 else "<span class='ns'>{:.4f}</span>".format(v)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all(in_dir: Path):
    per_pat, elec = {}, {}
    for pat_dir in sorted(p for p in in_dir.iterdir() if p.is_dir()):
        pat = pat_dir.name
        c = pat_dir / "cotrain_conditions_{}.csv".format(pat)
        e = pat_dir / "cotrain_electrodes_{}.csv".format(pat)
        if c.exists():
            per_pat[pat] = pd.read_csv(c)
        if e.exists():
            elec[pat] = pd.read_csv(e)
    rsa_path = in_dir / "cotrain_rsa_summary.csv"
    rsa = pd.read_csv(rsa_path) if rsa_path.exists() else pd.DataFrame()
    return per_pat, elec, rsa


def _vals(df, condition, metric):
    return df[df["condition"] == condition][metric].dropna().values


# ---------------------------------------------------------------------------
# Overview figures
# ---------------------------------------------------------------------------

def fig_overview(per_pat: dict, metric: str, metric_label: str) -> str:
    patients = sorted(per_pat)
    n_pat = len(patients)
    fig, axes = plt.subplots(1, 2, figsize=(max(9, 2.7 * n_pat), 4.6), sharey=True)
    for ax, target in zip(axes, TARGETS):
        x = np.arange(n_pat)
        width = 0.8 / len(SRC_ORDER)
        for i, src in enumerate(SRC_ORDER):
            cond = COND[(target, src)]
            means = [float(np.mean(_vals(per_pat[p], cond, metric))) if len(_vals(per_pat[p], cond, metric)) else np.nan for p in patients]
            sems = [float(np.std(_vals(per_pat[p], cond, metric)) / np.sqrt(max(1, len(_vals(per_pat[p], cond, metric))))) for p in patients]
            offset = (i - len(SRC_ORDER) / 2 + 0.5) * width
            ax.bar(x + offset, np.nan_to_num(means), width, yerr=sems,
                   color=SRC_COLORS[src], alpha=0.85, label=SRC_LABELS[src],
                   capsize=3, error_kw={"lw": 1.1})
        ax.set_xticks(x); ax.set_xticklabels(patients, fontsize=9)
        ax.set_xlabel("Patient"); ax.set_ylabel(metric_label)
        ax.set_title(TARGET_LABELS[target], fontsize=10)
        ax.set_ylim(0, None); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("{} per patient (train source within each target task)".format(metric_label), fontsize=11)
    fig.tight_layout()
    return _img_tag(fig, alt="overview_" + metric)


def fig_delta(per_pat: dict, src: str, ref: str, title: str,
              metric: str = "word_bal_acc") -> str:
    """Delta of *src* minus *ref* condition for *metric*, per patient x target.
    Used for cross-vs-within (Q1) and pooled-vs-within (Q3)."""
    patients = sorted(per_pat)
    n_pat = len(patients)
    fig, axes = plt.subplots(1, 2, figsize=(max(9, 2.7 * n_pat), 4.2), sharey=True)
    for ax, target in zip(axes, TARGETS):
        x = np.arange(n_pat)
        deltas = []
        for p in patients:
            a = _vals(per_pat[p], COND[(target, src)], metric)
            b = _vals(per_pat[p], COND[(target, ref)], metric)
            deltas.append(float(np.mean(a) - np.mean(b)) if len(a) and len(b) else np.nan)
        ax.bar(x, np.nan_to_num(deltas), 0.6, color=SRC_COLORS[src], alpha=0.85)
        for xi, d in zip(x, deltas):
            if not np.isnan(d):
                ax.text(xi, d + (0.003 if d >= 0 else -0.012), "{:+.3f}".format(d),
                        ha="center", fontsize=7, color="#333")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(patients, fontsize=9)
        ax.set_xlabel("Patient"); ax.set_ylabel("Δ " + metric)
        ax.set_title(TARGET_LABELS[target], fontsize=10); ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=11); fig.tight_layout()
    return _img_tag(fig, alt="delta_{}_{}".format(src, metric))


def fig_rsa(rsa: pd.DataFrame) -> str:
    if rsa.empty:
        return "<p class='subtle'>(no RSA data)</p>"
    rsa = rsa.sort_values("patient")
    patients = rsa["patient"].tolist()
    x = np.arange(len(patients)); width = 0.27
    cols = [("rdm_pic_vs_aud", "pic vs aud (neural)", "#6a3d9a"),
            ("rdm_pic_vs_glove", "pic vs GloVe", "#1f77b4"),
            ("rdm_aud_vs_glove", "aud vs GloVe", "#ff7f0e")]
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(patients)), 4.2))
    for i, (c, lab, col) in enumerate(cols):
        ax.bar(x + (i - 1) * width, rsa[c].values, width, color=col, alpha=0.85, label=lab)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(patients); ax.set_ylabel("Spearman RDM correlation")
    ax.set_title("Cross-task RSA of per-word geometry (Spearman of RDMs)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _img_tag(fig, alt="rsa")


def fig_electrodes_overview(elec: dict) -> str:
    """Pooled scatter of RSA(pic) vs RSA(aud) for all electrodes, all patients."""
    if not elec:
        return "<p class='subtle'>(no electrode data)</p>"
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for pat, df in sorted(elec.items()):
        ax.scatter(df["rsa_pic"], df["rsa_aud"], s=14, alpha=0.5, label=pat)
    allv = pd.concat(elec.values())
    lim = [min(allv["rsa_pic"].min(), allv["rsa_aud"].min()) - 0.02,
           max(allv["rsa_pic"].max(), allv["rsa_aud"].max()) + 0.02]
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
    ax.axhline(0, color="#bbb", lw=0.6); ax.axvline(0, color="#bbb", lw=0.6)
    ax.set_xlabel("RSA vs GloVe (picture)"); ax.set_ylabel("RSA vs GloVe (auditory)")
    ax.set_title("Per-electrode semantic encoding, both tasks", fontsize=10)
    ax.legend(fontsize=7, ncol=2, title="patient")
    fig.tight_layout()
    return _img_tag(fig, alt="elec_overview")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def overview_table(per_pat: dict, metric: str = "word_bal_acc") -> str:
    rows = []
    for pat in sorted(per_pat):
        df = per_pat[pat]
        row = {"patient": pat, "metric": metric}
        for target in TARGETS:
            for src in SRC_ORDER:
                v = _vals(df, COND[(target, src)], metric)
                row["{}_{}".format(target, src)] = float(np.mean(v)) if len(v) else np.nan
        rows.append(row)
    return _df_to_html(pd.DataFrame(rows))


def paired_wilcoxon(df, cond_a, cond_b, label_a, label_b):
    """Paired (by bootstrap_id) Wilcoxon for each metric: a vs b, two-sided."""
    out = []
    a = df[df["condition"] == cond_a]; b = df[df["condition"] == cond_b]
    merged = pd.merge(a, b, on="bootstrap_id", suffixes=("_a", "_b"))
    for col, _ in METRICS:
        va = merged[col + "_a"].dropna(); vb = merged[col + "_b"].dropna()
        n = min(len(va), len(vb))
        if n < 5:
            continue
        va, vb = va.values[:n], vb.values[:n]
        try:
            _, p = wilcoxon(va, vb, zero_method="zsplit")
        except ValueError:
            p = np.nan
        out.append({"comparison": "{} vs {}".format(label_a, label_b), "metric": col,
                    "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
                    "mean_diff": float((va - vb).mean()), "n": n,
                    "wilcoxon_p": float(p) if np.isfinite(p) else np.nan})
    return out


def _wilcoxon_html(stat_df) -> str:
    if stat_df is None or stat_df.empty:
        return "<p class='subtle'>(insufficient bootstraps)</p>"
    p_cols = {"wilcoxon_p"}
    float_cols = set(stat_df.select_dtypes(include="float").columns)
    rows = []
    for _, r in stat_df.iterrows():
        cells = []
        for c in stat_df.columns:
            v = r[c]
            if c in p_cols:
                cells.append("<td>{}</td>".format(_highlight_p(v)))
            elif c in float_cols:
                try:
                    cells.append("<td>{:.4f}</td>".format(float(v)))
                except (TypeError, ValueError):
                    cells.append("<td>{}</td>".format(v))
            else:
                cells.append("<td class='text'>{}</td>".format(v))
        rows.append("<tr>" + "".join(cells) + "</tr>")
    head = "".join("<th>{}</th>".format(c) for c in stat_df.columns)
    return "<table class='results'><thead><tr>" + head + "</tr></thead><tbody>" + "\n".join(rows) + "</tbody></table>"


# ---------------------------------------------------------------------------
# Per-patient figures
# ---------------------------------------------------------------------------

def plot_patient_bars(pat: str, df) -> str:
    fig, axes = plt.subplots(2, len(METRICS), figsize=(4.6 * len(METRICS), 8.0), sharey="col")
    for row_i, target in enumerate(TARGETS):
        for col_i, (col, title) in enumerate(METRICS):
            ax = axes[row_i, col_i]
            means, sems, colors, labels = [], [], [], []
            for src in SRC_ORDER:
                v = _vals(df, COND[(target, src)], col)
                means.append(float(np.mean(v)) if len(v) else np.nan)
                sems.append(float(np.std(v) / np.sqrt(max(1, len(v)))))
                colors.append(SRC_COLORS[src]); labels.append(SRC_LABELS[src].split(" ")[0])
            x = np.arange(len(means))
            ax.bar(x, means, yerr=sems, color=colors, capsize=4, alpha=0.85, error_kw={"lw": 1.3})
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
            ax.set_title("{}\n{}".format(TARGET_LABELS[target], title), fontsize=8.5)
            if col_i == 0:
                ax.set_ylabel("Score")
            ax.grid(axis="y", alpha=0.3)
            for xi, (m, s) in enumerate(zip(means, sems)):
                if not np.isnan(m):
                    ax.text(xi, m + s + 0.004, "{:.3f}".format(m), ha="center", fontsize=7, color="#333")
    fig.suptitle("Patient {}: co-training conditions".format(pat), fontsize=11)
    fig.tight_layout()
    return _img_tag(fig, alt="bars_" + pat)


def plot_seen_unseen(pat: str, df) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, target in zip(axes, TARGETS):
        x = np.arange(len(SRC_ORDER)); width = 0.35
        seen_m, seen_e, uns_m, uns_e = [], [], [], []
        for src in SRC_ORDER:
            cond = COND[(target, src)]
            sv = _vals(df, cond, "word_acc_seen"); uv = _vals(df, cond, "word_acc_unseen")
            seen_m.append(float(np.mean(sv)) if len(sv) else np.nan)
            seen_e.append(float(np.std(sv) / np.sqrt(max(1, len(sv)))))
            uns_m.append(float(np.mean(uv)) if len(uv) else np.nan)
            uns_e.append(float(np.std(uv) / np.sqrt(max(1, len(uv)))))
        colors = [SRC_COLORS[s] for s in SRC_ORDER]
        ax.bar(x - width / 2, seen_m, width, yerr=seen_e, color=colors, alpha=0.85, capsize=3, label="seen (train vocab)")
        ax.bar(x + width / 2, uns_m, width, yerr=uns_e, color=colors, alpha=0.4, capsize=3,
               hatch="//", edgecolor="white", label="unseen (zero-shot)")
        ax.set_xticks(x); ax.set_xticklabels([SRC_LABELS[s].split(" ")[0] for s in SRC_ORDER], fontsize=8)
        ax.set_title(TARGET_LABELS[target], fontsize=9); ax.set_ylabel("Word balanced accuracy")
        ax.grid(axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.legend(fontsize=8)
    fig.suptitle("Patient {}: seen vs unseen (zero-shot) word accuracy".format(pat), fontsize=11)
    fig.tight_layout()
    return _img_tag(fig, alt="seen_unseen_" + pat)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def section_overview(per_pat: dict) -> str:
    return (
        "<h2>Cross-patient overview</h2>"
        "<div class='box'><b>How to read this.</b> For each evaluation target (picture or "
        "auditory test trials), three decoders are compared: <b>within</b> (trained on the "
        "same task), <b>cross</b> (trained on the other task only), and <b>pooled</b> "
        "(trained on both tasks). Error bars are bootstrap SEM. "
        "<i>cross≈within</i> &rarr; shared representation (Q1); "
        "<i>pooled≥within</i> &rarr; one decoder serves both (Q3).</div>"
        + _legend_html()
        + "<h3>Word balanced accuracy</h3>" + fig_overview(per_pat, "word_bal_acc", "Word balanced accuracy")
        + "<h3>Category-independent balanced accuracy</h3>"
        + fig_overview(per_pat, "cat_indep_bal_acc", "Category-independent balanced accuracy")
        + "<h3>Cosine similarity (predicted vs true embedding)</h3>"
        + fig_overview(per_pat, "cosine_mean", "Cosine similarity")
        + "<h3>Mean word_bal_acc table (target × source)</h3>" + overview_table(per_pat, "word_bal_acc")
        + "<h3>Mean cat_indep_bal_acc table (target × source)</h3>" + overview_table(per_pat, "cat_indep_bal_acc")
    )


def section_q1(per_pat: dict, rsa: pd.DataFrame) -> str:
    return (
        "<h2>Q1 — Is the semantic representation the same across tasks?</h2>"
        "<div class='qbox'><b>Test.</b> If a decoder trained on one task still works on the "
        "other (cross ≈ within), the two tasks share a semantic code. The bars below show "
        "<b>cross − within</b> (negative = loss when crossing tasks). The RSA panel "
        "correlates the per-word neural geometry (RDM) between tasks and against GloVe; "
        "<i>rdm_pic_vs_aud</i> &gt; 0 indicates shared geometry independent of any decoder. "
        "Category-independent accuracy is shown alongside word accuracy because coarse "
        "semantic (category) structure is more robust and often transfers even when "
        "word-level identity does not.</div>"
        + "<h3>Cross − within: word balanced accuracy</h3>"
        + fig_delta(per_pat, "cross", "within", "Cross-task minus within-task (word_bal_acc)", "word_bal_acc")
        + "<h3>Cross − within: category-independent balanced accuracy</h3>"
        + fig_delta(per_pat, "cross", "within", "Cross-task minus within-task (cat_indep_bal_acc)", "cat_indep_bal_acc")
        + "<h3>Representational similarity (RDM correlations)</h3>" + fig_rsa(rsa)
        + _df_to_html(rsa[["patient", "n_shared_words", "rdm_pic_vs_aud", "rdm_pic_vs_glove", "rdm_aud_vs_glove"]] if not rsa.empty else rsa)
    )


def section_q3(per_pat: dict) -> str:
    return (
        "<h2>Q3 — Can one decoder serve both tasks?</h2>"
        "<div class='qbox'><b>Test.</b> Pooled-vs-within shows whether co-training helps or "
        "hurts each task. Bars are <b>pooled − within</b> (positive = pooling helps; the "
        "smaller task is expected to gain). Per-patient paired Wilcoxon tests are in the "
        "per-patient sections below.</div>"
        + "<h3>Pooled − within: word balanced accuracy</h3>"
        + fig_delta(per_pat, "pooled", "within", "Pooled minus within-task (word_bal_acc)", "word_bal_acc")
        + "<h3>Pooled − within: category-independent balanced accuracy</h3>"
        + fig_delta(per_pat, "pooled", "within", "Pooled minus within-task (cat_indep_bal_acc)", "cat_indep_bal_acc")
    )


def section_q2(elec: dict) -> str:
    # cross-patient top amodal table
    rows = []
    for pat in sorted(elec):
        df = elec[pat].copy()
        n_both_pos = int(((df["rsa_pic"] > 0) & (df["rsa_aud"] > 0)).sum())
        top = df.sort_values("amodal_score", ascending=False).head(3)
        rows.append({"patient": pat, "n_electrodes": len(df),
                     "n_rsa_pos_both": n_both_pos,
                     "top_channels": ", ".join(top["channel"].astype(str).tolist()),
                     "max_amodal_score": float(df["amodal_score"].max())})
    tbl = _df_to_html(pd.DataFrame(rows)) if rows else "<p class='subtle'>(no data)</p>"
    return (
        "<h2>Q2 — Are there amodal (modality-independent) electrodes?</h2>"
        "<div class='qbox'><b>Test.</b> Each electrode's semantic encoding is scored by RSA of "
        "its per-word activity geometry against GloVe, separately in each task. <b>Amodal "
        "candidates</b> lie in the upper-right (high RSA in <i>both</i> tasks) and have "
        "consistent cross-task tuning (amodal_score = min(rsa_pic, rsa_aud) × "
        "max(0, consistency)). Map the top channels to anatomy to test for hub regions "
        "(vATL / LpMTG / angular-IPS).</div>"
        + fig_electrodes_overview(elec)
        + "<h3>Top amodal electrodes per patient</h3>" + tbl
    )


def section_per_patient(in_dir: Path, per_pat: dict, elec: dict) -> str:
    blocks, all_stats = [], []
    for pat, df in sorted(per_pat.items()):
        stats = []
        for target in TARGETS:
            stats += paired_wilcoxon(df, COND[(target, "cross")], COND[(target, "within")],
                                     "cross_" + target, "within_" + target)
            stats += paired_wilcoxon(df, COND[(target, "pooled")], COND[(target, "within")],
                                     "pooled_" + target, "within_" + target)
        stat_df = pd.DataFrame(stats)
        if not stat_df.empty:
            stat_df.insert(0, "patient", pat)
            all_stats.append(stat_df)
        elec_png = _file_img_tag(in_dir / pat / "cotrain_{}_electrodes.png".format(pat), "elec " + pat)
        block = (
            "<h2>Patient {}</h2>".format(pat)
            + "<div class='figrow'>"
            + "<div><h3>All metrics by target and train source</h3>" + plot_patient_bars(pat, df) + "</div>"
            + "<div><h3>Seen vs unseen (zero-shot) word accuracy</h3>" + plot_seen_unseen(pat, df) + "</div>"
            + "</div>"
            + "<details><summary>Paired Wilcoxon (cross/pooled vs within, by bootstrap)</summary>"
            + _wilcoxon_html(stat_df) + "</details>"
            + "<details><summary>Per-electrode amodal scatter (saved figure)</summary>" + elec_png + "</details>"
        )
        blocks.append(block)
    master = ""
    if all_stats:
        master = "<h2>Master paired Wilcoxon (all patients)</h2>" + _wilcoxon_html(pd.concat(all_stats, ignore_index=True))
    return "\n".join(blocks) + master


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="HTML report for cross_task_cotrain results")
    p.add_argument("--in-dir", default=str(DEFAULT_IN_DIR))
    p.add_argument("--out", default=None)
    p.add_argument("--model", default=None, help="Filter to one model (default: first present)")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out) if args.out else (in_dir / "cross_task_cotrain_report.html")
    if not in_dir.exists():
        print("ERROR: in-dir does not exist:", in_dir); return 1

    per_pat, elec, rsa = load_all(in_dir)
    if not per_pat:
        print("No per-patient CSVs found."); return 1

    # choose model
    models = sorted(set().union(*[set(df["model"].unique()) for df in per_pat.values()]))
    model = args.model or ("kernel_pls" if "kernel_pls" in models else models[0])
    per_pat = {p_: d[d["model"] == model].copy() for p_, d in per_pat.items()}
    print("Patients:", ", ".join(sorted(per_pat)), "| model:", model,
          ("| other models present: " + ", ".join(m for m in models if m != model) if len(models) > 1 else ""))

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    method = (
        "<div class='box'><b>Method.</b>&nbsp; Picture- and auditory-naming trials of each "
        "patient are pooled into one dataset on the intersection of their channels (arranged "
        "identically), each task taken at its own loose-category peak bin. A single regressor "
        "(<b>{}</b>) is trained to predict the GloVe embedding. Per bootstrap (default N=50), a "
        "fraction of shared words is held fully out (zero-shot unseen); remaining trials "
        "are split per word. Six conditions are evaluated on shared test sets: within / cross / "
        "pooled training for each of the two evaluation targets. Decoding metric: 1-NN cosine "
        "retrieval against the target task's word-embedding database (chance ~ 1/n_words for "
        "word accuracy, 1/n_categories for category accuracy).</div>".format(model)
    )

    body = (
        "<h1>Cross-task co-training: picture + auditory naming as one task</h1>"
        "<p class='subtle'>Generated {} &middot; source: <code>{}</code> &middot; model: <code>{}</code></p>".format(generated, in_dir, model)
        + method
        + section_overview(per_pat)
        + section_q1(per_pat, rsa)
        + section_q3(per_pat)
        + section_q2(elec)
        + section_per_patient(in_dir, per_pat, elec)
    )
    html = ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>" + CSS
            + "<title>Cross-task co-training report</title></head><body>" + body + "</body></html>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print("Wrote:", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
