# cross_task_transfer_report.py
# HTML report from cross_task_transfer.py CSV outputs.
#
# Sections:
#   1. Cross-patient overview -- mean word_bal_acc per arm x direction x patient
#   2. Transfer gain -- delta over no_transfer baseline per patient
#   3. Per-patient detail -- bar charts (all metrics), seen/unseen split,
#      Wilcoxon arm > no_transfer statistics
#
# Inputs (default: results/cross_task_transfer/):
#   cross_task_transfer_summary.csv
#   <patient>/cross_task_transfer_<patient>.csv   (per-bootstrap rows)
#   <patient>/transfer_arms_bars.png              (already-saved static figure)
#
# Output:
#   results/cross_task_transfer/cross_task_transfer_report.html
#
# Usage:
#   python -m main.analysis.cross_task.cross_task_transfer_report
#   python -m main.analysis.cross_task.cross_task_transfer_report --in-dir <dir> --out report.html

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
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

from utils.paths import results_dir  # noqa: E402  (after the sys.path insert above)

# Was main/tests/results/cross_task_transfer, a root that no longer exists — the results
# tree was consolidated under main/results/. Resolve it through utils.paths so the two
# cannot drift again.
DEFAULT_IN_DIR = results_dir("cross_task_transfer", create=False)

# ---------------------------------------------------------------------------
# Arms / colours / metrics
# ---------------------------------------------------------------------------
ARM_ORDER = ["no_transfer", "transfer", "cca", "pca_cca"]
ARM_COLORS = {
    "no_transfer": "#7f7f7f",
    "transfer":    "#1f77b4",
    "cca":         "#2ca02c",
    "pca_cca":     "#9467bd",
}
ARM_LABELS = {
    "no_transfer": "No transfer\n(KernelPLS)",
    "transfer":    "Transfer\n(Ridge/T-space)",
    "cca":         "CCA\n(HGA-space)",
    "pca_cca":     "PCA-CCA\n(multibin)",
}
ARM_LABELS_FLAT = {k: v.replace("\n", " ") for k, v in ARM_LABELS.items()}

DIRECTIONS = ["pic_to_aud", "aud_to_pic"]
DIRECTION_LABELS = {
    "pic_to_aud": "Picture \u2192 Auditory",
    "aud_to_pic": "Auditory \u2192 Picture",
}

METRICS = [
    ("word_bal_acc",      "Word balanced accuracy"),
    ("cat_indep_bal_acc", "Category-independent balanced accuracy"),
    ("cosine_mean",       "Cosine similarity"),
]


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig, dpi: int = 130) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img_tag(fig, alt: str = "", dpi: int = 130) -> str:
    b64 = _fig_to_b64(fig, dpi=dpi)
    return '<img alt="{}" src="data:image/png;base64,{}" />'.format(alt, b64)


def _file_img_tag(path: Path, alt: str = "") -> str:
    if not path.exists():
        return "<p class='subtle'>(figure not found: {})</p>".format(path.name)
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return '<img alt="{}" src="data:image/png;base64,{}" />'.format(alt, b64)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

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
.box { background: #F5F7FA; padding: 10px 14px; border-left: 3px solid #1565C0;
       margin: 12px 0; font-size: 13px; }
.arm-legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 8px 0 14px 0; font-size: 13px; }
.arm-swatch { display: inline-block; width: 14px; height: 14px; border-radius: 3px;
              vertical-align: middle; margin-right: 4px; }
.figrow { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
.figrow > div { flex: 1 1 480px; }
.sig { color: #2E7D32; font-weight: 600; }
.ns  { color: #9E9E9E; }
</style>
"""


def _arm_legend_html() -> str:
    items = "".join(
        "<span><span class='arm-swatch' style='background:{}'></span>{}</span>".format(
            ARM_COLORS[a], ARM_LABELS_FLAT[a]
        )
        for a in ARM_ORDER
    )
    return "<div class='arm-legend'>{}</div>".format(items)


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
    if v < 0.05:
        return "<b class='sig'>{:.4f}</b>".format(v)
    return "<span class='ns'>{:.4f}</span>".format(v)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all(in_dir: Path):
    """Load summary CSV + per-patient bootstrap CSVs.

    Returns (summary_df, {patient: per_bootstrap_df}).
    """
    summary_path = in_dir / "cross_task_transfer_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    per_pat = {}
    for pat_dir in sorted(p for p in in_dir.iterdir() if p.is_dir()):
        pat = pat_dir.name
        csv = pat_dir / "cross_task_transfer_{}.csv".format(pat)
        if csv.exists():
            per_pat[pat] = pd.read_csv(csv)
    return summary, per_pat


# ---------------------------------------------------------------------------
# Cross-patient overview figures
# ---------------------------------------------------------------------------

def fig_cross_patient_bars(per_pat: dict) -> str:
    """Grouped bar chart: word_bal_acc per arm, 2 subplots (one per direction)."""
    rows = []
    for pat, df in per_pat.items():
        for direction in DIRECTIONS:
            sub = df[df["direction"] == direction]
            for arm in ARM_ORDER:
                vals = sub[sub["arm"] == arm]["word_bal_acc"].dropna()
                rows.append({
                    "patient": pat, "direction": direction, "arm": arm,
                    "mean": float(vals.mean()) if len(vals) > 0 else np.nan,
                    "sem":  float(vals.sem())  if len(vals) > 1 else 0.0,
                })
    if not rows:
        return "<p class='subtle'>(no data)</p>"
    agg = pd.DataFrame(rows)
    patients = sorted(agg["patient"].unique())
    n_pat = len(patients)
    fig, axes = plt.subplots(1, 2, figsize=(max(9, 2.6 * n_pat), 4.5), sharey=True)
    for ax, direction in zip(axes, DIRECTIONS):
        sub = agg[agg["direction"] == direction]
        x = np.arange(n_pat)
        width = 0.8 / len(ARM_ORDER)
        for i, arm in enumerate(ARM_ORDER):
            arm_sub = sub[sub["arm"] == arm].set_index("patient").reindex(patients)
            means = arm_sub["mean"].fillna(0).values
            sems  = arm_sub["sem"].fillna(0).values
            offset = (i - len(ARM_ORDER) / 2 + 0.5) * width
            ax.bar(x + offset, means, width, yerr=sems,
                   color=ARM_COLORS[arm], alpha=0.85,
                   label=ARM_LABELS_FLAT[arm],
                   capsize=3, error_kw={"lw": 1.2})
        ax.set_xticks(x)
        ax.set_xticklabels(patients, fontsize=9)
        ax.set_xlabel("Patient")
        ax.set_ylabel("Word balanced accuracy")
        ax.set_title(DIRECTION_LABELS[direction], fontsize=10)
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7.5, ncol=2)
    fig.suptitle("Cross-task transfer: word balanced accuracy per patient", fontsize=11)
    fig.tight_layout()
    return _img_tag(fig, alt="cross_patient_bars")


def fig_transfer_gain(per_pat: dict) -> str:
    """Delta over no_transfer baseline per patient x direction."""
    rows = []
    for pat, df in per_pat.items():
        for direction in DIRECTIONS:
            sub = df[df["direction"] == direction]
            nt_vals = sub[sub["arm"] == "no_transfer"]["word_bal_acc"].dropna()
            nt_mean = float(nt_vals.mean()) if len(nt_vals) > 0 else np.nan
            for arm in [a for a in ARM_ORDER if a != "no_transfer"]:
                vals = sub[sub["arm"] == arm]["word_bal_acc"].dropna()
                m = float(vals.mean()) if len(vals) > 0 else np.nan
                delta = (m - nt_mean) if not (np.isnan(m) or np.isnan(nt_mean)) else np.nan
                rows.append({
                    "patient": pat, "direction": direction, "arm": arm,
                    "delta": delta,
                })
    if not rows:
        return "<p class='subtle'>(no data)</p>"
    agg = pd.DataFrame(rows)
    transfer_arms = [a for a in ARM_ORDER if a != "no_transfer"]
    patients = sorted(agg["patient"].unique())
    n_pat = len(patients)
    fig, axes = plt.subplots(1, 2, figsize=(max(9, 2.6 * n_pat), 4.2), sharey=True)
    for ax, direction in zip(axes, DIRECTIONS):
        sub = agg[agg["direction"] == direction]
        x = np.arange(n_pat)
        width = 0.75 / len(transfer_arms)
        for i, arm in enumerate(transfer_arms):
            arm_sub = sub[sub["arm"] == arm].set_index("patient").reindex(patients)
            deltas = arm_sub["delta"].fillna(0).values
            offset = (i - len(transfer_arms) / 2 + 0.5) * width
            ax.bar(x + offset, deltas, width, color=ARM_COLORS[arm], alpha=0.85,
                   label=ARM_LABELS_FLAT[arm])
            for xi, d in zip(x + offset, deltas):
                if not np.isnan(d):
                    ypos = d + (0.004 if d >= 0 else -0.014)
                    ax.text(xi, ypos, "{:+.3f}".format(d),
                            ha="center", fontsize=6.5, color="#333")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(patients, fontsize=9)
        ax.set_xlabel("Patient")
        ax.set_ylabel("\u0394 word_bal_acc vs no_transfer")
        ax.set_title(DIRECTION_LABELS[direction], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Transfer gain over no-transfer baseline", fontsize=11)
    fig.tight_layout()
    return _img_tag(fig, alt="transfer_gain")


def build_summary_table(per_pat: dict) -> str:
    """Flat HTML table: patient x direction x arm -> mean +/- SEM for word_bal_acc."""
    rows = []
    for pat, df in per_pat.items():
        for direction in DIRECTIONS:
            sub = df[df["direction"] == direction]
            for arm in ARM_ORDER:
                vals = sub[sub["arm"] == arm]["word_bal_acc"].dropna()
                rows.append({
                    "patient": pat,
                    "direction": direction,
                    "arm": arm,
                    "n_bootstrap": int(len(vals)),
                    "word_bal_acc_mean": float(vals.mean()) if len(vals) > 0 else np.nan,
                    "word_bal_acc_sem":  float(vals.sem())  if len(vals) > 1 else 0.0,
                })
    if not rows:
        return "<p class='subtle'>(no data)</p>"
    return _df_to_html(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Per-patient figures
# ---------------------------------------------------------------------------

def plot_patient_bars(pat: str, df) -> str:
    """2 x 3 subplot: direction rows, metric columns; bar + SEM + value annotation."""
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(2, n_metrics,
                             figsize=(4.5 * n_metrics, 8.0), sharey="col")
    for row_i, direction in enumerate(DIRECTIONS):
        sub = df[df["direction"] == direction]
        for col_i, (col, title) in enumerate(METRICS):
            ax = axes[row_i, col_i]
            arms_present = [a for a in ARM_ORDER if a in sub["arm"].unique()]
            means, sems, colors, tick_labels = [], [], [], []
            for arm in arms_present:
                vals = sub[sub["arm"] == arm][col].dropna().values
                means.append(float(vals.mean()) if len(vals) > 0 else np.nan)
                sems.append(float(vals.std() / np.sqrt(max(1, len(vals)))))
                colors.append(ARM_COLORS[arm])
                tick_labels.append(ARM_LABELS[arm])
            x = np.arange(len(means))
            ax.bar(x, means, yerr=sems, color=colors, capsize=4,
                   alpha=0.85, error_kw={"lw": 1.4})
            ax.set_xticks(x)
            ax.set_xticklabels(tick_labels, fontsize=7)
            ax.set_title("{}\n{}".format(DIRECTION_LABELS[direction], title), fontsize=9)
            if col_i == 0:
                ax.set_ylabel("Score")
            ax.grid(axis="y", alpha=0.3)
            for xi, (m, s) in enumerate(zip(means, sems)):
                if not np.isnan(m):
                    ax.text(xi, m + s + 0.005, "{:.3f}".format(m),
                            ha="center", fontsize=7, color="#333")
    fig.suptitle("Patient {}: cross-task transfer arms".format(pat), fontsize=11)
    fig.tight_layout()
    return _img_tag(fig, alt="bars_{}".format(pat))


def plot_seen_unseen(pat: str, df) -> str:
    """Paired bar chart: word_acc_seen (solid) vs word_acc_unseen (hatched) per direction."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, direction in zip(axes, DIRECTIONS):
        sub = df[df["direction"] == direction]
        arms_present = [a for a in ARM_ORDER if a in sub["arm"].unique()]
        x = np.arange(len(arms_present))
        width = 0.35
        seen_m, seen_e, unseen_m, unseen_e = [], [], [], []
        for arm in arms_present:
            asub = sub[sub["arm"] == arm]
            sv = asub["word_acc_seen"].dropna().values
            uv = asub["word_acc_unseen"].dropna().values
            seen_m.append(float(sv.mean()) if len(sv) > 0 else np.nan)
            seen_e.append(float(sv.std() / np.sqrt(max(1, len(sv)))))
            unseen_m.append(float(uv.mean()) if len(uv) > 0 else np.nan)
            unseen_e.append(float(uv.std() / np.sqrt(max(1, len(uv)))))
        arm_colors = [ARM_COLORS[a] for a in arms_present]
        ax.bar(x - width / 2, seen_m, width, yerr=seen_e, color=arm_colors,
               alpha=0.85, capsize=3, label="anchor-seen")
        ax.bar(x + width / 2, unseen_m, width, yerr=unseen_e, color=arm_colors,
               alpha=0.45, capsize=3, hatch="//", edgecolor="white", label="anchor-unseen")
        ax.set_xticks(x)
        ax.set_xticklabels([ARM_LABELS_FLAT[a] for a in arms_present],
                            fontsize=7.5, rotation=10)
        ax.set_title(DIRECTION_LABELS[direction], fontsize=9)
        ax.set_ylabel("Word balanced accuracy")
        ax.grid(axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.legend(fontsize=8)
    fig.suptitle("Patient {}: seen vs unseen word accuracy".format(pat), fontsize=11)
    fig.tight_layout()
    return _img_tag(fig, alt="seen_unseen_{}".format(pat))


def compute_wilcoxon(pat: str, df) -> pd.DataFrame:
    """Wilcoxon signed-rank: each arm > no_transfer, per direction x metric.

    Bonferroni correction across arms (per direction x metric group).
    Returns DataFrame with columns:
        patient, direction, metric, arm, mean_arm, mean_no_transfer, mean_diff,
        n_bootstrap, wilcoxon_p_one_sided, wilcoxon_p_bonf
    """
    stat_rows = []
    for direction in DIRECTIONS:
        sub = df[df["direction"] == direction]
        nt_df = sub[sub["arm"] == "no_transfer"]
        n_arms_compared = len([a for a in ARM_ORDER if a != "no_transfer"])
        for arm in [a for a in ARM_ORDER if a != "no_transfer"]:
            arm_df = sub[sub["arm"] == arm]
            for col, _title in METRICS:
                nt_vals  = nt_df[col].dropna().values
                arm_vals = arm_df[col].dropna().values
                n = min(len(nt_vals), len(arm_vals))
                if n < 5:
                    continue
                try:
                    _, p = wilcoxon(
                        arm_vals[:n], nt_vals[:n],
                        alternative="greater", zero_method="zsplit"
                    )
                except ValueError:
                    p = np.nan
                p_bonf = min(1.0, float(p) * n_arms_compared) if np.isfinite(p) else np.nan
                stat_rows.append({
                    "patient":              pat,
                    "direction":            direction,
                    "metric":               col,
                    "arm":                  arm,
                    "mean_arm":             float(arm_vals[:n].mean()),
                    "mean_no_transfer":     float(nt_vals[:n].mean()),
                    "mean_diff":            float((arm_vals[:n] - nt_vals[:n]).mean()),
                    "n_bootstrap":          n,
                    "wilcoxon_p_one_sided": float(p) if np.isfinite(p) else np.nan,
                    "wilcoxon_p_bonf":      p_bonf,
                })
    return pd.DataFrame(stat_rows)


def _wilcoxon_html(stat_df) -> str:
    """Render Wilcoxon table with highlighted significant p-values."""
    if stat_df is None or stat_df.empty:
        return "<p class='subtle'>(insufficient bootstraps for Wilcoxon)</p>"
    float_cols = {"mean_arm", "mean_no_transfer", "mean_diff",
                  "wilcoxon_p_one_sided", "wilcoxon_p_bonf"}
    p_cols = {"wilcoxon_p_one_sided", "wilcoxon_p_bonf"}
    html_rows = []
    for _, row in stat_df.iterrows():
        cells = []
        for col in stat_df.columns:
            val = row[col]
            if col in p_cols:
                cells.append("<td>{}</td>".format(_highlight_p(val)))
            elif col in float_cols:
                try:
                    cells.append("<td>{:.4f}</td>".format(float(val)))
                except (TypeError, ValueError):
                    cells.append("<td>{}</td>".format(val))
            else:
                cells.append("<td class='text'>{}</td>".format(val))
        html_rows.append("<tr>" + "".join(cells) + "</tr>")
    header_cells = "".join("<th>{}</th>".format(c) for c in stat_df.columns)
    return (
        "<table class='results'><thead><tr>"
        + header_cells
        + "</tr></thead><tbody>"
        + "\n".join(html_rows)
        + "</tbody></table>"
    )


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def section_overview(per_pat: dict) -> str:
    bars_img = fig_cross_patient_bars(per_pat)
    gain_img = fig_transfer_gain(per_pat)
    tbl      = build_summary_table(per_pat)
    return (
        "<h2>Cross-patient overview</h2>"
        "<div class='box'>"
        "<b>What this section shows:</b> across all available patients, how does each "
        "alignment arm compare to the within-target baseline (<i>no_transfer</i>)? "
        "Both transfer directions are shown. Error bars are bootstrap SEM."
        "</div>"
        + _arm_legend_html()
        + "<h3>Word balanced accuracy per patient and arm</h3>"
        + bars_img
        + "<h3>Transfer gain over no-transfer baseline (&Delta;word_bal_acc)</h3>"
        + gain_img
        + "<h3>Summary table (mean word_bal_acc over all bootstraps)</h3>"
        + tbl
    )


def section_per_patient(in_dir: Path, per_pat: dict) -> str:
    blocks = []
    all_stats = []
    for pat, df in sorted(per_pat.items()):
        bars_img   = plot_patient_bars(pat, df)
        seen_img   = plot_seen_unseen(pat, df)
        stat_df    = compute_wilcoxon(pat, df)
        static_png = _file_img_tag(
            in_dir / pat / "transfer_arms_bars.png",
            alt="summary bars {}".format(pat)
        )
        if not stat_df.empty:
            all_stats.append(stat_df)
        block = (
            "<h2>Patient {}</h2>".format(pat)
            + "<div class='figrow'>"
            + "<div><h3>All metrics by arm and direction</h3>" + bars_img + "</div>"
            + "<div><h3>Seen vs unseen word accuracy</h3>" + seen_img + "</div>"
            + "</div>"
            + "<details><summary>Wilcoxon statistics"
            + " (arm &gt; no_transfer, one-sided, Bonferroni-corrected)</summary>"
            + _wilcoxon_html(stat_df)
            + "</details>"
            + "<details><summary>Summary figure"
            + " (transfer_arms_bars.png saved by cross_task_transfer.py)</summary>"
            + static_png
            + "</details>"
        )
        blocks.append(block)

    master_html = ""
    if all_stats:
        master = pd.concat(all_stats, ignore_index=True)
        master_html = (
            "<h2>Master Wilcoxon statistics (all patients)</h2>"
            + _wilcoxon_html(master)
        )

    return "\n".join(blocks) + master_html


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="HTML report for cross_task_transfer results"
    )
    p.add_argument(
        "--in-dir", default=str(DEFAULT_IN_DIR),
        help="Directory with per-patient sub-folders and summary CSV"
    )
    p.add_argument(
        "--out", default=None,
        help="Output HTML path (default: <in-dir>/cross_task_transfer_report.html)"
    )
    args = p.parse_args()

    in_dir   = Path(args.in_dir)
    out_path = Path(args.out) if args.out else (in_dir / "cross_task_transfer_report.html")

    print("Loading from : {}".format(in_dir), flush=True)
    if not in_dir.exists():
        print("ERROR: in-dir does not exist.", flush=True)
        return 1

    summary, per_pat = load_all(in_dir)
    if not per_pat:
        print("No per-patient CSVs found -- nothing to report.", flush=True)
        return 1

    print("Patients     : {}".format(", ".join(sorted(per_pat))), flush=True)
    print("Building HTML ...", flush=True)

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    method_box = (
        "<div class='box'>"
        "<b>Method.</b>&nbsp; For each patient and both transfer directions "
        "(picture&rarr;auditory, auditory&rarr;picture), a bootstrap loop (default N=50) "
        "repeatedly splits trials into train/test sets, samples K=8 anchor words shared "
        "between the two tasks, and evaluates four arms: "
        "(1)&nbsp;<b>no_transfer</b> &mdash; kernel-PLS trained on target trials only; "
        "(2)&nbsp;<b>transfer</b> &mdash; source PLS frozen, ridge maps target HGA to "
        "source PLS T-space via anchor words; "
        "(3)&nbsp;<b>cca</b> &mdash; CCA aligns word-averaged target HGA to source HGA, "
        "source PLS predicts; "
        "(4)&nbsp;<b>pca_cca</b> &mdash; like CCA but on PCA-compressed multibin lagged "
        "HGA representations. "
        "Decoding: 1-NN cosine retrieval against the target-task word embedding database."
        "</div>"
    )

    body = (
        "<h1>Cross-task transfer: picture naming \u2194 auditory naming</h1>"
        "<p class='subtle'>Generated {} &middot; source: <code>{}</code></p>".format(
            generated, in_dir)
        + method_box
        + section_overview(per_pat)
        + section_per_patient(in_dir, per_pat)
    )

    html = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        + CSS
        + "<title>Cross-task transfer report</title></head><body>"
        + body
        + "</body></html>"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print("Wrote: {}".format(out_path), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
