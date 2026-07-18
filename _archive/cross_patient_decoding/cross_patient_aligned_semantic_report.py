# -*- coding: utf-8 -*-
"""
tests/cross_patient_decoding/cross_patient_aligned_semantic_report.py
======================================================================
HTML report from cross_patient_aligned_semantic.py CSV outputs.

Sections per (target, embedding):
    1. Peak-bin accuracy by arm  (bar chart + Wilcoxon vs no_transfer)
    2. Time course over alignment window
    3. Seen vs unseen vocabulary split

The single-patient semantic regression baseline ("original") is overlaid
when --baseline-run is supplied, mirroring cross_patient_few_shot_report.py.

Usage (from main/):
    python -m _archive.cross_patient_decoding.cross_patient_aligned_semantic_report
    python -m _archive.cross_patient_decoding.cross_patient_aligned_semantic_report \\
        --baseline-run <run_folder> --mode overall
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAIN_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

from analysis.helpers._cross_patient_helpers import (   # noqa: E402
    DEFAULT_SOURCE_PATIENT,
    DEFAULT_TARGET_PATIENTS,
    DEFAULT_EMBEDDINGS,
    DEFAULT_ARM3_RESULTS_ROOT,
    get_out_dir,
    load_arm3_baseline,
    load_arm3_chance,
    header,
    step,
)

# ── Colours / labels ──────────────────────────────────────────────────────
ARM_COLORS = {
    "cca_align":   "#1f77b4",
    "joint_pca":   "#ff7f0e",
    "mcca":        "#9467bd",
    "no_transfer": "#d62728",
    "original":    "#2ca02c",
}
ARM_LABELS = {
    "cca_align":   "CCA alignment",
    "joint_pca":   "Joint PCA",
    "mcca":        "MCCA alignment",
    "no_transfer": "No transfer (target-only KernelPLS)",
    "original":    "Single-patient baseline (semantic regression)",
}
ARM_ORDER = ["cca_align", "joint_pca", "mcca", "no_transfer"]

METRICS_OVERALL = [
    ("cosine_mean",       "Cosine similarity"),
    ("word_bal_acc",      "Word balanced accuracy"),
    ("cat_indep_bal_acc", "Category-independent balanced accuracy"),
]
METRICS_UNSEEN = [
    ("cosine_unseen",   "Cosine similarity (unseen words)"),
    ("word_acc_unseen", "Word balanced accuracy (unseen words)"),
    ("cat_acc_unseen",  "Category balanced accuracy (unseen words)"),
]
METRICS = METRICS_UNSEEN   # default; overridden by --mode in main()

# Mapping from METRICS column names to per_time_scores.csv column names
_BASELINE_COL = {
    "cosine_mean":       "cosine_mean",
    "word_bal_acc":      "word_balanced_acc",
    "cat_indep_bal_acc": "category_balanced_acc",
    "cosine_unseen":     "cosine_mean",
    "word_acc_unseen":   "word_balanced_acc",
    "cat_acc_unseen":    "category_balanced_acc",
}


# ── HTML helpers ──────────────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img_tag(fig, alt: str = "") -> str:
    return f'<img alt="{alt}" src="data:image/png;base64,{_fig_to_b64(fig)}" />'


def _add_chance_line(ax, col: str, chance: dict) -> None:
    """Add a horizontal chance reference line (word / category) to ax."""
    if "word" in col and chance.get("word_chance") is not None:
        ax.axhline(chance["word_chance"], ls=":", lw=1.0, color="#777777",
                   label=f"chance (1/{chance['n_unique_words']})")
    elif "cat" in col and chance.get("cat_chance") is not None:
        ax.axhline(chance["cat_chance"], ls=":", lw=1.0, color="#777777",
                   label=f"chance (1/{chance['n_unique_categories']})")


# ── 1. Peak-bin bar chart ─────────────────────────────────────────────────

def plot_peak_bars(
    peak_df: pd.DataFrame,
    baseline_df,
    target: str,
    embedding: str,
    chance: dict | None = None,
) -> tuple[str, pd.DataFrame]:
    """Bar chart comparing arms at peak bin with bootstrap SEM error bars.

    Returns (img_html, wilcoxon_stats_df).
    """
    arms = [a for a in ARM_ORDER if a in peak_df["arm"].unique()]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4))
    if len(METRICS) == 1:
        axes = [axes]

    stat_rows: list = []
    nt_vals_cache: dict[str, np.ndarray] = {}

    for ax, (col, title) in zip(axes, METRICS):
        means, sems, colors, tick_labels = [], [], [], []

        for arm in arms:
            vals = peak_df[peak_df["arm"] == arm][col].dropna().values
            if len(vals) == 0:
                continue
            means.append(float(vals.mean()))
            sems.append(float(vals.std() / np.sqrt(len(vals))))
            colors.append(ARM_COLORS[arm])
            tick_labels.append(ARM_LABELS[arm])
            if arm == "no_transfer":
                nt_vals_cache[col] = vals

        # Optional single-patient baseline ceiling bar
        if baseline_df is not None:
            bc = _BASELINE_COL.get(col)
            if bc and bc in baseline_df.columns:
                ceil_val = float(np.nanmax(baseline_df[bc].values))
                means.append(ceil_val)
                sems.append(0.0)
                colors.append(ARM_COLORS["original"])
                tick_labels.append(ARM_LABELS["original"])

        x = np.arange(len(means))
        ax.bar(x, means, yerr=sems, color=colors, capsize=4, alpha=0.82,
               error_kw={"lw": 1.5})
        ax.set_xticks(x)
        ax.set_xticklabels(
            [lbl.replace(" (", "\n(").replace(" alignment", "\nalignment")
             for lbl in tick_labels],
            fontsize=7,
        )

        if chance is not None:
            _add_chance_line(ax, col, chance)
            ax.legend(fontsize=7)

        ax.set_ylabel(title)
        ax.set_title(title, fontsize=10)

        # Wilcoxon: each alignment arm vs no_transfer (one-sided: arm > no_transfer)
        nt_vals = nt_vals_cache.get(col, np.array([]))
        for arm in arms:
            if arm == "no_transfer":
                continue
            arm_vals = peak_df[peak_df["arm"] == arm][col].dropna().values
            n = min(len(arm_vals), len(nt_vals))
            if n < 5:
                continue
            try:
                _, p = wilcoxon(arm_vals[:n], nt_vals[:n],
                                alternative="greater", zero_method="zsplit")
            except ValueError:
                p = np.nan
            stat_rows.append(dict(
                target=target, embedding=embedding, metric=col, arm=arm,
                mean_arm=float(arm_vals.mean()),
                mean_no_transfer=float(nt_vals[:n].mean()),
                mean_diff=float((arm_vals[:n] - nt_vals[:n]).mean()),
                n_bootstrap=n,
                wilcoxon_p_one_sided=float(p) if np.isfinite(p) else np.nan,
            ))

    fig.suptitle(f"{target} / {embedding}  —  peak-bin accuracy by arm", fontsize=12)
    fig.tight_layout()

    stat_df = pd.DataFrame(stat_rows)
    if len(stat_df) > 0:
        n_comp = stat_df.groupby(
            ["target", "embedding", "metric"]
        )["arm"].transform("nunique")
        stat_df["wilcoxon_p_bonf"] = (
            stat_df["wilcoxon_p_one_sided"] * n_comp
        ).clip(upper=1.0)

    return _img_tag(fig, alt=f"peak_bars_{target}_{embedding}"), stat_df


# ── 2. Time course ────────────────────────────────────────────────────────

def plot_time_courses(
    tc_df: pd.DataFrame,
    peak_df: pd.DataFrame,
    baseline_df,
    target: str,
    embedding: str,
    bin_size_ms: int = 100,
    chance: dict | None = None,
) -> str:
    """Line plot of decoding accuracy over alignment-window bins per arm."""
    if tc_df.empty:
        return ("<p><i>No timecourse data — rerun with "
                "<code>--n-bootstrap-timecourse &gt; 0</code>.</i></p>")

    arms = [a for a in ARM_ORDER if a in tc_df["arm"].unique()]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 4),
                             sharex=False)
    if len(METRICS) == 1:
        axes = [axes]

    # Modal peak bin for a vertical reference line
    peak_bin_marker: int | None = None
    if not peak_df.empty and "bin_index" in peak_df.columns:
        mode_series = pd.Series(peak_df["bin_index"].values).mode()
        if len(mode_series) > 0:
            peak_bin_marker = int(mode_series.iloc[0])

    for ax, (col, title) in zip(axes, METRICS):
        for arm in arms:
            sub = tc_df[tc_df["arm"] == arm]
            if sub.empty:
                continue
            agg = sub.groupby("bin_index")[col].agg(["mean", "std", "count"])
            t_s = agg.index.values * bin_size_ms / 1000.0
            ax.plot(t_s, agg["mean"].values, lw=1.8, color=ARM_COLORS[arm],
                    label=ARM_LABELS[arm])
            sem = agg["std"].values / np.sqrt(agg["count"].values.clip(1))
            ax.fill_between(t_s, agg["mean"].values - sem, agg["mean"].values + sem,
                            color=ARM_COLORS[arm], alpha=0.18)

        # Single-patient baseline time-course overlay
        if baseline_df is not None:
            bc = _BASELINE_COL.get(col)
            if bc and bc in baseline_df.columns:
                t_s3 = baseline_df["bin_index"].values * bin_size_ms / 1000.0
                ax.plot(t_s3, baseline_df[bc].values, lw=1.5, ls="--",
                        color=ARM_COLORS["original"], label=ARM_LABELS["original"])

        if peak_bin_marker is not None:
            ax.axvline(peak_bin_marker * bin_size_ms / 1000.0,
                       ls=":", color="grey", alpha=0.6,
                       label=f"src peak bin={peak_bin_marker}")

        if chance is not None:
            if "cosine" in col and chance.get("cosine_chance_per_bin") is not None:
                t_sc = chance["cosine_chance_bins"] * bin_size_ms / 1000.0
                ax.plot(t_sc, chance["cosine_chance_per_bin"],
                        lw=1.0, ls=":", color="#777777", label="chance (shuffled)")
            else:
                _add_chance_line(ax, col, chance)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(f"{target} / {embedding}  —  time course", fontsize=12)
    fig.tight_layout()
    return _img_tag(fig, alt=f"tc_{target}_{embedding}")


# ── 3. Seen vs unseen bar chart ───────────────────────────────────────────

def plot_seen_vs_unseen(peak_df: pd.DataFrame, target: str, embedding: str) -> str:
    """Paired bars comparing seen vs unseen vocabulary accuracy per arm."""
    arms = [a for a in ARM_ORDER if a in peak_df["arm"].unique()]
    metric_pairs = [
        ("cosine_seen",   "cosine_unseen",  "Cosine: seen vs unseen"),
        ("word_acc_seen", "word_acc_unseen", "Word acc: seen vs unseen"),
        ("cat_acc_seen",  "cat_acc_unseen",  "Cat acc: seen vs unseen"),
    ]
    fig, axes = plt.subplots(1, len(metric_pairs),
                             figsize=(5 * len(metric_pairs), 4))
    if len(metric_pairs) == 1:
        axes = [axes]

    x = np.arange(len(arms))
    width = 0.35

    for ax, (c_s, c_u, title) in zip(axes, metric_pairs):
        seen_m, seen_e, unseen_m, unseen_e = [], [], [], []
        for arm in arms:
            sub = peak_df[peak_df["arm"] == arm]
            sv = sub[c_s].dropna().values
            uv = sub[c_u].dropna().values
            seen_m.append(sv.mean() if len(sv) > 0 else np.nan)
            seen_e.append(sv.std() / np.sqrt(max(1, len(sv))))
            unseen_m.append(uv.mean() if len(uv) > 0 else np.nan)
            unseen_e.append(uv.std() / np.sqrt(max(1, len(uv))))

        arm_colors = [ARM_COLORS[a] for a in arms]
        ax.bar(x - width / 2, seen_m, width, yerr=seen_e,
               color=arm_colors, alpha=0.85, capsize=3, label="seen")
        ax.bar(x + width / 2, unseen_m, width, yerr=unseen_e,
               color=arm_colors, alpha=0.45, capsize=3,
               hatch="//", label="unseen")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [ARM_LABELS[a].split()[0] for a in arms], fontsize=8, rotation=15
        )
        ax.set_title(title, fontsize=10)
        if ax is axes[0]:
            ax.legend(fontsize=8)

    fig.suptitle(
        f"{target} / {embedding}  —  seen vs unseen vocabulary split", fontsize=12
    )
    fig.tight_layout()
    return _img_tag(fig, alt=f"seen_unseen_{target}_{embedding}")


# ── Report builder ────────────────────────────────────────────────────────

def build_report(
    csv_paths: list,
    source_patient: str,
    targets: list,
    embeddings: list,
    baseline_run: str | None,
    out_html: str,
    arm3_results_root: str | None = None,
    show_chance: bool = True,
    bin_size_ms: int = 100,
) -> None:
    all_df = []
    for p in csv_paths:
        try:
            all_df.append(pd.read_csv(p))
        except Exception as e:
            step(f"  failed to read {p}: {e}")
    if not all_df:
        raise RuntimeError("No CSVs loaded; nothing to report.")
    df = pd.concat(all_df, ignore_index=True)

    parts = [
        "<html><head><meta charset='utf-8'>"
        "<title>Cross-Patient Aligned Semantic Decoding Report</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;margin:24px;color:#222;max-width:1400px}"
        "h1{border-bottom:2px solid #1f77b4}"
        "h2{margin-top:36px;color:#1f4a72;"
        "border-bottom:1px solid #ccc;padding-bottom:4px}"
        "h3{color:#555;margin-top:24px}"
        "table{border-collapse:collapse;margin:8px 0;font-size:13px}"
        "th,td{border:1px solid #ccc;padding:4px 8px;text-align:right}"
        "th{background:#f0f0f0;text-align:center}"
        "img{max-width:100%;height:auto;margin:8px 0}"
        "code{background:#f4f4f4;padding:2px 4px;border-radius:3px}"
        "details{margin:8px 0}"
        "</style></head><body>",
    ]
    parts.append("<h1>Cross-Patient Aligned Semantic Decoding Report</h1>")
    parts.append(
        f"<p>Source: <b>{source_patient}</b> &nbsp; "
        f"Targets: <b>{', '.join(targets)}</b> &nbsp; "
        f"Embeddings: <b>{', '.join(embeddings)}</b></p>"
    )

    all_stats: list = []

    for tgt in targets:
        parts.append(f"<h2>Target: {tgt}</h2>")
        for emb in embeddings:
            pair = df[(df["target_patient"] == tgt) & (df["embedding"] == emb)]
            if pair.empty:
                parts.append(f"<p><i>No data for {tgt} / {emb}.</i></p>")
                continue

            baseline_df = None
            chance = None
            if baseline_run is not None:
                baseline_df = load_arm3_baseline(
                    tgt, emb, baseline_run, results_root=arm3_results_root
                )
                if baseline_df is None:
                    step(f"  baseline missing for {tgt}/{emb}/{baseline_run}")
                if show_chance:
                    chance = load_arm3_chance(
                        tgt, emb, baseline_run, results_root=arm3_results_root
                    )

            peak_df = pair[pair["phase"] == "peak"].copy()
            tc_df   = pair[pair["phase"] == "timecourse"].copy()

            parts.append(f"<h3>{tgt} / {emb}</h3>")

            parts.append("<h4>1. Peak-bin accuracy by arm</h4>")
            img, stats = plot_peak_bars(peak_df, baseline_df, tgt, emb,
                                        chance=chance)
            parts.append(img)
            if len(stats) > 0:
                all_stats.append(stats)
                parts.append(
                    "<details><summary>"
                    "Wilcoxon arm &gt; no_transfer stats"
                    "</summary>"
                )
                parts.append(
                    stats.to_html(index=False,
                                  float_format=lambda x: f"{x:.4g}")
                )
                parts.append("</details>")

            parts.append("<h4>2. Time course over alignment window</h4>")
            parts.append(
                plot_time_courses(
                    tc_df, peak_df, baseline_df, tgt, emb,
                    bin_size_ms=bin_size_ms, chance=chance,
                )
            )

            parts.append("<h4>3. Seen vs unseen vocabulary split</h4>")
            parts.append(plot_seen_vs_unseen(peak_df, tgt, emb))

    if all_stats:
        master = pd.concat(all_stats, ignore_index=True)
        parts.append("<h2>Master statistics table</h2>")
        parts.append(
            master.to_html(index=False, float_format=lambda x: f"{x:.4g}")
        )
        stats_csv = os.path.splitext(out_html)[0] + "_stats.csv"
        master.to_csv(stats_csv, index=False)
        parts.append(f"<p>Stats CSV: <code>{stats_csv}</code></p>")

    parts.append("</body></html>")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"  wrote {out_html}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    global METRICS
    parser = argparse.ArgumentParser(
        description="HTML report for cross_patient_aligned_semantic results."
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE_PATIENT)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGET_PATIENTS)
    parser.add_argument("--embeddings", nargs="+", default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--results-dir", default=None,
                        help="Directory containing the aligned_semantic CSVs "
                             "(default: main/test_results/).")
    parser.add_argument("--baseline-run", default=None,
                        help="Run folder under --arm3-results-root for the "
                             "single-patient semantic regression overlay.")
    parser.add_argument("--arm3-results-root", default=DEFAULT_ARM3_RESULTS_ROOT,
                        help="Root for semantic regression results "
                             "(default: main/results/semantic_regression).")
    parser.add_argument("--show-chance", dest="show_chance",
                        action="store_true", default=True)
    parser.add_argument("--no-show-chance", dest="show_chance",
                        action="store_false")
    parser.add_argument("--bin-size-ms", type=int, default=100,
                        help="Bin width in ms for time-axis labels (default 100).")
    parser.add_argument("--out", default=None,
                        help="Output HTML path.")
    parser.add_argument("--mode", choices=["unseen", "overall"], default="unseen",
                        help="'unseen': score test trials NOT in the train-word set. "
                             "'overall': all test trials.")
    args = parser.parse_args()

    METRICS = METRICS_UNSEEN if args.mode == "unseen" else METRICS_OVERALL
    import _archive.cross_patient_decoding.cross_patient_aligned_semantic_report as _self
    _self.METRICS = METRICS

    results_dir = args.results_dir or get_out_dir()
    csv_paths = []
    for t in args.targets:
        for e in args.embeddings:
            p = os.path.join(
                results_dir,
                f"cross_patient_aligned_semantic_{args.source}_to_{t}_{e}.csv",
            )
            if os.path.exists(p):
                csv_paths.append(p)
            else:
                step(f"  missing: {p}")
    if not csv_paths:
        raise SystemExit("No CSVs found; run cross_patient_aligned_semantic first.")

    out_html = args.out or os.path.join(
        results_dir,
        f"cross_patient_aligned_semantic_report_{args.mode}.html",
    )

    header("BUILDING CROSS-PATIENT ALIGNED SEMANTIC REPORT")
    print(f"  source            : {args.source}")
    print(f"  targets           : {args.targets}")
    print(f"  embeddings        : {args.embeddings}")
    print(f"  mode              : {args.mode}")
    print(f"  metrics           : {[m[0] for m in METRICS]}")
    print(f"  csv inputs        : {len(csv_paths)}")
    print(f"  baseline_run      : {args.baseline_run}")
    print(f"  arm3_results_root : {args.arm3_results_root}")
    print(f"  show_chance       : {args.show_chance}")
    print(f"  bin_size_ms       : {args.bin_size_ms}")
    print(f"  out html          : {out_html}")

    build_report(
        csv_paths,
        source_patient=args.source,
        targets=args.targets,
        embeddings=args.embeddings,
        baseline_run=args.baseline_run,
        out_html=out_html,
        arm3_results_root=args.arm3_results_root,
        show_chance=args.show_chance,
        bin_size_ms=args.bin_size_ms,
    )


if __name__ == "__main__":
    main()
