# -*- coding: utf-8 -*-
"""
tests/cross_patient_decoding/cross_patient_few_shot_report.py
=============================================================
HTML report from cross_patient_few_shot CSV + map-pkl outputs.

Sections per (target, embedding):
    1. Time-vs-accuracy curves
    2. Sample-efficiency at peak with Wilcoxon stats
    3. Seen vs unseen anchor-word split
    4. Transferred-PLS analysis (M_X SVD, quiver, rotation consistency)
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import warnings
from typing import List, Optional

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

from tests.cross_patient_decoding._cross_patient_helpers import (
    DEFAULT_SOURCE_PATIENT,
    DEFAULT_TARGET_PATIENTS,
    PHONEME_EMBEDDINGS,
    get_out_dir,
    load_arm3_baseline,
    load_map_records,
    header,
    step,
)
from sklearn.decomposition import PCA

ARM_COLORS = {"transfer": "#1f77b4", "no_transfer": "#d62728", "original": "#2ca02c"}
ARM_LABELS = {
    "transfer": "Arm 1: transfer",
    "no_transfer": "Arm 2: X-only (no transfer)",
    "original": "Arm 3: X full data (existing baseline)",
}
METRICS = [
    ("cosine_mean", "Cosine similarity"),
    ("word_bal_acc", "Word balanced accuracy"),
    ("cat_indep_bal_acc", "Category-independent balanced accuracy"),
]


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img_tag(fig, alt: str = "") -> str:
    return f'<img alt="{alt}" src="data:image/png;base64,{_fig_to_b64(fig)}" />'


# --- 1. Time-vs-accuracy curves ---

def plot_time_courses(df_pair, arm3_df, target, embedding,
                      bin_size_ms: int = 100, k_for_timecourse=None) -> str:
    if k_for_timecourse is None:
        ks = sorted(df_pair["k"].unique())
        k_for_timecourse = ks[len(ks) // 2]
    sub = df_pair[df_pair["k"] == k_for_timecourse].copy()
    fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 4), sharex=True)
    if len(METRICS) == 1:
        axes = [axes]
    for ax, (col, title) in zip(axes, METRICS):
        for arm in ["transfer", "no_transfer"]:
            arm_sub = sub[sub["arm"] == arm]
            if arm_sub.empty:
                continue
            agg = arm_sub.groupby("time_bin")[col].agg(["mean", "std", "count"])
            t_s = agg.index.values * bin_size_ms / 1000.0
            ax.plot(t_s, agg["mean"], lw=1.8, color=ARM_COLORS[arm], label=ARM_LABELS[arm])
            sem = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
            ax.fill_between(t_s, agg["mean"] - sem, agg["mean"] + sem,
                            color=ARM_COLORS[arm], alpha=0.18)
        if arm3_df is not None:
            arm3_col_map = {"cosine_mean": "cosine_mean",
                            "word_bal_acc": "word_balanced_acc",
                            "cat_indep_bal_acc": "category_balanced_acc"}
            base_col = arm3_col_map.get(col)
            if base_col and base_col in arm3_df.columns:
                t_s3 = arm3_df["bin_index"].values * bin_size_ms / 1000.0
                ax.plot(t_s3, arm3_df[base_col].values, lw=1.5, ls="--",
                        color=ARM_COLORS["original"], label=ARM_LABELS["original"])
        for pb in sub.loc[sub["is_peak"], "time_bin"].unique():
            ax.axvline(pb * bin_size_ms / 1000.0, ls=":", color="grey", alpha=0.6)
        ax.set_title(f"{title}  (k = {k_for_timecourse})", fontsize=10)
        ax.set_xlabel("Time bin (s)")
        ax.set_ylabel(title)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"{target}  /  {embedding}   - time courses at k={k_for_timecourse}", fontsize=12)
    fig.tight_layout()
    return _img_tag(fig, alt=f"timecourse_{target}_{embedding}")


# --- 2. Sample-efficiency at peak ---

def plot_sample_efficiency(df_pair, target, embedding):
    peak = df_pair[df_pair["is_peak"]].copy()
    ks = sorted(peak["k"].unique())
    fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 4))
    if len(METRICS) == 1:
        axes = [axes]
    stat_rows = []
    for ax, (col, title) in zip(axes, METRICS):
        for arm in ["transfer", "no_transfer"]:
            arm_sub = peak[peak["arm"] == arm]
            agg = arm_sub.groupby("k")[col].agg(["mean", "std", "count"])
            ax.plot(agg.index, agg["mean"], "-o", color=ARM_COLORS[arm],
                    label=ARM_LABELS[arm], lw=1.8, markersize=5)
            sem = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
            ax.fill_between(agg.index, agg["mean"] - sem, agg["mean"] + sem,
                            color=ARM_COLORS[arm], alpha=0.18)
        for k in ks:
            a1 = peak[(peak.k == k) & (peak.arm == "transfer")].sort_values("bootstrap_id")
            a2 = peak[(peak.k == k) & (peak.arm == "no_transfer")].sort_values("bootstrap_id")
            common = sorted(set(a1.bootstrap_id) & set(a2.bootstrap_id))
            if len(common) < 5:
                continue
            v1 = a1.set_index("bootstrap_id").loc[common, col].values
            v2 = a2.set_index("bootstrap_id").loc[common, col].values
            try:
                _, p = wilcoxon(v1, v2, alternative="greater", zero_method="zsplit")
            except ValueError:
                p = np.nan
            stat_rows.append(dict(target=target, embedding=embedding, metric=col, k=k,
                                  median_transfer=float(np.median(v1)),
                                  median_no_transfer=float(np.median(v2)),
                                  median_diff=float(np.median(v1 - v2)),
                                  n_paired=len(common),
                                  wilcoxon_p_one_sided=float(p) if np.isfinite(p) else np.nan))
        ax.set_xscale("symlog" if max(ks) > 30 else "linear")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks], fontsize=8)
        ax.set_xlabel("k (shots)")
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"{target} / {embedding}  - sample efficiency at peak", fontsize=12)
    fig.tight_layout()
    img = _img_tag(fig, alt=f"sample_eff_{target}_{embedding}")
    stat_df = pd.DataFrame(stat_rows)
    if len(stat_df) > 0:
        n_k = stat_df.groupby(["target", "embedding", "metric"])["k"].transform("nunique")
        stat_df["wilcoxon_p_bonf"] = (stat_df["wilcoxon_p_one_sided"] * n_k).clip(upper=1.0)
    return img, stat_df


# --- 3. Seen vs unseen ---

def plot_seen_vs_unseen(df_pair, target, embedding) -> str:
    peak = df_pair[df_pair["is_peak"]].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metric_cols = [
        ("cosine_seen", "cosine_unseen", "Cosine: seen vs unseen"),
        ("word_acc_seen", "word_acc_unseen", "Word acc: seen vs unseen"),
        ("cat_acc_seen", "cat_acc_unseen", "Cat acc: seen vs unseen"),
    ]
    for ax, (c_s, c_u, title) in zip(axes, metric_cols):
        for arm, marker in (("transfer", "o"), ("no_transfer", "x")):
            sub = peak[peak.arm == arm]
            agg = sub.groupby("k").agg(seen=(c_s, "mean"), unseen=(c_u, "mean"),
                                       seen_std=(c_s, "std"), unseen_std=(c_u, "std")).reset_index()
            ax.errorbar(agg["k"], agg["seen"], yerr=agg["seen_std"], marker=marker, ls="-",
                        color=ARM_COLORS[arm], label=f"{ARM_LABELS[arm]} (seen)", alpha=0.9, capsize=2)
            ax.errorbar(agg["k"], agg["unseen"], yerr=agg["unseen_std"], marker=marker, ls=":",
                        color=ARM_COLORS[arm], label=f"{ARM_LABELS[arm]} (unseen)", alpha=0.5, capsize=2)
        ax.set_xlabel("k")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle(f"{target} / {embedding}  - seen vs unseen anchor words", fontsize=12)
    fig.tight_layout()
    return _img_tag(fig, alt=f"seen_unseen_{target}_{embedding}")


# --- 4. Transferred-PLS analysis ---

def _records_for(records, arm, k):
    return [r for r in records if r["arm"] == arm and r["k"] == k]


def plot_svd_spectrum(records, target, embedding) -> str:
    transfer = [r for r in records if r["arm"] == "transfer"]
    if not transfer:
        return ""
    ks = sorted({r["k"] for r in transfer})
    fig, axes = plt.subplots(2, len(ks), figsize=(3.2 * len(ks), 6.0), squeeze=False)
    for j, k in enumerate(ks):
        sub = _records_for(transfer, "transfer", k)
        ax = axes[0, j]
        for r in sub:
            s = r["svd"]["s"]
            ax.plot(np.arange(1, len(s) + 1), s, color=ARM_COLORS["transfer"], alpha=0.25, lw=0.8)
        max_len = max(len(r["svd"]["s"]) for r in sub)
        spectra = np.full((len(sub), max_len), np.nan)
        for i, r in enumerate(sub):
            s = r["svd"]["s"]
            spectra[i, :len(s)] = s
        med = np.nanmedian(spectra, axis=0)
        ax.plot(np.arange(1, max_len + 1), med, color="black", lw=1.8, label="median")
        ax.set_title(f"k = {k} singular values", fontsize=9)
        ax.set_xlabel("rank index")
        ax.set_ylabel("singular value")
        ax.legend(fontsize=7)
        ax2 = axes[1, j]
        eff = [r["svd"]["effective_rank"] for r in sub]
        ax2.hist(eff, bins=20, color=ARM_COLORS["transfer"], alpha=0.7)
        ax2.axvline(np.median(eff), color="black", lw=1.6, label=f"median={np.median(eff):.2f}")
        ax2.set_title(f"k = {k} effective rank (PR)", fontsize=9)
        ax2.set_xlabel("participation ratio")
        ax2.legend(fontsize=7)
    fig.suptitle(f"{target} / {embedding} - SVD of M_X (Arm 1 transfer)", fontsize=11)
    fig.tight_layout()
    return _img_tag(fig, alt=f"svd_{target}_{embedding}")


def plot_quiver_anchors(payload, target, embedding, k_for_quiver=None) -> str:
    records = payload["records"]
    meta = payload["metadata"]
    transfer = [r for r in records if r["arm"] == "transfer"]
    if not transfer:
        return ""
    ks = sorted({r["k"] for r in transfer})
    if k_for_quiver is None:
        k_for_quiver = ks[len(ks) // 2]
    sub = _records_for(transfer, "transfer", k_for_quiver)
    if not sub:
        return ""
    T_full = meta["T_anchors_full"]
    src_words = list(T_full.keys())
    Tmat = np.stack([T_full[w] for w in src_words], axis=0)
    pca = PCA(n_components=2).fit(Tmat)
    T_anchors_2d = pca.transform(Tmat)
    src_word_to_2d = {w: T_anchors_2d[i] for i, w in enumerate(src_words)}
    fig, ax = plt.subplots(figsize=(8, 7))
    word_pred_2d = {}
    for r in sub:
        for i, w in enumerate(r["anchor_words"]):
            p2 = pca.transform(r["pred_anchors"][i:i + 1])[0]
            word_pred_2d.setdefault(str(w), []).append(p2)
    for w, p in src_word_to_2d.items():
        ax.scatter(p[0], p[1], s=30, color="#cccccc", edgecolors="none", zorder=1)
    cmap = plt.get_cmap("tab20", max(20, len(word_pred_2d)))
    for ci, (w, preds) in enumerate(word_pred_2d.items()):
        if w not in src_word_to_2d:
            continue
        col = cmap(ci % 20)
        P = np.stack(preds, axis=0)
        gt = src_word_to_2d[w]
        ax.scatter(P[:, 0], P[:, 1], s=14, color=col, alpha=0.35, edgecolors="none", zorder=2)
        cent = P.mean(axis=0)
        ax.annotate("", xy=gt, xytext=cent,
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.3, alpha=0.9), zorder=3)
        ax.scatter(*gt, marker="*", s=120, color=col, edgecolors="black", linewidths=0.6, zorder=4)
        ax.text(gt[0] * 1.02, gt[1] * 1.02, w, fontsize=7, color="black", alpha=0.75)
    ax.axhline(0, ls=":", color="grey", lw=0.5)
    ax.axvline(0, ls=":", color="grey", lw=0.5)
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 of T_RB ({var[0]:.0%})")
    ax.set_ylabel(f"PC2 of T_RB ({var[1]:.0%})")
    ax.set_title(f"{target} / {embedding}  -  k = {k_for_quiver}\n"
                 f"Predicted X->T_RB (clouds + arrows) vs ground-truth T_RB (star)", fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return _img_tag(fig, alt=f"quiver_{target}_{embedding}_k{k_for_quiver}")


def plot_rotation_consistency(records, target, embedding) -> str:
    transfer = [r for r in records if r["arm"] == "transfer"]
    if not transfer:
        return ""
    ks = sorted({r["k"] for r in transfer})
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in ks:
        sub = _records_for(transfer, "transfer", k)
        U1 = np.stack([r["svd"]["U"][:, 0] for r in sub], axis=0)
        U1n = U1 / (np.linalg.norm(U1, axis=1, keepdims=True) + 1e-12)
        sim = np.abs(U1n @ U1n.T)
        iu = np.triu_indices(len(sim), k=1)
        ax.hist(sim[iu], bins=25, alpha=0.55, label=f"k = {k}")
    ax.set_xlabel("|cosine| of top singular direction across bootstrap pairs")
    ax.set_ylabel("pair count")
    ax.set_title(f"{target} / {embedding} - rotation consistency (Arm 1)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _img_tag(fig, alt=f"rotation_{target}_{embedding}")


# --- top-level report ---

def _maybe_load_maps(csv_path):
    pkl_path = csv_path.replace(".csv", "_maps.pkl")
    if not os.path.exists(pkl_path):
        return None
    try:
        return load_map_records(pkl_path)
    except Exception as e:
        step(f"  failed to load map pkl {pkl_path}: {e}")
        return None


def build_report(csv_paths, source_patient, targets, embeddings,
                 baseline_run, out_html, quiver_k=None):
    all_df = []
    for p in csv_paths:
        try:
            all_df.append(pd.read_csv(p))
        except Exception as e:
            step(f"  Failed to read {p}: {e}")
    if not all_df:
        raise RuntimeError("No CSVs loaded; nothing to report.")
    df = pd.concat(all_df, ignore_index=True)
    parts = ["<html><head><meta charset='utf-8'>"
             "<title>Cross-Patient Few-Shot Transfer Report</title>"
             "<style>body{font-family:system-ui,sans-serif;margin:24px;color:#222;max-width:1400px}"
             "h1{border-bottom:2px solid #1f77b4}h2{margin-top:36px;color:#1f4a72;"
             "border-bottom:1px solid #ccc;padding-bottom:4px}h3{color:#666;margin-top:24px}"
             "table{border-collapse:collapse;margin:8px 0;font-size:13px}"
             "th,td{border:1px solid #ccc;padding:4px 8px;text-align:right}"
             "th{background:#f0f0f0;text-align:center}img{max-width:100%;height:auto;margin:8px 0}"
             "code{background:#f4f4f4;padding:2px 4px;border-radius:3px}</style></head><body>"]
    parts.append("<h1>Cross-Patient Few-Shot Transfer Learning Report</h1>")
    parts.append(f"<p>Source: <b>{source_patient}</b> Targets: <b>{', '.join(targets)}</b> "
                 f"Embeddings: <b>{', '.join(embeddings)}</b></p>")
    all_stats = []
    for tgt in targets:
        parts.append(f"<h2>Target: {tgt}</h2>")
        for emb in embeddings:
            pair = df[(df.target_patient == tgt) & (df.embedding == emb)]
            if pair.empty:
                parts.append(f"<p><i>No data for {tgt} / {emb}.</i></p>")
                continue
            arm3_df = None
            if baseline_run is not None:
                arm3_df = load_arm3_baseline(tgt, emb, baseline_run)
                if arm3_df is None:
                    step(f"  Arm 3 missing for {tgt}/{emb}/{baseline_run}.")
            parts.append(f"<h3>{tgt} / {emb}</h3>")
            parts.append("<h4>1. Time courses</h4>")
            parts.append(plot_time_courses(pair, arm3_df, tgt, emb))
            parts.append("<h4>2. Sample efficiency at peak</h4>")
            img, stats = plot_sample_efficiency(pair, tgt, emb)
            parts.append(img)
            if len(stats) > 0:
                all_stats.append(stats)
                parts.append("<details><summary>Wilcoxon Arm1 &gt; Arm2 stats</summary>")
                parts.append(stats.to_html(index=False, float_format=lambda x: f"{x:.4g}"))
                parts.append("</details>")
            parts.append("<h4>3. Seen vs unseen anchor words</h4>")
            parts.append(plot_seen_vs_unseen(pair, tgt, emb))
            csv_for_pair = next(
                (p for p in csv_paths if (f"_{tgt}_" in os.path.basename(p))
                 and p.endswith(f"_{emb}.csv")), None,
            )
            maps_payload = _maybe_load_maps(csv_for_pair) if csv_for_pair else None
            if maps_payload is not None:
                parts.append("<h4>4. Transferred-PLS analysis (M_X SVD + quiver)</h4>")
                parts.append("<h5>4a. Singular value spectrum + effective rank</h5>")
                parts.append(plot_svd_spectrum(maps_payload["records"], tgt, emb))
                parts.append("<h5>4b. 2-D quiver of predicted T-hat vs source T_RB</h5>")
                parts.append(plot_quiver_anchors(maps_payload, tgt, emb, k_for_quiver=quiver_k))
                parts.append("<h5>4c. Rotation consistency across bootstraps</h5>")
                parts.append(plot_rotation_consistency(maps_payload["records"], tgt, emb))
            else:
                parts.append("<p><i>Map pickle not found; rerun with --save-maps.</i></p>")
    if all_stats:
        master_stats = pd.concat(all_stats, ignore_index=True)
        parts.append("<h2>Master stats table</h2>")
        parts.append(master_stats.to_html(index=False, float_format=lambda x: f"{x:.4g}"))
        stats_csv = os.path.splitext(out_html)[0] + "_stats.csv"
        master_stats.to_csv(stats_csv, index=False)
        parts.append(f"<p>Stats CSV: <code>{stats_csv}</code></p>")
    parts.append("</body></html>")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"  wrote {out_html}")


def main():
    parser = argparse.ArgumentParser(description="Build HTML report.")
    parser.add_argument("--source", default=DEFAULT_SOURCE_PATIENT)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGET_PATIENTS)
    parser.add_argument("--embeddings", nargs="+", default=PHONEME_EMBEDDINGS)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--baseline-run", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--quiver-k", type=int, default=None)
    args = parser.parse_args()
    results_dir = args.results_dir or get_out_dir()
    csv_paths = []
    for t in args.targets:
        for e in args.embeddings:
            p = os.path.join(results_dir,
                             f"cross_patient_few_shot_{args.source}_to_{t}_{e}.csv")
            if os.path.exists(p):
                csv_paths.append(p)
            else:
                step(f"  missing: {p}")
    if not csv_paths:
        raise SystemExit("No CSVs found; run cross_patient_few_shot first.")
    out_html = args.out or os.path.join(results_dir, "cross_patient_few_shot_report.html")
    header("BUILDING CROSS-PATIENT FEW-SHOT REPORT")
    print(f"  csv inputs : {len(csv_paths)}")
    print(f"  baseline   : {args.baseline_run}")
    print(f"  out html   : {out_html}")
    build_report(csv_paths, source_patient=args.source, targets=args.targets,
                 embeddings=args.embeddings, baseline_run=args.baseline_run,
                 out_html=out_html, quiver_k=args.quiver_k)


if __name__ == "__main__":
    main()
