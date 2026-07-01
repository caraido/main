# cross_task_cotrain_report.py
# HTML report from cross_task_cotrain.py CSV outputs.  Same visual style as
# cross_task_transfer_report.py.
#
# Sections:
#   1. Cross-patient overview -- word_bal_acc / cosine per (eval-target x train-source)
#      grouped by patient, with within/cross/pooled pairwise significance stars.
#      Within-patient tests use the Nadeau-Bengio CORRECTED RESAMPLED t-test:
#      the bootstrap resamples are overlapping train/test re-splits of one dataset,
#      so a plain paired t / Wilcoxon is anti-conservative (p shrinks as J grows);
#      the variance is inflated by (1/J + n_test/n_train) to correct for it.
#   2. Across-patient group summary -- within/cross/pooled grand means with
#      paired-BY-PATIENT significance (Wilcoxon signed-rank + paired t-test;
#      patients ARE independent, so an ordinary paired test is valid here).
#   3. Q1 (shared representation): cross-vs-within retention + RSA of per-word
#      neural geometry across tasks.
#   4. Q3 (one decoder for both): pooled-vs-within gain.
#   5. Q2 (amodal electrodes): per-electrode RSA(pic) vs RSA(aud), top channels.
#   6. Per-patient detail -- all metrics x (target,source) with significance stars
#      (corrected resampled t-test), seen/unseen split, and saved figures.
#
# Inputs: a single run folder produced by cross_task_cotrain.py. If --in-dir is
# the parent (main/tests/results/cross_task_cotrain/), the LATEST run subfolder is
# selected automatically; pass --in-dir <run> to report on a specific run.
#   <run>/cotrain_conditions_summary.csv, cotrain_rsa_summary.csv
#   <run>/<patient>/cotrain_conditions_<patient>.csv  (per-bootstrap rows)
#   <run>/<patient>/cotrain_electrodes_<patient>.csv
#   <run>/<patient>/cotrain_<patient>_bars.png, cotrain_<patient>_electrodes.png
#
# Output (written into the resolved run folder):
#   <run>/cross_task_cotrain_report.html
#
# Usage:
#   python -m main.tests.cross_task.cross_task_cotrain_report                 # latest run
#   python -m main.tests.cross_task.cross_task_cotrain_report --in-dir <run> --model kernel_pls

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
from scipy.stats import wilcoxon, ttest_rel, t as t_dist

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
# Significance annotation on figures
# ---------------------------------------------------------------------------

def _p_to_stars(p) -> str:
    """p-value -> star code; '' when the test could not be run, 'n.s.' otherwise."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _draw_sig_brackets(ax, xs, tops, pair_p, show_ns: bool = True) -> float:
    """Draw stacked significance brackets between bars on *ax*.

    xs      : bar-center x positions
    tops    : per-bar top y (bar height + error) used as the baseline
    pair_p  : {(i, j): p_value} for the bars to compare
    Returns the highest y drawn (so the caller can expand ylim for headroom).
    """
    finite = [t for t in tops if np.isfinite(t)]
    if not finite or not pair_p:
        return 0.0
    y0 = max(finite)
    # Scale step with the data so brackets stay compact regardless of metric range.
    # Old floor of 0.02 was too large when bars are small (e.g. word_bal_acc ~0.06).
    step = 0.08 * (abs(y0) if y0 else 1.0)
    # adjacent comparisons first (lower), spanning comparisons higher
    pairs = sorted(pair_p.keys(), key=lambda k: (k[1] - k[0], k[0]))
    lvl = 0
    top_used = y0
    for (i, j) in pairs:
        stars = _p_to_stars(pair_p[(i, j)])
        if stars == "" or (stars == "n.s." and not show_ns):
            continue
        y = y0 + step * (1.2 + 1.6 * lvl)
        ax.plot([xs[i], xs[i], xs[j], xs[j]],
                [y, y + step * 0.3, y + step * 0.3, y], lw=1.0, color="#444")
        is_sig = stars != "n.s."
        ax.text((xs[i] + xs[j]) / 2.0, y + step * 0.32, stars, ha="center", va="bottom",
                fontsize=11 if is_sig else 8,
                color="#2E7D32" if is_sig else "#9E9E9E",
                fontweight="bold" if is_sig else "normal")
        top_used = y + step * 1.2
        lvl += 1
    return top_used


# Fraction of trials held out as test in each resample (read from run_metadata.json
# in main(); used only as a fallback when a run's CSV lacks the n_test column).
_FALLBACK_TEST_FRAC = 0.3


def _corrected_resampled_t(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled paired t-test.

    The J resamples are overlapping random train/test re-splits of one fixed
    dataset, so the per-resample paired differences are NOT independent and a
    plain paired t / Wilcoxon is anti-conservative (p shrinks as J grows). The
    correction inflates the variance of the mean difference by (1/J + n_test/
    n_train) to account for the train-set overlap between resamples.

    a, b    : paired per-resample scores (same test set within each resample)
    n_train : mean training-set size across the resamples (per condition pair)
    n_test  : mean (shared) test-set size across the resamples
    Returns a two-sided p-value (Student-t, df = J-1)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    J = d.size
    if J < 2 or not (n_train and n_train > 0):
        return np.nan
    dbar = d.mean()
    var = d.var(ddof=1)
    if var <= 0:
        return 0.0 if dbar != 0 else 1.0
    rho = float(n_test) / float(n_train)          # train/test overlap correction
    denom = np.sqrt((1.0 / J + rho) * var)
    if denom == 0:
        return np.nan
    tstat = dbar / denom
    return float(2.0 * t_dist.sf(abs(tstat), df=J - 1))


def _pairwise_p(df, target, metric):
    """Pairwise significance between train sources, paired by bootstrap and
    corrected for resample overlap (Nadeau-Bengio). Returns {(i, j): p}."""
    has_ntest = "n_test" in df.columns
    cols = ["bootstrap_id", metric, "n_train"] + (["n_test"] if has_ntest else [])
    out = {}
    for i in range(len(SRC_ORDER)):
        for j in range(i + 1, len(SRC_ORDER)):
            a = df[df["condition"] == COND[(target, SRC_ORDER[i])]][cols]
            b = df[df["condition"] == COND[(target, SRC_ORDER[j])]][cols]
            m = pd.merge(a, b, on="bootstrap_id", suffixes=("_a", "_b")).dropna(
                subset=[metric + "_a", metric + "_b"])
            if len(m) < 3:
                out[(i, j)] = np.nan
                continue
            # mean train size over both conditions; shared test set within the pair
            n_train = float(np.nanmean(np.concatenate(
                [m["n_train_a"].values, m["n_train_b"].values])))
            if has_ntest:
                n_test = float(np.nanmean(m["n_test_a"].values))
            else:
                n_test = n_train * (_FALLBACK_TEST_FRAC / (1.0 - _FALLBACK_TEST_FRAC))
            out[(i, j)] = _corrected_resampled_t(
                m[metric + "_a"].values, m[metric + "_b"].values, n_train, n_test)
    return out


def _group_patient_means(per_pat: dict, target: str, metric: str):
    """For one (target, metric): each patient's mean per train source.
    Returns (patients, {src: [per-patient mean,...]})."""
    patients = sorted(per_pat)
    data = {src: [] for src in SRC_ORDER}
    for p in patients:
        for src in SRC_ORDER:
            v = _vals(per_pat[p], COND[(target, src)], metric)
            data[src].append(float(np.mean(v)) if len(v) else np.nan)
    return patients, data


def _paired_across_patients(data: dict):
    """Paired-by-patient tests between train-source groups.
    data: {src: [per-patient mean]}. Returns {(i, j): (p_wilcoxon, p_ttest, n)}."""
    out = {}
    for i in range(len(SRC_ORDER)):
        for j in range(i + 1, len(SRC_ORDER)):
            a = np.asarray(data[SRC_ORDER[i]], float)
            b = np.asarray(data[SRC_ORDER[j]], float)
            mask = np.isfinite(a) & np.isfinite(b)
            a, b = a[mask], b[mask]
            n = int(len(a))
            pw = pt = np.nan
            if n >= 2:
                try:
                    _, pt = ttest_rel(a, b)
                except Exception:
                    pt = np.nan
                try:
                    _, pw = wilcoxon(a, b, zero_method="zsplit")
                except ValueError:
                    pw = np.nan
            out[(i, j)] = (float(pw) if np.isfinite(pw) else np.nan,
                           float(pt) if np.isfinite(pt) else np.nan, n)
    return out


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
    width = 0.8 / len(SRC_ORDER)
    offsets = [(i - len(SRC_ORDER) / 2 + 0.5) * width for i in range(len(SRC_ORDER))]

    fig, axes = plt.subplots(1, 2, figsize=(max(9, 2.7 * n_pat), 4.8), sharey=True)
    y_ceil = 0.0
    for ax, target in zip(axes, TARGETS):
        x = np.arange(n_pat)
        # First pass: draw bars and collect per-patient, per-source tops
        tops_all = [[np.nan] * len(SRC_ORDER) for _ in range(n_pat)]
        for i, src in enumerate(SRC_ORDER):
            cond = COND[(target, src)]
            means, sems = [], []
            for pi, p in enumerate(patients):
                v = _vals(per_pat[p], cond, metric)
                m = float(np.mean(v)) if len(v) else np.nan
                s = float(np.std(v) / np.sqrt(max(1, len(v))))
                means.append(m); sems.append(s)
                tops_all[pi][i] = (m + s) if np.isfinite(m) else np.nan
            ax.bar(x + offsets[i], np.nan_to_num(means), width, yerr=sems,
                   color=SRC_COLORS[src], alpha=0.85, label=SRC_LABELS[src],
                   capsize=3, error_kw={"lw": 1.1})
        # Second pass: draw per-patient significance brackets
        for pi, p in enumerate(patients):
            pair_p = _pairwise_p(per_pat[p], target, metric)
            xs_local = [x[pi] + off for off in offsets]
            need = _draw_sig_brackets(ax, xs_local, tops_all[pi], pair_p, show_ns=True)
            y_ceil = max(y_ceil, need)
        ax.set_xticks(x); ax.set_xticklabels(patients, fontsize=9)
        ax.set_xlabel("Patient"); ax.set_ylabel(metric_label)
        ax.set_title(TARGET_LABELS[target], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    if y_ceil > 0:
        axes[0].set_ylim(0, y_ceil * 1.08)  # sharey=True propagates to right panel
    fig.suptitle("{} — brackets: within/cross/pooled pairwise (corrected resampled t-test)".format(metric_label),
                 fontsize=10)
    fig.tight_layout()
    return _img_tag(fig, alt="overview_" + metric)


def overview_sig_table(per_pat: dict, metric: str) -> str:
    """Per-patient, per-target pairwise significance table (stars + p-value)."""
    PAIRS = [(0, 1, "within vs cross"), (0, 2, "within vs pooled"), (1, 2, "cross vs pooled")]
    rows = []
    for p in sorted(per_pat):
        for target in TARGETS:
            pair_p = _pairwise_p(per_pat[p], target, metric)
            row = {"patient": p, "target": target}
            for i, j, label in PAIRS:
                pv = pair_p.get((i, j), np.nan)
                stars = _p_to_stars(pv)
                pstr = "{:.4f}".format(pv) if np.isfinite(pv) else "—"
                row[label] = "{} ({})".format(stars, pstr) if stars else "— ({})".format(pstr)
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return ""
    # Render manually so star cells can be coloured
    head = "".join("<th>{}</th>".format(c) for c in df.columns)
    body_rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = str(r[c])
            if c in ("patient", "target"):
                cells.append("<td class='text'>{}</td>".format(v))
            else:
                is_sig = v.startswith("*")
                cls = "sig" if is_sig else "ns"
                cells.append("<td class='{}'>{}</td>".format(cls, v))
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return ("<table class='results'><thead><tr>" + head + "</tr></thead><tbody>"
            + "\n".join(body_rows) + "</tbody></table>")


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


def fig_group_summary(per_pat: dict, metric: str, metric_label: str):
    """Across-patient group summary for one metric.

    Each patient contributes one mean per train source (within/cross/pooled);
    bars show the grand mean across patients (±SEM across patients), patient
    points are overlaid and connected, and brackets mark paired-by-patient
    significance.  Returns (img_html, stat_df)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    stat_rows = []
    for ax, target in zip(axes, TARGETS):
        _, data = _group_patient_means(per_pat, target, metric)
        arr = np.array([data[s] for s in SRC_ORDER], float)  # (n_src, n_pat)
        x = np.arange(len(SRC_ORDER))
        gmeans = [np.nanmean(arr[i]) for i in range(len(SRC_ORDER))]
        gsems = [float(np.nanstd(arr[i]) / np.sqrt(max(1, np.sum(np.isfinite(arr[i])))))
                 for i in range(len(SRC_ORDER))]
        ax.bar(x, np.nan_to_num(gmeans), 0.62, yerr=gsems,
               color=[SRC_COLORS[s] for s in SRC_ORDER], alpha=0.85, capsize=4,
               error_kw={"lw": 1.3})
        # per-patient points + connecting lines (paired view)
        for pi in range(arr.shape[1]):
            ax.plot(x, arr[:, pi], color="#555", lw=0.7, alpha=0.5,
                    marker="o", ms=3.5, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([SRC_LABELS[s].split(" ")[0] for s in SRC_ORDER], fontsize=9)
        ax.set_title(TARGET_LABELS[target], fontsize=10)
        ax.set_ylabel(metric_label); ax.grid(axis="y", alpha=0.3)
        # paired-by-patient significance brackets (Wilcoxon for stars)
        pair = _paired_across_patients(data)
        tops = [np.nanmax(arr[i]) if np.isfinite(arr[i]).any() else np.nan
                for i in range(len(SRC_ORDER))]
        need = _draw_sig_brackets(ax, x, tops, {k: v[0] for k, v in pair.items()})
        if need:
            ax.set_ylim(top=max(ax.get_ylim()[1], need))
        for (i, j), (pw, pt, n) in pair.items():
            stat_rows.append({"target": target,
                              "comparison": "{} vs {}".format(SRC_ORDER[i], SRC_ORDER[j]),
                              "metric": metric, "n_patients": n,
                              "wilcoxon_p": pw, "ttest_p": pt})
    fig.suptitle("Across-patient group summary — {}  (stars = paired Wilcoxon across patients)".format(metric_label),
                 fontsize=10)
    fig.tight_layout()
    return _img_tag(fig, alt="group_" + metric), pd.DataFrame(stat_rows)


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


def paired_resampled_t(df, cond_a, cond_b, label_a, label_b):
    """Paired (by bootstrap_id) Nadeau-Bengio corrected resampled t-test per
    metric: a vs b, two-sided. Corrects for the overlap between resamples."""
    out = []
    has_ntest = "n_test" in df.columns
    a = df[df["condition"] == cond_a]; b = df[df["condition"] == cond_b]
    merged = pd.merge(a, b, on="bootstrap_id", suffixes=("_a", "_b"))
    n_train = float(np.nanmean(np.concatenate(
        [merged["n_train_a"].values, merged["n_train_b"].values]))) if len(merged) else np.nan
    n_test = (float(np.nanmean(merged["n_test_a"].values)) if has_ntest and len(merged)
              else n_train * (_FALLBACK_TEST_FRAC / (1.0 - _FALLBACK_TEST_FRAC)))
    for col, _ in METRICS:
        m = merged[[col + "_a", col + "_b"]].dropna()
        n = len(m)
        if n < 3:
            continue
        va, vb = m[col + "_a"].values, m[col + "_b"].values
        p = _corrected_resampled_t(va, vb, n_train, n_test)
        out.append({"comparison": "{} vs {}".format(label_a, label_b), "metric": col,
                    "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
                    "mean_diff": float((va - vb).mean()), "n": n,
                    "corrected_t_p": float(p) if np.isfinite(p) else np.nan})
    return out


def _wilcoxon_html(stat_df) -> str:
    if stat_df is None or stat_df.empty:
        return "<p class='subtle'>(insufficient bootstraps)</p>"
    p_cols = {c for c in stat_df.columns if c.endswith("_p")}
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
    nM = len(METRICS)
    fig, axes = plt.subplots(2, nM, figsize=(4.8 * nM, 8.8), sharey="col")
    col_topneed = [0.0] * nM
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
            # significance brackets (Nadeau-Bengio corrected resampled t between sources)
            tops = [(m + s) if np.isfinite(m) else np.nan for m, s in zip(means, sems)]
            need = _draw_sig_brackets(ax, x, tops, _pairwise_p(df, target, col))
            col_topneed[col_i] = max(col_topneed[col_i], need)
    for col_i in range(nM):
        if col_topneed[col_i] > 0:
            axes[0, col_i].set_ylim(top=col_topneed[col_i])  # sharey='col' propagates
    fig.suptitle("Patient {}: co-training conditions  "
                 "(*** p<.001, ** p<.01, * p<.05, corrected resampled t-test)".format(pat),
                 fontsize=10)
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
    sig_note = ("Brackets show pairwise comparisons (within vs cross, within vs pooled, "
                "cross vs pooled) within each patient. Stars: <b class='sig'>***</b> p&lt;.001, "
                "<b class='sig'>**</b> p&lt;.01, <b class='sig'>*</b> p&lt;.05, "
                "<span class='ns'>n.s.</span> not significant. "
                "Test: <b>Nadeau-Bengio corrected resampled t-test</b> — a paired t-test on the "
                "per-resample score differences whose variance is inflated by (1/J + n_test/n_train) "
                "to account for the overlap between the random train/test re-splits (a plain "
                "paired t / Wilcoxon would be anti-conservative here, since the resamples are not "
                "independent). P-values in tables below each figure.")
    return (
        "<h2>Cross-patient overview</h2>"
        "<div class='box'><b>How to read this.</b> For each evaluation target (picture or "
        "auditory test trials), three decoders are compared: <b>within</b> (trained on the "
        "same task), <b>cross</b> (trained on the other task only), and <b>pooled</b> "
        "(trained on both tasks). Error bars are bootstrap SEM. "
        "<i>cross≈within</i> &rarr; shared representation (Q1); "
        "<i>pooled≥within</i> &rarr; one decoder serves both (Q3). " + sig_note + "</div>"
        + _legend_html()
        + "<h3>Word balanced accuracy</h3>"
        + fig_overview(per_pat, "word_bal_acc", "Word balanced accuracy")
        + "<details open><summary>Pairwise significance table — word balanced accuracy</summary>"
        + overview_sig_table(per_pat, "word_bal_acc") + "</details>"
        + "<h3>Category-independent balanced accuracy</h3>"
        + fig_overview(per_pat, "cat_indep_bal_acc", "Category-independent balanced accuracy")
        + "<details><summary>Pairwise significance table — category-independent balanced accuracy</summary>"
        + overview_sig_table(per_pat, "cat_indep_bal_acc") + "</details>"
        + "<h3>Cosine similarity (predicted vs true embedding)</h3>"
        + fig_overview(per_pat, "cosine_mean", "Cosine similarity")
        + "<details><summary>Pairwise significance table — cosine similarity</summary>"
        + overview_sig_table(per_pat, "cosine_mean") + "</details>"
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


def section_group_summary(per_pat: dict) -> str:
    parts = [
        "<h2>Across-patient group summary (within vs cross vs pooled)</h2>",
        "<div class='box'><b>How to read this.</b> Each patient is collapsed to one mean "
        "value per train source; bars are the <b>grand mean across patients</b> (error bars "
        "= SEM across the {} patients), with each patient's value overlaid as connected dots "
        "(paired view). Brackets show paired-by-patient tests between the three groups "
        "(<b>*** p&lt;.001, ** p&lt;.01, * p&lt;.05</b>; stars use the Wilcoxon signed-rank "
        "test, paired t-test p is in the table). With only ~6 patients the signed-rank test "
        "is conservative (its smallest possible two-sided p is ~0.03).</div>".format(len(per_pat)),
    ]
    all_stats = []
    for metric, label in METRICS:
        img, sdf = fig_group_summary(per_pat, metric, label)
        parts.append("<h3>{}</h3>".format(label) + img)
        if not sdf.empty:
            all_stats.append(sdf)
    if all_stats:
        parts.append("<details open><summary>Paired across-patient tests "
                     "(Wilcoxon signed-rank + paired t-test)</summary>"
                     + _wilcoxon_html(pd.concat(all_stats, ignore_index=True)) + "</details>")
    return "".join(parts)


def section_per_patient(in_dir: Path, per_pat: dict, elec: dict) -> str:
    blocks, all_stats = [], []
    for pat, df in sorted(per_pat.items()):
        stats = []
        for target in TARGETS:
            stats += paired_resampled_t(df, COND[(target, "cross")], COND[(target, "within")],
                                        "cross_" + target, "within_" + target)
            stats += paired_resampled_t(df, COND[(target, "pooled")], COND[(target, "within")],
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
            + "<details><summary>Corrected resampled t-test (cross/pooled vs within)</summary>"
            + _wilcoxon_html(stat_df) + "</details>"
            + "<details><summary>Per-electrode amodal scatter (saved figure)</summary>" + elec_png + "</details>"
        )
        blocks.append(block)
    master = ""
    if all_stats:
        master = "<h2>Master corrected resampled t-test (all patients)</h2>" + _wilcoxon_html(pd.concat(all_stats, ignore_index=True))
    return "\n".join(blocks) + master


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _resolve_run_dir(in_dir: Path) -> Path:
    """Point at an actual run folder.

    cross_task_cotrain.py now writes each run into its own timestamped subfolder.
    If ``in_dir`` is already a run folder (contains the summary CSV) use it as-is;
    otherwise pick the most recent run subfolder (names are timestamp-prefixed, so
    lexical max = newest).  Falls back to ``in_dir`` unchanged for legacy layouts.
    """
    # A real run folder carries run_metadata.json; the legacy flat root does not,
    # so check that first to avoid leftover root-level CSVs shadowing newer runs.
    if (in_dir / "run_metadata.json").exists():
        return in_dir
    runs = [d for d in in_dir.iterdir()
            if d.is_dir() and (d / "run_metadata.json").exists()]
    if runs:
        chosen = max(runs, key=lambda d: d.name)
        print("[cotrain_report] using latest run:", chosen.name)
        return chosen
    # Legacy flat layout (pre run-folder) — use in_dir as-is if it has summaries.
    return in_dir


def main() -> int:
    p = argparse.ArgumentParser(description="HTML report for cross_task_cotrain results")
    p.add_argument("--in-dir", default=str(DEFAULT_IN_DIR))
    p.add_argument("--out", default=None)
    p.add_argument("--model", default=None, help="Filter to one model (default: first present)")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    # Allow passing a bare run-folder name (resolved under the default results dir).
    if not in_dir.exists() and (DEFAULT_IN_DIR / args.in_dir).exists():
        in_dir = DEFAULT_IN_DIR / args.in_dir
    if not in_dir.exists():
        print("ERROR: in-dir does not exist:", in_dir); return 1
    in_dir = _resolve_run_dir(in_dir)
    out_path = Path(args.out) if args.out else (in_dir / "cross_task_cotrain_report.html")

    # Read test_frac from this run's metadata for the corrected-t fallback
    # (only used when a run's CSV predates the n_test column).
    meta_path = in_dir / "run_metadata.json"
    if meta_path.exists():
        try:
            import json
            global _FALLBACK_TEST_FRAC
            _FALLBACK_TEST_FRAC = float(json.loads(meta_path.read_text())["test_frac"])
        except Exception:
            pass

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
        + section_group_summary(per_pat)
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
