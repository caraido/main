# cross_task_channel_importance_report.py
# HTML report from cross_task_channel_importance.py CSV outputs.
#
# Inputs (from --in-dir, default: main/tests/results/cross_task_cotrain/):
#   channel_importance_all.csv
#   <PAT>/channel_importance_{PAT}_{metric_slug}.png
#   <PAT>/channel_jacobian_{PAT}_{metric_slug}.png
#
# Output (default):
#   <in-dir>/channel_importance_report.html
#
# Usage:
#   python -m main.tests.cross_task.cross_task_channel_importance_report
#   python -m main.tests.cross_task.cross_task_channel_importance_report --metric cat_indep_bal_acc --top 5
#   python -m main.tests.cross_task.cross_task_channel_importance_report --in-dir <dir> --out <out.html>

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

def _build_channel_map(pat: str, data_dir: Path) -> dict:
    """Return {csv_label: electrode_name} for a patient. Returns {} on failure.

    Resolution rules (from CLAUDE.md):
      AZ / LH / WBH  — ch{N}  → clean_channel_names[N] from *_channels.pkl
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
            clean = ch_df[ch_df["clean"]]["channel_name"].tolist()
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


# ---------------------------------------------------------------------------
# Per-patient channel tables
# ---------------------------------------------------------------------------

def _channel_table_pic(df_pat: pd.DataFrame, top_n: int) -> str:
    top = df_pat.sort_values("perm_imp_pic", ascending=False).head(top_n)
    thead = (
        "<thead><tr>"
        "<th>Rank</th><th>Channel</th>"
        "<th>&#916;acc&nbsp;(pic)</th><th>&#916;acc&nbsp;(aud)</th>"
        "<th>p_pic&nbsp;(raw)</th><th>q_pic&nbsp;(BH)</th>"
        "<th>Jac&nbsp;(pic)</th><th>Jac&nbsp;(aud)</th>"
        "</tr></thead>"
    )
    rows = []
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        rows.append(
            "<tr><td>{rank}</td>"
            "<td class='text{rcls}'>{ch}</td>"
            "{pv}{av}"
            "<td>{pp:.4f}</td><td>{qp:.4f}</td>"
            "<td>{jp:.3f}</td><td>{ja:.3f}</td></tr>".format(
                rank=rank,
                rcls=" top1" if rank == 1 else (" top2" if rank == 2 else ""),
                ch=r["channel"],
                pv=_delta_cell(r["perm_imp_pic"], rank),
                av=_delta_cell(r["perm_imp_aud"]),
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
        "<th>p_aud&nbsp;(raw)</th><th>q_aud&nbsp;(BH)</th>"
        "<th>Jac&nbsp;(pic)</th><th>Jac&nbsp;(aud)</th>"
        "</tr></thead>"
    )
    rows = []
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        rows.append(
            "<tr><td>{rank}</td>"
            "<td class='text{rcls}'>{ch}</td>"
            "{av}{pv}"
            "<td>{pa:.4f}</td><td>{qa:.4f}</td>"
            "<td>{jp:.3f}</td><td>{ja:.3f}</td></tr>".format(
                rank=rank,
                rcls=" top1" if rank == 1 else (" top2" if rank == 2 else ""),
                ch=r["channel"],
                av=_delta_cell(r["perm_imp_aud"], rank),
                pv=_delta_cell(r["perm_imp_pic"]),
                pa=r["p_aud"], qa=r["q_aud"],
                jp=r["jac_sens_pic"], ja=r["jac_sens_aud"],
            )
        )
    return label_suffix, "<table class='results'>{}<tbody>{}</tbody></table>".format(thead, "".join(rows))


# ---------------------------------------------------------------------------
# Cross-patient overview table
# ---------------------------------------------------------------------------

def section_overview(df: pd.DataFrame) -> str:
    rows = []
    for pat in sorted(df["patient"].unique()):
        dp = df[df["patient"] == pat]
        best_pic = dp.loc[dp["perm_imp_pic"].idxmax()]
        best_aud = dp.loc[dp["perm_imp_aud"].idxmax()]
        gc = _group_counts(dp)
        rows.append(
            "<tr>"
            "<td class='text'>{pat}</td><td>{n}</td>"
            "<td class='text top1'>{bpc}</td>{bpv}"
            "<td class='text top1'>{bac}</td>{bav}"
            "<td>{pp:.4f}</td>"
            "<td>{jp:.3f}</td><td>{ja:.3f}</td>"
            "<td>{both}</td><td>{po}</td><td>{ao}</td><td>{ne}</td>"
            "</tr>".format(
                pat=pat, n=len(dp),
                bpc=best_pic["channel"],
                bpv=_delta_cell(best_pic["perm_imp_pic"]),
                bac=best_aud["channel"],
                bav=_delta_cell(best_aud["perm_imp_aud"]),
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
        "<th>p_pic&nbsp;(raw)</th>"
        "<th>Jac&nbsp;(pic)</th><th>Jac&nbsp;(aud)</th>"
        "<th>both</th><th>pic_only</th><th>aud_only</th><th>neither</th>"
        "</tr></thead>"
    )
    note = (
        "<div class='box'>"
        "<b>How to read.</b>&nbsp;"
        "Best picture / auditory channel = highest mean &#916;acc on that task's test set "
        "across bootstraps. p_pic (raw) is the averaged per-bootstrap p-value for the best "
        "picture channel. Group counts (both / pic_only / aud_only / neither) reflect "
        "BH-FDR significance at &alpha;&nbsp;=&nbsp;0.05."
        "</div>"
    )
    return (
        "<h2>Cross-patient overview</h2>"
        + note
        + "<table class='results'>{}<tbody>{}</tbody></table>".format(thead, "".join(rows))
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
# Per-patient section
# ---------------------------------------------------------------------------

def section_patient(pat: str, df_pat, slug: str, top_n: int) -> str:
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
    )
    jac_img = _make_scatter(
        df_pat, "jac_sens_pic", "jac_sens_aud",
        "{} · Jacobian sensitivity (‖∂ŷ/∂x‖)".format(pat),
        "sensitivity picture", "sensitivity auditory",
    )
    return (
        "<h2>Patient {pat} &mdash; {n} channels</h2>"
        "<div class='pat-grid'>"
        "<div>"
        "<h3>Top channels &mdash; picture test set</h3>{pic_tbl}"
        "<h3>Top channels &mdash; auditory test set{aud_suffix}</h3>{aud_tbl}"
        "<p class='subtle'>Groups (BH-FDR &alpha;=0.05): {group_str}</p>"
        "</div>"
        "<div>"
        "<h3>Permutation importance scatter</h3>{imp_img}"
        "<h3>Jacobian sensitivity scatter</h3>{jac_img}"
        "</div>"
        "</div>"
    ).format(
        pat=pat, n=len(df_pat),
        pic_tbl=pic_tbl,
        aud_suffix=aud_suffix,
        aud_tbl=aud_tbl,
        group_str=group_str,
        imp_img=imp_img,
        jac_img=jac_img,
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
        "<b>Kernel-PLS limitation.</b>&nbsp;"
        "The Nystroem-RBF map distributes information across 100 landmark projections, diluting "
        "single-channel permutation effects. If attribution remains inconclusive, consider "
        "<code>--models pls</code> (linear PLS), for which analytic attribution via the "
        "coefficient matrix or VIP scores is exact and requires no permutation sampling."
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
    ap.add_argument("--data-dir", default=None,
                    help="Patient data directory containing *_channels.pkl files (default: main/data/)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    metric = args.metric
    slug = METRIC_SLUG.get(metric) or metric.replace("_", "")
    out_path = Path(args.out) if args.out else (in_dir / "channel_importance_report.html")

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

    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
    df_m = _apply_channel_maps(df_m, data_dir)

    patients = sorted(df_m["patient"].unique())
    print("Patients: {} | metric: {}".format(", ".join(patients), metric))

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
        "Significance: per-bootstrap p-values (each bootstrap's &#916;acc vs. that bootstrap's "
        "label-shuffle null), averaged across bootstraps, BH-FDR corrected at "
        "&alpha;&nbsp;=&nbsp;0.05. Electrode names are pre-resolved in "
        "<code>channel_importance_all.csv</code>."
        "</div>"
    ).format(metric=metric)

    per_patient_html = "\n".join(
        section_patient(pat, df_m[df_m["patient"] == pat].copy(), slug, args.top)
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
        "{per_pat}\n"
        "{ranking}\n"
        "{interp}\n"
        "<p class='subtle' style='margin-top:32px'>"
        "CSV source: <code>{all_csv}</code>. "
        "DR/RB integer channel indices resolved from picture-naming dataframe pkls; "
        "AZ/LH/WBH <code>ch{{N}}</code> labels resolved from <code>semantic_regression_results.pkl</code>."
        "</p>\n"
    ).format(
        gen=generated, src=str(in_dir), metric=metric,
        method=method,
        overview=section_overview(df_m),
        per_pat=per_patient_html,
        ranking=section_ranking(df_m),
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
