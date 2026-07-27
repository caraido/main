# -*- coding: utf-8 -*-
"""Assemble auditory_alignment_report.html — the side-by-side alignment comparison.

Reads the cached per-cell npz/meta (via aggregate.load_all), builds every figure, embeds
them as inline base64 PNGs (report/helper/html_utils.fig_to_base64), writes the plotted
source CSVs, and lays out the sections. Report + figures + source_data all live under
results/auditory_alignment/ (self-contained pilot output).
"""

import os
import sys
import datetime as _dt

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAIN_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)
_FIGS_FOR_PAPER = os.path.join(_MAIN_DIR, "figures_for_paper")
if _FIGS_FOR_PAPER not in sys.path:
    sys.path.insert(0, _FIGS_FOR_PAPER)

from utils.paths import results_dir                        # noqa: E402
from report.helper.html_utils import fig_to_base64         # noqa: E402
from tests.auditory_alignment import config                # noqa: E402
from tests.auditory_alignment import aggregate as A        # noqa: E402
from tests.auditory_alignment import stats as S            # noqa: E402
from tests.auditory_alignment import figures as F          # noqa: E402
from paper_common import display_id                        # noqa: E402


CSS = """
<style>
 body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
        margin: 24px 32px; color: #1a1a1a; max-width: 1500px; line-height: 1.45; }
 h1 { font-size: 22px; border-bottom: 2px solid #333; padding-bottom: 6px; }
 h2 { font-size: 17px; margin-top: 34px; border-bottom: 1px solid #ccc; padding-bottom: 3px; }
 h3 { font-size: 14px; margin-top: 20px; color: #333; }
 p, li { font-size: 13px; }
 .note { background: #f5f7fa; border-left: 4px solid #4a7; padding: 8px 14px; margin: 12px 0;
         font-size: 12.5px; }
 .warn { background: #fff6f0; border-left: 4px solid #e07a3f; padding: 8px 14px; margin: 12px 0;
         font-size: 12.5px; }
 img { max-width: 100%; height: auto; display: block; margin: 8px 0 4px; }
 table.t { border-collapse: collapse; margin: 10px 0; font-size: 12px; }
 table.t th, table.t td { border: 1px solid #ccc; padding: 3px 8px; text-align: right; }
 table.t th { background: #eef1f4; }
 table.t td.l, table.t th.l { text-align: left; }
 .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: start; }
 .muted { color: #666; font-size: 11.5px; }
 code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 11.5px; }
</style>
"""


def _img(fig, fig_dir, stem):
    """Save PNG (provenance) then return an inline base64 <img> tag."""
    png = os.path.join(fig_dir, stem + ".png")
    try:
        fig.savefig(png, dpi=140, bbox_inches="tight")
    except Exception:
        pass
    b64 = fig_to_base64(fig, dpi=140)
    return f'<img alt="{stem}" src="data:image/png;base64,{b64}" />'


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}"


def _summary_tables(summ, cues, metrics):
    html = []
    for metric in metrics:
        sub = summ[summ["metric"] == metric]
        rows = [
            "<table class='t'><tr><th class='l'>Alignment</th><th>peak mean</th><th>± s.e.m.</th>"
            "<th>latency mean (s)</th><th>latency s.d. (s)</th><th>FWHM (s)</th><th>n</th></tr>"
        ]
        for cue_key in cues:
            r = sub[sub["cue_key"] == cue_key]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            rows.append(
                f"<tr><td class='l'>{config.CUE_LABELS[cue_key]}</td>"
                f"<td>{_fmt(r['peak_mean'])}</td><td>{_fmt(r['peak_sem'])}</td>"
                f"<td>{_fmt(r['latency_mean_s'], 2)}</td><td>{_fmt(r['latency_sd_s'], 2)}</td>"
                f"<td>{_fmt(r['fwhm_mean_s'], 2)}</td><td>{int(r['n'])}</td></tr>"
            )
        rows.append("</table>")
        html.append(f"<h3>{config.METRIC_LABEL[metric]}</h3>" + "".join(rows))
    return "".join(html)


def _vote_table(tally, cues, metrics):
    rows = ["<table class='t'><tr><th class='l'>Metric</th>"
            + "".join(f"<th>{config.CUE_LABELS[c]}</th>" for c in cues) + "</tr>"]
    for m in metrics:
        cells = "".join(f"<td>{int(tally.loc[m, c]) if (m in tally.index and c in tally.columns) else 0}</td>"
                        for c in cues)
        rows.append(f"<tr><td class='l'>{config.METRIC_LABEL[m]}</td>{cells}</tr>")
    rows.append("</table>")
    return "".join(rows)


def _cell_inventory(records, cues, patients):
    rows = ["<table class='t'><tr><th class='l'>Patient</th>"
            + "".join(f"<th>{config.CUE_LABELS[c]}</th>" for c in cues)
            + "<th class='l'>n trials</th><th class='l'>words</th></tr>"]
    for p in patients:
        cells, ntr, nw = [], "—", "—"
        for c in cues:
            rec = records.get((c, p))
            if rec is None:
                cells.append("<td>—</td>")
            else:
                m = rec["meta"]
                cells.append(f"<td>{m['n_bins']} bins</td>")
                ntr, nw = m.get("n_trials", "—"), m.get("n_unique_words", "—")
        rows.append(f"<tr><td class='l'>{display_id(p)} ({p})</td>{''.join(cells)}"
                    f"<td class='l'>{ntr}</td><td class='l'>{nw}</td></tr>")
    rows.append("</table>")
    return "".join(rows)


def _write_source_csvs(src_dir, records, peak_df, summ, tally, cues, metrics, patients, alpha):
    os.makedirs(src_dir, exist_ok=True)
    peak_df.to_csv(os.path.join(src_dir, "peak_table.csv"), index=False)
    summ.to_csv(os.path.join(src_dir, "peak_summary.csv"), index=False)
    tally.to_csv(os.path.join(src_dir, "argmax_vote_tally.csv"))
    # group time-courses + per-bin significance
    gt_rows, gs_rows = [], []
    for cue_key in cues:
        for metric in metrics:
            pp = A.present_patients(records, cue_key, metric, patients)
            if not pp:
                continue
            gt = A.group_timecourse(records, cue_key, metric, pp)
            gt.insert(0, "metric", metric); gt.insert(0, "cue_key", cue_key)
            gt_rows.append(gt)
            gs = S.group_perbin(records, cue_key, metric, pp, alpha=alpha)
            if len(gs):
                gs.insert(0, "metric", metric); gs.insert(0, "cue_key", cue_key)
                gs_rows.append(gs)
    if gt_rows:
        pd.concat(gt_rows, ignore_index=True).to_csv(
            os.path.join(src_dir, "group_timecourse.csv"), index=False)
    if gs_rows:
        pd.concat(gs_rows, ignore_index=True).to_csv(
            os.path.join(src_dir, "group_perbin_significance.csv"), index=False)


def build_report(patients=None, cues=None, metrics=None, alpha=0.05, pctile=None,
                 exclude=(), out_name="auditory_alignment_report.html"):
    """Load cached cells, build every figure + tables, write the HTML report. Returns path."""
    patients = list(patients or config.AUD_PATIENTS)
    cues = list(cues or config.CUES.keys())
    metrics = list(metrics or config.METRIC_KEYS)
    pctile = config.DEFAULTS["pctile"] if pctile is None else pctile

    out_root = results_dir(config.ANALYSIS)
    fig_dir = str(results_dir(config.ANALYSIS, "figures"))
    src_dir = str(results_dir(config.ANALYSIS, "source_data"))

    records = A.load_all(cues, patients, pctile=pctile)
    if not records:
        raise RuntimeError("No computed cells found — run the compute step first "
                           "(python -m tests.auditory_alignment.run).")
    # patients actually present anywhere
    present = [p for p in patients if any((c, p) in records for c in cues)]
    kept = [p for p in present if p not in exclude]

    peak_df = A.peak_table(records, cues, present, metrics)
    summ = A.peak_summary(peak_df, kept)
    vote = A.argmax_vote(peak_df, kept, metrics)
    tally = A.vote_tally(vote, cues, metrics)

    _write_source_csvs(src_dir, records, peak_df, summ, tally, cues, metrics, present, alpha)

    parts = ["<!DOCTYPE html><html><head><meta charset='utf-8'>", CSS,
             "</head><body>"]
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts.append(f"<h1>Auditory alignment — which cue triggers the semantic signal?</h1>")
    parts.append(
        f"<p class='muted'>Built {ts} · patients present: "
        f"{', '.join(display_id(p) for p in present)} · alignments: "
        f"{', '.join(config.CUE_LABELS[c] for c in cues)} · embedding: GloVe · "
        f"per-bin null pctile {pctile} · BH-FDR α={alpha}</p>")

    parts.append(
        "<div class='note'><b>Design.</b> Instead of time-warping, each trial's raw "
        "high-gamma is re-referenced to one behavioral cue at a time "
        "(<code>--warp none --align {cue}</code> in <code>semantic_regression.py</code>): "
        "the window is trimmed to whole 100&nbsp;ms bins so the cue lands exactly on a bin "
        "boundary (x=0) and to the shortest window across that patient's trials, then the "
        "full clean+kernel-PLS pipeline runs. The cue that yields the strongest and most "
        "temporally <i>locked</i> semantic readout de-smeared the informative bin, and is "
        "the likely trigger.</div>")

    parts.append(
        "<div class='warn'><b>Read the statistics honestly.</b> With n=6 patients a one-"
        "sided Wilcoxon signed-rank has an exact minimum p of 1/2<sup>6</sup>&nbsp;≈&nbsp;0.016 "
        "— so ** / *** are unreachable that way; a single * is the ceiling. The primary "
        "group per-bin test is therefore <b>Fisher's combination of the six per-patient "
        "permutation p-values</b> (each from the shuffled null the fit already computed), "
        "BH-FDR corrected across bins. <b>Cosine</b> has no stored per-bin null so it is "
        "shown descriptively (no significance); metrics with a real null are category "
        "accuracy, word top-1/3/5 and R². Auditory decoding is weak overall (few trials) — "
        "this compares <i>relative</i> alignment, not absolute performance. Each per-bin "
        "point integrates up to 1&nbsp;s of causal history, so a &lsquo;+0.25&nbsp;s peak&rsquo; "
        "means confident-by-then, not absent-earlier. "
        "<b>Epoch resolution:</b> each patient's permutation p can only reach "
        "&#8776;1/(epochs+1) (&#8776;0.048 at 20 epochs, &#8776;0.020 at 50), so a fast "
        "low-epoch pilot can show an <i>empty</i> significance raster even where the group "
        "mean clearly beats chance; it populates at the full 50-epoch run.</div>")

    parts.append("<h2>Cells computed</h2>")
    parts.append(_cell_inventory(records, cues, present))

    # ── Headline ──────────────────────────────────────────────────────────────
    parts.append("<h2>1 · Headline — peak height × temporal locking</h2>")
    parts.append("<p>Each marker is one alignment. The trigger cue sits <b>high</b> "
                 "(strong decoding) and <b>left</b> (low cross-patient latency jitter).</p>")
    parts.append(_img(F.fig_locking_scatter(peak_df, kept, cues, metrics),
                      fig_dir, "01_locking_scatter"))
    if config.ATYPICAL_PATIENT in present and config.ATYPICAL_PATIENT not in exclude:
        kept_noRB = [p for p in kept if p != config.ATYPICAL_PATIENT]
        if kept_noRB:
            parts.append(f"<h3>Same, excluding {config.ATYPICAL_PATIENT} "
                         f"(atypical go-cue geometry)</h3>")
            parts.append(_img(F.fig_locking_scatter(peak_df, kept_noRB, cues, metrics),
                              fig_dir, "01b_locking_scatter_noRB"))

    # ── Vote + heatmap ────────────────────────────────────────────────────────
    parts.append("<h2>2 · Which alignment wins — within-patient vote &amp; overview</h2>")
    parts.append("<div class='grid2'>")
    parts.append("<div>" + _img(F.fig_vote(tally, cues, metrics), fig_dir, "02_vote")
                 + _vote_table(tally, cues, metrics) + "</div>")
    parts.append("<div>" + _img(F.fig_heatmap(peak_df, kept, cues, metrics),
                                 fig_dir, "03_heatmap") + "</div>")
    parts.append("</div>")

    # ── Per-metric detail ─────────────────────────────────────────────────────
    parts.append("<h2>3 · Per-metric detail</h2>")
    for mi, metric in enumerate(metrics):
        parts.append(f"<h3>{config.METRIC_LABEL[metric]}</h3>")
        parts.append(_img(F.fig_timecourse_grid(records, cues, metric, kept, alpha=alpha),
                          fig_dir, f"10_timecourse_{metric}"))
        parts.append("<div class='grid2'>")
        parts.append("<div>" + _img(F.fig_peak_box(peak_df, metric, kept, cues),
                                     fig_dir, f"11_peakbox_{metric}") + "</div>")
        parts.append("<div>" + _img(F.fig_latency_box(peak_df, metric, kept, cues),
                                     fig_dir, f"12_latency_{metric}") + "</div>")
        parts.append("</div>")

    # ── Summary tables ────────────────────────────────────────────────────────
    parts.append("<h2>4 · Summary tables</h2>")
    parts.append(_summary_tables(summ, cues, metrics))

    # ── Per-patient supplement ────────────────────────────────────────────────
    parts.append("<h2>5 · Per-patient detail (supplement)</h2>")
    supp_metrics = [m for m in ("cosine", "category_indep", "word_top1") if m in metrics]
    for metric in supp_metrics:
        parts.append(_img(F.fig_patient_detail(records, cues, metric, present),
                          fig_dir, f"20_patient_{metric}"))

    parts.append(
        "<div class='warn'><b>Caveats.</b> "
        f"<b>{config.ATYPICAL_PATIENT}</b>: go-cue precedes the auditory prompt "
        "(atypical cue geometry) — the headline is shown with and without it. "
        f"<b>{'/'.join(config.FEW_TRIAL_PATIENTS)}</b>: very few auditory trials → unstable / "
        "NaN word top-k (nan-safe aggregation; see the per-patient panels). Group curves are "
        "plotted only where all shown patients cover a bin.</div>")

    parts.append(f"<p class='muted'>Source data: <code>{src_dir}</code> · "
                 f"figures: <code>{fig_dir}</code></p>")
    parts.append("</body></html>")

    out_path = os.path.join(str(out_root), out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print(f"[report] wrote {out_path}", flush=True)
    return out_path
