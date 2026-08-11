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
from tests.auditory_alignment import compare_warped as CW  # noqa: E402
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


def _arm_label(key):
    """Display label for a table row. The four cue alignments come from config.CUE_LABELS;
    the warped run is a fifth arm that is not a cue, so it is not in that mapping."""
    if key == CW.WARPED_KEY:
        return CW.WARPED_LABEL
    return config.CUE_LABELS.get(key, key)


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
                f"<tr><td class='l'>{_arm_label(cue_key)}</td>"
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


def _warped_section(peak_df, warped_df, summ_all, kept, cues, metrics, epochs_pilot):
    """Section: the warped run (AUD_RUN) as a fifth arm against the four cue alignments.

    Reports peak height per metric for all five arms, a paired Wilcoxon of warped vs each
    cue on per-patient peaks, and the provenance + comparability caveats. The caveats are
    not optional garnish: two of the three are reasons a number in this table must NOT be
    read as a like-for-like difference."""
    prov = CW.warped_provenance()
    arms = list(cues) + [CW.WARPED_KEY]
    html = []

    html.append("<h2>5 · Comparison with the linearly warped run</h2>")
    html.append(
        "<p>The four arms above re-reference raw data to a cue and do <b>not</b> warp "
        "(<code>--warp none</code>). The shipped auditory decoder instead time-warps each "
        "trial so the spoken prompt occupies a common duration, then aligns to "
        "<code>aud_stim_onset</code>. That run is <code>utils.config.AUD_RUN</code> and is "
        "the fifth arm here.</p>")
    html.append(
        "<div class='note'><b>The contrast that isolates the warp is Stim on vs Warped.</b> "
        "Both align to <code>aud_stim_onset</code>; they differ only in whether the prompt "
        "was stretched to a common duration. The other three cue arms differ from the warped "
        "run in <i>two</i> ways at once (alignment and warping), so a difference there does "
        "not attribute to either one.</div>")
    html.append(
        "<p class='muted'>Warped run: <code>{run}</code> · warp={warp}/{scope} → "
        "target {tgt} s ({src}) · align={align} · ROI gate {atlas} · history "
        "{h} bins × {b} ms · {ep} epochs · n={n}</p>".format(
            run=prov.get("run_id", "?"), warp=prov.get("auditory_warp"),
            scope=prov.get("auditory_warp_scope"),
            tgt=_fmt(prov.get("auditory_warp_target_sec"), 4),
            src=prov.get("auditory_warp_target_source"), align=prov.get("align_cue"),
            atlas=prov.get("roi_atlas"), h=prov.get("n_bins_history"),
            b=prov.get("bin_size_ms"), ep=prov.get("n_epochs"),
            n=len(prov.get("patients") or [])))

    # The history caveat is DERIVED, not typed: the warped arm was repointed to a 10-bin run
    # on 2026-08-11 while this pilot still runs at the repo default of 5. If the two ever
    # match again this sentence disappears on its own rather than going quietly stale.
    _h_warp = prov.get("n_bins_history")
    _h_pilot = config.DEFAULTS["n_bins_history"]
    _hist_caveat = ""
    if _h_warp and _h_warp != _h_pilot:
        _hist_caveat = (
            "<b>(4) The two sides no longer share a history window.</b> This pilot uses "
            f"{_h_pilot} bins ({_h_pilot * config.DEFAULTS['bin_size']}&nbsp;ms) and the "
            f"warped arm now uses {_h_warp} ({_h_warp * int(prov.get('bin_size_ms') or 0)}"
            "&nbsp;ms), because <code>figures_for_paper/semantic_regression</code> was "
            "repointed to the 10-bin auditory run on 2026-08-11. Doubling history is worth "
            "roughly +1 to +7% on auditory metrics on its own "
            "(<code>docs/experiments/001</code>), so part of any warped-vs-cue gap below is "
            "the window, not the alignment. Re-run this pilot with "
            f"<code>--history-bins {_h_warp}</code> to remove this term."
        )
    html.append(
        "<div class='warn'><b>%s ways this comparison is not like-for-like.</b> "
        % ("Four" if _hist_caveat else "Three") +
        "<b>(1) Peak height is comparable; latency is not</b>, except for Stim on. The four "
        "cue arms have a cue-relative axis with x=0 at the cue; the warped arm's axis is "
        "<i>warped</i> time, with every prompt stretched or compressed to "
        f"{_fmt(prov.get('auditory_warp_target_sec'), 3)}&nbsp;s. A latency in the warped "
        "column is a position in warped time, so the latency and FWHM columns are shown for "
        "completeness and must not be differenced across the boundary. "
        "<b>(2) Per-bin significance is not comparable.</b> This pilot ran at "
        f"{epochs_pilot} epochs, flooring each patient's permutation p at "
        f"&#8776;{1.0 / (epochs_pilot + 1):.4f}; the warped run ran at "
        f"{prov.get('n_epochs')} epochs, flooring at "
        f"&#8776;{1.0 / (int(prov.get('n_epochs') or 1) + 1):.4f}. Fewer significant bins on "
        "the pilot side can be resolution, not absence of signal. "
        "<b>(3) The pilot is <code>--warp none</code>.</b> It is not the warped run under a "
        "different alignment — it is a different treatment of the time axis, which is the "
        "whole point of putting them side by side. " + _hist_caveat + "</div>")

    # Peak height, all five arms
    html.append("<h3>Peak height by metric (per-patient argmax over the full window, "
                "then mean across patients)</h3>")
    # Only metrics the warped arm actually carries. R2 is in neither warped source
    # (source_data.csv holds the four retrieval metrics; per_time_scores.csv has r2_mean
    # but no matching per-bin null), so a comparison row for it cannot be built -- and an
    # R2 table showing only the cue arms would read as a comparison that was made and lost.
    have = set(warped_df["metric"].unique())
    skipped = [m for m in metrics if m not in have]
    if skipped:
        html.append("<p class='muted'>Not compared (absent from the warped run's source "
                    "data): " + ", ".join(config.METRIC_LABEL.get(m, m) for m in skipped)
                    + ".</p>")
    # Every cue x metric paired test is one family: 4 cues x 5 metrics = 20 tests. At
    # alpha=0.05 that expects ~1 false positive on its own, so an uncorrected single star
    # in this table is not evidence of anything. BH-FDR across the whole family, computed
    # in one pass BEFORE rendering so no cell can be shown without its q.
    both = pd.concat([peak_df, warped_df], ignore_index=True)
    cmp_metrics = [m for m in metrics if m in have]
    keys, pvals = [], []
    for metric in cmp_metrics:
        for arm in cues:
            p, n = S.paired_wilcoxon_peaks(both, metric, arm, CW.WARPED_KEY)
            keys.append((metric, arm, n))
            pvals.append(p)
    rej, qvals = S.benjamini_hochberg(np.asarray(pvals, dtype=float), alpha=0.05)
    stat = {(m, a): (p, q, bool(r), n)
            for (m, a, n), p, q, r in zip(keys, pvals, qvals, rej)}
    n_raw = int(np.sum(np.asarray(pvals, dtype=float) < 0.05))
    n_fdr = int(np.sum(rej))
    html.append(
        f"<div class='note'><b>Multiplicity.</b> {len(pvals)} paired tests "
        f"({len(cmp_metrics)} metrics × {len(cues)} alignments), each two-sided Wilcoxon "
        "signed-rank on per-patient peaks vs the warped run. "
        f"<b>{n_raw}</b> reach p&lt;0.05 uncorrected; <b>{n_fdr}</b> survive BH-FDR at "
        "q&lt;0.05. At 20 tests, ~1 uncorrected hit is the null expectation — read the q "
        "column, not the p column.</div>")

    for metric in cmp_metrics:
        sub = summ_all[summ_all["metric"] == metric]
        if sub.empty:
            continue
        rows = ["<table class='t'><tr><th class='l'>Arm</th><th>peak mean</th>"
                "<th>± s.e.m.</th><th>latency (s)</th><th>n</th>"
                "<th>vs warped: p</th><th>q (BH)</th></tr>"]
        for arm in arms:
            r = sub[sub["cue_key"] == arm]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            if arm == CW.WARPED_KEY:
                pcell = "<td class='l'>—</td><td class='l'>—</td>"
            else:
                p, q, ok, n = stat.get((metric, arm), (np.nan, np.nan, False, 0))
                pcell = (f"<td class='l'>{_fmt(p, 4)} (n={n})</td>"
                         f"<td class='l'>{_fmt(q, 4)} {S.stars(q) if ok else 'n.s.'}</td>"
                         if np.isfinite(p) else "<td class='l'>—</td><td class='l'>—</td>")
            lat = ("<td class='muted'>warped time</td>" if arm == CW.WARPED_KEY
                   else f"<td>{_fmt(r['latency_mean_s'], 2)}</td>")
            rows.append(
                f"<tr><td class='l'>{_arm_label(arm)}</td><td>{_fmt(r['peak_mean'])}</td>"
                f"<td>{_fmt(r['peak_sem'])}</td>{lat}<td>{int(r['n'])}</td>{pcell}</tr>")
        rows.append("</table>")
        html.append(f"<h4>{config.METRIC_LABEL[metric]}</h4>" + "".join(rows))

    html.append("<p class='muted'>Stars follow the q column, not p.</p>")

    # Provenance anchor
    anch = CW.warped_group_anchor(kept)
    if len(anch):
        html.append(
            "<h3>Provenance check — the warped run's own published summary</h3>"
            "<p class='muted'>Reproduced from the same source data behind the shipped "
            "figure, using <i>that</i> file's estimator: the cohort mean at a single common "
            "bin t*, where t* is the argmax of the across-participant mean curve over t≥0 "
            "restricted to bins every participant covers. These values match "
            "<code>figures_for_paper/semantic_regression/source_data/peak_rise_stats.csv</code> "
            "and exist to prove this section reads the run the paper reports. <b>They are a "
            "different estimator from the table above and are lower by construction</b> — a "
            "per-patient argmax capitalises on each patient's own noise, a common-bin mean "
            "does not. Do not compare a number across the two tables.</p>")
        rows = ["<table class='t'><tr><th class='l'>Metric</th><th>t* (s)</th>"
                "<th>cohort mean at t*</th><th>± s.e.m.</th><th>empirical chance</th>"
                "<th>n</th></tr>"]
        for _, r in anch.iterrows():
            rows.append(
                f"<tr><td class='l'>{config.METRIC_LABEL.get(r['metric'], r['metric'])}</td>"
                f"<td>{_fmt(r['t_star_s'], 1)}</td><td>{_fmt(r['peak_acc_mean'], 4)}</td>"
                f"<td>{_fmt(r['peak_acc_sem'], 4)}</td><td>{_fmt(r['emp_chance'], 4)}</td>"
                f"<td>{int(r['n_patients'])}</td></tr>")
        rows.append("</table>")
        html.append("".join(rows))
    return "".join(html)


def _coverage_table(records, cues, patients):
    """Per cue: the window every patient covers, and which patient limits each end.

    `aggregate.group_timecourse` keeps only bins ALL shown patients cover, so one
    short-windowed participant silently truncates a whole arm's group curve. That is not
    visible in the curve itself — it just starts later — and would otherwise read as a
    property of the cue rather than of one patient's trial timing. Returns (html, df)."""
    rows = []
    for cue_key in cues:
        lo_by_p, hi_by_p = {}, {}
        for p in patients:
            rec = records.get((cue_key, p))
            if rec is None:
                continue
            ks = [int(k) for k in rec["meta"].get("k", [])]
            if not ks:
                continue
            lo_by_p[p], hi_by_p[p] = min(ks), max(ks)
        if not lo_by_p:
            continue
        bin_s = next(iter(records[(cue_key, p)]["meta"]["bin_size_ms"]
                          for p in lo_by_p)) / 1000.0
        k_lo, k_hi = max(lo_by_p.values()), min(hi_by_p.values())
        lim_lo = max(lo_by_p, key=lambda q: lo_by_p[q])
        lim_hi = min(hi_by_p, key=lambda q: hi_by_p[q])
        rows.append(dict(
            cue_key=cue_key, n_patients=len(lo_by_p),
            group_start_s=k_lo * bin_s, group_end_s=k_hi * bin_s,
            group_span_s=(k_hi - k_lo) * bin_s,
            union_start_s=min(lo_by_p.values()) * bin_s,
            union_end_s=max(hi_by_p.values()) * bin_s,
            limiting_patient_start=lim_lo, limiting_patient_end=lim_hi,
        ))
    df = pd.DataFrame(rows)
    if df.empty:
        return "", df
    html = ["<table class='t'><tr><th class='l'>Alignment</th>"
            "<th>group window (s)</th><th>span (s)</th><th>widest single patient (s)</th>"
            "<th class='l'>limits the start</th><th class='l'>limits the end</th></tr>"]
    for _, r in df.iterrows():
        html.append(
            f"<tr><td class='l'>{_arm_label(r['cue_key'])}</td>"
            f"<td>{_fmt(r['group_start_s'], 1)} … {_fmt(r['group_end_s'], 1)}</td>"
            f"<td>{_fmt(r['group_span_s'], 1)}</td>"
            f"<td>{_fmt(r['union_start_s'], 1)} … {_fmt(r['union_end_s'], 1)}</td>"
            f"<td class='l'>{display_id(r['limiting_patient_start'])} "
            f"({r['limiting_patient_start']})</td>"
            f"<td class='l'>{display_id(r['limiting_patient_end'])} "
            f"({r['limiting_patient_end']})</td></tr>")
    html.append("</table>")
    return "".join(html), df


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

    # The warped run (AUD_RUN) as a fifth arm. Kept OUT of peak_df/vote/tally on purpose:
    # the vote and the locking scatter ask "which CUE wins", and the warped run is not a
    # cue -- folding it in would silently change what those two figures mean. It joins only
    # in the dedicated comparison section, on peak height.
    try:
        warped_df = CW.warped_peak_table(present)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  [warn] warped comparison unavailable: {exc}", flush=True)
        warped_df = pd.DataFrame(columns=peak_df.columns)
    summ_all = (A.peak_summary(pd.concat([peak_df, warped_df], ignore_index=True), kept)
                if len(warped_df) else summ)

    _write_source_csvs(src_dir, records, peak_df, summ, tally, cues, metrics, present, alpha)
    if len(warped_df):
        CW.write_source_csv(src_dir, present)
        # The five-arm summary is what section 5 renders; ship it as source data so the
        # table and its numbers stay one unit.
        summ_all.to_csv(os.path.join(src_dir, "peak_summary_with_warped.csv"), index=False)
        _rows = []
        _both = pd.concat([peak_df, warped_df], ignore_index=True)
        _have = set(warped_df["metric"].unique())
        for _m in [m for m in metrics if m in _have]:
            for _a in cues:
                _p, _n = S.paired_wilcoxon_peaks(_both, _m, _a, CW.WARPED_KEY)
                _rows.append(dict(metric=_m, cue_key=_a, vs="warped", p=_p, n=_n))
        if _rows:
            _st = pd.DataFrame(_rows)
            _rej, _q = S.benjamini_hochberg(_st["p"].to_numpy(dtype=float), alpha=alpha)
            _st["q_bh"], _st["reject_bh"] = _q, _rej
            _st.to_csv(os.path.join(src_dir, "warped_paired_tests.csv"), index=False)

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

    # Derived from the cohort actually loaded, not typed. This paragraph used to hardcode
    # the n=6 floor (1/2^6 = 0.016) and the claim that ** / *** are unreachable. At n=10
    # the floor is 0.00098 and *** IS reachable, so the hardcoded version did not merely
    # go stale -- it understated what the test could show, in a report whose whole purpose
    # is weighing evidence. config.WILCOXON_FLOOR existed for this and was read nowhere.
    _n_pat = len(present)
    _floor = 1.0 / (2 ** _n_pat) if _n_pat else float("nan")
    # Ladder must cover the case where the floor exceeds 0.05 (n<=4): there NOTHING is
    # reachable, not "a single * is the ceiling".
    _reach = ("*** is reachable" if _floor < 0.001 else
              "** is reachable but *** is not" if _floor < 0.01 else
              "a single * is the ceiling" if _floor < 0.05 else
              "not even a single * is reachable")
    parts.append(
        f"<div class='warn'><b>Read the statistics honestly.</b> With n={_n_pat} patients a "
        f"one-sided Wilcoxon signed-rank has an exact minimum p of 1/2<sup>{_n_pat}</sup>"
        f"&nbsp;&#8776;&nbsp;{_floor:.5g} &mdash; so {_reach} by that route. The primary "
        f"group per-bin test is nonetheless <b>Fisher's combination of the {_n_pat} "
        "per-patient permutation p-values</b> (each from the shuffled null the fit already "
        "computed), BH-FDR corrected across bins. <b>Cosine</b> has no stored per-bin null so it is "
        "shown descriptively (no significance); metrics with a real null are category "
        "accuracy, word top-1/3/5 and R². Auditory decoding is weak overall (few trials) — "
        "this compares <i>relative</i> alignment, not absolute performance. Each per-bin "
        "point integrates up to 1&nbsp;s of causal history, so a &lsquo;+0.25&nbsp;s peak&rsquo; "
        "means confident-by-then, not absent-earlier. "
        "<b>Epoch resolution:</b> each patient's permutation p can only reach "
        "&#8776;1/(epochs+1) (&#8776;0.048 at 20 epochs, &#8776;0.020 at 50), so a fast "
        "low-epoch pilot can show an <i>empty</i> significance raster even where the group "
        "mean beats chance. <b>That is not the explanation here.</b> This report is the "
        "full 50-epoch run and the raster is still empty — an earlier version of this "
        "paragraph predicted it would populate, and that prediction is falsified. Fisher "
        "combination of ten floored p-values would reach p&#8776;3e-9, so the floor is not "
        "binding; the group per-bin p simply does not get small enough. See the "
        "significant-bin counts below.</div>")

    parts.append("<h2>Cells computed</h2>")
    parts.append(_cell_inventory(records, cues, present))

    cov_html, cov_df = _coverage_table(records, cues, kept)
    if cov_html:
        parts.append("<h3>Window each alignment's group curve actually covers</h3>")
        parts.append(
            "<p class='muted'>Group curves are averaged only over bins <b>every</b> shown "
            "patient covers, so the shortest window sets the arm's span. Where the group "
            "window is much narrower than the widest single patient, one participant's "
            "trial timing — not the cue — is what shortened it. Peak search in the tables "
            "below is <i>per patient</i> over that patient's own full window, so it is "
            "unaffected; the group time-course figures are.</p>")
        parts.append(cov_html)
        cov_df.to_csv(os.path.join(src_dir, "group_window_coverage.csv"), index=False)

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

    # ── Group per-bin significance, stated as a count ─────────────────────────
    # An all-zero significance raster is easy to skim past as "the figure looks normal".
    # State the count and the closest miss explicitly, so a null result reads as a result.
    gs_path = os.path.join(src_dir, "group_perbin_significance.csv")
    if os.path.isfile(gs_path):
        gs = pd.read_csv(gs_path)
        testable = [m for m in metrics if config.METRIC_HAS_NULL.get(m, False)]
        gs_t = gs[gs["metric"].isin(testable)]
        n_sig = int(gs_t["sig_fdr"].sum()) if "sig_fdr" in gs_t else 0
        n_bins = int(len(gs_t))
        parts.append("<h3>Group per-bin significance</h3>")
        tab = (gs_t.groupby(["cue_key", "metric"])["sig_fdr"].sum().unstack(fill_value=0)
               if n_bins else None)
        if n_sig == 0 and n_bins:
            best = gs_t.loc[gs_t["p_fisher"].idxmin()] if gs_t["p_fisher"].notna().any() else None
            extra = ""
            if best is not None:
                extra = (f" The smallest Fisher-combined p over all {n_bins} tested bins is "
                         f"<b>{_fmt(best['p_fisher'], 4)}</b> ({_arm_label(best['cue_key'])}, "
                         f"{config.METRIC_LABEL.get(best['metric'], best['metric'])}, "
                         f"t={_fmt(best['t_s'], 1)} s; group {_fmt(best['group_obs'], 4)} vs "
                         f"null {_fmt(best['group_null'], 4)}), giving q="
                         f"{_fmt(best['q_fisher'], 3)}.")
            parts.append(
                "<div class='warn'><b>No bin is significant in any alignment.</b> "
                f"0 of {n_bins} tested bins survive BH-FDR at α={alpha}, across all "
                f"{len(cues)} alignments and all {len(testable)} metrics that carry a null."
                + extra +
                " The group mean exceeds its null in the large majority of bins, so this is "
                "a power statement, not a sign reversal — but it means <b>no alignment in "
                "this pilot demonstrates significant per-bin decoding</b>, and the "
                "alignment comparison below is between arms none of which clear that bar. "
                "Report it that way.</div>")
        elif n_bins:
            parts.append(f"<p>{n_sig} of {n_bins} tested bins survive BH-FDR at "
                         f"α={alpha}.</p>")
        if tab is not None:
            hdr = "".join(f"<th>{config.METRIC_LABEL.get(c, c)}</th>" for c in tab.columns)
            trs = "".join(
                f"<tr><td class='l'>{_arm_label(i)}</td>"
                + "".join(f"<td>{int(v)}</td>" for v in row) + "</tr>"
                for i, row in tab.iterrows())
            parts.append("<table class='t'><tr><th class='l'>Alignment</th>"
                         + hdr + "</tr>" + trs + "</table>")

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

    # ── Warped-run comparison ─────────────────────────────────────────────────
    if len(warped_df):
        _eps = [r["meta"].get("epochs") for r in records.values()
                if r["meta"].get("epochs") is not None]
        # Read from the cells rather than from the CLI arg: on --report-only the arg is a
        # default, not what actually produced the cells on disk.
        _ep = int(max(_eps)) if _eps else 0
        if len(set(_eps)) > 1:
            parts.append(
                "<div class='warn'><b>Mixed epochs across cells</b> "
                f"({sorted(set(_eps))}) — the pilot's permutation-p floor is not one number "
                "here. The comparison section quotes the largest.</div>")
        parts.append(_warped_section(peak_df, warped_df, summ_all, kept, cues, metrics, _ep))

    # ── Per-patient supplement ────────────────────────────────────────────────
    parts.append("<h2>6 · Per-patient detail (supplement)</h2>")
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
