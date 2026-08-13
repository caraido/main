# -*- coding: utf-8 -*-
"""
figures_for_paper/cross_task/compute_cross_task_data.py
=======================================================
Heavy step (run once, Speech conda env, from project root
``d:/.../Speech``).  Reads the existing cross-task analysis outputs under
``main/results/cross_task_cotrain/`` plus the new prediction-MDS run,
maps internal initials -> NUE display IDs, and writes tidy per-panel
source-data CSVs (+ ``group_inference.csv``) into ``./source_data/``.  The
CSV-only ``cross_task_panels.py`` renders from these; no project pkls needed
downstream.

Sources — all at the `tpm`/h10 pair with `balance=downsample` since 2026-08-13. Every one
is a named pin in utils.config, never a literal here:
  * co-training conditions     : <COTRAIN_RUN>/cotrain_conditions_summary.csv
  * ROI region importance      : <ROI_DIR>/region_importance_<atlas>_all.csv (permutation Δacc
                                 + Jacobian + ROI-only sufficiency, region-organized)
  * chance band                : figures_for_paper/semantic_regression/panels_cache_*.npz
                                 (the shuffled category-independent nulls)

**Rebuilt 2026-08-13 around the ROI story.** The figure is now four panels — co-training
generalization, Jacobian ROI ranking, ROI-only decoder accuracy, region knockout — and the
MDS/PCA/RSA panels are retired. `mds()` and `rsa()` are DELIBERATELY still defined but are
no longer called by `main()`: their shipped CSVs are the input to the pending co-trained
latent-space work, and regenerating them at the new cohort would destroy the N=9 copies for
no gain. Do not "tidy" them into the call chain or delete them without reading
docs/experiments/018.

**The cohort is 7, not 9, and it is DERIVED** — participants with at least one significant
category-independent time bin in BOTH tasks, from
`analysis.cross_task.cross_task_region_importance_report.significant_participants()`. Never
type the list here; it is read from the semantic_regression figure's shipped source data and
will move when that figure does.

The figure ran on `tp`/h5 + `balance=none` until 2026-08-12. Two things changed with it and
both belong in the caption: the channel set is `tpm` (18 regions, adding insula, cingulate,
entorhinal, parahippocampal, precuneus), so "temporal-parietal" no longer describes it; and
the pooled training set is now class-balanced by downsampling rather than not resampled.

Reproduce:
    python figures_for_paper/cross_task/compute_cross_task_data.py
    python figures_for_paper/cross_task/cross_task_panels.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS_ROOT = os.path.dirname(HERE)            # figures_for_paper/
MAIN_DIR = os.path.dirname(FIGS_ROOT)        # main/
sys.path.insert(0, FIGS_ROOT)
from paper_common import display_id          # noqa: E402  (also puts main/ on sys.path)
from utils.config import (CROSS_TASK_FIGURE_COTRAIN_RUN,                 # noqa: E402
                          CROSS_TASK_FIGURE_ROI_DIR,
                          CROSS_TASK_FIGURE_MDS_RUN,
                          OLD_STIMULUS_SET_PATIENTS,
                          AUD_RUN, PIC_RUN, p_stars)
from utils import config as _cfg   # noqa: E402
from utils.paths import latest_run_dir                                  # noqa: E402
from utils.roi_palette import color_of as _color_of                     # noqa: E402
# The cohort rule and the three derived-column helpers are DEFINED in the report module and
# imported, not restated. A second copy of "which participants" or "how a per-electrode value
# is formed" is exactly the drift this repo has been bitten by before: the figure and the
# report would silently stop describing the same thing.
from analysis.cross_task.cross_task_region_importance_report import (  # noqa: E402
    significant_participants, add_per_channel, add_standardized, region_colors)

SRC = os.path.join(HERE, "source_data")
os.makedirs(SRC, exist_ok=True)

RESULTS = os.path.join(MAIN_DIR, "results", "cross_task_cotrain")
# Renamed from NONE_RUN 2026-08-13: the figure moved to balance=downsample, and a name
# asserting `none` while pointing at the downsampled run is how a caption ends up
# describing a resampling scheme the numbers did not use.
COTRAIN_RUN = os.path.join(RESULTS, CROSS_TASK_FIGURE_COTRAIN_RUN)
# Region-importance output is keyed on the resampling setting (2026-07-23) and, since the
# scope ladder, on scope/history too — hence a pinned sub-path rather than a bare
# balance_<x>. The sufficiency columns present in this arm are not read here.
ROI_DIR = os.path.join(RESULTS, *CROSS_TASK_FIGURE_ROI_DIR.split("/"))

#: The reported cohort: participants with >= 1 significant category-independent time bin in
#: BOTH tasks (Alec, 2026-08-13). Seven of the nine that have both tasks; AZ and DR drop.
#:
#: Applied to EVERY panel, not just the ROI ones, so the figure carries a single N. The cost
#: is real and is stated in the caption: a two-sided paired Wilcoxon at n=7 cannot return a
#: p below 2/2^7 = 0.0156, so every picture contrast lands on that floor. Effect sizes barely
#: move (picture retention 0.818 -> 0.810), i.e. the two dropped participants cost resolution
#: rather than effect.
#:
#: Was `list(_cfg.SHARED_PATIENTS)` (9). The significance is read from the semantic_regression
#: figure's shipped source_data.csv, whose picture arm is `tp`/h5 and auditory arm `tpfm`/h10
#: -- a DIFFERENT configuration from this figure's runs. That caveat is in the caption; it is
#: the price of not loading the 92 MB per-patient pkls to recompute per-bin significance here.
PATIENTS = significant_participants()
if not PATIENTS:
    raise SystemExit(
        "significant_participants() returned nothing -- "
        "figures_for_paper/semantic_regression/source_data/source_data.csv is missing or has "
        "no significant rows. The cohort is derived from it and must not be typed here.")
#: Everyone with both tasks, kept only to report what the cohort filter dropped.
ENROLLED_BOTH_TASKS = list(_cfg.SHARED_PATIENTS)
METRIC_MAIN = "cat_indep_bal_acc"
METRICS = ["cat_indep_bal_acc", "word_bal_acc", "cosine_mean"]
#: Which atlas arm feeds the ROI panel. NMM is primary; the DK arm is computed and
#: archived alongside it, and which of them the FIGURE shows is an editorial decision
#: deferred until both sets of numbers exist.
ROI_ATLAS = _cfg.ROI_ATLAS_DEFAULT

# NOTE (2026-08-13): `_pick_representative` and `SELECTION_RULE` were deleted with the
# single-participant ROI bar panel they served. The three ROI panels are now
# cross-participant aggregates, so no exemplar is chosen and there is nothing left to
# justify. `panel_c_roi_coverage.csv` keeps the atlas-coverage columns and loses
# `is_representative` / `selection_rule`.

# Chance for cat_indep_bal_acc is 1 / (categories that participant actually has),
# and the cohort does NOT share one taxonomy.  Five participants ran the current
# auditory stimulus set (animal, body part, food/fruit, nature, object/tool,
# vehicle -> 6 categories, chance 0.167).  CP and RB ran an older set that adds
# abstract and action and drops vehicle: RB ends up with 5 (chance 0.200) and CP
# with 7 (chance 0.143).  A single hard-coded 1/6 was therefore wrong for RB
# already and would be wrong for CP too, so it is derived per participant below
# rather than typed.  See per_patient_chance().
_CHANCE_FALLBACK_N = 6

# Distinct category palette. The six categories of the current stimulus set get
# maximally separated hues; abstract & action occur only for the older-set pair
# (CP, RB) so they take leftover colours (not prioritised for separation).
CATEGORY_COLORS = {
    "animal": "#2ca02c",       # green
    "body part": "#e41a1c",    # red
    "food/fruit": "#377eb8",   # blue
    "nature": "#ff7f00",       # orange
    "object/tool": "#984ea3",  # purple
    "vehicle": "#17becf",      # teal
    "abstract": "#999999",     # grey  (RB only)
    "action": "#f781bf",       # pink  (RB only)
}
_SPARE_COLORS = ["#a65628", "#bcbd22", "#000000"]

# report's train-source ordering + condition map (cross_task_cotrain_report.py)
SRC_ORDER = ["within", "cross", "pooled"]
#: Which pairs get tested. Indices into SRC_ORDER = [within, cross, pooled], i.e.
#: **within-vs-cross** and **cross-vs-pooled** (Alec, 2026-08-13, revised the same day).
#:
#: Both contrasts are against **cross**, the transfer baseline: the question the figure puts
#: is whether a decoder beats naive train-on-one/test-on-the-other, and both the within-task
#: and the co-trained decoder are asked it.
#:
#: **within-vs-pooled is deliberately NOT tested**, which has a consequence that has to stay
#: visible rather than being quietly absorbed: the retention shortfall (co-trained decoder
#: vs the within-task ceiling, 81 % picture / 92 % auditory) now has **no significance test
#: behind it** and must be reported as a descriptive ratio. It was tested until this change
#: and read p = 0.0156 / 0.047. Do not re-add the contrast to recover a p-value for a
#: sentence; change the sentence.
COMPARISONS = [(0, 1), (1, 2)]
COND = {
    ("pic", "within"): "within_pic", ("pic", "cross"): "cross_a2p",
    ("pic", "pooled"): "pooled_pic",
    ("aud", "within"): "within_aud", ("aud", "cross"): "cross_p2a",
    ("aud", "pooled"): "pooled_aud",
}
TARGETS = ["pic", "aud"]


def _stars(p: float) -> str:
    """Star ladder — thresholds come from utils.config.p_stars (one ladder, repo-wide)."""
    return p_stars(p)


def _bh_fdr(p):
    """Benjamini-Hochberg q-values for a 1-D array of p, NaN-safe (NaN in, NaN out)."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx])]
    m = len(order)
    ranked = p[order] * m / np.arange(1, m + 1)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    return q


def did(pat: str) -> str:
    return display_id(pat)


# ── generalization (co-training: within / cross / pooled) ──────────────────

def generalization():
    summ = pd.read_csv(os.path.join(COTRAIN_RUN, "cotrain_conditions_summary.csv"))
    rows = []
    for pat in PATIENTS:
        for target in TARGETS:
            for source in SRC_ORDER:
                cond = COND[(target, source)]
                r = summ[(summ.patient == pat) & (summ.condition == cond)]
                if r.empty:
                    continue
                r = r.iloc[0]
                for m in METRICS:
                    rows.append(dict(display_id=did(pat), patient=pat,
                                     target=target, source=source, metric=m,
                                     value=float(r[f"{m}_mean"])))
    per_pat = pd.DataFrame(rows)
    per_pat.to_csv(os.path.join(SRC, "panel_b_generalization.csv"), index=False)

    grp, stats = [], []
    for target in TARGETS:
        for m in METRICS:
            data = {s: [] for s in SRC_ORDER}
            for pat in PATIENTS:
                for s in SRC_ORDER:
                    v = per_pat[(per_pat.patient == pat) & (per_pat.target == target)
                                & (per_pat.source == s) & (per_pat.metric == m)]
                    data[s].append(float(v.value.iloc[0]) if not v.empty else np.nan)
            for s in SRC_ORDER:
                a = np.array(data[s], float)
                a = a[np.isfinite(a)]
                grp.append(dict(target=target, metric=m, source=s,
                                mean=a.mean(), sem=a.std(ddof=1) / np.sqrt(len(a)),
                                n=len(a)))
            for i, j in COMPARISONS:
                a = np.array(data[SRC_ORDER[i]], float)
                b = np.array(data[SRC_ORDER[j]], float)
                mask = np.isfinite(a) & np.isfinite(b)
                a, b = a[mask], b[mask]
                pw = pt = np.nan
                if len(a) >= 2:
                    try:
                        _, pt = ttest_rel(a, b)
                    except Exception:
                        pass
                    try:
                        _, pw = wilcoxon(a, b, zero_method="zsplit")
                    except ValueError:
                        pass
                stats.append(dict(target=target, metric=m,
                                  comparison=f"{SRC_ORDER[i]}-{SRC_ORDER[j]}",
                                  p_wilcoxon=pw, p_ttest=pt, n=len(a),
                                  stars=_stars(pw)))
    stats = pd.DataFrame(stats)
    # BH-corrected companion, SHIPPED BUT NOT PLOTTED. The figure reports the uncorrected
    # Wilcoxon, as it always has, and the caption says so; silently swapping in a corrected
    # p would be changing a statistical method inside a regeneration. It is computed here so
    # the corrected values exist to be quoted, and so the decision to keep the uncorrected
    # ones is made against a number rather than in the dark. Family = all tests in this CSV.
    stats["q_bh"] = _bh_fdr(stats["p_wilcoxon"].to_numpy(dtype=float))
    stats["stars_bh"] = [_stars(q) for q in stats["q_bh"]]
    pd.DataFrame(grp).to_csv(
        os.path.join(SRC, "panel_b_generalization_group.csv"), index=False)
    stats.to_csv(os.path.join(SRC, "panel_b_generalization_stats.csv"), index=False)
    return per_pat, pd.DataFrame(grp), stats


def retention(per_pat: pd.DataFrame):
    rows = []
    for pat in PATIENTS:
        for target in TARGETS:
            def val(src):
                v = per_pat[(per_pat.patient == pat) & (per_pat.target == target)
                            & (per_pat.source == src)
                            & (per_pat.metric == METRIC_MAIN)]
                return float(v.value.iloc[0]) if not v.empty else np.nan
            w, p = val("within"), val("pooled")
            rows.append(dict(display_id=did(pat), patient=pat, target=target,
                             within=w, pooled=p,
                             retention=(p / w if w else np.nan)))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(SRC, "table_r1_retention.csv"), index=False)
    return tab


# ── semantic-organization MDS (separate per-task decoders) ─────────────────

def _latest_mds_run():
    """The PINNED ``*_prediction_mds_*`` run (``utils.config.CROSS_TASK_FIGURE_MDS_RUN``).

    Was "newest matching glob" until 2026-07-30, which made panel a the only figure input
    in the repo without a pin: re-running the MDS silently repointed the panel, and the run
    the shipped figure depended on read ``unreferenced`` in docs/results_index.md — which
    AGENTS.md authorises pruning. Falls back to newest-glob ONLY if the pin is missing from
    disk, and says so loudly rather than substituting silently.

    NB the fallback is now more dangerous than it was, not less: the newest matching glob
    is whichever configuration ran last, and there are `tp`/h5 and `tpm`/h10 MDS runs side
    by side. It stays only because failing loudly beats failing silently.
    """
    from pathlib import Path
    pinned = os.path.join(RESULTS, CROSS_TASK_FIGURE_MDS_RUN)
    if os.path.isdir(pinned):
        return pinned
    print(f"  [mds] WARNING pinned CROSS_TASK_FIGURE_MDS_RUN {CROSS_TASK_FIGURE_MDS_RUN} not "
          f"found on disk — falling back to the newest *_prediction_mds_* run, which may be "
          f"a DIFFERENT scope/history. Repin utils/config.py:CROSS_TASK_FIGURE_MDS_RUN.")
    return str(latest_run_dir(Path(RESULTS), "*_prediction_mds_*", fallback_to_root=False))


def mds():
    """RETIRED FROM THE FIGURE 2026-08-13, DELIBERATELY STILL HERE.

    The semantic-organization MDS panel and its S1/S2 supplements were dropped when the
    figure was rebuilt around the ROI story. This function is **not called by main()** and
    its outputs (``panel_a_mds_points.csv``, ``panel_a_mds_alignment.csv``) are **not
    regenerated** — the shipped copies are at the previous N = 9 cohort and are the input to
    the pending co-trained latent-space work (docs/experiments/018). Running it now would
    overwrite them at N = 7 and buy nothing.

    Do not delete it, and do not wire it back into main() without deciding what happens to
    those CSVs.
    """
    run = _latest_mds_run()
    print(f"  [mds] using {os.path.basename(run)}")
    frames = []
    missing = []
    for pat in PATIENTS:
        f = os.path.join(run, f"prediction_mds_{pat}.csv")
        if not os.path.exists(f):
            # Loud, and re-raised below: a participant silently dropped here is a panel
            # whose N disagrees with every other panel in the figure.
            print(f"  [mds] WARNING missing {pat}")
            missing.append(pat)
            continue
        d = pd.read_csv(f)
        d.insert(0, "display_id", did(pat))
        frames.append(d)
    if missing:
        raise FileNotFoundError(
            f"MDS run {os.path.basename(run)} is missing {len(missing)} of {len(PATIENTS)} "
            f"participants: {missing}. Panel a would ship a different N from every other "
            f"panel. Re-run cross_task_prediction_mds.py for the current cohort, or pass "
            f"an MDS run that has them.")
    pts = pd.concat(frames, ignore_index=True)
    pts.to_csv(os.path.join(SRC, "panel_a_mds_points.csv"), index=False)

    align = pd.read_csv(os.path.join(run, "prediction_mds_alignment_summary.csv"))
    # Filter to the analysed cohort. The per-patient loop above reads PATIENTS one by one,
    # but this summary is a whole-run file and carried every participant the MDS run was
    # executed with -- including any since retired. Unfiltered it put a retired participant
    # into panel a AND into the representative candidate pool below, so the panel's N
    # disagreed with every other panel in the figure and the showcase participant was
    # chosen from a cohort that is not the reported one.
    align = align[align["patient"].isin(PATIENTS)].copy()
    align.insert(0, "display_id", align["patient"].map(did))
    # representative = a clean-taxonomy patient (max shared categories) with the
    # strongest significant cross-task alignment; ties broken by alignment.  This
    # avoids RB as the MDS showcase (its PN/AN runs use inconsistent category
    # label sets, only 4 shared), while RB still headlines the VIP panel.
    cand = align[align["cat_centroid_alignment_p"] < 0.1]
    cand = cand if not cand.empty else align
    rep = cand.sort_values(["n_shared_categories", "cat_centroid_alignment"],
                           ascending=[False, False])["patient"].iloc[0]
    align["is_representative"] = (align["patient"] == rep)
    align.to_csv(os.path.join(SRC, "panel_a_mds_alignment.csv"), index=False)

    # fixed shared category palette — distinct hues, common categories prioritised
    cats = sorted(pts["category"].astype(str).unique())
    spare = iter(_SPARE_COLORS)
    pal = pd.DataFrame([
        {"category": c, "color": CATEGORY_COLORS.get(c) or next(spare, "#777777")}
        for c in cats])
    pal.to_csv(os.path.join(SRC, "category_style.csv"), index=False)
    return align, rep, cats


# ── ROI / region importance (VIP + permutation Δacc + Jacobian) ────────────

def roi():
    """Per patient x region table feeding all three ROI panels, from
    region_importance_<atlas>_all.csv: permutation Δacc pic/aud + significance, Jacobian
    pic/aud, ROI-only sufficiency accuracy, neural-GloVe covariance, and the whole-brain
    ceiling / share.

    The derived columns the panels actually plot are added HERE, by the report's own
    helpers, so the figure and the HTML report cannot disagree about what "per electrode"
    or "enrichment" means:
      * ``perm_imp_{pic,aud}_pc``   = region Δacc / its channel count            (panel d)
      * ``jac_sens_{pic,aud}_std``  = per-electrode ‖∂ŷ/∂x‖ ÷ that participant's
                                      whole-brain per-electrode average FOR THE SAME TASK
                                      (panel b)
      * ``suff_pooled_{pic,aud}``   = ROI-only decoder held-out accuracy, raw    (panel c)

    ``suff_delta_*`` / ``suff_null_*`` / ``suff_p_*`` are carried even though they are NaN
    in every row of this arm. That is deliberate: the pass ran with ``--suff-null-draws 0``,
    so panel c has **no matched-N size control**, and shipping the empty columns makes that
    visible in the data rather than only in the caption.

    Plain-PLS VIP was removed from the pipeline 2026-07-23 (it attributed a linear
    surrogate the paper does not report, and as a region total it was an
    electrode-count proxy), so `vip`/`vip_std` are no longer requested."""
    csv = os.path.join(ROI_DIR, f"region_importance_{ROI_ATLAS}_all.csv")
    if not os.path.exists(csv):
        raise SystemExit(
            f"{csv} not found. The ROI analysis is now per-atlas: run\n"
            f"  python -m analysis.cross_task.cross_task_region_importance "
            f"--atlas {ROI_ATLAS} --analysis both --single-modality --roi-sufficiency\n"
            f"(region_importance_all.csv was the retired primary_roi output and is not "
            f"read any more.)")
    d = pd.read_csv(csv)
    d = d[(d.metric == METRIC_MAIN) & (d.patient.isin(PATIENTS))].copy()
    have = sorted(d.patient.unique())
    missing = [p for p in PATIENTS if p not in have]
    if missing:
        raise SystemExit(
            f"{csv} is missing {missing} -- the ROI panels would carry a different N from "
            f"panel a. Re-run cross_task_region_importance for the reported cohort.")

    # Cohort filter BEFORE the derived columns: add_standardized builds a per-participant
    # whole-brain reference, so deriving first would reference participants the figure does
    # not show.
    add_per_channel(d)                      # <col>_pc
    add_standardized(d)                     # <col>_std
    d.insert(0, "display_id", d["patient"].map(did))
    keep = [c for c in [
        "display_id", "patient", "region", "n_channels",
        "perm_imp_pic", "perm_imp_aud", "perm_imp_pic_per_ch",
        "perm_imp_aud_per_ch", "perm_imp_pic_pc", "perm_imp_aud_pc",
        "p_pic", "p_aud", "q_pic", "q_aud", "group",
        "jac_sens_pic", "jac_sens_aud", "jac_sens_pic_std", "jac_sens_aud_std",
        "suff_pooled_pic", "suff_pooled_aud",
        "suff_adj_pooled_pic", "suff_adj_pooled_aud",
        "suff_delta_pic", "suff_delta_aud", "suff_p_pic", "suff_p_aud",
        "n_cats_pic", "n_cats_aud",
        "cov_nc_pic", "cov_nc_aud",
        "wb_imp_pic", "wb_imp_aud", "wb_p_pic", "wb_p_aud",
        "frac_wb_pic", "frac_wb_aud",
    ] if c in d.columns]
    d[keep].to_csv(os.path.join(SRC, "panel_c_roi.csv"), index=False)

    cov = pd.DataFrame([
        dict(display_id=did(p), patient=p, has_roi=(p in have), roi_atlas=ROI_ATLAS,
             note="" if p in have
             else "No ROI atlas available for this participant")
        for p in PATIENTS])
    cov.to_csv(os.path.join(SRC, "panel_c_roi_coverage.csv"), index=False)
    return d, have


def roi_aggregate(d):
    """The cross-participant table the ROI markers are DRAWN from — one row per region.

    Aggregator is the **mean** across participants (Alec, 2026-08-13; the HTML report
    defaults to the median and offers both). Unweighted: a participant with 3 contacts in a
    region counts as much as one with 20. ``n_participants`` is what marker size encodes and
    is the only reliability cue on the panel now that the ``(n)`` label suffix is gone —
    four of the 17 regions come from one or two participants.

    Shipped rather than recomputed in the plotting script because rule 2 of
    figures_for_paper/README.md is that every plotted table lives in source_data/.
    """
    d = d.copy()
    d["jac_enrich"] = d[["jac_sens_pic_std", "jac_sens_aud_std"]].mean(axis=1)
    agg = (d.groupby("region")
            .agg(ko_pic_pc=("perm_imp_pic_pc", "mean"),
                 ko_aud_pc=("perm_imp_aud_pc", "mean"),
                 jac_enrich=("jac_enrich", "mean"),
                 suff_pooled_pic=("suff_pooled_pic", "mean"),
                 suff_pooled_aud=("suff_pooled_aud", "mean"),
                 suff_adj_pooled_pic=("suff_adj_pooled_pic", "mean"),
                 suff_adj_pooled_aud=("suff_adj_pooled_aud", "mean"),
                 n_participants=("patient", "nunique"),
                 mean_channels=("n_channels", "mean"))
            .sort_values("jac_enrich", ascending=False)
            .reset_index())
    agg.to_csv(os.path.join(SRC, "panel_roi_aggregate.csv"), index=False)
    return agg


def roi_style(d):
    """region -> colour, resolved ONCE here so the plotting script stays CSV-only.

    ``region_colors`` is the report's function and the vendored ``utils.roi_palette`` is
    still authoritative: the 13 whitelisted regions keep their vendored colours (the same
    ones the electrode_labeling brain figures use), and the regions the `tpm` scope adds --
    which the vendored palette cannot express and renders in one indistinguishable grey --
    get report-only colours assigned by sorted name so they are stable across runs.
    """
    rcol = region_colors(set(d["region"].astype(str)))
    out = pd.DataFrame(
        [{"region": r, "color": rcol[r], "vendored": rcol[r] == _color_of(r)}
         for r in sorted(rcol)])
    out.to_csv(os.path.join(SRC, "roi_style.csv"), index=False)
    return out


def roi_chance_band():
    """The shuffled-null chance band drawn on panel c — one row per task.

    Centre = the mean over participants of each participant's mean label-shuffled
    category-independent accuracy; half-width = the **SEM across participants**, i.e. the
    precision of the cohort's chance estimate, which is the quantity on the same footing as
    the markers (each marker is itself a cross-participant mean). Two alternatives were
    tried and rejected: a 2.5-97.5 percentile across participants (CI-like, width tracks n)
    and mean ± the pooled shuffles' SD (0.029 picture / 0.071 auditory, ~70x and ~14x the
    between-participant spread, because it measures how far a *single shuffle* moves — it
    swallowed the panel). Full record: docs/experiments/017.

    **RB is excluded from the band but stays in the markers** (Alec, 2026-08-13), via
    ``utils.config.OLD_STIMULUS_SET_PATIENTS`` ∩ cohort rather than by name — that
    membership already changed once, when CP was retired. RB ran the earlier stimulus set
    (7 picture / 5 auditory categories against 6/6), which puts its measured auditory null
    at 0.199 against ~0.167 for everyone else and made it the sole author of the band's
    upper edge. The markers and the reference therefore cover different cohorts; the caption
    says so.

    Averaged over ALL time bins, not the peak bin: chance does not depend on time (the two
    differ by < 0.004, measured) and a peak bin would tie this table to a run id.

    Source is the semantic_regression figure's shipped npz caches (< 1 MB), so no per-patient
    pkl is read. Those nulls come from the WHOLE-BRAIN decoder while panel c's markers are
    ROI-only decoders with far fewer channels, so a small region's own null would be wider:
    the band marks where chance sits, it does not test a region.
    """
    excluded = sorted(set(PATIENTS) & set(OLD_STIMULUS_SET_PATIENTS))
    keep = [p for p in PATIENTS if p not in set(OLD_STIMULUS_SET_PATIENTS)] or list(PATIENTS)
    rows = []
    for task in ("picture", "auditory"):
        npz = os.path.join(FIGS_ROOT, "semantic_regression",
                           f"panels_cache_{task}_GloVe.npz")
        if not os.path.exists(npz):
            raise SystemExit(
                f"{npz} not found -- panel c's chance band is measured from the shuffled "
                f"nulls and must not be replaced by a theoretical 1/n_categories. "
                f"Regenerate figures_for_paper/semantic_regression first.")
        z = np.load(npz)
        per_pat, absent = [], []
        for p in keep:
            k = f"{p}__category_indep__null"
            if k not in z:
                absent.append(p)
                continue
            v = np.asarray(z[k], dtype=float).ravel()
            v = v[np.isfinite(v)]
            if v.size:
                per_pat.append(float(v.mean()))
        if len(per_pat) < 2:
            raise SystemExit(f"only {len(per_pat)} participant null(s) in {npz}; "
                             f"a band needs at least 2.")
        a = np.asarray(per_pat)
        sem = float(a.std(ddof=1) / np.sqrt(a.size))
        rows.append(dict(task=task, centre=round(float(a.mean()), 6),
                         lo=round(float(a.mean() - sem), 6),
                         hi=round(float(a.mean() + sem), 6),
                         sem=round(sem, 6), n_participants=a.size,
                         excluded="; ".join(excluded),
                         missing_from_cache="; ".join(absent),
                         source_npz=os.path.basename(npz)))
    band = pd.DataFrame(rows)
    band.to_csv(os.path.join(SRC, "roi_chance_band.csv"), index=False)
    return band


# ── RSA (per-word neural geometry across tasks) ────────────────────────────

def rsa():
    """RETIRED FROM THE FIGURE 2026-08-13 (was supplement S7). Same status as ``mds()``:
    not called by main(), ``panel_s7_rsa.csv`` is left at its shipped N = 9 state."""
    r = pd.read_csv(os.path.join(COTRAIN_RUN, "cotrain_rsa_summary.csv"))
    r = r[r.patient.isin(PATIENTS)].copy()
    r.insert(0, "display_id", r["patient"].map(did))
    r.to_csv(os.path.join(SRC, "panel_s7_rsa.csv"), index=False)
    return r


# ── per-participant chance ─────────────────────────────────────────────────

def per_patient_chance(patients=PATIENTS):
    """Per (participant, task) chance for ``cat_indep_bal_acc`` = 1 / n_categories.

    Chance is task-specific, not a single cohort constant.  In
    ``cross_task_cotrain._run_conditions`` every condition that *tests* on
    picture is scored against ``db_pic`` and every condition that tests on
    auditory against ``db_aud``, so the denominator is the number of categories
    in the test task for that participant -- ``within_pic``/``cross_a2p``/
    ``pooled_pic`` use the picture set, ``within_aud``/``cross_p2a``/
    ``pooled_aud`` the auditory one.

    Counts are read from each run's own per-trial ``true_category`` rather than
    a nominal taxonomy, so they reflect what was actually scored.  Writes
    ``source_data/chance_by_participant.csv`` so every chance line in the figure
    traces back to a run id.
    """
    sem_dir = os.path.join(MAIN_DIR, "results", "semantic_regression")
    rows = []
    for pat in patients:
        for task, run in (("picture", PIC_RUN), ("auditory", AUD_RUN)):
            f = os.path.join(sem_dir, run, pat, "top1_decoding_source_data.csv")
            if not os.path.exists(f):
                continue
            col = pd.read_csv(f, usecols=["true_category"])["true_category"]
            cats = sorted({str(c) for c in col.dropna().unique()})
            k = len(cats) or _CHANCE_FALLBACK_N
            rows.append(dict(patient=pat, display_id=display_id(pat), task=task,
                             n_categories=k, chance=round(1.0 / k, 4),
                             categories="; ".join(cats)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SRC, "chance_by_participant.csv"), index=False)
    return df


# ── group inference (every headline number the Results text cites) ─────────

def group_inference(grp, stats, tab, roi_d, agg, band, have_roi):
    def gm(target, source, metric=METRIC_MAIN):
        row = grp[(grp.target == target) & (grp.source == source)
                  & (grp.metric == metric)].iloc[0]
        return row["mean"], row["sem"]

    pp_m, pp_s = gm("pic", "pooled")
    pa_m, pa_s = gm("aud", "pooled")
    wp_m, _ = gm("pic", "within")
    wa_m, _ = gm("aud", "within")
    cp_m, cp_s = gm("pic", "cross")
    ca_m, ca_s = gm("aud", "cross")

    rows = []
    add = lambda q, v, d="": rows.append(dict(quantity=q, value=v, detail=d))
    dropped = [p for p in ENROLLED_BOTH_TASKS if p not in PATIENTS]
    add("n_participants", len(PATIENTS),
        f"{' '.join(display_id(p) for p in PATIENTS)}; >=1 significant "
        f"category-independent time bin in BOTH tasks, from the semantic_regression "
        f"figure's source_data.csv (a different configuration: tp/h5 picture, tpfm/h10 "
        f"auditory). Dropped from the {len(ENROLLED_BOTH_TASKS)} with both tasks: "
        f"{', '.join(display_id(p) for p in dropped) or 'none'}")
    add("wilcoxon_p_floor", round(2.0 / 2 ** len(PATIENTS), 4),
        f"smallest attainable two-sided paired Wilcoxon p at n={len(PATIENTS)}; every "
        f"contrast at this floor is 'as significant as this cohort can show', not a "
        f"measured effect size")
    # Chance is per participant AND per task -- the cohort spans two auditory
    # stimulus sets, so one number cannot stand in for it.  Report each task's
    # mean with its spread; the full table is chance_by_participant.csv.
    ch = per_patient_chance()
    for task in ("picture", "auditory"):
        t = ch[ch.task == task]
        if t.empty:
            continue
        lo, hi = t["chance"].min(), t["chance"].max()
        spread = f"{lo:.4f}" if lo == hi else f"{lo:.4f}-{hi:.4f}"
        add(f"chance_cat_indep_{task}_mean", round(float(t["chance"].mean()), 4),
            f"mean of 1 / n_categories over {len(t)} participants; "
            f"per-participant range {spread}; n_categories "
            f"{int(t['n_categories'].min())}-{int(t['n_categories'].max())}")
    add("pooled_pic_cat_indep_mean", round(pp_m, 4), f"sem {pp_s:.4f}")
    add("pooled_aud_cat_indep_mean", round(pa_m, 4), f"sem {pa_s:.4f}")
    add("within_pic_cat_indep_mean", round(wp_m, 4), "full-data ceiling")
    add("within_aud_cat_indep_mean", round(wa_m, 4), "full-data ceiling")
    add("retention_pic", round(pp_m / wp_m, 3), "pooled / within (picture)")
    add("retention_aud", round(pa_m / wa_m, 3), "pooled / within (auditory)")
    add("cross_pic_cat_indep_mean", round(cp_m, 4),
        f"train aud test pic (cross_a2p); sem {cp_s:.4f}")
    add("cross_aud_cat_indep_mean", round(ca_m, 4),
        f"train pic test aud (cross_p2a); sem {ca_s:.4f}")
    for _, s in stats.iterrows():
        add(f"p_{s.target}_{s.comparison}_{s.metric}",
            f"{s.p_wilcoxon:.4g}",
            f"paired Wilcoxon (n={s.n}) {s.stars}; BH over the {len(stats)} tests in "
            f"panel_b_generalization_stats.csv q={s.q_bh:.4g} {s.stars_bh}. "
            f"THE FIGURE PLOTS THE UNCORRECTED p.")
    # ── ROI panels: the numbers the Results text cites ─────────────────────
    add("roi_atlas_scope_history", f"{ROI_ATLAS} / tpm / h10",
        f"{CROSS_TASK_FIGURE_ROI_DIR}; {agg['region'].nunique()} regions")
    add("roi_regions_low_n",
        "; ".join(f"{r} n={int(n)}" for r, n in
                  agg.loc[agg.n_participants <= 2, ["region", "n_participants"]].values),
        "regions contributed by 1-2 participants; kept and marked by marker size, not "
        "dropped")
    for tag, col, lab in (("jac", "jac_enrich", "per-electrode Jacobian enrichment"),
                          ("ko_pic", "ko_pic_pc", "picture knockout Δacc / electrode"),
                          ("ko_aud", "ko_aud_pc", "auditory knockout Δacc / electrode"),
                          ("suff_pic", "suff_pooled_pic", "ROI-only accuracy, picture"),
                          ("suff_aud", "suff_pooled_aud", "ROI-only accuracy, auditory")):
        top = agg.sort_values(col, ascending=False).head(3)
        add(f"roi_top3_{tag}",
            "; ".join(f"{r} {v:.4f}" for r, v in top[["region", col]].values),
            f"mean across participants, {lab}")
    for _, b in band.iterrows():
        add(f"chance_band_{b.task}", f"{b.centre:.4f} [{b.lo:.4f}, {b.hi:.4f}]",
            f"shuffled-null mean ± 1 SEM over {b.n_participants} participants"
            + (f"; excludes {', '.join(display_id(p) for p in b.excluded.split('; '))} "
               f"(earlier stimulus set)" if b.excluded else "")
            + f"; from {b.source_npz}")
    add("roi_sufficiency_size_control", "NONE",
        "suff_delta_* / suff_null_* / suff_p_* are NaN in every row: this arm ran with "
        "--suff-null-draws 0, so panel c has no matched-N null. Raw ROI-only accuracy "
        "rises with electrode count, so its cross-region ordering is partly an "
        "implant-coverage ordering")
    if "wb_imp_pic" in roi_d.columns:
        wb = roi_d.groupby("patient")[["wb_imp_pic", "wb_imp_aud"]].first()
        add("wb_ceiling_pic_mean", round(float(wb["wb_imp_pic"].mean()), 4),
            "mean whole-brain knockout Δcat-indep (picture) across participants")
        add("wb_ceiling_aud_mean", round(float(wb["wb_imp_aud"].mean()), 4),
            "mean whole-brain knockout Δcat-indep (auditory) across participants")
    if "wb_p_pic" in roi_d.columns:
        wbp = roi_d.groupby("patient")[["wb_p_pic", "wb_p_aud"]].first()
        for tag in ("pic", "aud"):
            v = wbp[f"wb_p_{tag}"].dropna()
            add(f"wb_ceiling_{tag}_n_significant",
                f"{int((v < _cfg.ALPHA).sum())}/{len(v)}",
                f"participants whose whole-brain knockout clears p<{_cfg.ALPHA} "
                f"(range {v.min():.3f}-{v.max():.3f})")
    n_sig = (roi_d[roi_d.group.isin(["both", "picture_only", "auditory_only"])]
             .groupby("patient").size()) if "group" in roi_d.columns else None
    if n_sig is not None:
        add("n_sig_regions_mean", round(float(n_sig.reindex(have_roi)
            .fillna(0).mean()), 2),
            "mean number of BH-FDR significant regions per participant (region knockout)")
    add("roi_coverage", "/".join(display_id(p) for p in have_roi),
        f"participants with an ROI atlas ({len(have_roi)}/{len(PATIENTS)} covered)")
    gi = pd.DataFrame(rows)
    gi.to_csv(os.path.join(SRC, "group_inference.csv"), index=False)
    return gi


def main():
    print("[compute_cross_task] writing ->", SRC)
    print(f"  cohort ({len(PATIENTS)}): {', '.join(PATIENTS)}  "
          f"[{', '.join(display_id(p) for p in PATIENTS)}]")
    per_pat, grp, stats = generalization()
    tab = retention(per_pat)
    # mds() and rsa() are NOT called -- see their docstrings. Their CSVs stay at N=9.
    roi_d, have_roi = roi()
    agg = roi_aggregate(roi_d)
    style = roi_style(roi_d)
    band = roi_chance_band()
    gi = group_inference(grp, stats, tab, roi_d, agg, band, have_roi)
    print(f"  ROI coverage: {[display_id(p) for p in have_roi]}")
    print(f"  regions: {len(agg)} ({ROI_ATLAS}); "
          f"{int((~style['vendored']).sum())} outside the vendored palette")
    print(f"  chance band: " + "  ".join(
        f"{b.task} {b.centre:.4f} [{b.lo:.4f},{b.hi:.4f}] n={b.n_participants}"
        for _, b in band.iterrows()))
    print("  group_inference.csv:")
    for _, r in gi.iterrows():
        print(f"    {r['quantity']:32s} {r['value']}  {r['detail']}")
    print("[compute_cross_task] done.")


if __name__ == "__main__":
    main()
