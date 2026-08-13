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
  * co-training conditions/RSA : <COTRAIN_RUN>/cotrain_conditions_summary.csv, cotrain_rsa_summary.csv
  * ROI region importance      : <ROI_DIR>/region_importance_<atlas>_all.csv (permutation Δacc
                                 + Jacobian, region-organized; the sufficiency columns in that
                                 file are NOT read here)
  * semantic-organization MDS  : utils.config.CROSS_TASK_FIGURE_MDS_RUN — pinned 2026-07-30
                                 (as MDS_RUN); was "latest matching glob" before that

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
                          AUD_RUN, PIC_RUN, p_stars)
from utils import config as _cfg   # noqa: E402
from utils.paths import latest_run_dir                                  # noqa: E402

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

#: Repointed to utils.config 2026-08-08 (PV and SE joined; 8 -> 10).
PATIENTS = list(_cfg.SHARED_PATIENTS)
METRIC_MAIN = "cat_indep_bal_acc"
METRICS = ["cat_indep_bal_acc", "word_bal_acc", "cosine_mean"]
#: Which atlas arm feeds the ROI panel. NMM is primary; the DK arm is computed and
#: archived alongside it, and which of them the FIGURE shows is an editorial decision
#: deferred until both sets of numbers exist.
ROI_ATLAS = _cfg.ROI_ATLAS_DEFAULT

#: The participant shown in the per-participant ROI panel -- DERIVED, not typed.
#:
#: Rule (decided 2026-08-09): **the participant whose strongest region effect is
#: significant in BOTH tasks and largest.** Concretely: restrict to regions labelled
#: ``group == "both"`` (knockout clears BH-FDR in picture and in auditory), score each
#: such region by ``min(perm_imp_pic_per_ch, perm_imp_aud_per_ch)`` -- the weaker of its
#: two tasks, so a region only counts as strong if it is strong in both -- and take the
#: participant owning the highest-scoring region.
#:
#: Why derived. ``LH`` used to be hard-coded because it had a strong signal in
#: ``post depth``, a ``primary_roi`` placeholder for depth-shank contacts that exists in
#: neither new atlas. The reason for the choice disappeared with the parcellation, and a
#: hand-picked exemplar whose justification no longer exists is indistinguishable from a
#: cherry-pick. ``panel a`` already derives its own representative this way; this makes
#: the two panels select by the same kind of documented procedure.
#:
#: Per-electrode, not totals: region totals correlate ~0.99 with channel count, so a
#: total-based pick would largely select the participant with the biggest implant.
#:
#: **The fallback is load-bearing.** At n=8 no region in any participant cleared BH-FDR
#: (``n_sig_regions_mean`` was 0.0), so ``group == "both"`` may be empty at n=10 too. When
#: it is, the same score is applied WITHOUT the significance restriction and the panel's
#: coverage CSV records ``selection_rule = "strongest-in-both (significance not
#: attainable)"``. That distinction has to survive into the source data, because "strongest
#: region that was significant in both" and "strongest region, none significant" are
#: different claims and the figure must not present the second as the first.
SELECTION_RULE = "strongest-in-both"


def _pick_representative(d):
    """(participant, rule_actually_used) under the rule documented above.

    *d* is the region table for METRIC_MAIN, already restricted to PATIENTS.
    """
    pic = "perm_imp_pic_per_ch" if "perm_imp_pic_per_ch" in d.columns else "perm_imp_pic"
    aud = "perm_imp_aud_per_ch" if "perm_imp_aud_per_ch" in d.columns else "perm_imp_aud"
    if pic not in d.columns or aud not in d.columns:
        raise SystemExit(
            f"cannot pick a representative: {pic}/{aud} missing from the region CSV. "
            f"Was the run made without --analysis permutation?")

    # skipna=False is load-bearing: pandas' default SKIPS NaN, so a region missing one
    # task's effect would score on the other task alone -- exactly inverting the meaning
    # of min() here, which is "strong in BOTH". A region with either task missing must be
    # ineligible, not eligible on half the evidence.
    scored = d.assign(_score=d[[pic, aud]].min(axis=1, skipna=False))
    rule = SELECTION_RULE
    pool = scored[scored.get("group", "") == "both"] if "group" in scored.columns \
        else scored.iloc[0:0]
    if pool.empty:
        # Expected at this cohort size; see the note above. Not silent.
        rule = SELECTION_RULE + " (significance not attainable)"
        pool = scored
        print("  [roi] no region is significant in BOTH tasks for any participant -- "
              "falling back to the strongest region regardless of significance. "
              "This is recorded in panel_c_roi_coverage.csv.")
    pool = pool[np.isfinite(pool["_score"])]
    if pool.empty:
        raise SystemExit("cannot pick a representative: no finite knockout scores.")
    best = pool.loc[pool["_score"].idxmax()]
    print(f"  [roi] representative = {best['patient']} "
          f"(region {best['region']}, score {best['_score']:.4f}, rule: {rule})")
    return str(best["patient"]), rule

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
            for i, j in [(0, 1), (0, 2), (1, 2)]:
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
    pd.DataFrame(grp).to_csv(
        os.path.join(SRC, "panel_b_generalization_group.csv"), index=False)
    pd.DataFrame(stats).to_csv(
        os.path.join(SRC, "panel_b_generalization_stats.csv"), index=False)
    return per_pat, pd.DataFrame(grp), pd.DataFrame(stats)


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
    """Single region-organized product feeding the consolidated ROI figure:
    per patient x region, from region_importance_all.csv (permutation Δacc pic/aud
    + significance, Jacobian pic/aud, neural-GloVe covariance) plus the whole-brain
    ceiling / share. All 6 patients now have an atlas.

    Plain-PLS VIP was removed from the pipeline 2026-07-23 (it attributed a linear
    surrogate the paper does not report, and as a region total it was an
    electrode-count proxy), so `vip`/`vip_std` are no longer requested and the S4
    VIP supplement is gone. Covariance columns are carried instead."""
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

    representative, selection_rule = _pick_representative(d)
    d.insert(0, "display_id", d["patient"].map(did))
    keep = [c for c in [
        "display_id", "patient", "region", "n_channels",
        "perm_imp_pic", "perm_imp_aud", "perm_imp_pic_per_ch",
        "perm_imp_aud_per_ch", "p_pic", "p_aud", "q_pic", "q_aud", "group",
        "jac_sens_pic", "jac_sens_aud", "cov_nc_pic", "cov_nc_aud",
        "wb_imp_pic", "wb_imp_aud", "wb_p_pic", "wb_p_aud",
        "frac_wb_pic", "frac_wb_aud",
    ] if c in d.columns]
    d[keep].to_csv(os.path.join(SRC, "panel_c_roi.csv"), index=False)

    cov = pd.DataFrame([
        dict(display_id=did(p), patient=p, has_roi=(p in have),
             is_representative=(p == representative),
             selection_rule=selection_rule, roi_atlas=ROI_ATLAS,
             note="" if p in have
             else "No ROI atlas available for this participant")
        for p in PATIENTS])
    cov.to_csv(os.path.join(SRC, "panel_c_roi_coverage.csv"), index=False)
    return d, have, representative, selection_rule


# ── RSA (per-word neural geometry across tasks) ────────────────────────────

def rsa():
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

def group_inference(grp, stats, tab, align, rep_mds, roi_d, have_roi,
                    roi_rep, roi_rule):
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
    add("n_participants", len(PATIENTS),
        f"{' '.join(PATIENTS)} (both PN + AN)")
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
    for _, s in stats[(stats.metric == METRIC_MAIN)].iterrows():
        add(f"p_{s.target}_{s.comparison}_cat_indep",
            f"{s.p_wilcoxon:.4g}", f"paired Wilcoxon (n={s.n}) {s.stars}")
    # ── ROI region importance headline numbers ─────────────────────────────
    # top region per patient by picture Δacc, and its share of the whole-brain
    # ceiling (frac_wb_pic); mean whole-brain ceiling across participants.
    if "wb_imp_pic" in roi_d.columns:
        wb = roi_d.groupby("patient")["wb_imp_pic"].first()
        add("wb_ceiling_pic_mean", round(float(wb.mean()), 4),
            "mean whole-brain knockout Δcat-indep (picture) across participants")
    if "frac_wb_pic" in roi_d.columns:
        top_share = (roi_d.sort_values("perm_imp_pic", ascending=False)
                     .groupby("patient").first())
        add("top_region_share_pic_mean",
            round(float(top_share["frac_wb_pic"].mean()), 3),
            "mean share of the whole-brain ceiling held by each patient's top "
            "picture region")
    n_sig = (roi_d[roi_d.group.isin(["both", "picture_only", "auditory_only"])]
             .groupby("patient").size()) if "group" in roi_d.columns else None
    if n_sig is not None:
        add("n_sig_regions_mean", round(float(n_sig.reindex(have_roi)
            .fillna(0).mean()), 2),
            "mean number of significant regions per participant")
    add("roi_representative", did(roi_rep),
        f"{roi_rep}: representative region-importance participant "
        f"(selected by rule: {roi_rule})")
    ar = align[align.patient == rep_mds].iloc[0]
    add("mds_representative", did(rep_mds),
        f"cross-task category-centroid alignment "
        f"{ar.cat_centroid_alignment:.3f}, p {ar.cat_centroid_alignment_p:.3g}")
    add("mds_alignment_mean",
        round(float(align.cat_centroid_alignment.mean()), 3),
        "mean cross-task category-centroid alignment across participants")
    add("roi_coverage", "/".join(display_id(p) for p in have_roi),
        f"participants with an ROI atlas ({len(have_roi)}/{len(PATIENTS)} covered)")
    gi = pd.DataFrame(rows)
    gi.to_csv(os.path.join(SRC, "group_inference.csv"), index=False)
    return gi


def main():
    print("[compute_cross_task] writing ->", SRC)
    per_pat, grp, stats = generalization()
    tab = retention(per_pat)
    align, rep_mds, cats = mds()
    roi_d, have_roi, roi_rep, roi_rule = roi()
    rsa()
    gi = group_inference(grp, stats, tab, align, rep_mds, roi_d, have_roi,
                         roi_rep, roi_rule)
    print(f"  representative: MDS={display_id(rep_mds)} ({rep_mds}), "
          f"ROI={display_id(roi_rep)} ({roi_rep}) via {roi_rule}")
    print(f"  categories: {cats}")
    print(f"  ROI coverage: {[display_id(p) for p in have_roi]}")
    print("  group_inference.csv:")
    for _, r in gi.iterrows():
        print(f"    {r['quantity']:32s} {r['value']}  {r['detail']}")
    print("[compute_cross_task] done.")


if __name__ == "__main__":
    main()
