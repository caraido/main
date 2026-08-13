# -*- coding: utf-8 -*-
"""Repo-level settings: pinned runs, statistical thresholds, figure style.

The counterpart to ``utils/paths.py``.  ``paths`` answers *where output goes*;
this module answers *which run is authoritative* and *what counts as
significant*.  Both exist for the same reason: the value used to live in half a
dozen places at once, and the copies drifted.

Before this module there were three independent ``PIC_RUN_DEFAULT`` constants
with identical values, three ``AUD_RUN_DEFAULT`` (two of them split across
source lines, so a naive grep found only one), an absolute ``D:\\OneDrive - …``
path in the now-archived ``_archive/legacy/seen_unseen_analysis.py``, two
identical ``SIG_ALPHA = 0.05``, and
five separate implementations of the same ``***``/``**``/``*`` ladder.
Repointing the pinned auditory run meant editing seven files and hoping.

Import from here instead of typing a literal::

    from utils.config import AUD_RUN, PIC_RUN, ALPHA, PCTILE, p_stars

Every tree can reach this: the import graph is a strict DAG pointing *into*
``utils/`` (``figures_for_paper`` -> ``analysis`` -> ``utils``, plus the root
scripts and ``tests/``), and this module imports nothing outside the stdlib and
``utils.paths``.  Scripts under ``figures_for_paper/`` need ``MAIN_DIR`` on
``sys.path`` first -- they already compute it.

WHY THIS IS A ``.py`` UNDER ``utils/`` AND NOT A JSON AT THE REPO ROOT
---------------------------------------------------------------------
``utils/audit_runs.py`` decides whether a run directory is ``PINNED`` or
``unreferenced`` by grepping tracked source for the literal run-id string, over
``SCAN_DIRS = ("figures_for_paper", "analysis", "tests", "notebooks", "report",
"utils")`` with ``SCAN_SUFFIXES = (".py", ".ipynb", ".md")``.  A ``.py`` file
under ``utils/`` is inside that surface, so run ids parked here keep their pins.
A root-level ``config.json`` would be outside it on *both* axes, every pinned
run would flip to ``unreferenced``, and ``AGENTS.md`` authorises pruning those --
roughly 50 GB.  If this file is ever converted to a data file, extend
``audit_runs.SCAN_DIRS``/``SCAN_SUFFIXES`` in the same commit.

Superseded runs stay named here for the same reason: a run that is still the
provenance of a shipped figure must not read as unreferenced just because the
default moved on.
"""

from __future__ import annotations

from pathlib import Path

from utils.paths import results_dir

# ── Pinned runs ───────────────────────────────────────────────────────────────
# Run ids under results/semantic_regression/.  Change them HERE, nowhere else;
# every consumer keeps its own CLI flag for a one-off override.
#
# KEEP EACH RUN ID ON ONE LINE.  ``audit_runs`` matches the literal against the
# directory name, so an implicitly concatenated string ("…group_" "align-…")
# yields only the first fragment, the run reads as ``unreferenced``, and the
# pruning plan in docs/repo_layout.md would then treat it as deletable.  This is
# not hypothetical: two of the three old AUD_RUN_DEFAULT constants were split
# that way and a full-string grep found only one of them.  Long lines here are
# correct; do not let a formatter wrap them.
# fmt: off
# flake8: noqa: E501

#: Picture naming, 100 epochs, **15 participants, NMM-gated, 5-bin history**.  The
#: paper's picture-naming run.  Repinned 2026-08-09.
#:
#: Not an extension of PIC_RUN_100EP_N13 below -- it is a different analysis.  The
#: channel set is the 13-region temporal-parietal whitelist applied to ``nmm_roi``
#: (633 channels kept across the cohort, against whole-brain before), and the
#: feature window is 500 ms rather than 1000 ms.  Numbers from the two are not
#: comparable and must never be pooled.
PIC_RUN = "2026-08-09_10-17-27_picture_naming_roi-nmm_h5_kernel_pls_cosine_100ep"

#: Auditory naming, group-warped, aligned to auditory stimulus onset, 100 epochs,
#: **10 participants, NMM-gated, 5-bin history**.  Repinned 2026-08-09.
#:
#: The warp target was RECOMPUTED for this cohort rather than pinned: PV and SE
#: joined, so the pooled median moved 3.5800 s -> **3.560 s** and every participant
#: was re-warped.  ``meta.json`` records ``auditory_warp_target_source: computed``.
#: That is deliberate -- the whole auditory arm is being replaced, so continuity
#: with the retired runs buys nothing and 3.5800 s described a cohort that no
#: longer exists.  420 channels kept across the cohort.
AUD_RUN = "2026-08-09_09-04-16_auditory_naming_warp-stim-group_align-aud_stim_onset_roi-nmm_h5_kernel_pls_cosine_100ep"

#: The 13-participant, whole-brain, 10-bin predecessor of PIC_RUN.  Superseded
#: 2026-08-09 by the NMM-gated re-run.  Retained and named for the reason given at
#: the top of this section: it is the provenance of every figure not yet
#: regenerated, and an unnamed run id reads as ``unreferenced`` in
#: docs/results_index.md, which AGENTS.md then authorises pruning.
PIC_RUN_100EP_N13 = "2026-06-02_17-25-11_picture_naming_kernel_pls_cosine_100ep"

#: The 8-participant, whole-brain, 10-bin predecessor of AUD_RUN, group-warped to
#: 3.5800 s.  Superseded 2026-08-09.  Retained for the same reason.
AUD_RUN_100EP_N8 = "2026-07-28_16-59-35_auditory_naming_warp-stim-group_align-aud_stim_onset_kernel_pls_cosine_100ep"

#: The 6-participant predecessor of AUD_RUN (AA AZ DR LH RB WBH), group-warped to
#: 3.500 s.  Superseded 2026-07-28 and retained for exactly the reason given at the
#: top of this section: it is the provenance of every auditory figure not yet
#: regenerated against the 7-participant run, and a run id that stops appearing in
#: tracked source reads as ``unreferenced`` in docs/results_index.md, which
#: AGENTS.md then authorises pruning.
AUD_RUN_N6 = "2026-07-13_11-58-22_auditory_naming_warp-linear-group_align-aud_stim_onset_kernel_pls_cosine_100ep"

#: Picture naming, 50 epochs.  Superseded by PIC_RUN.  As of 2026-07-30 it has NO
#: consumers: cross_task_cotrain, cross_task_regression, open_vocab_retrieval.predict_io
#: and figures_for_paper/cross_task all moved to PIC_RUN, which ended the epoch asymmetry
#: (a 50-epoch picture arm against a 100-epoch auditory one left the two arms'
#: permutation nulls unequally resolved -- p floors at ~1/(n_epochs+1)).
#: It stays named here anyway, and that is the whole point of this block: it is the
#: provenance of every cross_task / open_vocab / extendability figure produced before
#: that date, and a run id that stops appearing in tracked source reads ``unreferenced``
#: in docs/results_index.md, which AGENTS.md then authorises pruning.  Do not delete.
PIC_RUN_50EP = "2026-04-08_17-05-14_kernel_pls_cosine_50ep"

#: Auditory naming, per-participant warp, 50 epochs, same 6 participants as
#: AUD_RUN.  Superseded, retained for the same reason as PIC_RUN_50EP.
AUD_RUN_50EP = "2026-05-07_22-26-06_auditory_naming_warp-linear_align-aud_stim_onset_kernel_pls_cosine_50ep"

#: Unbalanced-class control run, read by the cross-task ROI importance figure.
#: Repointed 2026-08-09 to the TEN-participant, NMM-gated run (PV and SE added); its
#: inputs are the current PIC_RUN and AUD_RUN, both 100 epochs, both 5-bin.
NONE_BALANCE_RUN = "2026-08-09_20-42-51_kernel_pls_balance-none_50boot"

#: The 8-participant, whole-brain predecessor.  Superseded 2026-08-09; retained and
#: named so it does not read as unreferenced.
NONE_BALANCE_RUN_N8 = "2026-07-30_15-39-14_kernel_pls_balance-none_50boot"

#: The class-BALANCED counterpart of NONE_BALANCE_RUN: identical inputs, with the
#: pooled training set downsampled to equalise the picture and auditory arms.
#:
#: New on 2026-08-09.  ``region_importance`` has always been run for both balance
#: settings, but ``cross_task_cotrain`` itself never was -- no
#: ``balance-downsample`` cotrain run existed in any cohort before this one, so the
#: co-training conditions had no imbalance control at all.  Picture outnumbers
#: auditory by roughly 3:1 overall and by 5:1 in the largest participant, which is
#: exactly the regime where a pooled decoder can look like it transfers when it is
#: really just fitting the majority task.
#:
#: Not read by any figure yet -- it is the control the figure's claim should be
#: checked against, not a replacement for NONE_BALANCE_RUN.
DOWNSAMPLE_BALANCE_RUN = "2026-08-09_23-15-22_kernel_pls_balance-downsample_50boot"

#: Semantic-organization MDS run (cross_task_prediction_mds.py), read by panel a of the
#: cross-task figure.  Pinned 2026-07-30.  Before that this was the ONLY input to a paper
#: figure resolved by "newest matching glob" rather than by a pin: re-running the MDS
#: silently repointed panel a, and — worse — the run the shipped figure depended on read
#: ``unreferenced`` in docs/results_index.md, which AGENTS.md authorises pruning.
MDS_RUN = "2026-08-09_20-45-49_prediction_mds_separate_kfold5_seed42"

#: The 8-participant, whole-brain predecessor of MDS_RUN.  Superseded 2026-08-09.
MDS_RUN_N8 = "2026-07-30_15-43-11_prediction_mds_separate_kfold5_seed42"

#: The two runs that isolate the 2026-07-30 changes from each other.  Both hold the
#: SEVEN pre-KAW participants and differ only in the picture arm's epoch count, so
#: diffing their ``cotrain_conditions_summary.csv`` measures the 50->100 epoch effect
#: with the cohort held fixed -- which is why they are kept rather than pruned.  That
#: diff was run on 2026-07-30: every picture-involving condition moved by +0.000 to
#: +0.006 cat_indep_bal_acc, and ``within_aud`` moved by EXACTLY 0.000, confirming the
#: auditory arm was untouched.  ``_50EP`` is additionally the provenance of every
#: cross_task figure shipped before 2026-07-30.
NONE_BALANCE_RUN_N7_50EP = "2026-07-28_20-09-58_kernel_pls_balance-none_50boot"
NONE_BALANCE_RUN_N7_100EP = "2026-07-30_15-23-26_kernel_pls_balance-none_50boot"

#: The 6-participant predecessor of NONE_BALANCE_RUN.  Superseded 2026-07-28,
#: retained for the same reason: it is the provenance of the cross_task figure
#: as shipped before CP joined the auditory cohort.
NONE_BALANCE_RUN_N6 = "2026-06-30_12-54-54_kernel_pls_balance-none_50boot"

#: The auditory arm of figures_for_paper/semantic_regression: the **23-region `tpfm` scope
#: at 10 bins / 1000 ms of history** (723 contacts), against PIC_RUN's 13 regions at 5 bins.
#: Adopted 2026-08-11 on Alec's instruction.
#:
#: **That figure's two tasks now differ on BOTH axes — channels and history.** The methods
#: must say so; "temporal-parietal cortex" no longer describes its auditory arm, which
#: includes frontal and medial/deep contacts.
#:
#: Chosen on the figure's own estimator, over the other three cells of the 2x2
#: (docs/experiments/001).  It has the highest cohort peak on all four metrics, and the
#: count of participants with any significant bin rises on three: category 4/10 -> 8/10,
#: top-5 6/10 -> 8/10, top-3 7/10, top-1 10/10 -> 9/10 (the one regression).
#:
#: Two caveats that must travel with it.  (1) No individual auditory contrast in the 2x2
#: survives BH-FDR — the arm was selected on effect size and n_sig, not on significance.
#: (2) The auditory runs are time-warped, so a longer window is also more tolerant of
#: residual warp misalignment; accumulation and misalignment have NOT been separated.
#: The palette problem does not apply here — these panels colour by participant, not region.
#:
#: Deliberately NOT AUD_RUN: repointing that would change ~10 other modules (cross_task,
#: open_vocab_retrieval, the co-training arms), none of which have
#: been re-run at this configuration.  Every other auditory analysis still uses AUD_RUN at
#: `tp`/5 bins, and this figure is currently the only one that does not.
AUD_RUN_FIGURE = "2026-08-11_04-49-25_auditory_naming_warp-stim-group_align-aud_stim_onset_roi-nmm_scope-tpfm_h10_kernel_pls_cosine_100ep"

# ── Diagnostic runs — NOT paper runs ─────────────────────────────────────────
#: The 2026-08-11 history x ROI-scope factorial, run to find out which of the two changes
#: made in the 2026-08 re-run (narrower channel gate, shorter history window) cost ~5% of
#: decoding performance.  Full record: docs/experiments/001-history-and-scope-diagnostic.md.
#:
#: Named here so audit_runs sees them as pinned rather than prunable -- audit_runs does not
#: scan docs/, so the experiment entry alone would leave them deletable.
#:
#: **Two of them are no longer diagnostic-only** (amended 2026-08-11).  The `tpfm`/h10 pair
#: -- picture `..._11-37-57_...` and auditory `..._04-49-25_...` -- is read by
#: AUD_RUN_FIGURE above (the semantic_regression auditory panels) and by the cross-task
#: co-training + ROI-importance arm under
#: results/cross_task_cotrain/scope-tpfm_h10/.  The other four are still read by nothing.
#:
#: Do not promote any of them to PIC_RUN/AUD_RUN without re-running with the full embedding
#: set: they are GloVe-only.  That is not a problem for the consumers above -- the
#: semantic_regression panels and the whole cross-task chain read GloVe only -- but it is
#: for anything that reads a second embedding.
#:
#: 'tpfm' is the diagnostic 23-region scope (utils.roi_scopes); the baseline corner of the
#: 2x2 is PIC_RUN / AUD_RUN themselves, which are 'tp' at 5 bins.
#: One id per line -- audit_runs matches the literal string, so a wrapped id is invisible.
SCOPE_DIAGNOSTIC_RUNS = (
    "2026-08-11_00-51-28_auditory_naming_warp-stim-group_align-aud_stim_onset_roi-nmm_scope-tpfm_h5_kernel_pls_cosine_100ep",
    "2026-08-11_02-50-41_auditory_naming_warp-stim-group_align-aud_stim_onset_roi-nmm_scope-tp_h10_kernel_pls_cosine_100ep",
    "2026-08-11_04-49-25_auditory_naming_warp-stim-group_align-aud_stim_onset_roi-nmm_scope-tpfm_h10_kernel_pls_cosine_100ep",
    "2026-08-11_06-56-18_picture_naming_roi-nmm_scope-tpfm_h5_kernel_pls_cosine_100ep",
    "2026-08-11_10-01-39_picture_naming_roi-nmm_scope-tp_h10_kernel_pls_cosine_100ep",
    "2026-08-11_11-37-57_picture_naming_roi-nmm_scope-tpfm_h10_kernel_pls_cosine_100ep",
)

#: The cross-task co-training runs on that same `tpfm`/h10 pair, 2026-08-11.  Side-by-side
#: with NONE_BALANCE_RUN / DOWNSAMPLE_BALANCE_RUN, which stay pointed at the `tp`/h5 arm and
#: still feed the shipped cross-task figure -- nothing here is read by a figure.
#:
#: Their ROI-importance outputs live under
#: results/cross_task_cotrain/scope-tpfm_h10/balance_{none,downsample}/, written with --out
#: so the pinned balance_* directories were never touched.  Those passes ran WITHOUT
#: --roi-sufficiency (Alec, 2026-08-11): 38 columns rather than 54, and the report has no
#: sufficiency section.  Do not diff them against a 54-column CSV and read the missing
#: columns as a regression.
SCOPE_DIAGNOSTIC_COTRAIN_RUNS = (
    "2026-08-11_16-11-01_kernel_pls_balance-none_50boot",
    "2026-08-11_16-14-16_kernel_pls_balance-downsample_50boot",
)

#: The `tpm` rung of the scope ladder, 2026-08-12.  `tpm` is `tp` + MEDIAL (18 regions), so
#: `tp` -> `tpm` -> `tpfm` separates the medial half of the tpfm gain from the frontal half;
#: `tpfm` moved both at once and could not attribute it.  All four are n=9 (CP retired) and
#: GloVe-only, like SCOPE_DIAGNOSTIC_RUNS -- do not promote either pair to PIC_RUN/AUD_RUN.
#:
#: The h5 and h10 auditory runs share a warp target of **3.5600 s**: the h10 run computed it
#: over its own 10 participants and the h5 run pinned it to that value, so the h5-vs-h10
#: contrast is history-only and no warp difference is confounded with it.  NB the h10
#: auditory run's 10 participants still include CP -- it predates nothing, CP is simply
#: filtered downstream, which is why every cotrain and ROI output below is n=9.
TPM_LADDER_RUNS = (
    "2026-08-11_23-42-55_picture_naming_roi-nmm_scope-tpm_h10_kernel_pls_cosine_100ep",
    "2026-08-12_09-14-11_auditory_naming_warp-stim-group_align-aud_stim_onset_roi-nmm_scope-tpm_h10_kernel_pls_cosine_100ep",
    "2026-08-12_10-14-58_picture_naming_roi-nmm_scope-tpm_h5_kernel_pls_cosine_100ep",
    "2026-08-12_16-01-46_auditory_naming_warp-stim-group_align-aud_stim_onset_roi-nmm_scope-tpm_h5_kernel_pls_cosine_100ep",
)

#: The cross-task co-training runs on the `tpm` pairs above, both balance settings.  The
#: **downsample** arm is the control the ladder is read against: picture outnumbers auditory
#: ~3:1, and a pooled decoder can look like it transfers while fitting the majority task.
#: **The tpm/h10 downsample arm is no longer diagnostic** (2026-08-12): it is the input to
#: figures_for_paper/cross_task, pinned again below as CROSS_TASK_FIGURE_COTRAIN_RUN so the
#: figure's provenance does not depend on reading this tuple's ordering.  The other three
#: still feed nothing.
TPM_LADDER_COTRAIN_RUNS = (
    "2026-08-12_18-09-39_kernel_pls_balance-none_50boot",        # tpm/h10
    "2026-08-12_18-17-20_kernel_pls_balance-downsample_50boot",  # tpm/h10, control
    "2026-08-12_19-41-18_kernel_pls_balance-none_50boot",        # tpm/h5
    "2026-08-12_19-47-11_kernel_pls_balance-downsample_50boot",  # tpm/h5, control
)

#: Their ROI-importance outputs, written with --out so the pinned balance_* directories were
#: never touched.  Named here as directory literals because audit_runs cannot pin a name with
#: no timestamp in it any other way, and an unpinned directory reads ``unreferenced``.
#:
#: Like the tpfm arm these ran WITHOUT --roi-sufficiency: **38 columns, not 54, and the HTML
#: report has no sufficiency section.**  NMM only -- there is no DK or merged arm.  Do not
#: diff them against a 54-column CSV and read the missing columns as a regression.
TPM_LADDER_ROI_DIRS = (
    "scope-tpm_h10",
    "scope-tpm_h5",
)

# ── The cross-task paper figure's inputs ─────────────────────────────────────
#: figures_for_paper/cross_task moved from `tp`/h5 + balance=none to **`tpm`/h10 +
#: balance=downsample** on 2026-08-12 (Alec).  Named separately from the ladder tuples above
#: because these three say "the figure reads this", which is a different claim from "this run
#: exists" -- and because compute_cross_task_data.py must import a name, never a literal.
#:
#: Two consequences that have to travel with the figure, both in its caption:
#:   1. `tpm` is 18 regions, adding insula, cingulate, entorhinal, parahippocampal and
#:      precuneus.  **"Temporal-parietal cortex" no longer describes this figure's channel
#:      set** -- the same caveat already carried by AUD_RUN_FIGURE.
#:   2. Trials are now class-balanced by downsampling.  The previous figure pooled with no
#:      resampling and its Results text argued for that explicitly; that argument is void.
#:
#: The superseded `tp`/h5 arm stays named above (NONE_BALANCE_RUN, MDS_RUN) -- it is the
#: provenance of every cross-task figure shipped before this date, and an unnamed run id
#: reads ``unreferenced`` in docs/results_index.md, which AGENTS.md authorises pruning.
CROSS_TASK_FIGURE_COTRAIN_RUN = "2026-08-12_18-17-20_kernel_pls_balance-downsample_50boot"

#: Region-importance output for the same pair, relative to results/cross_task_cotrain/.
#: NMM only -- there is no DK or merged arm at `tpm`, so the figure cannot be rendered on
#: the DK atlas without a new pass.
#:
#: Written 2026-08-13 with ``--roi-sufficiency --suff-null-draws 0``: the raw ROI-only
#: accuracy columns are present, and ``suff_delta_*`` / ``suff_null_*`` / ``suff_p_*`` are
#: NaN by construction.  **The report for this arm therefore carries no size control** and
#: says so; raw sufficiency rises with electrode count, so a cross-region ranking on it is
#: substantially an implant-coverage ranking.  The figure reads none of these columns.
#:
#: Its knockout columns are reproducible from that recorded command -- a control pass
#: without ``--roi-sufficiency`` matched it exactly.  The arm's *previous* contents (19:40,
#: same day) did not match, and its flags are unrecoverable because region-importance
#: writes no manifest.  The sibling arms (``scope-tpm_h5``, ``scope-tpfm_h10``, the
#: ``balance_*`` pair) are still in that unattributable state, so rung-to-rung ladder
#: differences below ~0.03 acc are not safely readable as scope or history effects.
CROSS_TASK_FIGURE_ROI_DIR = "scope-tpm_h10/balance_downsample"

#: Panel a's semantic-organization MDS at the same `tpm`/h10 pair, run 2026-08-13 so the
#: whole figure sits on one configuration.  MDS_RUN above stays pinned as the `tp`/h5
#: predecessor and the provenance of panel a as shipped before this date.
CROSS_TASK_FIGURE_MDS_RUN = "2026-08-13_00-22-11_prediction_mds_separate_kfold5_seed42"


def run_dir(run_id: str, analysis: str = "semantic_regression") -> Path:
    """Return the results directory for ``run_id`` without creating it.

    ``create=False`` is deliberate: a typo in a run id should raise where it is
    read, not silently mkdir an empty directory that then shows up in
    ``docs/results_index.md`` as ``incomplete``.
    """
    return results_dir(analysis, run_id, create=False)


# ── Cohort ────────────────────────────────────────────────────────────────────
# Added 2026-08-08.  The same duplication this module was created to kill, one
# axis over: ``["AA","AZ","CP","DR","KAW","LH","RB","WBH"]`` was typed verbatim in
# five live files and the picture list in four more, so adding a participant meant
# editing nine files and hoping.  Import these; keep each consumer's --patients
# flag for a one-off override.
#
# These are the participants the ANALYSIS uses.  They are NOT the participants on
# disk: ``utils.patient_data.discover_patients`` returns whatever has a
# ``{PAT}_{task}_df.pkl``, so a run launched without ``--patients`` takes its
# cohort from the filesystem and silently changes when data lands.  Pass
# ``--patients`` explicitly whenever a run must reproduce an existing result.

#: Participants withdrawn from the analysis, mapped to WHY.  **This is the one switch.**
#:
#: A retired participant is not a deleted one: their data moves to ``data_archive/`` and
#: their existing runs stay on disk, because a result that was produced was produced.  What
#: retirement means is that nothing NEW includes them and nothing reported does.
#:
#: Shaped after ``utils.rois.EXCLUDED_CONTACTS`` -- table plus reason plus accessor -- which
#: is how this repository already does exclusions, one level down at the contact.
#:
#: Retiring someone here is NOT sufficient on its own.  The cohort has three sources and
#: this constant only governs the first:
#:   1. these tuples                       -- covered below
#:   2. ``utils.patient_data.discover_patients``, a scan of ``data/``  -- filtered there
#:   3. ``os.listdir(run_dir)`` in semantic_regression_panels          -- filtered there
#: An existing run directory still contains the retired participant's outputs, so anything
#: walking a run directory must filter too.  See ``docs/experiments/015``.
RETIRED_PATIENTS = {
    "CP": "retired 2026-08-12 by group consensus; results are not reported",
}

#: The ENROLLED rosters -- the historical record, retained deliberately.  The analysis
#: cohorts below are DERIVED from them, so retiring someone is one edit above rather than a
#: hunt through tuples; the same reason ``_cross_patient_helpers`` derives its target list
#: rather than typing it.  Do not delete a name from these: that would erase the fact that
#: the participant was ever enrolled.
_ENROLLED_PICTURE = ("AA", "AP", "AZ", "CP", "DR", "EH", "EM", "KAW", "LH", "MM", "PV", "RB", "SE", "VB", "WBH")
_ENROLLED_AUDITORY = ("AA", "AZ", "CP", "DR", "KAW", "LH", "PV", "RB", "SE", "WBH")


def active_patients(patients):
    """*patients* with the retired ones dropped, order preserved."""
    return tuple(p for p in patients if p not in RETIRED_PATIENTS)


def assert_not_retired(patients):
    """Raise if *patients* names a retired participant.

    Deliberately a hard error rather than a silent drop.  ``--patients`` is an explicit
    request, and quietly returning a smaller cohort than was asked for would change N --
    and therefore every Bonferroni denominator and every "k of N" count -- behind the
    reader's back.  Better to stop and say whose data it is and why it is gone.
    """
    hit = [p for p in patients if p in RETIRED_PATIENTS]
    if hit:
        detail = "\n".join(f"    {p}: {RETIRED_PATIENTS[p]}" for p in hit)
        raise SystemExit(
            f"{len(hit)} requested participant(s) are retired:\n{detail}\n"
            f"  Their data is under data_archive/ and their existing runs are kept, but no "
            f"new analysis may include them. Drop them from --patients.")


#: Picture naming.  Enrolled 15; **14 analysed** (CP retired 2026-08-12).
PICTURE_PATIENTS = active_patients(_ENROLLED_PICTURE)

#: Auditory naming.  CP joined 2026-07-28, KAW 2026-07-30, PV and SE 2026-08-06.
#: Enrolled 10; **9 analysed** (CP retired 2026-08-12).
AUDITORY_PATIENTS = active_patients(_ENROLLED_AUDITORY)

#: Participants with BOTH tasks -- the cross-task cohort.  Equal to
#: ``AUDITORY_PATIENTS`` as a matter of *fact* (every auditory participant also did
#: picture naming), not by definition.  Aliased rather than re-typed so the two
#: cannot drift, but kept as its own name because the two mean different things and
#: a future participant could break the equality.
SHARED_PATIENTS = AUDITORY_PATIENTS

#: Participants who ran the EARLIER auditory stimulus set: longer spoken prompts
#: (median 4.72 s CP / 4.64 s RB against ~3.2-3.6 s for the current set) and a different
#: category inventory -- it adds `abstract` and `action` and drops `vehicle`.
#:
#: A property of those recordings, not a cohort definition, which is why it lives here as
#: its own name rather than being folded into the tuples above.  It is why chance for
#: `cat_indep_bal_acc` is PER PARTICIPANT (0.143-0.200) and never a flat 1/6.
#:
#: CP stays listed after retirement: the fact remains true of CP's archived data, and the
#: consumers intersect this with the run's actual cohort.  **With CP retired, RB is the
#: only old-set participant in the analysis** -- keep saying so in Methods rather than
#: letting the caveat quietly disappear (Alec, 2026-08-12).
OLD_STIMULUS_SET_PATIENTS = ("CP", "RB")

#: The two ECoG participants; the rest are sEEG.  Count deliberately not stated -- it is
#: ``len(PICTURE_PATIENTS) - 2`` and a literal here went stale the moment CP was retired.
ECOG_PATIENTS = ("MM", "VB")


# ── Feature window ────────────────────────────────────────────────────────────
# Set 2026-08-08.  Was 10 bins (1000 ms), typed independently in nine live files
# plus three hard-coded call sites; the 1000 ms results are retired.
#
# A wrong value here does not raise -- it silently rescales every reported latency,
# because the report modules convert bin index to seconds.  Which is why they now
# read ``n_bins_history`` from the run's own ``meta.json`` and use this only as the
# default for a NEW run.

#: Preceding time bins fed to the model as history.  5 bins x 100 ms = 500 ms.
N_BINS_HISTORY = 5

#: Bin width in milliseconds.
BIN_SIZE_MS = 100


# ── Brain-region atlas ────────────────────────────────────────────────────────

#: Which atlas column gates channel selection when a run does not say.  Both arms
#: are supported everywhere; NMM is primary for the paper's non-ROI figures and DK
#: is the peer arm, so a run that does not name an atlas gets the primary one.
#: The vocabulary itself is ``utils.rois``; the colours are ``utils.roi_palette``.
ROI_ATLAS_DEFAULT = "nmm"

#: The atlases a run may be gated by.  ``none`` disables the region filter entirely
#: and exists only to reproduce runs that predate it -- it is not a valid choice for
#: new work, and the pipeline prints a deprecation line when it is used.
ROI_ATLAS_CHOICES = ("nmm", "dk", "none")


# ── Statistics ────────────────────────────────────────────────────────────────

#: The p-value cutoff, repo-wide.  Set 2026-07-27 (was an effective 0.01 in the
#: per-bin permutation test and 0.05 everywhere else -- the repo disagreed with
#: itself).  Everything below is derived from this; do not type a cutoff.
ALPHA = 0.05

#: ALPHA expressed as a null percentile, for the per-bin permutation test in
#: figures_for_paper/semantic_regression (a bin is significant iff the observed
#: mean exceeds this percentile of the shuffled null at that bin).  Float by
#: construction, so captions render "95.0th percentile".
PCTILE = 100.0 * (1.0 - ALPHA)

#: Permutation counts for the retrieval nulls.  ``N_PERM_GRADED`` is lower
#: because the graded-relevance null is far more expensive per draw.
N_PERM = 1000
N_PERM_GRADED = 200

#: Star ladder, strictest first.  Paired with ``p_stars`` below.
STAR_LADDER = ((0.001, "***"), (0.01, "**"), (0.05, "*"))


def p_stars(p, ns: str = "n.s.") -> str:
    """Return the significance-star string for a p-value.

    One implementation, replacing five near-identical inline ladders that used
    three different spellings of the same thresholds (``0.001`` vs ``1e-3``).
    ``None``/NaN returns ``ns`` rather than raising, so a missing test in a
    table renders as not-significant instead of crashing a figure.
    """
    if p is None or p != p:          # None or NaN
        return ns
    for thresh, star in STAR_LADDER:
        if p < thresh:
            return star
    return ns


# ── Figure style ──────────────────────────────────────────────────────────────
# Read by figures_for_paper/paper_common.apply_paper_style().  Figure scripts
# should import these from paper_common, which re-exports them, rather than
# reaching in here directly -- see figures_for_paper/README.md.

#: PNG resolution: single panels vs the combined multi-panel figure.
DPI_PANEL = 200
DPI_COMBINED = 300

#: Type sizes (pt).  Nature-style: small, and uniform across every figure.
FONT_SIZE = 8
AXES_TITLE_SIZE = 11
AXES_LABEL_SIZE = 8
TICK_SIZE = 7
LEGEND_SIZE = 7.5

#: Vector output with editable (not outlined) text, so labels stay selectable
#: and a co-author can fix a typo without regenerating the figure.
VECTOR_FONTTYPE = 42
