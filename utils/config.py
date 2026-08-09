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

#: Picture naming, 100 epochs, 12 participants.  The paper's picture-naming run.
PIC_RUN = "2026-06-02_17-25-11_picture_naming_kernel_pls_cosine_100ep"

#: Auditory naming, group-warped, aligned to auditory stimulus onset, 100
#: epochs, 7 participants (AA AZ CP DR LH RB WBH).  Repinned 2026-07-28, when CP
#: entered the auditory cohort.  Under ``--warp-scope group`` the warp target is
#: the median over the pooled trials of EVERY patient in the run, so adding CP
#: moved it 3.500 s -> 3.580 s and re-warped the other six as well: this run
#: supersedes AUD_RUN_N6 rather than extending it, and the two are not mergeable.
#: The slug reads ``warp-stim-group`` where the predecessor reads
#: ``warp-linear-group`` only because ``--warp linear`` became a deprecated alias
#: for ``--warp stim``; the warping behaviour is identical.
AUD_RUN = "2026-07-28_16-59-35_auditory_naming_warp-stim-group_align-aud_stim_onset_kernel_pls_cosine_100ep"

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
#: Repointed 2026-07-30 to the EIGHT-participant run (KAW added); its inputs are
#: PIC_RUN and AUD_RUN, both 100 epochs.
NONE_BALANCE_RUN = "2026-07-30_15-39-14_kernel_pls_balance-none_50boot"

#: Semantic-organization MDS run (cross_task_prediction_mds.py), read by panel a of the
#: cross-task figure.  Pinned 2026-07-30.  Before that this was the ONLY input to a paper
#: figure resolved by "newest matching glob" rather than by a pin: re-running the MDS
#: silently repointed panel a, and — worse — the run the shipped figure depended on read
#: ``unreferenced`` in docs/results_index.md, which AGENTS.md authorises pruning.
MDS_RUN = "2026-07-30_15-43-11_prediction_mds_separate_kfold5_seed42"

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

#: Picture naming, 15 participants.  PV and SE joined 2026-08-06.
PICTURE_PATIENTS = ("AA", "AP", "AZ", "CP", "DR", "EH", "EM", "KAW", "LH", "MM", "PV", "RB", "SE", "VB", "WBH")

#: Auditory naming, 10 participants.  CP joined 2026-07-28, KAW 2026-07-30, PV and SE 2026-08-06.
AUDITORY_PATIENTS = ("AA", "AZ", "CP", "DR", "KAW", "LH", "PV", "RB", "SE", "WBH")

#: Participants with BOTH tasks -- the cross-task cohort.  Equal to
#: ``AUDITORY_PATIENTS`` as a matter of *fact* (every auditory participant also did
#: picture naming), not by definition.  Aliased rather than re-typed so the two
#: cannot drift, but kept as its own name because the two mean different things and
#: a future participant could break the equality.
SHARED_PATIENTS = AUDITORY_PATIENTS

#: The two ECoG participants; the other 13 are sEEG.
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
