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
#: epochs, 6 participants (AA AZ DR LH RB WBH).  Pinned 2026-07-27.  Expected to
#: change again once the group alignment is settled -- that is the one edit.
AUD_RUN = "2026-07-13_11-58-22_auditory_naming_warp-linear-group_align-aud_stim_onset_kernel_pls_cosine_100ep"

#: Picture naming, 50 epochs.  Superseded by PIC_RUN, retained: it is the
#: provenance of the currently shipped cross_task / open_vocab / extendability
#: figures, which have not yet been regenerated against the 100-epoch runs.
PIC_RUN_50EP = "2026-04-08_17-05-14_kernel_pls_cosine_50ep"

#: Auditory naming, per-participant warp, 50 epochs, same 6 participants as
#: AUD_RUN.  Superseded, retained for the same reason as PIC_RUN_50EP.
AUD_RUN_50EP = "2026-05-07_22-26-06_auditory_naming_warp-linear_align-aud_stim_onset_kernel_pls_cosine_50ep"

#: Unbalanced-class control run, read by the cross-task ROI importance figure.
NONE_BALANCE_RUN = "2026-06-30_12-54-54_kernel_pls_balance-none_50boot"


def run_dir(run_id: str, analysis: str = "semantic_regression") -> Path:
    """Return the results directory for ``run_id`` without creating it.

    ``create=False`` is deliberate: a typo in a run id should raise where it is
    read, not silently mkdir an empty directory that then shows up in
    ``docs/results_index.md`` as ``incomplete``.
    """
    return results_dir(analysis, run_id, create=False)


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
