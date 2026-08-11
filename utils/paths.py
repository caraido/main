# -*- coding: utf-8 -*-
"""Canonical filesystem locations for analysis output.

Every analysis used to hard-code its own output root, and they disagreed.  The
three that accumulated were ``main/results/``, ``main/test_results/`` and
``main/tests/results/`` -- plus a fourth, ``<project>/test_results/``, created
whenever a script that built a *relative* output path was launched from the
project root instead of ``main/``.  One analysis suite
(``phoneme_semantic_dissociation``) ended up split across two of them, half its
results outside the repository entirely.

Import from here instead of composing a root by hand::

    from utils.paths import results_dir
    OUT_ROOT = results_dir("cross_task_cotrain")

The directory name ``results`` is load-bearing and must not be "improved":
``.gitignore`` excludes ``*results``, so renaming this tree to ``runs/`` or
``output/`` would stage 169 GB of pickled model state on the next ``git add``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

#: ``main/`` -- this file lives at ``main/utils/paths.py``.
MAIN_DIR = Path(__file__).resolve().parent.parent

#: The single root for all analysis output, keyed by analysis underneath.
RESULTS_ROOT = MAIN_DIR / "results"

#: Where paper-ready figures and their source data live (one folder per figure).
FIGURES_FOR_PAPER = MAIN_DIR / "figures_for_paper"

#: Exploratory, per-run figure output.  Gitignored and safe to prune, with the
#: exception noted in ``docs/results_index.md``.
FIGURES_DIR = MAIN_DIR / "figures"

#: Raw stdout tees.  Gitignored *as a directory*, which is why no tracked record may
#: live here: git cannot re-include a file whose parent directory is excluded.  The
#: tracked run index lives in ``docs/`` for exactly that reason.
LOGS_ROOT = MAIN_DIR / "logs"

#: Raw acquisition data.  Read-only by repository rule; see ``derived_data_path``
#: for the one counted exception.
DATA_ROOT = MAIN_DIR / "data"

#: Which lifecycle folder owns an analysis name.  Stage is *derived* from this rather
#: than stored, so ``results/auditory_alignment/`` is classified as a pilot today
#: without moving a single byte under ``results/``.
LIFECYCLE_ROOTS = {
    "pilot": MAIN_DIR / "tests",
    "promoted": MAIN_DIR / "analysis",
    "published": MAIN_DIR / "figures_for_paper",
}

#: Directory basenames that ``.gitignore`` excludes AT ANY DEPTH -- the Python-packaging
#: block plus this project's own roots.  A tracked file underneath one of these exists on
#: disk, resolves every path that points at it, and is invisible to git.  That has already
#: happened twice here (``data*`` hid a routed context file; ``*cache*`` hid the 18
#: ``cache_*.csv`` that determine rendered figure output).  Checked, not remembered:
#: ``git check-ignore -v <path>``.
_GITIGNORE_HOSTILE = frozenset({
    "build", "develop-eggs", "dist", "downloads", "eggs", ".eggs", "lib", "lib64",
    "parts", "sdist", "var", "wheels", "venv", "ENV", "env", ".venv",
    "figures", "logs", "embeddings", "supportive_repos", ".vector_cache",
})


def _assert_inside(path: Path) -> Path:
    """Raise if *path* escapes ``main/``.

    The failure this exists to prevent is in this module's own docstring: a script that
    built a *relative* output path and was launched from the project root instead of
    ``main/`` wrote its results outside the repository entirely.  Every accessor below
    routes through here, so that can no longer happen silently.
    """
    resolved = Path(path).resolve()
    if resolved != MAIN_DIR and MAIN_DIR not in resolved.parents:
        raise ValueError(
            f"refusing to return {resolved} -- it is outside {MAIN_DIR}. "
            f"This is the relative-path escape utils/paths.py exists to prevent."
        )
    return path


def _reject_hostile_names(parts) -> None:
    """Raise if any component would be swallowed by ``.gitignore`` at any depth.

    Only meaningful for destinations that are meant to be *tracked*; the untracked roots
    are already excluded wholesale and do not care.
    """
    for part in parts:
        name = str(part)
        if name in _GITIGNORE_HOSTILE:
            raise ValueError(
                f"{name!r} is excluded by .gitignore at any depth, so a tracked file "
                f"under it would be invisible to git. Pick another name and verify with "
                f"`git check-ignore -v`."
            )
        if name.startswith("data"):
            raise ValueError(
                f"{name!r} matches the .gitignore rule `data*`, which matches on basename "
                f"at any depth. A tracked file named this way is invisible to git -- this "
                f"already happened once to docs/agent-context/data-conventions.md."
            )


def stage_of(analysis: str) -> str:
    """``'pilot' | 'promoted' | 'published' | 'unknown'`` for an analysis name.

    Derived from whichever lifecycle folder owns the name, never stored, so it is correct
    retroactively for every analysis already on disk.  ``tests/`` output and ``analysis/``
    output both land in ``results/<analysis>/`` and are otherwise indistinguishable there;
    this is what tells them apart without moving anything.

    A name owned by more than one folder resolves to the most-promoted match, because
    promotion is what the lifecycle tracks.

    Three shapes count as ownership, because results-directory names follow all three:
    a folder (``figures_for_paper/cross_task/``), a top-level module
    (``tests/auditory_alignment/``), or a module nested one level inside a topic folder
    (``analysis/cross_task/cross_task_cotrain.py`` owns ``results/cross_task_cotrain/``).
    Without the third, every analysis named after its module rather than its folder
    reported ``unknown``.
    """
    for stage in ("published", "promoted", "pilot"):
        root = LIFECYCLE_ROOTS[stage]
        if (root / analysis).is_dir() or (root / f"{analysis}.py").is_file():
            return stage
        if root.is_dir() and any((topic / f"{analysis}.py").is_file()
                                 for topic in root.iterdir() if topic.is_dir()):
            return stage
    return "unknown"


def results_dir(analysis: str, *parts: str, create: bool = True) -> Path:
    """Return ``results/<analysis>/<*parts>``, creating it by default.

    ``analysis`` should match the analysis's code folder or its
    ``figures_for_paper/`` folder, so a reader can move between code, results
    and figures by name alone.
    """
    path = _assert_inside(RESULTS_ROOT.joinpath(analysis, *parts))
    if create:
        os.makedirs(path, exist_ok=True)
    return path


#: A run-id directory name: the ``YYYY-MM-DD_HH-MM-SS`` stamp every pipeline prefixes
#: its run folder with.  Kept in sync with ``utils.audit_runs.RUN_ID_RE`` by eye — this
#: one only has to *recognise* a run dir, not validate it.
_RUN_ID_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")


def latest_run_dir(root: Path, pattern: str = "*", fallback_to_root: bool = True) -> Path:
    """Return the most recent timestamped run directory under ``root``.

    Run dirs sort lexically by recency because every pipeline prefixes them with
    ``YYYY-MM-DD_HH-MM-SS``, so "latest" is just the last one.  ``pattern`` narrows the
    candidates (e.g. ``"*_prediction_mds_*"``).

    ``fallback_to_root`` returns ``root`` itself when no run dir matches, which is what
    lets a reader keep working against the *legacy* layout: ``cross_task_regression`` and
    ``cross_task_transfer`` wrote straight to ``<root>/<patient>/`` with no run dir until
    2026-07-30, so those trees still exist and still need to be readable.

    NB this resolves *latest*, not *pinned*.  A figure that ships numbers must pin its
    input in ``utils/config.py`` instead — "whatever ran most recently" is not provenance.
    """
    cands = sorted(p for p in root.glob(pattern)
                   if p.is_dir() and _RUN_ID_PREFIX_RE.match(p.name))
    if cands:
        return cands[-1]
    if fallback_to_root:
        return root
    raise FileNotFoundError(f"No timestamped run directory matching {pattern!r} under {root}")


def figures_dir(analysis: str, *parts: str, create: bool = True) -> Path:
    """Return ``figures/<analysis>/<*parts>``, creating it by default.

    The exploratory-output counterpart to ``results_dir``, with the same keying so
    a run's figures and its results share a name.  It exists for the same reason:
    the training scripts used to build this path *relative to the working
    directory*, which put the output outside the repository whenever one of them
    was launched from the project root instead of ``main/``.
    """
    path = _assert_inside(FIGURES_DIR.joinpath(analysis, *parts))
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def report_path(analysis: str, name: str, run_id: str = None, *,
                suffix: str = "", create: bool = True) -> Path:
    """The one sanctioned destination for a generated HTML report.

    ``run_id`` given -> ``results/<analysis>/<run_id>/report/<name>_<run_id><suffix>.html``
    ``run_id`` None  -> ``results/<analysis>/report/<name><suffix>.html``

    A report describes a run, so it belongs inside that run and should die with it.  The
    ``run_id``-less form is for genuinely cross-run reports, which have no single run to
    live in.

    ``suffix`` is how a re-render over a participant subset gets its own filename instead
    of overwriting the cohort report -- ``suffix="_036_031"``.  That is a real workflow,
    not a hypothetical: "write an html with just 036 and 031" has been asked twice.

    Before this existed, reports landed in three conventions, one of which was a bare
    filename resolved against the *current working directory*.  That is why ``.gitignore``
    carries a ``/*.html`` rule for the repository root.
    """
    if run_id:
        stem = f"{name}_{run_id}{suffix}.html"
        directory = results_dir(analysis, run_id, "report", create=create)
    else:
        stem = f"{name}{suffix}.html"
        directory = results_dir(analysis, "report", create=create)
    return _assert_inside(directory / stem)


def paper_dir(analysis: str, *parts: str, create: bool = True) -> Path:
    """``figures_for_paper/<analysis>/<*parts>`` -- the tracked deliverable root.

    Every figure script already computes this as ``dirname(__file__)``; this accessor
    exists so that anything *outside* the folder can name it without hand-composing, and
    so the gitignore-hostile-name check applies.
    """
    _reject_hostile_names(parts)
    path = _assert_inside(FIGURES_FOR_PAPER.joinpath(analysis, *parts))
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def paper_source_data(analysis: str, *parts: str, create: bool = True) -> Path:
    """``figures_for_paper/<analysis>/source_data/<*parts>`` -- shipped paper source data.

    This is the one destination in the repository that is both generated and *tracked*, so
    it is the one where a gitignored name does real damage: the file would exist on disk,
    satisfy every path that reads it, and be invisible to ``git status`` while drifting out
    of sync with the committed figure.  ``_reject_hostile_names`` is why this is not just
    ``paper_dir(analysis, "source_data", ...)``.
    """
    _reject_hostile_names(parts)
    return paper_dir(analysis, "source_data", *parts, create=create)


def figure_path(analysis: str, name: str, *, ext: str = "png",
                run_id: str = None, create: bool = True) -> Path:
    """An exploratory figure file.

    ``run_id`` given -> inside that run: ``results/<analysis>/<run_id>/figures/<name>.<ext>``
    ``run_id`` None  -> cross-run scratch: ``figures/<analysis>/<name>.<ext>``

    Publication figures do not come through here -- they use ``paper_dir``.
    """
    ext = str(ext).lstrip(".")
    if run_id:
        directory = results_dir(analysis, run_id, "figures", create=create)
    else:
        directory = figures_dir(analysis, create=create)
    return _assert_inside(directory / f"{name}.{ext}")


def log_path(analysis: str, run_id: str, *, legacy_stem: str = None,
             create: bool = True) -> Path:
    """``logs/<analysis>/<run_id>.log``, or the legacy flat name when asked.

    ``legacy_stem`` reproduces the pre-existing flat layout, ``logs/<stem>_<run_id>.log``,
    byte for byte.  The three root pipelines pass it because a refactor must not rename an
    output: 61 existing logs are named that way and are matched by eye against run ids.
    The per-analysis grouping is for the analyses that log nothing at all today, which is
    most of them.
    """
    if legacy_stem:
        directory = LOGS_ROOT
        stem = f"{legacy_stem}_{run_id}.log"
    else:
        directory = LOGS_ROOT / analysis
        stem = f"{run_id}.log"
    if create:
        os.makedirs(directory, exist_ok=True)
    return _assert_inside(directory / stem)


def derived_data_path(name: str, *, create_parent: bool = True) -> Path:
    """A DERIVED file that lives under the otherwise read-only ``data/`` tree.

    ``data/`` is raw acquisition and repository rule makes it read-only, but one cache
    (``_warp_segment_durations.json``) is written there and is depended on by name.  This
    accessor exists so the exception is *countable* -- grep for callers -- and so
    relocating it later is a one-line change here rather than a search across pipelines.

    Do not add callers.  A new derived output belongs under ``results/``.
    """
    if create_parent:
        os.makedirs(DATA_ROOT, exist_ok=True)
    return _assert_inside(DATA_ROOT / name)
