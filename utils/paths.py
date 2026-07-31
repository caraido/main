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


def results_dir(analysis: str, *parts: str, create: bool = True) -> Path:
    """Return ``results/<analysis>/<*parts>``, creating it by default.

    ``analysis`` should match the analysis's code folder or its
    ``figures_for_paper/`` folder, so a reader can move between code, results
    and figures by name alone.
    """
    path = RESULTS_ROOT.joinpath(analysis, *parts)
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
    path = FIGURES_DIR.joinpath(analysis, *parts)
    if create:
        os.makedirs(path, exist_ok=True)
    return path
