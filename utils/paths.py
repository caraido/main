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
