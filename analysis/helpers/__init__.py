# -*- coding: utf-8 -*-
"""
analysis.helpers — shared library for the promoted analyses.

Moved here from `tests/helper/` in the 2026-07 reorganisation; `analysis/README.md`
classifies it **library**, so changing an API here breaks callers across
`analysis/cross_task/`, `analysis/embedding_sweeps/`, `analysis/model_diagnostics/`
and `analysis/open_vocab_retrieval/`. Grep for importers before moving anything.

  visual_layer_sweep_report
      HTML report and console summary for the visual model layer sweep.
      Imported by analysis.embedding_sweeps.visual_layer_sweep; not a standalone CLI.
      NB this is a *renderer* living in a package named "helpers" -- it belongs next
      to its only caller, and moving it is queued in the output/report refactor.

  _cross_patient_helpers, _phoneme_semantic_helpers
      Suite-specific helpers that outlived the CLIs they shipped with. Both are still
      imported far beyond their namesake suites.

NB "helpers" is a name this repository is retiring: it means "miscellaneous", which is
why a path root, a model factory and an HTML renderer all ended up here. The layer
vocabulary is `utils` (repo-wide) / `common` (per-tree) / `render` (markup); see
`docs/repo_layout.md`.
"""

import pickle as pk

from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline

from utils.paths import results_dir

from .visual_layer_sweep_report import generate_html_report, print_console_summary

__all__ = ['generate_html_report', 'print_console_summary', 'load_results_pkl']

# ``create=False`` is load-bearing: this runs at IMPORT time, and results_dir() creates by
# default.  A directory brought into existence as an import side-effect shows up in
# docs/results_index.md as an ``incomplete`` run that nobody launched.
SEM_REG_DIR = results_dir("semantic_regression", create=False)


# --- shared test helpers (extracted from duplicate definitions across tests/) ---

def make_pipeline(n_components):
    return Pipeline([
        ('nystroem', Nystroem(kernel='rbf')),
        ('pls', PLSRegression(n_components=n_components, scale=False)),
    ])

def load_results_pkl(run_folder: str, patient: str) -> dict:
    """Unpickle the heavy results pkl. Requires the project's `models` package on PYTHONPATH.

    Memory note: pkl files are 100MB-2.6GB. Run on a machine with enough RAM
    (16GB+ recommended). If you only need peak bins (not the trained models),
    use `load_per_time_scores` instead.
    """
    pkl_path = SEM_REG_DIR / run_folder / patient / "semantic_regression_results.pkl"
    with open(pkl_path, "rb") as f:
        return pk.load(f)
