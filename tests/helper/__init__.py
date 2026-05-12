# -*- coding: utf-8 -*-
"""
tests.helper — Internal helpers for the tests package.

  visual_layer_sweep_report
      HTML report and console summary for the visual model layer sweep.
      Imported by tests.visual_layer_sweep; not a standalone CLI.
"""

from .visual_layer_sweep_report import generate_html_report, print_console_summary

__all__ = ['generate_html_report', 'print_console_summary']


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
