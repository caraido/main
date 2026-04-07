"""
report — Post-hoc analysis and HTML report generation for regression runs.

Package layout
--------------
  __main__.py                    CLI for semantic regression reports
  semantic_regression_report.py  HTML generator for semantic regression runs
  phoneme_regression_report.py   Standalone CLI+HTML for phoneme regression runs
  helper/                        Shared analysis modules (no HTML generation)
    config.py                    Shared constants (embedding names, groupings)
    results_loader.py            Load PKL/CSV results from a run folder
    significance_testing.py      Wilcoxon signed-rank tests (Bonferroni-corrected)
    word_bias_analysis.py        Favorite-word / prediction entropy analysis
    metric_dissociation.py       R² vs. category acc vs. word acc dissociation
    embedding_norms.py           L2 norm analysis for embedding-space bias

Semantic regression usage (from main/):
    python -m report <run_dir>          # full path
    python -m report <run_id>           # auto-resolved to results/semantic_regression/<run_id>
    python -m report latest             # most recently modified run folder

Phoneme regression usage (from main/):
    python report/phoneme_regression_report.py <run_dir>
    python report/phoneme_regression_report.py latest

HTML output filenames include the run_id suffix from meta.json:
    semantic_regression_report_<run_id>.html
    phoneme_regression_report_<run_id>.html

See ``python -m report --help`` for all options.
"""
