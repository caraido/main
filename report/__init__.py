"""
report — Post-hoc analysis and HTML report generation for semantic regression runs.

This package reads results from a single run folder (as produced by
``semantic_regression.py``) and generates:

  1. Significance testing (Wilcoxon signed-rank vs. shuffled null, Bonferroni-corrected)
  2. Word prediction bias analysis (favorite-word detection, entropy)
  3. Metric dissociation (R² vs. category acc vs. word acc)
  4. Embedding norm analysis (which words are nearest the PCA centroid)
  5. A full HTML report combining all analyses

Usage (from main/):
    python -m report <run_dir>
    python -m report results/semantic_regression/2026-03-27_14-30-00_KRR_l2_50ep

See ``python -m report --help`` for all options.
"""
