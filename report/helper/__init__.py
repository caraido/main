# -*- coding: utf-8 -*-
"""
report.helper — Shared helper modules for the report package.

These modules load data, compute analyses, and provide constants
used by the HTML-generating report scripts. They do NOT generate
HTML files directly.

Modules
-------
config              Shared constants (embedding names, model groupings)
results_loader      Load regression results from PKL and CSV files
significance_testing Wilcoxon signed-rank significance tests (semantic)
word_bias_analysis  Word prediction bias / favorite-word analysis
metric_dissociation Dissociation between R², category acc, word acc
embedding_norms     L2 norm analysis for embedding-space bias
"""

from .config import EMBEDDING_NAMES, SEM_MODELS, VIS_MODELS
from .results_loader import (
    load_pkl_raw,
    load_patient_from_pkl,
    load_patient_from_csv,
    extract_null_from_html,
)
from .significance_testing import compute_significance
from .word_bias_analysis import compute_word_bias
from .metric_dissociation import compute_metric_dissociation
from .embedding_norms import compute_norm_analysis
