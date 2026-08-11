# -*- coding: utf-8 -*-
"""
report.helper — shared COMPUTE for the report package. Emits no HTML documents.

Despite the folder name, nothing here is "miscellaneous": these modules load data,
compute analyses, and provide constants for the report scripts that do the markup.
The markup layer is `report/render/`; keep the two apart. The name `helper` is being
retired repo-wide (see `docs/repo_layout.md`), but renaming this package is deferred:
it is imported by a paper pipeline
(`figures_for_paper/semantic_regression/semantic_regression_panels.py`) and by a
multi-megabyte tracked notebook, so the rename would restage a large blob for a
cosmetic gain.

Modules
-------
config              Shared constants (embedding names, model groupings)
results_loader      Load regression results from PKL and CSV files
significance_testing Wilcoxon signed-rank significance tests (semantic)
word_bias_analysis  Word prediction bias / favorite-word analysis
metric_dissociation Dissociation between R², category acc, word acc
embedding_norms     L2 norm analysis for embedding-space bias
html_utils          The one exception to "no HTML": a matplotlib->base64 embedder,
                    plus a PARSER that recovers numeric arrays out of saved Plotly HTML.
                    It exists only because reports had no data-side artifact, and is
                    scheduled for deletion once every report emits its tables as CSV
                    (see report.render.table's `name=` argument). Do not add callers.
                    Its sibling `extract_null_from_html` was deleted 2026-08-11 with the
                    significance CSV fallback -- that one fed a statistical test, and on
                    failure silently swapped the empirical null for theoretical chance.
"""

from .config import EMBEDDING_NAMES, SEM_MODELS, VIS_MODELS
from .results_loader import (
    load_pkl_raw,
    load_patient_from_pkl,
)
from .significance_testing import compute_significance
from .word_bias_analysis import compute_word_bias
from .metric_dissociation import compute_metric_dissociation
from .embedding_norms import compute_norm_analysis
