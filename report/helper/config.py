"""
report.helper.config — Shared constants and helper utilities for the report package.

These constants define the canonical embedding order (used in all plots and
tables) and the semantic/visual grouping used for model-type annotations.
"""

# Canonical embedding order — matches semantic_regression.py EMBEDDING_NAMES.
# First four are text-based semantic models; last two are image-based visual models.
EMBEDDING_NAMES = ['GloVe', 'FastText', 'Word2Vec', 'ConceptNet', 'DINOv2', 'SimCLR']

# Groupings for model-type annotation in significance tables
SEM_MODELS = {'GloVe', 'FastText', 'Word2Vec', 'ConceptNet'}  # text-based semantic embeddings
VIS_MODELS = {'DINOv2', 'SimCLR'}                              # image-based visual embeddings
