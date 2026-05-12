# -*- coding: utf-8 -*-
"""
tests/ -- Experiments and analyses for the iEEG decoding pipeline.

The folder is organized by research topic.  Each experiment is a standalone
CLI script with its own argparse parser, runnable as `python -m tests.<topic>.<name>`.

Proper unit tests (pytest-style) live in `main/pytest/`, not here.

──────────────────────────────────────────────────────────────────────────────
Topic folders
──────────────────────────────────────────────────────────────────────────────

phoneme_semantic_dissociation/
    Does phoneme decoding pick up genuine phonological information, or
    merely reflect semantic co-variance in the neural signal?

      Test 1: cross_category_generalization   -- phoneme decoding across categories
      Test 2: semantic_residual_regression    -- phoneme decoding on semantic residuals
      Test 3: partial_rsa                     -- partial RSA controlling for semantics
      Test 4: subspace_angle_analysis         -- angles between phon vs sem PLS subspaces
      Test A: ensemble_retrieval              -- ensemble with mixing weight alpha
      Test B: banded_ridge_encoding           -- banded ridge encoding (sem+phon)
      Test C: commonality_analysis            -- commonality analysis on retrieval variance
      Test D: joint_embedding_pls             -- concatenated (joint) embedding target PLS

dyso_dissociation/
    Geometric dissociation analyses via DySO (utils.dyso).

      semantic_phoneme_dyso     -- semantic vs phoneme geometry
      lexical_visual_dyso       -- lexical-semantic vs visual geometry

model_diagnostics/
    Methodological diagnostics: model selection, tuning, retrieval method.

      regression_model_comparison -- Linear Ridge / KRR / PLS / Kernel PLS
      pls_components_sweep        -- find PLS overfitting knee
      pca_and_deflation_retrieval -- where word info lives in neural feature space

cross_task/
    Cross-task transfer analyses.

      cross_task_regression       -- picture-naming <-> auditory-naming

embedding_sweeps/
    Embedding-model layer/variant sweeps.

      visual_layer_sweep          -- DINOv2 / SimCLR intermediate-layer regression

helpers/
    Shared support modules (not standalone CLIs).

      _phoneme_semantic_helpers   -- data prep used by phoneme_semantic_dissociation/*
      visual_layer_sweep_report   -- HTML report generation for visual_layer_sweep
      __init__                    -- make_pipeline, load_results_pkl (shared scaffold)

_archive/
    Retired experiments (kept for reference, not run).

──────────────────────────────────────────────────────────────────────────────
Invocation pattern
──────────────────────────────────────────────────────────────────────────────

  python -m tests.phoneme_semantic_dissociation.commonality_analysis --patients AA AZ --epochs 20
  python -m tests.dyso_dissociation.semantic_phoneme_dyso --smoke --patient AA
  python -m tests.model_diagnostics.regression_model_comparison --patients AA AZ --epochs 10
  python -m tests.cross_task.cross_task_regression --patients AA AZ
  python -m tests.embedding_sweeps.visual_layer_sweep --patients AA --epochs 10

Results are written to `tests/results/<topic>/<run_id>/` or, for legacy
scripts, `tests/results/<run_id>/`.
"""
