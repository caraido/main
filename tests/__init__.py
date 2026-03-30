"""
tests — Model comparison and diagnostic tests for semantic regression.

This package contains scripts for testing:

  model_comparison
      Compare four model variants (Linear Ridge, KRR, PLS, Kernel PLS)
      on identical train/test splits.  Answers: does the kernel help?
      Does PLS fix prediction bias?
      Usage: python -m tests.model_comparison --patients AA AZ --epochs 10

  pls_learning_curve
      Sweep n_components for PLS and Kernel PLS, recording train vs test R²
      at each setting.  Detects overfitting (growing train–test gap) and
      identifies the optimal number of components.
      Usage: python -m tests.pls_learning_curve --patients AA --embedding GloVe

  layer_sweep
      Systematic intermediate-layer regression for DINOv2 (13 ViT layers)
      and SimCLR (5 ResNet-50 stages).  Tests whether mid-level visual
      representations predict neural HGA better than pooled final features.
      Includes paired Wilcoxon tests vs pooled baseline with Bonferroni
      correction, cross-patient consistency analysis, and optional
      multi-layer PCA combination.
      Usage: python -m tests.layer_sweep --patients AA AZ VB --epochs 10

All scripts save CSV results and an HTML visual summary to tests/results/.
"""
