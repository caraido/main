"""
tests — Experiments and reports for the semantic regression pipeline.

── Experiment scripts (run models, write CSVs) ──────────────────────────────

  regression_model_comparison
      Head-to-head comparison of Linear Ridge, KRR, PLS, and Kernel PLS on
      identical train/test splits across patients and embeddings.
      Answers: does the kernel help? Does PLS fix prediction bias?
      Usage: python -m tests.regression_model_comparison --patients AA AZ --epochs 20

  pls_components_sweep
      Sweep n_components (2–40) for PLS, recording train/test R², cosine
      similarity, word accuracy, and category accuracy at each setting.
      Saves per-patient CSVs incrementally; supports --resume.
      Usage: python -m tests.pls_components_sweep \\
                 --patients VB RB AA LH AZ EH EM \\
                 --embedding GloVe FastText Word2Vec DINOv2 SimCLR \\
                 --epochs 20 --closest cosine

  visual_layer_sweep
      Per-layer regression for DINOv2 (13 ViT layers) and SimCLR (5 stages).
      Tests whether intermediate visual representations predict neural HGA
      better than the final pooled layer.
      Usage: python -m tests.visual_layer_sweep --patients AA AZ VB --epochs 10

── Report scripts (read CSVs, generate HTML) ────────────────────────────────

  model_selection_report
      Reads model_comparison_*.csv + pls_learning_curve_*.csv.
      Answers: which model to use, and what n_components is optimal?
      Usage: python -m tests.model_selection_report --results_dir <path>

  pls_components_tradeoff_report
      Reads pls_learning_curve_*.csv.
      Explains *why* R²/cosine peak at n≈4 while word/cat accuracy keep
      rising — regression overfitting vs retrieval ranking trade-off.
      Usage: python -m tests.pls_components_tradeoff_report --results_dir <path>

── Internal helpers ─────────────────────────────────────────────────────────

  helper.visual_layer_sweep_report
      HTML report generation and console summary for visual_layer_sweep.
      Not a standalone CLI — imported by visual_layer_sweep.
"""
