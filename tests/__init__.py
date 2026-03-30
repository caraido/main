"""
tests — Experiments and reports for the semantic regression pipeline.

── Experiment scripts (run models, write CSVs) ──────────────────────────────

  model_comparison
      Head-to-head comparison of Linear Ridge, KRR, PLS, and Kernel PLS on
      identical train/test splits across patients and embeddings.
      Answers: does the kernel help? Does PLS fix prediction bias?
      Usage: python -m tests.model_comparison --patients AA AZ --epochs 20

  pls_learning_curve
      Sweep n_components (2–40) for PLS, recording train/test R², cosine
      similarity, word accuracy, and category accuracy at each setting.
      Saves per-patient CSVs incrementally; supports --resume.
      Usage: python -m tests.pls_learning_curve \\
                 --patients VB RB AA LH AZ EH EM \\
                 --embedding GloVe FastText Word2Vec DINOv2 SimCLR \\
                 --epochs 20 --closest cosine

  layer_sweep
      Per-layer regression for DINOv2 (13 ViT layers) and SimCLR (5 stages).
      Tests whether intermediate visual representations predict neural HGA
      better than the final pooled layer.
      Usage: python -m tests.layer_sweep --patients AA AZ VB --epochs 10

── Report scripts (read CSVs, generate HTML) ────────────────────────────────

  report_model_selection
      Reads model_comparison_*.csv + pls_learning_curve_*.csv.
      Answers: which model to use, and what n_components is optimal?
      Usage: python -m tests.report_model_selection --results_dir <path>

  report_ncomponents_tradeoff
      Reads pls_learning_curve_*.csv.
      Explains *why* R²/cosine peak at n≈4 while word/cat accuracy keep
      rising — regression overfitting vs retrieval ranking trade-off.
      Usage: python -m tests.report_ncomponents_tradeoff --results_dir <path>
"""
