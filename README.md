# Semantic Regression — iEEG Speech Decoding Pipeline

Neural decoding of semantic representations from intracranial EEG (iEEG/sEEG) high gamma activity (70–150 Hz) during picture naming.  Patients view images and name them aloud; this pipeline predicts word-level semantic embeddings from the neural response and evaluates retrieval accuracy.

**Lab:** Slutzky & Glaser Lab, Northwestern University

## Pipeline Overview

```
Neural HGA (channels × time bins)
    ↓
Flatten across history window (n_bins_history=10)
    ↓
Regression model  →  predicted embedding  →  L2 nearest-neighbor retrieval
    ↓                                              ↓
R² over time                               Word & category balanced accuracy
```

The default regression model is **Kernel Ridge Regression** (Nystroem RBF → Ridge, α=1.5) with PCA(10) on the target embedding space.  PLS and linear variants are available via `--model` (after applying the PLS patch).

## Project Structure

```
main/
├── semantic_regression.py      # Main batch pipeline (data → regression → figures)
├── PATCH_pls_support.py        # Auto-patcher to add PLS model support
├── embeddings.py               # Embedding loading utilities (GloVe, FastText, etc.)
├── hyperparameter_tuning.py    # Grid search over alpha, PCA components, etc.
│
├── models/
│   └── model.py                # BasicRegressor, BasicClassifier, BottleneckModel
│
├── report/                     # Post-hoc analysis package (run on completed results)
│   ├── __main__.py             # CLI: python -m report <run_dir>
│   ├── config.py               # Shared constants (embedding names, model groups)
│   ├── loader.py               # PKL/CSV/HTML data loading with torch stubs
│   ├── significance.py         # Wilcoxon signed-rank with Bonferroni correction
│   ├── bias.py                 # Prediction collapse / favorite-word detection
│   ├── dissociation.py         # R² vs retrieval accuracy dissociation analysis
│   ├── norms.py                # Embedding norm analysis (bias root cause)
│   └── html_report.py          # Generates standalone HTML report
│
├── tests/                      # Diagnostic tests for model selection
│   ├── model_comparison.py     # 4-model comparison (Linear Ridge, KRR, PLS, Kernel PLS)
│   └── pls_learning_curve.py   # Overfitting diagnostic: n_components sweep
│
├── utils/
│   └── utils.py                # Preprocessing helpers (reformat, plot_accuracy_plotly)
│
├── data/                       # Patient data (not tracked in git)
│   ├── {patient}/
│   │   └── picture_naming_df.pkl
│   └── conceptnet-en-19.08.txt.gz
│
├── results/semantic_regression/ # Run-based output
│   └── {run_id}/
│       ├── meta.json
│       └── {patient}/
│           ├── semantic_regression_results.pkl
│           ├── top1_decoding_source_data.csv
│           └── per_time_scores.csv
│
└── figures/semantic_regression/ # Run-based figures
    └── {run_id}/
        └── {patient}/
            ├── r2_over_time.html
            ├── word_retrieval_balanced_acc.html
            └── ...
```

## Quick Start

### 1. Run the main pipeline

```bash
cd main/

# Default: all patients, 50 epochs, KRR model, L2 retrieval
python semantic_regression.py

# Specific patients, fewer epochs
python semantic_regression.py --patients AA AZ VB --epochs 20

# Use cosine similarity for retrieval
python semantic_regression.py --closest cosine
```

Output is organized by run: `results/semantic_regression/{timestamp}_KRR_l2_50ep/`.

### 2. Generate a report on a completed run

```bash
python -m report results/semantic_regression/2026-03-27_KRR_l2_50ep
```

The report includes significance testing (Wilcoxon + Bonferroni), prediction bias analysis, metric dissociation, and embedding norm analysis.  Output goes to `{run_dir}/report/`.

Options:
- `--skip-bias` — skip the word bias analysis
- `--skip-norms` — skip embedding norm analysis (avoids loading large PKLs)
- `--fig-dir` — override figure directory path
- `--out-dir` — override report output path

### 3. Add PLS model support

```bash
python PATCH_pls_support.py
```

This patches `semantic_regression.py` to accept a `--model` flag:

```bash
python semantic_regression.py --model pls --closest cosine
python semantic_regression.py --model kernel_pls
python semantic_regression.py --model linear_ridge
python semantic_regression.py --model krr          # default, unchanged
```

### 4. Run diagnostic tests

**Model comparison** — test all four models on the same splits:

```bash
python -m tests.model_comparison --patients AA AZ --epochs 10
```

Compares Linear Ridge, KRR, PLS, and Kernel PLS.  Reports R², retrieval accuracy, and prediction entropy (bias metric).  Answers: does the kernel help?  Does PLS fix the favorite-word problem?

**PLS learning curve** — detect overfitting by sweeping `n_components`:

```bash
python -m tests.pls_learning_curve --patients AA --embedding GloVe --epochs 10
python -m tests.pls_learning_curve --patients AA AZ --max-comp 30 --no-kernel
```

Plots train vs test R² as a function of `n_components`.  A growing gap between train and test indicates overfitting.  The optimal `n_components` is where test R² peaks.

Results and HTML reports are saved to `tests/results/`.

## Embeddings

Six embedding spaces are used, spanning semantic and visual modalities:

| Embedding  | Type     | Dim  | Source                          |
|------------|----------|------|---------------------------------|
| GloVe      | Semantic | 300  | torchtext (6B, 300d)            |
| FastText   | Semantic | 300  | torchtext (simple English wiki) |
| Word2Vec   | Semantic | 300  | gensim (Google News)            |
| ConceptNet | Semantic | 300  | ConceptNet Numberbatch 19.08    |
| DINOv2     | Visual   | 768  | facebook/dinov2-base            |
| SimCLR     | Visual   | 2048 | Pre-extracted from images       |

## Key Analysis Concepts

### Prediction Bias (Favorite-Word Problem)

Ridge L2 regularization shrinks predicted embeddings toward the origin in PCA space. L2 nearest-neighbor retrieval then consistently selects the word whose embedding has the smallest norm — the "favorite word."  This inflates category accuracy (if the favorite word's category is common) while producing low prediction entropy and poor word-level accuracy.

**Diagnostic:** Check `pred_entropy_norm` in the bias analysis.  Values near 0 indicate severe collapse; values near 1 indicate uniform predictions.

**Fix:** PLS regression replaces Ridge + PCA.  It jointly learns the projection and regression, avoiding the norm-shrinkage artifact.  Cosine similarity retrieval is a quicker partial fix.

### Significance Testing

For each (patient, embedding) pair, observed retrieval accuracy at the best time bin is compared against a permutation null (label-shuffled epochs) using a one-sided Wilcoxon signed-rank test.  P-values are corrected with global Bonferroni across all tests (patients × embeddings × metrics).

### Metric Dissociation

R² (regression fit) and retrieval accuracy (nearest-neighbor classification) can disagree.  The dissociation analysis identifies cases where R² is high but retrieval is at chance (model fits noise), or where R² is low but retrieval works (coarse category information survives even poor regression).

## Dependencies

```
numpy, pandas, scipy, scikit-learn, matplotlib, plotly
dill, nltk, gensim
torch, torchtext, torchvision  (for embeddings)
transformers                    (for DINOv2)
```

## Patients

The pipeline auto-discovers patients from `data/{patient}/picture_naming_df.pkl`.  Current cohort: AA, AB, AC, AE, AF, AG, AI, AZ, HB, VB, WBH, ZJ (12 patients with iEEG/sEEG coverage of temporal, frontal, and parietal cortex).
