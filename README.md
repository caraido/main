# iEEG Speech Decoding Pipeline

Neural decoding of semantic and phonological representations from intracranial EEG (iEEG/sEEG) high gamma activity (70–150 Hz) during picture naming and auditory naming.  Patients view images or hear words and name them aloud; this pipeline regresses word-level embeddings from the neural response and evaluates retrieval accuracy over time.

**Lab:** Slutzky & Glaser Lab, Northwestern University

## Pipelines

Three main entry-point scripts, all sharing the same data and results layout:

| Script | What it does |
|--------|-------------|
| `semantic_regression.py` | Regresses semantic word embeddings (GloVe, FastText, etc.) from neural HGA; evaluates word & category retrieval accuracy over time. |
| `phoneme_regression.py` | Regresses phoneme embeddings (PWESuite panphon / token-IPA) from neural HGA; identical retrieval evaluation. |
| `semantic_vanilla_retrieval.py` | Baseline: LOO nearest-centroid directly in neural feature space — no regression, no embedding. |

All three pipelines support both **picture naming** and **auditory naming** tasks, and optional **time-warping** and **alignment-cue** windowing.

```
Neural HGA (channels × time bins)
    ↓
Flatten across history window  (n_bins_history = 10 × bin_size)
    ↓
Regression model  →  predicted embedding  →  cosine nearest-neighbor retrieval
    ↓                                              ↓
cosine similarity over time             Word & category balanced accuracy
```

Default model: **Kernel PLS** (Nystroem RBF features → PLS, 10 components, cosine retrieval).

## Project Structure

```
main/
├── semantic_regression.py          # Semantic embedding regression pipeline
├── phoneme_regression.py           # Phoneme embedding regression pipeline
├── semantic_vanilla_retrieval.py   # Vanilla neural-space LOO retrieval (baseline)
├── embeddings.py                   # Embedding loading (GloVe, FastText, DINOv2, …)
├── pursuit_simulation.py           # Pursuit task simulation
│
├── models/
│   └── model.py                    # BasicRegressor, BasicClassifier, BottleneckModel
│
├── utils/
│   ├── utils.py                    # BACKWARD-COMPAT shim re-exporting everything below
│   ├── __init__.py                 # Module index
│   ├── io.py                       # File/dir I/O: save_figure_and_source_data, load_all_data
│   ├── preprocessing.py            # align_data, reformat, reformat_raw, switch_2_*
│   ├── text.py                     # remove_number, get_sentence_tense/subject/person, nlp
│   ├── plotting.py                 # plot_accuracy_plotly, plot_on_channel, get_channel_colors
│   ├── interactive.py              # interactive_3d_scatter_plot, interactive_channel_importance, interactive_confusion_accuracy
│   ├── decoder.py                  # GeneralDecoder class
│   ├── logging.py                  # _header, _section, _progress  (pipeline console)
│   ├── confusion_matrices.py       # _make_cm, _plot_cm_grid, _per_word_stats, _plot_count_vs_metric
│   ├── run_meta.py                 # git_hash, git_dirty, write_meta, find_repo_root
│   ├── patient_data.py             # discover_patients, find_df_path, is_valid_answer, extract_col
│   ├── cli.py                      # common_parser, add_*_flags — shared argparse builders
│   └── dyso.py                     # Dissociation / dysochrony utilities
│
├── report/                         # Standalone post-hoc report scripts
│   ├── __main__.py                 # python -m report <run_dir>  (semantic regression)
│   ├── semantic_regression_report.py
│   ├── phoneme_regression_report.py
│   ├── auditory_naming_regression_report.py
│   ├── vanilla_retrieval_report.py
│   ├── cross_task_regression_report.py
│   ├── model_selection_report.py
│   ├── model_vs_vanilla_report.py
│   ├── pca_deflation_report.py
│   ├── peak_time_report.py
│   ├── pls_components_tradeoff_report.py
│   ├── phoneme_semantic_separation_report.py
│   ├── semantic_phoneme_dyso_report.py
│   └── helper/                     # Shared report utilities
│
├── tests/                          # Analysis modules (run as python -m tests.<name>)
│   ├── regression_model_comparison.py   # Linear Ridge vs KRR vs PLS vs Kernel PLS
│   ├── pls_components_sweep.py          # n_components overfitting diagnostic
│   ├── cross_task_regression.py         # Cross-task (picture ↔ auditory) generalization
│   ├── cross_category_generalization.py # Cross-category hold-out
│   ├── semantic_phoneme_dyso.py         # Semantic vs phoneme dissociation
│   ├── commonality_analysis.py          # Shared variance across embedding spaces
│   ├── partial_rsa.py                   # Partial RSA controlling for confounds
│   ├── banded_ridge_encoding.py         # Banded ridge encoding model
│   ├── ensemble_retrieval.py            # Ensemble of semantic + phoneme retrieval
│   ├── joint_embedding_pls.py           # Joint semantic-phoneme PLS
│   ├── lexical_visual_dyso.py           # Lexical vs visual dissociation
│   ├── subspace_angle_analysis.py       # Principal angles between embedding subspaces
│   ├── pca_and_deflation_retrieval.py   # PCA deflation retrieval diagnostic
│   ├── visual_layer_sweep.py            # Visual model layer sweep (DINOv2 layers)
│   └── results/                         # Test output (HTML reports, CSVs)
│
├── data/                           # Patient data — not tracked in git
│   ├── {PATIENT}/
│   │   ├── {PATIENT}_picture_naming_df.pkl
│   │   ├── {PATIENT}_picture_naming_labels.pkl
│   │   ├── {PATIENT}_picture_naming_features_0.1sbin_align{cue}.pkl
│   │   ├── {PATIENT}_auditory_naming_df.pkl
│   │   ├── {PATIENT}_auditory_naming_labels.pkl
│   │   ├── {PATIENT}_auditory_naming_features_0.1sbin_align{cue}.pkl
│   │   └── {PATIENT}_channels.pkl
│   └── conceptnet-en-19.08.txt.gz
│
├── embeddings/                     # Pre-extracted image embeddings (DINOv2, SimCLR)
│   └── pictureNaming extended all/
│
├── results/
│   ├── semantic_regression/
│   │   └── {run_id}/               # e.g. 2026-04-08_kernel_pls_cosine_50ep
│   │       ├── meta.json
│   │       ├── {PATIENT}/
│   │       │   ├── semantic_regression_results.pkl
│   │       │   ├── per_time_scores.csv
│   │       │   └── top1_decoding_source_data.csv
│   │       └── report/
│   ├── phoneme_regression/
│   │   └── {run_id}/               # e.g. 2026-04-06_kernel_pls_cosine_50ep
│   │       ├── meta.json
│   │       ├── {PATIENT}/
│   │       │   ├── phoneme_regression_results.pkl
│   │       │   ├── per_time_scores.csv
│   │       │   └── top1_decoding_source_data.csv
│   │       └── report/
│   └── semantic_vanilla_retrieval/
│       └── {run_id}/
│           ├── meta.json
│           └── {PATIENT}/
│               └── ...
│
└── figures/                        # Per-run interactive figures (Plotly HTML)
    └── semantic_regression/
        └── {run_id}/
            └── {PATIENT}/
```

### Run ID format

```
{YYYY-MM-DD}_{HH-MM-SS}[_auditory_naming][_warp-{warp}][_align-{cue}]_{model}_{metric}_{N}ep
```

Examples:
- `2026-04-08_01-02-28_kernel_pls_cosine_50ep`
- `2026-05-07_12-45-41_auditory_naming_warp-linear_kernel_pls_cosine_50ep`
- `2026-04-16_16-44-20_kernel_pls_cosine_50ep_voicealign`

### Data file naming

Pre-extracted feature files follow the pattern:
```
{PATIENT}_{task}_features_0.1sbin_align{cue}.pkl
```
where `{cue}` is one of: `trialonset`, `gocueonset`, `voiceonset`, `audstimonset`, `audstimoffset`, `audstimmidpoint`.

## Quick Start

All commands are run from `main/`.

### 1. Semantic regression

```bash
# All patients, 50 epochs, kernel_pls, cosine retrieval (defaults)
python semantic_regression.py

# Specific patients / epochs
python semantic_regression.py --patients AA AZ VB --epochs 20

# Auditory naming with linear time-warp, aligned to auditory stimulus onset
python semantic_regression.py --task auditory_naming --warp linear \
    --align aud_stim_onset --patients AA AZ
```

Key flags:
- `--model {krr,linear_ridge,pls,kernel_pls}` — regression model (default: `kernel_pls`)
- `--closest {l2,cosine}` — retrieval metric (default: `cosine`)
- `--embedding GloVe FastText …` — subset of embeddings to run
- `--bin-size 100` — temporal bin size in ms
- `--align {none,trial_onset,go_cue,voice_onset,voice_offset,aud_stim_onset,aud_stim_offset}` — event to align trials around
- `--align-back` / `--align-forward` — seconds before/after the alignment cue

### 2. Phoneme regression

```bash
# All patients, picture naming (default)
python phoneme_regression.py

# Voice-onset aligned
python phoneme_regression.py --align-voice --voice-back 2.5 --voice-forward 1.5

# Auditory naming with warp, aligned to go cue
python phoneme_regression.py --task auditory_naming --warp linear \
    --align go_cue --patients AA AZ
```

Additional flags over semantic_regression:
- `--n-components 10` — PLS components
- `--align-voice` — shorthand for `--align voice_onset` (legacy flag)
- `--embedding panphon token_ipa` — phoneme embedding type(s)

### 3. Vanilla retrieval (baseline)

```bash
python semantic_vanilla_retrieval.py
python semantic_vanilla_retrieval.py --task auditory_naming --warp linear --shuffles 50
```

### 4. Generate HTML reports

**Semantic regression report** (`python -m report`):
```bash
python -m report 2026-04-08_01-02-28_kernel_pls_cosine_50ep
python -m report latest
python -m report latest --skip-bias --skip-norms
```
Options: `--fig-dir`, `--out-dir`, `--skip-bias`, `--skip-norms`, `--data-dir`.

**Phoneme regression report**:
```bash
python report/phoneme_regression_report.py --run_dir latest
python report/phoneme_regression_report.py --run_dir latest --with-significance
```
Options: `--out-dir`, `--with-significance`, `--max-pkl-mb`.

Both accept a bare run ID, a `results/<pipeline>/` path, or `latest`.  Output goes to `{run_dir}/report/` by default.

### 5. Analysis tests

All tests are run as `python -m tests.<module>` from `main/`.  Common flags: `--patients`, `--epochs`, `--embedding`, `--smoke` (quick sanity check).

```bash
# Regression model comparison (Linear Ridge / KRR / PLS / Kernel PLS)
python -m tests.regression_model_comparison --patients AA AZ --epochs 10

# PLS n_components overfitting sweep
python -m tests.pls_components_sweep --patients AA --embedding GloVe --epochs 10

# Semantic vs phoneme dissociation
python -m tests.semantic_phoneme_dyso --smoke --patient AA

# Cross-task generalization (picture → auditory)
python -m tests.cross_task_regression --patients AA AZ

# Commonality analysis (shared variance across embedding spaces)
python -m tests.commonality_analysis --patients VB --epochs 20

# Partial RSA (controlling for word frequency / phonological confounds)
python -m tests.partial_rsa --patients VB CP AA --epochs 20

# Ensemble retrieval (semantic + phoneme combined)
python -m tests.ensemble_retrieval --patients VB --phon-embs panphon

# Visual layer sweep (DINOv2 layers)
python -m tests.visual_layer_sweep --patients AA --epochs 10

# Banded ridge encoding model
python -m tests.banded_ridge_encoding --patients VB WBH

# Subspace angle analysis
python -m tests.subspace_angle_analysis --patients VB CP AA
```

Results and HTML reports are saved to `tests/results/`.

## Embeddings

### Semantic (used in semantic_regression.py)

| Embedding  | Type     | Dim  | Source                          |
|------------|----------|------|---------------------------------|
| GloVe      | Semantic | 300  | torchtext (6B, 300d)            |
| FastText   | Semantic | 300  | torchtext (simple English wiki) |
| Word2Vec   | Semantic | 300  | gensim (Google News)            |
| ConceptNet | Semantic | 300  | ConceptNet Numberbatch 19.08 (`data/conceptnet-en-19.08.txt.gz`) |
| DINOv2     | Visual   | 768  | facebook/dinov2-base            |
| SimCLR     | Visual   | 2048 | Pre-extracted (`embeddings/`)   |

### Phoneme (used in phoneme_regression.py)

| Embedding  | Dim | Source |
|------------|-----|--------|
| panphon    | 24  | PWESuite — IPA → articulatory feature vectors |
| token_ipa  | varies | PWESuite — IPA character token embedding |

## Key Concepts

### Prediction Bias (Favorite-Word Problem)

Ridge L2 regularization shrinks predicted embeddings toward the origin in PCA space. L2 nearest-neighbor retrieval then consistently selects the word whose embedding has the smallest norm — the "favorite word."  This inflates category accuracy while producing poor word-level accuracy and low prediction entropy.

**Fix:** Kernel PLS jointly learns the projection and regression, avoiding norm shrinkage.  Cosine similarity retrieval is a quicker partial fix.

### Significance Testing

For each (patient, embedding) pair, observed retrieval accuracy at the peak time bin is compared against a permutation null (label-shuffled epochs) using a one-sided Wilcoxon signed-rank test.  P-values are Bonferroni-corrected across all tests (patients × embeddings).  Available via `--with-significance` in the phoneme report or always included in the semantic report.

### CLI Conventions

All scripts use `argparse` with dash-form flags (e.g. `--out-dir`, `--pls-components`, `--run-dir`). Underscore variants (`--out_dir`, `--pls_components`, `--run_dir`) are accepted as legacy aliases for back-compat.

Shared flag builders live in `utils/cli.py`:

```python
from utils.cli import common_parser
parser = common_parser(prog='my_script', description='...',
                       flag_groups=['patient', 'training', 'paths'])
# adds --patients --patient --task --model --closest --epochs --pls-components
#      --pca-components --n-splits --seed --shuffles --run-dir --results-dir
#      --out-dir --fig-dir --data-dir
args = parser.parse_args()
```

Available groups: `patient`, `training`, `paths`, `alignment`, `smoke`, `quiet`.

### Temporal Alignment

Trials can be windowed around any behavioral event (`--align`), with optional linear time-warping of the auditory stimulus segment (`--warp linear`) to normalize across variable stimulus durations.  The resulting time axis is stored in `meta.json` (`actual_back_sec`, `actual_forward_sec`) and reflected in all report figures.

### Cross-Task Regression

The notebook `notebooks/cross_semantic_regression.ipynb` and the test module `tests/cross_task_regression.py` extend the cross-decoding paradigm to predict continuous word embeddings instead of discrete categories.

**Ranked accuracy metrics** on `BasicRegressor` (`models/model.py`):
- `all_ranked_accuracy` — top-1: predicted embedding's closest neighbor is the correct word
- `all_top_k_accuracy` — dict of top-k accuracy for k ∈ {1, 3, 5, 10}, enabled via `compute_ranked_accuracy=True` and `top_k_values=[1,3,5,10]` in `fit()`

**Auditory naming support.** Picture naming and auditory naming differ significantly in stimulus duration (≈0.3-0.8 s vs ≈2-3 s) and semantic processing timing (immediate vs late-in-presentation).  Three time-alignment strategies are available for cross-task work:
- `time_warp` (recommended) — align comparable cognitive stages across tasks by warping the auditory stimulus segment
- `voice_onset_align` — align both tasks to voice onset, allowing comparison of late prep / production-related activity
- `stimulus_onset_align` — naïve alignment, useful as a baseline

Use the `--warp` and `--align` CLI flags on `semantic_regression.py` and `phoneme_regression.py` to select.

*Full dev notes for both features live in `docs/_archive/CROSS_REGRESSION_README.md` and `docs/_archive/CROSS_REGRESSION_AUDITORY_SUPPORT.md`.*

## Dependencies

```
numpy, pandas, scipy, scikit-learn, matplotlib, plotly
dill, nltk, gensim
torch, torchtext, torchvision  (for semantic embeddings)
transformers                    (for DINOv2)
panphon, pwesuite               (for phoneme embeddings)
```

## Patients

The pipeline auto-discovers patients from `data/{PATIENT}/` subdirectories.  Current cohort with data: **AA, AP, AZ, CP, DR, EH, EM, LH, MM, RB, VB, WBH** (iEEG/sEEG coverage of temporal, frontal, and parietal cortex; both picture naming and auditory naming paradigms available for most patients).
