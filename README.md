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
│   ├── helper/                     # compute for reports; emits no markup
│   └── render/                     # markup: Document, table(), callout(),
│                                   # assets/report.css. Computes nothing.
│
│                                   # ── Analysis lifecycle ──
│                                   # tests/ -> analysis/ -> figures_for_paper/
│                                   # pilot     promoted     published
│                                   #              +-> _archive/ (didn't pan out)
│                                   # See docs/repo_layout.md
│
├── tests/                          # STAGE 1 — pilot sandbox. Throwaway; nothing
│                                   # outside tests/ may import it. Currently empty.
│
├── analysis/                       # STAGE 2 — promoted; the paper depends on this
│   │                               # (run as `python -m analysis.<topic>.<name>`)
│   │                               # Per-module status: analysis/README.md
│   ├── open_vocab_retrieval/       # library: imported by extendability +
│   │                               # extendability_co_trained +
│   │                               # semantic_regression/compute_within_category_null
│   ├── cross_task/
│   │   ├── cross_task_cotrain.py             # library + regen path
│   │   ├── cross_task_regression.py          # library: peak-bin helpers used by
│   │   │                                     # open_vocab_retrieval/predict_io
│   │   ├── cross_task_region_importance.py   # regen path (ROI: permutation + Jacobian)
│   │   ├── cross_task_prediction_mds.py      # regen path (MDS panel)
│   │   └── cross_task_transfer.py            # supplementary: complete, no figure yet
│   ├── embedding_sweeps/
│   │   └── visual_layer_sweep.py             # feeds language_vs_visual panel f
│   ├── model_diagnostics/
│   │   └── pls_components_sweep.py           # feeds the pls_components figure
│   └── helpers/                              # shared support modules (not CLIs)
│       ├── __init__.py                       # make_pipeline, load_results_pkl
│       ├── _phoneme_semantic_helpers.py      # shared well beyond its namesake suite
│       ├── _cross_patient_helpers.py         # 19 fns used by cross_task_transfer
│       └── visual_layer_sweep_report.py
│
├── _archive/                       # Retired: piloted, no paper figure, not
│                                   # maintained. Reasons in _archive/README.md
│   ├── phoneme_semantic_dissociation/  # Tests 1-4 + A-D (4 never ran)
│   ├── dyso_dissociation/              # superseded by language_vs_visual
│   ├── cross_patient_decoding/         # 4 CLIs (its helper was promoted)
│   ├── model_diagnostics/              # regression_model_comparison, pca_deflation
│   ├── cross_task_reports/             # superseded by cross_task_panels.py
│   └── legacy/                         # the former tests/_archive
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

### 5. Analysis experiments

Promoted analyses live under `analysis/<topic>/` and are run as `python -m analysis.<topic>.<module>` from `main/`.  Common flags: `--patients`, `--epochs`, `--embedding`, `--smoke` (quick sanity check).  Retired experiments are under `_archive/` and are not maintained; new pilots start in `tests/`.

```bash
# === Cross-task (co-training, ROI region importance, prediction MDS) ===
# Co-train one kernel-PLS on pooled picture + auditory trials
python -m analysis.cross_task.cross_task_cotrain --patients AA AZ

# ROI/region attribution: permutation region-knockout + Jacobian
python -m analysis.cross_task.cross_task_region_importance --analysis both

# Semantic-organization MDS of the two separate per-task decoders
python -m analysis.cross_task.cross_task_prediction_mds

# Naive picture<->auditory transfer (the negative control; supplementary)
python -m analysis.cross_task.cross_task_transfer --patients AA AZ

# === Open-vocabulary / zero-shot retrieval ===
# NB: --patient takes ONE patient; call run(...) directly for the full cohort
python -m analysis.open_vocab_retrieval.run --patient AA

# === Model diagnostics ===
# PLS n_components overfitting sweep -> results/pls_components/
python -m analysis.model_diagnostics.pls_components_sweep --patients AA --embedding GloVe --epochs 10

# === Embedding sweeps ===
# Visual model layer sweep (DINOv3 / MoCo layers)
python -m analysis.embedding_sweeps.visual_layer_sweep --patients AA --epochs 10
```

Retired suites (`phoneme_semantic_dissociation`, `dyso_dissociation`,
`cross_patient_decoding`, `regression_model_comparison`,
`pca_and_deflation_retrieval`) are under `_archive/` and are not maintained — see
[`_archive/README.md`](_archive/README.md) for why each was retired.

Results and HTML reports are saved to `results/<analysis>/`, the single output root.  Scripts should obtain it via `utils.paths.results_dir("<analysis>")` rather than composing a path by hand.  Which individual runs are safe to delete is recorded in [`docs/results_index.md`](docs/results_index.md) (regenerate with `python -m utils.audit_runs --write`) — runs marked `PINNED` are named in tracked source and feed paper figures.

There is no unit test suite in this repository (see `docs/agent-context/validation.md`).
Validation is `py_compile` on touched files plus the figure-regeneration and ledger checks
described there.

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

All scripts use `argparse` with **dash-form flags, never underscores** (`--out-dir`, not
`--out_dir`). Underscore variants are accepted as legacy aliases for back-compat, written
as `parser.add_argument('--run-dir', '--run_dir', dest='run_dir', ...)`.

There is no shared parser factory. `utils/cli.py` held one from 2026-07 until it was
**deleted on 2026-08-11**: it was 7.5 KB written to be adopted, never imported once by any
of the ~31 argparse scripts, while `README.md` described it as live. A helper nobody calls
is a claim the codebase does not honour. Reuse here is a per-script migration nobody has
scheduled; until someone does, copy the spelling below.

The canonical flag vocabulary — this table is the convention, and it outlived the module:

| Flag | Meaning |
|---|---|
| `--patients ID [ID ...]` / `--patient ID` | cohort, or a single participant |
| `--task {picture_naming,auditory_naming}` | which speech task |
| `--warp {none,linear}` · `--warp-target-sec FLOAT` | auditory time-warping; pin the target when *extending* a cohort |
| `--align CUE` · `--align-back FLOAT` · `--align-forward FLOAT` | alignment event and window |
| `--bin-size INT` · `--history-bins INT` | temporal bin size (ms) and history length |
| `--roi-atlas {nmm,dk,none}` · `--roi-scope NAME` | the atlas picks the column, the scope picks the regions — independent |
| `--model {kernel_pls,pls,krr,linear_ridge}` · `--closest {l2,cosine}` | estimator and retrieval metric |
| `--embedding NAME [NAME ...]` · `--pls-components INT` · `--pca-components INT` | |
| `--epochs INT` · `--n-splits INT` · `--seed INT` · `--shuffles INT` | |
| `--run-dir PATH` · `--results-dir PATH` · `--out-dir PATH` · `--fig-dir PATH` · `--data-dir PATH` · `--in-dir PATH` | **obtain these via `utils.paths`, never hand-composed** |
| `--why TEXT` · `--supersedes RUN_ID` | recorded into the run's `meta.json` |
| `--smoke` · `--quiet` · `--resume` | |

### Temporal Alignment

Trials can be windowed around any behavioral event (`--align`), with optional linear time-warping of the auditory stimulus segment (`--warp linear`) to normalize across variable stimulus durations.  The resulting time axis is stored in `meta.json` (`actual_back_sec`, `actual_forward_sec`) and reflected in all report figures.

### Cross-Task Regression

The notebook `notebooks/cross_semantic_regression.ipynb` and the module `analysis/cross_task/cross_task_regression.py` extend the cross-decoding paradigm to predict continuous word embeddings instead of discrete categories.

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
