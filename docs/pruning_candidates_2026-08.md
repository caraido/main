# Pruning candidates — staged for review, 2026-08-08

**Nothing here has been deleted.** This is the "report and stop" step of the
`results-hygiene` procedure. Each row needs its own explicit yes.

Ledger regenerated with `python -m utils.audit_runs --write` immediately before this table;
statuses below are quoted from `docs/results_index.md`, not remembered. Every candidate was
also grepped for importers across `*.py`, `*.md`, `*.ipynb` — a `PINNED` marking is a grep
result, so an absent pin is necessary but not sufficient.

**Read `docs/results_index.md` before acting on any row; do not act on this file alone.**
It is a snapshot and will go stale the moment a run lands.

## Context

Every run on disk predates three changes that land together in 2026-08: the 13-region
temporal-parietal whitelist, atlas-gated channel selection (`--roi-atlas`), and the 500 ms /
5-bin history window. No existing run can be extended into the new scheme — the channel set
and the feature dimension both change — so all of them become historical the moment the new
runs land. That is an argument for pruning *after* the re-run succeeds, not before: until
then they are the only results that exist.

## A. Patient-scoped directories

The category the request named. Two of the three are genuinely eligible; the headline one is
not, and that is the useful finding.

| Path | Ledger | Size | Verdict |
|---|---|---|---|
| `results/layer_sweep_KAW/` | loose files, 1 file | 39 KB | **DO NOT DELETE YET.** Not cruft — a deliberate shard. `compute_language_vs_visual_data.py:63-64` globs `results/layer_sweep_*/layer_sweep.csv` and concatenates, so this directory is the only route by which KAW reaches panel f. Deleting it now silently returns that panel to N=12 with no error. `run_vision_layer_sweep.py` now merges instead of overwriting (fixed 2026-08-08), so the correct sequence is: sweep all 15 into `results/layer_sweep/`, confirm the row count, then delete this and simplify the glob to a single path. |
| `results/VB/semantic_regression/` | not in the ledger (loose `.pk`, no run structure) | 24 KB | **Eligible.** Three `*_layerwise_r2_curves.pk` from March. No code reference; the only mention is `docs/repo_layout.md:97` calling it "Legacy — superseded by `layer_sweep/`". Superseded by a pipeline that still exists. |
| `results/label_generation/{KAW,PV,SE}/` | `unreferenced` (KAW) / `incomplete` (PV, SE) | 10.2 + 3.8 + 6.9 = **20.9 MB** | **Eligible but I recommend keeping.** No code reads them. They hold `{PAT}_*_channels.pkl`, `{PAT}_*_labels.pkl`, electrode-localisation screenshots and `entry and target.pdf` — i.e. the *provenance* of the ROI atlas, not a regenerable analysis output. 21 MB is a cheap price for being able to answer "where did this contact's label come from". Delete only if you have that provenance elsewhere. |

The `incomplete` status on PV and SE means "0 patients", which is an artifact of
`audit_runs` counting subdirectories in a directory that holds loose files. It is not
evidence of an aborted run.

## B. Large unreferenced runs

Not patient-scoped, but they dominate the 169 GB and the question will come up. **All are
`unreferenced` or `incomplete` in the ledger and none is named by tracked code.**

| Run (under `results/semantic_regression/` unless noted) | Ledger | Size |
|---|---|---|
| `2026-04-08_01-02-28_kernel_pls_cosine_50ep` | incomplete | 28.1 GB |
| `2026-04-06_23-55-15_kernel_pls_cosine_50ep` | unreferenced | 24.4 GB |
| `2026-04-03_12-03-53_kernel_pls_cosine_50ep` | incomplete | 15.6 GB |
| `original_KRR_l2_50ep` | unreferenced | 13.5 GB |
| `2026-07-15_16-45-02_picture_naming_warp-voice-group_kernel_pls_cosine_50ep` | incomplete | 8.1 GB |
| `2026-04-13_22-37-15_kernel_pls_cosine_50ep` | unreferenced | 4.8 GB |
| `2026-04-06_12-34-48_kernel_pls_cosine_50ep` | unreferenced | 4.6 GB |
| `results/phoneme_regression/` (5 runs) | all unreferenced | **~36 GB** |
| `results/semantic_vanilla_retrieval/` (2 runs) | unreferenced | 2.2 GB |

Two cautions on this block specifically:

- **`phoneme_regression`'s 36 GB has no consumer today, but the module was just updated**
  (5-bin window, ROI gate) rather than retired, so it is expected to be re-run. Deleting the
  old output is fine; deleting it *because* nothing reads it would be the wrong reason.
- Several of these are `incomplete` only because a sibling run in the same task bucket has
  more patients. That is a relative classification, not a statement that the run failed.

## C. Explicitly NOT candidates

| Path | Why |
|---|---|
| Anything marked `PINNED` | 11 runs, ~50 GB. Refuse on sight — `utils/config.py` keeps superseded run ids named precisely so they stay pinned. |
| `2026-04-08_17-05-14_…` (13.6 GB), `2026-06-02_17-25-11_…` (18.9 GB) | Read as stale April/June runs; both are PINNED paper-figure defaults. **Never prune by date.** |
| `results/cross_task_cotrain/balance_{none,downsample}/` | Current, 54-column, feeds the cross-task figure. Superseded only once the new atlas-gated passes land. |
| The bare `results/cross_task_{cotrain,regression,transfer}/{PAT}/` dirs | Stale pre-`--balance` output, but structurally referenced (`OUT_ROOT/{PAT}`) and small. Low value, non-zero risk. |

## If any of this is approved

Per the procedure: prefer a same-volume `mv` to a staging directory over `rm` — everything
under `results/` is a OneDrive Files-On-Demand placeholder, and `mv` within the volume is an
instant metadata rename while `cp`/checksum/move-outside-OneDrive hydrates the whole tree.
Afterwards re-run `python -m utils.audit_runs --write`, confirm no new `incomplete` or
"referenced in code but not present on disk" entry appeared, and report the **measured**
reclaimed size.
