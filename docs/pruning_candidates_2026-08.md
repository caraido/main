# Pruning candidates — staged for review, 2026-08-08

> ## EXECUTED 2026-08-10 — 80.30 GB staged out, measured
>
> Approved by Alec: **groups A, B and D** of the 2026-08-10 list (below the correction
> banner). Staged first to `../_pruned_2026-08-10/` — a sibling of `main/` inside the
> OneDrive root, so the move was a same-volume metadata rename and took **1 second**;
> a `cp` would have hydrated ~80 GB of placeholders. Alec then approved a hard delete,
> and the staging directory was removed after a guarded check that its top level held
> only `figures/` and `results/` and exactly the 16 leaf directories that were moved
> into it (710 files, 80.30 GB). **This is now irreversible.** The deletion also
> propagates to the OneDrive cloud copy.
>
> | group | what | results | + figure twins |
> |---|---|---|---|
> | A | `phoneme_regression`, 5 runs | 35.67 GB | 0.75 GB |
> | B | `semantic_regression` 2026-04-03 and 2026-04-08_01-02-28 | 43.71 GB | 0.16 GB |
> | D | `dyso_dissociation/semantic_phoneme`, `VB/semantic_regression` | 0.01 GB | — |
> | | **measured total in staging** | | **80.30 GB** |
>
> Figure twins were included because each group's approved size was quoted with them;
> leaving them would have created 0.9 GB of fresh orphans. The now-empty `results/VB/`
> was removed with `rmdir`, which only succeeds on an empty directory.
>
> Post-move validation, `python -m utils.audit_runs --write`:
>
> - PINNED **23 / 77.05 GB — unchanged**
> - incomplete **19 / 25.97 GB — unchanged**; no new `incomplete` entry
> - per-patient **22**, derived **3** — unchanged
> - unreferenced 13 / 81.64 GB → **4 / 2.22 GB**
> - no new "referenced in code but not present on disk" entry
> - `results/` total 184.69 GB → **105.27 GB**
>
> **Not approved, still on disk:** the two `semantic_vanilla_retrieval` runs (2.2 GB) and
> the two `cross_task_{regression,transfer}` 2026-07-30 runs (7 MB — the paper's negative
> control; `unreferenced` only because that analysis has no figure to pin it).
>
> **Renamed 2026-08-10, not deleted:** `figures/semantic_regression/original_KSS_l2_50ep`
> → `original_KRR_l2_50ep` (176 MB). It was the figure tree of a pinned run under a
> misspelled name; the ledger now reports it `twin:PINNED` where it previously read
> `orphan`. Pre-flight confirmed the target name was free and that the two directories
> held identical 12-participant sets.
>
> **Not approved, still on disk:** the remaining **507.7 MB** of `figures/` orphans
> (11 directories, down from 12 now that the rename resolved one).
>
> §A and §B below are the **2026-08-08** staging table and are superseded by this. Their
> sizes and verdicts were computed against a ledger with two defects (see the correction
> banner immediately following). Do not act on them.

> **CORRECTION 2026-08-10 — one row below is wrong and must not be acted on.**
>
> `original_KRR_l2_50ep` (13.5 GB) is listed in §B as `unreferenced`. **It is referenced**,
> by `notebooks/semantic_regression_retrieval_metrics_comparison.ipynb:12,68`, which reads
> `results_root / "original_KRR_l2_50ep" / "report"`.
>
> The ledger could not see that. `utils/audit_runs.RUN_ID_RE` matched pins only against
> timestamp-shaped names, so a directory whose name carries no date was structurally
> unpinnable and always read `unreferenced`, however much code named it. Fixed 2026-08-10
> by adding a second reference class — whole-token directory literals. Seven directories
> moved `unreferenced` → `PINNED` on the first regeneration, including this one and
> `balance_none`/`balance_downsample`, which a paper pipeline reads
> (`figures_for_paper/cross_task/compute_cross_task_data.py:52`).
>
> Also corrected in the same pass: directories that were never runs no longer read
> `incomplete`. `results/auditory_alignment/{figures,source_data}` and the per-patient
> `cross_task_*/{PAT}/` trees are now `derived` and `per-patient`. Ten directories moved
> out of `incomplete`, which §A already suspected ("an artifact of `audit_runs` counting
> subdirectories in a directory that holds loose files").
>
> **Regenerate and re-read `docs/results_index.md` before using anything in this file.**
> The measured `unreferenced` total is now **37.9 GB**, not the ~51 GB implied below.

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
