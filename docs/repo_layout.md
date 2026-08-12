# Repository layout and the analysis lifecycle

## The lifecycle

```
tests/   ->   analysis/   ->   figures_for_paper/
pilot         promoted          published
                  |
                  +-> _archive/     (piloted, did not pan out)
```

| Folder | Stage | Expectation |
|---|---|---|
| [`tests/`](../tests/) | 1 — pilot | Throwaway. Nothing outside `tests/` may import from it. Empty *between* pilots, not permanently — what's currently live is a fast-moving fact, not documented here; check `tests/` itself or `python -m utils.audit_runs --status`. |
| [`analysis/`](../analysis/) | 2 — promoted | Expected to keep working. Per-module status in [`analysis/README.md`](../analysis/README.md). |
| [`figures_for_paper/`](../figures_for_paper/) | 3 — published | Manuscript deliverables. Conventions in [`figures_for_paper/README.md`](../figures_for_paper/README.md). |
| [`_archive/`](../_archive/) | retired | Not maintained, not deleted. Reasons in [`_archive/README.md`](../_archive/README.md). |

The rule that makes this work: **promotion is about who depends on you, not
about age or folder.** Before the 2026-07 reorganization, five figure pipelines
imported from a folder called `tests/`, and two of the modules that looked most
archivable (`cross_task_regression`, `_cross_patient_helpers`) turned out to be
libraries. Grep for importers before moving anything.

## The output contract

Every destination has a name and one sanctioned writer. If you are composing a path
by hand, you are writing to the wrong place.

| Root | Holds | Tracked | Sanctioned writer |
|---|---|---|---|
| `results/<analysis>/<run_id>/` | everything one run produced — arrays, pkls, `meta.json`, its `source_data/*.csv`, its `figures/*.png`, its `report/*.html` | no | `utils.paths.results_dir` |
| `figures/<analysis>/` | exploratory, cross-run, throwaway plots | no | `utils.paths.figures_dir` |
| `figures_for_paper/<analysis>/` | the only tracked deliverable: `*.png`/`*.pdf` + `source_data/*.csv` | **yes** | that analysis's `*_panels.py` |
| `logs/` | raw stdout tee — crash forensics only, never read whole | no | the pipeline's own tee |
| `docs/` | the tracked record: `results_index.md`, `experiments/` | **yes** | `utils.audit_runs`, and a human |
| `data/` | raw acquisition and its caches | partly | **nothing** — read-only |

Two consequences worth stating, because both were violated before this table existed:

- **A run's report and source data belong to the run, not to `figures/`.** A report
  describes one run and should die with it. `figures/` is what its own docstring says:
  "exploratory, per-run figure output… safe to prune."
- **`tmp/` is not on this list.** Everything that accumulated there was pilot-grade
  analysis that got misfiled because starting a pilot felt expensive. Anything worth
  naming is a `tests/<slug>/` pilot; anything not worth naming does not belong in the
  repository at all. **Retired 2026-08-11:** the three scripts became
  `tests/significance_test_comparison/`, and `tests/_template/` now exists so that the
  "starting a pilot is expensive" premise no longer holds. `tmp/` still holds the
  *outputs* those scripts wrote beside themselves — stale the moment either pinned run id
  moves, kept only until someone says to delete them.

Some existing output predates this table and is **grandfathered, not moved** — anything
under `results/` or `figures/` needs the `results-hygiene` procedure, and a copy would
hydrate the OneDrive tree. See "Untracked inputs that tracked figures depend on" below.

### Stage is derived, not stored

`tests/` output and `analysis/` output both land in `results/<analysis>/`, which makes a
pilot indistinguishable from a promoted analysis on disk. The stage is recoverable
without moving anything: it is whichever lifecycle folder owns the analysis name —
`results/<analysis>/` is a **pilot** if and only if `tests/<analysis>/` exists on disk right
now. That is a property to read (`utils.paths.stage_of`), not a directory to create, and not
a fact worth restating here since which pilots exist changes quickly.

## Results

One root, `results/<analysis>/`, keyed by a name matching either the analysis's
code folder or its `figures_for_paper/` folder. Write to it via
`utils.paths.results_dir("<analysis>")` — never a hand-composed path.

Before consolidation there were **three** competing roots plus a relative
fallback that escaped the repository entirely, and `phoneme_semantic_dissociation`
was split across two of them with Tests 1–4 outside the repo and Tests A–D
inside it. That is the failure mode `utils/paths.py` exists to prevent.

*Which* run is authoritative is the companion question, answered by
**`utils/config.py`**: pinned run ids (`PIC_RUN`, `AUD_RUN`, plus the superseded
50-epoch pair, kept named so the audit does not mark them prunable), the
repo-wide p-value cutoff `ALPHA` with `PCTILE` derived from it, permutation
counts, and figure type sizes/DPI. Repointing a pinned run is one edit there.
It is a `.py` under `utils/` on purpose: `utils/audit_runs.py` only scans
`.py`/`.ipynb`/`.md` under `figures_for_paper, analysis, tests, notebooks,
report, utils`, so run ids in a root-level JSON would make every pinned run read
as `unreferenced` — and unreferenced runs are what the pruning plan below
deletes. Converting this file to a data file means editing `audit_runs.SCAN_*`
in the same commit.

| `results/` folder | Produced by | Consumed by |
|---|---|---|
| `semantic_regression/` (19 runs, 131 GB) | `semantic_regression.py` | `figures_for_paper/semantic_regression/`, `language_vs_visual/` |
| `phoneme_regression/` (5 runs, 36 GB) | `phoneme_regression.py` | — |
| `semantic_vanilla_retrieval/` (2 runs) | `semantic_vanilla_retrieval.py` | — |
| `cross_task_cotrain/` | `analysis/cross_task/cross_task_{cotrain,region_importance,prediction_mds}.py` | `figures_for_paper/cross_task/` |
| `cross_task_transfer/` | `analysis/cross_task/cross_task_transfer.py` | — (supplementary) |
| `cross_task_regression/` | `analysis/cross_task/cross_task_regression.py` | — |
| `pls_components/` | `analysis/model_diagnostics/pls_components_sweep.py` | `figures_for_paper/pls_components/` |
| `layer_sweep/` | `analysis/embedding_sweeps/visual_layer_sweep.py` | `figures_for_paper/language_vs_visual/` panel f |
| `dyso_dissociation/`, `phoneme_semantic_dissociation/`, `cross_patient_decoding/`, `model_diagnostics/` | archived suites | — |

Which individual runs are safe to touch is in
[`results_index.md`](results_index.md), regenerated by
`python -m utils.audit_runs --write`. **Runs marked `PINNED` are named in tracked
source and must not be deleted.**

### Two operational constraints

1. **OneDrive Files-On-Demand.** Everything under `results/` and `figures/` is a
   cloud placeholder. A same-volume `mv` is a metadata rename — instant, safe
   even for the 131 GB tree. A `cp`, a checksum, or a move outside the OneDrive
   root forces hydration and downloads the whole thing.
2. **The directory name `results` is load-bearing.** `.gitignore` excludes
   `*results`, so renaming this tree to `runs/` or `output/` would stage 169 GB
   on the next `git add`. It is also why `results_index.md` lives in `docs/`:
   git cannot re-include a file under an excluded directory.

## Proposed pruning of `results/` — not executed

`results/` is 169 GB. Nothing was moved or deleted; this is the plan for a later,
separately-approved pass.

**Never touch** the 6 `PINNED` runs (~50 GB). Two of them —
`2026-04-08_17-05-14` (13.6 GB) and `2026-06-02_17-25-11` (17.6 GB) — look like
stale March/April runs and are in fact hard-coded defaults behind paper figures.
This is precisely why pruning by date is unsafe and must be driven by
`utils.audit_runs`.

Candidates, in ascending order of risk:

1. **Aborted runs** — 6 directories under `figures/semantic_regression/` with no
   participant sub-directories, and `results/.../2026-07-13_10-46-47` with a
   single `meta.json`. Each retains only that `meta.json`, which records the
   command line and git commit of the attempt. Tiny; kept for provenance. Delete
   only if you want the tidiness.
2. **Superseded auditory sweeps** — `2026-05-04`/`05-06`/`05-07` are four
   warp-linear runs within three days, of which only the last is pinned (~8 GB
   recoverable). Same pattern for the four `2026-04-06` phoneme runs.
3. **Legacy** — `results/VB/semantic_regression/*.pk` (superseded by
   `layer_sweep/`) and three stray `feature_importance_*.mp4`.
   **`original_KRR_l2_50ep` was listed here and has been removed from the list:
   it is `PINNED`, not unreferenced** — see below.

**Executed 2026-08-10 — 80.30 GB staged out of the tree.** Groups A, B and D of the
list in `pruning_candidates_2026-08.md` were approved and moved, not deleted, to
`../_pruned_2026-08-10/` (a sibling of `main/`, inside the OneDrive root, so the
move was a same-volume metadata rename — it took 1 second). Restoring is the
reverse `mv`.

| | before | after |
|---|---|---|
| `results/` total | 184.69 GB | **105.27 GB** |
| PINNED | 77.05 GB (23) | 77.05 GB (23) — unchanged |
| unreferenced | 81.64 GB (13) | 2.22 GB (4) |
| incomplete | 25.97 GB (19) | 25.97 GB (19) — unchanged |

The 169 GB figure quoted elsewhere in this repository was already stale before
this; it is now doubly so. Regenerating any of the rest is expensive, so each
further deletion wants its own explicit yes.

**What was kept, deliberately:** the two `semantic_vanilla_retrieval` runs (2.2 GB —
deleting both would make `report/model_vs_vanilla_report.py` permanently unrunnable),
and `cross_task_regression/2026-07-30_21-18-23_GloVe` plus
`cross_task_transfer/2026-07-30_21-19-27_GloVe_50boot` (7 MB). Those last two read
`unreferenced` because the transfer analysis has no figure to pin them — but
`analysis/README.md` calls `cross_task_transfer` the negative control behind the
paper's framing, so deleting it would discard the evidence for a claim the paper
rests on. **An `unreferenced` marking is a grep result, not a judgement about
scientific value.**

### The `original_KRR` / `original_KSS` mismatch — resolved 2026-08-10

Long-standing open question; the answer is **`KRR` is authoritative and the run is
referenced.**

- `results/semantic_regression/original_KRR_l2_50ep/` and
  `figures/semantic_regression/original_KSS_l2_50ep/` hold the **same 12
  participants** (AA AP AZ CP DR EH EM LH MM RB VB WBH). The results side also has
  the `report/` the figures side never had. They are one run under two spellings —
  the ordinary results↔figures twin pair, with the figure directory misspelled.
- `notebooks/semantic_regression_retrieval_metrics_comparison.ipynb` names it
  **`original_KRR_l2_50ep`** and reads
  `results_root / "original_KRR_l2_50ep" / "report"` (lines 12, 68). Its own
  markdown describes the notebook as comparing
  `2026-03-27_12-35-02_KRR_cosine_50ep` against `original_KRR_l2_50ep` — i.e. KRR
  with cosine retrieval versus KRR with L2. `KSS` has no meaning in that scheme.
- `KSS` occurs nowhere in any `.py`, `.ipynb` or `.md` except the four documents
  that described the mismatch, plus the generated orphan row in
  `results_index.md`. No code has ever referenced it.

Two consequences:

- **`original_KRR_l2_50ep` (13.5 GB) must not be deleted.** It read `unreferenced`
  only because `audit_runs` could not pin a directory whose name carries no
  timestamp; that was fixed 2026-08-10 and it now reads `PINNED`. It had been
  staged for deletion in `pruning_candidates_2026-08.md`.
- **RENAMED 2026-08-10.** `figures/semantic_regression/original_KSS_l2_50ep/`
  (176 MB) was the typo'd figure directory of a pinned run — not junk, but that
  run's figure tree. It is now `figures/semantic_regression/original_KRR_l2_50ep/`
  and the ledger reports it as `twin:PINNED` against
  `results/semantic_regression/original_KRR_l2_50ep`, where before it read
  `orphan`. **`KSS` no longer names any directory on disk.** It survives only in
  this section, `pruning_candidates_2026-08.md` and the `results-hygiene` skill —
  i.e. only in the documents that record what it was and how it was resolved,
  which is where it should stay.

## Known open issues

These were found during the reorganization and deliberately **not** acted on,
because each is a scientific or in-flight-work decision rather than a cleanup:

- **`semantic_regression` figures are stale relative to their own code.**
  Re-running `semantic_regression_panels.py` today changes the caption from
  "p < 0.05 / aligned to trial onset" to "p < 0.01 / aligned to auditory stimulus
  onset" and rewrites 8782 lines of source data. Cause:
  `panels_cache_auditory_GloVe.npz` was rebuilt 2026-07-18 11:13, 46 minutes
  after the last commit, and the cache was gitignored so nothing flagged the
  drift. The caches are tracked now, so this cannot recur silently. Decide which
  version the manuscript should cite, then re-render.
- **`within_category_null` — resolved 2026-08-11.** The half-finished
  `figures_for_paper/within_category_null/` split closed by going *backwards*: the
  analysis lives in `figures_for_paper/semantic_regression/` as a compute/render pair,
  `compute_within_category_null.py` + `within_category_null_panels.py`, now run for both
  tasks. It writes `source_data/within_category_null_{topk,group}.csv` and ships as
  `S5_within_category_null.{png,pdf}` (**a** picture naming, **b** auditory naming) plus
  two task standalones. `12_within_category_null.{png,pdf}` and the three earlier S5
  standalones are retired — `12` starred each k from the *median of the per-participant
  permutation p-values*, which is not a group test, and disagreed with S5 on top-1.
- **`figures/open_vocab_retrieval/source_data/` is a pilot directory that
  production reads.** See the table below — it is the largest entry.
- **`notebooks/` was left untouched.** 11 of 17 notebooks are superseded, but
  `report/helper/results_loader.py` is imported by `semantic_regression_panels.py`,
  so archiving is its own scoped job.

## The report layer

An HTML report is split at the compute/markup seam, so a visualization change
touches one stylesheet and a few functions rather than every generator:

| Package | Holds | Rule |
|---|---|---|
| `report/helper/` | **compute**: loaders, significance, bias, norms | emits no markup |
| `report/render/` | **markup**: `Document` (folds + contents nav), `table()`, `callout()`, `assets/report.css` | computes nothing |

Both halves were reconstructed from what the generators had already duplicated:
`fig_to_base64` existed in 9 copies, the fold/contents system in 1 (with its
contents list module-level, so two reports in one process shared it), and the same
five callout roles were spelled ten ways across four palettes. `report.css` carries
an alias block mapping the legacy class names onto the canonical five — delete a row
once no generator emits that class.

Two rules that are easy to violate:

- **`table(df, name=..., out_dir=...)` writes the paired CSV.** Prefer it. Two
  parsers in `report/helper/html_utils.py` exist only to recover numbers by regexing
  saved Plotly HTML, and they can be deleted once the reports feeding them emit
  `source_data` instead. Do not add callers to those parsers.
- **`report/lib/` is gitignored** (`.gitignore` excludes `lib/` at any depth), which
  is why the markup package is `render/`. Check `git check-ignore` before adding a
  directory here.

## Untracked inputs that tracked figures depend on

Paper figures are tracked; some of their inputs are not. `figures/` is gitignored and
full of genuine junk, so deleting the wrong folder there silently breaks a paper
pipeline — and the breakage only appears when a figure is *regenerated*. Nothing in
this table may be pruned without regenerating its producer first.

| Untracked input | Produced by | Read by |
|---|---|---|
| `figures/open_vocab_retrieval/source_data/trial_predictions_picture_naming.csv` (47 MB) and `…_auditory_naming.csv` (8.3 MB) | `analysis/open_vocab_retrieval/run.py` | `figures_for_paper/extendability/compute_extendability_data.py`, `figures_for_paper/extendability/extendability_panels.py`, `figures_for_paper/semantic_regression/compute_within_category_null.py` |
| `figures/language_vs_visual/source_data/cache_null_means_100ep.csv` | `figures_for_paper/language_vs_visual/compute_null_means.py` — **and also by `notebooks/language_vs_visual.ipynb`, which writes the same filename** | `figures_for_paper/language_vs_visual/compute_language_vs_visual_data.py` |

Two notes, both measured rather than remembered:

- **The auditory open-vocab CSV now exists, and the auditory arm now runs.** An earlier
  revision of this file said "only the picture CSV exists — there is no auditory
  counterpart, so the auditory arm of `within_category_null` cannot currently run."
  Both files are present as of the 2026-08-10 re-run, and the auditory arm ships as
  panel **b** of S5 as of 2026-08-11. Do not carry the old caveat forward.
- **`cache_null_means_100ep.csv` has two writers racing on one filename**, one of them a
  notebook, in a gitignored directory, with a paper figure downstream. Route both
  through a single constant before anything else; whether the file should move into
  tracked `figures_for_paper/language_vs_visual/source_data/` — which is what
  `.gitignore`'s own NB says about the other 18 `cache_*.csv` — is a separate decision.

The 47 MB CSV is why the second one is not simply the answer for both: tracking that
file is not on the table.
