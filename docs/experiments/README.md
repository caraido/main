# `docs/experiments/` — the experiment record

The tracked writing memory: what was asked, what was run, what it showed, what was decided.
One file per **question**, never one file per run.

> **The index at the foot of this file is generated** by `python -m utils.audit_runs --write`,
> alongside `docs/results_index.md`. Everything else here is hand-written.
>
> `--status` is the companion view: what is running right now and what finished recently,
> read from one `meta.json` per run and never sizing directories, so it stays usable while
> a run is still going.

## Why this is not a log

A single growing log is unreadable — by a person and, more sharply, by an agent that has to load
it to answer "have we tried this?". So the record is tiered, and each tier answers exactly one
question:

| Tier | Where | Written by | Tracked | Answers |
|---|---|---|---|---|
| raw stdout | `logs/` | the pipeline's tee | no | "why did it crash" |
| run manifest | `results/<analysis>/<run_id>/meta.json` | the pipeline | no | "what exactly was run" |
| run index | [`results_index.md`](../results_index.md) | **generated** by `utils.audit_runs` | yes | "is it done, where is it, is it safe to delete" |
| experiment record | this directory | a human | yes | "what did we try, and what did it show" |

The raw log is allowed to be huge and uncategorised precisely because nothing reads it whole.
**This directory is not.** An entry over 120 lines is a failure, not a thorough entry — if a
question needs more, it is more than one question.

A tracked record cannot live under `logs/`: `.gitignore` excludes that directory, and git cannot
re-include a file whose parent directory is excluded. That is the same constraint that put
[`results_index.md`](../results_index.md) in `docs/`.

## Schema

Filename: `<NNN>-<kebab-slug>.md`, `NNN` zero-padded and never reused.

```markdown
---
id: 007
kind: experiment          # experiment | decision | manuscript
title: Does auditory semantic info lock to stimulus onset or to the go cue?
status: open              # open | answered | abandoned | superseded
analysis: auditory_alignment
opened: 2026-08-04
closed:
runs:
  - 2026-08-10_12-00-00_auditory_naming_warp-none_align-aud_stim_onset_roi-nmm_h5
report: results/auditory_alignment/auditory_alignment_report.html
answer:
---

## Question
One paragraph. Why this is worth compute.

## What was tried
One line per run, newest first, each naming its run id.

## Result
Two sentences maximum. Every number cites the run id it came from.

## Next
One bullet, or "none — answered".
```

**`kind`** — `experiment` means compute was launched. `decision` is a methodological choice
waiting on a person. `manuscript` is a claim in the draft that disagrees with an artifact.
The three are kept in one directory because they cross-reference constantly; the field is what
lets a reader filter.

**`runs:`** — one run id per line, always. `utils.audit_runs` matches the literal string against
the directory name, so an id split or wrapped across lines is an id it cannot see.

## Rules

1. **One entry per question, not per run.** Runs are already recorded, automatically, in their
   own `meta.json`. This directory exists for the part a machine cannot write.
2. **Open it when compute is launched.** Before that, an undecided question is task state and
   belongs in the open-questions file; see the routing table in
   [`../agent-context/README.md`](../agent-context/README.md).
3. **Every number cites a run id.** A number with no run behind it does not go in.
4. **Report the negative result.** An `abandoned` entry saying "this did not work, here is the
   run that shows it" is the most valuable thing in this directory, because it is the one thing
   nobody reconstructs later.
5. **Do not delete an entry.** Set `status: superseded` and name the entry that replaced it.
6. **An entry does not pin a run.** `utils.audit_runs` deliberately does not scan `docs/`
   for pins — this file lists run ids, and scanning it would mark every run in the repository
   as pinned. A run named only here is still prunable, so the `results-hygiene` procedure
   updates the entry to record the deletion *before* deleting.

## Index

**Generated — do not hand-edit between the markers.** Refresh with
`python -m utils.audit_runs --write`, which rewrites only the block below; everything
else in this file is hand-written and preserved.

<!-- BEGIN GENERATED INDEX -- python -m utils.audit_runs --write -->

| id | kind | status | title | analysis | runs cited |
|---|---|---|---|---|---|
| [001](001-history-and-scope-diagnostic.md) | experiment | answered | Did the 2026-08 re-run lose ~5% to a narrower channel gate or to a shorter history window? | `semantic_regression` | 6 |
| [002](002-pls-components-paragraph-vs-n12-figure.md) | manuscript | open | The PLS-components paragraph describes a 7-participant sweep; the figure is N=12 | `pls_components` | 0 |
| [003](003-cross-task-paragraph-overclaims.md) | manuscript | answered | The draft's cross-task paragraph quotes retired numbers and overclaims an amodal code | `cross_task_cotrain` | 0 |
| [004](004-regression-window-and-repeat-count.md) | manuscript | answered | The draft contradicts itself about the regression window and the repeat count | `semantic_regression` | 0 |
| [005](005-svm-comparison-not-yet-quotable.md) | manuscript | open | Joon's fixed-class SVM comparison is claimed in Results but has no number behind it | `semantic_regression` | 0 |
| [006](006-method-descriptions-that-do-not-match-the-artifacts.md) | manuscript | open | Four method/display descriptions in the draft do not match the artifacts that produced them | `open_vocab_retrieval` | 0 |
| [007](007-manuscript-fields-awaiting-content.md) | manuscript | open | Empty manuscript fields that invite an unsupported claim if filled from memory | `semantic_regression` | 0 |
| [008](008-known-broken-code-left-alone.md) | decision | open | Known-broken code deliberately left alone, and what each one costs | `semantic_regression` | 0 |
| [009](009-tracked-doc-corrections-not-applied.md) | decision | answered | Tracked documentation that describes things which no longer exist | `semantic_regression` | 0 |
| [010](010-epoch-asymmetry-between-picture-and-auditory-arms.md) | decision | answered | Should the picture arm stay at 50 epochs while the auditory arm ran 100? | `cross_task_cotrain` | 2 |
| [011](011-adding-cp-to-the-auditory-cohort.md) | decision | superseded | What did adding CP to the auditory cohort expose, and what must the Methods now say? | `semantic_regression` | 1 |
| [012](012-downstream-regeneration-after-the-aud-run-repoint.md) | decision | superseded | Downstream regeneration after the 2026-07-27 AUD_RUN repoint | `cross_task_cotrain` | 5 |
| [013](013-report-render-layer.md) | decision | open | The report layer split at the compute/markup seam, and what each generator still owes | `report` | 0 |
| [014](014-report-fig-dir-null.md) | experiment | answered | report/__main__ inferred fig_dir with a forward slash, so it was always None on Windows | `report` | 1 |
| [015](015-retiring-cp.md) | decision | answered | CP is retired from the analysis, behind a single switch | `semantic_regression` | 1 |
| [016](016-cross-task-figure-at-tpm-h10-downsample.md) | decision | answered | The cross-task figure moves to tpm/h10 with class balancing, and the retention claim weakens | `cross_task` | 5 |
| [017](017-roi-report-four-page-convention.md) | decision | answered | The ROI importance report emits four pages per arm, and accuracy is de-trended not divided | `cross_task` | 1 |

<!-- END GENERATED INDEX -->
