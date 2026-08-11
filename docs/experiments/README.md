# `docs/experiments/` — the experiment record

The tracked writing memory: what was asked, what was run, what it showed, what was decided.
One file per **question**, never one file per run.

> **The index table at the foot of this file will be generated** by
> `python -m utils.audit_runs --write`, alongside `docs/results_index.md`. Until that lands,
> the entries below are maintained by hand and the schema here is what a new entry must match.

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

*Generated. Do not hand-edit once `utils.audit_runs` owns this section.*

| id | kind | status | title | analysis | latest run |
|---|---|---|---|---|---|
| — | — | — | *no entries yet* | — | — |
