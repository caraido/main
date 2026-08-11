---
id: 008
kind: decision
title: Known-broken code deliberately left alone, and what each one costs
status: open
analysis: semantic_regression
opened: 2026-07-27
closed:
runs:
report:
answer:
---

## Question

Defects found during the 2026-07 reorganisation and deliberately not acted on, each because
fixing it is a scoped job rather than a cleanup. Recorded so none is rediscovered as a
surprise.

**RB's V-shank exclusion never fires at the `semantic_regression.py` stage.** RB's channel
labels there are integers, so `str(cn).startswith('V')` is never true. A latent gap for any
participant with integer labels at that stage. See
[`../agent-context/channel-and-roi-naming.md`](../agent-context/channel-and-roi-naming.md).

**`figures/open_vocab_retrieval/source_data/` is a pilot directory that production reads.**
Three modules consume it — `extendability/compute_extendability_data.py`,
`extendability/extendability_panels.py`, and
`semantic_regression/within_category_null.py`. `figures/` is gitignored and full of genuine
junk, so deleting the wrong folder there silently breaks two paper pipelines, and only on
regeneration. Now recorded in the untracked-inputs table in
[`../repo_layout.md`](../repo_layout.md).

**`notebooks/` and `report/` were left untouched.** 11 of 17 notebooks and 11 of 13 report
generators are superseded, but `report/helper/results_loader.py` is imported by
`figures_for_paper/semantic_regression/semantic_regression_panels.py`, so archiving either
directory is its own scoped job.

## What was tried

No compute. Each was verified by reading importers.

## Result

All three still stand. Two facts recorded here in 2026-07 have since changed and are
corrected in the sources:

- the open-vocab **auditory** CSV now exists (it did not when this was written), and the
  picture CSV is **47 MB**, not 38;
- `figures_for_paper/within_category_null/` was listed here as unable to run. It can: its
  fourth candidate input path exists, and its palette and alpha violations were fixed
  2026-08-10. It has still never been executed.

## Next

- RB V-shank: decide whether to normalise labels at that stage or accept the gap; it changes
  channel counts, so it is a sample-definition change, not a fix.
- `report/` and `notebooks/`: addressed by the report-layer and notebook-policy work.
