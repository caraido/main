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
- `figures_for_paper/within_category_null/` was listed here as unable to run. It has now
  been executed for **both** tasks (2026-08-11), and the open-vocab auditory dependency
  named under **Next** is discharged.

## Next

**Closed 2026-08-11.** The analysis lives under `figures_for_paper/semantic_regression/` as
`compute_within_category_null.py` (compute) + `within_category_null_panels.py` (render),
writes `source_data/within_category_null_{topk,group}.csv`, and ships the single-panel
`S5_within_category_null` — picture naming, N=15, Holm over its own three tests
(p=0.010/0.011/0.010). `12_within_category_null` is retired.

**The auditory arm is computed but NOT shipped** (Alec's call: it needs a team discussion
first). Its rows stay in both CSVs with `shipped = False`; `within_category_null_panels.py
--task auditory_naming` renders it as an uncaptioned diagnostic. It is a null result at
**both** decoder configurations — observed retrieval sits below the category-preserving null
at every k under `AUD_RUN_FIGURE` (23-region, 10 bins; excess −0.005/−0.008/−0.017) and under
`AUD_RUN` (13-region, 5 bins; −0.005/−0.017/−0.024), Holm p=1.000 throughout. Open-vocab
retrieval was re-run at `AUD_RUN_FIGURE` on 2026-08-11 to rule the configuration out. Ruled
out as causes: OOV stimulus words taking zero GloVe vectors, and the true word's presence in
its own null pool — both make the excess *more* negative. Unexplained: in 7/10 participants
the decoder ranks the true word below its own category-mates. That is the open question for
the team.

Remaining, unchanged:

- RB V-shank: decide whether to normalise labels at that stage or accept the gap; it changes
  channel counts, so it is a sample-definition change, not a fix.
- `report/` and `notebooks/`: addressed by the report-layer and notebook-policy work.
