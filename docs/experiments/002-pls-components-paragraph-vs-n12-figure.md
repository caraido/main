---
id: 002
kind: manuscript
title: The PLS-components paragraph describes a 7-participant sweep; the figure is N=12
status: open
analysis: pls_components
opened: 2026-07-28
closed:
runs:
report:
answer:
---

## Question

A manuscript-text problem, not a figure problem: the figure, its caption and both
`figures_for_paper/pls_components/source_data/*.csv` agree with each other. The active
draft does not agree with them.

Draft says — seven-participant sweep, category accuracy **0.339** at n=10, metric named
"mean-centered cosine", train−test gaps **0.04 / 0.15 / 0.22 / 0.27** at n=2/10/15/20.

Current N=12 source data say — category accuracy **0.305**, word accuracy **0.050**, plain
predicted-vs-true cosine gaps **0.096 / 0.255 / 0.308 / 0.348**.

The metric name is the substantive part. `test_cosine`/`train_cosine` come directly from
`all_cosine_sim`; mean-centring (`utils/retrieval.mean_center_db`) is applied **only on the
retrieval path**. So "mean-centered cosine" does not describe the quantity plotted.

## What was tried

Nothing to run. The artifacts already exist and were checked against the draft on
2026-07-28.

## Result

Not a compute question. The discrepancy is between the draft and artifacts that are
already correct.

## Next

**Decision, Alec 2026-08-11: wait for N=15.** The paragraph will be updated once the N=15
sweep finishes, not against the N=12 numbers above. His expectation is that the conclusion
will not change much — but that is an expectation, not a result, and the paragraph should
quote whatever N=15 actually produces.

- Blocked on the `pls_components` sweep at N=15. `results/pls_components/` currently holds
  12 `pls_lc_*.csv`; the last attempt to extend it to 15 failed (2026-08-09).
- The metric-name correction is **independent of N** and can land now: the plotted quantity
  is plain predicted-vs-true cosine, not "mean-centered cosine", because mean-centring is
  applied only on the retrieval path.
- Do not reconcile in the other direction: the figure and its source data are the record.

**Metric name corrected 2026-08-11** (the half that was independent of N): the fit and
plotted quantity is plain predicted-vs-true cosine, so "mean-centered cosine" is struck
in paras 85, 125 and 130 as tracked changes. Mean-centring on the RETRIEVAL path is real
and paras 132/136 are deliberately untouched. **Still open: the N=15 sweep** -- this
paragraph keeps its N=12 numbers and "seven participants" until that run exists.
