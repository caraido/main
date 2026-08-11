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

- Revise the paragraph to the N=12 plain-cosine values above, **unless** Alec identifies a
  different intended metric and names the artifact that produced it.
- Do not reconcile in the other direction: the figure and its source data are the record.
