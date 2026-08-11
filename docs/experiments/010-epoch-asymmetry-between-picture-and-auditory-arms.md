---
id: 010
kind: decision
title: Should the picture arm stay at 50 epochs while the auditory arm ran 100?
status: answered
analysis: cross_task_cotrain
opened: 2026-07-28
closed: 2026-07-30
runs:
  - 2026-07-28_20-09-58_kernel_pls_balance-none_50boot
  - 2026-07-30_15-23-26_kernel_pls_balance-none_50boot
report:
answer: >
  Migrated. Both arms are 100 epochs. Measured effect on every picture-involving condition
  was +0.000 to +0.006 cat_indep_bal_acc; within_aud moved exactly 0.000. No headline claim
  changes in kind.
---

## Question

Epoch count sets the resolution of the permutation null — p floors at roughly 1/(n+1) — so
pairing a 50-epoch picture arm with a 100-epoch auditory arm left the two arms' nulls
unequally resolved. Keep the asymmetry and document it, or migrate the picture arm?

## What was tried

The effect was measured **with the cohort held fixed**, by running cotrain at seven
participants on each picture run: `NONE_BALANCE_RUN_N7_50EP` against
`NONE_BALANCE_RUN_N7_100EP`. Both are pinned in `utils/config.py`, so this is reproducible.

## Result

Every picture-involving condition moved **+0.000 to +0.006** `cat_indep_bal_acc`.
`within_aud` moved **exactly 0.000**, confirming the auditory arm was untouched. Maximum
per-participant shift **0.045**. Cross-task stays near chance; pooled stays just below
within-task — no headline claim changes in kind.

Also verified: the per-patient common-channel sets are **identical** across the two picture
runs for all seven participants, so `_resolve_to_electrode_names` was unaffected and never
fell back to positional pairing.

## Next

None — answered. Alec chose migration 2026-07-30. `PIC_RUN_DEFAULT` moved `PIC_RUN_50EP` →
`PIC_RUN` in `cross_task_cotrain.py`, `cross_task_regression.py`,
`open_vocab_retrieval/predict_io.py` and
`figures_for_paper/cross_task/compute_cross_task_data.py`. `PIC_RUN_50EP` now has no
consumers but stays named in `utils/config.py`, or the ledger flips it to `unreferenced` and
`AGENTS.md` then authorises pruning it.
