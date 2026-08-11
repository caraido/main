---
id: 004
kind: manuscript
title: The draft contradicts itself about the regression window and the repeat count
status: answered
analysis: semantic_regression
opened: 2026-07-28
closed: 2026-08-11
runs:
report:
answer: >
  Both halves settled 2026-08-11. Repeat count is 100 for every analysis. Window length
  is PER TASK: 500 ms (5 bins) for picture naming, 1000 ms (10 bins) for auditory naming
  -- the draft had asserted 1 s for both. Applied to the draft as tracked changes:
  para 84 (the global 1 s claim), para 119 (a 500-ms sliding window that named no task),
  and para 85 (50 -> 100 splits, which had contradicted para 130's 100).
---

## Question

The draft states two different windows and two different repeat counts:

- overview Methods — current bin + **nine** preceding bins (**1 s**), **50** repeated splits;
- detailed Methods — current + **four** preceding bins (**500 ms**), and later **100** iterations.

At the time this was raised (2026-07-28) the headline results had been generated with 1 s
windows, so the tracked 500-ms sentence described results that did not exist.

## What was tried

- **The 500 ms rerun has since landed.** `utils.config.N_BINS_HISTORY` is now **5**, the
  1000 ms results are retired, and the pinned runs carry `_h5` in their ids. The
  `n_bins_history=10` literals formerly in `models/model.py` and
  `analysis/model_diagnostics/pls_components_sweep.py` are gone — checked 2026-08-11.
- Entry [001](001-history-and-scope-diagnostic.md) is now measuring what that window change
  cost, jointly with the channel gate.

## Result

The *code* contradiction is resolved. The *manuscript* contradiction is not: the draft still
carries both sentences, and neither yet matches the shipped runs.

## Next

**Decision, Alec 2026-08-11, split in two:**

- **Repeat count is settled: 100 iterations, for every analysis.** The draft's "50 repeated
  splits" is wrong wherever it appears. This can be corrected now.
- **Window length is deliberately still open**, and — importantly — **may end up differing
  between picture and auditory naming**. So the Methods must not assert a single window for
  both tasks until that is decided. Entry
  [001](001-history-and-scope-diagnostic.md) is the experiment that will inform it.

That second point changes what "fix the contradiction" means: the fix is not to pick one of
the two sentences, it is to make the window **per task** and leave it unstated until
measured. Writing "500 ms" globally now would be a new error, not a correction — it would
assert parity between the tasks that has not been established.

Read `n_bins_history` and `n_epochs` from each run's `meta.json` rather than from memory.

**Applied 2026-08-11.** Six tracked edits landed in `Semantic decoding paper_Draft.docx`
(paras 84, 119, 85 x2, 125, 130); the last three belong to entry 002's metric-name fix.
The window is now stated per task rather than globally, which is what makes the sentence
true rather than merely consistent.
