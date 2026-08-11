---
id: 013
kind: decision
title: The report layer split at the compute/markup seam, and what each generator still owes
status: open
analysis: report
opened: 2026-08-11
closed:
runs:
report:
answer:
---

## Question

Alec: *"I found the html reports are very useful. We should keep this format. But the
code that render html changes a lot (different visualization requirement etc). The
reporting system needs to be able to take frequent changes."*

A visualization change touched up to fifteen files, because every generator carried
its own markup layer. Measured across `report/*.py` + the two under
`analysis/cross_task/`: figure→base64 PNG in **9** copies (+1 dead); a `<style>` block
in **15 of 15**, 17–170 lines, in **3 drifted palettes**; DataFrame→table as 3
incompatible helpers + 9 inline row loops; callouts as 5 roles × 10 class names × 4
palettes; the fold/contents nav in 1 (with its contents list module-level).

## What was tried

No compute — a refactor. Verification was differential, not visual.

**`report/render/`** now holds the markup layer, the counterpart
`report/helper/__init__.py` already documented but never had: `Document` (folds,
contents nav, assembly), `table()` (per-cell class hooks + optional paired CSV),
`callout()` (five roles, one palette), `assets/report.css`.

Ported and **verified**:

- **`fig_to_base64`** — 9 duplicates removed. `vanilla_retrieval_report` output
  **byte-identical**; the donor's `_fig_to_img` identical on a synthetic figure.
- **`cross_task_region_importance_report.py`** (the fold/TOC donor) — 18 figures
  byte-identical, 8 sections, 8 contents entries in order, **172 numbers identical**.
  Only the stylesheet differs, which is the point.
- **`vanilla_retrieval_report.py`** — figures, all 50 numbers, full text identical.
- **`cross_task_transfer_report.py`**, **`cross_task_regression_report.py`** — CSS
  moved to the shared sheet; compile + import verified, **not yet re-run on real
  input**.

Fixed in passing: the donor's contents list was a module-level `_TOC`, so two reports
built in one process appended to the same one. Instance state now; the self-test
asserts two Documents do not share it.

## Result

Not a compute question. Verification is per file above; the unported generators are
unchanged, not broken.

## Next

**Alec's decision needed on the keep/port/archive column.** 12 of 15 generators are
not referenced by `report/__main__.py` — each is its own CLI. Not evidence they are
dead, but it is why nobody noticed the drift.

| Generator | State | Proposed |
|---|---|---|
| `semantic_regression_report.py` | dispatched by `__main__` | **port** — primary |
| `auditory_naming_regression_report.py` | dispatched by `__main__` | **port** — primary |
| `cross_task_region_importance_report.py`, `vanilla_retrieval_report.py` | ported ✔ | keep |
| `cross_task_transfer_report.py`, `cross_task_regression_report.py` | CSS ported ✔ | keep — needs real-input re-run |
| `peak_time_report.py` | own 130-line CSS | port |
| `phoneme_regression_report.py` | own CSS; only generator writing its own CSV | port |
| `phoneme_semantic_separation_report.py` | no callouts at all | port |
| `model_vs_vanilla_report.py`, `pca_deflation_report.py` | 90–170-line CSS, matplotlib→raw SVG | port |
| `model_selection_report.py`, `pls_components_tradeoff_report.py` | **converted to matplotlib ✔** | keep |
| `semantic_phoneme_dyso_report.py` | **archived ✔** | done |

**Decided by Alec 2026-08-11 and done:**

- **`semantic_phoneme_dyso_report.py` archived** to `_archive/dyso_dissociation/`, beside
  the producer it renders. It had been unrunnable since its input was pruned
  2026-08-10. The renderer outliving its producer by two weeks is the lesson: archiving
  an analysis did not archive the report depending on it.
- **Both hand-rolled-SVG reports converted to matplotlib.** ~600 lines of manual axis
  transforms, tick loops and three-`<line>` error-bar caps replaced. Signatures keep
  their pixel `W`/`H` args so no call site changed. Verified by rendering every chart
  function and asserting valid PNG output, including the edge cases the SVG handled
  (negative bars, single x value, empty input). Two defects the conversion exposed and
  fixed: value labels collided with error-bar caps, and axis padding scaled by endpoint
  rather than range, so a small negative bar's label landed on the tick labels.
  **This changes how the charts look** — intended, but it is figure-expression, so
  compare before adopting any figure downstream.

Still open: **two independent from-scratch Plotly embedders** (`auditory_naming` ~15
helpers, `cross_task_regression` a smaller one). Its own job.

The `fig_dir` path defect found while reading this layer is entry [014](014-report-fig-dir-null.md).
