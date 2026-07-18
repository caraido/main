# `_archive/` — retired analyses

Stage 3 of the analysis lifecycle: piloted, did not pan out, kept for reference.

Nothing here is maintained or expected to run. Imports were repointed when
`tests/` was reorganized so the code still reads correctly, but no attempt is
made to keep it working. **Kept on disk rather than deleted** — these are small
text files, and "did you try X?" is a question that comes up long after the
fact, particularly at a committee meeting.

Results were not deleted either; each suite's output now lives under
`results/<analysis>/` alongside everything else.

| Archived | Retired | Why | Results |
|---|---|---|---|
| [`phoneme_semantic_dissociation/`](phoneme_semantic_dissociation/) (8 scripts) | 2026-07 | No paper figure. Never finished: 4 of the 8 scripts (`ensemble_retrieval`, `joint_embedding_pls`, `commonality_analysis`, `banded_ridge_encoding`) produced no output at all, `commonality_{LH,VB}.csv` are 2 bytes (failed runs), and `subspace_angle_analysis` covers 3 of 12 participants. Last substantive edit predates the May 2026 reorg. | `results/phoneme_semantic_dissociation/` |
| [`dyso_dissociation/`](dyso_dissociation/) (2 scripts) | 2026-07 | The lexical-vs-visual contrast it tested now ships as `figures_for_paper/language_vs_visual/` using a different method. Its own panels were cut and survive only as pilot PNGs under `figures/language_vs_visual/`. | `results/dyso_dissociation/` |
| [`cross_patient_decoding/`](cross_patient_decoding/) (4 CLIs) | 2026-07 | No paper figure; cold since 2026-05-21. **Note:** its `_cross_patient_helpers.py` was *not* archived — `cross_task_transfer` imports 19 functions from it, so it was promoted to `analysis/helpers/`. | `results/cross_patient_decoding/` |
| [`model_diagnostics/`](model_diagnostics/) (2 scripts) | 2026-07 | `regression_model_comparison` and `pca_and_deflation_retrieval` never reached the paper and only ever ran for 3 of 12 participants (Mar–Apr 2026). Their folder-mate `pls_components_sweep` was promoted — it feeds a supplementary figure. | `results/model_diagnostics/` |
| [`cross_task_reports/`](cross_task_reports/) (2 scripts) | 2026-07 | HTML-report generators superseded by `figures_for_paper/cross_task/cross_task_panels.py`, which produces the publication figures from the same CSVs. The *compute* scripts they wrapped were promoted to `analysis/cross_task/`. | HTML under `results/cross_task_cotrain/` |
| [`legacy/`](legacy/) (3 scripts) | 2026-05 | The pre-existing `tests/_archive/`, folded in here: `dPCA_differences`, `hyperparameter_tuning`, `hyperparameter_tuning_irregular`. | — |

## Restoring something

`git mv` it back to `analysis/`, repoint its `analysis.*` / `_archive.*` imports,
and add a row to `analysis/README.md` with its status. Check
`docs/results_index.md` first for whether the result runs it expects still exist.
