# Figure caption — Selection of the number of PLS components

> **NOT REGENERATED for the 2026-08 re-run — do not ship without re-reading this note.**
>
> Every other figure in this directory was rebuilt on 2026-08-10 against the NMM-gated,
> 5-bin, 15-participant runs. This one was deliberately left alone: the sweep is ~600 model
> fits and was estimated at 4–8 h, which was not worth blocking the re-run for a component
> count that is already settled at 10.
>
> So the numbers below describe the **superseded** analysis: **N=11**, whole-brain channels
> (no temporal-parietal ROI restriction), 1000 ms history. They are not comparable with the
> other figures, and the participant count differs from the N=14 stated elsewhere.
>
> **N=11, not 12, since 2026-08-12**: CP was retired (`docs/experiments/015-retiring-cp.md`)
> and `compute_pls_components_data.py` now skips `RETIRED_PATIENTS` when it globs
> `results/pls_components/pls_lc_*.csv`. The sweep itself was NOT re-run — the per-participant
> CSVs are independent, so dropping one participant is exactly equivalent to recomputing
> without it. The component count is unchanged at 10.
>
> To regenerate, in this order:
> `python -m analysis.model_diagnostics.pls_components_sweep --patients <all 15> --embedding GloVe --epochs 10`,
> then `python figures_for_paper/pls_components/compute_pls_components_data.py
> --expect-patients 15` (the authority for both `source_data/*.csv`), then
> `pls_components_selection.ipynb` for the panels. GloVe alone is sufficient — the figure reads
> no other embedding, so the four-embedding sweep the previous run used costs ~4× for data
> nothing consumes.

Selection of the number of PLS components. Held-out performance metric as a function of the
number of PLS components for picture naming task (kernel PLS regression: Nystroem kernel
followed by PLS regression onto GloVe word-embedding targets). For every component the model
was refit over repeated random train/test splits (70/30). Metrics are shown at the
best-performing time bin. **a** Balanced category accuracy and **b** balanced word accuracy
obtained from retrieval of the predicted embeddings against the true word embeddings. Shading
denotes ± 1 s.e.m. Grey lines show individual participant. **c** Cosine similarity between
predicted and true embeddings, shown for the training set (grey, dashed, square markers) and
the held-out test set (blue, solid, circular markers). Yellow line (triangular markers):
per-participant train-minus-test difference. Thin lines show individual participants for all
three quantities; bold lines are across-participant means. **a, b, c** N=11.
