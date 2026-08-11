# `significance_test_comparison` — pilot

## The question

The shipped per-bin significance rule compares **one number** — the epoch-mean of the
observed score — against the **distribution of individual shuffled draws**
(`mean(obs) > percentile_{PCTILE}(null)`). A natural-looking alternative compares the two
distributions' **means**, with a standard error that shrinks like `1/sqrt(n_epochs)`.
The two families disagree, sometimes dramatically.

This pilot asks two things, in order:

1. **How far apart are they, on the same data?** Six one-sided `obs > null` rules run on
   the cached obs/null arrays of the pinned picture and auditory runs, broken down per
   participant and per time bin.
2. **Which one is calibrated?** Measured, not asserted — the decisive diagnostic is
   whether a rule's *pre-onset* false-positive rate depends on how many resamples were
   drawn. It must not.

**This pilot does not propose changing the shipped test.**
`docs/agent-context/scientific-integrity.md` records the test family as settled: every
t-test variant was tried and failed the same way — a reliable ~0.01 obs-over-null offset
at baseline passes any t-test at n=100 epochs, putting 30–45 % of pre-onset bins over
threshold. This exists to make that reproducible from the arrays rather than remembered.

## The scripts

Run from `main/`, in the `Speech` conda env. `r2_cache_build` is imported by the other
two and does not usually need running on its own.

| Script | What it does |
|---|---|
| `r2_cache_build.py` | Extracts the R² obs/null arrays (`all_test_score` vs `all_chance`) from the pinned PN/AN run pickles into `r2_cache_{task}_{embedding}.npz` + a JSON sidecar carrying `run_dir`, so the cache self-invalidates when `utils.config.PIC_RUN`/`AUD_RUN` move. Mirrors `semantic_regression_panels.build_cache` key-for-key. R² is here because it is the one *continuous-fit* metric with a stored matched null — `cosine` has none anywhere in the repo (`models/model.py` never scores a shuffled cosine), so no test of any kind can be run on it. |
| `perbin_test_comparison.py` | The six rules — the shipped percentile rule, two permutation rules, and unpaired-t / paired-t / Wilcoxon (the last three Bonferroni-corrected within participant, at their own `--alpha-sample`). Emits an HTML report with per-participant significance rasters, plus three CSVs. Cross-checks its own `pctile` rule against the `significant` column of the shipped `figures_for_paper/semantic_regression/source_data/source_data.csv`. |
| `why_discrepancy.py` | Four diagnostics on the same arrays: **D1** baseline offset on pre-onset bins, **D2** yardstick ratio (SD of null draws ÷ SE of the mean difference, ≈ √E — this *is* the size of the disagreement), **D3** obs↔null correlation across epochs, **D4** the decisive one: re-run both families on random subsets of the 100 epochs and plot the significant-bin rate against `n_epochs`. |

```bash
python -m tests.significance_test_comparison.r2_cache_build
python -m tests.significance_test_comparison.perbin_test_comparison
python -m tests.significance_test_comparison.perbin_test_comparison --alpha-sample 0.001
python -m tests.significance_test_comparison.why_discrepancy
```

Cutoffs come from the CLI, defaulting to `utils.config.ALPHA` / `PCTILE`. There is no
module-level p-value literal anywhere in this package, and there must not be.

## Outputs belong in `results/significance_test_comparison/`

Every path comes from `utils.paths.results_dir("significance_test_comparison")`:

```
results/significance_test_comparison/
  r2_cache_{picture,auditory}_GloVe.npz  (+ .npz.json sidecar)
  perbin_test_comparison.html
  why_discrepancy.html
  source_data/
    perbin_test_comparison.csv            pooled totals
    perbin_test_comparison_bypatient.csv  per participant, with the Bonferroni m
    perbin_test_comparison_perbin.csv     per participant per bin, raw and corrected p
    why_discrepancy_summary.csv
    why_discrepancy_perbin.csv
    why_discrepancy_epoch_sweep.csv
```

**This is the one thing the migration changed.** In `tmp/` these files were written beside
the code — that is the anti-pattern `docs/repo_layout.md` §"The output contract" exists to
remove, and it is why `tmp/` is gone. Do not reintroduce a hand-composed path here.

The pre-migration copies of those outputs were **left in `tmp/`, not moved and not
deleted**: they are stale the moment either pinned run id moves, and deleting anything
needs an explicit decision. Re-running any script above regenerates them in the right
place.

## Inputs it reads

- `figures_for_paper/semantic_regression/panels_cache_{task}_GloVe.npz` — the retrieval
  metrics. Refuses to run if its sidecar `run_dir` disagrees with the pinned run.
- The pinned run pickles under `results/semantic_regression/`, via
  `report.helper.results_loader.load_pkl_raw`, for R² only.
- `figures_for_paper/paper_common.py` for participant display IDs and fixed colours.

Importing *into* a pilot is fine; nothing outside `tests/` may import this package.
