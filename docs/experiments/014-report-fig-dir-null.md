---
id: 014
kind: experiment
title: report/__main__ inferred fig_dir with a forward slash, so it was always None on Windows
status: answered
analysis: report
opened: 2026-08-11
closed: 2026-08-11
runs: 2026-08-09_10-17-27_picture_naming_roi-nmm_h5_kernel_pls_cosine_100ep
report:
answer: >
  Real defect, confirmed on all four argument forms, and fixed. NO published number is
  affected: fig_dir is read only by the CSV fallback in load_patient_from_csv, that
  fallback fires only when a PKL will not load, and on the pinned picture run all 15
  PKLs load -- so it fired for no patient and the bug was inert. Fixed while provably
  inert. Had it fired it would have swapped the empirical shuffled null for theoretical
  chance, a median gap of -0.002 but up to -0.024 (~14% relative) and anti-conservative
  for 2/15 patients on category and 5/15 on word. A SECOND defect is left open by
  design: the 1/6 and 1/60 constants contradict AGENTS.md, which says category chance is
  per participant; correcting them would change numbers, so it needs a decision.
---

## Question

Does `report/__main__.py`'s `run_dir.replace('results/', 'figures/', 1)` silently produce
`fig_dir=None` on Windows, and if so has it changed any reported number?

## What was tried

`report/__main__.py` inferred the figures twin with
`run_dir.replace('results/', 'figures/', 1)` — a **forward** slash. `_resolve_run_dir`
returns an absolute path built by `utils.paths`, so on Windows the needle never matched.
Measured across all four accepted argument forms: **`fig_dir` was None on every one**
except a literal relative forward-slash path that happened to exist as typed.

`fig_dir` is read in exactly one place — `load_patient_from_csv`, the fallback used when
a patient's PKL will not load. There, `results_loader.py:269-270` substitutes
**theoretical chance (1/6 category, 1/60 word) for the empirical shuffled null** when
`fig_dir` is None. So the defect silently swaps the null rather than failing.

**Was any published number affected? No — measured, not assumed.** On the pinned picture
run all 15 PKLs load (largest 0.7 GB), so the fallback fires for **no patient** and
`fig_dir` is never consulted. The bug was inert. It was fixed while provably inert,
which is the safe moment.

Had it fired, the substitution is not negligible. Empirical shuffled null vs the
hard-coded constants, across the 15 patients of the pinned run:

| | median gap | range | direction |
|---|---|---|---|
| category | −0.0022 | −0.0240 .. +0.0066 | theoretical **higher** in 13/15 |
| word | −0.0021 | −0.0061 .. +0.0019 | theoretical **higher** in 10/15 |

Mostly conservative, but not uniformly: 2/15 category and 5/15 word would have been
**anti**-conservative. Worst cases (EM −0.0240, RB −0.0233, VB −0.0230) are ~14 %
relative error on the category null.

**The second defect — the constants themselves — is resolved by deletion.** They
contradicted `AGENTS.md`: category chance is **per participant** (0.143–0.200; the
measured nulls span 0.1427–0.1733), never a flat 1/6, and word chance depends on
vocabulary size, not a flat 1/60.

**Decision, Alec 2026-08-11: always use the empirical null; delete the fallback.** Done:

- `load_patient_from_csv` and `extract_null_from_html` **deleted**, with their exports.
  The PKL — which carries the run's own shuffled null — is now the only source, so the
  hard-coded constants have no code path left to reach.
- `compute_significance` loses `fig_dir` and gains `allow_missing`.
- `report/__main__.py` loses `--fig-dir`, which had no consumer left, and gains
  `--allow-missing-pkl`. **The path fix made earlier the same day is therefore gone
  too** — deleting the consumer retired it.

**Pass condition met: the full 90-row × 21-column significance table for the pinned run
is byte-identical before and after** (md5 `1b95e7de…`), all 15 patients, Bonferroni
n=4500, cat 90/90 and word 88/90 significant either way. Deleting a code path that
never executed moved nothing, as predicted.

**A silent drop was introduced by the deletion and closed deliberately.** Without the
fallback, a failed PKL would have hit `continue` behind one line of stdout — and a
participant vanishing from the cohort changes N, the Bonferroni denominator, and every
corrected p-value. So a missing PKL now **raises**, naming the patient and the reason;
`allow_missing=True` proceeds but records the casualties in `df.attrs['dropped_patients']`.
Verified on a synthetic run: raises naming GHOST, and with the flag returns 12 rows for
2 patients with the drop recorded.

Fixed in passing: the patient scan counted **any** subdirectory, so a run's own `report/`
directory would have become a phantom patient and — under the new rule — a hard error. It
now excludes `utils.audit_runs.DERIVED_DIR_NAMES`, the repo's single list of such names.
This codebase has made the `report/`-is-a-patient mistake once before, in `_dir_stats`.
