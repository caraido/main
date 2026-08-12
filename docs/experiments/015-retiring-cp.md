---
id: 015
kind: decision
title: CP is retired from the analysis, behind a single switch
status: answered
analysis: semantic_regression
opened: 2026-08-12
closed: 2026-08-12
runs: 2026-08-09_10-17-27_picture_naming_roi-nmm_h5_kernel_pls_cosine_100ep
report:
answer: >
  Group consensus 2026-08-12: CP is retired and not reported. Cohort 15/10 -> 14/9. The
  exclusion lives in ONE place, utils.config.RETIRED_PATIENTS, from which both cohort tuples
  derive and which discover_patients filters on; an explicit --patients CP is a hard error.
  data/CP moved to data_archive/CP (6.44 GB, 9 ms). Legacy runs are kept -- a retired
  participant is not a deleted one. The warp target is PINNED at 3.5600 s rather than
  recomputed, which is what allows this without re-fitting a single model. RB stays and is now
  the only old-stimulus-set participant, which must keep being said in Methods.
---

## Question

Retire CP without hard-coding the exclusion in the nine-plus places that choose a cohort, and
without silently invalidating results that were legitimately produced.

## What was tried

**The cohort had three independent sources**, so a config edit alone would have half-worked:

1. the tuples in `utils/config.py`;
2. `utils.patient_data.discover_patients`, a scan of `data/` — the cohort for every run
   launched without `--patients`;
3. `os.listdir(run_dir)` in `semantic_regression_panels._patient_dirs` — the flagship figure,
   which has no `--patients` flag and would have gone on plotting CP from the pinned run.

All three now go through `RETIRED_PATIENTS`. The `_ENROLLED_*` rosters are kept and the
analysis cohorts derived from them, so the record of who was enrolled survives the retirement.

**Two cohort couplings were measured, not assumed** — this is what decided the cost.

*The group warp target.* Under `--warp-scope group` the target is the pooled median over the
run's trials and every participant is stretched to it, so the cohort leaks into each
individual's features. Recomputing without CP moves it **3.5600 s → 3.4960 s (−64 ms)** and
re-warps all nine. **Pinned at 3.5600 s instead** (Alec): each participant's warp then depends
only on their own trials and the constant, so the existing `AUD_RUN` already *is* the
nine-participant run. Reproducing 3.5600 s exactly from `data/_warp_segment_durations.json`
confirmed the formula was being read correctly before relying on it.

*The open-vocabulary gallery.* `run_co_trained_retrieval.py:201` builds the retrieval gallery
from the union of every participant's true words. Measured per arm:

| arm | CP's words | unique to CP | gallery |
|---|---|---|---|
| picture | 48 | **0** | unchanged, union stays 104 |
| auditory | 78 | **43** | **changes**, 258 → 215 |

CP's unique auditory words (`fignewton`, `boardorpreboard`, `earlyday`, …) are the old
stimulus set. So the auditory arm needed re-scoring against a changed gallery; the picture arm
did not. The held-out split is per participant (`:177` unions that one patient's own words with
a per-patient seeded RNG), so the decoder is not cohort-coupled either way.

## Result

**No model was re-fit for the retirement** — not `semantic_regression` (60–185 min per run),
not `phoneme_regression`, not `semantic_vanilla_retrieval`. What was re-run is aggregation,
plus re-scoring where the gallery genuinely moved.

`semantic_regression` regenerated at **N=14 picture / N=9 auditory**; peak/rise stats now
report `n_sig` out of 14 and 9. The auto-generated caption sentence about the two stimulus sets
narrowed to RB alone without being edited by hand, because it reads
`OLD_STIMULUS_SET_PATIENTS` and intersects it with the run's cohort.

`data/CP/` → `data_archive/CP/`: 12 files, 6.44 GB, `os.rename` in **9 ms** (same volume, so
metadata only — no OneDrive hydration), byte counts identical, and invisible to git since
`.gitignore:48` (`data*`) covers both trees.

## Next

Kept deliberately:

- **CP's runs stay on disk** — 60 directories, 3.5 GB, including inside `PIC_RUN`, `AUD_RUN`
  and `AUD_RUN_FIGURE`. A result that was produced was produced. Consumers filter by cohort
  rather than the data being deleted.
- **CP stays in `figures_for_paper/participants.json`** (display id `NUE030`, colour
  `#d62728`). That file is a registry of *identity*, not of cohort: already-published figures
  must keep resolving the id, and the colour must not be silently reassigned.
- **CP stays in `OLD_STIMULUS_SET_PATIENTS`.** The fact remains true of the archived data, and
  consumers intersect it with the run's cohort.

Open, and surfaced rather than fixed here:

- **`NUE030` vs `NUEx030`.** The repo uses `NUE###`; the manuscript uses `NUEx###`. A
  find-and-replace must target the right one in each.
- **Manuscript ¶68 is *about* CP and sits inside a `<w:ins>`**, invisible to python-docx's
  `paragraph.text`. "significantly so in fourteen … in the remaining one, NUEx030, p = 0.11"
  becomes "in all fourteen" — the exception vanishes and the claim *strengthens*. That is a
  rewrite, not a renumber. The `p ≤ 0.022` bound was also CP's.
- **Wilcoxon floors move**: n=15→14 takes the one-sided floor 3.05e-5 → 6.10e-5, so every
  headline `p = 3.1 × 10⁻⁵` changes value. n=10→9 takes 9.77e-4 → 1.95e-3.
- **De-identification leak, tracked**:
  `figures_for_paper/extendability_co_trained/results_section.md:10-11` prints the
  initials↔display-id mapping in plain text; manuscript ¶79 names `RB` directly.
- **Four un-ledgered `scope-tpm` runs** contain CP and are named in no tracked source.

Supersedes [011](011-adding-cp-to-the-auditory-cohort.md), which recorded CP's addition and
its consequences; every one of those consequences is now reversed except the stimulus-set
caveat, which survives attached to RB.
