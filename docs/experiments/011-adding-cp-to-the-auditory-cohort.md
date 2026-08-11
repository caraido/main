---
id: 011
kind: decision
title: What did adding CP to the auditory cohort expose, and what must the Methods now say?
status: answered
analysis: semantic_regression
opened: 2026-07-28
closed: 2026-07-28
runs:
  - 2026-07-28_16-59-35_auditory_naming_warp-stim-group_align-aud_stim_onset_kernel_pls_cosine_100ep
report:
answer: >
  CP is in the analyzed auditory cohort (n=6 -> n=7 at the time; 10 today). Inclusion
  exposed a second stimulus set, per-participant chance, and a warp-target coupling that
  makes the n=7 numbers not cleanly comparable to the n=6 ones.
---

## Question

CP's auditory data was reprocessed (`data/CP/CP_auditory_naming_*.pkl`, 2026-06-18). Should
CP enter the analyzed auditory cohort, and what does that change?

## What was tried

The decoder was re-run over `AA AZ CP DR LH RB WBH`. Methods can state n=7 for both
"performed" and "analyzed" as of that date. *(The cohort has since grown to 10.)*

## Result

Four consequences, all of which must reach the Methods:

- **CP and RB ran an older auditory stimulus set.** All 49 of CP's distinct prompt durations
  also occur in RB's (Jaccard 0.98) versus ≤11 shared with any other participant; median
  prompt **4.64 s** for the pair against **3.34 s** for the other five. The pair's category
  inventory adds `abstract`/`action` and drops `vehicle`. A real covariate, not a labelling
  quirk — `participants.json` tags both "AN (old)".
- **Chance is per participant, 0.143–0.200, not a flat 1/6.** Per-participant category counts
  run 5–7. The old hard-coded `N_CATEGORIES = 6` was already wrong for RB on both tasks. Now
  derived in `figures_for_paper/cross_task/source_data/chance_by_participant.csv`.
- **The n=7 numbers are not cleanly comparable to the n=6 ones.** Under `--warp-scope group`
  the warp target is a pooled median, so adding CP moved it **3.500 s → 3.580 s** and
  re-warped the other six. No matched n=6 control was run (Alec's call) and the decoder is
  unseeded, so per-patient deltas confound CP's effect with run-to-run variance. **Do not
  report those deltas as CP's contribution.**
- **The one-sided Wilcoxon floor moved from 1/64 = 0.0156 to 1/128 = 0.0078.**

## Next

Still to state wherever rank metrics are reported: roughly 30 space-stripped multi-word
auditory labels are OOV in GloVe and their trials are dropped; AA has 52 unique words across
53 auditory trials, so its auditory arm is inherently zero-shot; CP's picture–auditory shared
vocabulary is only 19 words against 30–58 for the others, which limits its cross-task
estimates specifically.
