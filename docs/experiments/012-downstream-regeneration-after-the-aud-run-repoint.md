---
id: 012
kind: decision
title: Downstream regeneration after the 2026-07-27 AUD_RUN repoint
status: superseded
analysis: cross_task_cotrain
opened: 2026-07-27
closed: 2026-08-09
runs:
  - 2026-07-13_11-58-22_auditory_naming_warp-linear-group_align-aud_stim_onset_kernel_pls_cosine_100ep
report:
answer: >
  Superseded by the 2026-08 re-run, which replaced every auditory and picture run rather
  than repointing to the 2026-07-13 one. Both pins have moved; the listed sequence no
  longer describes work that needs doing.
---

## Question

On 2026-07-27 `utils/config.py` was repointed so that **every** analysis used the auditory
run `2026-07-13_11-58-22_…_100ep`. Only `figures_for_paper/semantic_regression/` was
regenerated in that change, so the cross-task, extendability and open-vocab figures were
left stale relative to their own defaults — still built from the superseded
`2026-05-07_22-26-06_…50ep` run.

Both runs held the same six participants (AA AZ DR LH RB WBH), so **no N and no caption N
changed**; what changed was the alignment (`warp-linear` → `warp-linear-group`) and the
epoch count (50 → 100). The regeneration was multi-hour to multi-day compute and was left
as Alec's call to schedule.

## What was tried

The listed sequence was never executed as written. It was overtaken by a larger change:
commits `f9b73a6` (ROI whitelist gate, 5-bin history, cohort 15/10) and `75db493` (re-run
every analysis on that cohort) replaced the pins entirely.

## Result

Both pins have moved past the run this entry was about — `PIC_RUN` is now
`2026-08-09_10-17-27_…_roi-nmm_h5_…` and `AUD_RUN` is
`2026-08-09_09-04-16_…_roi-nmm_h5_…`. The 2026-07-13 auditory run is still PINNED in
`utils/config.py` as a superseded reference, which is why it has not been pruned.

## Next

- **Verify rather than assume.** This entry asserts the regeneration was overtaken, on the
  evidence of the two commits and the current pins; it does not assert that every
  downstream figure was rebuilt. Confirm per figure with
  `git diff --stat figures_for_paper/*/source_data/` after a re-run before quoting any
  cross-task, extendability or open-vocab number.
