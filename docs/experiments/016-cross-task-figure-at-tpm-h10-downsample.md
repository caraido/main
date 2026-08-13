---
id: 016
kind: decision
title: The cross-task figure moves to tpm/h10 with class balancing, and the retention claim weakens
status: answered
analysis: cross_task
opened: 2026-08-13
closed: 2026-08-13
runs: >
  2026-08-12_18-17-20_kernel_pls_balance-downsample_50boot,
  2026-08-13_00-22-11_prediction_mds_separate_kfold5_seed42,
  2026-08-11_23-42-55_picture_naming_roi-nmm_scope-tpm_h10_kernel_pls_cosine_100ep,
  2026-08-12_09-14-11_auditory_naming_warp-stim-group_align-aud_stim_onset_roi-nmm_scope-tpm_h10_kernel_pls_cosine_100ep
report: results/cross_task_cotrain/scope-tpm_h10/balance_downsample/region_importance_report.html
answer: >
  Alec, 2026-08-13: figures_for_paper/cross_task is promoted to the tpm/h10 pair with
  balance=downsample on all four panels. The three inputs are pinned as CROSS_TASK_FIGURE_
  constants in utils/config.py. The previously shipped tp/h5 + balance=none arm stays pinned
  as its predecessor.
---

## What changed

All four panels now come from one configuration:

| Panel | Input | Pin |
|---|---|---|
| a (MDS) | `2026-08-13_00-22-11_prediction_mds_separate_kfold5_seed42` | `CROSS_TASK_FIGURE_MDS_RUN` |
| b, S7 | `2026-08-12_18-17-20_kernel_pls_balance-downsample_50boot` | `CROSS_TASK_FIGURE_COTRAIN_RUN` |
| c, S3 | `scope-tpm_h10/balance_downsample` | `CROSS_TASK_FIGURE_ROI_DIR` |

Panel a's MDS run was created for this promotion; no `tpm`/h10 MDS existed.

## The finding: the retention claim was an imbalance artifact

Picture retention falls **0.987 → 0.818** and pooled-vs-within becomes significant in both
tasks (picture p = 0.0039, auditory p = 0.012; both were p = 0.098, n.s.).

Three factors moved together, so they were separated by holding two fixed at a time over
the same nine participants, using the `balance=none` cotrain run at the *same* `tpm`/h10
pair (`2026-08-12_18-09-39`) as the middle term:

| | ret_pic | ret_aud |
|---|---|---|
| `tp`/h5 `none` (was shipped) | 0.987 | 0.940 |
| `tpm`/h10 `none` | 0.989 | 0.863 |
| `tpm`/h10 `downsample` (now) | 0.818 | 0.919 |

Scope + history at fixed balance: **+0.002** picture. Balance at fixed scope + history:
**−0.171** picture. Within-task ceilings are unchanged (0.2926 → 0.2932), so the pooled
decoder moved, not its baseline. Auditory goes the other way (+0.056) under balancing.

**Conclusion:** the ~99 % picture retention reported before was substantially an artifact of
picture trials outnumbering auditory ~3:1 in the pooled training set. `DOWNSAMPLE_BALANCE_RUN`
was created on 2026-08-09 precisely to test this, and the previous result did not survive it.
Co-training still clears cross-decoding (0.19) and per-participant chance (0.164 / 0.170).

Picture pooled-vs-within has now read n.s. → significant → n.s. → significant across four
analyses. The first three flips are cohort size; this one is the control. Report it as
established *under class balancing*, with the unbalanced value stated alongside.

## Other numbers that moved

Centroid alignment significant in 4/9 (was 5/9). Whole-brain **picture** knockout ceiling
significant in 3/9 (was 8/9); auditory 0/9 unchanged. Top-region ceiling share 51 % (was
59 %). ROI representative NUE041 (was NUE044) — derived, not chosen. Exactly one region
clears BH-FDR (NUE036 supramarginal, q = 0.050) where the previous text said none.

Two claims in the old Results could not be reproduced and are flagged in
`results_section.md` rather than carried over: the covariance/channel-count correlation
(needs raw `cov_*`, only `cov_nc_*` ships) and the independent-decoder agreement (needs a
`--single-modality` pass). A third **changed sign**: per-electrode knockout vs ROI size was
ρ ≈ −0.33 and is +0.10 / +0.17 here.

## Two defects fixed in passing

1. **S3 silently dropped three of nine participants.** `fig_roi_all` built a hard-coded 2×3
   grid and `zip`-truncated against the participant list; the blank-out loop could never
   fire. The grid now derives its row count from the cohort. This was in the shipped
   supplement, independent of the promotion.
2. **The ROI report advertised a size control it had not computed.** Under
   `--suff-null-draws 0` the `suff_delta_*` columns exist as all-NaN, so the delta panel was
   dropped while the heading and note still described the matched-N null. Heading and note
   now switch on whether the null is finite.

The ROI report's four-page convention, introduced alongside this promotion, is its own
entry: `017-roi-report-four-page-convention.md`.

## Open

- **`--roi-sufficiency` ran at K = 0**, so this arm has raw ROI-only accuracy and no
  size-controlled Δ or p. Alec is considering retiring the matched-N panel entirely; if it
  stays, this arm needs a K = 50 pass (order 5–7 h, not measured).
- **Sibling ladder arms are unattributable.** See the AGENTS.md note: region-importance
  writes no manifest, and this directory's previous contents differed from a recorded re-run
  by up to 0.025 acc for reasons now unrecoverable. `scope-tpm_h5`, `scope-tpfm_h10` and the
  `balance_*` pair are still in that state, so rung-to-rung ladder comparisons are not yet
  safe.
- **`balance_downsample/` is internally inconsistent** — its DK and merged CSVs are
  54-column while its NMM CSV is 38-column after a 15:29 overwrite on 2026-08-12. The only
  54-column NMM copy is the rescued file in `ladder_n9/tp_h5/balance_downsample/`.
