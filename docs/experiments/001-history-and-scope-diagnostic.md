---
id: 001
kind: experiment
title: Did the 2026-08 re-run lose ~5% to a narrower channel gate or to a shorter history window?
status: answered
analysis: semantic_regression
opened: 2026-08-11
closed: 2026-08-11
runs:
  - 2026-08-11_00-51-28_auditory_naming_warp-stim-group_align-aud_stim_onset_roi-nmm_scope-tpfm_h5_kernel_pls_cosine_100ep
  - 2026-08-11_02-50-41_auditory_naming_warp-stim-group_align-aud_stim_onset_roi-nmm_scope-tp_h10_kernel_pls_cosine_100ep
  - 2026-08-11_04-49-25_auditory_naming_warp-stim-group_align-aud_stim_onset_roi-nmm_scope-tpfm_h10_kernel_pls_cosine_100ep
  - 2026-08-11_06-56-18_picture_naming_roi-nmm_scope-tpfm_h5_kernel_pls_cosine_100ep
  - 2026-08-11_10-01-39_picture_naming_roi-nmm_scope-tp_h10_kernel_pls_cosine_100ep
  - 2026-08-11_11-37-57_picture_naming_roi-nmm_scope-tpfm_h10_kernel_pls_cosine_100ep
report:
answer: >
  Channels, not history. Adding frontal+medial cortex (tpfm) raises picture naming 5-11%
  (3 of 5 metrics survive BH-FDR, 11-13 of 15 participants); doubling history to 10 bins
  LOWERS it 1-3% (3-6 of 15). The kernel-bandwidth confound is ruled out: tp/h10 has more
  features than tpfm/h5 and performs worst. Auditory agrees on channels but not history,
  and nothing there survives correction.
---

## Question

The 2026-08 re-run changed two things at once — the channel gate (whole-brain → the 13-region
temporal-parietal whitelist) and the history window (10 → 5 bins) — and decoding got worse.
Measured on shared participants, peak per patient, GloVe, from `per_time_scores.csv`:
auditory (8 shared) −5.7/−6.9/−5.6/−3.0/−5.0%, picture (13 shared) −7.2/−3.7/−4.2/−5.3/−5.4%
on cosine/cat/word/top-3/top-5. **Not auditory-specific**, which is how it was first framed.

A 2×2 at the current cohort; the `tp`/h5 corner is `PIC_RUN`/`AUD_RUN`, so three new arms per
task. `tpfm` = the 13 + the `FRONTAL` and `MEDIAL` families (23 regions), excluding
sensorimotor, occipital, subcortical and auditory belt — reasons in
[`../agent-context/roi-vocabulary.md`](../agent-context/roi-vocabulary.md#scopes-added-2026-08-11).
Required a new `--roi-scope` axis; the region set had no CLI mechanism before.

## What was tried

Six arms, GloVe only, 100 epochs, `--roi-atlas nmm`, explicit `--patients`. All 15/15 and
10/10, **0 failures**, ~12.5 h. Verified at the pkl level (`clean_channel_rois` ⊆ the named
scope, `n_bins_history` matching the run-id token), not only against `meta.json`. `tp` arms
reproduced the baseline channel sets exactly (picture 633, auditory 420); both auditory arms
recomputed the warp target to 3.5600000000000005, identical to `AUD_RUN`.

## Result

**Picture (n=15)**, change vs `PIC_RUN`; p / q(BH over the 15 picture tests) / patients improving:

| arm | ch | n_feat | cosine | cat | word | top-3 | top-5 |
|---|---|---|---|---|---|---|---|
| `tp`/h5 = `PIC_RUN` | 633 | 3,165 | 0.1136 | 0.3024 | 0.0468 | 0.1113 | 0.1721 |
| `tpfm`/h5 | 1,047 | 5,235 | **+11.3%** | **+6.1%** | +6.1% | +5.3% | **+6.7%** |
| | | | .005/**.04**/13 | .008/**.04**/13 | .107/.20/12 | .055/.12/11 | .007/**.04**/11 |
| `tp`/h10 | 633 | 6,330 | −3.0% | −2.2% | −1.4% | −2.1% | −1.3% |
| | | | .055/.12/5 | .030/.11/3 | .847/.85/8 | .252/.42/6 | .277/.42/6 |
| `tpfm`/h10 | 1,047 | 10,470 | +4.3% | +2.7% | +3.3% | +2.9% | +4.0% |
| | | | .359/.45/9 | .524/.61/8 | .599/.64/8 | .330/.45/9 | .055/.12/9 |

**Auditory (n=10)**, change vs `AUD_RUN`: `tpfm`/h5 +1.9/+5.1/+2.2/−3.9/−3.1%; `tp`/h10
+6.9/+3.6/+1.9/+1.3/+4.2%; `tpfm`/h10 +7.1/+11.3/+5.4/+0.3/+3.9%. **0 of 15 survive BH-FDR.**
The only cell under p=0.05 uncorrected is `tpfm`/h10 category (p=0.010, q=0.15, 8/10).
Patients improving run 3/10–8/10 — coin flips.

**The kernel-bandwidth confound is ruled out for picture.** `tp`/h10 carries *more* features
than `tpfm`/h5 (6,330 vs 5,235) and is the *worst* arm; under `gamma = 1/n_features` it should
have been the best. The auditory arms *do* order by `n_feat`, one more reason to weight them
lightly.

**A noise floor turned up for free.** VB has no frontal/medial coverage (41/50 channels under
both scopes), so its `tpfm`/h5 run and `PIC_RUN` are the same configuration differing only in
random splits. VB moves +1.3% cosine, +1.4% cat, −0.7% top-3, −1.4% top-5, and **+12.3% word**.
The four small ones are the four metrics that reached significance; word, the noisiest, is the
one that did not (p=0.107). n=1, and it also absorbs the code-state/embedding difference, so it
is an upper bound.

## Next

- **A decision for the group, not a conclusion this entry can make.** `tpfm` is diagnostic: no
  palette coverage (`utils/roi_palette.py` is vendored and cannot be extended here), and
  adopting it means the paper is no longer about temporal-parietal cortex. That is a scope
  claim, not a hyperparameter. The performance case is real on picture; the cost is the framing.
- If adopted, re-run the winning arm with the **full embedding set** — these are GloVe-only.
- The 500 ms history standard is **supported** on picture naming; do not revert it.

**Limitations travelling with every number above.** (1) No noise floor except VB (n=1, upper
bound) — there is no random seed anywhere, `train_test_split` and `KFold` both run
`random_state=None`; the cohort-wide replicate was declined. (2) The baseline corner differs in
more than the two factors — different code state, 6 embeddings not 1 (GloVe sees identical data,
but the code state stands). (3) All six launched from a dirty tree; the commit made immediately
after is the state that ran, as no code changed in between. (4) Auditory and picture disagree
about history (+6.9% vs −3.0% cosine), unexplained. (5) `tpfm` was never run under DK, so there
is no cross-atlas check.
