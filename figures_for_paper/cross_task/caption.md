# Caption for `00_cross_task_combined.pdf` (panels also shipped as `01_`, `02_`, `03_`)

**Figure | Cross-task semantic decoding from a single co-trained decoder.** Held-out
decoding of picture naming and auditory naming in the participants who performed both
tasks (N = 9). High-gamma activity at each task's peak bin was mapped onto GloVe word
embeddings by kernel partial-least-squares regression and scored by nearest-neighbour
retrieval; the pooled decoder was trained on both tasks' trials with the majority task
downsampled to equalise them. **a** Semantic organization of the two separate per-task
decoders for one participant (NUE036): held-out predicted GloVe vectors (word-stratified
five-fold out-of-fold) from the picture-only and auditory-only decoders, embedded jointly
by metric MDS on cosine distance. Dot: one trial, coloured by semantic category. Both
subplots share one MDS coordinate frame. Bar plot: cross-task category-centroid alignment
per participant; red: permutation p < 0.05 (4 of 9). **b** Within-task decoder (grey),
other-task decoder (cross, red) and pooled co-trained decoder (blue), on held-out picture
(top) and auditory (bottom) trials, for category-independent balanced accuracy, word
balanced accuracy and cosine similarity. Bar: mean ± s.e.m. across participants; dot: one
participant; dashed line: mean per-participant chance. **c** Region importance of the
co-trained model for one participant (NUE041), organized by `nmm_roi`. Left: permutation
Δcategory-independent accuracy when a region's history block is jointly shuffled. Right:
region analytic Jacobian sensitivity per electrode. Dashed lines: whole-brain knockout
ceiling. Regions share one y-order (picture Δacc). Inset: each participant's top region as
a share of its whole-brain picture ceiling. Stars: paired Wilcoxon across participants
(\*\* p < 0.01, \* p < 0.05, n.s. not significant), computed separately per metric. The
pooled decoder reaches 0.82 of the within-task ceiling for picture (p = 0.0039) and 0.92
for auditory (p = 0.012). Channel set is the 18-region `tpm` scope — temporal-parietal
regions plus insula, cingulate, entorhinal, parahippocampal and precuneus — with 1000 ms
of history, so it is not temporal-parietal cortex alone. Chance is per participant
(1 / n_categories; picture mean 0.164, auditory mean 0.170): NUE031 ran an earlier
auditory stimulus set and shares only 4 categories. Magnitudes are per electrode because
region totals track channel count. The whole-brain knockout ceiling is significant in 3 of
9 participants for picture and 0 of 9 for auditory (p = 0.08–0.34). Picture and auditory
agree closely in **c** because one co-trained model scores both tasks through a single
shared map; this is a property of the model, not evidence of amodal coding.

## Notes — not part of the caption

**The numbers changed on 2026-08-13 and every one of them moved.** The figure was rebuilt
at the `tpm`/h10 pair with `balance=downsample`; it previously used `tp`/h5 with no
resampling. Superseded values that must not be quoted from any earlier draft: retention
97 %/86 % (or 99 %/94 %), within-vs-pooled p = 0.055 / 0.016 (or 0.098 / 0.098), picture
ceiling significant in 8/9, alignment significant in 5/9, ROI representative NUE044.

**The retention drop is the resampling change, not the scope/history change.** Holding
factors fixed one at a time across the same nine participants: `tp`/h5→`tpm`/h10 at
`balance=none` moves picture retention by +0.002 (0.987→0.989); `none`→`downsample` at
`tpm`/h10 moves it by −0.171 (0.989→0.818). Within-task ceilings are essentially unchanged
(0.2926→0.2932), so this is the pooled decoder, not the baseline. Auditory retention moves
the other way under downsampling (+0.056).

**Inputs** (pinned in `utils/config.py`):
- co-training + RSA — `CROSS_TASK_FIGURE_COTRAIN_RUN` = `2026-08-12_18-17-20_kernel_pls_balance-downsample_50boot`
- ROI importance — `CROSS_TASK_FIGURE_ROI_DIR` = `scope-tpm_h10/balance_downsample`
- panel a MDS — `CROSS_TASK_FIGURE_MDS_RUN` = `2026-08-13_00-22-11_prediction_mds_separate_kfold5_seed42`
- upstream pair — picture
  `2026-08-11_23-42-55_picture_naming_roi-nmm_scope-tpm_h10_kernel_pls_cosine_100ep`, auditory
  `2026-08-12_09-14-11_auditory_naming_warp-stim-group_align-aud_stim_onset_roi-nmm_scope-tpm_h10_kernel_pls_cosine_100ep`

**Representative selection is derived, not chosen.** `panel_c_roi_coverage.csv` records
`selection_rule = "strongest-in-both (significance not attainable)"`: no region clears
BH-FDR in both tasks in any participant, so the fallback rule applies and the panel shows
the strongest region regardless of significance. `n_sig_regions_mean` is 0.11.

**Auditory word-retrieval and cosine contrasts are all n.s.** in **b** (within-vs-cross
p = 0.43, within-vs-pooled p = 0.36). Only category-independent accuracy separates the
arms for auditory; the caption's stars carry this per metric.

## Supplements

**S1, Semantic-organization MDS for all participants** — as **a**, every participant.
Cross-task category-centroid alignment is significant in 4 of 9 (n.s.: NUE044 p = 0.108,
NUE041 p = 0.214, NUE045 p = 0.251, NUE051 p = 0.295, NUE038 p = 0.377). No participant
has a negative point estimate.

**S2, Semantic-organization PCA for all participants** — as **S1**, but the shared 2D
space is PCA fit on both tasks' predicted embeddings jointly instead of cosine-MDS. The
alignment values are identical (computed on the 300-D predicted vectors, independent of
the 2D projection).

**S1 (3D) / S2 (3D)** (`*_all_3d`) — the same MDS and PCA maps reduced to **three**
components (each reducer still fit on both tasks jointly), one 3D scatter per task with
shared x/y/z limits. PCA is nested, so its first two axes match S2; the 3-component MDS is
a separate fit.

**S3, Region (ROI) knockout importance for all participants** — Δcategory-independent
accuracy when a whole `nmm_roi` region is jointly shuffled, picture vs auditory, regions
ordered by picture Δacc; dashed lines mark each participant's whole-brain ceiling. All
nine participants are shown. *(Until 2026-08-13 the grid was a hard-coded 2×3 and silently
truncated to the first six; three participants were missing from the shipped supplement.)*

*(S4, region-total plain-PLS VIP, was removed 2026-07-23. VIP has no well-defined
input-space analogue under the Nyström map, so it attributed a linear surrogate the paper
does not report, and as a region total it tracked ROI electrode count (ρ = 0.98 within
participant). The kernel-approximation control it was meant to provide is better served by
the neural–GloVe covariance, which is model-free.)*

**S7, Cross-task representational similarity** — Spearman correlation of the per-word
neural RDM between tasks (pic ↔ aud) and of each task with the GloVe RDM.

_Display IDs (NUE###) map to internal initials in `participants.json`. Source data:
`figures_for_paper/cross_task/source_data/`; regenerate with `compute_cross_task_data.py`
then `cross_task_panels.py`._
