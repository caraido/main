_Figure package: `main/figures_for_paper/cross_task/`. Every number below is in
`source_data/group_inference.csv` unless marked otherwise; reproduce with
`compute_cross_task_data.py` (Speech env) then `cross_task_panels.py` (any env).
Underlying analyses: `main/analysis/cross_task/cross_task_prediction_mds.py` (panel a) and
the **`balance=downsample`** co-training run + ROI region importance
(`cross_task_region_importance.py`) under `main/results/cross_task_cotrain/` (panels b, c
and supplements). All three inputs are pinned as `CROSS_TASK_FIGURE_*` in
`utils/config.py`._

> **⚠ Note for co-authors — reported numbers changed again on 2026-08-13, and the
> headline retention claim did not survive.** The figure moved to the `tpm`/h10 channel
> set with the pooled training set **class-balanced by downsampling**. It previously used
> `tp`/h5 with **no resampling**, and the paragraph below used to argue that no-resampling
> was the honest choice. That argument is withdrawn: downsampling is the control for
> picture trials outnumbering auditory ~3:1, and the previous numbers do not survive it.
>
> Picture retention falls from **99 % to 82 %**, and pooled-vs-within becomes
> **statistically detectable in both tasks** (picture p = 0.0039, auditory p = 0.012;
> both were p = 0.098, n.s.). **The cause is the resampling change, not the channel-set or
> history change**, established by holding factors fixed one at a time over the same nine
> participants: `tp`/h5 → `tpm`/h10 at `balance=none` moves picture retention by **+0.002**
> (0.987 → 0.989), while `none` → `downsample` at `tpm`/h10 moves it by **−0.171**
> (0.989 → 0.818). Within-task ceilings are essentially unchanged (0.2926 → 0.2932), so
> this is the pooled decoder changing and not its baseline. Auditory retention moves the
> other way under downsampling (0.863 → 0.919), as expected when the majority task stops
> crowding the minority one.
>
> **Superseded values — do not quote from any earlier draft:** retention 97 %/86 % or
> 99 %/94 %; within-vs-pooled p = 0.055 / 0.016 or 0.098 / 0.098; picture whole-brain
> ceiling significant in 8/9; centroid alignment significant in 5/9; ROI representative
> NUE044; top-region ceiling share 59 %.
>
> **Marker [AL7.1] changes with it.** There *is* now a resampling step to describe: the
> pooled training set is downsampled to equalise the two tasks; within-task baselines are
> still not resampled. Marker **[AL8.1]** is unchanged — the electrode-level importance
> panel remains replaced by ROI/region importance.
>
> **Earlier changes, retained for the record.** (2026-07-28) NUE030 joined the auditory
> cohort and two defects were fixed that both still stand: a category label silently
> truncated to 10 characters, and picture/auditory channels being paired *by position*
> rather than by electrode for the three participants whose runs used different
> channel-label vocabularies. (2026-07-30) NUE047 joined both cohorts and the picture arm
> moved from the 50- to the 100-epoch run; the epoch change alone moved every
> picture-involving condition by +0.000 to +0.006 cat-indep. (2026-08-12) **NUE030 was
> retired** by group consensus (`docs/experiments/015-retiring-cp.md`); n = 9. Removing it
> did not re-warp anyone — the group warp target is pinned at 3.5600 s — verified
> max|diff| = 0.0.
>
> **Picture pooled-vs-within has now read n.s. → significant → n.s. → significant across
> four analyses.** It is n.s. at n = 6 (p = 0.094), significant at n = 7 (p = 0.016), n.s.
> at n = 9 unbalanced (p = 0.098), and significant at n = 9 downsampled (p = 0.0039). The
> first three flips are cohort size; the last is the resampling control. It should be
> described as *established only under class balancing*, with the unbalanced result stated
> alongside it.

## A single decoder co-trained on both tasks decodes semantic category from either

**The two task-specific decoders already share a semantic geometry.** Before co-training
we asked whether the independently trained picture-naming and auditory-naming decoders
organize their outputs the same way. For each participant we took the held-out
(out-of-fold) GloVe vectors predicted at the peak bin by each task's own decoder and
embedded the two tasks' trials jointly with metric MDS on cosine distance (Fig. R3a). The
same semantic categories occupied the same regions of the shared space in **4 of 9**
participants (cross-task category-centroid alignment, trial-label permutation p < 0.05;
representative NUE036 r = 0.441, p = 0.002; group mean r = 0.258; Fig. R3a, S1). The five
non-significant participants are not averaged away: NUE044 (r = 0.212, p = 0.108),
NUE041 (r = 0.107, p = 0.214), NUE045 (r = 0.092, p = 0.251), NUE051 (r = 0.086,
p = 0.295) and NUE038 (r = 0.056, p = 0.377). **No participant has a negative point
estimate**, so the non-significant cases are better read as underpowered than as evidence
against shared organization. Because both decoders map neural activity into the *same*
GloVe space, this shared organization motivated training a single decoder for both tasks.

**One co-trained decoder serves both tasks, at a measurable cost.** We pooled the trials
of both tasks, downsampling the majority task so the two contribute equally, and trained a
single kernel-PLS semantic regressor (Methods). The co-trained decoder performed above
chance on held-out trials of both tasks (category-independent balanced accuracy: picture
0.240 ± 0.013, auditory 0.251 ± 0.012; n = 9). Chance is per participant rather than a
single value — 1 / n_categories, mean 0.164 for picture and 0.170 for auditory — because
NUE031 ran an earlier auditory stimulus set with a different category inventory (Methods).
The co-trained decoder reached **82 % of the picture-naming ceiling and 92 % of the
auditory ceiling**, and both shortfalls are statistically detectable (pooled vs within-task
paired Wilcoxon: picture p = 0.0039, auditory p = 0.012; Fig. R3b, Table R1). For picture
naming the same pattern holds for word-level balanced accuracy and cosine similarity (both
p = 0.0039); **for auditory naming neither of those metrics separates any pair of arms**
(within-vs-cross p = 0.43, within-vs-pooled p = 0.36, cross-vs-pooled p = 0.34 for word
accuracy; p = 0.13 / 0.36 / 0.098 for cosine), so the auditory result rests on the
category-level metric alone. Crucially, a decoder trained on one task alone did **not**
transfer to the other: cross-task decoding sat near chance (0.191 picture, 0.198 auditory)
and was below both the within-task and the pooled decoders in both tasks (paired Wilcoxon
p = 0.0039). A single co-trained model — not reuse of a task-specific one — is therefore
what decodes semantic category from either modality, at a cost relative to bespoke
per-task decoders that is now detectable rather than negligible.

**A few brain regions carry the shared code.** Single-electrode attribution is
uninformative here — under the Nystroem-RBF map information is spread redundantly across
electrodes, so dropping any one channel barely moves accuracy and no electrode reaches
BH-FDR significance. We therefore interrogated the co-trained model at the level of brain
regions (`nmm_roi`), with permutation region-knockout Δaccuracy (the population-level drop
when a whole region is removed), analytic Jacobian sensitivity, and the model-free
neural–GloVe cross-covariance (Fig. R3c, S3). Region scores are read **per electrode**: as
totals the Jacobian tracks how many contacts happened to land in an ROI (within
participant, median Spearman ρ with channel count = +0.96), while the knockout total is
less size-driven (ρ = +0.39) and its per-electrode form is close to size-independent
(ρ = +0.10 picture, +0.17 auditory). A participant's top picture region held on average
**51 %** of the whole-brain picture knockout ceiling (mean whole-brain picture Δaccuracy
0.077). We report ceiling shares for picture only: the whole-brain **auditory** knockout
clears p < 0.05 in **none of the nine** participants (p = 0.08–0.34), against **3 of 9**
for picture (p = 0.005–0.33), so for most participants an auditory share would divide by a
denominator indistinguishable from zero. Under the Nystroem-RBF map region knockout clears
BH-FDR for exactly **one** region in one participant (NUE036 supramarginal, q = 0.050,
8 contacts) and for none in auditory, so the region *ranking* and *picture ceiling share*,
not per-region certification, carry the signal. All nine participants have an ROI atlas,
so this analysis runs for every one.

Two features of this analysis should not be over-read. First, ROI size and ROI identity
are collinear by implant design — depth shanks and MTG strips carry ~20 contacts while
ventral gyral ROIs carry 3–6 — so per-electrode normalization removes most but not all of
the size dependence. Second, because a single co-trained model scores both tasks through
one shared map, its picture and auditory ROI rankings agree almost perfectly by
construction (Jacobian per-electrode ρ = +1.00; knockout per electrode ρ = +0.30, range
+0.14 to +0.72); that agreement is a property of the model, not evidence that the regions
are amodal. These regions — informative for semantic decoding in both tasks — are
therefore candidate targets for a future implant, and candidate sites of the **shared,
alignable subspace** characterized above, rather than demonstrated sites of an amodal code.

> **Two figures from the previous draft are not reproduced here and must be recomputed
> before use.** (1) The neural–GloVe covariance's correlation with channel count (quoted as
> ρ = 0.96): `panel_c_roi.csv` ships only the size-normalized `cov_nc_*` columns, so it
> cannot be checked from the figure's source data. (2) The per-electrode agreement between
> two *independently* trained single-task decoders (quoted as ρ = 0.02–0.43): this needs
> the `_solo` columns from a `--single-modality` pass, which the current arm did not run.
> A third claim has **changed sign** and has been corrected above: per-electrode enrichment
> was reported to retain a *negative* correlation with ROI size (ρ ≈ −0.33); at this
> configuration it is +0.10 / +0.17.

### Methods note
For each participant, picture- and auditory-naming trials (high-gamma activity at each
task's own loose-category peak bin, on the shared channel set) were pooled into one
training set with the **majority task downsampled** so the two tasks contribute equally,
and a single kernel-PLS regressor (Nystroem-RBF, 100 landmarks → PLS, 10 components) was
fit to GloVe. Within-task baselines are not resampled. Held-out evaluation used a per-word
train/test split with a 30 % zero-shot word holdout, 50 bootstraps; within-task, cross-task
and pooled conditions share the same test sets so comparisons are paired. Group statistics
are paired Wilcoxon signed-rank across the nine participants. Panel-a predicted embeddings
are word-stratified 5-fold out-of-fold predictions from each task's own decoder; the shared
2D layout is metric cosine-MDS (Fig. R3a, S1) and, equivalently, PCA fit on both tasks
jointly (Fig. S2). ROI knockout is computed for all nine participants (each has an
`nmm_roi` atlas); the region null uses a separate label-shuffle stream (20 shuffles) and
region scores are totals summed over each region's electrodes.

**Channel set.** These analyses use the 18-region `tpm` scope — the temporal-parietal
whitelist plus insula, cingulate, entorhinal, parahippocampal and precuneus — with 1000 ms
(10 bins) of history. "Temporal-parietal cortex" therefore does not describe this figure's
channel set, and the region counts here are not comparable with analyses run on the
13-region `tp` scope.

### Table R1 — retention of the within-task ceiling (`table_r1_retention.csv`)
Per-participant within-task vs pooled category-independent balanced accuracy for each task,
and their ratio (pooled ÷ within). Group means: picture **82 %**, auditory **92 %**
(n = 9). Per-participant picture retention ranges 0.75 (NUE051) to 0.90 (NUE036); auditory
ranges 0.83 (NUE050) to 1.05 (NUE041, the only participant whose pooled decoder exceeds its
within-task auditory ceiling).
