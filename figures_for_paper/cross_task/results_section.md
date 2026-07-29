_Figure package: `main/figures_for_paper/cross_task/`. Every number below is in
`source_data/group_inference.csv`; reproduce with `compute_cross_task_data.py`
(Speech env) then `cross_task_panels.py` (any env). Underlying analyses:
`main/analysis/cross_task/cross_task_prediction_mds.py` (panel a) and the reused
`balance=none` co-training run + ROI region importance
(`cross_task_region_importance.py`) under
`main/results/cross_task_cotrain/` (panels b, c and supplements)._

> **⚠ Note for co-authors — reported numbers changed from the earlier draft.**
> The draft paragraph's accuracies (picture 0.241, auditory 0.251; retention
> 78 %/87 %) came from a run that **upsampled** auditory trials. Per the current
> decision we pool the two tasks with **no resampling** (the within-task
> baselines were never resampled either — confirmed). The honest no-resampling
> numbers are picture **0.275**, auditory **0.235**, retention **97 %/85 %**
> (below). This resolves marker **[AL7.1]**: there is no upsampling step to
> describe — trials are simply pooled. Marker **[AL8.1]**: the electrode-level
> importance panel was **replaced by ROI/region importance** (single-channel
> effects are weak under the Nystroem-RBF dilution). Region numbers below are
> regenerated, not placeholders: all 7 participants have an ROI atlas and region
> importance runs for every one.
>
> **Second change, 2026-07-28.** NUEx030 joined the auditory cohort (n = 6 → 7),
> which under the group time-warp also re-warped the other six, so every number
> here moved. Two defects were fixed in the same pass and both affected these
> results: a category label silently truncated to 10 characters, and — more
> seriously — picture and auditory channels being paired **by position** rather
> than by electrode for the three participants whose two runs used different
> channel-label vocabularies. The prior figures were computed before that fix.
> One conclusion changed direction: picture pooled-vs-within is now significant
> (p = 0.016) where it was previously n.s. (p = 0.094), so the co-trained decoder
> does pay a small, detectable cost on picture naming.

## A single decoder co-trained on both tasks generalizes across modalities

**The two task-specific decoders already share a semantic geometry.** Before
co-training we asked whether the independently trained picture-naming and
auditory-naming decoders organize their outputs the same way. For each
participant we took the held-out (out-of-fold) GloVe vectors predicted at the
peak bin by each task's own decoder and embedded the two tasks' trials jointly
with metric MDS on cosine distance (Fig. R3a). The same semantic categories
occupied the same regions of the shared space in both tasks — the picture and
auditory category centroids were aligned in 5 of 7 participants
(cross-task category-centroid alignment, trial-label permutation p < 0.05;
representative NUEx036 r = 0.43, p = 0.006; group mean r = 0.24; Fig. R3a, S1).
The two exceptions are informative and are not averaged away: NUEx038 is
non-significant (r = 0.14, p = 0.19) and NUEx030 shows no cross-task alignment at
all (r = −0.14, p = 0.91). NUEx030 also has the smallest picture–auditory shared
vocabulary in the cohort (19 words), so its null result is at least partly a
power limitation rather than clear evidence against shared organization.
Because both decoders map neural activity into the *same* GloVe space, this
shared organization motivated training a single decoder for both tasks.

**One co-trained decoder serves both tasks.** We pooled the trials of both tasks
and trained a single kernel-PLS semantic regressor with no resampling (Methods).
The co-trained decoder performed above chance on held-out trials of both tasks
(category-independent balanced accuracy: picture 0.275 ± 0.025, auditory
0.235 ± 0.019; n = 7). Chance is per participant rather than a single value —
1 / n_categories, mean 0.160 for picture and 0.168 for auditory — because two
participants ran an earlier auditory stimulus set with a different category
inventory (Methods). The co-trained decoder retained 97 % of the picture-naming
ceiling and 85 % of the auditory ceiling, but in both cases the small loss is
statistically detectable (pooled vs within-task paired Wilcoxon: picture
p = 0.016, auditory p = 0.031) (Fig. R3b, Table R1). This held for word-level
balanced accuracy and cosine similarity as well (Fig. R3b). Crucially, a decoder
trained on one task alone did **not** transfer to the other: cross-task decoding
sat near chance (0.193 picture, 0.205 auditory) and was significantly below both
the within-task and the pooled decoders (paired Wilcoxon p = 0.016). A single
co-trained model —
not reuse of a task-specific one — is therefore what decodes semantic category
from either modality, at modest cost relative to bespoke per-task decoders.

**A few brain regions carry the shared code.** Single-electrode attribution is
uninformative here — under the Nystroem-RBF map information is spread redundantly
across electrodes, so dropping any one channel barely moves accuracy and no
electrode reaches BH-FDR significance. We therefore interrogated the co-trained
model at the level of brain regions (`primary_roi`), with three complementary
measures: permutation region-knockout Δaccuracy (the population-level drop when a
whole region is removed), analytic Jacobian sensitivity, and the model-free
neural–GloVe cross-covariance (Fig. R3c, S3). Region scores are read **per
electrode**: as totals, the magnitude measures track how many contacts happened to
land in an ROI (within participant, ρ with channel count = 0.99 for the Jacobian
and 0.96 for covariance) rather than any property of the tissue; only the knockout
is size-robust (ρ = 0.19). A
small number of regions per participant carried a large share of the whole-brain
**picture** knockout ceiling — a participant's top picture region held on average
**46 %** of it (mean whole-brain picture Δaccuracy 0.148) — and the methods
concur on the leading regions (e.g. NUEx038 post depth is the top region under all
of them, at 82 % of its picture ceiling). We report ceiling shares for picture only:
the whole-brain **auditory** knockout does not reach significance in any
participant (p = 0.23–0.42), so an auditory share divides by a denominator
indistinguishable from zero. Under the Nystroem-RBF map even whole-region knockout
rarely clears BH-FDR significance (≈0.17 significant regions per participant), so
the region *ranking* and *picture ceiling share*, not per-region certification,
carry the signal; no ROI survives a BH-corrected group-level test of per-electrode
enrichment across participants. All seven participants have an ROI atlas, so this
analysis runs for every one.

Two features of this analysis should not be over-read. First, ROI size and ROI
identity are collinear by implant design — depth shanks and MTG strips carry ~20
contacts while ventral gyral ROIs carry 3–6 — so per-electrode enrichment retains a
negative correlation with ROI size (ρ ≈ −0.33) that normalization cannot remove.
Second, because a single co-trained model scores both tasks through one shared map,
its picture and auditory ROI rankings agree near-perfectly by construction
(Jacobian ρ = +0.99 per electrode); that agreement is a property of the model, not
evidence that the regions are amodal. Comparing two *independently* trained
single-task decoders instead, per-electrode agreement is weak (ρ = 0.02–0.43).
These regions — informative for semantic decoding in both tasks — are therefore
candidate targets for a future implant, and candidate sites of the **shared,
alignable subspace** characterized above, rather than demonstrated sites of an
amodal code.

### Methods note
For each participant, picture- and auditory-naming trials (high-gamma activity at
each task's own loose-category peak bin, on the shared channel set) were pooled
into one training set with **no resampling**, and a single kernel-PLS regressor
(Nystroem-RBF, 100 landmarks → PLS, 10 components) was fit to GloVe. Held-out
evaluation used a per-word train/test split with a 30 % zero-shot word holdout,
50 bootstraps; within-task, cross-task and pooled conditions share the same test
sets so comparisons are paired. Group statistics are paired Wilcoxon signed-rank
across the seven participants. Panel-a predicted embeddings are word-stratified
5-fold out-of-fold predictions from each task's own decoder; the shared 2D layout
is metric cosine-MDS (Fig. R3a, S1) and, equivalently, PCA fit on both tasks
jointly (Fig. S2), both trained on the two tasks together. ROI knockout is
computed for all seven participants (each has a `primary_roi` atlas); the region
null uses a separate label-shuffle stream (20 shuffles) and region scores are
totals summed over each region's electrodes.

### Table R1 — retention of the within-task ceiling (`table_r1_retention.csv`)
Per-participant within-task vs pooled category-independent balanced accuracy for
each task, and their ratio (pooled ÷ within). Group means: picture 97 %,
auditory 85 % (n = 7).
