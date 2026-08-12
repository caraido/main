> [!WARNING]
> **STALE — do not quote any number in this file (flagged 2026-08-12).**
>
> Two independent reasons, and the second is the bigger one:
>
> 1. **CP was retired 2026-08-12** (`docs/experiments/015-retiring-cp.md`). Every `n = 10`
>    below is now 9, "all 8 participants" is 9, and the paragraph naming NUE030 as a
>    cross-task-alignment exception describes a participant who is no longer in the
>    analysis. That paragraph needs deleting, not renumbering.
> 2. **It was already stale before that.** The prose quotes NUE047 at r = 0.26, p = 0.052;
>    the current `source_data/panel_a_mds_alignment.csv` has 0.381, p = 0.004. So this text
>    predates the run it claims to describe, and patching only the CP references would make
>    it *look* current while still being wrong.
>
> The caption (`caption.md`) and `source_data/` ARE current — they were regenerated
> 2026-08-12 and are the authority. Rewrite this file from them when the cross-task
> material is next needed; note that `docs/experiments/003` retired it from the manuscript
> as a whole, so it may not be needed at all.

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
> numbers are picture **0.279**, auditory **0.234**, retention **97 %/86 %**
> (below). This resolves marker **[AL7.1]**: there is no upsampling step to
> describe — trials are simply pooled. Marker **[AL8.1]**: the electrode-level
> importance panel was **replaced by ROI/region importance** (single-channel
> effects are weak under the Nystroem-RBF dilution). Region numbers below are
> regenerated, not placeholders: all 8 participants have an ROI atlas and region
> importance runs for every one.
>
> **Second change, 2026-07-28.** NUE030 joined the auditory cohort (n = 6 → 7),
> which under the group time-warp also re-warped the other six, so every number
> here moved. Two defects were fixed in the same pass and both affected these
> results: a category label silently truncated to 10 characters, and — more
> seriously — picture and auditory channels being paired **by position** rather
> than by electrode for the three participants whose two runs used different
> channel-label vocabularies. The prior figures were computed before that fix.
> One conclusion changed direction: picture pooled-vs-within is now significant
> (p = 0.016) where it was previously n.s. (p = 0.094), so the co-trained decoder
> does pay a small, detectable cost on picture naming.
>
> **Third change, 2026-07-30.** NUE047 joined both cohorts (picture n = 12 → 13,
> both-task n = 7 → 8), and the picture arm of every cross-task analysis moved from
> the 50-epoch to the 100-epoch run, ending an epoch asymmetry that left the two
> arms' permutation nulls unequally resolved. The two effects were separated by
> re-running the co-train at the *old* cohort on both picture runs: the epoch change
> alone moved every picture-involving condition by +0.000 to +0.006 cat-indep, and
> `within_aud` by exactly 0.000. Adding NUE047 left the other seven participants'
> per-participant values bit-identical, as it must — those are computed independently
> per participant.
>
> **Two conclusions moved, both toward weaker claims.** (1) Picture
> pooled-vs-within has flipped **back to n.s.** (p = 0.055, from p = 0.016 above;
> it was n.s. before that). Across three cohorts this comparison has read n.s. →
> significant → n.s., so it should be described as *not established*, not as a
> detectable cost. Retention itself barely moved (97 %). (2) **No region in any
> participant now clears BH-FDR** (0.0 significant regions per participant, down
> from 0.43). That is not a regression: the strongest prior unit cleared its
> threshold by 4 × 10⁻⁵ against ~3 × 10⁻³ run-to-run movement, so per-region
> certification was always threshold noise. The ROI conclusions rest on ranking and
> ceiling share, which are unchanged.

## A single decoder co-trained on both tasks generalizes across modalities

**The two task-specific decoders already share a semantic geometry.** Before
co-training we asked whether the independently trained picture-naming and
auditory-naming decoders organize their outputs the same way. For each
participant we took the held-out (out-of-fold) GloVe vectors predicted at the
peak bin by each task's own decoder and embedded the two tasks' trials jointly
with metric MDS on cosine distance (Fig. R3a). The same semantic categories
occupied the same regions of the shared space in both tasks — the picture and
auditory category centroids were aligned in 5 of 8 participants
(cross-task category-centroid alignment, trial-label permutation p < 0.05;
representative NUE036 r = 0.43, p = 0.006; group mean r = 0.24; Fig. R3a, S1).
The three exceptions are informative and are not averaged away: NUE047 falls just
short of threshold (r = 0.26, p = 0.052) at an effect size larger than two
participants that do clear it, so it is better read as underpowered than as
negative; NUE038 is non-significant (r = 0.14, p = 0.20); and NUE030 shows no
cross-task alignment at all (r = −0.11, p = 0.84). NUE030 also has the smallest
picture–auditory shared vocabulary in the cohort (19 words), so its null result is
at least partly a power limitation rather than clear evidence against shared
organization.
Because both decoders map neural activity into the *same* GloVe space, this
shared organization motivated training a single decoder for both tasks.

**One co-trained decoder serves both tasks.** We pooled the trials of both tasks
and trained a single kernel-PLS semantic regressor with no resampling (Methods).
The co-trained decoder performed above chance on held-out trials of both tasks
(category-independent balanced accuracy: picture 0.289 ± 0.020, auditory
0.234 ± 0.004; n = 10). Chance is per participant rather than a single value —
1 / n_categories, mean 0.162 for picture and 0.168 for auditory — because two
participants ran an earlier auditory stimulus set with a different category
inventory (Methods). The co-trained decoder retained 97 % of the picture-naming
ceiling and 86 % of the auditory ceiling. For auditory naming that small loss is
statistically detectable (pooled vs within-task paired Wilcoxon p = 0.016); for
picture naming it is **not** (p = 0.055) (Fig. R3b, Table R1). The same pattern
holds for word-level balanced accuracy and cosine similarity in picture naming
(both p = 0.016); in auditory naming neither of those two metrics separates the
pooled from the within-task decoder (p = 0.109 and p = 0.461), so the auditory
cost is detectable only at the category level (Fig. R3b). Crucially, a decoder
trained on one task alone did **not** transfer to the other: cross-task decoding
sat near chance (0.191 picture, 0.204 auditory) and was significantly below both
the within-task and the pooled decoders (paired Wilcoxon p = 0.008). A single
co-trained model —
not reuse of a task-specific one — is therefore what decodes semantic category
from either modality, at modest cost relative to bespoke per-task decoders.

**A few brain regions carry the shared code.** Single-electrode attribution is
uninformative here — under the Nystroem-RBF map information is spread redundantly
across electrodes, so dropping any one channel barely moves accuracy and no
electrode reaches BH-FDR significance. We therefore interrogated the co-trained
model at the level of brain regions (`nmm_roi`), with three complementary
measures: permutation region-knockout Δaccuracy (the population-level drop when a
whole region is removed), analytic Jacobian sensitivity, and the model-free
neural–GloVe cross-covariance (Fig. R3c, S3). Region scores are read **per
electrode**: as totals, the magnitude measures track how many contacts happened to
land in an ROI (within participant, ρ with channel count = 0.99 for the Jacobian
and 0.96 for covariance) rather than any property of the tissue; only the knockout
is size-robust (ρ = 0.19). A
small number of regions per participant carried a large share of the whole-brain
**picture** knockout ceiling — a participant's top picture region held on average
**59 %** of it (mean whole-brain picture Δaccuracy 0.124). We report ceiling
shares for picture only:
the whole-brain **auditory** knockout clears p < 0.05 in **none of the ten**
participants (p = 0.10–0.36; the picture ceiling clears in 8/10), so for
almost every participant an auditory share would divide by a denominator
indistinguishable from zero. Under the Nystroem-RBF map even whole-region knockout
does not clear BH-FDR significance for **any** region in **any** participant
(0.0 significant regions per participant), so the region *ranking* and *picture
ceiling share*, not per-region certification, carry the signal; no ROI survives a
BH-corrected group-level test of per-electrode enrichment across participants.
Per-region certification here is threshold noise rather than a stable result: the
strongest unit in the previous 7-participant, 50-epoch analysis (NUE041 pFus)
cleared its BH threshold by 4 × 10⁻⁵ while the same statistic moves by ~3 × 10⁻³
between runs, so it was never robustly significant in either direction. All eight
participants have an ROI atlas, so this analysis runs for every one.

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
across the eight participants. Panel-a predicted embeddings are word-stratified
5-fold out-of-fold predictions from each task's own decoder; the shared 2D layout
is metric cosine-MDS (Fig. R3a, S1) and, equivalently, PCA fit on both tasks
jointly (Fig. S2), both trained on the two tasks together. ROI knockout is
computed for all ten participants (each has an `nmm_roi` atlas); the region
null uses a separate label-shuffle stream (20 shuffles) and region scores are
totals summed over each region's electrodes.

### Table R1 — retention of the within-task ceiling (`table_r1_retention.csv`)
Per-participant within-task vs pooled category-independent balanced accuracy for
each task, and their ratio (pooled ÷ within). Group means: picture 98 %,
auditory 94 % (n = 10).
