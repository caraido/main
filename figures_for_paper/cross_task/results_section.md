_Figure package: `main/figures_for_paper/cross_task/`. Every number below is in
`source_data/group_inference.csv`; reproduce with `compute_cross_task_data.py`
(Speech env) then `cross_task_panels.py` (any env). Underlying analyses:
`main/tests/cross_task/cross_task_prediction_mds.py` (panel a) and the reused
`balance=none` co-training run + channel/region importance under
`main/tests/results/cross_task_cotrain/` (panels b, c and supplements)._

> **⚠ Note for co-authors — reported numbers changed from the earlier draft.**
> The draft paragraph's accuracies (picture 0.241, auditory 0.251; retention
> 78 %/87 %) came from a run that **upsampled** auditory trials. Per the current
> decision we pool the two tasks with **no resampling** (the within-task
> baselines were never resampled either — confirmed). The honest no-resampling
> numbers are picture **0.302**, auditory **0.231**, retention **98 %/78 %**
> (below). This resolves marker **[AL7.1]**: there is no upsampling step to
> describe — trials are simply pooled. Marker **[AL8.1]** is confirmed
> (participant NUEx031: top electrodes V2, V3, V4).

## A single decoder co-trained on both tasks generalizes across modalities

**The two task-specific decoders already share a semantic geometry.** Before
co-training we asked whether the independently trained picture-naming and
auditory-naming decoders organize their outputs the same way. For each
participant we took the held-out (out-of-fold) GloVe vectors predicted at the
peak bin by each task's own decoder and embedded the two tasks' trials jointly
with metric MDS on cosine distance (Fig. R3a). The same semantic categories
occupied the same regions of the shared space in both tasks — the picture and
auditory category centroids were aligned in 5 of 6 participants
(cross-task category-centroid alignment, trial-label permutation p < 0.05;
representative NUEx036 r = 0.36, p = 0.006; group mean r = 0.32; Fig. R3a, S1).
Because both decoders map neural activity into the *same* GloVe space, this
shared organization motivated training a single decoder for both tasks.

**One co-trained decoder serves both tasks.** We pooled the trials of both tasks
and trained a single kernel-PLS semantic regressor with no resampling (Methods).
The co-trained decoder performed above chance on held-out trials of both tasks
(category-independent balanced accuracy: picture 0.302 ± 0.026, auditory
0.231 ± 0.022; chance 0.167; n = 6), retaining essentially all of the
picture-naming ceiling (98 %; pooled vs within-task paired Wilcoxon p = 0.094,
n.s.) and the large majority of the auditory ceiling (78 %; p = 0.031)
(Fig. R3b, Table R1). This held for word-level balanced accuracy and cosine
similarity as well (Fig. R3b). Crucially, a decoder trained on one task alone did
**not** transfer to the other: cross-task decoding sat at chance
(0.186 for both directions) and was significantly below both the within-task and
the pooled decoders (paired Wilcoxon p = 0.031). A single co-trained model —
not reuse of a task-specific one — is therefore what decodes semantic category
from either modality, at modest cost relative to bespoke per-task decoders.

**A sparse, task-general electrode set carries the shared code.** Interrogating
the co-trained model with the PLS variable-importance-in-projection (VIP) score —
which weights each electrode by its contribution to the shared neural→GloVe
mapping — roughly one third of electrodes per participant exceeded average
importance (VIP > 1; mean 36 %, range 29–44 %), a sparse set of high-importance,
task-general electrodes (e.g. participant NUEx031: V2, V3, V4; Fig. R3c, S3).
Permutation and region-knockout importance point to the same sites, though
single-channel effects are diluted under the Nystroem-RBF map and none reach
BH-FDR significance (Fig. S4, S5). These electrodes — informative for semantic
decoding in both tasks — are candidate targets for a future implant and candidate
sites of an amodal, shared semantic representation.

### Methods note
For each participant, picture- and auditory-naming trials (high-gamma activity at
each task's own loose-category peak bin, on the shared channel set) were pooled
into one training set with **no resampling**, and a single kernel-PLS regressor
(Nystroem-RBF, 100 landmarks → PLS, 10 components) was fit to GloVe. Held-out
evaluation used a per-word train/test split with a 30 % zero-shot word holdout,
50 bootstraps; within-task, cross-task and pooled conditions share the same test
sets so comparisons are paired. Group statistics are paired Wilcoxon signed-rank
across the six participants. Panel-a predicted embeddings are word-stratified
5-fold out-of-fold predictions from each task's own decoder; the shared 2D layout
is metric cosine-MDS (Fig. R3a, S1) and, equivalently, PCA fit on both tasks
jointly (Fig. S2), both trained on the two tasks together. ROI knockout is
available only for participants with a `primary_roi` atlas (NUEx041/044/038/036);
NUEx045 and NUEx031 have none.

### Table R1 — retention of the within-task ceiling (`table_r1_retention.csv`)
Per-participant within-task vs pooled category-independent balanced accuracy for
each task, and their ratio (pooled ÷ within). Group means: picture 98 %,
auditory 78 % (n = 6).
