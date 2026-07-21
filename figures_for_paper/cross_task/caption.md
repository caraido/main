# Figure R3 — A single decoder co-trained on both speech tasks generalizes across modalities

**a, Semantic organization of the two separate per-task decoders.** Held-out
predicted GloVe vectors (peak bin, word-stratified 5-fold out-of-fold) from the
picture-only and auditory-only decoders of one participant (NUEx036), embedded
jointly by metric MDS on cosine distance; each dot is a trial, coloured by
semantic category. The two subplots share one MDS coordinate frame (identical x-
and y-limits), so a category occupying the same region in both tasks reflects a
shared organization (e.g. body part low in both). Bar plot: cross-task
category-centroid alignment per participant
(red, permutation p < 0.05; 5/6 significant). **b, Co-training generalizes.**
Category-independent balanced accuracy, word balanced accuracy and cosine
similarity for the within-task decoder (grey), the other-task decoder
(cross, red) and the pooled co-trained decoder (blue), on held-out
picture-naming (top) and auditory-naming (bottom) trials. Bars mean ± s.e.m.,
dots participants, dashed line chance; stars paired Wilcoxon across participants
(* p < 0.05, n.s. not significant). The single-task decoder does not transfer
(cross ≈ chance); the pooled decoder retains 98 % (picture) and 78 % (auditory)
of the within-task ceiling. **c, Task-general brain regions.** Region (ROI)
importance of the co-trained model for one participant (NUEx038), organized by
`primary_roi`: (left) permutation Δcategory-independent accuracy when a whole
region's history block is jointly shuffled — the population-level drop when the
region is removed — picture vs auditory, dashed lines the whole-brain knockout
ceiling; (middle) region-total analytic Jacobian sensitivity; (right)
region-total plain-PLS VIP. Regions share one y-order (by picture Δacc). Bar plot
inset: each participant's top region as a share of its whole-brain ceiling.
n = 6 participants (both picture and auditory naming). No trial resampling;
chance 0.167 (six categories). Single-channel attribution is not shown — under
the Nystroem-RBF dilution single-electrode effects are weak; the region view is
the population-level signal.

## Supplements

**S1, Semantic-organization MDS for all participants** — as **a**, every
participant; cross-task category-centroid alignment is significant in 5/6 (LH /
NUEx038 n.s.).

**S2, Semantic-organization PCA for all participants** — as **S1**, but the
shared 2D space is PCA fit on both tasks' predicted embeddings jointly (instead of
cosine-MDS). The cross-task alignment values are identical (computed on the 300-D
predicted vectors, independent of the 2D projection).

**S1 (3D) / S2 (3D)** (`*_all_3d`) — the same MDS and PCA maps reduced to **three**
components instead of two (each reducer still fit on both tasks jointly), one 3D
scatter per task with shared x/y/z limits. PCA is nested, so its first two axes
match S2; the 3-component MDS is a separate fit (MDS is not nested).

**S3, Region (ROI) knockout importance for all participants** — Δ
category-independent accuracy when a whole `primary_roi` is jointly shuffled,
picture vs auditory, regions ordered by picture Δacc; dashed lines mark each
participant's whole-brain knockout ceiling. All six participants now have an ROI
atlas, so every panel is populated (no placeholder).

**S4, Region-total VIP for all participants** — plain-PLS Variable-Importance-in-
Projection summed over each region's feature columns, from a linear PLS fit on all
pooled trials. A fast linear complement to the kernel-PLS permutation knockout in
S3; regions ordered by VIP.

**S7, Cross-task representational similarity** — Spearman correlation of the
per-word neural RDM between tasks (pic ↔ aud) and of each task with the GloVe RDM.
Both tasks track GloVe while their mutual neural geometry is weak/inconsistent —
consistent with cross-task decoding failing yet co-training succeeding.

_Display IDs (NUEx###) map to internal initials in participants.json. Source data:
`figures_for_paper/cross_task/source_data/`; regenerate with
`compute_cross_task_data.py` then `cross_task_panels.py`._
