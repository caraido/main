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
of the within-task ceiling. **c, Task-general electrodes.** Per-electrode
Variable-Importance-in-Projection (VIP) of the co-trained model for one
participant (NUEx031), ranked; red above the average-importance threshold
(VIP = 1); the four highest are labelled (V2, V3, V4, T1). Bar plot: fraction of
electrodes with VIP > 1 per participant (dotted line 1/6). n = 6 participants
(both picture and auditory naming). No trial resampling; chance 0.167 (six
categories).

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

**S3, VIP electrode importance for all participants** — ranked VIP from the
co-trained model; dashed line VIP = 1; red above average. 29–44 % of electrodes
exceed average importance (mean 36 %).

**S4, Per-channel permutation importance for all participants** — ΔΔ
category-independent accuracy when each electrode's history block is shuffled,
picture vs auditory; the strongest few electrodes are labelled. Under the
Nystroem-RBF map single-channel effects are diluted, so no electrode reaches
BH-FDR significance (all grey), yet the strongest sites recur (e.g. NUEx041 T4,
NUEx044 S3, NUEx031 V2/V3, NUEx036 PC13).

**S5, Region (ROI) knockout importance** — Δ category-independent accuracy when a
whole `primary_roi` is jointly shuffled, picture vs auditory. Available for the
four participants with an ROI atlas (NUEx041/044/038/036); NUEx045 and NUEx031
have no atlas (placeholder).

**S7, Cross-task representational similarity** — Spearman correlation of the
per-word neural RDM between tasks (pic ↔ aud) and of each task with the GloVe RDM.
Both tasks track GloVe while their mutual neural geometry is weak/inconsistent —
consistent with cross-task decoding failing yet co-training succeeding.

_Display IDs (NUEx###) map to internal initials in participants.json. Source data:
`figures_for_paper/cross_task/source_data/`; regenerate with
`compute_cross_task_data.py` then `cross_task_panels.py`._
