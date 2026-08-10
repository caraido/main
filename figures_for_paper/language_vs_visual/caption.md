# Figure caption — Decoded picture-naming information reflects linguistic rather than visual structure

Decoded picture-naming high-gamma activity reflects linguistic rather than visual structure
(N = 15). High-gamma activity was regressed onto two families of word-referent embedding that are
blind to one another by construction: a **language** family — GloVe and Word2Vec, trained only on
lexical co-occurrence — and a **vision** family — DINOv3 (vision transformer) and MoCo
(convolutional network), trained only on images by self-supervision. Kernel-PLS decoder
(Nystroem RBF → PLS); all accuracies are held-out (cross-validated test trials).
**a** Procrustes similarity between the four embedding spaces on the shared stimulus set
(1 − Procrustes disparity of the top-10-PC concept geometries; higher = more similar); black lines
separate the language from the vision block. **b** Variance explained (R² − shuffled chance) for the
language vs. vision family (mean ± s.e.m.); bars below zero mark bins where language > vision (per-bin
linear mixed model, participant random intercept, BH-FDR q < 0.05; pre-onset bins not tested).
Vertical lines/bands: cue times (Go cue, voice onset, voice offset) mean ± 1 s.d. across participants.
**c** As **b** for category effect (balanced accuracy − shuffled chance). **d** At the semantic
peak bin (group category-accuracy peak, ~1.1 s), pairwise difference Δ = language − vision for each
language>vision model pair, in R² (left), category (middle) and word (right) decoding; bar =
group mean, error = s.e.m., dots = participants (jittered); stars = one-sided Wilcoxon signed-rank
(language > vision), BH-FDR over the 12 pair×metric tests (\*\*\* q<0.001, \*\* q<0.01, \* q<0.05,
n.s. q≥0.05). **e** Between-participant difference Δ = language − vision in post-stimulus mean
accuracy, ranked per participant for category and word; blue = favours language, red = favours
vision. **f** Category (left) and word (right) accuracy of DINOv3 and MoCo across layer depth
(1-indexed; mean ± s.e.m. over participants); dashed line = language decoder (GloVe, Word2Vec) pooled
peak accuracy ± s.e.m. Dotted vertical line: picture onset (0 s). Participants are identified by
display ID (NUEx###). N = 15, except the final four time bins (4.1-4.4 s), where N = 14: one
participant's recording window ends at 4.0 s. Per-bin participant counts are in the `count`
column of `source_data/panel_b_category_timecourse.csv`.

Channels are restricted to the 13-region temporal-parietal whitelist applied to `nmm_roi`
(633 of 1,360 recorded contacts across the cohort), and the feature window is 500 ms
(5 x 100 ms bins). Numbers are therefore not comparable with the pre-2026-08 whole-brain,
1000 ms version of this figure.
