# Results — Decoded picture-naming information reflects linguistic rather than visual structure

*Figure: `main/figures_for_paper/language_vs_visual/` (panels a–e). Numbers are emitted to
`source_data/group_inference.csv`; decoding is held-out (cross-validated test trials), N = 12
picture-naming participants.*

---

A central concern for any semantic decoder operating on a picture-naming task is whether it reads
out the meaning of the intended word or merely the visual properties of the stimulus. We addressed
this directly by regressing high-gamma activity onto two families of embedding model that are, by
construction, blind to one another's information: language-only distributional models (GloVe,
Word2Vec), trained purely on lexical co-occurrence, and vision-only self-supervised models with
different architectures (DINOv3, a vision transformer; MoCo, a convolutional network), trained
purely on images with no text or labels under a self-supervised paradigm. The two families are
genuinely distinct representations of the same stimuli: a Procrustes analysis of the models'
embedding spaces showed high within-family but low cross-family similarity — the two language models
aligned at 0.73 and the two vision models at 0.51, whereas every language–vision pair aligned at only
0.29–0.44 (Procrustes similarity on the shared stimulus set; Fig. a).

Both families decoded semantic category above chance and followed a similar time course, rising after
picture onset and peaking ~1.1 s later; but the language family carried more information throughout.
The advantage was strongest and most sustained in variance explained: the language−vision R²
difference was significant across 36 bins spanning ~0.3–3.9 s (per-bin linear mixed model,
participant random intercept, BH-FDR q < 0.05; Fig. b). The category-accuracy difference between the
language and vision families was likewise significantly positive across 20 time bins, concentrated
0.4–1.5 s after picture onset (standalone panel `03_category_timecourse`, no longer part of the
combined figure — see `caption.md`).

At the moment semantic decoding peaked (~1.1 s), a pairwise model comparison confirmed the
dissociation across metrics (one-sided Wilcoxon signed-rank, language > vision, BH-FDR over the 12
pair×metric tests; Fig. c). In variance explained, every language model exceeded every vision model:
GloVe and Word2Vec each beat both DINOv3 and MoCo (Δ R² = 0.009–0.020, all q < 0.01, 11–12 of 12
participants). In category accuracy, both language models beat the transformer DINOv3
(GloVe > DINOv3 q < 0.01, Word2Vec > DINOv3 q < 0.05) and GloVe also beat MoCo (q < 0.05); only
Word2Vec vs. the convolutional MoCo — the strongest vision competitor — was non-significant.
Word-level decoding did not separate the families in any pairwise comparison.

The effect was consistent across the full cohort. Category decoding favored language over vision
embeddings in 11 of 12 participants (one-sided Wilcoxon signed-rank p = 7.3 × 10⁻⁴; mean Δ = 0.006),
while the weaker word-level decoding favored language in 8 of 12 (p = 0.088; Fig. d). Finally, the
language advantage was not an artifact of using the vision models' default pooled representation:
category and word decoding rose steadily from early to late layers of both vision networks (e.g.
DINOv3 category accuracy 0.16 → 0.29 from layer 1 to 12; MoCo 0.23 → 0.28 to layer 5) — consistent
with later layers encoding progressively higher-level, more semantic content and earlier layers
low-level shape, colour and contour — yet the best-decoding layer of each still reached only the
language decoder's level, not beyond it (best category accuracy DINOv3 0.290, MoCo 0.282, vs. the
language reference 0.296; Fig. e).

Decoded picture-naming activity in temporoparietal cortex therefore reflects linguistic-semantic
structure more than visual-perceptual structure, consistent with read-out from a post-perceptual,
lexical-semantic stage rather than from visual processing of the stimulus.
