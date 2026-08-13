# Results — Decoded picture-naming information reflects linguistic rather than visual structure

*Figure: `main/figures_for_paper/language_vs_visual/` (panels a–e). Numbers are emitted to
`source_data/group_inference.csv`; decoding is held-out (cross-validated test trials), **N = 14**
picture-naming participants. Regenerated 2026-08-12 after CP was retired
(`docs/experiments/015-retiring-cp.md`); the one-sided Wilcoxon floor is 1/2¹⁴ = 6.1 × 10⁻⁵.*

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
aligned at 0.735 and the two vision models at 0.512, whereas every language–vision pair aligned at
only 0.285–0.439 (Procrustes similarity on the shared stimulus set; Fig. a).

Both families decoded semantic category above chance and followed a similar time course, rising after
picture onset and peaking ~1.1 s later; but the language family carried more information throughout.
The advantage was strongest and most sustained in variance explained: the language−vision R²
difference was significant across 37 bins spanning ~0.7–4.4 s (per-bin linear mixed model,
participant random intercept, BH-FDR q < 0.05; Fig. b). The category-accuracy difference between the
language and vision families was likewise significantly positive, though across only 4 time bins,
concentrated 1.0–4.3 s after picture onset (standalone panel `03_category_timecourse`, no longer part of the
combined figure — see `caption.md`).

At the moment semantic decoding peaked (~1.1 s), a pairwise model comparison confirmed the
dissociation across metrics (one-sided Wilcoxon signed-rank, language > vision, BH-FDR over the 12
pair×metric tests; Fig. c). In variance explained, every language model exceeded every vision model:
GloVe and Word2Vec each beat both DINOv3 and MoCo (Δ R² = 0.011–0.024, all q < 0.001, **14 of 14
participants in every one of the four comparisons**). In category accuracy, both language models
beat the transformer DINOv3 (GloVe > DINOv3 q < 0.01, 12/14; Word2Vec > DINOv3 q < 0.05, 11/14);
neither beat the convolutional MoCo — the strongest vision competitor — after correction
(GloVe > MoCo q = 0.19, 7/14; Word2Vec > MoCo q = 0.51, 7/14). Word-level decoding did not
separate the families in any pairwise comparison (all q ≥ 0.20).

The effect was consistent across the full cohort. Category decoding favored language over vision
embeddings in 13 of 14 participants (one-sided Wilcoxon signed-rank p = 6.1 × 10⁻⁴; mean Δ = 0.0030),
while the weaker word-level decoding favored language in 11 of 14 (p = 8.3 × 10⁻³; Fig. d). Finally, the
language advantage was not an artifact of using the vision models' default pooled representation:
category and word decoding rose steadily from early to late layers of both vision networks (e.g.
DINOv3 category accuracy 0.162 → 0.274 from layer 0 to 12; MoCo 0.223 → 0.282 to layer 4) — consistent
with later layers encoding progressively higher-level, more semantic content and earlier layers
low-level shape, colour and contour — yet the best-decoding layer of each still reached only the
language decoder's level, not beyond it (best category accuracy DINOv3 0.290, MoCo 0.282, vs. the
language reference 0.292 ± 0.009; Fig. e).

Decoded picture-naming activity in temporoparietal cortex therefore reflects linguistic-semantic
structure more than visual-perceptual structure, consistent with read-out from a post-perceptual,
lexical-semantic stage rather than from visual processing of the stimulus.

---

## Note on the change from N = 12

This section reported N = 12 until 2026-08-12. It is now N = 14, and the numbers come from
`source_data/group_inference.csv` as regenerated on that date. The gap spans more than CP's
retirement: the N = 12 text described the **whole-brain, 1000 ms** analysis, whereas the
current figure is **NMM-gated (13 regions), 500 ms**. Numbers from the two are not
comparable, so this is a rewrite against current data rather than a renumbering.

**Two claims weakened, and both should be read as weaker rather than smoothed over:**

- **Category accuracy, language > vision, per-bin.** Significant across **4** bins
  (1.0–4.3 s), against 20 bins (0.4–1.5 s) in the N = 12 text. The R² contrast is
  unaffected and if anything stronger (37 bins, 0.7–4.4 s, 14/14 participants on all four
  model pairs).
- **GloVe > MoCo in category accuracy has reversed.** It was reported significant
  (q < 0.05); it is now **q = 0.19, 7/14 participants — not significant**. Neither language
  model now beats MoCo after correction. MoCo, a convolutional network, remains the
  strongest vision competitor, and the honest statement is that the language advantage in
  *category accuracy* rests on the comparison against DINOv3, not against MoCo.

The cohort-level claim strengthened slightly (13/14 for category, 11/14 for word, against
11/12 and 8/12), and the word-level comparison remains null in every pairwise test.

The overall conclusion — that decoded picture-naming activity reflects linguistic rather
than visual structure — still rests on the R² contrast, which is unambiguous. It no longer
rests on category accuracy against MoCo.
