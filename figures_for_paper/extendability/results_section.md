# Results — Extendability of the regression-and-retrieval decoder

*(Figure reference "Fig. X" is a placeholder for the extendability figure,
`00_extendability_combined`; panels a–f. Every number here is reproduced by
`extendability_panels.py` from `figures/open_vocab_retrieval/source_data/` and is
mirrored in this folder's `source_data/`. **Picture naming, N = 14 participants.**
Regenerated 2026-08-12 after CP was retired — `docs/experiments/015-retiring-cp.md`.
The one-sided Wilcoxon floor is therefore 1/2¹⁴ = 6.1 × 10⁻⁵, not the 2.4 × 10⁻⁴ of the
N = 12 draft.)*

## Open-vocabulary and zero-shot retrieval

Because the decoder predicts a point in a continuous word-embedding space rather than
selecting among a fixed set of labels, its predictions can be matched against an arbitrary
word list without refitting. We tested this by ranking each trial's predicted GloVe vector,
by cosine similarity, against an open gallery that contained the stimulus words together
with up to 5,000 additional words never presented to any participant (part-of-speech– and
frequency-matched distractors); the rank of the true word among the *N* gallery words
indexes retrieval (chance median percentile rank 0.5; chance top-*k* accuracy *k*/*N*).
Cross-validation held 30% of the unique words entirely out of training, so held-out trials
test retrieval of words the decoder was never fit to produce (zero-shot).

**Open-vocabulary retrieval.** Against the full 5,000-word gallery, the true word fell in
the top 2.3% of the ranked list (group median percentile rank 0.023; per-participant
0.010–0.035; Wilcoxon signed-rank against chance *p* = 6.1 × 10⁻⁵, the minimum attainable at
*n* = 14; within-participant permutation *p* < 0.001 in **every** participant). Its median
rank was 52–174 of 5,000. The true word was the single top choice on 0.6–7.1% of trials
(chance 0.02%), fell within the top 10 on 16.9 ± 1.4% of trials (mean ± s.e.m.; chance 0.2%),
and within the top 100 on 47.7 ± 1.8% (chance 2%; Fig. Xa,b). Retrieval declined only
gradually as the gallery grew: expanding it 25-fold (200 → 5,000 words) lowered the group
median percentile rank from 0.156 to 0.023 — the true word's median rank rose only ≈ 3-fold
(e.g. NUE041, 17 → 52) against the 25-fold increase in distractors — and top-10 accuracy fell
only from 0.252 to 0.169 (Fig. Xa). Results were essentially identical for a raw, unmatched
gallery (median percentile rank at *N* = 5,000, 0.023 matched vs 0.023 raw).

**Zero-shot retrieval of untrained words.** Words held entirely out of training were still
retrieved far above chance. The group median percentile rank was 0.036 for held-out words
(95% CI 0.033–0.042) versus 0.017 for in-vocabulary words (95% CI 0.014–0.023), both
*p* = 6.1 × 10⁻⁵ against chance (Fig. Xc). Every participant placed held-out words well below
chance (per-participant 0.021–0.058; top 2.1–5.8% of the 5,000-word gallery) and above their
own in-vocabulary level; across participants the seen-to-unseen change in median percentile
rank was roughly twofold (0.017 → 0.036) rather than a return toward chance (2,354 held-out
trials in total). The full per-trial held-out distribution and its graceful widening with
gallery size are shown per participant in Supplementary Fig. S1.

**Retrieved neighbours are semantically related.** When the true word was not ranked first,
the retrieved words lay near it in meaning. Graded against WordNet Wu–Palmer similarity — a
semantic hierarchy independent of the GloVe decode target — the neural ranking reached a mean
nDCG@100 of 0.644 (95% CI 0.639–0.651). The mean Wu–Palmer similarity of the top-10 retrieved
neighbours to the true word exceeded a matched random-draw null in **every** participant
(observed 0.540–0.599 vs null 0.518–0.562; median difference +0.020), significantly so in
**all fourteen** (within-participant permutation *p* ≤ 0.05, largest *p* = 0.050), with an
across-participant difference of +0.020 (Wilcoxon *p* = 6.1 × 10⁻⁵; Fig. Xd). The same neural
ranking also organised the whole retrieved list semantically: its nDCG@100 exceeded a matched
permutation null in every participant (observed 0.629–0.672 vs null 0.600–0.646; median
difference +0.020), significantly within-participant in **all fourteen** (largest
*p* = 0.015) and across participants (Wilcoxon *p* = 6.1 × 10⁻⁵; Fig. Xe).
Projecting a well-decoded participant's predictions into two dimensions (cosine MDS; NUE027)
makes this concrete: across diverse semantic categories the predicted embeddings land on the
ground-truth word and its near-synonyms — e.g. *mango*→*peach*, *cat*→*deer*, *spring*→*fall*
(Fig. Xf). Per-trial percentile rank was only weakly related to word frequency
(|*r*| = 0.01–0.19 across participants), indicating that retrieval was not explained by
frequency. Illustrative best-case retrievals per participant are tabulated in Supplementary
Table S2.

---

## Note on the change from N = 12 / N = 15

This section previously reported N = 12, and briefly N = 15. It is now N = 14. Three claims
changed in kind rather than in decimal, and all three moved the same way — **the retired
participant was the floor on every open-vocabulary metric**:

| | with CP (N = 15) | without CP (N = 14) |
|---|---|---|
| single top choice | 0–7.1% of trials | **0.6**–7.1% |
| worst median rank | 313 of 5,000 | **174** |
| within-participant permutation | *p* ≤ 0.022; < 0.001 in 14/15 | **< 0.001 in 14/14** |
| near-miss null exceeded | significant in 14/15 | **14/14** |
| nDCG null exceeded | significant in 14/15 | **14/14** |

The "one participant where it was not significant" sentences are therefore gone — not
renumbered. That participant was CP in every case. The effect did not get stronger; the
weakest participant was removed, which is a fact about the cohort and should be read as
such.
