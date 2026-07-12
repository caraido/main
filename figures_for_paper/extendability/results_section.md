# Results — Extendability of the regression-and-retrieval decoder

*(Figure reference "Fig. X" is a placeholder for the extendability figure,
`00_extendability_combined`; panels a–f. Numbers are reproduced by
`extendability_panels.py` from `figures/open_vocab_retrieval/source_data/`.
Picture naming, N = 12 participants.)*

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
the top 2.1% of the ranked list (group median percentile rank 0.021; per-participant
0.011–0.042; Wilcoxon signed-rank against chance *p* = 2.4 × 10⁻⁴, the minimum attainable at
*n* = 12; within-participant permutation *p* ≤ 0.01 in every participant, *p* < 0.001 in ten
of twelve). Its median rank was 56–212 of 5,000. The true word was the single top choice on
0.7–6.5% of trials (chance 0.02%), fell within the top 10 on 17.2 ± 1.6% of trials
(mean ± s.e.m.; chance 0.2%), and within the top 100 on 49.5 ± 2.1% (chance 2%; Fig. Xa,b).
Retrieval declined only gradually as the gallery grew: expanding it 25-fold (200 → 5,000
words) lowered the group median percentile rank from 0.154 to 0.022 — the true word's median
rank rose only ≈ 3-fold (e.g. NUEx041, 18 → 56) against the 25-fold increase in
distractors — and top-10 accuracy fell only from 0.255 to 0.172 (Fig. Xa). Results were
essentially identical for a raw, unmatched gallery (median percentile rank at *N* = 5,000,
0.022 for both).

**Zero-shot retrieval of untrained words.** Words held entirely out of training were still
retrieved far above chance. The group median percentile rank was 0.031 for held-out words
(95% CI 0.027–0.052) versus 0.016 for in-vocabulary words (95% CI 0.014–0.023), both
*p* = 2.4 × 10⁻⁴ against chance (Fig. Xc). Every participant placed held-out words well below
chance (per-participant 0.016–0.110; top 1.6–11.0% of the 5,000-word gallery) and above their
own in-vocabulary level; across participants the seen-to-unseen change in median percentile
rank was roughly twofold (0.016 → 0.031) rather than a return toward chance (1,840 held-out
trials in total). The full per-trial held-out distribution and its graceful widening with
gallery size are shown per participant in Supplementary Fig. S1.

**Retrieved neighbours are semantically related.** When the true word was not ranked first,
the retrieved words lay near it in meaning. Graded against WordNet Wu–Palmer similarity — a
semantic hierarchy independent of the GloVe decode target — the neural ranking reached a mean
nDCG@100 of 0.65 (95% CI 0.635–0.654; per-participant 0.603–0.675). The mean Wu–Palmer
similarity of the top-10 retrieved neighbours to the true word exceeded a matched random-draw
null in every participant (observed 0.507–0.599 vs null 0.497–0.561), significantly so in ten
(within-participant permutation *p* = 0.005); in the remaining two, NUEx030 and NUEx045, the
differences were small and non-significant (*p* = 0.11 and *p* = 0.42), with an
across-participant difference of +0.022 (Wilcoxon *p* = 2.4 × 10⁻⁴; Fig. Xd). The same neural
ranking also organised the whole retrieved list semantically: its nDCG@100 exceeded a matched
permutation null in every participant (observed 0.603–0.675 vs null 0.590–0.641; median
difference +0.018), significantly within-participant in eleven of twelve (the exception
NUEx045 at *p* = 0.055) and across participants (Wilcoxon *p* = 2.4 × 10⁻⁴; Fig. Xe).
Projecting a well-decoded participant's predictions into two dimensions (cosine MDS; NUEx027) makes
this concrete: across diverse semantic categories the predicted embeddings land on the
ground-truth word and its near-synonyms — e.g. *mango*→*peach*, *cat*→*deer*, *spring*→*fall*
(Fig. Xf). Per-trial percentile rank was only weakly related to word frequency
(|*r*| = 0.01–0.19 across participants), indicating that retrieval was not explained by
frequency. Illustrative best-case retrievals per participant are tabulated in Supplementary
Table S2.
