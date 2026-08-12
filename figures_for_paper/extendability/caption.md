# Figure caption — Extendability of the regression-and-retrieval decoder

Extendability of the regression-and-retrieval decoder (picture naming; 14 participants).
The kernel-PLS decoder (Nystroem RBF kernel followed by PLS regression onto GloVe word-embedding
targets) predicts an embedding per trial; the predicted vector is ranked by cosine similarity
against an open word gallery of 5000 words (the stimulus words plus POS- and frequency-matched
distractors never presented to any participant), and the rank of the true word is the score.
Chance: median percentile rank 0.5; top-k accuracy k/N. **a** Median percentile rank (rank/N;
lower is better) versus gallery size N (200–5000 words); box, interquartile range and median across
participants; coloured points, individual participants; bold black, across-participant mean; stars,
Wilcoxon signed-rank versus chance per N. **b** Top-k retrieval accuracy versus rank k at N=5000
(same box/points/mean convention; log y; dashed, chance k/N; stars, Wilcoxon versus chance per k).
**c** Median percentile rank at N=5000 for words seen in training (in-vocab) versus words held
entirely out of training (held-out, zero-shot; 30% of unique words held out per cross-validation
split); box + points across participants; bracket, paired Wilcoxon (in-vocab versus held-out).
**d** Wu–Palmer WordNet similarity between the true word and its top-10 retrieved neighbours, for a
matched random-draw null versus the neural retrieval (WordNet grade is independent of the GloVe
decode target). **e** nDCG@100 of the neural ranking versus the same matched permutation null
(whole-list semantic organisation under the independent grade). In **d**,**e**: box + points across
participants; bracket, group Wilcoxon of the observed-minus-null difference. **f** Two-dimensional
MDS (cosine) of a best-participant semantic-neighbourhood showcase: for several stimulus words of
diverse semantic category, the predicted word (blue, bold; the top-retrieved gallery word at its own
embedding) is shown beside the ground-truth word (black, bold) and their nearest gallery neighbours
(grey); predictions land on the true word and its near-synonyms. In **a**–**e**: box, interquartile
range and median across participants; coloured points, individual participants (one fixed colour per
participant); bold black, across-participant mean; dashed grey, chance. Group tests are Wilcoxon
signed-rank (see Results). Participants identified by display ID (NUE###). **a**–**f** N=14.
Supplements: S1, per-participant held-out per-trial percentile distributions across N; S2,
qualitative best-case retrievals; S3–S4, semantic-neighbourhood showcases for three further
participants (S3: NUE031, NUE041; S4: NUE036).
