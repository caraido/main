# Figure caption — Extendability of the CO-TRAINED regression-and-retrieval decoder

Extendability of a single decoder co-trained on pooled picture- and auditory-naming trials,
evaluated on auditory naming trials (7 participants with both tasks: NUEx041, NUEx044, NUEx030, NUEx045, NUEx038, NUEx031, NUEx036).
The kernel-PLS decoder (Nystroem RBF kernel followed by PLS regression onto
GloVe word-embedding targets) is fit on the intersection of the two tasks' electrodes and predicts
an embedding per trial; the predicted vector is ranked by cosine similarity against an open word
gallery of 5000 words (the stimulus words plus POS- and frequency-matched distractors never
presented), and the rank of the true word is the score. Predictions are out-of-fold; a fraction
of the unique words across BOTH tasks is held entirely out of training in either modality
(zero-shot), so an in-vocab word may have been seen only cross-modally. Chance: median percentile
rank 0.5; top-k accuracy k/N. **a** Median percentile rank (rank/N; lower is better) versus gallery
size N (200–5000 words); box, interquartile range and median across participants; coloured points,
participants; bold black, mean; stars, Wilcoxon versus chance per N. **b** Top-k retrieval accuracy
versus rank k at N=5000 (log y; dashed, chance k/N; stars, Wilcoxon versus chance per k).
**c** Median percentile rank at N=5000 for words seen in training (in-vocab) versus held entirely
out (zero-shot); bracket, paired Wilcoxon. **d** Wu–Palmer similarity between the true word and its
top-10 retrieved neighbours, matched null versus neural (WordNet grade independent of the GloVe
decode target). **e** nDCG@100 of the neural ranking versus the matched permutation null. In
**d**,**e**: bracket, group Wilcoxon of the observed-minus-null difference. **f** 2D MDS (cosine)
of a best-participant semantic-neighbourhood showcase: the predicted word (blue, bold; top-retrieved
gallery word at its own embedding) beside the ground-truth word (black, bold) and their nearest
neighbours (grey). In **a**–**e**: coloured points, participants (fixed colour each); bold black,
mean; dashed grey, chance. Group tests are Wilcoxon signed-rank. Participants identified by display
ID (NUEx###). N=7. Auditory naming has few trials and few repeated words, so auditory panels are
noisier and closer to chance — expected for the weaker modality. A companion comparison figure
juxtaposes picture- versus auditory-test performance for panels a–e. Supplements: S1, per-participant
held-out per-trial percentile distributions across N; S2, qualitative best-case retrievals; S3–S4,
semantic-neighbourhood showcases for further participants.
