# Results — Extendability of the *co-trained* regression-and-retrieval decoder

*(Figure references "Fig. X" / "Fig. Y" are placeholders for the co-trained
extendability figures, `00_extendability_combined_picture` and
`00_extendability_combined_auditory`, panels a–f, and the picture-vs-auditory
`00_extendability_combined_comparison`. Numbers are reproduced by
`extendability_panels.py` from `figures_for_paper/extendability_co_trained/source_data/`.
A single decoder is co-trained on pooled picture- and auditory-naming trials and
evaluated on each task separately; N = 6 participants with both tasks:
NUEx041 (AA), NUEx044 (AZ), NUEx045 (DR), NUEx038 (LH), NUEx031 (RB), NUEx036 (WBH).)*

## One decoder, open-vocabulary retrieval in both modalities

A single kernel-PLS decoder was co-trained on the pooled picture- and auditory-naming
trials of each participant, using the electrodes common to both tasks, and its per-trial
predicted GloVe vector was ranked by cosine similarity against an open gallery of up to
5,000 words (the stimulus words plus part-of-speech– and frequency-matched distractors
never presented). Predictions were out-of-fold; 30% of the unique words *across both
tasks* were held entirely out of training in either modality, so a held-out (zero-shot)
word was never seen by the decoder in picture *or* auditory naming, whereas an in-vocab
word may have been seen only in the other modality. Chance median percentile rank is 0.5;
chance top-*k* accuracy is *k*/*N*. We report the two tasks in turn, then the direct
comparison.

**Picture naming.** Against the full 5,000-word gallery, the co-trained decoder placed the
true word in the top 1.7% of the ranked list (group median percentile rank 0.017;
per-participant 0.012–0.032; Wilcoxon signed-rank against chance *p* = 0.016, the minimum
attainable at *n* = 6; within-participant permutation *p* < 0.05 in all six). Its median
rank was 61–159 of 5,000. The true word was the single top choice on 1.4–5.6% of trials
(chance 0.02%), fell within the top 10 on 17.4 ± 1.5% of trials (mean ± s.e.m.; chance
0.2%) and within the top 100 on 51.5 ± 2.5% (chance 2%; Fig. Xa,b). Retrieval declined only
gradually as the gallery grew: enlarging it 25-fold (200 → 5,000 words) lowered the group
median percentile rank from 0.143 to 0.017 (Fig. Xa).

Words held entirely out of training were still retrieved far above chance: group median
percentile rank 0.030 for held-out versus 0.014 for in-vocab words (both *p* = 0.016 against
chance; per-participant held-out 0.022–0.074; Fig. Xc; 1,131 held-out trials). When the true
word was not ranked first the retrieved words lay near it in meaning: graded against
WordNet Wu–Palmer similarity (a hierarchy independent of the GloVe decode target), the
neural ranking reached mean nDCG@100 = 0.655 (per-participant 0.639–0.669), exceeding a
matched permutation null in every participant (all six within-participant permutation
*p* < 0.05; group Wilcoxon *p* = 0.016; Fig. Xe); the mean Wu–Palmer similarity of the
top-10 neighbours likewise exceeded its matched null (observed 0.568–0.604 vs null
0.532–0.568; significant within-participant in five of six; group *p* = 0.016; Fig. Xd).
Projecting a well-decoded participant's predictions into two dimensions (cosine MDS;
NUEx041) makes this concrete — across diverse categories the predicted word lands on the
ground-truth word and its near-synonyms (e.g. *bear*→*deer*, *orange*→*strawberry*,
*apple*→*banana*, *cat*→*cow*; Fig. Xf). Per-trial percentile rank was only weakly related to
word frequency (|*r*| = 0.07–0.23).

**Auditory naming.** The same co-trained decoder decoded auditory-naming trials open-
vocabulary as well — a regime in which the auditory-only decoder is essentially at chance,
because auditory naming has few trials and few repeated words. Against the 5,000-word
gallery the true word fell in the top 3.0% of the list (group median percentile rank 0.030;
per-participant 0.019–0.041; *p* = 0.016 against chance; within-participant permutation
*p* < 0.05 in five of six). Its median rank was 93–206 of 5,000; the true word was top-1 on
0–3.9% of trials, top-10 on 10.4 ± 2.3% and top-100 on 44.2 ± 2.9% (Fig. Ya,b), and
retrieval again degraded gracefully with gallery size (group median percentile rank
0.203 → 0.029 from N = 200 → 5,000; Fig. Ya).

The seen/unseen split exposes the source of this generalisation. In-vocab auditory words —
those the decoder had encountered in training, in most cases *only through picture naming* —
were retrieved very accurately (group median percentile rank 0.019; per-participant
0.010–0.033), whereas words held out of both modalities (genuinely zero-shot) were retrieved
less well but still far above chance (group 0.068; per-participant 0.017–0.111; both
*p* = 0.016; Fig. Yc; 259 held-out trials). Thus most of the auditory decoding rides on
lexical structure the model learned in the other modality — the co-training payoff. Retrieved
auditory neighbours were also semantically related (top-10 Wu–Palmer observed 0.503–0.591 vs
matched null 0.433–0.565; significant within-participant in four of six; group Wilcoxon
*p* = 0.016; Fig. Yd). Whole-list organisation was weaker and less consistent than for
picture naming: nDCG@100 exceeded its matched null in only two of six participants
individually, and the group difference was a non-significant trend (mean nDCG 0.628; group
Wilcoxon *p* = 0.078; Fig. Ye) — expected given the small auditory test sets. Frequency
coupling was modest in most participants but larger in the smallest-sample cases
(|*r*| = 0.06–0.59), a caveat for the auditory numbers.

**Picture versus auditory.** Juxtaposing the two evaluations of the one decoder (Fig. Z,
panels a–e) shows picture-naming retrieval to be the stronger of the two, but only by a
margin that narrows as the gallery grows. Picture beat auditory in paired within-participant
tests at every smaller gallery size (median percentile rank at N = 200–2,000, all paired
Wilcoxon *p* = 0.031) and at every top-*k* through k = 50 (top-10 17.4% vs 10.4%; paired
*p* = 0.016), but the difference was no longer significant at the full 5,000-word gallery
(median percentile rank 0.020 vs 0.030, *p* = 0.078) or at top-100 (*p* = 0.078). Throughout,
auditory retrieval stayed far above chance and shared the picture task's qualitative
signature — graceful scaling with gallery size, in-vocab < held-out < chance, and
semantically related near-misses. The headline result is that a *single* co-trained decoder
extends open-vocabulary, zero-shot word retrieval to both speech modalities, with the
auditory side decoding largely by reusing lexical structure learned during picture naming.

*Caveats.* The co-trained decoder is restricted to the six participants with both tasks and
to the electrodes common to the two runs, so its picture-naming numbers are not directly
comparable to the 12-participant, all-channel picture-only figure. Auditory test sets are
small; the auditory panels are correspondingly noisier and the whole-list nDCG effect does
not reach significance. Per-participant held-out distributions across gallery size
(Supplementary Fig. S1), qualitative best-case retrievals (Supplementary Table S2) and
additional semantic-neighbourhood showcases (Supplementary Figs. S3–S4) are provided for
each task.
