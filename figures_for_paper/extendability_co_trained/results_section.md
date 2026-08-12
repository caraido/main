# Results — Extendability of the *co-trained* regression-and-retrieval decoder

*(Figure references "Fig. X" / "Fig. Y" are placeholders for the co-trained
extendability figures, `00_extendability_combined_picture` and
`00_extendability_combined_auditory`, panels a–f, and the picture-vs-auditory
`00_extendability_combined_comparison`. Numbers are reproduced by
`extendability_panels.py` from `figures_for_paper/extendability_co_trained/source_data/`.
A single decoder is co-trained on pooled picture- and auditory-naming trials and
evaluated on each task separately; N = 7 participants with both tasks:
NUE041 (AA), NUE044 (AZ), NUE030 (CP), NUE045 (DR), NUE038 (LH), NUE031 (RB),
NUE036 (WBH). NUE030 joined the auditory cohort on 2026-07-28; all numbers below
were regenerated against the 7-participant auditory run and differ from the earlier
6-participant draft.)*

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
per-participant 0.013–0.056; Wilcoxon signed-rank against chance *p* = 0.0078, the minimum
attainable at *n* = 7; within-participant permutation *p* < 0.05 in all seven). Its median
rank was 64–278 of 5,000. The true word was the single top choice on 1.4–5.3% of trials
(chance 0.02%), fell within the top 10 on 15.0 ± 1.8% of trials (mean ± s.e.m.; chance
0.2%) and within the top 100 on 47.3 ± 4.0% (chance 2%; Fig. Xa,b). Retrieval declined only
gradually as the gallery grew: enlarging it 25-fold (200 → 5,000 words) lowered the group
median percentile rank from 0.169 to 0.027 (Fig. Xa).

Words held entirely out of training were still retrieved far above chance: group median
percentile rank 0.034 for held-out versus 0.014 for in-vocab words (both *p* = 0.0078 against
chance; per-participant held-out 0.022–0.079; Fig. Xc; 1,178 held-out trials). When the true
word was not ranked first the retrieved words lay near it in meaning: graded against
WordNet Wu–Palmer similarity (a hierarchy independent of the GloVe decode target), the
neural ranking reached mean nDCG@100 = 0.650 (per-participant 0.619–0.670), exceeding a
matched permutation null at the group level (Wilcoxon *p* = 0.0078) though within-participant
only in five of seven — with NUE030 added, this is no longer significant in every
participant (Fig. Xe); the mean Wu–Palmer similarity of the
top-10 neighbours likewise exceeded its matched null (observed 0.535–0.603 vs null
0.525–0.571; significant within-participant in five of seven; group *p* = 0.0078; Fig. Xd).
Projecting a well-decoded participant's predictions into two dimensions (cosine MDS;
NUE041) makes this concrete — across diverse categories the predicted word lands on the
ground-truth word and its near-synonyms (e.g. *bear*→*deer*, *orange*→*strawberry*,
*apple*→*banana*, *cat*→*cow*; Fig. Xf). Per-trial percentile rank was only weakly related to
word frequency (|*r*| = 0.07–0.23).

**Auditory naming.** The same co-trained decoder decoded auditory-naming trials open-
vocabulary as well — a regime in which the auditory-only decoder is essentially at chance,
because auditory naming has few trials and few repeated words. Against the 5,000-word
gallery the true word fell in the top 3.3% of the list (group median percentile rank 0.033;
per-participant 0.019–0.059; *p* = 0.0078 against chance; within-participant permutation
*p* < 0.05 in four of seven). Its median rank was 93–293 of 5,000; the true word was top-1 on
0.9–3.9% of trials, top-10 on 12.0 ± 1.9% and top-100 on 40.5 ± 3.3% (Fig. Ya,b), and
retrieval again degraded gracefully with gallery size (group median percentile rank
0.227 → 0.035 from N = 200 → 5,000; Fig. Ya).

The seen/unseen split exposes the source of this generalisation. In-vocab auditory words —
those the decoder had encountered in training, in most cases *only through picture naming* —
were retrieved very accurately (group median percentile rank 0.020; per-participant
0.013–0.045), whereas words held out of both modalities (genuinely zero-shot) were retrieved
less well but still far above chance (group 0.052; per-participant 0.024–0.122; both
*p* = 0.0078; Fig. Yc; 265 held-out trials). Thus most of the auditory decoding rides on
lexical structure the model learned in the other modality — the co-training payoff. Retrieved
auditory neighbours were also semantically related (top-10 Wu–Palmer observed 0.494–0.611 vs
matched null 0.418–0.566; significant within-participant in six of seven; group Wilcoxon
*p* = 0.0078; Fig. Yd). Whole-list organisation remains weaker and far less consistent than
for picture naming: nDCG@100 exceeded its matched null in only two of seven participants
individually. The group test is now significant (mean nDCG 0.625; group Wilcoxon
*p* = 0.0078; Fig. Ye) where it was previously a non-significant trend, but that reflects all
seven participants falling on the same side of their null rather than any individual effect
being strong — with only two of seven significant on their own, the group result should not
be read as evidence of reliable within-participant whole-list organisation. Frequency
coupling was modest in most participants but larger in the smallest-sample cases
(|*r*| = 0.00–0.66), a caveat for the auditory numbers.

**Picture versus auditory.** Juxtaposing the two evaluations of the one decoder (Fig. Z,
panels a–e) shows picture-naming retrieval to be the stronger of the two, but only by a
margin that narrows as the gallery grows. Picture beat auditory in paired within-participant
tests at every gallery size (median percentile rank at N = 200–5,000, paired Wilcoxon
*p* = 0.0078–0.039) and at every top-*k* from k = 5 through k = 100 (top-10 15.0% vs 12.0%;
paired *p* = 0.023). With the seventh participant added, picture now also beats auditory at
the full 5,000-word gallery (0.027 vs 0.035, *p* = 0.039) and at top-100 (*p* = 0.039), where
the six-participant analysis had found no significant difference; only top-1 remains
non-significant (*p* = 0.055). Throughout,
auditory retrieval stayed far above chance and shared the picture task's qualitative
signature — graceful scaling with gallery size, in-vocab < held-out < chance, and
semantically related near-misses. The headline result is that a *single* co-trained decoder
extends open-vocabulary, zero-shot word retrieval to both speech modalities, with the
auditory side decoding largely by reusing lexical structure learned during picture naming.

*Caveats.* The co-trained decoder is restricted to the seven participants with both tasks and
to the electrodes common to the two runs, so its picture-naming numbers are not directly
comparable to the 12-participant, all-channel picture-only figure. Auditory test sets are
small; the auditory panels are correspondingly noisier, and although the whole-list nDCG
effect is now significant at the group level it holds within only two of seven participants
individually. Two participants (NUE030, NUE031) ran an earlier auditory stimulus set with
longer prompts and a different category inventory, so the auditory cohort is not homogeneous. Per-participant held-out distributions across gallery size
(Supplementary Fig. S1), qualitative best-case retrievals (Supplementary Table S2) and
additional semantic-neighbourhood showcases (Supplementary Figs. S3–S4) are provided for
each task.
