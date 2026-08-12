# Supplementary figure S5 — within-category (category-preserving) null

Caption for `S5_within_category_null.{png,pdf}`. The main figure's caption is `caption.md`,
beside this file.

---

Word-level retrieval against a category-preserving null, picture naming.

Bars: cohort mean ± SEM of retrieval accuracy at top-1, top-3 and top-5. Grey (`chance`):
the uniform-substitution null — the candidate word is drawn uniformly from that
participant's own stimulus vocabulary and its rank read off the same fixed ranking. Teal
(`category-only`): the same substitution restricted to the true word's semantic category, so
category structure is retained and sub-category word identity is destroyed; 10,000 draws per
participant. Amber (`category+word identity`): the observed decoder. Black dots: individual
participants, jittered horizontally; each participant's three dots share an x-offset.

Ranking is mean-centred cosine between a trial's predicted GloVe vector and the GloVe target
vector of each of that participant's unique stimulus words (kernel-PLS decoder; the
`utils.retrieval.mean_center_db` convention). All decoded trials are pooled, in-vocabulary
and held-out alike. No re-fitting and no re-ranking: a permutation substitutes a word and
reads its rank off the same fixed ranking.

Brackets: one-sided Wilcoxon signed-rank across participants on the per-participant
difference (observed − category-only), Holm-corrected over the three tests in this panel;
`*** p<0.001, ** p<0.01, * p<0.05`, `n.s.` otherwise. Star ladder from `utils.config.p_stars`,
cutoff from `utils.config.ALPHA`. Holm-adjusted p: top-1 p=0.010, top-3 p=0.011, top-5
p=0.010. Group excess (observed − category-only): +0.014 / +0.022 / +0.034. One-sided
Wilcoxon signed-rank floors at 1/2^n = 3.05e-5 at N=15, so after Holm over three tests the
smallest attainable adjusted p is 9.2e-5.

Stimulus vocabulary is 44–64 words per participant and 6–7 semantic categories, so uniform
chance is per participant and the grey bar is the mean of the per-participant nulls, not a
single 1/W.

Rendered by `within_category_null_panels.py` from
`source_data/within_category_null_topk.csv` and `source_data/within_category_null_group.csv`,
computed by `compute_within_category_null.py` from
`figures/open_vocab_retrieval/source_data/trial_predictions_picture_naming.csv` (`PIC_RUN`;
13-region temporal-parietal whitelist, 5 bins of history — the configuration behind panels
**a**–**d** of the main figure).

**Limitations.** The input CSV lives in a gitignored directory (see the untracked-inputs
table in `docs/repo_layout.md`), so this figure cannot be regenerated from a clean checkout
alone. Observed accuracy and the null are both averaged over trials, so a word produced on
many trials contributes many times; averaging both per unique word instead reduces the excess
from +0.014 / +0.022 / +0.034 to +0.006 / +0.010 / +0.019 (top-1 / top-3 / top-5). The
direction is unchanged and no significance test was re-run under that weighting.

Participants are identified by display ID (NUEx###). **N = 15** (picture naming).

---

## Not shipped: auditory naming

`compute_within_category_null.py` also computes the auditory arm (N=10), and its rows are
kept in both source CSVs with `shipped = False`. It is **not plotted and not part of this
figure**, pending discussion with the team. `within_category_null_panels.py --task
auditory_naming` renders it as `S5_within_category_null_auditory_naming.{png,pdf}`, a
diagnostic with no caption.

What it shows, recorded here so it is not rediscovered: observed retrieval sits at the
uniform-substitution null and **below** the category-preserving null at every k (excess
−0.005 / −0.008 / −0.017; Holm p=1.000 for all three), at both auditory decoder
configurations — `AUD_RUN_FIGURE` (23-region, 10 bins) and `AUD_RUN` (13-region, 5 bins;
−0.005 / −0.017 / −0.024). Ruled out as causes: OOV stimulus words receiving zero GloVe
vectors (restricting to in-GloVe words makes the excess more negative), and the true word's
presence in its own null pool (removing it also makes it more negative). Unexplained: in
7/10 participants the decoder ranks the true word below its own category-mates.
