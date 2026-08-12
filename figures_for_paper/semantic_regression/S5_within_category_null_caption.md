# Supplementary figure S5 caption — within-category (category-preserving) null

The paragraph below is the caption as it should appear in the manuscript — copy it whole.
Everything under "Notes" is provenance for this repository and is not part of the caption.
Hand-written (this figure's renderer writes no caption); keep it in the same Nature legend
style as `caption.md` — see `figures_for_paper/README.md` §4.

**Supplementary Figure S5 | Word-level retrieval against a category-preserving null.**
Retrieval accuracy at top-1, top-3 and top-5 during picture naming (N = 15). Bars: cohort
mean ± s.e.m. Grey (chance): uniform substitution, in which the candidate word is drawn
uniformly from that participant's own stimulus vocabulary and its rank read off the same
fixed ranking. Teal (category-only): the same substitution restricted to the true word's
semantic category, so category structure is retained and sub-category word identity is
destroyed; 10,000 draws per participant. Amber (category + word identity): the observed
decoder. Black dots: individual participants, jittered horizontally, each participant's
three dots sharing an x-offset. Ranking is mean-centred cosine similarity between a trial's
predicted GloVe vector and the GloVe target vector of each of that participant's unique
stimulus words; all decoded trials are pooled, in-vocabulary and held-out alike, and a
permutation substitutes a word and reads its rank off the same fixed ranking, without
re-fitting and without re-ranking. Brackets: one-sided Wilcoxon signed-rank test across
participants on the per-participant difference (observed − category-only), Holm-corrected
over the three tests in this panel (*** p < 0.001, ** p < 0.01, * p < 0.05, n.s.
otherwise); Holm-adjusted p = 0.010, 0.011 and 0.010 at top-1, top-3 and top-5. Stimulus
vocabulary is 44–64 words and 6–7 semantic categories per participant, so uniform chance is
per participant and the grey bar is the mean of the per-participant nulls rather than a
single 1/W. Participants are identified by display ID.

## Notes — not part of the caption

- Figure: `S5_within_category_null.{png,pdf}`; plotted values:
  `source_data/within_category_null_topk.csv` (per-participant points) and
  `source_data/within_category_null_group.csv` (bar heights, s.e.m., stars).
- Rendered by `within_category_null_panels.py`, which computes nothing; the Wilcoxon and
  the Holm correction belong to `compute_within_category_null.py`. Star ladder from
  `utils.config.p_stars`, cutoff from `utils.config.ALPHA`.
- Input: `figures/open_vocab_retrieval/source_data/trial_predictions_picture_naming.csv`
  (`PIC_RUN`; 13-region temporal-parietal whitelist, 5 bins of history — the configuration
  behind the main figure, `caption.md`).
- Group excess (observed − category-only): +0.014 / +0.022 / +0.034 at top-1 / top-3 /
  top-5. Kept out of the caption as a result; quote it in the Results text, not here.
- One-sided Wilcoxon signed-rank floors at 1/2^n = 3.05e-5 at N = 15, so after Holm over
  three tests the smallest attainable adjusted p is 9.2e-5.

**Limitations.** The input CSV lives in a gitignored directory (see the untracked-inputs
table in `docs/repo_layout.md`), so this figure cannot be regenerated from a clean checkout
alone. Observed accuracy and the null are both averaged over trials, so a word produced on
many trials contributes many times; averaging both per unique word instead reduces the
excess from +0.014 / +0.022 / +0.034 to +0.006 / +0.010 / +0.019 (top-1 / top-3 / top-5).
The direction is unchanged and no significance test was re-run under that weighting.

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
