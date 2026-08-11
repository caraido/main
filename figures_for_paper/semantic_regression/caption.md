# Figure caption — Cross-patient semantic-decoding time courses

Cross-patient semantic-decoding time courses (GloVe). Held-out decoding accuracy as a
function of time in two naming tasks — picture naming (N=15, aligned to trial onset); auditory naming (N=10, aligned to auditory stimulus onset) — with kernel-PLS (Nystroem RBF kernel followed by
PLS regression onto GloVe word-embedding targets); each participant in a distinct colour,
kept the same in every panel. Columns = task, rows = metric.

*Picture naming* (**a**, **b**, **c**, **d**; N=15). **a** Category accuracy. **b** Top-1 word-retrieval accuracy. **c** Top-3 word-retrieval accuracy. **d** Top-5 word-retrieval accuracy.

*Auditory naming* (**e**, **f**, **g**, **h**; N=10). **e** Category accuracy. **f** Top-1 word-retrieval accuracy. **g** Top-3 word-retrieval accuracy. **h** Top-5 word-retrieval accuracy.

Within a metric family the y-scale is shared across panels and across tasks (the word top-k rows share one scale; the category row has its own), so accuracies are directly comparable between tasks.

The auditory cohort spans two stimulus sets: NUEx030 and NUEx031 heard an earlier set with longer spoken prompts and a different category inventory (it adds abstract and action and omits vehicle). The number of semantic categories therefore differs between participants, so chance for category accuracy is per participant and the dashed line is the mean of the per-participant shuffled nulls, not a single 1/n_categories.

Coloured bars below the chance line are a per-participant significance raster (rows ordered by peak
accuracy, highest at top): time bins after the alignment cue where the observed mean accuracy
exceeds the 95.0th percentile of the shuffled-null distribution at that bin (per-bin one-sided
permutation test, p < 0.05; bins before the alignment cue are not tested). Dashed
line: mean shuffled chance across participants. Dotted vertical line at 0 s: that task's alignment
cue. Shaded vertical bands: mean cue time across participants ± 1 s.d.; cues identical across
participants (the group-warped auditory stimulus offset) are drawn as a single line without a band.
The alignment cue itself, and cues falling outside a panel's time window, are excluded. x-axis in
seconds. Participants are identified by display ID (NUEx###).

---

# Supplementary figure S5 — within-category (category-preserving) null

**a** Retrieval accuracy decomposed into a category component and a word component, cohort
mean ± SEM at top-1, top-3 and top-5. Grey: uniform chance, the accuracy expected with no
information. Teal: the category-preserving null, in which the word↔trial correspondence is
permuted *within* each semantic category, so category structure is retained and
sub-category word identity is destroyed. Amber: the observed decoder. The amber-to-teal gap
is word-level information — accuracy the decoder achieves beyond what knowing the category
alone would buy. **b** That excess against k, one line per participant, cohort mean in
black; the dashed line is the category-only expectation. **c** Per participant at top-5:
observed accuracy against the category-only 95% null band, ordered by excess.

Group means: top-1 observed 0.042 against a category-only null of 0.028 (uniform chance
0.017), excess +0.014; top-3 observed 0.107 against 0.085 (uniform 0.051), excess +0.022;
top-5 observed 0.172 against 0.139 (uniform 0.084), excess +0.034. Participants exceeding
their own category-only null at p < 0.05: 7/15 at top-1, 8/15 at top-3, 8/15 at top-5.
One-sided permutation test against the within-category null; cutoff from `utils.config.ALPHA`.
Participants are identified by display ID (NUEx###). **N = 15** (picture naming).

Rendered by `within_category_null_panels.py` from
`source_data/within_category_null_topk.csv`, which `within_category_null.py` computes from
`figures/open_vocab_retrieval/source_data/trial_predictions_picture_naming.csv`.

**Limitations.** The auditory arm is not shown: it needs
`trial_predictions_auditory_naming.csv`, which now exists but has not been put through this
analysis. The input CSV lives in a gitignored directory (see the untracked-inputs table in
`docs/repo_layout.md`), so this figure cannot be regenerated from a clean checkout alone.
