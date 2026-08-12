---
name: open-vocab-retrieval
description: Running and interpreting the zero-shot / open-vocabulary word-retrieval pipeline in analysis/open_vocab_retrieval/ — headline metric, gallery construction, held-out split semantics, and the significance tests that make nDCG and near-miss interpretable. Use when running the retrieval pipeline or reporting any retrieval number.
---

# Open-vocabulary retrieval

## Purpose

Rank each trial's predicted GloVe vector by cosine against an open gallery (the stimulus
words plus up to 5000 POS/frequency-matched distractors), to show the decoder extends beyond
a fixed class set.

## Trigger conditions

- Running `analysis.open_vocab_retrieval.run`.
- Touching gallery, metrics, relevance, or stats code in that package.
- Reporting a percentile rank, top-k, MRR, nDCG, or near-miss number.
- Building the `extendability` paper figure (**picture naming only**; the auditory and
  co-trained arms were retired 2026-08-12).

## Required inputs

- `Speech` conda env, cwd = `main/`. GloVe 840B via torchtext, cached at `main/.vector_cache`.
- A picture and/or auditory results run.
- Optional: `--subtlex-file`, `--concreteness-file`.

## Procedure

Run as `python -m analysis.open_vocab_retrieval.run`.

**`--patient` takes exactly ONE patient.** To run the full cohort, call `run(...)` directly
with the list — the shipped picture run is 14 (CP retired 2026-08-12; entry 015).

Modules, in pipeline order: `gallery, predict_io, retrieval, metrics, relevance, stats,
sweeps, figures, run`.

### Design decisions — verify against docstrings before changing any of these

- **Headline metric is median percentile rank** (rank/N; chance 0.5, invariant to gallery
  size). Also reported: top-k, MRR, nDCG@100, top-10 Wu–Palmer near-miss similarity.
- Decode target is GloVe 840B, and **gallery words are embedded with the same model** so
  query and gallery share a space.
- **Frequency proxy is GloVe vocab rank** (`itos` is frequency-ordered), so no SUBTLEX
  download is needed; `log_freq = -log10(rank+1)`.
- **Concreteness is optional.** Without `--concreteness-file` the "matched" gallery falls
  back to POS + frequency-band matching with a loud warning and a NaN concreteness column.
  No norms file ships by default.
- The true word for gallery lookup is `clean_word(reg.labels)` — strip the `(category)`
  disambiguation suffix (`mouse(object/tool)` → `mouse`). **The held-out zero-shot split is
  at the clean-word level**, so both senses hold out together.
- Gallery POS filter is noun-**dominant** WordNet + nltk stopwords + `len>=3`. A plain "has
  a noun synset" test lets function words (in, be, a) through.

### Statistics — the part that is easy to get wrong

- **Never pool trials across patients.** Significance is a within-patient trial→word
  permutation null, then a group Wilcoxon versus chance.
- Group Wilcoxon floors at **1.2e-4** for n=13 and **0.0039** for n=8 (the auditory cohort
  since KAW joined on 2026-07-30; it was 0.0078 at n=7 and 0.0156 at n=6), one-sided. When
  every patient is significant you are at the floor — say so rather than implying more.
  Hitting the floor
  means only that every patient fell on the same side of its null, which is not the same as
  any individual effect being strong: check the within-patient count before claiming more.
- **Absolute nDCG (~0.65) is uninterpretable**: chance nDCG is ~0.59–0.64, not 0. It MUST be
  read against `perm_p_ndcg` / `ndcg_null_mean` (per patient) and `ndcg_vs_null` (group).
  Near-miss is likewise tested against a matched permutation null (`perm_p_near_miss`,
  group `near_miss_vs_null`).
- `n_perm_graded` default 200 → per-patient permutation p floors at ~0.005.

## Decision points

| Situation | Action |
|---|---|
| Need all 12 patients | Call `run(...)` directly; `--patient` will not take a list |
| No concreteness norms available | Proceed; note the matched gallery is POS+frequency only |
| Reporting nDCG | Always alongside its permutation p — never the absolute value alone |
| Auditory stimulus labels | ~30 are multi-word phrases with spaces removed (`a bird`→`abird`), OOV in GloVe, dropped from the gallery and excluded from rank metrics. Standard handling — state it |
| A result is n.s. | Report it. Auditory nDCG vs null was p=0.078 at n=6; at n=7 the group test hit the floor (p=0.0078) while holding within only **2/7** patients — report both numbers, not the group one alone. Re-check the within-patient count after the n=8 regeneration; the floor moves to 0.0039 but the per-patient tally is the honest number |

## Validation

Sanity check on AA (picture): median %rank all ≈ 0.03; in-vocab (0.024) < held-out (0.052)
< chance (0.5); near-miss permutation significant. If in-vocab is not below held-out, the
split is wrong.

## Failure handling

- A full 6-patient run at headline N=5000 with 1000 permutations is heavy — it loads every
  results pkl, and nDCG's ideal-DCG is ~N × n_trials WordNet calls. Correct, just slow.
- Set `PYTHONIOENCODING=utf-8 PYTHONUTF8=1`.

## Outputs

`per_patient_metrics_*.csv`, `group_inference_*.json`, `trial_predictions_*.csv`; figures to
`main/figures/open_vocab_retrieval/`, all CSVs to `.../source_data/`.

**Note:** `figures/open_vocab_retrieval/source_data/` looks exploratory but two paper
pipelines read it, including a 38 MB `trial_predictions_picture_naming.csv`. Do not prune it.

## References

- `analysis/open_vocab_retrieval/` — module docstrings are authoritative
- `figures_for_paper/extendability/` — the one consumer
- Skill **paper-figure** for anything rendered
- `utils/retrieval.py` — `mean_center_db`, `normalize_rows`
