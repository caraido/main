# Scientific integrity

Non-negotiable. These rules bind every claim this repository produces — in code comments,
figure captions, HTML reports, chat replies, and manuscript text. Routed from `AGENTS.md`
for any task that states a number, a statistic, or an exclusion.

Promoted 2026-07-27 from machine-local user configuration, where they were invisible to
Codex and to every other machine.

## Numbers

- **Never fabricate or extrapolate a number.** Every figure, statistic, or p-value must
  come from a run that can be pointed at. If it has not been computed, say it has not been
  computed — do not estimate, do not carry a number over from an older cohort, do not
  round a remembered value into a new claim.
- **Do not treat generated or simulated values as measured data.** Permutation nulls,
  bootstrap resamples, and synthetic checks are labelled as such wherever they appear.
- **Numbers and the artifact that produced them are one unit.** A quoted value must name
  its run id or its source-data CSV. When the two disagree, the artifact wins and the prose
  is wrong.
- Draft-manuscript values are **not** a source. Where a draft and a current run disagree,
  the disagreement is an open question (`.claude/open-questions.md`), not something to
  paper over.

## Statistics

- **State N, the test, and the multiple-comparison correction wherever a claim is made.**
- **Never change a statistical method because it makes a result significant.** Choosing a
  test is a design decision made before seeing the outcome; changing it afterwards is
  p-hacking regardless of intent.
- **Report negative and non-significant results as plainly as positive ones.** Examples
  already standing in this repo: auditory nDCG vs null is n.s. (p = 0.078, 2/6 patients);
  peak-bin word-level language-vs-vision contrasts are all n.s.; no ROI survives BH
  correction in the cross-task analysis. None of these are hidden or softened.
- **Know your floor.** With n = 6 the one-sided Wilcoxon p floors at 0.0156, so `**`/`***`
  are unreachable — say so rather than implying the effect is weak. With n = 12 the floor
  is 2.4e-4. Per-patient permutation p floors at ≈ 1/(n_epochs + 1).
- Absolute values of scale-free metrics are uninterpretable without their null. nDCG@100
  sits at ~0.65 where chance is ~0.59–0.64; it **must** be read against the matched
  permutation null, never reported bare.
- **Do not re-suggest a plain t-test for per-bin significance.** The per-bin test for the
  decoding time courses was iterated a long way — scalar-mean Wilcoxon+BH → paired
  Wilcoxon+BH → paired t-test+Bonferroni → one-sample t vs mean chance+Bonferroni → the
  current **99th-percentile permutation** (a bin is significant iff the observed mean
  exceeds the 99th percentile of the shuffled null at that bin, ≈ p<0.01). Every t-test
  variant failed the same way: observed accuracy sits a reliable ~0.01 above null even at
  baseline, and with n = 100 epochs that offset passes any t-test, so 30–45 % of *pre-onset*
  bins came out significant. Only a distribution/effect-size criterion drives pre-onset to
  ~0 %. This is a settled design decision, not an open choice.
- A structural correlation is not evidence. The co-trained Jacobian gives ρ(pic, aud) =
  +0.99 *per electrode* because one shared map scores both tasks — that is the model's
  architecture, not amodality. Task-specificity claims come only from the knockouts and the
  single-modality control.

## Exclusions and missing data

- **Do not silently discard missing values or failed observations.** State what was
  excluded, how many, and why, at the point the number is reported.
- **Annotate low-n; do not filter it.** Cross-participant plots keep small ROIs and
  low-trial participants and make the unreliability visible (`n=` annotations, median rings
  sized by participant count) rather than dropping them.
- Known standing exclusions that must be restated wherever affected numbers appear: ~30
  auditory stimulus labels are space-stripped multi-word phrases, OOV in GloVe, and their
  trials are dropped from rank metrics; AA has 52 unique words across 53 auditory trials,
  so its auditory arm is inherently zero-shot.
- **Do not conceal warnings, failed tests, or excluded data.** A pipeline that printed a
  warning and a pipeline that ran clean are different results.

## Framing

- **Never overclaim.** Name the soft spots — a stated limitation is a feature, not a
  concession.
- The cross-modal result is a **"shared, alignable subspace"**, never "an amodal code".
  Naive PN↔AN transfer is at chance; only co-training plus low-dimensional alignment
  recovers shared structure.
- Distinguish observed fact, assumption, and interpretation explicitly. If a mechanism is
  being proposed rather than measured, say which.
- Where a competing method wins, show it. MoCo (a vision CNN) is a strong competitor on
  category decoding and is reported, not hidden.

## Provenance and reproducibility

- Raw data and acquisition files under `data/` are **read-only**. Derived outputs are
  rewritable. Never write a derived output into a raw-data directory.
- Every figure must be reproducible from its own `source_data/`; see `data-conventions.md`.
- Results are written only through `utils.paths.results_dir("<analysis>")`, so a run is
  always locatable from its analysis name.
- `docs/results_index.md` (regenerate with `python -m utils.audit_runs --write`) is the
  ledger of which runs are `PINNED` to paper figures. Consult it before quoting a run id
  and before deleting anything.
- A run made on a dirty working tree is marked `git_dirty` and its configuration cannot be
  reconstructed from the commit. Two 2026-05 auditory runs have identical recorded configs
  but different bin counts for exactly this reason — treat `git_dirty` runs as
  provenance-incomplete.
