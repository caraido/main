# Change protocol — classifying, scoping, and landing a change

Routed from `AGENTS.md` for any nontrivial change. This file holds only what the rest of
the tracked context does not: how to classify a change before making it, what to inventory
first, and the traps specific to this pipeline. Where another file owns a rule, this one
points at it instead of restating it — a second copy is how the repo ended up with three
`PIC_RUN_DEFAULT` constants and a caption that disagreed with its own code.

## Classify the change first

State the category in the plan, before editing. Never combine two silently.

| Category | Moves a reported number? |
|---|---|
| implementation-only | no |
| parameter | yes |
| scientific-method | yes |
| sample-definition | yes |
| data | yes |
| results / provenance | no, but changes what is citable |
| figure-expression-only | no |
| manuscript / documentation | no |

A refactor is *implementation-only by definition*: it must not change statistics, sample
inclusion, missing-data behaviour, alignment, seeds, thresholds, output names, or figures.
If it does, it was a parameter/method/sample/data change all along — re-plan it and get it
approved as one.

## Audit before redesign

The first pass is read-only. Produce an inventory with exact `path:line` for:

- duplicate or overlapping functions; reusable logic trapped in an entry-point or figure
  script;
- imports from the pilot `tests/` tree;
- hand-composed result paths;
- hard-coded run ids, participant lists, alpha thresholds, palettes, or identities;
- `dropna`, `nan*` reducers, imputation, `isfinite` filters, broad `except:`, warning
  suppression;
- reshapes, merges, concatenations, sorting, indexing, and masking that could change
  alignment;
- nondeterministic code and unrecorded seeds;
- loaders with no schema or count validation;
- modules with downstream figure or report consumers (`analysis/README.md` records these);
- tracked figures whose source data or caches may be stale;
- claims or citations with no traceable source;
- stale documentation, especially commands and paths that no longer exist.

For each finding, say which it is: **observed fact**, **risk**, **uncertainty needing
investigation**, or **proposed change**. Do not start implementing while the inventory is
incomplete, unless the task was scoped to one already-understood issue.

This list earns its keep. The 2026-07-28 pass over it found three top-level pipelines
hand-composing *relative* result paths, three modules defaulting to a `tests/results/`
root that no longer exists, and a routing table pointing at a context file that was never
written.

## The small-change loop

1. Write a plan: scope, category, risks, files, consumers, rollback point.
2. Identify (or add) the check that would catch a regression, *before* changing behaviour.
3. Implement one coherent step.
4. Read the diff for anything unrelated to that step.
5. Run the checks in `docs/agent-context/validation.md`.
6. Compare counts, missingness, and numerical output against the pre-change values.
7. Update the record that owns each changed fact — not a second record that restates it.
8. Report the commands actually run and what they printed, including the ones skipped.
9. Stop when an ambiguous scientific choice appears; it is Alec's, not yours.

For deduplication and helper extraction specifically, the `python-refactor` skill owns the
procedure and its own checklist — follow it rather than improvising from this list.

### Stop and report immediately if

- participant, trial, channel, ROI, label, or NaN counts change unexpectedly;
- a result moves beyond the tolerance you declared;
- a `PINNED` run becomes unreferenced, or a new `incomplete` run appears;
- a figure's `source_data/*.csv` changes without an approved reason;
- an API, data field, or number cannot be verified;
- the change would alter a scientific interpretation.

## Seeds — read this before writing any characterization test

**The decoder is not deterministic, and nothing records a seed.** In `models/model.py`:
`KFold(..., random_state=None)` and an unseeded `train_test_split`, plus
`np.random.permutation` on the global RNG for the shuffle baseline. The pool initialiser
suppresses warnings and reseeds nothing. `semantic_regression.py` contains no seed at all,
and `meta.json` has no seed field.

Two consequences:

- **Exact-value characterization tests are impossible today.** Either seed first, or write
  tolerance-based statistical assertions and say that is what they are. Do not describe
  the pipeline as reproducible.
- **A single `random_state` would destroy the analysis.** The design averages over
  `n_epochs` *independent* random repeats, and `_fit` receives the repeat index and
  discards it. Setting one shared `random_state=42` makes every repeat identical,
  collapsing both the variance estimate and the permutation null. The correct design is a
  per-repeat derived seed — `random_state = base + i` from that discarded index — with
  `base` recorded in `meta.json` and defaulting to `None` so current behaviour stays
  bit-identical until someone opts in.

Adding seeds is method-adjacent: it changes what future runs produce relative to the runs
already behind the paper. It needs Alec's explicit go-ahead, and it comes *before* any
characterization work that assumes determinism, not after.

## Missing data

The default for an unexpected missing or non-finite value is to **stop with a diagnostic**.
Never add `dropna`, `nanmean`, `fillna`, or an `isfinite` filter merely to make code run.

An approved policy may drop, propagate, or impute — but it must name the affected unit
(observation, trial, word, channel, ROI, participant, time bin, embedding), report counts
and fractions before and after, preserve the ids of excluded records, state its source,
and record the operation in the run manifest. Imputation is a method step, not cleanup:
fit it inside the training fold. Sample-composition changes need explicit approval.
`scientific-integrity.md` §Exclusions owns the reporting rules.

**Grandfather clause.** As of 2026-07-28 the tree already contains ~304 occurrences of
`nanmean`/`nanmedian`/`dropna`/`fillna`/`isfinite`, concentrated in the three root
pipelines. They are documented, not blocking: inventory them when you touch the
surrounding code, and review them one at a time. Only *new* occurrences are gated by the
paragraph above. Treating all 304 as blockers means the audit phase never terminates.

## Two rules that protect the run ledger

- **Keep each run id on one source line.** `utils/audit_runs.py` matches the literal
  string against the directory name, so an implicitly concatenated id
  (`"…group_" "align-…"`) yields only its first fragment, the run reads `unreferenced`,
  and the pruning plan in `docs/repo_layout.md` treats it as deletable. Two of the three
  old `AUD_RUN_DEFAULT` constants were split exactly that way. Full rationale lives in
  `utils/config.py`; do not let a formatter wrap those lines.
- **A new top-level directory that could name a run id must be added to
  `audit_runs.SCAN_DIRS` in the same commit.** The scan covers `figures_for_paper`,
  `analysis`, `tests`, `notebooks`, `report`, `utils` only. A fixture, verification, or
  research tree outside that set silently unpins every run it names — roughly 50 GB of
  pinned output is in scope. This is the same trap as moving `utils/config.py` to a
  root-level JSON.

## Do not invent

- **Package APIs.** Check the version installed in the `Speech` env and its actual
  signature or local source before calling something unfamiliar; the newest online docs
  may not match. If you cannot verify, mark the call unverified rather than presenting it
  as working.
- **Repository facts.** Search before asserting that a module is unused, a run is stale, a
  test exists, or a path is canonical. Folder and age are both poor proxies here.
- **Data schemas.** Inspect real columns, dtypes, ids, and shapes. Do not infer a field
  name from convention.
- **Literature.** Work from a DOI, PMID, or supplied paper. Use `TODO: citation required`
  when the evidence is missing, and keep what a source reports separate from your reading
  of it.

Numbers have their own rule, and it is stricter: `scientific-integrity.md` §Numbers.

## Validation

`docs/agent-context/validation.md` owns the hierarchy and what to report when a check
cannot run. Inspection is never a substitute for execution.
