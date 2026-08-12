# Speech — semantic decoding from intracranial signals

Shared instructions for every coding agent working in this repository (Claude Code reads
this through `CLAUDE.md`; Codex reads it natively). This file is authoritative: when it and
a generated memory disagree, this file wins.

## Repository purpose

Decode word-level meaning from high-gamma activity (70–200 Hz) recorded during picture
naming (PN) and auditory naming (AN). The core method is a kernel-PLS regression→retrieval
decoder: `Nystroem(rbf, 100) -> PLSRegression(10)` onto GloVe word embeddings, scored by
1-NN cosine retrieval (`word_bal_acc`, `cat_indep_bal_acc`).

**CP was RETIRED on 2026-08-12 by group consensus and is not reported.** The analysed cohort
is **14 picture / 9 auditory** — `AA AZ DR KAW LH PV RB SE WBH` have both tasks. CP's data
moved to `data_archive/CP/` and CP's existing runs are kept on disk; a retired participant is
not a deleted one. **`utils.config.RETIRED_PATIENTS` is the single switch** — every cohort
constant derives from it, `discover_patients` filters on it, and an explicit
`--patients CP` is a hard error. Do not hand-write an exclusion anywhere else. Full record:
`docs/experiments/015-retiring-cp.md`.

Enrolled but not analysed: 15 picture / 10 auditory (the `_ENROLLED_*` tuples in
`utils/config.py`, kept as the historical record). KAW joined both cohorts 2026-07-30, PV and
SE landed 2026-08-06.

**"On disk" is not "in the paper," and this paragraph has itself been wrong.** It used to
assert that no figure included PV or SE and that every shipped figure was N=13 / N=8. Both
statements were stale by 2026-08-09: `PICTURE_PATIENTS` and `AUDITORY_PATIENTS` include PV
and SE, and the shipped `semantic_regression` source data is N=14 picture / N=9 auditory as
of 2026-08-12. (`PATIENTS`, `DEFAULT_TARGET_PATIENTS` and `AUD_PATIENTS` were listed here as
config constants and **do not exist** in `utils/config.py` — they are downstream aliases.)

The rule that survives: **do not restate a caption's N from this file.** Read it off the
figure's own `source_data/*.csv`, which is the only thing that cannot drift from the figure.
`discover_patients` returns 14 / 9 (retired participants filtered), so a run launched without
an explicit `--patients` still takes its cohort from the filesystem and changes when data
lands.

**The auditory cohort still spans two stimulus sets, and this is not cosmetic.** CP and RB
ran an older set whose spoken prompts are ~1.3 s longer (median 4.72 s CP / 4.64 s RB against
~3.2–3.6 s) and whose categories differ: it adds `abstract` and `action` and drops `vehicle`.
**With CP retired, RB is the only old-set participant left** — the caveat gets *smaller*, not
gone, and must stay in Methods rather than quietly disappearing (Alec, 2026-08-12). The pair
is named once, in `utils.config.OLD_STIMULUS_SET_PATIENTS`; consumers intersect it with the
run's own cohort. So "68 words / 6 semantic categories" describes the *current* set only — the
per-participant category count actually ranges 5–7, and chance for `cat_indep_bal_acc` is
therefore per participant (0.143–0.200), never a flat 1/6. KAW, PV and SE ran the
**current** set (6 categories,
chance 1/6 — read off their `*_labels.pkl` `class` column, not off a duration comparison;
`data/_aud_stim_durations.json` has no PV/SE entry yet).

**The group warp target couples participants unless you pin it.** Under `--warp-scope group`
the target is the median over the *pooled* trials of every patient in the run, so adding a
participant shifts it and silently re-warps everyone already in the cohort — which is why
adding CP superseded the auditory run rather than extending it (3.500 s → 3.580 s).
`--warp-target-sec` (added 2026-07-30) breaks that coupling by supplying the target instead
of computing it: the new participant depends on the constant and nobody depends on the new
participant. `meta.json` records which happened as `auditory_warp_target_source`
(`computed` | `pinned`), and under `pinned` it leaves `auditory_warp_target_patients` null
rather than claiming the run's own patients defined the target. KAW was added this way, at
the pre-existing 3.5800 s.

**Retiring CP is pinned, not recomputed** (Alec, 2026-08-12). Dropping CP would move the
pooled median 3.5600 s → 3.4960 s and re-warp all nine remaining participants, voiding every
auditory fit. Pinning at **3.5600 s** severs that: each participant's warp then depends only
on their own trials and the constant, so the existing `AUD_RUN` already *is* the
nine-participant run and **no auditory model was re-fit**. The cost, which belongs in Methods:
the retained nine stay warped to a target computed from a cohort that included CP, who had the
longest segments — a ~64 ms inflation.

**For the 2026-08 re-run the target is recomputed, not pinned** (decided 2026-08-08). Every
auditory run is being replaced, so continuity with the retired runs buys nothing, and 3.5800 s
is a property of a cohort that no longer exists. Omit `--warp-target-sec`; the pooled median
over all 10 participants becomes the target and `meta.json` records
`auditory_warp_target_source: computed`. **PV and SE are absent from
`data/_warp_segment_durations.json`** (13 keys), so that first run also re-reads their
multi-GB trial pkls to measure their prompt durations — expect it to be slow once, then
cached. Pin with `--warp-target-sec` only when *extending* a cohort whose runs must stay
comparable.

Chapter 1 of Alec's thesis, co-first-authored with **Joon** (Joon Hei Lee), who owns the
fixed-class SVM classifier arm. The current draft is tracked at
`Semantic decoding paper_Draft.docx` in `main/`.

## Core working principles

- **Inspect before you modify.** Read the code, its importers, and the routed context in
  the table below before changing behaviour.
- **Grep for importers before moving, renaming, or deleting anything.** In this repo
  neither folder nor age predicts whether something matters. Two modules classified as dead
  during the 2026-07 reorg turned out to be libraries behind paper figures, and the
  breakage would only have surfaced on regeneration.
- **Make the smallest coherent change**, matching the conventions of the nearest existing
  implementation.
- **Prefer the authoritative artifact over prose.** `docs/results_index.md` over a
  remembered run id; `participants.json` over a hard-coded palette; a docstring over a
  summary of it.
- **A figure and the numbers on it are one unit.** Change one, regenerate the other, then
  diff the tracked `source_data/*.csv` — rendered PDFs always differ (embedded timestamps),
  so the CSVs are the real signal.
- **Distinguish observed fact, assumption, and interpretation** in everything you report,
  and say which one you are giving.
- **Do not claim completion before the checks in `docs/agent-context/validation.md` have
  actually run.** State what you ran and what it printed, including checks you skipped.

## Critical boundaries

- **Never hand-compose a results path.** Always `utils.paths.results_dir("<analysis>")`.
  Three competing roots plus a relative fallback that escaped the repository is how
  `phoneme_semantic_dissociation` ended up split across two directories, half of it outside
  the project.
- **Never hard-code a run id or a p-value cutoff.** Both live in `utils/config.py` —
  `PIC_RUN`/`AUD_RUN` and `ALPHA` (0.05), with `PCTILE` derived from `ALPHA`. Before it
  existed the same auditory run id was typed into three modules (two of them splitting the
  string across lines, so a grep found one), and the repo simultaneously claimed p<0.01 in
  its code and p<0.05 in its shipped caption. Keep a per-invocation CLI flag; do not
  reintroduce a module-level literal. `utils/config.py` must stay a `.py` under `utils/`
  or `audit_runs` stops seeing the pins — the reason is in `docs/repo_layout.md` §Results.
- **Never delete anything under `results/` or `figures/` without checking
  `docs/results_index.md` first** (regenerate with `python -m utils.audit_runs --write`).
  Runs marked `PINNED` are named in tracked source and feed paper figures.
  `2026-04-08_17-05-14` and `2026-06-02_17-25-11` read as stale April runs and are
  pinned defaults worth ~31 GB. **Never prune by date.** Use the `results-hygiene` skill.
- **Never add a blanket pattern to `.gitignore`.** A `*cache*` rule silently untracked 23
  files, including the 18 `cache_*.csv` that *determine* rendered figure output; they
  drifted out of sync with the committed figures with nothing showing in `git status`.
  `.gitignore` carries a comment saying so — do not undo it.
- **`results` is a load-bearing directory name.** `.gitignore` excludes `*results`, so
  renaming that tree would stage 169 GB. It is also why `results_index.md` lives in `docs/`:
  git cannot re-include a file under an excluded directory.
- **Raw data and acquisition files under `data/` are read-only.** Derived outputs are
  rewritable; never write a derived output into a raw-data directory.
- **OneDrive Files-On-Demand:** everything under `results/` and `figures/` is a cloud
  placeholder. A same-volume `mv` is an instant metadata rename. A `cp`, a checksum, or a
  move outside the OneDrive root hydrates the whole tree.
- **Never fabricate or extrapolate a number.** Every figure, statistic, or p-value must come
  from a run that can be pointed at. If it has not been computed, say so.
- **Never change a statistical method because it makes a result significant.** Full rules in
  `docs/agent-context/scientific-integrity.md`.
- **Never overclaim a result.** In particular the cross-modal finding is a **"shared,
  alignable subspace"** — never "an amodal code". Naive PN↔AN transfer is at chance; only
  co-training plus low-dimensional alignment recovers shared structure.
- **Never commit or push without an explicit ask.** Never `--no-verify`; never bypass commit
  signing. Confirm before anything destructive or irreversible, and before anything that
  leaves this machine.

## Non-obvious project facts

- **Environment:** conda env `Speech` — `conda run -n Speech python …`. Project pickles need
  `dill`; the standalone Python 3.10 install cannot load them. See
  `docs/agent-context/environments.md`.
- **Run from `main/`**, as `python -m analysis.<topic>.<name>`.
- **Set `PYTHONIOENCODING=utf-8 PYTHONUTF8=1`** for anything that logs box-drawing
  characters or a `▸` glyph, or it crashes on cp1252.
- **Feature layout:** the column for channel `c`, history bin `k` is `c + k * n_channels`.
- **The Nystroem nonlinearity dilutes single-channel effects** — few channels reach
  BH-FDR significance. This is why cross-task importance is region/ROI-only.
- **`tests/` is the pilot stage** of `tests/ -> analysis/ -> figures_for_paper/`, with dead
  pilots going to `_archive/`. It is empty *between* pilots, not permanently —
  `tests/auditory_alignment/` is currently live. Nothing outside `tests/` may import it.
- **All 15 participants have an ROI atlas** (`data/{PAT}/{PAT}_*channels.pkl`). Prefer
  the picture-naming file; ROI info is task-invariant. A glob of `*_picture_naming_channels.pkl`
  silently drops AA, whose file is just `AA_channels.pkl`.
- **`primary_roi` is retired (2026-08-08).** ROI analysis is keyed on `nmm_roi` or `dk_roi`,
  selected with `--atlas`, and the coarse anterior/posterior merge (`--merge-regions`) is gone
  with it. The two atlases are **peers**, not a default and a variant: each gates channel
  selection as well as grouping, so an NMM run and a DK run are different channel sets, and
  they name the same region for only 442 of the 718 contacts either whitelists. Say which
  atlas any number came from. Full reference: `docs/agent-context/roi-vocabulary.md`.
- **Only temporal-parietal cortex is in the analysis** — a 13-region whitelist
  (`utils.rois.IN_ANALYSIS`), vendored from `electrode_labeling`. It is a *region* filter, not
  an electrode-type filter: a depth contact in supramarginal is in. 634 contacts under NMM /
  683 under DK, after artifact rejection. Verify the copy against the sibling repo with
  `python scripts/check_roi_vocabulary.py --sibling <path>`.
- **The region set is now a named scope, and 13 regions is the *default*, not the only option**
  (added 2026-08-11). `--roi-scope` selects from `utils.roi_scopes.SCOPES`: `tp` = the 13
  (every paper run), `tpfm` = 23, adding the `FRONTAL` and `MEDIAL` families. **`tpfm` is
  diagnostic only** — `utils/roi_palette.py` is vendored and cannot be extended from this repo,
  so figures built from a `tpfm` run render the ten added regions in one grey and drop them from
  legends *without raising*. The atlas picks the column, the scope picks the regions; they are
  independent. `utils/rois.py` is never edited to change a scope — `utils/roi_scopes.py` derives
  them from its public API, which is what keeps `check_roi_vocabulary.py` passing.
- **500 ms history (5 bins) is the standard** (`utils.config.N_BINS_HISTORY`). The 1000 ms
  results are retired. Run ids now carry `_roi-<atlas>_scope-<scope>_h<bins>` unconditionally,
  because before that a 5-bin gated run and a 10-bin whole-brain run produced the same directory
  name. A run id with no `_scope-` token predates 2026-08-11 and is `tp`.
- **"KAW has no fusiform coverage" was a `primary_roi` statement and no longer holds as
  written.** 0 aFus/pFus under `primary_roi`, but **4 under `nmm_roi` and 3 under `dk_roi`**.
  Fusiform counts differ across all three columns for most patients (table in
  `docs/agent-context/channel-and-roi-naming.md`), so any claim about a participant's coverage
  must name the atlas. Do not carry the old caveat forward, and do not silently drop it
  either — restate it under the atlas the analysis actually ran on. KAW's `shared_vocab` of 58
  is unaffected and still makes it a strong addition for the cross-task analyses.

## Glossary

| Term | Meaning |
|---|---|
| TP | temporoparietal cortex — the decoding target |
| PN / AN | picture naming / auditory naming — the two speech tasks |
| HGA / high-gamma | 70–200 Hz broadband; the signal modality |
| regression-retrieval | kernel-PLS (Nyström-RBF + PLS-10) HGA→GloVe, then nearest-neighbour retrieval. Contrast with Joon's SVM classifier |
| the sharpened claim | "Semantic info decodable from TP cortex during production is *lexical* (word-level, amodal, continuous), not *perceptual* (stimulus-level, modality-bound, categorical)" |
| shared/alignable subspace | The correct framing of the cross-modal result. Never "amodal code" |

## Context routing

| Working on | Load |
|---|---|
| Anything under `figures_for_paper/` | skill **paper-figure** |
| Cross-task ROI / region importance | skill **cross-task-roi** |
| Open-vocabulary / zero-shot retrieval | skill **open-vocab-retrieval** |
| Deleting, pruning, or moving results | skill **results-hygiene** |
| Deduplicating or extracting helpers | skill **python-refactor** |
| Making any nontrivial change | `docs/agent-context/change-protocol.md` |
| Any claim, statistic, or exclusion | `docs/agent-context/scientific-integrity.md` |
| Knowing when work is done | `docs/agent-context/validation.md` |
| IDs, units, indexing, file naming | `docs/agent-context/data-conventions.md` |
| Channel-name → electrode → ROI plumbing | `docs/agent-context/channel-and-roi-naming.md` |
| ROI vocabulary, inclusion whitelist, region colours | `docs/agent-context/roi-vocabulary.md` |
| Env, encoding, OneDrive behaviour | `docs/agent-context/environments.md` |
| How shared context and memory are organised | `docs/agent-context/README.md` |
| Repo layout, lifecycle, results map | `docs/repo_layout.md` |
| Which runs are safe to touch | `docs/results_index.md` |
| Which modules are load-bearing | `analysis/README.md` |
| CLI flags and exact invocations | `README.md` §Quick Start; `analysis/cross_task/README.md` |
| What is still undecided | `docs/experiments/` — entries with `status: open` |

Skills are canonical in `.agents/skills/` (read directly by Codex) and mirrored into
`.claude/skills/` for Claude Code by `python scripts/sync_agent_skills.py`.

## Validation requirements

Full hierarchy, including what to do when a check cannot run:
**`docs/agent-context/validation.md`**. The short form:

1. `python -m py_compile <touched .py files>` — there is no test suite in this repo.
2. If a figure pipeline changed: re-run its `*_panels.py`, then `git diff --stat` the
   tracked `source_data/*.csv`. No diff is the pass condition.
3. If a results path changed: `python -m utils.audit_runs --write`, then confirm
   `docs/results_index.md` gained no new `incomplete` entry.
4. State what you ran and what it printed.
