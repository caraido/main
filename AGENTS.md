# Speech — semantic decoding from intracranial signals

Shared instructions for every coding agent working in this repository (Claude Code reads
this through `CLAUDE.md`; Codex reads it natively). This file is authoritative: when it and
a generated memory disagree, this file wins.

## Repository purpose

Decode word-level meaning from high-gamma activity (70–200 Hz) recorded during picture
naming (PN) and auditory naming (AN). The core method is a kernel-PLS regression→retrieval
decoder: `Nystroem(rbf, 100) -> PLSRegression(10)` onto GloVe word embeddings, scored by
1-NN cosine retrieval (`word_bal_acc`, `cat_indep_bal_acc`). Cohort: 12 participants
(2 ECoG, 10 sEEG), 68 words / 6 semantic categories; 6 have both tasks
(AA AZ DR LH RB WBH).

Chapter 1 of Alec's thesis, co-first-authored with **Joon** (Joon Hei Lee), who owns the
fixed-class SVM classifier arm. Current draft:
`Semantic decoding paper_Draft 1_with new results_2026-06-11.docx` — not the older May-8
copies.

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
- **Never delete anything under `results/` or `figures/` without checking
  `docs/results_index.md` first** (regenerate with `python -m utils.audit_runs --write`).
  Runs marked `PINNED` are named in tracked source and feed paper figures.
  `2026-04-08_17-05-14` and `2026-06-02_17-25-11` read as stale April runs and are
  hard-coded defaults worth ~31 GB. **Never prune by date.** Use the `results-hygiene` skill.
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
- **All 12 participants now have an ROI atlas** (`data/{PAT}/{PAT}_*channels.pkl`). Prefer
  the picture-naming file; ROI info is task-invariant. A glob of `*_picture_naming_channels.pkl`
  silently drops AA, whose file is just `AA_channels.pkl`.

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
| Any claim, statistic, or exclusion | `docs/agent-context/scientific-integrity.md` |
| Knowing when work is done | `docs/agent-context/validation.md` |
| IDs, units, indexing, file naming | `docs/agent-context/data-conventions.md` |
| Channel-name → electrode → ROI plumbing | `docs/agent-context/channel-and-roi-naming.md` |
| Env, encoding, OneDrive behaviour | `docs/agent-context/environments.md` |
| How shared context and memory are organised | `docs/agent-context/README.md` |
| Repo layout, lifecycle, results map | `docs/repo_layout.md` |
| Which runs are safe to touch | `docs/results_index.md` |
| Which modules are load-bearing | `analysis/README.md` |
| CLI flags and exact invocations | `README.md` §Quick Start; `analysis/cross_task/README.md` |
| What is still undecided | `.claude/open-questions.md` (local, not tracked) |

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
