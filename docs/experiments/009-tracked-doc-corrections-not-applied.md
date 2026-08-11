---
id: 009
kind: decision
title: Tracked documentation that describes things which no longer exist
status: answered
analysis: semantic_regression
opened: 2026-07-26
closed: 2026-08-11
runs:
report:
answer: >
  Applied. VIP struck from analysis/README.md:24 and README.md:94,262 (recast as
  "retired 2026-07-23", matching the framing already used in analysis/cross_task/README.md
  and figures_for_paper/cross_task/caption.md). The main/pytest/ section removed from
  README.md (the tree entry and the "6. Unit tests" section); validation.md's own pointer to
  that section corrected too. tests/ description in repo_layout.md trimmed to the structural
  lifecycle role, dropping the specific-content claims that go stale. AGENTS.md:56-57's claim
  that the tracked draft was removed from the repository is also now wrong in the opposite
  direction -- fixed to name the actual tracked file. .vscode/settings.json: still moot, no
  .vscode/ directory on this machine.
---

## Question

Edits to **git-tracked** files, proposed 2026-07-26 and held back pending approval. Each is
documentation asserting something that is not true, which is worse than silence: an agent
routed by these files acts on them.

## What was tried

Re-checked on disk 2026-08-11; counts below are current, not remembered.

**VIP is described as a live region-importance method.** It was deleted 2026-07-23. Three
places: `analysis/README.md:24`, `README.md:94`, `README.md:262`. Nothing breaks —
`figures_for_paper/cross_task/` still reads a `vip` column from shipped CSVs that retain it
— but the docs are wrong, and entry [003](003-cross-task-paragraph-overclaims.md) says VIP
must not appear in the manuscript either.

**A `main/pytest/` directory is documented and has never existed.** Four places, all in
`README.md`: `:116`, `:293`, `:296`, `:297`, including runnable-looking `pytest pytest/`
commands. There are no test files, no `conftest.py`, and no pytest/ruff/black/pre-commit
config anywhere in `main/`. `tests/__init__.py` carried the same claim and was corrected
2026-08-10; `docs/agent-context/validation.md` already states plainly that there is no test
suite.

**`.vscode/settings.json` `livePreview`** points at `main/tests/results/…`, a pre-2026-07
path. No `.vscode/` directory exists on this machine, so this is probably already moot.

## Result

Not a compute question. All three are still present except the two already corrected.

## Next

**Decision, Alec 2026-08-11 — approved, all of it:**

- **Remove the VIP analysis.** Not "mark retired" — remove. It is already gone from the code
  (2026-07-23) and is being removed from the manuscript with the rest of the cross-task
  material (entry [003](003-cross-task-paragraph-overclaims.md)); the tracked descriptions
  are the last place it survives.
- **Update `README.md` to the current state**, and generally: where something no longer
  exists, say so rather than leaving the old description. That covers the `main/pytest/`
  section, which documents runnable-looking `pytest pytest/` commands for a directory that
  has never existed.
- **Do not describe `tests/` in `docs/repo_layout.md`.** Its contents change quickly, and
  the run ledger plus `--status` now report what is actually there — a prose description of
  a fast-moving directory is a thing that is wrong most of the time.
- **`.vscode/settings.json`**: confirmed moot on this machine — there is no `.vscode/`
  directory. Left as a note in case it exists elsewhere.
