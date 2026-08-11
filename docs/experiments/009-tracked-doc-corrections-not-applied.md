---
id: 009
kind: decision
title: Tracked documentation that describes things which no longer exist
status: open
analysis: semantic_regression
opened: 2026-07-26
closed:
runs:
report:
answer:
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

- Strike VIP from the three tracked descriptions, or mark it explicitly retired.
- Delete the `main/pytest/` section from `README.md`, or create the directory. Documenting a
  test suite that does not exist is how "the checks passed" gets said without checks.
- Confirm the `.vscode/` item is moot and drop it.
