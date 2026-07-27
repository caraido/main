---
name: python-refactor
description: 'Safe Python deduplication and utility extraction in this repository. Use for repeated code blocks, shared helper creation, script-to-utils refactors, and post-change validation (especially analysis/ pipelines and utils/).'
---

# Python Refactor Workflow

## Purpose
Use this skill to perform safe, repeatable Python refactors that reduce duplication while preserving behavior.

This workflow is optimized for this repository's analysis scripts and utility modules.

## When To Use
- Multiple Python files repeat near-identical functions/classes.
- A script contains helper logic that should live in a shared utils module.
- You want to standardize preprocessing/alignment/decoder patterns.
- You need a conservative refactor with explicit validation.

## Inputs
- Scope folder (for example: `analysis/`, `utils/`, `figures_for_paper/`, `report/`).
- Refactor intent:
  - `dedupe`: remove repeated blocks across files.
  - `extract-utils`: move helpers into a utility module.
  - `cleanup`: simplify and normalize existing helper usage.
- Optional safety rule: keep public APIs and script entrypoints unchanged.

## Workflow
1. Discover repeated code
- Search Python files in scope.
- Identify duplicated function/class definitions and repeated pipelines.
- Prioritize high-impact duplicates used in >=2 files.

2. Select extraction target
- Prefer an existing scope-local utility file (for example `utils.py`).
- If none exists, create one in a predictable location and keep names stable.

3. Extract shared helpers
- Move duplicated logic into utility functions/classes.
- Keep signatures compatible with current call sites.
- Add concise docstrings and minimal comments where behavior is non-obvious.

4. Refactor call sites
- Replace local duplicates with imports from the utility module.
- Preserve external behavior, data flow, and expected outputs.
- Avoid unrelated formatting or structural changes.

5. Validate
- Run static error checks on all touched files.
- Ensure imports resolve and no duplicate old definitions remain.
- If tests or notebooks are relevant, run the smallest sanity check available.

6. Report
- Summarize what was extracted and where.
- List touched files and compatibility notes.
- Mention any residual risks or pre-existing issues not introduced by the refactor.

## Repository Conventions
- Prefer conservative edits over broad rewrites.
- Do not alter unrelated files in a dirty working tree.
- Keep utility APIs explicit and reusable.
- Favor ASCII and existing code style.
- **Grep for importers before moving or renaming anything.** In this repo, folder and age
  are poor proxies for whether something matters: `cross_task_regression.py` and
  `helpers/_cross_patient_helpers.py` were both classified as dead and both turned out to
  be libraries behind paper figures. The breakage only surfaces on regeneration.
  `analysis/README.md` records which modules are **library** / **pipeline** /
  **supplementary** and who depends on each.
- Never hand-compose a results path in extracted code — use
  `utils.paths.results_dir("<analysis>")`.
- `python -m py_compile` is the only check available; there is no test suite.

## Validation Checklist
See [validation checklist](./references/validation-checklist.md).

## Done Criteria
- Duplicate logic is centralized.
- Call sites use shared helpers.
- Touched files pass diagnostics.
- Final summary includes risks and follow-up suggestions.
