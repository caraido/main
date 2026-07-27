# Validation — what "done" means here

Routed from `AGENTS.md`. Run the checks that apply, in order, and report what each one
printed. Reporting a check you did not run as passed is the one unrecoverable failure mode
in this document.

## There is no test suite

`main/` has no test files, no `conftest.py`, and no pytest, ruff, black, or pre-commit
configuration. `README.md` §6 and `tests/__init__.py` describe a `main/pytest/` directory
that does not exist. `tests/` is the **pilot stage** of the analysis lifecycle, not a test
suite.

Consequence: static compilation plus artifact diffing is the whole safety net. Treat it
accordingly.

## The hierarchy

### 1. Static check — always

```bash
python -m py_compile <every .py file you touched>
```

Confirm imports resolve and no duplicate old definition survives a refactor.

### 2. Figure-pipeline regression — whenever a figure or its inputs changed

```bash
python figures_for_paper/<analysis>/<analysis>_panels.py     # cwd = main/
git diff --stat figures_for_paper/<analysis>/source_data/
```

**No diff is the pass condition.** An unexplained diff is a regression, not an update.
Rendered `.pdf` files always differ (embedded timestamps) — ignore them; the tracked CSVs
are the real signal. Confirm both `.png` and `.pdf` exist for every panel, and that no CSV
was written outside `{analysis}/source_data/`.

If the diff is intended, say what changed and why in the same breath as reporting it.

### 3. Results-path check — whenever a path, run id, or output location changed

```bash
python -m utils.audit_runs --write
git diff docs/results_index.md
```

Confirm no new `incomplete` entry and no new "referenced in code but not present" entry
appeared. Never resolve a run-id disagreement by hand-picking; resolve it here.

### 4. Data validation — whenever a loader, exclusion, or channel mapping changed

Confirm channel counts, `primary_roi` coverage, and per-patient trial counts against
`channel-and-roi-naming.md`. A silent count change is how the LH shank bug survived: names
were reconstructed from the pre-exclusion column, so `ch{N}` pointed at the wrong
electrode, and nothing failed loudly.

### 5. Statistical checks — whenever a number will be reported

Per `scientific-integrity.md`: state N, the test, the correction, and the achievable p
floor. Verify a metric against its null rather than its absolute value. Confirm any in-plot
`p < …` annotation is derived from the actual threshold, not hard-coded.

### 6. Visual inspection — for any rendered figure

Render and look at it. Panel letters flow left→right then top→bottom; significance rasters
sit clear of the data; no participant appears by initials.

### 7. Reproducibility — before anything is called final

The figure regenerates from its own `source_data/` in a clean checkout, and every quoted
number traces to a run id in `docs/results_index.md`.

## Agent-configuration changes

Changes to `AGENTS.md`, `CLAUDE.md`, `.agents/skills/`, `docs/agent-context/`, or
`scripts/` are validated by:

```bash
python scripts/sync_agent_skills.py     # regenerate the Claude mirror
python scripts/validate_agent_config.py # must exit 0
```

## When a check cannot run

Say so, name the check, and give the reason. Acceptable reasons include: the data are not
on this machine, the run would take hours, the environment lacks a dependency, or the check
requires a decision only Alec can make.

Never substitute inspection for execution. "The code looks correct" is not a validation
result, and neither is a check that was skipped and left unmentioned.
