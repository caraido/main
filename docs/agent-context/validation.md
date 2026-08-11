# Validation — what "done" means here

Routed from `AGENTS.md`. Run the checks that apply, in order, and report what each one
printed. Reporting a check you did not run as passed is the one unrecoverable failure mode
in this document.

## There is no test suite

`main/` has no test files, no `conftest.py`, and no pytest, ruff, black, or pre-commit
configuration. A `main/pytest/` directory that never existed was documented in `README.md`
and `tests/__init__.py` until 2026-08-11; both are corrected. `tests/` is the **pilot stage**
of the analysis lifecycle, not a test suite.

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

### What that script now enforces beyond the two-tier layout

Added 2026-08-11. All five are read-only and stdlib-only, and all five were checked
against a deliberate violation to confirm they can actually go red.

| Check | Fails on |
|---|---|
| `check_output_paths` | a **relative** output root (`os.path.join('results', …)`, `base_dir='results'`) or a path naming the deleted `main/tests/results`. Narrow on purpose: an absolute hand-composed root is style, a relative one is the bug that put output outside the repository. Skips comments and docstrings — this repo documents its traps at length, and a checker that reads its own documentation as a violation gets switched off. `utils/paths.py` and `utils/audit_runs.py` are exempt because they *define* the roots |
| `check_experiments` | an entry in `docs/experiments/` with unparseable frontmatter, a bad `kind`/`status`, a duplicate `id`, `status: answered` and an empty `answer:`, or **more than 120 lines** — the mechanical form of "this is a record, not a log". A cited run id missing from `results/` is a WARN, matched by prefix since entries elide long ids |
| `check_scripts_documented` | a `scripts/*.py` absent from `scripts/README.md` |
| `check_shared_modules_adopted` | a module in `SHARED_MODULES` with **zero importers**. The retired `utils/cli` module is why it exists: 7.5 KB of shared argparse builders, written to be adopted, never imported once, still described as live in `README.md`. It was **deleted 2026-08-11** and the `KNOWN_UNADOPTED` allowlist is now **empty** — keep it that way; an entry there is an unmade adopt-or-delete decision, not an exemption. A package is matched by its directory name, not `__init__` |
| `check_notebooks` | **(a)** a data file a notebook *writes* that tracked non-notebook `.py` also names, and **(b)** a `results/` run directory pinned only from a `.ipynb`. Notebooks are exploratory or demo output, so nothing outside one may depend on what it writes. Both rules fire today and both entries are allowlisted: `cache_null_means_100ep.csv` in `KNOWN_NOTEBOOK_SHARED` (two writers on one filename in a gitignored directory, with a paper figure reading it), and `2026-03-27_12-35-02_KRR_cosine_50ep` in `KNOWN_NOTEBOOK_PINS` (13.5 GB named only by `semantic_regression_retrieval_metrics_comparison.ipynb`). Pins come from `utils.audit_runs.find_pins`, never a second parser. Two deliberate narrownesses: only **code cells** are read for (a), because a saved output cell renders a past run rather than stating what the notebook does; and the on-disk test in (b) is **exact, not prefix**, because output cells render ids truncated and a fragment otherwise prefix-matches a run that *is* pinned from `utils/config.py` |

`KNOWN_LEGACY`, `KNOWN_UNADOPTED`, `KNOWN_NOTEBOOK_SHARED` and `KNOWN_NOTEBOOK_PINS` exist
so the checks exit 0 the day they land. A gate that goes red immediately is a gate someone
disables; every one of those lists may only shrink.

### Answering "is it done?"

```bash
python -m utils.audit_runs --status     # running now + last 14 days; ~2 s, no sizing
python -m utils.audit_runs --json       # the same rows, for an agent
```

Distinct from `--write`, which walks the whole tree to size it. `--status` reads one
`meta.json` per run, so it stays usable *while* a run is going.

## When a check cannot run

Say so, name the check, and give the reason. Acceptable reasons include: the data are not
on this machine, the run would take hours, the environment lacks a dependency, or the check
requires a decision only Alec can make.

Never substitute inspection for execution. "The code looks correct" is not a validation
result, and neither is a check that was skipped and left unmentioned.
