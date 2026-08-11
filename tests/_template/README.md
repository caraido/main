# `tests/_template/` — start a pilot in one copy

```bash
cp -r tests/_template tests/<slug>          # git bash; PowerShell: Copy-Item -Recurse
python -m tests.<slug>.run --why "the one question this pilot answers"
```

This directory exists because starting a pilot used to feel expensive, so pilot-grade work
accumulated in `tmp/` instead — untracked, unnamed, with its outputs sitting beside its
code. Copying four small files is cheaper than that, which is the whole argument.

Nothing here is imported by anything. Edit the copy, never this directory.

## The pilot contract

**Throwaway.** `tests/` is stage 1 of `tests/ → analysis/ → figures_for_paper/`
(`docs/repo_layout.md`). Nothing here is expected to keep working. A pilot that stops
being interesting is deleted or moved to `_archive/`; it is not maintained in place.

**Nothing outside `tests/` may import from `tests/`.** That is what makes the folder
throwaway rather than load-bearing, and it is the rule the 2026-07 reorganisation exists
to restore — five figure pipelines had come to import from the old `tests/`, so "archive
the tests folder" was unsafe. Import *into* a pilot freely (`utils`, `analysis`,
`figures_for_paper/paper_common.py`, the root training scripts); never the other way.

**Graduate when something depends on you.** Promotion is about who depends on you, not
about age or how finished the code feels. The moment a figure script, a report, or another
analysis wants to import a pilot module, move it to `analysis/<topic>/` and give it a row
in `analysis/README.md`. Until then it stays here.

**Outputs go to `results/<slug>/` via `utils.paths.results_dir` — never a hand-composed
path.** Three competing output roots plus a relative fallback that resolved against the
working directory is how one analysis suite ended up split across two directories, half of
it outside the repository. `run.py` gets every path from `utils.run_context.open_run`, so
there is no path to compose by hand. `scripts/validate_agent_config.py` fails a relative
output root, and `results/` is a `.gitignore`d tree, so a wrong path fails silently
otherwise.

A run's report, figures and source data belong **to the run**:
`results/<slug>/<run_id>/{figures,source_data,report}/`. They describe one run and should
die with it. `figures/<slug>/` is for genuinely cross-run scratch only.

**Write a manifest, which means using `open_run`.** It is not optional and cannot be
forgotten, because it happens on entry: `meta.json` is written before any work runs, so a
pilot killed at minute one still records its command line, git commit and question. It is
also what makes the run visible to `python -m utils.audit_runs --status` *while it is
still going*, and what keeps it out of the `unreferenced` bucket the pruning plan deletes.

**`--why` is required on purpose.** One line, captured at launch when the answer is still
known. That plus `--supersedes` is the entire human writing burden per run.

## Files

| File | What it is |
|---|---|
| `run.py` | the CLI entry point: argparse → `open_run` → per-participant loop → `run.headline()` |
| `__init__.py` | makes `python -m tests.<slug>.run` work; say what the pilot asks |
| `README.md` | this file — replace it with what your pilot asks and what it found |

Split into more modules only when `run.py` stops fitting on a screen.
`tests/auditory_alignment/` is the reference implementation of a pilot that did
(`config.py`, `metrics.py`, `stats.py`, `figures.py`, `report.py`, `run.py`).

## When the pilot ends

- **It worked and something wants to import it** → move it to `analysis/<topic>/`, add a
  row to `analysis/README.md`, and re-point the run ids the figures use.
- **It answered a question and stops there** → record the answer in `docs/experiments/`
  (schema in `docs/experiments/README.md`; entries stay under 80 lines) and delete the
  code. The entry is the deliverable, not the script.
- **It did not pan out** → `_archive/`, with the reason in `_archive/README.md`.

In all three cases the `results/<slug>/` tree it produced is governed by
`docs/results_index.md` and the `results-hygiene` skill — never delete a run directory
without checking there first.
