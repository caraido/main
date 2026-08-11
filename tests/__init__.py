# -*- coding: utf-8 -*-
"""tests/ -- the pilot sandbox. Stage 1 of the analysis lifecycle.

    tests/  ->  analysis/  ->  figures_for_paper/
    pilot       promoted        published
                    |
                    +-> _archive/   (piloted, did not pan out)

New ideas start here: a standalone CLI script with its own argparse parser,
runnable as `python -m tests.<name>`. Nothing here is expected to keep working,
and nothing outside tests/ should import from it -- that is the whole point of
the folder. Once something is worth depending on, promote it to `analysis/` and
record its status in `analysis/README.md`.

This folder is empty *between* pilots, not permanently. That is the intended steady
state: a live pilot here is the lifecycle working as designed, and an empty folder
means nothing is in flight. `tests/auditory_alignment/` is currently live.

Why it was emptied (2026-07): everything that had accumulated here was either
paper-critical or dead, with nothing in the layout to tell the two apart. Five
production figure pipelines imported from `tests/`, and two read their source
data out of `tests/results/`, so "archive the tests folder" was unsafe. The
promoted code moved to `analysis/`, the dead pilots to `_archive/`, and all
analysis output to `results/<analysis>/`.

Results
-------
Write output via `utils.paths.results_dir("<analysis>")`, never a hand-composed
path. Three competing output roots (`main/results/`, `main/test_results/`,
`main/tests/results/`) plus a relative fallback that escaped to the project root
is how one analysis suite ended up split across two directories, half of it
outside the repository entirely.

Reports, figures and source data belong to the run that produced them, so a pilot
writes them *inside* its own results tree -- `results/<analysis>/{figures,source_data}/`
and the report HTML beside them. `tests/auditory_alignment/` is the reference
implementation. See `docs/repo_layout.md` for the full output contract.

There is no unit-test suite. This folder is the pilot stage of the analysis
lifecycle, not a test runner: `main/` has no test files, no `conftest.py`, and no
pytest/ruff/black/pre-commit configuration. Earlier revisions of this docstring and
of `README.md` described a `main/pytest/` directory; it has never existed. What
stands in for a test suite is documented in `docs/agent-context/validation.md`.
"""
