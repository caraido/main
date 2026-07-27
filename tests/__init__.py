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

This folder is currently empty, which is the intended steady state.

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

Proper unit tests (pytest-style) live in `main/pytest/`, not here.
"""
