# -*- coding: utf-8 -*-
"""Audit the result-run tree against the runs that analysis code pins.

A run directory under ``results/{analysis}/{run_id}/`` is expensive to
regenerate (tens of GB of pickled model state), so it must never be pruned on
the basis of its date.  Several runs from March/April are still hard-coded as
defaults in paper-figure code while much newer runs are throwaway debug passes.
This module recovers that distinction mechanically:

  pinned       run_id appears literally in tracked source
  superseded   not pinned, and a later run covers the full patient cohort
  incomplete   fewer patient sub-directories than the cohort (aborted run)

Run it whenever the pinned set might have moved::

    python -m utils.audit_runs                 # markdown to stdout
    python -m utils.audit_runs --write         # refresh docs/results_index.md

The report is written to ``docs/`` rather than beside the data it describes:
``.gitignore`` excludes ``*results``, and git cannot re-include a file whose
parent directory is excluded, so a manifest under ``results/`` would itself be
untracked - the exact failure mode it exists to document.

Only the standard library is used, so it runs under any interpreter.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(MAIN_DIR, "results")
REPORT_PATH = os.path.join(MAIN_DIR, "docs", "results_index.md")

# A run_id always begins with an ISO date + wall-clock time; the trailing config
# slug is free-form (task, warp/align flags, model, loss, epochs).
RUN_ID_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}[A-Za-z0-9_.-]*")

# Where to look for pins.  results/ and figures/ are excluded deliberately: a
# run_id occurring inside its own output is not a reference to it.
SCAN_DIRS = ("figures_for_paper", "analysis", "tests", "notebooks", "report", "utils")
SCAN_SUFFIXES = (".py", ".ipynb", ".md")
SKIP_DIR_PARTS = ("__pycache__", ".ipynb_checkpoints", ".git")


def _iter_source_files():
    """Yield every tracked-ish source file that could pin a run."""
    for rel in SCAN_DIRS:
        root = os.path.join(MAIN_DIR, rel)
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_PARTS]
            for name in filenames:
                if name.endswith(SCAN_SUFFIXES):
                    yield os.path.join(dirpath, name)
    for name in os.listdir(MAIN_DIR):
        if name.endswith(".py"):
            yield os.path.join(MAIN_DIR, name)


def find_pins():
    """Map run_id -> sorted list of "relpath:lineno" that mention it."""
    pins = defaultdict(set)
    for path in _iter_source_files():
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError:
            continue
        rel = os.path.relpath(path, MAIN_DIR).replace(os.sep, "/")
        for lineno, line in enumerate(lines, 1):
            for match in RUN_ID_RE.findall(line):
                pins[match.rstrip("._-")].add("%s:%d" % (rel, lineno))
    return {k: sorted(v) for k, v in pins.items()}


def _dir_stats(path):
    """(n_patient_subdirs, total_bytes, n_files) for one run directory."""
    n_sub = total = n_files = 0
    for entry in os.scandir(path):
        if entry.is_dir():
            n_sub += 1
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            n_files += 1
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return n_sub, total, n_files


def scan_runs(with_size=True):
    """Discover every run directory grouped by analysis folder."""
    runs = defaultdict(list)
    if not os.path.isdir(RESULTS_ROOT):
        return runs
    for analysis in sorted(os.listdir(RESULTS_ROOT)):
        adir = os.path.join(RESULTS_ROOT, analysis)
        if not os.path.isdir(adir):
            continue
        for run_id in sorted(os.listdir(adir)):
            rdir = os.path.join(adir, run_id)
            if not os.path.isdir(rdir):
                continue
            if with_size:
                n_sub, total, n_files = _dir_stats(rdir)
            else:
                n_sub, total, n_files = -1, -1, -1
            runs[analysis].append(
                {"run_id": run_id, "n_patients": n_sub, "bytes": total, "n_files": n_files}
            )
    return runs


def _human(n):
    if n < 0:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "%.1f %s" % (n, unit) if unit != "B" else "%d B" % n
        n /= 1024.0


def task_of(run_id):
    """Which naming task a run belongs to.

    Cohorts differ by task - picture naming has 12 participants, auditory only
    the 6 with both tasks - so completeness must be judged within a task or
    every auditory run looks half-finished.
    """
    if "auditory_naming" in run_id:
        return "auditory"
    if "picture_naming" in run_id:
        return "picture"
    return "other"


def classify(run, pins, cohort_max):
    if run["run_id"] in pins:
        return "PINNED"
    if 0 <= run["n_patients"] < cohort_max:
        return "incomplete"
    return "unreferenced"


def build_report(with_size=True):
    pins = find_pins()
    runs = scan_runs(with_size=with_size)

    out = ["# Pinned and superseded result runs", ""]
    out.append("Generated by `python -m utils.audit_runs --write`. Do not hand-edit.")
    out.append("")
    out.append("`PINNED` = the run_id appears literally in tracked source, so paper")
    out.append("figures depend on it. **Never delete a PINNED run.** `incomplete` = fewer")
    out.append("patient sub-directories than the largest run in the same analysis, i.e. an")
    out.append("aborted pass. `unreferenced` means only that no code names it - confirm")
    out.append("against this table before pruning anything.")
    out.append("")

    referenced_but_missing = set(pins)

    for analysis in sorted(runs):
        rows = runs[analysis]
        if not rows:
            continue
        cohort = {}
        for r in rows:
            t = task_of(r["run_id"])
            cohort[t] = max(cohort.get(t, 0), r["n_patients"])
        out.append("## %s" % analysis)
        out.append("")
        out.append("| run_id | task | status | patients | size | pinned at |")
        out.append("|---|---|---|---|---|---|")
        for r in rows:
            task = task_of(r["run_id"])
            status = classify(r, pins, cohort[task])
            where = ", ".join("`%s`" % p for p in pins.get(r["run_id"], [])) or "-"
            referenced_but_missing.discard(r["run_id"])
            pats = "?" if r["n_patients"] < 0 else str(r["n_patients"])
            out.append(
                "| `%s` | %s | %s | %s | %s | %s |"
                % (r["run_id"], task, status, pats, _human(r["bytes"]), where)
            )
        out.append("")

    # Some analyses write loose per-participant CSVs rather than run directories.
    # Without this they would produce no rows at all and silently vanish from the
    # index -- the opposite of what it is for.
    flat = []
    if os.path.isdir(RESULTS_ROOT):
        for analysis in sorted(os.listdir(RESULTS_ROOT)):
            adir = os.path.join(RESULTS_ROOT, analysis)
            if not os.path.isdir(adir) or runs.get(analysis):
                continue
            n = total = 0
            for dirpath, _d, filenames in os.walk(adir):
                for name in filenames:
                    n += 1
                    try:
                        total += os.path.getsize(os.path.join(dirpath, name))
                    except OSError:
                        pass
            if n:
                flat.append((analysis, n, total))
    if flat:
        out.append("## Analyses stored as loose files (no run directories)")
        out.append("")
        out.append("| analysis | files | size |")
        out.append("|---|---|---|")
        for analysis, n, total in flat:
            out.append("| `%s` | %d | %s |" % (analysis, n, _human(total)))
        out.append("")

    stale_pins = sorted(p for p in referenced_but_missing if RUN_ID_RE.fullmatch(p))
    if stale_pins:
        out.append("## Referenced in code but not present on disk")
        out.append("")
        out.append("Either the run was deleted or the reference is a stale example in a")
        out.append("docstring/notebook output. Worth resolving so the pin set stays honest.")
        out.append("")
        for run_id in stale_pins:
            out.append("- `%s` - %s" % (run_id, ", ".join("`%s`" % p for p in pins[run_id])))
        out.append("")

    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true",
                    help="write docs/results_index.md instead of printing")
    ap.add_argument("--no-size", action="store_true",
                    help="skip directory sizing (much faster on the 169 GB tree)")
    args = ap.parse_args(argv)

    report = build_report(with_size=not args.no_size)
    if args.write:
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
        print("wrote %s" % REPORT_PATH)
    else:
        sys.stdout.write(report + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
