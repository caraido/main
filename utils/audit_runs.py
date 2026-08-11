# -*- coding: utf-8 -*-
"""Audit the result-run tree against the runs that analysis code pins.

A run directory under ``results/{analysis}/{run_id}/`` is expensive to
regenerate (tens of GB of pickled model state), so it must never be pruned on
the basis of its date.  Several runs from March/April are still hard-coded as
defaults in paper-figure code while much newer runs are throwaway debug passes.
This module recovers that distinction mechanically:

  PINNED       the directory name appears literally in tracked source
  incomplete   fewer patient sub-directories than the cohort (aborted run)
  unreferenced no tracked source names it
  derived      not a run at all -- a run's own output subdirectory
  per-patient  not a run at all -- the pre-run-directory ``<root>/<PATIENT>/`` layout

Two of those five were added 2026-08-10 to stop the ledger reporting things that were
never runs as ``incomplete``, i.e. as aborted passes worth deleting.  The pilot's own
``results/auditory_alignment/{figures,source_data}/`` read that way, and so did every
per-patient directory written before ``cross_task_*`` grew run folders on 2026-07-30.

Pins come from TWO reference classes, and the second one also dates from 2026-08-10:

  1. a timestamped run id matched by ``RUN_ID_RE``
  2. a non-timestamped *directory name* matched as a whole token

Class 2 exists because class 1 could not see a directory whose name has no date in it, so
such a directory always read ``unreferenced`` no matter how much code named it.  Measured
consequences at the time it was added: ``original_KRR_l2_50ep`` (13.5 GB) is read by
``notebooks/semantic_regression_retrieval_metrics_comparison.ipynb`` and was staged for
deletion in ``docs/pruning_candidates_2026-08.md``; ``balance_none`` is read by
``figures_for_paper/cross_task/compute_cross_task_data.py``, a paper pipeline.  Both read
``unreferenced``.  A false "referenced" costs disk; a false "unreferenced" costs
irreplaceable compute, so this errs toward referenced.

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
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(MAIN_DIR, "results")
FIGURES_ROOT = os.path.join(MAIN_DIR, "figures")
REPORT_PATH = os.path.join(MAIN_DIR, "docs", "results_index.md")

# Direct children of results/<analysis>/ that are a run's OWN OUTPUT, not runs.  Without
# this they are counted as run directories with zero patient sub-directories and therefore
# classified ``incomplete`` -- which reads as "aborted pass, safe to delete" and points at
# the pilot's report, figures and source data.
DERIVED_DIR_NAMES = frozenset({
    "figures", "source_data", "report", "reports", "logs", "comparison_figures",
})

# The pre-run-directory layout: cross_task_{cotrain,regression,transfer} wrote straight to
# <root>/<PATIENT>/ until 2026-07-30, and label_generation/ still holds one directory per
# participant of atlas provenance.  Structural label only -- it says what the directory IS,
# not whether it is worth keeping.
PATIENT_DIR_RE = re.compile(r"^[A-Z]{2,4}$")

# A directory name is eligible to be pinned as a literal only if it is distinctive enough
# that a whole-token match is evidence rather than coincidence: long, and containing an
# underscore.  Patient initials (AA, WBH) and bare words are excluded by construction --
# they appear in every patient list in the repository and would pin everything.
MIN_LITERAL_LEN = 6

# A run_id always begins with an ISO date + wall-clock time; the trailing config
# slug is free-form (task, warp/align flags, model, loss, epochs).
RUN_ID_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}[A-Za-z0-9_.-]*")

# Where to look for pins.  results/ and figures/ are excluded deliberately: a
# run_id occurring inside its own output is not a reference to it.
#
# ADDING A TOP-LEVEL DIRECTORY THAT MAY NAME A RUN ID MEANS EXTENDING THIS TUPLE
# IN THE SAME COMMIT.  A fixture, verification, or research tree left outside the
# scan makes every run it pins read as ``unreferenced``, and unreferenced runs are
# what the pruning plan in docs/repo_layout.md deletes -- roughly 50 GB of pinned
# output is in range.  Same trap as moving utils/config.py to a root-level JSON.
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


def _literal_pin_candidates(runs):
    """Directory names worth searching for as literals, with the traps excluded.

    Excluded, in order of how badly each would mislead:

    * anything ``RUN_ID_RE`` already handles -- class 1 owns those;
    * anything shorter than ``MIN_LITERAL_LEN`` or without an underscore, which is how
      patient initials are kept out;
    * a name that is also an analysis directory name (``results/VB/semantic_regression``
      would otherwise pin on the ~200 mentions of the *analysis* ``semantic_regression``,
      which say nothing about that legacy directory);
    * ``DERIVED_DIR_NAMES``, which are classified before pins are consulted anyway.
    """
    reserved = set(SCAN_DIRS) | set(DERIVED_DIR_NAMES)
    if os.path.isdir(RESULTS_ROOT):
        reserved |= set(os.listdir(RESULTS_ROOT))
    out = set()
    for rows in runs.values():
        for run in rows:
            name = run["run_id"]
            if RUN_ID_RE.fullmatch(name) or name in reserved:
                continue
            if len(name) < MIN_LITERAL_LEN or "_" not in name:
                continue
            out.add(name)
    return out


def find_dir_literal_pins(names):
    """Map directory-name -> sorted "relpath:lineno" for names appearing as whole tokens.

    Whole-token means not embedded in a longer identifier or filename stem, so
    ``balance_none`` matches ``os.path.join(RESULTS, "balance_none")`` and
    ``results/x/balance_none/y.csv`` but not ``balance_none_v2``.

    Scans exactly the same files as ``find_pins`` -- in particular NOT ``results/`` or
    ``figures/``, because a directory naming itself inside its own output is not a
    reference to it.
    """
    if not names:
        return {}
    pattern = re.compile(
        r"(?<![A-Za-z0-9_.\-])(" + "|".join(re.escape(n) for n in sorted(names))
        + r")(?![A-Za-z0-9_.\-])"
    )
    pins = defaultdict(set)
    for path in _iter_source_files():
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError:
            continue
        rel = os.path.relpath(path, MAIN_DIR).replace(os.sep, "/")
        for lineno, line in enumerate(lines, 1):
            for match in pattern.findall(line):
                pins[match].add("%s:%d" % (rel, lineno))
    return {k: sorted(v) for k, v in pins.items()}


def _dir_stats(path):
    """(n_patient_subdirs, total_bytes, n_files) for one run directory.

    ``DERIVED_DIR_NAMES`` are excluded from the subdirectory count.  A run's own
    ``report/`` is not a participant, and counting it as one inflates that run's patient
    count by one -- which inflates the cohort maximum for its whole task bucket, which
    then reports every genuinely complete sibling as ``incomplete``.  Measured instance:
    ``original_KRR_l2_50ep`` has no ``meta.json``, so it fell back to a raw count of 13
    (12 participants plus ``report/``), and two complete 12-participant runs totalling
    43.7 GB read as aborted passes because of it.
    """
    n_sub = total = n_files = 0
    for entry in os.scandir(path):
        if entry.is_dir() and entry.name not in DERIVED_DIR_NAMES:
            n_sub += 1
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            n_files += 1
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return n_sub, total, n_files


def _metadata_patient_count(path, fallback):
    """Prefer the run manifest's patient list over counting every subdirectory.

    Semantic-regression runs also contain a top-level ``report/`` directory, so a raw
    directory count is one too high once all participant outputs are present.  Older
    pipelines have no manifest; for those, retain the legacy directory-count fallback.
    """
    meta_path = os.path.join(path, "meta.json")
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError, TypeError):
        return fallback
    for key in ("succeeded_patients", "patients"):
        patients = meta.get(key)
        if isinstance(patients, list):
            return len(set(map(str, patients)))
    return fallback


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
                n_sub = _metadata_patient_count(rdir, n_sub)
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


#: Where the tracked experiment record lives.  Read as a SEPARATE, WEAKER reference class
#: -- never added to ``SCAN_DIRS``.  ``docs/`` must stay out of the pin scan because
#: ``docs/results_index.md`` lists every run id in the repository; scanning it would mark
#: every run ``PINNED`` and destroy the distinction the ledger exists to draw.
JOURNAL_DIR = os.path.join(MAIN_DIR, "docs", "experiments")


def find_journaled():
    """Map run_id -> ["docs/experiments/<file>", ...] for ids named in the record.

    A ``journaled`` run is still prunable -- an entry is a note that a question was asked,
    not a declaration that paper output depends on the run. What it buys is that
    ``results-hygiene`` can update the entry to record the deletion *before* deleting,
    instead of silently invalidating a record that cites the run.
    """
    out = defaultdict(set)
    if not os.path.isdir(JOURNAL_DIR):
        return {}
    for name in sorted(os.listdir(JOURNAL_DIR)):
        if not name.endswith(".md") or name == "README.md":
            continue
        path = os.path.join(JOURNAL_DIR, name)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        for match in RUN_ID_RE.findall(text):
            out[match.rstrip("._-")].add("docs/experiments/%s" % name)
    return {k: sorted(v) for k, v in out.items()}


#: The generated table in ``docs/experiments/README.md`` sits between these. Everything
#: outside them is hand-written and preserved: the schema and the rules are prose a
#: generator has no business rewriting.
_INDEX_BEGIN = "<!-- BEGIN GENERATED INDEX -- python -m utils.audit_runs --write -->"
_INDEX_END = "<!-- END GENERATED INDEX -->"


def _frontmatter(text):
    """Top-level keys of a leading ``---`` block. Nested values are skipped.

    A deliberate 20-line parser rather than PyYAML: this module is stdlib-only so it runs
    in a bare checkout, and there is no tracked dependency manifest in this repository to
    declare a third-party import in.
    """
    lines = text.lstrip("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    keys = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return keys
        if not line.strip() or line.startswith((" ", "\t", "#")):
            continue
        key, sep, value = line.partition(":")
        if sep:
            keys[key.strip()] = value.strip().strip("'\"")
    return {}


def build_experiments_index():
    """The one table an agent reads to answer "what have we already tried?"."""
    rows = []
    if os.path.isdir(JOURNAL_DIR):
        for name in sorted(os.listdir(JOURNAL_DIR)):
            if not name.endswith(".md") or name == "README.md":
                continue
            try:
                with open(os.path.join(JOURNAL_DIR, name), "r",
                          encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            fm = _frontmatter(text)
            if not fm:
                continue
            n_runs = len(set(RUN_ID_RE.findall(text)))
            rows.append((fm.get("id", "?"), fm.get("kind", "?"), fm.get("status", "?"),
                         fm.get("title", name), fm.get("analysis", "-"), n_runs, name))
    out = [_INDEX_BEGIN, ""]
    if not rows:
        out += ["*No entries yet.*", ""]
    else:
        out.append("| id | kind | status | title | analysis | runs cited |")
        out.append("|---|---|---|---|---|---|")
        for r in sorted(rows):
            out.append("| [%s](%s) | %s | %s | %s | `%s` | %d |"
                       % (r[0], r[6], r[1], r[2], r[3], r[4], r[5]))
        out.append("")
    out.append(_INDEX_END)
    return "\n".join(out)


def write_experiments_index():
    """Replace the generated block in ``docs/experiments/README.md``. Returns a message."""
    path = os.path.join(JOURNAL_DIR, "README.md")
    if not os.path.isfile(path):
        return "skipped %s (absent)" % os.path.relpath(path, MAIN_DIR)
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    block = build_experiments_index()
    if _INDEX_BEGIN in text and _INDEX_END in text:
        head = text.split(_INDEX_BEGIN)[0]
        tail = text.split(_INDEX_END, 1)[1]
        new = head + block + tail
    else:
        new = text.rstrip("\n") + "\n\n" + block + "\n"
    if new == text:
        return "unchanged %s" % os.path.relpath(path, MAIN_DIR)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new)
    return "wrote %s" % os.path.relpath(path, MAIN_DIR)


def classify(run, pins, cohort_max, journaled=None):
    """Status for one directory.  Order matters.

    "Is this even a run?" is asked before "is it referenced?", because the two
    not-a-run cases have zero patient sub-directories and would otherwise fall through
    to ``incomplete`` -- the classification the pruning plan treats as an aborted pass.

    Ranking: ``PINNED`` > ``journaled`` > ``incomplete`` > ``unreferenced``.
    """
    name = run["run_id"]
    if name in DERIVED_DIR_NAMES:
        return "derived"
    if PATIENT_DIR_RE.match(name):
        return "per-patient"
    if name in pins:
        return "PINNED"
    if journaled and name in journaled:
        return "journaled"
    if 0 <= run["n_patients"] < cohort_max:
        return "incomplete"
    return "unreferenced"


def build_cohort_coverage(rows):
    """Which participants each run actually produced, and which it is missing.

    The cohort has grown four times (CP, KAW, then PV and SE together) and each growth
    silently superseded a set of runs. "Which runs predate PV and SE" was answered by
    memory; this answers it from the manifests. The reference set is the union of every
    participant seen in that (analysis, task), not a hard-coded list, so it cannot go
    stale the way ``SHARED_PATIENTS`` did in five files.
    """
    groups = defaultdict(list)
    for r in rows:
        if r["succeeded_patients"] or r["patients"]:
            groups[(r["analysis"], r["task"])].append(r)

    out = []
    if not groups:
        return out
    out.append("## Cohort coverage")
    out.append("")
    out.append("Per run, from its own `meta.json`. `missing` is relative to the union of")
    out.append("every participant seen in that analysis and task -- derived, never a")
    out.append("hard-coded list.")
    out.append("")
    for (analysis, task) in sorted(groups):
        rs = groups[(analysis, task)]
        universe = set()
        for r in rs:
            universe |= set(r["succeeded_patients"]) | set(r["patients"])
        out.append("### %s / %s  (%d participants seen)" % (analysis, task, len(universe)))
        out.append("")
        out.append("| run | n | missing |")
        out.append("|---|---|---|")
        for r in sorted(rs, key=lambda r: r["run_id"], reverse=True):
            got = set(r["succeeded_patients"]) or set(r["patients"])
            missing = sorted(universe - got)
            out.append("| `%s` | %d | %s |" % (
                r["run_id"], len(got),
                ", ".join(missing) if missing else "—"))
        out.append("")
    return out


def scan_figures(with_size=True):
    """Inventory ``figures/`` against ``results/``.  This is NOT a pin scan.

    ``figures/`` has never had a ledger: ``scan_runs`` walks ``results/`` only, so several
    GB of per-run figure output could not be classified at all, and an orphan -- a figure
    directory whose run has already been deleted -- was indistinguishable from a live one.

    **The per-run mirror layout ended on 2026-08-11.** The three root pipelines now write
    to ``results/<analysis>/<run_id>/figures/``, so a run owns its own plots and there is
    no twin to inventory. 31 twins were relocated then; what is left here is 11 ORPHANS,
    whose runs are already gone. A ``twin`` row appearing now therefore means a *stale
    duplicate* -- output of a run that also has figures inside it -- not a live mirror.
    ``figures/`` keeps only cross-run, throwaway output (and some of it, notably
    ``figures/open_vocab_retrieval/source_data/``, is read by paper pipelines: see the
    untracked-inputs table in ``docs/repo_layout.md`` before deleting anything here).

    **Do not "fix" this by adding ``figures`` to ``SCAN_DIRS``.** That tuple is the list of
    places a run id may be *referenced from*, and ``figures/`` is excluded from it on
    purpose: a run id occurring inside its own output is not a reference to it. This
    function is an inventory of a different tree, not another source of pins.

    Returns ``(mirrored, unmirrored)``:
      mirrored   analysis -> rows for each ``figures/<analysis>/<name>/``, each carrying
                 whether the ``results/`` twin exists
      unmirrored rows for ``figures/<x>/`` where no ``results/<x>/`` exists at all -- these
                 are not run mirrors and are inventoried only as a total
    """
    mirrored, unmirrored = defaultdict(list), []
    if not os.path.isdir(FIGURES_ROOT):
        return mirrored, unmirrored
    for analysis in sorted(os.listdir(FIGURES_ROOT)):
        fdir = os.path.join(FIGURES_ROOT, analysis)
        if not os.path.isdir(fdir):
            continue
        rdir = os.path.join(RESULTS_ROOT, analysis)
        if not os.path.isdir(rdir):
            _n, total, n_files = _dir_stats(fdir) if with_size else (0, -1, -1)
            unmirrored.append({"analysis": analysis, "bytes": total, "n_files": n_files})
            continue
        for name in sorted(os.listdir(fdir)):
            fpath = os.path.join(fdir, name)
            if not os.path.isdir(fpath):
                continue
            _n, total, n_files = _dir_stats(fpath) if with_size else (0, -1, -1)
            mirrored[analysis].append({
                "run_id": name,
                "bytes": total,
                "n_files": n_files,
                "has_twin": os.path.isdir(os.path.join(rdir, name)),
            })
    return mirrored, unmirrored


def _read_meta(path):
    """The run's ``meta.json`` as a dict, or ``{}``."""
    try:
        with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError, TypeError):
        return {}


def _rel(path):
    """Display a recorded path relative to ``main/`` when it is inside it.

    ``log_path`` is absolute in runs from the three root pipelines, because their own
    ``_build_meta`` supplies it and the caller wins over ``open_run``'s POSIX-relative
    version -- deliberate, since overwriting it would have broken the guarantee that a
    migration only ever adds manifest keys. So normalise for DISPLAY, never in the file.
    """
    if not path:
        return path
    try:
        if os.path.isabs(path) and os.path.commonpath([os.path.abspath(path), MAIN_DIR]) == MAIN_DIR:
            return os.path.relpath(path, MAIN_DIR).replace(os.sep, "/")
    except (ValueError, OSError):
        pass
    return str(path).replace(os.sep, "/")


def _fmt_duration(seconds):
    if seconds is None:
        return "-"
    seconds = int(float(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return "%dh%02dm" % (h, m)
    if m:
        return "%dm%02ds" % (m, s)
    return "%ds" % s


def collect_runs():
    """One structured row per run directory that carries a ``meta.json``.

    **Deliberately does no directory sizing.** ``scan_runs`` walks every file to total
    bytes, which on this tree means stat-ing a six-figure number of OneDrive placeholders;
    that is fine for the once-a-change ledger and far too slow for "is it done?". This
    reads one small JSON per run instead, which is what makes ``--status`` usable while a
    run is still going.
    """
    rows = []
    if not os.path.isdir(RESULTS_ROOT):
        return rows
    for analysis in sorted(os.listdir(RESULTS_ROOT)):
        adir = os.path.join(RESULTS_ROOT, analysis)
        if not os.path.isdir(adir):
            continue
        for run_id in sorted(os.listdir(adir)):
            rdir = os.path.join(adir, run_id)
            if not os.path.isdir(rdir):
                continue
            meta = _read_meta(rdir)
            if not meta:
                continue
            succeeded = meta.get("succeeded_patients") or []
            requested = meta.get("patients") or []
            rows.append({
                "analysis": analysis,
                "run_id": run_id,
                "status": meta.get("status"),
                "stage": meta.get("stage"),
                "task": task_of(run_id),
                "started_at": meta.get("started_at") or meta.get("timestamp_local"),
                "finished_at": meta.get("finished_at"),
                "duration_sec": meta.get("duration_sec"),
                "n_succeeded": len(succeeded),
                "n_requested": len(requested),
                "succeeded_patients": list(succeeded),
                "patients": list(requested),
                "question": meta.get("question"),
                "supersedes": meta.get("supersedes"),
                "log_path": meta.get("log_path"),
                "report_paths": meta.get("report_paths") or [],
                "git_commit": meta.get("git_commit"),
                "git_dirty": meta.get("git_dirty"),
                "dir": os.path.relpath(rdir, MAIN_DIR).replace(os.sep, "/"),
            })
    return rows


def _recent(rows, days=14):
    """Rows started within ``days``, newest first. Rows with no parseable start sort last."""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    out = [r for r in rows if (r["started_at"] or "") >= cutoff]
    return sorted(out, key=lambda r: r["started_at"] or "", reverse=True)


def build_status(days=14):
    """The fast view: what is running now, and what finished recently.

    This exists because "is it done?", "what's the status?" and "where's the html?" were
    asked roughly ten times across two sessions and were unanswerable without walking a
    100+ GB tree by hand.
    """
    rows = collect_runs()
    out = []

    running = [r for r in rows if r["status"] == "running"]
    out.append("RUNNING (%d)" % len(running))
    if not running:
        out.append("  nothing in flight")
    for r in sorted(running, key=lambda r: r["started_at"] or ""):
        elapsed = None
        try:
            elapsed = (datetime.now()
                       - datetime.fromisoformat(r["started_at"])).total_seconds()
        except (TypeError, ValueError):
            pass
        out.append("  %s  elapsed %-8s %s" % (
            (r["started_at"] or "?")[-8:], _fmt_duration(elapsed), r["analysis"]))
        out.append("    %s" % r["run_id"])
        done, want = r["n_succeeded"], r["n_requested"]
        out.append("    %d/%d participants%s" % (
            done, want, ("  ·  " + _rel(r["log_path"])) if r["log_path"] else ""))
        if r["question"]:
            out.append("    why: %s" % r["question"])
    out.append("")

    recent = [r for r in _recent(rows, days) if r["status"] != "running"]
    out.append("LAST %d DAYS (%d)" % (days, len(recent)))
    if not recent:
        out.append("  nothing")
    else:
        out.append("  %-19s %-17s %-8s %-6s %-26s %s"
                   % ("status", "started", "duration", "pats", "analysis", "run"))
        for r in recent:
            out.append("  %-19s %-17s %-8s %-6s %-26s %s" % (
                # A manifest with no `status` predates utils.run_context; say that
                # rather than "?", which reads as "something went wrong".
                (r["status"] or "pre-context")[:19],
                (r["started_at"] or "?").replace("T", " ")[:16],
                _fmt_duration(r["duration_sec"]),
                "%d/%d" % (r["n_succeeded"], r["n_requested"]),
                r["analysis"][:26],
                r["run_id"][:64]))
    out.append("")

    failed = [r for r in rows if (r["status"] or "").startswith("failed")
              or r["status"] == "interrupted"]
    if failed:
        out.append("NEEDS ATTENTION (%d)" % len(failed))
        for r in sorted(failed, key=lambda r: r["started_at"] or "", reverse=True)[:10]:
            out.append("  %-19s %s/%s" % (r["status"], r["analysis"], r["run_id"][:56]))
        out.append("")

    n_pre = len([r for r in rows if not r["status"]])
    out.append("%d runs carry a manifest; %d of them predate utils.run_context and so have "
               "no status," % (len(rows), n_pre))
    out.append("shown as `pre-context`. Runs with no manifest at all appear only in "
               "docs/results_index.md.")
    return "\n".join(out)


def build_report(with_size=True):
    runs = scan_runs(with_size=with_size)

    # Two reference classes, merged into one pin map.  Class 2 is why a directory whose
    # name carries no timestamp can be pinned at all; see the module docstring.
    pins = find_pins()
    journaled = find_journaled()
    literal_pins = find_dir_literal_pins(_literal_pin_candidates(runs))
    for name, sites in literal_pins.items():
        pins[name] = sorted(set(pins.get(name, [])) | set(sites))

    out = ["# Pinned and superseded result runs", ""]
    out.append("Generated by `python -m utils.audit_runs --write`. Do not hand-edit.")
    out.append("")
    out.append("`PINNED` = the directory name appears literally in tracked source, so")
    out.append("something depends on it. **Never delete a PINNED run.** `incomplete` = fewer")
    out.append("patient sub-directories than the largest run in the same analysis, i.e. an")
    out.append("aborted pass. `unreferenced` means only that no code names it - confirm")
    out.append("against this table before pruning anything.")
    out.append("")
    out.append("`journaled` = named in `docs/experiments/` but not in code. Still prunable —")
    out.append("an entry records that a question was asked, not that paper output depends on")
    out.append("the run. Update the entry to record the deletion *before* deleting.")
    out.append("")
    out.append("`derived` and `per-patient` are **not runs** and must not be read as aborted")
    out.append("passes: `derived` is a run's own output subdirectory (`figures/`,")
    out.append("`source_data/`, `report/`), and `per-patient` is the pre-run-directory")
    out.append("`<root>/<PATIENT>/` layout that `cross_task_*` used before 2026-07-30 and")
    out.append("that `label_generation/` still uses for atlas provenance.")
    out.append("")
    out.append("A name is matched two ways: as a timestamped run id, and - since 2026-08-10 -")
    out.append("as a whole-token directory literal. Without the second, any directory whose")
    out.append("name has no date in it read `unreferenced` however much code named it;")
    out.append("`original_KRR_l2_50ep` (13.5 GB, read by a notebook) and `balance_none` (read")
    out.append("by a paper pipeline) both did, and the first was staged for deletion.")
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
            status = classify(r, pins, cohort[task], journaled)
            where = ", ".join("`%s`" % p
                              for p in pins.get(r["run_id"],
                                                journaled.get(r["run_id"], []))) or "-"
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

    # ── figures/ ─────────────────────────────────────────────────────────────
    fig_mirrored, fig_unmirrored = scan_figures(with_size=with_size)
    if fig_mirrored or fig_unmirrored:
        out.append("## `figures/` - per-run figure output")
        out.append("")
        out.append("Since 2026-08-11 a run owns its plots: the three root pipelines write")
        out.append("`results/<analysis>/<run_id>/figures/`, and the 31 per-run twins that")
        out.append("used to live here were moved into their runs. What remains is mostly")
        out.append("`orphan` -- no `results/` twin, i.e. the run it described is already gone")
        out.append("-- which is the safest thing in this file to delete. A `twin` row now")
        out.append("means a STALE DUPLICATE rather than a live mirror. Note `figures/` also")
        out.append("holds cross-run data that paper pipelines read; see the untracked-inputs")
        out.append("table in `docs/repo_layout.md` before deleting.")
        out.append("")
        for analysis in sorted(fig_mirrored):
            rows = fig_mirrored[analysis]
            if not rows:
                continue
            out.append("### figures/%s" % analysis)
            out.append("")
            out.append("| directory | status | size | twin |")
            out.append("|---|---|---|---|")
            for r in rows:
                name = r["run_id"]
                if name in DERIVED_DIR_NAMES:
                    status, twin = "derived", "-"
                elif not r["has_twin"]:
                    status, twin = "**orphan**", "missing"
                else:
                    status = "twin:PINNED" if name in pins else "twin"
                    twin = "`results/%s/%s`" % (analysis, name)
                out.append("| `%s` | %s | %s | %s |"
                           % (name, status, _human(r["bytes"]), twin))
            out.append("")
        if fig_unmirrored:
            out.append("### figures/ subtrees with no `results/` counterpart")
            out.append("")
            out.append("Not run mirrors - these have no corresponding analysis under")
            out.append("`results/` at all. Inventoried, not classified.")
            out.append("")
            out.append("| subtree | files | size |")
            out.append("|---|---|---|")
            for r in fig_unmirrored:
                out.append("| `figures/%s` | %d | %s |"
                           % (r["analysis"], r["n_files"], _human(r["bytes"])))
            out.append("")

    out.extend(build_cohort_coverage(collect_runs()))

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
    ap.add_argument("--status", action="store_true",
                    help="what is running now and what finished recently; reads one "
                         "meta.json per run and never sizes directories, so it is fast "
                         "enough to use while a run is still going")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="the same structured rows as --status, as JSON, for an agent")
    ap.add_argument("--days", type=int, default=14,
                    help="how far back --status looks (default: 14)")
    args = ap.parse_args(argv)

    if args.as_json:
        json.dump(collect_runs(), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    if args.status:
        sys.stdout.write(build_status(days=args.days) + "\n")
        return 0

    report = build_report(with_size=not args.no_size)
    if args.write:
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
        print("wrote %s" % REPORT_PATH)
        print(write_experiments_index())
    else:
        sys.stdout.write(report + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
