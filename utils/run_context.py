# -*- coding: utf-8 -*-
"""One run, one directory, one manifest -- so a pipeline cannot write anywhere else.

Every output a run produces is reachable only through the ``RunContext`` handed out by
``open_run``. There is no other way to get a path: no ``os.path.join``, no ``MAIN_DIR /
"results"``, no bare filename resolved against the working directory. That is the whole
point of the module.

Why it also owns the manifest
-----------------------------
Before this existed, 12 of the 15 analyses under ``results/`` wrote **no run-level
manifest at all**, so "is it done", "what did it produce", "which participants succeeded"
and "where is the report" were unanswerable without walking a 100+ GB tree. The three that
did write one -- the root training scripts, via ``utils.run_meta`` -- had 45 well-chosen
fields, and ``analysis/cross_task/cross_task_cotrain.py`` had a fourth, different schema
under a fourth filename (``run_metadata.json``).

So the manifest is not a side effect here; it is the reason the context exists. Writing it
is not optional and cannot be forgotten, because it happens on entry.

**There is deliberately no append-only JSONL ledger.** ``meta.json`` *is* the ledger row and
``utils.audit_runs`` already walks every one of them. A second file recording the same
facts is a second source of truth that can disagree with the tree it describes -- the exact
class of bug ``utils/config.py`` and ``docs/results_index.md`` exist to prevent.

Usage
-----
::

    from utils.run_context import open_run

    with open_run("auditory_alignment", run_id, why="which cue does decoding lock to?",
                  meta={"n_epochs": 50, "cues": CUES}) as run:
        for patient in patients:
            ...
            np.savez(run.path(patient, "perbin.npz"), **arrays)
            df.to_csv(run.table("peak_summary"), index=False)
            fig.savefig(run.figure("01_locking"))
        run.headline(word_bal_acc=0.31, cat_indep_bal_acc=0.44)
        doc.write(run.report("auditory_alignment"))

On exit -- including on an exception -- the manifest gains ``finished_at``,
``duration_sec`` and a ``status`` of ``ok``, ``failed:<ExcType>`` or ``interrupted``.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import socket
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import run_meta
from .logging import tee_stdout
from .paths import MAIN_DIR, figures_dir, log_path, report_path, results_dir, stage_of

#: Lifecycle fields. A caller may NEVER set these: they describe what happened to the run,
#: which only the context can know. A caller passing ``status='ok'`` up front would make
#: the manifest assert success before any work ran.
_LIFECYCLE_OWNED = frozenset({
    "analysis", "stage", "status", "started_at", "started_at_utc",
    "finished_at", "duration_sec", "report_paths", "headline",
})

#: Provenance fields the context computes, but which a caller may legitimately already
#: have computed itself. **The caller wins.** These are facts about the run, not claims
#: about its outcome, and the two computations agree.
#:
#: This split exists because the three root pipelines build a 45-field manifest of their
#: own and that manifest is the richer one -- it records the ROI whitelist in full, the
#: warp target and who defined it, per-patient channel-selection counts. Making them drop
#: those fields to satisfy a reserved-word check would have *removed* provenance in the
#: name of tidying it up.
_PROVENANCE_FILLED = frozenset({
    "run_id", "command_line", "cwd", "host", "log_path", "git_commit", "git_dirty",
    "question", "supersedes",
    "python_version", "platform",
    "numpy_version", "pandas_version", "sklearn_version", "torch_version",
})


def _versions():
    """Interpreter and key library versions, best effort.

    Recorded because there is **no tracked dependency manifest anywhere in this
    repository** -- no environment.yml, requirements.txt or pyproject.toml. The run
    manifest is therefore the only environment provenance that exists, which makes it
    worth a few import attempts.
    """
    out = {"python_version": platform.python_version(), "platform": platform.platform()}
    for mod, key in (("numpy", "numpy_version"), ("pandas", "pandas_version"),
                     ("sklearn", "sklearn_version"), ("torch", "torch_version")):
        try:
            out[key] = __import__(mod).__version__
        except Exception:
            pass
    return out


@dataclass
class RunContext:
    """Handle to one run's directory. Every path a pipeline writes comes from here."""

    analysis: str
    run_id: str
    stage: str
    dir: Path
    meta: dict = field(repr=False)
    #: Additional directories the manifest is mirrored into. The root pipelines write
    #: meta.json into both their results/ and figures/ run directory, so a figure tree
    #: read on its own still says what produced it. Preserved rather than simplified away.
    mirror_dirs: tuple = ()

    # ── paths ────────────────────────────────────────────────────────────────
    def path(self, *parts: str) -> Path:
        """An arbitrary file inside this run. Parent directories are created."""
        target = results_dir(self.analysis, self.run_id, *parts[:-1]) if len(parts) > 1 \
            else results_dir(self.analysis, self.run_id)
        return target / parts[-1] if parts else target

    def figure(self, name: str, ext: str = "png") -> Path:
        """``<run>/figures/<name>.<ext>`` -- this run's own figure tree."""
        return results_dir(self.analysis, self.run_id, "figures") / f"{name}.{str(ext).lstrip('.')}"

    def table(self, name: str) -> Path:
        """``<run>/source_data/<name>.csv``.

        Named ``source_data`` to match the convention ``figures_for_paper/`` already uses:
        the numbers actually plotted, beside the thing that plots them.
        """
        stem = name if name.endswith(".csv") else f"{name}.csv"
        return results_dir(self.analysis, self.run_id, "source_data") / stem

    def report(self, name: str, suffix: str = "") -> Path:
        """``<run>/report/<name>_<run_id><suffix>.html``, recorded in the manifest.

        ``suffix`` is how a re-render over a participant subset gets its own filename
        instead of overwriting the cohort report.
        """
        path = report_path(self.analysis, name, self.run_id, suffix=suffix)
        self.meta.setdefault("report_paths", [])
        rel = os.path.relpath(path, results_dir(self.analysis, self.run_id, create=False))
        entry = {"path": rel.replace(os.sep, "/"),
                 "generated_at": datetime.now().isoformat(timespec="seconds")}
        self.meta["report_paths"] = [e for e in self.meta["report_paths"]
                                     if e.get("path") != entry["path"]] + [entry]
        self._flush()
        return path

    def log(self) -> Path:
        return Path(self.meta["log_path"])

    def scratch_figure(self, name: str, ext: str = "png") -> Path:
        """A cross-run exploratory figure under ``figures/<analysis>/``.

        Use sparingly: a figure that describes one run belongs to that run.
        """
        return figures_dir(self.analysis) / f"{name}.{str(ext).lstrip('.')}"

    # ── manifest ─────────────────────────────────────────────────────────────
    def note(self, **fields) -> None:
        """Merge fields into the manifest and rewrite it now.

        Rewriting immediately rather than at exit is deliberate: a run killed halfway
        still leaves an honest manifest describing how far it got.
        """
        clashes = _LIFECYCLE_OWNED & set(fields)
        if clashes:
            raise ValueError(f"{sorted(clashes)} describe the run's outcome and are owned "
                             f"by RunContext; note() is for analysis parameters")
        self.meta.update(fields)
        self._flush()

    def headline(self, **numbers) -> None:
        """The few numbers this run exists to produce.

        Kept small on purpose -- this is what a status view and a journal entry quote, and
        a manifest field that grows without bound stops being readable. Not a results
        store: the full arrays live in the run's own files.
        """
        self.meta.setdefault("headline", {}).update(numbers)
        self._flush()

    def fail(self, patient: str, reason: str) -> None:
        """Record a per-participant failure without aborting the run."""
        failed = self.meta.setdefault("failed_patients", [])
        if patient not in failed:
            failed.append(patient)
        self.meta.setdefault("failures", {})[patient] = str(reason)
        self._flush()

    def succeed(self, patient: str) -> None:
        done = self.meta.setdefault("succeeded_patients", [])
        if patient not in done:
            done.append(patient)
        self._flush()

    def _flush(self) -> None:
        run_meta.write_meta(self.meta, str(self.dir), *[str(d) for d in self.mirror_dirs])


@contextlib.contextmanager
def open_run(analysis: str,
             run_id: str,
             *,
             stage: str = None,
             meta: dict = None,
             why: str = None,
             supersedes: str = None,
             tee: bool = True,
             legacy_log_stem: str = None,
             mirror_dirs=(),
             argv=None):
    """Create ``results/<analysis>/<run_id>/``, tee stdout, and own the manifest.

    Parameters
    ----------
    analysis, run_id
        ``run_id`` should match ``utils.audit_runs.RUN_ID_RE`` -- a timestamp prefix plus a
        config slug. A run whose directory name the auditor cannot match reads
        ``unreferenced`` forever and is what the pruning plan deletes.
    why
        One line: the question this run is meant to answer. Captured at launch, when the
        answer is known, and it travels with the run so it cannot drift from it. This plus
        ``supersedes`` is the entire human writing burden per run.
    supersedes
        The ``run_id`` this replaces, so a chain of re-runs is reconstructable.
    legacy_log_stem
        Reproduces the pre-existing flat log name ``logs/<stem>_<run_id>.log`` exactly.
        The three root pipelines pass it because a refactor must not rename an output.
    """
    # VALIDATE BEFORE CREATING ANYTHING. results_dir() and log_path() both mkdir, so a
    # call that is going to be rejected must be rejected first -- otherwise a typo leaves
    # an empty run directory behind, and an empty run directory is exactly what
    # audit_runs reports as `incomplete`, i.e. an aborted pass worth deleting. Caught by
    # the module's own self-test, which left two such directories on the first attempt.
    full = dict(meta or {})
    clashes = _LIFECYCLE_OWNED & set(full)
    if clashes:
        raise ValueError(f"{sorted(clashes)} describe the run's outcome and are owned by "
                         f"open_run; remove them from meta")

    stage = stage or stage_of(analysis)
    started = datetime.now()
    run_dir = results_dir(analysis, run_id)
    # Only reserve a log path when we are actually going to write one. Recording a
    # log_path under tee=False put a path to a non-existent file in the manifest, and
    # created an empty logs/<analysis>/ directory to go with it.
    lp = log_path(analysis, run_id, legacy_stem=legacy_log_stem) if tee else None

    # Provenance the context can supply -- but only where the caller has not already.
    # setdefault, not update: a pipeline with its own richer manifest keeps every field
    # it built, so migrating one can only ADD keys, never drop or overwrite them.
    provenance = {
        "run_id": run_id,
        "command_line": list(argv if argv is not None else sys.argv),
        "cwd": os.getcwd(),
        "host": socket.gethostname(),
        # POSIX separators on purpose. report/__main__.py derives a figures path with
        # `run_dir.replace('results/', 'figures/')`, which silently no-ops on Windows
        # backslashes -- a stored path should never depend on the separator it was
        # written with.
        "log_path": os.path.relpath(lp, MAIN_DIR).replace(os.sep, "/") if lp else None,
        "git_commit": run_meta.git_hash(),
        "git_dirty": run_meta.git_dirty(),
        "question": why,
        "supersedes": supersedes,
    }
    provenance.update(_versions())
    for key, value in provenance.items():
        full.setdefault(key, value)
    if why is not None:
        full["question"] = why            # an explicit --why always wins
    if supersedes is not None:
        full["supersedes"] = supersedes

    # Lifecycle: always the context's.
    full.update({
        "analysis": analysis,
        "stage": stage,
        "status": "running",
        "started_at": started.isoformat(timespec="seconds"),
        "started_at_utc": started.astimezone(timezone.utc).isoformat(timespec="seconds"),
    })
    full.setdefault("report_paths", [])

    ctx = RunContext(analysis=analysis, run_id=run_id, stage=stage, dir=run_dir, meta=full,
                     mirror_dirs=tuple(mirror_dirs))
    # Written BEFORE any work: a run killed at minute one still leaves a manifest naming
    # its command line and git commit. docs/repo_layout.md values exactly that.
    ctx._flush()

    status = "ok"
    try:
        if tee:
            with tee_stdout(lp):
                yield ctx
        else:
            yield ctx
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except BaseException as exc:
        status = f"failed:{type(exc).__name__}"
        raise
    finally:
        finished = datetime.now()
        if ctx.meta.get("failed_patients") and status == "ok":
            status = "ok-with-failures"
        ctx.meta.update({
            "status": status,
            "finished_at": finished.isoformat(timespec="seconds"),
            "duration_sec": round((finished - started).total_seconds(), 1),
        })
        ctx._flush()


def read_status(analysis: str, run_id: str) -> dict:
    """The manifest of a run, or ``{}``. Thin wrapper so callers need one import."""
    return run_meta.read_meta(str(results_dir(analysis, run_id, create=False)))
