# -*- coding: utf-8 -*-
"""
utils.run_meta -- git and run-metadata helpers shared by training scripts.

These were previously duplicated (with docstring drift) across
phoneme_regression.py, semantic_regression.py, and semantic_vanilla_retrieval.py.

Public API:
    git_hash(repo_dir=None)    -- short git HEAD hash, or None
    git_dirty(repo_dir=None)   -- True if working tree dirty; None on error
    write_meta(meta, *dirs)    -- write meta dict as meta.json into each dir
    find_repo_root(start=None) -- walk up to find the directory containing .git

Callers can use no-arg forms (`git_hash()`); repo_dir defaults to the nearest
ancestor of cwd containing a .git folder.
"""

import json
import os
import subprocess
from pathlib import Path


def find_repo_root(start_dir=None):
    """Walk up from start_dir (or cwd) looking for a directory containing .git."""
    cur = Path(start_dir or os.getcwd()).resolve()
    for p in [cur, *cur.parents]:
        if (p / '.git').is_dir() or (p / '.git').is_file():
            return str(p)
    return str(cur)


def git_hash(repo_dir=None):
    """Return short git HEAD hash; None if not in a git repo or git missing."""
    repo_dir = repo_dir or find_repo_root()
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=repo_dir, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def git_dirty(repo_dir=None):
    """Return True if the working tree has uncommitted changes; None on error."""
    repo_dir = repo_dir or find_repo_root()
    try:
        out = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            cwd=repo_dir, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return len(out) > 0
    except Exception:
        return None


def write_meta(meta, *dirs):
    """Write meta dict as meta.json into each of the given directories."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)


def read_meta(run_dir):
    """The run's meta.json as a dict, or {} if absent or unreadable."""
    try:
        with open(os.path.join(run_dir, 'meta.json'), encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def read_window(run_dir):
    """``(n_bins_history, bin_size_ms)`` for *run_dir*, read from its own meta.json.

    Raises ``SystemExit`` when the run does not record them.

    Why this refuses to guess: every report that turns a bin index into a time does
    ``(bin_index - n_bins_history) * bin_size``.  Those two numbers used to be module
    constants hard-coded to 10 and 100, so reading a run with a different window shifted
    every reported latency by the difference -- silently, with no error and a plausible
    plot.  A run that does not say what window it used cannot be converted to seconds at
    all, and saying so is the only safe answer.

    Runs made before ``n_bins_history`` was recorded are the reason this is a hard failure
    rather than a fallback: for those, pass the window explicitly on the command line so
    the assumption is visible in the shell history rather than buried in a default.
    """
    meta = read_meta(run_dir)
    n_hist = meta.get('n_bins_history')
    bin_ms = meta.get('bin_size_ms')
    if n_hist is None or bin_ms is None:
        raise SystemExit(
            f"{run_dir}: meta.json does not record n_bins_history/bin_size_ms, so bin "
            f"indices cannot be converted to seconds. Pass --history-bins/--bin-size "
            f"explicitly for this run rather than assuming the current default.")
    return int(n_hist), float(bin_ms)
