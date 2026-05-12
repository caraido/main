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
