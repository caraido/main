# -*- coding: utf-8 -*-
"""
utils.logging — console output helpers used by the model/training scripts.

These were previously duplicated verbatim across phoneme_regression.py,
semantic_regression.py, and semantic_vanilla_retrieval.py.

NB this module shadows the stdlib ``logging`` name. That is a latent trap, but the
stdlib module is used nowhere in this repository — all logging here is ``print`` plus
the stdout tee below — so renaming it is a separate, purely cosmetic change with ~6
importers. Do not ``import logging`` inside ``utils/`` expecting the standard library.

Public API:
    _sep        — print a horizontal rule
    _header     — print a boxed header
    _section    — print a sub-section header
    _progress   — print/redraw an in-place progress bar
    tee_stdout  — context manager duplicating stdout+stderr into a log file
"""

import contextlib
import sys

def _sep(char='─', width=72):
    print(char * width)

def _header(msg):
    print()
    _sep('═')
    print(f'  {msg}')
    _sep('═')

def _section(msg):
    print()
    _sep()
    print(f'  >> {msg}')
    _sep()

def _progress(current, total, label=''):
    bar_len = 40
    filled = int(bar_len * current / total) if total else 0
    bar = '█' * filled + '░' * (bar_len - filled)
    print(f'\r        [{bar}] {current}/{total}  {label}', end='', flush=True)


class _Tee:
    """Duplicate writes to both the original stream and a log file.

    Extracted from three verbatim copies: semantic_regression.py, phoneme_regression.py
    and semantic_vanilla_retrieval.py each defined this class identically.
    """

    def __init__(self, log_file, original_stream):
        self._log = log_file
        self._term = original_stream

    def write(self, data):
        self._term.write(data)
        self._term.flush()
        # Carriage returns become newlines so the log file stays readable: the progress
        # bar redraws in place on a terminal, which would otherwise collapse to one line.
        self._log.write(data.replace('\r', '\n'))
        self._log.flush()

    def flush(self):
        self._term.flush()
        self._log.flush()

    def isatty(self):
        return False


@contextlib.contextmanager
def tee_stdout(path, *, encoding='utf-8'):
    """Duplicate stdout and stderr into ``path`` for the duration of the block.

    Unlike the three copies this replaces, it is a context manager and restores both
    streams in a ``finally``. Those copies restored stdout only on the success path, so a
    pipeline that raised left a ``_Tee`` wrapping a closed file installed as ``sys.stdout``
    for the rest of the process -- which is how a traceback could end up truncated in the
    very log written to diagnose it.

    Yields the open file handle, so a caller can record its name in a run manifest.
    """
    handle = open(path, 'w', encoding=encoding, buffering=1)
    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(handle, sys.__stdout__)
    sys.stderr = _Tee(handle, sys.__stderr__)
    try:
        yield handle
    finally:
        sys.stdout, sys.stderr = saved_out, saved_err
        handle.close()
