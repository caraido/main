# -*- coding: utf-8 -*-
"""
utils.logging — console output helpers used by the model/training scripts.

These were previously duplicated verbatim across phoneme_regression.py,
semantic_regression.py, and semantic_vanilla_retrieval.py.

Public API:
    _sep      — print a horizontal rule
    _header   — print a boxed header
    _section  — print a sub-section header
    _progress — print/redraw an in-place progress bar
"""

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
