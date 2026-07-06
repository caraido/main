# -*- coding: utf-8 -*-
"""
figures_for_paper/paper_common.py — shared conventions for every paper figure.

Import this from any figure script or notebook in ``figures_for_paper/`` so that
all figures share ONE participant mapping and ONE plotting style. See README.md
in this folder for the full standard.

Usage (script or notebook)::

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # figures_for_paper/
    # (from a notebook, insert the absolute path to figures_for_paper instead)
    from paper_common import display_id, apply_paper_style, PARTICIPANTS

    apply_paper_style()                 # editable-text vector output + house rcParams
    label = display_id("AA")            # -> "NUEx041"

Rules enforced here:
  * Figures and published source-data tables must use ``display_id`` (NUEx###),
    NEVER the internal initials.
  * PDFs use editable text (``pdf.fonttype 42``) so labels remain selectable.
"""

import os
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARTICIPANTS_JSON = os.path.join(_HERE, 'participants.json')


def load_participants():
    """Return the list of participant dicts from participants.json (source of truth)."""
    with open(_PARTICIPANTS_JSON, encoding='utf-8') as f:
        return json.load(f)['participants']


PARTICIPANTS = load_participants()
_DISPLAY = {p['initials']: p['display_id'] for p in PARTICIPANTS}
_INITIALS = {p['display_id']: p['initials'] for p in PARTICIPANTS}


def display_id(initials, strict=False):
    """Map internal initials (e.g. 'AA') to the paper display ID (e.g. 'NUEx041').

    Unknown initials fall back to the input unchanged (so a figure never crashes on
    an unmapped participant) unless ``strict=True``, which raises — use strict in
    tests/CI to catch a participant that was added to the data but not to
    participants.json.
    """
    if strict and initials not in _DISPLAY:
        raise KeyError(f"{initials!r} not in participants.json — add it before publishing.")
    return _DISPLAY.get(initials, initials)


def initials_of(display):
    """Inverse of display_id: paper display ID -> internal initials."""
    return _INITIALS.get(display, display)


def display_ids(initials_list, strict=False):
    """Vectorised display_id over an iterable of initials."""
    return [display_id(x, strict=strict) for x in initials_list]


def apply_paper_style():
    """Apply the house Matplotlib rcParams for paper figures.

    Editable-text vector output (fonttype 42) + restrained Nature-style defaults.
    Call once at the top of a figure script/notebook, after importing matplotlib.
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        'pdf.fonttype': 42,      # editable text in PDF
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'font.size': 8,
        'axes.titlesize': 11,
        'axes.titleweight': 'bold',
        'axes.labelsize': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7.5,
        'legend.frameon': False,
    })
