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
  * A participant's plotting colour is fixed in participants.json and reused in every
    figure; cue marker colours/labels are fixed in cue_style.json. Neither is
    hard-coded in figure scripts.
  * PDFs use editable text (``pdf.fonttype 42``) so labels remain selectable.

Division of labour with ``utils/config.py``: that module owns repo-wide values
(pinned run ids, the p-value cutoff, type sizes, DPI) because ``analysis/`` and
the root scripts need them too. This module owns figure *identity* — which
participant is which colour, which cue is which label — which is meaningful
only here. The style/statistics names are re-exported below so a figure script
needs one import, not two.
"""

import os
import sys
import json
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAIN_DIR = os.path.dirname(_HERE)
_PARTICIPANTS_JSON = os.path.join(_HERE, 'participants.json')
_CUE_STYLE_JSON = os.path.join(_HERE, 'cue_style.json')
_EMBEDDING_STYLE_JSON = os.path.join(_HERE, 'embedding_style.json')

# figures_for_paper/ is not a package and is usually reached by putting *itself*
# on sys.path; utils/ then needs main/ there too. Bootstrap it here so every
# figure script inherits the ability to import utils.* just by importing this.
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

from utils.config import (ALPHA, PCTILE, p_stars,               # noqa: E402,F401
                          DPI_PANEL, DPI_COMBINED,
                          FONT_SIZE, AXES_TITLE_SIZE, AXES_LABEL_SIZE,
                          TICK_SIZE, LEGEND_SIZE, VECTOR_FONTTYPE)

# Brain-region identity, re-exported for the same reason the style names are: a figure
# script should need one import, not three. The ROI palette is figure identity in exactly
# the sense participants.json is -- which region is which colour, fixed everywhere -- but it
# is vendored from the electrode_labeling repo rather than owned here, so it lives in
# utils/roi_palette.py and is only surfaced from this module.
from utils.roi_palette import (REGION_COLORS, OTHER, OTHER_COLOR,   # noqa: E402,F401
                               color_of, legend_entries, ordered as roi_ordered,
                               display as roi_display, FAMILIES as ROI_FAMILIES)
from utils.rois import IN_ANALYSIS                                  # noqa: E402,F401

# Fallback palette for participants not yet given a colour in participants.json
# (a newly added participant still plots without a crash; give them a real colour there).
DEFAULT_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                   '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#393b79', '#e7298a']


def load_participants():
    """Return the list of participant dicts from participants.json (source of truth)."""
    with open(_PARTICIPANTS_JSON, encoding='utf-8') as f:
        return json.load(f)['participants']


PARTICIPANTS = load_participants()
_DISPLAY = {p['initials']: p['display_id'] for p in PARTICIPANTS}
_INITIALS = {p['display_id']: p['initials'] for p in PARTICIPANTS}
_COLOR = {p['initials']: p['color'] for p in PARTICIPANTS if p.get('color')}


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


def participant_color(initials, default=None):
    """Return the fixed plotting colour for a participant (from participants.json),
    or ``default`` if none is defined for that participant."""
    return _COLOR.get(initials, default)


def assign_colors(initials_list):
    """Return a colour per participant in `initials_list`, in that order.

    Uses each participant's fixed colour from participants.json; any participant
    without one is assigned the next unused DEFAULT_PALETTE colour so figures never
    crash on a newly added participant (but give them a real colour in the JSON)."""
    used = set(_COLOR.get(p) for p in initials_list if _COLOR.get(p))
    spare = [c for c in DEFAULT_PALETTE if c not in used]
    out, k = [], 0
    for p in initials_list:
        c = _COLOR.get(p)
        if c is None:
            c = spare[k % len(spare)] if spare else DEFAULT_PALETTE[k % len(DEFAULT_PALETTE)]
            k += 1
        out.append(c)
    return out


def load_cue_style():
    """Return an OrderedDict cue_name -> {'color', 'label'} from cue_style.json,
    preserving the file's cue order (the order cues are drawn / listed in legends)."""
    with open(_CUE_STYLE_JSON, encoding='utf-8') as f:
        cues = json.load(f)['cues']
    return OrderedDict((k, {'color': v['color'], 'label': v['label']}) for k, v in cues.items())


def load_embedding_style():
    """Return the embedding/family plotting style from embedding_style.json (source of truth).

    A dict with two keys: ``families`` (name -> {'color','label','models'}) and ``models``
    (name -> {'color','family', optional 'group_color'}). Figures contrasting decoding targets
    (e.g. language_vs_visual) read colours/labels from here instead of hard-coding them.
    Convenience accessors below flatten the common lookups.
    """
    with open(_EMBEDDING_STYLE_JSON, encoding='utf-8') as f:
        return json.load(f)


def embedding_colors():
    """Flatten embedding_style.json into the dicts figures usually want:
    (model_color, family_color, family_label, model_group_color).
    ``group_color`` falls back to the model's own colour when not defined."""
    es = load_embedding_style()
    model_color = {m: v['color'] for m, v in es['models'].items()}
    family_color = {f: v['color'] for f, v in es['families'].items()}
    family_label = {f: v['label'] for f, v in es['families'].items()}
    group_color = {m: v.get('group_color', v['color']) for m, v in es['models'].items()}
    return model_color, family_color, family_label, group_color


def apply_paper_style():
    """Apply the house Matplotlib rcParams for paper figures.

    Editable-text vector output (fonttype 42) + restrained Nature-style defaults.
    Call once at the top of a figure script/notebook, after importing matplotlib.
    Type sizes and DPI come from ``utils.config``; change them there, not here.
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        'pdf.fonttype': VECTOR_FONTTYPE,      # editable text in PDF
        'ps.fonttype': VECTOR_FONTTYPE,
        'svg.fonttype': 'none',
        'font.size': FONT_SIZE,
        'axes.titlesize': AXES_TITLE_SIZE,
        'axes.titleweight': 'bold',
        'axes.labelsize': AXES_LABEL_SIZE,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': TICK_SIZE,
        'ytick.labelsize': TICK_SIZE,
        'legend.fontsize': LEGEND_SIZE,
        'legend.frameon': False,
    })
