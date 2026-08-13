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
    label = display_id("AA")            # -> "NUE041"

Rules enforced here:
  * Figures and published source-data tables must use ``display_id`` (NUE###),
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
    """Map internal initials (e.g. 'AA') to the paper display ID (e.g. 'NUE041').

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


def place_labels(ax, labels, fontsize=7, pad_pt=6.0, gap_pt=2.0,
                 leader_color='#000000', leader_lw=0.5, marker_r_pt=7.0,
                 n_iter=800, fontweight='bold', clamp=True, margin_frac=(0.16, 0.01)):
    """Annotate ``labels`` = ``[(x, y, text)]`` so the text RINGS the point cloud.

    Each label STARTS on the ray from the centroid of the cloud through its own marker,
    just outside it, so the text sits on the outside of the blob rather than on a
    neighbour. The alternative -- a fixed offset, or left/right by which half of the axis a
    marker falls in -- puts most of the text on one side and, for a scatter whose points
    cluster along the diagonal, stacks it over the markers. That is the failure this fixes.

    It is then relaxed: labels repel each other and the markers, a weak spring holds each
    near its own starting anchor, and the whole thing is clamped inside the axes. Pushing
    only along the radius (the first version of this) cannot separate two regions that are
    nearly collinear with the centroid -- they slide outward together forever -- which is
    how ``aITG`` and ``insula`` printed on top of one another. The relaxation separates
    along whichever axis is cheaper, so collinear pairs come apart sideways.

    Geometry is computed in DISPLAY space (pixels), because these panels are equal-aspect
    while the two variables are not on the same numeric scale, so a data-space offset is not
    the offset that appears on the page. **Set the axis limits before calling**: the
    data->pixel transform is read here and does not update afterwards.

    Text extents are estimated from the character count rather than measured with a
    renderer. A real measurement needs a draw pass, and at <= ~30 short region names the
    estimate is what actually gets used.

    Returns the number of labels drawn with a leader line.
    """
    if not labels:
        return 0
    import numpy as _np

    dpi = ax.figure.dpi / 72.0                       # points -> pixels
    pad, gap = pad_pt * dpi, gap_pt * dpi
    mark_r = marker_r_pt * dpi
    fs = fontsize * dpi

    P = _np.array([ax.transData.transform((x, y)) for x, y, _ in labels], dtype=float)
    texts = [str(t) for _, _, t in labels]
    n = len(texts)
    centre = P.mean(axis=0)

    # Radial unit vectors. A marker sitting exactly on the centroid has no direction of its
    # own, so it is given one by index -- arbitrary but deterministic, which is what matters
    # for a figure that gets regenerated.
    U = P - centre
    norm = _np.linalg.norm(U, axis=1)
    for i in range(n):
        if norm[i] < 1e-9:
            ang = 2.0 * _np.pi * i / max(n, 1)
            U[i] = (_np.cos(ang), _np.sin(ang))
            norm[i] = 1.0
    U /= norm[:, None]

    # Half-extents, MEASURED with the renderer rather than estimated from the character
    # count. The estimate (0.55 em per character) under-measured bold text by ~18 %, which is
    # not a cosmetic error: the relaxation below stops when it detects no overlaps, so boxes
    # that are too small make it stop while the text is still visibly colliding. That is how
    # `temporal pole` printed across `pMTG`. Cost is one extra draw pass per panel.
    probes = [ax.text(0, 0, t, fontsize=fontsize, fontweight=fontweight,
                      transform=None, alpha=0.0) for t in texts]
    try:
        rend = ax.figure.canvas.get_renderer()
    except AttributeError:                      # backend without a cached renderer
        ax.figure.canvas.draw()
        rend = ax.figure.canvas.get_renderer()
    ext = [p.get_window_extent(rend) for p in probes]
    for p in probes:
        p.remove()
    hw = _np.array([0.5 * e.width for e in ext])
    hh = _np.array([0.5 * e.height for e in ext])

    # Anchor side is fixed from the initial radial direction and does NOT change during
    # relaxation: a label that flips ha/va mid-solve oscillates instead of settling.
    SX = _np.where(U[:, 0] > 0.20, 1.0, _np.where(U[:, 0] < -0.20, -1.0, 0.0))
    SY = _np.where(U[:, 1] > 0.20, 1.0, _np.where(U[:, 1] < -0.20, -1.0, 0.0))

    ideal = P + U * (mark_r + pad)
    pos = ideal.copy()
    box = lambda q: q + _np.column_stack([SX * hw, SY * hh])   # anchor -> box centre

    # Labels may leave the axes by ``margin_frac`` of its size on each side. A hard clamp to
    # the axes rectangle deadlocks whenever the markers are crowded into one corner -- which
    # they are on the knockout panel, where one region sits 4x further out than the rest and
    # squeezes the other twelve into the bottom-left: the labels then had nowhere to go and
    # the relaxation ended still overlapping. Letting them into the margin, with a leader
    # line back to the marker, is the readable resolution and is what a hand-drawn figure
    # does. ``annotation_clip=False`` below is what actually renders them there.
    #
    # ``margin_frac`` is (x, y) and the two are NOT symmetric by default. Text beside the
    # panel reads as a margin annotation; text above the top spine reads as the figure being
    # broken, which is what a single symmetric margin produced for `angular` and `aSTG`. So
    # the horizontal margin is generous and the vertical one is nearly nil.
    try:
        mfx, mfy = margin_frac
    except TypeError:
        mfx = mfy = margin_frac
    x0, x1 = sorted(ax.transAxes.transform([(0, 0), (1, 1)])[:, 0])
    y0, y1 = sorted(ax.transAxes.transform([(0, 0), (1, 1)])[:, 1])
    mx, my = mfx * (x1 - x0), mfy * (y1 - y0)
    x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - my, y1 + my

    for _ in range(n_iter):
        C = box(pos)
        # Convergence is tested on OVERLAPS REMAINING, not on how far things moved this
        # pass. Testing on movement stops early and wrongly: the spring below cancels most
        # of a small separation step, so a still-overlapping pair settles into a low-motion
        # standoff. That is exactly how `precuneus` printed on top of `aFus`.
        n_overlap = 0
        # label <-> label
        for i in range(n):
            for j in range(i + 1, n):
                dx, dy = C[i] - C[j]
                ox = hw[i] + hw[j] + gap - abs(dx)
                oy = hh[i] + hh[j] + gap - abs(dy)
                if ox <= 0 or oy <= 0:
                    continue
                n_overlap += 1
                # separate along whichever axis needs the least travel
                if ox < oy:
                    s = 0.5 * ox * (1.0 if dx >= 0 else -1.0)
                    pos[i, 0] += s; pos[j, 0] -= s; C[i, 0] += s; C[j, 0] -= s
                else:
                    s = 0.5 * oy * (1.0 if dy >= 0 else -1.0)
                    pos[i, 1] += s; pos[j, 1] -= s; C[i, 1] += s; C[j, 1] -= s
        # label <-> every marker (not just its own): text over another region's dot is
        # exactly as unreadable as text over another region's text.
        for i in range(n):
            for k in range(n):
                dx, dy = C[i] - P[k]
                ox = hw[i] + mark_r - abs(dx)
                oy = hh[i] + mark_r - abs(dy)
                if ox <= 0 or oy <= 0:
                    continue
                n_overlap += 1
                if ox < oy:
                    s = ox * (1.0 if dx >= 0 else -1.0)
                    pos[i, 0] += s; C[i, 0] += s
                else:
                    s = oy * (1.0 if dy >= 0 else -1.0)
                    pos[i, 1] += s; C[i, 1] += s
        # weak spring back toward the radial anchor, so labels do not wander off. Kept small
        # (2 %) so it biases the solution without fighting the separation.
        pos += 0.02 * (ideal - pos)
        if clamp:
            C = box(pos)
            pos[:, 0] += _np.clip(x0 + hw - C[:, 0], 0, None)
            pos[:, 0] -= _np.clip(C[:, 0] + hw - x1, 0, None)
            pos[:, 1] += _np.clip(y0 + hh - C[:, 1], 0, None)
            pos[:, 1] -= _np.clip(C[:, 1] + hh - y1, 0, None)
        if n_overlap == 0:
            break

    inv = ax.transData.inverted()
    n_leader = 0
    for i, (dx_data, dy_data, text) in enumerate(labels):
        tx, ty = inv.transform(pos[i])
        ha = 'left' if SX[i] > 0 else ('right' if SX[i] < 0 else 'center')
        va = 'bottom' if SY[i] > 0 else ('top' if SY[i] < 0 else 'center')
        far = _np.linalg.norm(pos[i] - P[i]) > mark_r + pad + 1.5 * dpi
        n_leader += int(far)
        ax.annotate(
            text, xy=(dx_data, dy_data), xytext=(tx, ty), textcoords='data',
            fontsize=fontsize, fontweight=fontweight, ha=ha, va=va, zorder=7,
            annotation_clip=False,
            arrowprops=(dict(arrowstyle='-', color=leader_color, lw=leader_lw,
                             shrinkA=0, shrinkB=2) if far else None))
    return n_leader


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
