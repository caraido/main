# -*- coding: utf-8 -*-
"""
report.render.blocks — the callout boxes.

Five semantic roles, one palette. The roles were recovered by reading what the
existing boxes were actually used FOR, not by what they were called: the same
"here is the method" box is spelled `.box`, `.method-box`, `.met`, `.methods` and
`.notes` across the generators, in four different colours, and one file's
`.warning` is red while another's is amber.

Naming is by role, not colour, so that recolouring the whole report set is a
change to `assets/report.css` and nothing else.
"""

from __future__ import annotations

CALLOUT_KINDS = {
    "summary":     "Executive summary / headline result.",
    "method":      "How something was computed. The most duplicated role.",
    "finding":     "A specific observation worth pulling out of a table.",
    "warning":     "A caveat, exclusion, or interpret-with-caution note.",
    "note":        "Neutral aside; the default when nothing stronger fits.",
}


def callout(kind: str, html: str, title: str = "") -> str:
    """Return one callout `<div>`.

    `kind` must be a key of CALLOUT_KINDS -- an unknown kind raises rather than
    silently rendering an unstyled div, because a silently unstyled caveat box is
    exactly the failure that loses a caveat.
    """
    if kind not in CALLOUT_KINDS:
        raise ValueError(
            "unknown callout kind {!r}; expected one of {}".format(
                kind, ", ".join(sorted(CALLOUT_KINDS))))
    head = "<h3>{}</h3>".format(title) if title else ""
    return '<div class="callout callout-{}">{}{}</div>'.format(kind, head, html)
