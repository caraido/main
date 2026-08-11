# -*- coding: utf-8 -*-
"""
report.render.document — collapsible sections, the contents nav, page assembly.

Extracted from `analysis/cross_task/cross_task_region_importance_report.py`, the
only generator that had grown a real one. That version kept its table of contents
in a module-level `_TOC = []`, which meant building two reports in one process
concatenated their contents lists; the state lives on the instance here.

The other fourteen generators have no collapsible sections at all, or one
hand-written `<details class="meta-box">` around a meta.json dump. That idiom is
`add_meta()` -- a fold that deliberately stays out of the contents nav, because a
parameter dump is not a section of the argument.
"""

from __future__ import annotations

import re
from datetime import datetime

from .tables import stylesheet

_TOC_SCRIPT = (
    "<script>"
    "function setAll(o){document.querySelectorAll('details.sec').forEach(function(d){d.open=o;});}"
    "document.querySelectorAll('nav.toc a').forEach(function(a){a.addEventListener('click',function(){"
    "var el=document.querySelector(a.getAttribute('href'));var p=el;"
    "while(p){if(p.tagName==='DETAILS')p.open=true;p=p.parentElement;}});});"
    "</script>")

_HEADING = re.compile(r"\s*<(h[12])>(.*?)</\1>(.*)", re.S)


class Document:
    """One HTML report under construction.

    sections are appended in order; `render()` returns the finished page with the
    contents nav inserted ahead of them.
    """

    def __init__(self, title: str, subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle
        self._toc = []       # (anchor_id, plain-text title, is_sub)
        self._parts = []

    # -- content ----------------------------------------------------------

    def add_html(self, html: str) -> "Document":
        """Append raw HTML outside any fold (intros, callouts, figure grids)."""
        self._parts.append(html)
        return self

    def fold(self, html: str, anchor: str, open: bool = False,
             sub: bool = False, in_toc: bool = True) -> str:
        """Return `html` wrapped in a collapsible <details>, registering its
        contents entry. Does NOT append -- this is the expression-position form,
        needed because nested sections must be composed before their parent is
        added. HTML with no leading <h1>/<h2> is returned unchanged rather than
        losing its heading to a summary that isn't there.

        `sub=True` indents the entry under the preceding part. Nesting works: the
        contents click handler opens every ancestor <details>.
        """
        m = _HEADING.match(html)
        if not m:
            return html
        _tag, heading, rest = m.group(1), m.group(2), m.group(3)
        if in_toc:
            self.add_toc_entry(anchor, heading, sub)
        return '<details id="{}" class="sec"{}><summary>{}</summary>{}</details>'.format(
            anchor, " open" if open else "", heading, rest)

    def add_section(self, html: str, anchor: str, open: bool = False,
                    sub: bool = False, in_toc: bool = True) -> "Document":
        """Fold `html` and append it. The common case."""
        return self.add_html(self.fold(html, anchor, open=open, sub=sub, in_toc=in_toc))

    def add_toc_entry(self, anchor: str, title: str, sub: bool = False) -> "Document":
        """File a contents entry for a section not yet built.

        Needed for a parent section whose children fold themselves: folds register
        on return, so folding the parent in the caller would file it AFTER its own
        children. The parent registers here first, then folds with in_toc=False.
        """
        self._toc.append((anchor, re.sub(r"<[^>]+>", "", title), sub))
        return self

    def add_meta(self, html: str, summary: str = "meta.json - all run parameters",
                 open: bool = False) -> "Document":
        """The run-parameter dump: a fold that stays out of the contents nav."""
        self._parts.append(
            '<details class="meta"{}><summary>{}</summary>{}</details>'.format(
                " open" if open else "", summary, html))
        return self

    # -- assembly ---------------------------------------------------------

    def _toc_html(self) -> str:
        if not self._toc:
            return ""
        items = "".join(
            '<li{}><a href="#{}">{}</a></li>'.format(
                ' class="sub"' if sub else "", anchor, title)
            for anchor, title, sub in self._toc)
        return ('<nav class="toc"><div class="toc-title">Contents</div>'
                '<div class="toolbar">'
                '<button type="button" onclick="setAll(true)">Expand all</button>'
                '<button type="button" onclick="setAll(false)">Collapse all</button>'
                '</div><ul>{}</ul></nav>'.format(items))

    def render(self, generated: str = "") -> str:
        """Return the complete page. `generated` stamps a generation time; pass a
        fixed string to make output reproducible for diffing."""
        stamp = generated or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sub = '<p class="subtitle">{}</p>'.format(self.subtitle) if self.subtitle else ""
        return (
            '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            "<title>{title}</title>\n{css}\n</head>\n<body>\n"
            "<h1>{title}</h1>\n{sub}\n{toc}\n{body}\n"
            '<p class="subtle">Generated {stamp}</p>\n'
            "{script}\n</body>\n</html>\n"
        ).format(title=self.title, css=stylesheet(), sub=sub,
                 toc=self._toc_html(), body="\n".join(self._parts),
                 stamp=stamp, script=_TOC_SCRIPT)
