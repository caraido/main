# -*- coding: utf-8 -*-
"""
report.render — the MARKUP layer. Computes nothing.

The counterpart to `report.helper`, which computes and emits no markup. A report
generator loads data through `report.helper` (or its own analysis module), then
renders it through here. Keeping the two apart is what lets a visualization change
touch one stylesheet and a handful of functions instead of fifteen generators.

Before this package, every generator carried its own copy of the same four things:
a `<style>` block (15 of 15 files, 17-170 lines each, in three drifted colour
lineages), a DataFrame-to-table renderer (three incompatible implementations plus
nine hand-rolled row loops), a set of coloured callout `<div>`s (the same five
semantic roles spelled `.box` / `.method-box` / `.met` / `.methods` / `.notes` in
four different palettes), and — in one file — a collapsible-section and
table-of-contents system.

Modules
-------
document   Document: collapsible <details> sections, a table-of-contents nav, and
           the final page assembly. Replaces a module-level _TOC global.
tables     table(): DataFrame -> <table>, with per-cell class hooks, and the
           optional paired CSV that lets a reader get the numbers without
           scraping the HTML.
blocks     callout(): the five semantic box kinds, on one palette.
assets/    report.css — the single stylesheet.

The CSV option is not a consolidation of existing behaviour; it is new. Only one
generator ever wrote a CSV of its own tables, which is why `report.helper.html_utils`
still carries two parsers that recover numbers by regexing saved Plotly HTML. Those
parsers can be deleted once the reports that feed them emit `source_data` instead.
"""

from .blocks import callout, CALLOUT_KINDS
from .document import Document
from .tables import table, stylesheet

__all__ = ["Document", "callout", "CALLOUT_KINDS", "table", "stylesheet"]
