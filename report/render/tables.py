# -*- coding: utf-8 -*-
"""
report.render.tables — DataFrame -> HTML table, and the paired CSV.

Three incompatible table renderers existed before this module: two thin wrappers
over `DataFrame.to_html(classes="results")`, one manual row loop taking a
`highlight_col`, and nine further hand-rolled loops written inline in the
generators. The hand-rolled ones dominate, and they exist for a real reason the
pandas wrappers cannot serve: they set a CSS class PER CELL (significance stars,
winner/loser shading, per-patient tiers). So `table()` takes a `cell_class` hook
rather than pretending formatting is only about floats.

The `name=` argument is the part that is new rather than consolidated. Passing it
writes the table's own source CSV next to the report. Only one generator ever did
this on its own, which is why `report.helper.html_utils` still carries two parsers
that recover numbers by regexing saved Plotly HTML -- a report that emits its
numbers cannot need scraping later.
"""

from __future__ import annotations

import html as _html
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(value, float_fmt: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if isinstance(value, (float, np.floating)):
        return float_fmt.format(value)
    return _html.escape(str(value))


def table(df: pd.DataFrame,
          name: str = "",
          out_dir=None,
          float_fmt: str = "{:.3f}",
          cell_class=None,
          row_class=None,
          css_class: str = "results",
          index: bool = False) -> str:
    """Render `df` as an HTML `<table>`, optionally writing `<out_dir>/<name>.csv`.

    cell_class(column, value, row) -> str | ""   CSS class for one cell.
    row_class(row) -> str | ""                   CSS class for one <tr>.

    Both hooks receive the RAW value, not the formatted string, so a threshold
    test reads `value < ALPHA` rather than parsing back out of "0.032".

    Passing `name` without `out_dir` raises: it means the caller asked for a
    source CSV and would otherwise get silence.
    """
    if df is None or len(df) == 0:
        return "<p class='subtle'>(no data)</p>"

    if name and out_dir is None:
        raise ValueError(
            "table(name={!r}) needs out_dir to write the paired CSV".format(name))
    if name:
        csv_path = Path(out_dir) / "{}.csv".format(name)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=index)

    frame = df.reset_index() if index else df
    head = "".join("<th>{}</th>".format(_html.escape(str(c))) for c in frame.columns)

    body = []
    for _, row in frame.iterrows():
        cells = []
        for col in frame.columns:
            value = row[col]
            klass = cell_class(col, value, row) if cell_class else ""
            cells.append("<td{}>{}</td>".format(
                ' class="{}"'.format(klass) if klass else "",
                _fmt(value, float_fmt)))
        rk = row_class(row) if row_class else ""
        body.append("<tr{}>{}</tr>".format(
            ' class="{}"'.format(rk) if rk else "", "".join(cells)))

    return '<table class="{}"><thead><tr>{}</tr></thead><tbody>{}</tbody></table>'.format(
        css_class, head, "".join(body))


def stylesheet(extra: str = "") -> str:
    """The shared stylesheet, inlined as a `<style>` element.

    Inlined rather than linked because these reports are single self-contained
    files that get copied to a run directory, emailed, and opened from disk; a
    linked stylesheet would silently render unstyled in all three cases.

    `extra` appends report-specific rules after the shared ones. Several reports
    have layout genuinely their own -- an embedding-toggle pill, a leakage matrix's
    diagonal shading, an interactive Plotly panel -- and those belong to the report,
    not in a stylesheet fifteen files share. Anything that turns out NOT to be
    specific should move into report.css instead of being pasted into a second
    `extra`.
    """
    css = Path(__file__).with_name("assets") / "report.css"
    body = css.read_text(encoding="utf-8")
    if extra:
        body += "\n/* ---- report-specific ---- */\n" + extra.strip() + "\n"
    return "<style>\n{}\n</style>".format(body)
