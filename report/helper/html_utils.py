# -*- coding: utf-8 -*-
"""
report.helper.html_utils -- shared utilities for HTML report scripts.

Public API:
    _decode_bdata          -- decode a base64'd numpy array string back into ndarray
    fig_to_base64          -- render a matplotlib Figure to a base64 PNG string
    extract_vanilla_html   -- pull neural/chance traces out of a vanilla report HTML
"""

import io
import re
import json
import base64

import numpy as np
import matplotlib.pyplot as plt


def _decode_bdata(bdata_str, dtype='f8'):
    """Decode a base64 binary data string from a Plotly trace into a numpy array."""
    raw = base64.b64decode(bdata_str)
    np_dtype = np.float64 if dtype == 'f8' else np.float32
    return np.frombuffer(raw, dtype=np_dtype)


def fig_to_base64(fig, dpi=140) -> str:
    """Render a matplotlib Figure to a base64-encoded PNG string and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def extract_vanilla_html(html_path):
    """Extract neural and chance traces from a vanilla retrieval Plotly HTML report.

    Returns
    -------
    (x_arr, neural_y, chance_y) : tuple of ndarray-or-None
        x: time bins (or indices), neural_y: neural retrieval accuracy,
        chance_y: chance/null accuracy.  Any of the three may be None if not found.

    Returns (None, None, None) if the file can't be opened, parsed, or contains
    no recognisable Plotly traces.
    """
    x_arr, neural_y, chance_y = None, None, None

    try:
        with open(html_path, encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return None, None, None

    match = re.search(
        r'Plotly\.newPlot\(\s*"[^"]+"\s*,\s*(\[.*?\])\s*,',
        content, re.DOTALL,
    )
    if not match:
        return None, None, None
    try:
        traces = json.loads(match.group(1))
    except Exception:
        return None, None, None

    for t in traces:
        y_data = t.get('y', {})
        x_data = t.get('x', {})
        trace_name = t.get('name', '')

        arr = None
        xarr = None
        if isinstance(y_data, dict) and 'bdata' in y_data:
            try:
                arr = _decode_bdata(y_data['bdata'], y_data.get('dtype', 'f8'))
                if isinstance(x_data, dict) and 'bdata' in x_data:
                    xarr = _decode_bdata(x_data['bdata'], x_data.get('dtype', 'f8'))
                elif isinstance(x_data, (list, tuple)):
                    xarr = np.array(x_data, dtype=np.float64)
            except Exception:
                arr, xarr = None, None
        elif isinstance(y_data, (list, tuple)):
            arr = np.array(y_data, dtype=np.float64)
            if isinstance(x_data, (list, tuple)):
                xarr = np.array(x_data, dtype=np.float64)

        if arr is None:
            continue

        if trace_name == 'Neural':
            neural_y = arr
            if xarr is not None:
                x_arr = xarr
        elif trace_name == 'chance':
            chance_y = arr

    return x_arr, neural_y, chance_y
