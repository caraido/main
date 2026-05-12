# -*- coding: utf-8 -*-
"""
utils.utils -- BACKWARD-COMPAT shim.

The original 1412-LOC utils.py was split into themed submodules during the
code-health cleanup.  This module re-exports the public API so existing
callers (`from utils.utils import remove_number, plot_accuracy_plotly`)
continue to work without modification.

For NEW code, prefer importing from the themed submodule directly:

    from utils.io           import load_all_data, save_figure_and_source_data
    from utils.preprocessing import align_data, reformat, reformat_raw, switch_2_number
    from utils.text         import remove_number, get_sentence_tense
    from utils.plotting     import plot_accuracy_plotly, get_channel_colors
    from utils.interactive  import interactive_3d_scatter_plot
    from utils.decoder      import GeneralDecoder
"""

# Re-exports preserving the original public API.
from .io import (
    _sanitize_path_component,
    _to_json_serializable,
    save_figure_and_source_data,
    load_all_data,
)
from .preprocessing import (
    align_data,
    reformat_raw,
    reformat,
    switch_2_number,
    switch_2_category,
    ind_func,
    fix_index,
)
from .text import (
    replace_underscores,
    add_space_after_comma,
    get_sentence_tense,
    get_sentence_subject_number,
    get_sentence_subject_person,
    remove_number,
    nlp,
)
from .plotting import (
    DATA_COLORS,
    LINE_COLORS,
    get_channel_colors,
    to_rgba,
    plot_on_channel,
    plot_accuracy_plotly,
    plot_3d_scatter,
)
from .interactive import (
    interactive_3d_scatter_plot,
    interactive_channel_importance,
    interactive_confusion_accuracy,
)
from .decoder import GeneralDecoder

__all__ = [
    '_sanitize_path_component', '_to_json_serializable',
    'save_figure_and_source_data', 'load_all_data',
    'align_data', 'reformat_raw', 'reformat',
    'switch_2_number', 'switch_2_category', 'ind_func', 'fix_index',
    'replace_underscores', 'add_space_after_comma',
    'get_sentence_tense', 'get_sentence_subject_number', 'get_sentence_subject_person',
    'remove_number', 'nlp',
    'DATA_COLORS', 'LINE_COLORS',
    'get_channel_colors', 'to_rgba',
    'plot_on_channel', 'plot_accuracy_plotly', 'plot_3d_scatter',
    'interactive_3d_scatter_plot', 'interactive_channel_importance',
    'interactive_confusion_accuracy',
    'GeneralDecoder',
]
