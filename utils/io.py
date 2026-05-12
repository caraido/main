# -*- coding: utf-8 -*-
"""utils.io -- file/directory I/O helpers."""

import os
import json
import pickle
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as sio
import mat73
import matplotlib.pyplot as plt

def _sanitize_path_component(name):
    """Return a filesystem-safe folder/file component for Windows/macOS/Linux."""
    if name is None:
        return "unknown"
    safe = re.sub(r'[<>:"/\\|?*]', '_', str(name)).strip()
    safe = re.sub(r'\s+', '_', safe)
    return safe or "unknown"


def _to_json_serializable(obj):
    """Convert common scientific Python objects to JSON-serializable structures."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    return obj


def save_figure_and_source_data(
    fig,
    source_data,
    patient,
    task,
    figure_name,
    base_dir='results',
    file_name=None,
    figure_formats=('html',),
    data_format='pkl',
    save_data=True,
    save_metadata=True,
):
    """
    Save a figure and optional source data using a standardized folder layout.

    Folder layout:
        <base_dir>/<patient>/<task>/

    Parameters
    ----------
    fig : plotly or matplotlib figure
        Figure object to save. Plotly and Matplotlib figures are supported.
    source_data : any
        Data used to create the figure. Recommended: dict of arrays/lists/scalars.
    patient : str
        Patient identifier (becomes top-level folder).
    task : str
        Task identifier (becomes subfolder under patient).
    figure_name : str
        Logical figure name used in filenames and metadata.
    base_dir : str
        Root output directory.
    file_name : str or None
        Optional custom filename stem. If None, uses figure_name.
    figure_formats : tuple/list of str
        Figure output formats. Examples: ('html',), ('html', 'png'), ('png',).
        For Plotly PNG export, kaleido must be installed.
    data_format : str
        Source data format: 'pkl', 'npz', or 'json'.
    save_data : bool
        If True, source_data will be saved.
    save_metadata : bool
        If True, writes a metadata JSON log file.

    Returns
    -------
    dict
        Paths of saved files and output directory.
    """
    safe_patient = _sanitize_path_component(patient)
    safe_task = _sanitize_path_component(task)
    safe_figure = _sanitize_path_component(figure_name)
    safe_stem = _sanitize_path_component(file_name) if file_name else safe_figure

    output_dir = Path(base_dir) / safe_patient / safe_task
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {'output_dir': str(output_dir), 'figure_paths': [], 'data_path': None, 'metadata_path': None}

    # Save figure in requested formats.
    for fmt in figure_formats:
        fmt_lower = str(fmt).lower().lstrip('.')
        fig_path = output_dir / f"{safe_stem}.{fmt_lower}"

        if hasattr(fig, 'write_html') and hasattr(fig, 'write_image'):
            if fmt_lower == 'html':
                fig.write_html(str(fig_path))
            else:
                fig.write_image(str(fig_path))
        elif hasattr(fig, 'savefig'):
            fig.savefig(str(fig_path), bbox_inches='tight')
        else:
            raise TypeError("Unsupported figure type. Provide a Plotly or Matplotlib figure.")

        saved_files['figure_paths'].append(str(fig_path))

    # Save source data in requested format.
    if save_data:
        data_fmt = str(data_format).lower()
        data_path = output_dir / f"{safe_stem}_source.{data_fmt}"

        if data_fmt == 'pkl':
            with open(data_path, 'wb') as f:
                pickle.dump(source_data, f)
        elif data_fmt == 'npz':
            if isinstance(source_data, dict):
                np.savez_compressed(str(data_path), **source_data)
            else:
                np.savez_compressed(str(data_path), data=source_data)
        elif data_fmt == 'json':
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(_to_json_serializable(source_data), f, indent=2)
        else:
            raise ValueError("data_format must be one of: 'pkl', 'npz', 'json'.")

        saved_files['data_path'] = str(data_path)

    # Write metadata log for reproducibility.
    if save_metadata:
        metadata = {
            'saved_at': datetime.now().isoformat(timespec='seconds'),
            'patient': patient,
            'task': task,
            'figure_name': figure_name,
            'file_stem': safe_stem,
            'figure_formats': [str(f).lower().lstrip('.') for f in figure_formats],
            'data_format': data_format if save_data else None,
            'figure_paths': saved_files['figure_paths'],
            'data_path': saved_files['data_path'],
            'working_directory': os.getcwd(),
        }
        metadata_path = output_dir / f"{safe_stem}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        saved_files['metadata_path'] = str(metadata_path)

    return saved_files


def load_all_data(path):
    try:
        data = sio.loadmat(path)['all_data']
    except:
        data = np.array(mat73.loadmat(path)['all_data'], dtype=object)
    return data
