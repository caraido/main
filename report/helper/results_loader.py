# -*- coding: utf-8 -*-
"""
report.helper.results_loader — Load per-patient data from a run folder.

Handles:
  - Loading PKL files (with torch stub to avoid requiring PyTorch)
  - Extracting per-epoch observed and null accuracy arrays
  - Decoding binary trace data from Plotly HTML figures
  - Chunked CSV reading for large patients (e.g., WBH)

Key functions:
  - load_patient_from_pkl(): Full data from PKL (per-epoch arrays for obs + null)
  - load_patient_from_csv(): Fallback when PKL is too large or unavailable
  - extract_null_from_html(): Extract chance baseline from Plotly HTML figures
"""

import os
import sys
import types
import json
import base64
import warnings
import numpy as np
import pandas as pd
from .config import EMBEDDING_NAMES

warnings.filterwarnings('ignore')

# ─── Module stubs for loading PKL without torch ──────────────────────────────
# The PKL files contain BasicRegressor objects that import torch at unpickle
# time. These stubs allow loading the objects without having PyTorch installed.
# All numpy arrays (the data we need) are accessible on the deserialized object.

def _install_stubs():
    """Create fake module stubs so dill can unpickle BasicRegressor objects."""
    for mod_name in ['torch', 'models', 'models.model']:
        sys.modules.setdefault(mod_name, types.ModuleType(mod_name))

    class FakeBasicRegressor:
        """Placeholder class that accepts any attribute set by dill."""
        pass

    sys.modules['models'].BasicRegressor = FakeBasicRegressor
    sys.modules['models.model'].BasicRegressor = FakeBasicRegressor

_install_stubs()

try:
    import dill
except ImportError:
    os.system(f"{sys.executable} -m pip install dill --break-system-packages -q")
    import dill


# ═══════════════════════════════════════════════════════════════════════════════
# PKL loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_pkl_raw(pkl_path, max_bytes=10_000_000_000):
    """
    Load a raw PKL dict from disk.

    Parameters
    ----------
    pkl_path : str
        Path to the .pkl file.
    max_bytes : int
        Skip files larger than this to prevent OOM (default 3 GB).

    Returns
    -------
    dict or None
        The unpickled dictionary, or None if the file exceeds max_bytes.
    """
    size = os.path.getsize(pkl_path)
    if size > max_bytes:
        print(f"    [skip] PKL too large ({size / 1e6:.0f} MB > {max_bytes / 1e6:.0f} MB)")
        return None
    with open(pkl_path, 'rb') as f:
        return dill.load(f)


def load_patient_from_pkl(pkl_path):
    """
    Extract per-epoch accuracy arrays from a patient's PKL file.

    Each embedding's regressor stores:
      - all_retrieval_category_balanced_acc       (n_epochs, n_bins)
      - all_retrieval_category_chance_balanced_acc (n_epochs, n_bins) [shuffled null]
      - all_retrieval_word_balanced_acc            (n_epochs, n_bins)
      - all_retrieval_chance_word_balanced_acc     (n_epochs, n_bins) [shuffled null]

    Returns
    -------
    dict[str, dict] or None
        Keys are embedding names; values contain 'cat_obs', 'cat_null',
        'word_obs', 'word_null' arrays. Returns None if PKL cannot be loaded.
    """
    data = load_pkl_raw(pkl_path)
    if data is None:
        return None

    records = {}
    for emb in EMBEDDING_NAMES:
        if emb not in data.get('regressors', {}):
            continue
        br = data['regressors'][emb]
        records[emb] = {
            'cat_obs':   np.array(br.all_retrieval_category_balanced_acc),
            'cat_null':  np.array(br.all_retrieval_category_chance_balanced_acc),
            'word_obs':  np.array(br.all_retrieval_word_balanced_acc),
            'word_null': np.array(br.all_retrieval_chance_word_balanced_acc),
        }
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# HTML null extraction (Plotly binary-encoded traces)
# ═══════════════════════════════════════════════════════════════════════════════

def _decode_bdata(bdata_str, dtype='f8'):
    """Decode a base64 binary data string from a Plotly trace."""
    b64 = bdata_str.replace('\u002f', '/').replace('\u003d', '=')
    raw = base64.b64decode(b64)
    np_dtype = np.float64 if dtype == 'f8' else np.float32
    return np.frombuffer(raw, dtype=np_dtype)


def extract_null_from_html(html_path):
    """
    Extract the 'chance' trace's y-values from a Plotly HTML figure.

    The Plotly figures store trace data as base64-encoded binary in the
    ``Plotly.newPlot(...)`` call. This function parses the JSON trace array,
    finds the trace named 'chance', and decodes its y-data.

    Parameters
    ----------
    html_path : str
        Path to the .html file.

    Returns
    -------
    np.ndarray or None
        Chance values per time bin, or None if parsing fails.
    """
    if not os.path.exists(html_path):
        return None

    with open(html_path, 'r') as f:
        content = f.read()

    # Find the last Plotly.newPlot(...) call and extract the trace JSON array
    idx = content.rfind('Plotly.newPlot(')
    if idx < 0:
        return None
    start = content.find('[', idx)
    depth = 0
    for i, ch in enumerate(content[start:]):
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
        if depth == 0:
            end = start + i + 1
            break

    traces = json.loads(content[start:end])
    chance_trace = next((t for t in traces if t.get('name') == 'chance'), None)
    if chance_trace is None:
        return None

    return _decode_bdata(
        chance_trace['y']['bdata'],
        chance_trace['y'].get('dtype', 'f8'),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CSV fallback loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_patient_from_csv(run_dir, patient, fig_dir=None):
    """
    Reconstruct per-epoch accuracies from top1_decoding_source_data.csv
    when the PKL is too large to load (e.g., WBH at 2.6 GB).

    Uses chunked reading for memory efficiency and extracts null baselines
    from the Plotly HTML figures.

    Parameters
    ----------
    run_dir : str
        Path to the run's results directory (e.g., results/semantic_regression/{run_id}).
    patient : str
        Patient ID (e.g., 'WBH').
    fig_dir : str or None
        Path to the run's figures directory for HTML null extraction.

    Returns
    -------
    dict or None
        Same format as load_patient_from_pkl but with 'from_csv' flag set.
    """
    patient_dir = os.path.join(run_dir, patient)
    top1_path   = os.path.join(patient_dir, 'top1_decoding_source_data.csv')
    pts_path    = os.path.join(patient_dir, 'per_time_scores.csv')
    if not os.path.exists(top1_path) or not os.path.exists(pts_path):
        return None

    pts = pd.read_csv(pts_path)

    # Identify best time bin per embedding for category and word accuracy
    best_bins = {}
    for emb in EMBEDDING_NAMES:
        sub = pts[pts['embedding'] == emb].sort_values('bin_index')
        if len(sub) == 0:
            continue
        cat_best  = int(sub.loc[sub['category_balanced_acc'].idxmax(), 'bin_index'])
        word_best = int(sub.loc[sub['word_balanced_acc'].idxmax(), 'bin_index'])
        best_bins[emb] = (cat_best, word_best)

    needed_bins = set()
    for cb, wb in best_bins.values():
        needed_bins.add(cb)
        needed_bins.add(wb)

    # Chunked read — only keep rows at the best bins (memory-efficient for 443 MB files)
    chunks = []
    for chunk in pd.read_csv(top1_path, chunksize=500_000):
        filtered = chunk[chunk['bin_index'].isin(needed_bins)]
        if len(filtered):
            chunks.append(filtered)
    top1 = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    # Extract null baselines from HTML figures
    cat_null_arr  = None
    word_null_arr = None
    if fig_dir is not None:
        cat_html  = os.path.join(fig_dir, patient, 'category_retrieval_balanced_acc.html')
        word_html = os.path.join(fig_dir, patient, 'word_retrieval_balanced_acc.html')
        cat_null_arr  = extract_null_from_html(cat_html)
        word_null_arr = extract_null_from_html(word_html)

    records = {}
    for emb in EMBEDDING_NAMES:
        if emb not in best_bins:
            continue
        cat_best, word_best = best_bins[emb]
        emb_df = top1[top1['embedding'] == emb]

        def _per_epoch_bal_acc(df, best_bin, true_col, correct_col):
            """Compute balanced accuracy per epoch at a specific time bin."""
            sub = df[df['bin_index'] == best_bin]
            accs = []
            for ep in sorted(sub['epoch'].unique()):
                ep_df = sub[sub['epoch'] == ep]
                r = ep_df.groupby(true_col)[correct_col].mean()
                accs.append(r.mean())
            return np.array(accs)

        cat_obs  = _per_epoch_bal_acc(emb_df, cat_best,  'true_category', 'category_correct')
        word_obs = _per_epoch_bal_acc(emb_df, word_best, 'true_word',     'word_correct')

        cn = cat_null_arr[cat_best]   if cat_null_arr  is not None else 1.0 / 6
        wn = word_null_arr[word_best] if word_null_arr is not None else 1.0 / 60

        records[emb] = {
            'obs_cat_at_best':  cat_obs,
            'obs_word_at_best': word_obs,
            'null_cat_mean':    cn,
            'null_word_mean':   wn,
            'cat_best_bin':     cat_best,
            'word_best_bin':    word_best,
            'from_csv':         True,
        }
    return records
