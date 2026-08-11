# -*- coding: utf-8 -*-
"""
report.helper.results_loader — Load per-patient data from a run folder.

The PKL is the only source. It carries the per-epoch observed AND shuffled-null arrays,
which is what every downstream test compares.

**The CSV/HTML fallback was deleted on 2026-08-11** (Alec). `load_patient_from_csv` and
`extract_null_from_html` reconstructed a patient from `top1_decoding_source_data.csv` and
scraped the null out of saved Plotly HTML. Two things were wrong with it:

1. When the HTML could not be read it silently substituted **theoretical chance** --
   `1/6` category, `1/60` word -- for the empirical shuffled null. Those constants also
   contradict `AGENTS.md`: category chance is per participant (0.143-0.200 measured; the
   pinned run spans 0.1427-0.1733), and word chance depends on vocabulary size.
2. Its trigger was unreachable in practice. The stated reason was "PKL too large, e.g.
   WBH at 2.6 GB"; WBH is now 0.7 GB and all 15 PKLs of the pinned run load, so the
   fallback fired for no patient. Removing it moved no number -- verified by diffing the
   full `compute_significance` table before and after.

A patient whose PKL will not load is now a hard error naming that patient, not a silent
drop. See `docs/experiments/014-report-fig-dir-null.md`.

Key function:
  - load_patient_from_pkl(): full data from PKL (per-epoch arrays for obs + null)
"""

import os
import sys
import types
import warnings
import numpy as np
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
    for emb in data.get('regressors', {}).keys():
        br = data['regressors'][emb]
        records[emb] = {
            # confounded category (category of predicted word) — kept for back-compat
            'cat_obs':        np.array(br.all_retrieval_category_balanced_acc),
            'cat_null':       np.array(br.all_retrieval_category_chance_balanced_acc),
            # independent category (separate nearest-centroid in category space)
            'cat_indep_obs':  np.array(br.all_retrieval_category_indep_balanced_acc),
            'cat_indep_null': np.array(br.all_retrieval_category_indep_chance_balanced_acc),
            # word
            'word_obs':       np.array(br.all_retrieval_word_balanced_acc),
            'word_null':      np.array(br.all_retrieval_chance_word_balanced_acc),
        }
    return records
