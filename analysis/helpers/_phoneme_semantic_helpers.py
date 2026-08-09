# -*- coding: utf-8 -*-
"""
tests/_phoneme_semantic_helpers.py
===================================
Shared utilities for phoneme-semantic separation tests.

These tests investigate whether phoneme regression picks up genuine
phonological information or merely reflects semantic co-variance in the
neural signal.
"""

import os
import sys
import warnings
import gc
import pickle as pk

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.cross_decomposition import PLSRegression

# --- cleanup batch 2: reformat moved to utils.utils ---
from utils.utils import reformat
from utils import config as _cfg

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────
#: Repo-wide since 2026-08-08; see utils/config.py. Was a local 10.
N_BINS_HISTORY   = _cfg.N_BINS_HISTORY
KRR_ALPHA        = 1.5
PLS_COMPONENTS   = 10
PHONEME_EMBEDDINGS = ['panphon']
SEMANTIC_EMBEDDINGS_TO_USE = ['GloVe']   # primary confound embedding(s)

_PWESUITE_FILES = {
    'panphon':   'pwesuite_panphon_embeddings.pk',
    'token_ipa': 'pwesuite_token_ipa_embeddings.pk',
}


# ── Data loading ─────────────────────────────────────────────────────────

def load_phoneme_embeddings_for_patient(pdata):
    """Load panphon + token_ipa phoneme embeddings aligned to patient trials.

    Returns dict[emb_name] -> ndarray (n_trials, 300).
    """
    labels = pdata['clean_answer_labels']
    emb_folder = os.path.join('embeddings', 'pictureNaming extended all')
    result = {}
    for name, fname in _PWESUITE_FILES.items():
        fpath = os.path.join(emb_folder, fname)
        with open(fpath, 'rb') as f:
            embed_dict = pk.load(f)
        result[name] = _map_phoneme_embed(embed_dict, labels)
    return result


def load_semantic_embeddings_for_patient(pdata, shared_models, names=None):
    """Load semantic embeddings aligned to patient trials.

    Uses build_patient_embeddings from semantic_regression but filters to
    the requested names.

    Returns dict[emb_name] -> ndarray (n_trials, D).
    """
    from semantic_regression import build_patient_embeddings
    all_embeds = build_patient_embeddings(pdata, shared_models)
    if names is None:
        names = SEMANTIC_EMBEDDINGS_TO_USE
    return {k: v for k, v in all_embeds.items() if k in names}


def _normalize_tokens(tokens):
    return np.array([str(t).strip().lower() for t in tokens])


def _remove_number(text):
    """Strip trailing picture number (e.g., 'bat1' -> 'bat')."""
    text = str(text)
    while text and text[-1].isdigit():
        text = text[:-1]
    return text


def _map_phoneme_embed(embed_dict, target_labels):
    """Map phoneme embeddings to patient trial labels."""
    words_norm = _normalize_tokens(np.asarray(embed_dict['words']))
    embeddings = np.asarray(embed_dict['phoneme_embedding'])
    dim = embeddings.shape[1]

    exact = {w: i for i, w in enumerate(words_norm)}
    base = {}
    for i, w in enumerate(words_norm):
        base.setdefault(_remove_number(w), []).append(i)

    labels_norm = _normalize_tokens(target_labels)
    out, missing = [], []
    for t_raw, t_norm in zip(target_labels, labels_norm):
        if t_norm in exact:
            out.append(embeddings[exact[t_norm]])
        elif t_norm in base:
            out.append(np.mean([embeddings[i] for i in base[t_norm]], axis=0))
        elif _remove_number(t_norm) in base:
            out.append(np.mean([embeddings[i] for i in base[_remove_number(t_norm)]], axis=0))
        else:
            bare = ''.join(c for c in t_norm if c.isalpha())
            if bare and bare in base:
                out.append(np.mean([embeddings[i] for i in base[bare]], axis=0))
            elif bare and bare in exact:
                out.append(embeddings[exact[bare]])
            else:
                out.append(np.full(dim, np.nan, dtype=np.float32))
                missing.append(t_raw)
    if missing:
        print(f"  [WARN] {len(missing)} labels not found in phoneme vocab "
              f"(NaN assigned, trial will be dropped): {missing[:5]}")
    return np.array(out, dtype=np.float32)


# ── Trial filtering ──────────────────────────────────────────────────────

def filter_nan_phoneme_trials(pdata, phon_embeds):
    """Remove trials whose phoneme embedding is NaN (ambiguous answered word).

    Returns (pdata_filtered, phon_embeds_filtered).  Only array fields whose
    first dimension matches the trial count are sliced; all other fields are
    passed through unchanged.
    """
    n_trials = len(pdata['clean_answer_labels'])
    valid = np.ones(n_trials, dtype=bool)
    for Y in phon_embeds.values():
        valid &= ~np.isnan(Y).any(axis=1)

    n_removed = int((~valid).sum())
    if n_removed > 0:
        print(f"  [INFO] Dropping {n_removed} trial(s) with NaN phoneme embeddings "
              f"(ambiguous answered word)")

    pdata_f = {}
    for key, val in pdata.items():
        if isinstance(val, np.ndarray) and val.shape[0] == n_trials:
            pdata_f[key] = val[valid]
        else:
            pdata_f[key] = val

    phon_embeds_f = {k: v[valid] for k, v in phon_embeds.items()}
    return pdata_f, phon_embeds_f


# ── Feature construction ─────────────────────────────────────────────────


# ── Pipeline construction ────────────────────────────────────────────────

def make_kernel_pls_pipeline(n_components=PLS_COMPONENTS):
    return Pipeline([
        ('nystroem', Nystroem(kernel='rbf')),
        ('pls', PLSRegression(n_components=n_components, scale=False)),
    ])


def make_pls_pipeline(n_components=PLS_COMPONENTS):
    return Pipeline([
        ('pls', PLSRegression(n_components=n_components, scale=False)),
    ])


# ── Retrieval database + functions ───────────────────────────────────────
# The canonical retrieval procedure (mean-per-word database + mean-centring
# before cosine) lives in utils.retrieval and is shared across the codebase.
# Re-exported here so the ~12 tests that import these names keep working.
from utils.retrieval import (
    mean_embedding_per_word,
    build_retrieval_db,
    cosine_sim_matrix,
    cosine_retrieval,
    category_indep_retrieval,
    compute_retrieval_metrics,
)


# ── Partial RSA helper ──────────────────────────────────────────────────

def partial_spearman(rdm_a, rdm_b, rdm_control):
    """Partial Spearman: corr(a, b) controlling for control."""
    r_ab, _ = spearmanr(rdm_a, rdm_b)
    r_ac, _ = spearmanr(rdm_a, rdm_control)
    r_bc, _ = spearmanr(rdm_b, rdm_control)
    denom = np.sqrt(max(1 - r_ac**2, 1e-12)) * np.sqrt(max(1 - r_bc**2, 1e-12))
    return (r_ab - r_ac * r_bc) / denom


# ── Output helpers ───────────────────────────────────────────────────────

def get_out_dir(args_out_dir=None):
    """Return the output directory, creating it if needed.

    The default used to be `dirname(__file__)/results`, i.e. tests/helpers/results,
    which has never existed -- so callers only worked when passed an explicit
    --out-dir, and a *relative* fallback elsewhere in this suite is what wrote
    Tests 1-4 to <project>/test_results/, outside the repository, while Tests A-D
    landed inside it. Both halves now live under results/phoneme_semantic_dissociation/.
    """
    from utils.paths import results_dir
    out = args_out_dir or results_dir('phoneme_semantic_dissociation')
    os.makedirs(out, exist_ok=True)
    return str(out)


def discover_patients(data_folder='data', task='picture_naming'):
    """List available patients from `data_folder` that have a {patient}_{task}_df.pkl."""
    from utils.patient_data import discover_patients as _discover
    return _discover(data_folder, task)


def header(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")

def step(msg):
    print(f"  {msg}")
