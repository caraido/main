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
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.cross_decomposition import PLSRegression

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────
N_BINS_HISTORY   = 10
KRR_ALPHA        = 1.5
PLS_COMPONENTS   = 10
PHONEME_EMBEDDINGS = ['panphon', 'token_ipa']
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
    labels = pdata['clean_target_labels']
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
                out.append(np.zeros(dim, dtype=np.float32))
                missing.append(t_raw)
    if missing:
        print(f"  [WARN] {len(missing)} labels not found: {missing[:5]}")
    return np.array(out, dtype=np.float32)


# ── Feature construction ─────────────────────────────────────────────────

def reformat(data, bins_per_feature):
    """Create lagged feature matrices.

    data: (n_trials, n_bins, n_channels)
    Returns list of (n_trials, n_features) arrays, one per time bin.
    """
    reformatted_data = []
    for i in range(data.shape[1]):
        reformatted = data[:, i - np.minimum(i, bins_per_feature - 1):i + 1, :]
        reformatted_data.append(reformatted.reshape(data.shape[0], -1))
    return reformatted_data


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


# ── Retrieval database ──────────────────────────────────────────────────

def build_retrieval_db(Y, labels, categories):
    """Build per-word mean embedding database + category mapping.

    Returns:
        db_embeds:       (n_unique_words, D) mean embedding per word
        unique_words:    (n_unique_words,) word labels
        word_to_cat_idx: (n_unique_words,) category index per word
        unique_cats:     (n_cats,) category labels
        word_to_idx:     dict {word_str: word_idx}
    """
    unique_words = np.unique(labels)
    word_to_idx = {w: i for i, w in enumerate(unique_words)}
    db_embeds = np.zeros((len(unique_words), Y.shape[1]), dtype=np.float64)
    db_counts = np.zeros(len(unique_words), dtype=np.int64)
    for trial_i in range(len(labels)):
        widx = word_to_idx[labels[trial_i]]
        db_embeds[widx] += Y[trial_i]
        db_counts[widx] += 1
    valid = db_counts > 0
    db_embeds[valid] /= db_counts[valid, None]

    word_cats = np.array([categories[np.where(labels == w)[0][0]]
                          for w in unique_words])
    unique_cats = np.unique(word_cats)
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    word_to_cat_idx = np.array([cat_to_idx[c] for c in word_cats])

    return db_embeds, unique_words, word_to_cat_idx, unique_cats, word_to_idx


# ── Retrieval functions ──────────────────────────────────────────────────

def cosine_retrieval(Y_pred, db_embeds):
    """Return (predicted word indices, cosine similarities to true word)."""
    pred_n = Y_pred / (np.linalg.norm(Y_pred, axis=1, keepdims=True) + 1e-10)
    db_n = db_embeds / (np.linalg.norm(db_embeds, axis=1, keepdims=True) + 1e-10)
    sims = pred_n @ db_n.T
    return np.argmax(sims, axis=1)


def category_indep_retrieval(Y_pred, db_embeds, word_to_cat_idx, n_cats):
    """Return predicted category indices via nearest centroid matching."""
    cat_centroids = np.zeros((n_cats, db_embeds.shape[1]), dtype=np.float64)
    cat_counts = np.zeros(n_cats, dtype=np.int64)
    for wi in range(len(db_embeds)):
        ci = word_to_cat_idx[wi]
        cat_centroids[ci] += db_embeds[wi]
        cat_counts[ci] += 1
    valid = cat_counts > 0
    cat_centroids[valid] /= cat_counts[valid, None]

    pred_n = Y_pred / (np.linalg.norm(Y_pred, axis=1, keepdims=True) + 1e-10)
    cat_n = cat_centroids / (np.linalg.norm(cat_centroids, axis=1, keepdims=True) + 1e-10)
    dists = 1 - pred_n @ cat_n.T
    return np.argmin(dists, axis=1)


def compute_retrieval_metrics(Y_pred, true_labels, true_cats,
                              db_embeds, unique_words, word_to_cat_idx,
                              unique_cats, word_to_idx):
    """Compute word accuracy, category-independent accuracy, and cosine sim.

    Returns dict with keys: word_bal_acc, cat_indep_bal_acc, cosine_mean.
    """
    n_cats = len(unique_cats)
    # Word-level retrieval
    pred_word_idx = cosine_retrieval(Y_pred, db_embeds)
    true_word_idx = np.array([word_to_idx[w] for w in true_labels])

    word_bal_acc = float(balanced_accuracy_score(true_word_idx, pred_word_idx))

    # Category-independent retrieval
    pred_cat_idx = category_indep_retrieval(Y_pred, db_embeds,
                                            word_to_cat_idx, n_cats)
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    true_cat_idx = np.array([cat_to_idx[c] for c in true_cats])
    cat_indep_bal_acc = float(balanced_accuracy_score(true_cat_idx, pred_cat_idx))

    # Cosine similarity: predicted vs true embedding
    true_embeds = db_embeds[true_word_idx]
    pred_n = Y_pred / (np.linalg.norm(Y_pred, axis=1, keepdims=True) + 1e-10)
    true_n = true_embeds / (np.linalg.norm(true_embeds, axis=1, keepdims=True) + 1e-10)
    cosine_mean = float(np.mean(np.sum(pred_n * true_n, axis=1)))

    return {
        'word_bal_acc': word_bal_acc,
        'cat_indep_bal_acc': cat_indep_bal_acc,
        'cosine_mean': cosine_mean,
    }


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
    """Return the output directory, creating it if needed."""
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        '..', 'test_results')
    out = args_out_dir or os.path.abspath(base)
    os.makedirs(out, exist_ok=True)
    return out


def discover_patients():
    """List available patients from data/ directory."""
    from semantic_regression import discover_patients as _discover
    return _discover()


def header(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")

def step(msg):
    print(f"  {msg}")
