# -*- coding: utf-8 -*-
"""utils/retrieval.py
====================
Canonical embedding-space retrieval procedure, shared across the codebase.

Every retrieval analysis that matches predicted embeddings against a
vocabulary database follows the SAME two conventions defined here:

  1. **Database = mean embedding per word.**  Each unique word's database
     entry is the mean of all its trials' (true or reference) embeddings,
     not a single representative trial.  See :func:`mean_embedding_per_word`.

  2. **Mean-centre before cosine / distance.**  The database centroid
     (``db_embeds.mean(axis=0)``) is subtracted from BOTH the database and the
     query predictions before any cosine similarity or L2 distance is taken.
     This removes the dominant shared direction common to all embeddings
     (the "average word meaning"), so retrieval reflects word-discriminating
     signal rather than proximity to the centroid.  Without it, a model that
     always predicts the centroid scores deceptively well.  See
     :func:`mean_center_db`.

These primitives are the single source of truth used by:
  * ``models.model.BasicRegressor._compute_retrieval_accuracy``
  * ``analysis.helpers._phoneme_semantic_helpers`` (re-exported; ~12 test files)
  * ``_archive.phoneme_semantic_dissociation.ensemble_retrieval``

``semantic_vanilla_retrieval.NeuralRetriever`` implements a different algorithm
(leave-one-out retrieval in *neural* feature space) but follows the identical
mean-per-word + mean-centre conventions.
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score

__all__ = [
    "mean_embedding_per_word",
    "build_retrieval_db",
    "mean_center_db",
    "normalize_rows",
    "cosine_sim_matrix",
    "cosine_retrieval",
    "category_indep_retrieval",
    "compute_retrieval_metrics",
]


# ── Low-level primitives ──────────────────────────────────────────────────

def normalize_rows(M, eps=1e-10):
    """Row-wise L2 normalisation (safe against zero rows)."""
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + eps)


def mean_center_db(db_embeds, queries):
    """Subtract the database centroid from both the DB and the queries.

    Returns ``(db_centered, queries_centered, db_mean)``.  This is the
    canonical centring step: ``db_mean`` is the mean across the per-word
    database rows, so both spaces are expressed as deviations from the
    "average word".
    """
    db_mean = db_embeds.mean(axis=0)
    return db_embeds - db_mean, queries - db_mean, db_mean


def mean_embedding_per_word(Y, labels):
    """Mean embedding across all trials for each unique word.

    Parameters
    ----------
    Y : ndarray (n_trials, D)
        Per-trial embeddings.
    labels : array-like (n_trials,)
        Word identity per trial.

    Returns
    -------
    unique_words : ndarray (n_words,)
        Unique words in ``np.unique`` (sorted) order.
    db_embeds : ndarray (n_words, D), float64
        ``db_embeds[i]`` is the mean of all ``Y`` rows whose label is
        ``unique_words[i]``.
    """
    labels = np.asarray(labels)
    Y = np.asarray(Y)
    unique_words, inv = np.unique(labels, return_inverse=True)
    db_embeds = np.zeros((len(unique_words), Y.shape[1]), dtype=np.float64)
    counts = np.zeros(len(unique_words), dtype=np.int64)
    np.add.at(db_embeds, inv, Y)
    np.add.at(counts, inv, 1)
    valid = counts > 0
    db_embeds[valid] /= counts[valid, None]
    return unique_words, db_embeds


# ── Retrieval database (per-word mean + category mapping) ─────────────────

def build_retrieval_db(Y, labels, categories):
    """Build per-word mean-embedding database + category mapping.

    The database entry for each unique word is the mean embedding across all
    of that word's trials (:func:`mean_embedding_per_word`).

    Returns:
        db_embeds:       (n_unique_words, D) mean embedding per word
        unique_words:    (n_unique_words,) word labels
        word_to_cat_idx: (n_unique_words,) category index per word
        unique_cats:     (n_cats,) category labels
        word_to_idx:     dict {word_str: word_idx}
    """
    labels = np.asarray(labels)
    categories = np.asarray(categories)
    unique_words, db_embeds = mean_embedding_per_word(Y, labels)
    word_to_idx = {w: i for i, w in enumerate(unique_words)}

    word_cats = np.array([categories[np.where(labels == w)[0][0]]
                          for w in unique_words])
    unique_cats = np.unique(word_cats)
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    word_to_cat_idx = np.array([cat_to_idx[c] for c in word_cats])

    return db_embeds, unique_words, word_to_cat_idx, unique_cats, word_to_idx


# ── Retrieval functions ───────────────────────────────────────────────────

def cosine_sim_matrix(Y_pred, db_embeds, center=True):
    """Cosine similarity matrix between queries and the DB.

    Returns an ``(n_query, n_db)`` matrix.  When ``center`` is True (default)
    the DB centroid is removed from both sides first (canonical convention).
    """
    if center:
        db_embeds, Y_pred, _ = mean_center_db(db_embeds, Y_pred)
    return normalize_rows(Y_pred) @ normalize_rows(db_embeds).T


def cosine_retrieval(Y_pred, db_embeds, center=True):
    """Return predicted word indices via nearest (mean-centred) cosine match."""
    sims = cosine_sim_matrix(Y_pred, db_embeds, center=center)
    return np.argmax(sims, axis=1)


def category_indep_retrieval(Y_pred, db_embeds, word_to_cat_idx, n_cats,
                             center=True):
    """Return predicted category indices via nearest centroid matching.

    Category centroids are the mean of the (mean-centred) per-word database
    rows belonging to each category, so prediction operates in the same
    centred space as :func:`cosine_retrieval`.
    """
    if center:
        db_embeds, Y_pred, _ = mean_center_db(db_embeds, Y_pred)

    cat_centroids = np.zeros((n_cats, db_embeds.shape[1]), dtype=np.float64)
    cat_counts = np.zeros(n_cats, dtype=np.int64)
    for wi in range(len(db_embeds)):
        ci = word_to_cat_idx[wi]
        cat_centroids[ci] += db_embeds[wi]
        cat_counts[ci] += 1
    valid = cat_counts > 0
    cat_centroids[valid] /= cat_counts[valid, None]

    dists = 1 - normalize_rows(Y_pred) @ normalize_rows(cat_centroids).T
    return np.argmin(dists, axis=1)


def _normalize_tokens(tokens):
    """Canonicalise string keys (strip + lowercase) for robust lookups."""
    return np.array([str(t).strip().lower() for t in tokens])


def compute_retrieval_metrics(Y_pred, true_labels, true_cats,
                              db_embeds, unique_words, word_to_cat_idx,
                              unique_cats, word_to_idx):
    """Compute word accuracy, category-independent accuracy, and cosine sim.

    All distances and cosine similarities are computed on mean-centred
    embeddings (DB centroid removed), matching the convention in
    ``model.BasicRegressor._compute_retrieval_accuracy``.

    Returns dict with keys: word_bal_acc, cat_indep_bal_acc, cosine_mean,
    n_word_dropped_unseen, n_cat_dropped_unseen.
    """
    n_cats = len(unique_cats)

    # Canonicalize string keys at lookup time so train/test casing/whitespace
    # differences do not create spurious unknown-label failures.
    norm_words = _normalize_tokens(unique_words)
    norm_true_labels = _normalize_tokens(true_labels)
    norm_cats = _normalize_tokens(unique_cats)
    norm_true_cats = _normalize_tokens(true_cats)

    norm_word_to_idx = {w: i for i, w in enumerate(norm_words)}
    norm_cat_to_idx = {c: i for i, c in enumerate(norm_cats)}

    # Mean-centre the DB once; every distance/cosine below operates on the
    # centred deviations.  Sub-calls receive already-centred inputs.
    db_c, Y_pred_c, _ = mean_center_db(db_embeds, Y_pred)

    # Word-level retrieval (computed only on mapped test labels)
    pred_word_idx_all = cosine_retrieval(Y_pred_c, db_c, center=False)
    true_word_idx_all = np.array([norm_word_to_idx.get(w, -1) for w in norm_true_labels], dtype=np.int64)
    word_valid = true_word_idx_all >= 0
    dropped_word = int((~word_valid).sum())

    if np.any(word_valid):
        true_word_idx = true_word_idx_all[word_valid]
        pred_word_idx = pred_word_idx_all[word_valid]
        word_bal_acc = float(balanced_accuracy_score(true_word_idx, pred_word_idx))

        # Cosine similarity: predicted vs true (centred) embedding for valid words
        true_embeds = db_c[true_word_idx]
        pred_valid = Y_pred_c[word_valid]
        pred_n = normalize_rows(pred_valid)
        true_n = normalize_rows(true_embeds)
        cosine_mean = float(np.mean(np.sum(pred_n * true_n, axis=1)))
    else:
        word_bal_acc = np.nan
        cosine_mean = np.nan

    # Category-independent retrieval (computed only on mapped categories)
    pred_cat_idx_all = category_indep_retrieval(Y_pred_c, db_c,
                                                word_to_cat_idx, n_cats,
                                                center=False)
    true_cat_idx_all = np.array([norm_cat_to_idx.get(c, -1) for c in norm_true_cats], dtype=np.int64)
    cat_valid = true_cat_idx_all >= 0
    dropped_cat = int((~cat_valid).sum())

    if np.any(cat_valid):
        true_cat_idx = true_cat_idx_all[cat_valid]
        pred_cat_idx = pred_cat_idx_all[cat_valid]
        cat_indep_bal_acc = float(balanced_accuracy_score(true_cat_idx, pred_cat_idx))
    else:
        cat_indep_bal_acc = np.nan

    return {
        'word_bal_acc': word_bal_acc,
        'cat_indep_bal_acc': cat_indep_bal_acc,
        'cosine_mean': cosine_mean,
        'n_word_dropped_unseen': dropped_word,
        'n_cat_dropped_unseen': dropped_cat,
    }
