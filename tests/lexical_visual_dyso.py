"""
tests/lexical_visual_dyso.py
============================
Test whether iEEG neural geometry is more aligned with lexical-semantic
(GloVe/FastText) or visual (DINOv2/SimCLR) embedding subspaces, using DySO
to separate the unique and shared components of each modality.

Pipeline
--------
1. Load embeddings (GloVe, FastText, DINOv2, SimCLR) per patient.
2. Reduce each to D_pca dims via PCA; concatenate within modality:
     X_lex  = hstack(GloVe_pca, FastText_pca)   # (n_words, 2*D_pca)
     X_vis  = hstack(DINOv2_pca, SimCLR_pca)    # (n_words, 2*D_pca)
3. Run DySO([X_lex, X_vis]) → orthonormal bases:
     U_lex_unique  (d × k_lex)  — directions unique to lexical-semantic
     U_vis_unique  (d × k_vis)  — directions unique to visual
     U_shared      (d × k_sh)   — directions shared by both modalities
4. RSA: at each time bin, compute per-word mean neural RDM and correlate
   (Spearman) with embedding RDMs in each subspace. Null via word permutation.
5. Regression: Kernel PLS neural→subspace targets (cross-validated epochs);
   cosine retrieval; word and category balanced accuracy per subspace.
6. Save per-patient and combined CSVs.

Premise: if neural geometry is lexical-semantic, RSA and retrieval should be
significantly better for U_lex_unique than U_vis_unique across patients.

Usage (from main/):
    python -m tests.lexical_visual_dyso
    python -m tests.lexical_visual_dyso --patients VB LH --pca_dims 30 --epochs 5

Output:
    tests/results/lexical_visual_dyso_rsa_{patient}.csv  — per-bin RSA stats
    tests/results/lexical_visual_dyso_reg_{patient}.csv  — per-bin regression stats
    tests/results/lexical_visual_dyso_rsa_all.csv
    tests/results/lexical_visual_dyso_reg_all.csv

Key columns (RSA CSV):
    patient, bin_index, time_ms,
    r_lex_unique, r_vis_unique, r_shared,       # Spearman r vs neural RDM
    z_lex_unique, z_vis_unique, z_shared,       # z-score vs permutation null
    null_mean_lex_unique, null_std_lex_unique, ... (same for vis / shared)
    k_lex_unique, k_vis_unique, k_shared, d_common, pca_dims

Key columns (Regression CSV):
    patient, bin_index, time_ms, epoch,
    word_acc_lex_unique, word_acc_vis_unique, word_acc_shared,
    cat_acc_lex_unique,  cat_acc_vis_unique,  cat_acc_shared,
    cosine_lex_unique,   cosine_vis_unique,   cosine_shared,
    k_lex_unique, k_vis_unique, k_shared, d_common
"""

import os
import sys
import argparse
import warnings
import gc
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._phoneme_semantic_helpers import (
    reformat,
    get_out_dir,
    build_retrieval_db,
    cosine_retrieval,
    category_indep_retrieval,
    compute_retrieval_metrics,
    N_BINS_HISTORY,
    header,
    step,
)
from semantic_regression import (
    load_patient_data,
    load_shared_embedding_models,
    build_patient_embeddings,
    discover_patients,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))
from dyso import dyso as run_dyso


# ── Constants ─────────────────────────────────────────────────────────────────
PCA_DIMS_DEFAULT   = 30      # per-embedding PCA; d_common = 2 * PCA_DIMS
PLS_COMPONENTS     = 10
N_EPOCHS_DEFAULT   = 5
N_SHUFFLE          = 500     # word permutations for RSA null distribution
BIN_SIZE_MS        = 100
SEED               = 42
VAR_CUTOFF         = 99.0    # DySO variance cutoff

LEX_NAMES = ['GloVe', 'FastText']
VIS_NAMES = ['DINOv2', 'SimCLR']

RESULTS_SUBDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'results'
)


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _rdm_cosine(X: np.ndarray) -> np.ndarray:
    """Upper-triangle cosine dissimilarity vector from (n × d) matrix."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    Xn = X / norms
    sim = Xn @ Xn.T
    n = X.shape[0]
    idx = np.triu_indices(n, k=1)
    return (1.0 - sim)[idx]


def _pca_fit_transform(X_fit: np.ndarray, n_components: int):
    """Fit StandardScaler+PCA on X_fit, return (X_reduced, scaler, pca)."""
    n_components = min(n_components, X_fit.shape[0] - 1, X_fit.shape[1])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fit)
    pca = PCA(n_components=n_components, random_state=SEED)
    X_reduced = pca.fit_transform(X_scaled)
    return X_reduced, scaler, pca


def _word_mean(X_trial: np.ndarray, labels: np.ndarray, unique_words: np.ndarray) -> np.ndarray:
    """Average trial-level embeddings per unique word.

    Returns (n_unique_words, d).
    """
    out = np.zeros((len(unique_words), X_trial.shape[1]), dtype=np.float64)
    w2i = {w: i for i, w in enumerate(unique_words)}
    cnts = np.zeros(len(unique_words), dtype=np.int64)
    for i, lab in enumerate(labels):
        if lab in w2i:
            out[w2i[lab]] += X_trial[i]
            cnts[w2i[lab]] += 1
    valid = cnts > 0
    out[valid] /= cnts[valid, None]
    return out


def _make_pipeline(n_components: int = PLS_COMPONENTS) -> Pipeline:
    return Pipeline([
        ('nystroem', Nystroem(kernel='rbf', random_state=SEED)),
        ('pls', PLSRegression(n_components=n_components, scale=False)),
    ])


def _word_stratified_split(labels: np.ndarray, unique_words, split: float,
                           rng: np.random.Generator):
    """Split trials ensuring every word appears in the test set."""
    n = len(labels)
    n_test_target = max(int(n * split), 1)
    word_trials: dict = {w: [] for w in unique_words}
    for i, lab in enumerate(labels):
        if lab in word_trials:
            word_trials[lab].append(i)

    test_set = set()
    singleton_set = set()
    for w in unique_words:
        trials = word_trials[w]
        if not trials:
            continue
        if len(trials) == 1:
            test_set.add(trials[0])
            singleton_set.add(trials[0])
        else:
            test_set.add(int(rng.choice(trials)))

    # Fill up to target size from remaining trials
    remaining = [i for i in range(n) if i not in test_set]
    rng.shuffle(remaining)
    needed = max(n_test_target - len(test_set), 0)
    test_set.update(remaining[:needed])

    test_idx  = np.array(sorted(test_set))
    train_idx = np.array(
        [i for i in range(n) if i not in test_set or i in singleton_set]
    )
    return train_idx, test_idx


# ── DySO subspace decomposition ───────────────────────────────────────────────

def build_dyso_bases(embed_dict: dict, labels: np.ndarray,
                     pca_dims: int) -> dict | None:
    """
    Compute DySO decomposition of GloVe+FastText vs DINOv2+SimCLR embeddings.

    Returns a dict with:
        U_lex, U_vis, U_shared  — orthonormal bases (d × k_*)
        X_lex_trial, X_vis_trial — (n_valid_trials × d) projected embeddings
        X_lex_word, X_vis_word   — (n_words × d) word-level projected embeddings
        unique_words             — (n_words,)
        valid_mask               — (n_trials,) bool selecting valid trials
        pca_dims, d_common, k_lex, k_vis, k_shared
    Returns None on failure.
    """
    # Require all four embeddings
    for name in LEX_NAMES + VIS_NAMES:
        if name not in embed_dict:
            print(f"  [DySO] SKIP: embedding '{name}' not found")
            return None
        if np.all(np.isnan(embed_dict[name])):
            print(f"  [DySO] SKIP: embedding '{name}' is all-NaN")
            return None

    # Build valid-trial mask: no NaN in any of the four embeddings
    valid_mask = np.ones(len(labels), dtype=bool)
    for name in LEX_NAMES + VIS_NAMES:
        valid_mask &= ~np.isnan(embed_dict[name]).any(axis=1)

    n_valid = int(valid_mask.sum())
    if n_valid < 10:
        print(f"  [DySO] SKIP: only {n_valid} valid trials (need ≥10)")
        return None

    valid_labels = labels[valid_mask]
    unique_words = np.unique(valid_labels)
    n_words = len(unique_words)

    # Adapt PCA dims so DySO has enough samples (rule-of-thumb: n_words > d)
    pca_dims_actual = min(pca_dims, max(n_words // 4, 5))
    d_common = 2 * pca_dims_actual
    step(f"DySO setup: n_words={n_words}, pca_dims={pca_dims_actual}, "
         f"d_common={d_common}, n_valid_trials={n_valid}")

    # Reduce each embedding via PCA fitted on word-level means, then
    # apply the same transform to trial-level embeddings.
    lex_word_parts, lex_trial_parts = [], []
    vis_word_parts, vis_trial_parts = [], []

    for name in LEX_NAMES:
        E_trial = embed_dict[name][valid_mask].astype(np.float64)
        E_word  = _word_mean(E_trial, valid_labels, unique_words)
        E_word_pca, scaler, pca = _pca_fit_transform(E_word, pca_dims_actual)
        E_trial_pca = pca.transform(scaler.transform(E_trial))
        lex_word_parts.append(E_word_pca)
        lex_trial_parts.append(E_trial_pca)

    for name in VIS_NAMES:
        E_trial = embed_dict[name][valid_mask].astype(np.float64)
        E_word  = _word_mean(E_trial, valid_labels, unique_words)
        E_word_pca, scaler, pca = _pca_fit_transform(E_word, pca_dims_actual)
        E_trial_pca = pca.transform(scaler.transform(E_trial))
        vis_word_parts.append(E_word_pca)
        vis_trial_parts.append(E_trial_pca)

    X_lex_word  = np.hstack(lex_word_parts)    # (n_words, d_common)
    X_vis_word  = np.hstack(vis_word_parts)
    X_lex_trial = np.hstack(lex_trial_parts)   # (n_valid_trials, d_common)
    X_vis_trial = np.hstack(vis_trial_parts)

    # Run DySO
    step("Running DySO decomposition (may take 1-2 min)…")
    t0 = time.time()
    try:
        result = run_dyso([X_lex_word, X_vis_word],
                          var_cutoff=VAR_CUTOFF, verbosity=0)
    except Exception as exc:
        print(f"  [DySO] FAILED: {exc}")
        return None

    dt = time.time() - t0
    U_lex    = result.unique.get((0,), np.zeros((d_common, 0)))
    U_vis    = result.unique.get((1,), np.zeros((d_common, 0)))
    U_shared = result.shared

    k_lex, k_vis, k_shared = U_lex.shape[1], U_vis.shape[1], U_shared.shape[1]
    step(f"DySO done in {dt:.1f}s — k_lex={k_lex}, k_vis={k_vis}, k_shared={k_shared}")

    # Log variance explained
    for cond_key, ve in result.var_explained.items():
        modality = 'lex' if cond_key == 'cond_0' else 'vis'
        parts = ', '.join(f"{k}: {v:.1f}%" for k, v in ve.items())
        step(f"  var explained [{modality}]: {parts}")

    return dict(
        U_lex=U_lex, U_vis=U_vis, U_shared=U_shared,
        X_lex_trial=X_lex_trial, X_vis_trial=X_vis_trial,
        X_lex_word=X_lex_word,   X_vis_word=X_vis_word,
        unique_words=unique_words, valid_mask=valid_mask,
        pca_dims=pca_dims_actual, d_common=d_common,
        k_lex=k_lex, k_vis=k_vis, k_shared=k_shared,
    )


# ── RSA ───────────────────────────────────────────────────────────────────────

def compute_rsa_timecourse(
    data: np.ndarray,             # (n_trials, n_bins, n_channels)
    labels: np.ndarray,           # (n_trials,)
    dyso_info: dict,
    n_shuffle: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Compute RSA at every time bin.

    For each bin:
      - Build per-word mean neural matrix → neural RDM
      - Compute embedding RDMs in each DySO subspace
      - Spearman r(neural, subspace) + permutation z-score
    """
    valid_mask   = dyso_info['valid_mask']
    unique_words = dyso_info['unique_words']
    U_lex        = dyso_info['U_lex']
    U_vis        = dyso_info['U_vis']
    U_shared     = dyso_info['U_shared']
    X_lex_word   = dyso_info['X_lex_word']
    X_vis_word   = dyso_info['X_vis_word']
    k_lex        = dyso_info['k_lex']
    k_vis        = dyso_info['k_vis']
    k_shared     = dyso_info['k_shared']
    d_common     = dyso_info['d_common']
    pca_dims     = dyso_info['pca_dims']

    # Filter to valid trials
    data_valid   = data[valid_mask]              # (n_valid, n_bins, n_ch)
    labels_valid = labels[valid_mask]

    n_bins = data_valid.shape[1]
    n_words = len(unique_words)

    # Pre-compute fixed embedding subspace RDMs
    def subspace_rdm(X_word, U):
        if U.shape[1] == 0:
            return None
        return _rdm_cosine(X_word @ U)

    rdm_lex    = subspace_rdm(X_lex_word, U_lex)
    rdm_vis    = subspace_rdm(X_vis_word, U_vis)
    # Shared: average projection from both modalities
    rdm_shared = None
    if k_shared > 0:
        Z_shared = (X_lex_word @ U_shared + X_vis_word @ U_shared) / 2.0
        rdm_shared = _rdm_cosine(Z_shared)

    # Pre-compute per-word mean neural at every bin → (n_bins, n_words, n_ch)
    step("Pre-computing per-word mean neural features per bin…")
    w2i = {w: i for i, w in enumerate(unique_words)}
    M_neural = np.zeros((n_bins, n_words, data_valid.shape[2]), dtype=np.float32)
    cnts = np.zeros(n_words, dtype=np.int64)
    for i, lab in enumerate(labels_valid):
        if lab in w2i:
            M_neural[:, w2i[lab], :] += data_valid[i]   # (n_bins, n_ch) broadcast
            cnts[w2i[lab]] += 1
    for wi in range(n_words):
        if cnts[wi] > 0:
            M_neural[:, wi, :] /= cnts[wi]

    rows = []
    for b in range(n_bins):
        Mn = M_neural[b]                               # (n_words, n_ch)
        rdm_n = _rdm_cosine(Mn.astype(np.float64))

        if np.any(np.isnan(rdm_n)) or np.std(rdm_n) < 1e-12:
            rows.append({'bin_index': b})
            continue

        def _r(rdm_emb):
            if rdm_emb is None or np.any(np.isnan(rdm_emb)):
                return np.nan
            return float(spearmanr(rdm_n, rdm_emb).statistic)

        r_lex    = _r(rdm_lex)
        r_vis    = _r(rdm_vis)
        r_shared = _r(rdm_shared)

        # Permutation null: shuffle word order in neural matrix
        null_lex    = np.full(n_shuffle, np.nan)
        null_vis    = np.full(n_shuffle, np.nan)
        null_shared = np.full(n_shuffle, np.nan)
        for si in range(n_shuffle):
            perm  = rng.permutation(n_words)
            rdm_p = _rdm_cosine(Mn[perm].astype(np.float64))
            null_lex[si]    = _r(rdm_lex)    if rdm_lex    is not None else np.nan
            null_vis[si]    = _r(rdm_vis)    if rdm_vis    is not None else np.nan
            null_shared[si] = _r(rdm_shared) if rdm_shared is not None else np.nan
            # Overwrite with permuted rdm
            null_lex[si]    = float(spearmanr(rdm_p, rdm_lex).statistic)    if rdm_lex    is not None else np.nan
            null_vis[si]    = float(spearmanr(rdm_p, rdm_vis).statistic)    if rdm_vis    is not None else np.nan
            null_shared[si] = float(spearmanr(rdm_p, rdm_shared).statistic) if rdm_shared is not None else np.nan

        def _z(r, null):
            mu = float(np.nanmean(null))
            sd = float(np.nanstd(null))
            return (r - mu) / max(sd, 1e-10) if not np.isnan(r) else np.nan

        rows.append(dict(
            bin_index=b,
            r_lex_unique=r_lex,
            r_vis_unique=r_vis,
            r_shared=r_shared,
            z_lex_unique=_z(r_lex, null_lex),
            z_vis_unique=_z(r_vis, null_vis),
            z_shared=_z(r_shared, null_shared),
            null_mean_lex_unique=float(np.nanmean(null_lex)),
            null_mean_vis_unique=float(np.nanmean(null_vis)),
            null_mean_shared=float(np.nanmean(null_shared)),
            null_std_lex_unique=float(np.nanstd(null_lex)),
            null_std_vis_unique=float(np.nanstd(null_vis)),
            null_std_shared=float(np.nanstd(null_shared)),
        ))

        if b % 5 == 0:
            print(f"  RSA bin {b}/{n_bins}  "
                  f"r_lex={r_lex:.3f}  r_vis={r_vis:.3f}  r_shared={r_shared:.3f}",
                  flush=True)

    df = pd.DataFrame(rows)
    df['time_ms']    = (df['bin_index'] - N_BINS_HISTORY) * BIN_SIZE_MS
    df['k_lex_unique'] = k_lex
    df['k_vis_unique'] = k_vis
    df['k_shared']     = k_shared
    df['d_common']     = d_common
    df['pca_dims']     = pca_dims
    return df


# ── Regression ────────────────────────────────────────────────────────────────

def compute_regression_epoch(
    data_bins: list,          # list[n_bins] of (n_trials, n_features)
    labels: np.ndarray,       # (n_trials,) — aligned to valid_mask-filtered trials
    categories: np.ndarray,
    dyso_info: dict,
    n_components: int,
    rng: np.random.Generator,
) -> list[dict]:
    """
    One cross-validated epoch: train Kernel PLS → retrieve in each subspace.

    Returns list of per-bin result dicts.
    """
    unique_words = dyso_info['unique_words']
    U_lex        = dyso_info['U_lex']
    U_vis        = dyso_info['U_vis']
    U_shared     = dyso_info['U_shared']
    X_lex        = dyso_info['X_lex_trial']   # (n_valid_trials, d)
    X_vis        = dyso_info['X_vis_trial']

    # Subspace target vectors per trial
    Y_lex    = X_lex @ U_lex    if U_lex.shape[1]    > 0 else None
    Y_vis    = X_vis @ U_vis    if U_vis.shape[1]    > 0 else None
    Y_shared = ((X_lex @ U_shared + X_vis @ U_shared) / 2.0
                if U_shared.shape[1] > 0 else None)

    # Train/test split (word-stratified)
    train_idx, test_idx = _word_stratified_split(
        labels, unique_words, split=0.3, rng=rng
    )
    labels_train = labels[train_idx]
    labels_test  = labels[test_idx]
    cats_train   = categories[train_idx]
    cats_test    = categories[test_idx]

    # Build retrieval databases from training trials
    def _make_db(Y):
        if Y is None or len(train_idx) == 0:
            return None
        return build_retrieval_db(Y[train_idx], labels_train, cats_train)

    db_lex    = _make_db(Y_lex)
    db_vis    = _make_db(Y_vis)
    db_shared = _make_db(Y_shared)

    subspaces = [
        ('lex_unique', Y_lex,    db_lex),
        ('vis_unique', Y_vis,    db_vis),
        ('shared',     Y_shared, db_shared),
    ]

    results = []
    for bin_i, X_bin in enumerate(data_bins):
        X_train = X_bin[train_idx]
        X_test  = X_bin[test_idx]
        row = {'bin_index': bin_i}

        for name, Y_target, db in subspaces:
            if Y_target is None or db is None:
                row[f'word_acc_{name}'] = np.nan
                row[f'cat_acc_{name}']  = np.nan
                row[f'cosine_{name}']   = np.nan
                continue

            db_embeds, uw, w2ci, uc, w2i = db
            try:
                model = _make_pipeline(n_components)
                model.fit(X_train, Y_target[train_idx])
                Y_pred = model.predict(X_test)
                metrics = compute_retrieval_metrics(
                    Y_pred, labels_test, cats_test,
                    db_embeds, uw, w2ci, uc, w2i
                )
                row[f'word_acc_{name}'] = metrics['word_bal_acc']
                row[f'cat_acc_{name}']  = metrics['cat_indep_bal_acc']
                row[f'cosine_{name}']   = metrics['cosine_mean']
            except Exception as exc:
                print(f"    [WARN] Bin {bin_i} subspace {name} failed: {exc}")
                row[f'word_acc_{name}'] = np.nan
                row[f'cat_acc_{name}']  = np.nan
                row[f'cosine_{name}']   = np.nan

        results.append(row)

    return results


# ── Per-patient runner ────────────────────────────────────────────────────────

def process_patient(patient: str, shared_models: dict,
                    args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Process one patient. Returns (rsa_df, reg_df) or None on failure."""
    header(f"PATIENT: {patient}")

    # Load neural data
    try:
        pdata = load_patient_data(patient)
    except FileNotFoundError as exc:
        print(f"  [SKIP] {exc}")
        return None

    # (n_trials, n_channels, n_bins) → (n_trials, n_bins, n_channels)
    data       = pdata['clean_data_binned'].swapaxes(1, 2)
    labels     = np.asarray(pdata['clean_target_labels'])
    categories = np.asarray(pdata['clean_word_category'])
    n_trials, n_bins, n_channels = data.shape
    step(f"data shape: {data.shape}  labels: {len(np.unique(labels))} unique words")

    # Build embeddings
    try:
        embed_dict = build_patient_embeddings(pdata, shared_models)
    except Exception as exc:
        print(f"  [SKIP] build_patient_embeddings failed: {exc}")
        return None

    # DySO decomposition
    dyso_info = build_dyso_bases(embed_dict, labels, args.pca_dims)
    if dyso_info is None:
        return None

    k_lex    = dyso_info['k_lex']
    k_vis    = dyso_info['k_vis']
    k_shared = dyso_info['k_shared']

    if k_lex == 0 and k_vis == 0:
        print("  [WARN] DySO found no unique subspaces for either modality — skipping")
        return None

    # Lagged feature matrices (one per bin, aligned to ALL trials)
    data_bins = reformat(data, N_BINS_HISTORY)

    # Subset data_bins to valid trials for regression
    valid_mask = dyso_info['valid_mask']
    data_bins_valid = [X[valid_mask] for X in data_bins]
    labels_valid    = labels[valid_mask]
    cats_valid      = categories[valid_mask]

    rng = np.random.default_rng(SEED)

    # ── RSA ─────────────────────────────────────────────────────────────────────
    header("RSA timecourse")
    rsa_df = compute_rsa_timecourse(data, labels, dyso_info,
                                    args.n_shuffle, rng)
    rsa_df.insert(0, 'patient', patient)

    # ── Regression ──────────────────────────────────────────────────────────────
    header("Regression timecourse")
    reg_rows = []
    for epoch in range(args.n_epochs):
        t0 = time.time()
        epoch_rng = np.random.default_rng(SEED + epoch + 1)
        bin_results = compute_regression_epoch(
            data_bins_valid, labels_valid, cats_valid,
            dyso_info, args.pls_components, epoch_rng
        )
        for b, row in enumerate(bin_results):
            row.update({
                'patient': patient,
                'epoch': epoch,
                'time_ms': (b - N_BINS_HISTORY) * BIN_SIZE_MS,
                'k_lex_unique': k_lex,
                'k_vis_unique': k_vis,
                'k_shared': k_shared,
                'd_common': dyso_info['d_common'],
            })
            reg_rows.append(row)
        elapsed = time.time() - t0
        step(f"Epoch {epoch + 1}/{args.n_epochs} done ({elapsed:.1f}s)")

    reg_df = pd.DataFrame(reg_rows)

    # Save per-patient CSVs
    os.makedirs(RESULTS_SUBDIR, exist_ok=True)
    rsa_path = os.path.join(RESULTS_SUBDIR, f'lexical_visual_dyso_rsa_{patient}.csv')
    reg_path = os.path.join(RESULTS_SUBDIR, f'lexical_visual_dyso_reg_{patient}.csv')
    rsa_df.to_csv(rsa_path, index=False)
    reg_df.to_csv(reg_path, index=False)
    step(f"Saved → {rsa_path}")
    step(f"Saved → {reg_path}")

    gc.collect()
    return rsa_df, reg_df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog='python -m tests.lexical_visual_dyso',
        description='Lexical vs visual subspace alignment test via DySO.',
    )
    parser.add_argument(
        '--patients', nargs='+', default=None,
        help='Patient IDs to process (default: all available)'
    )
    parser.add_argument(
        '--pca_dims', type=int, default=PCA_DIMS_DEFAULT,
        help=f'PCA dimensions per embedding before DySO stacking (default: {PCA_DIMS_DEFAULT})'
    )
    parser.add_argument(
        '--n_epochs', type=int, default=N_EPOCHS_DEFAULT,
        help=f'Number of cross-validation epochs for regression (default: {N_EPOCHS_DEFAULT})'
    )
    parser.add_argument(
        '--pls_components', type=int, default=PLS_COMPONENTS,
        help=f'Kernel PLS components (default: {PLS_COMPONENTS})'
    )
    parser.add_argument(
        '--n_shuffle', type=int, default=N_SHUFFLE,
        help=f'RSA null permutations (default: {N_SHUFFLE})'
    )
    parser.add_argument(
        '--out_dir', default=None,
        help='Override output directory (default: tests/results/)'
    )
    args = parser.parse_args()

    global RESULTS_SUBDIR
    if args.out_dir:
        RESULTS_SUBDIR = args.out_dir
    os.makedirs(RESULTS_SUBDIR, exist_ok=True)

    patients = args.patients or discover_patients()
    print(f"\nPatients: {patients}")
    print(f"pca_dims={args.pca_dims}, n_epochs={args.n_epochs}, "
          f"pls_components={args.pls_components}, n_shuffle={args.n_shuffle}")

    header("Loading shared embedding models")
    shared_models = load_shared_embedding_models()

    all_rsa, all_reg = [], []
    failed = []

    for patient in patients:
        result = process_patient(patient, shared_models, args)
        if result is not None:
            rsa_df, reg_df = result
            all_rsa.append(rsa_df)
            all_reg.append(reg_df)
        else:
            failed.append(patient)

    if all_rsa:
        combined_rsa = pd.concat(all_rsa, ignore_index=True)
        combined_reg = pd.concat(all_reg, ignore_index=True)

        rsa_all_path = os.path.join(RESULTS_SUBDIR, 'lexical_visual_dyso_rsa_all.csv')
        reg_all_path = os.path.join(RESULTS_SUBDIR, 'lexical_visual_dyso_reg_all.csv')
        combined_rsa.to_csv(rsa_all_path, index=False)
        combined_reg.to_csv(reg_all_path, index=False)

        header("Summary")
        print(f"Processed : {len(all_rsa)} patients")
        if failed:
            print(f"Failed    : {failed}")
        print(f"RSA  CSV  : {rsa_all_path}")
        print(f"Reg  CSV  : {reg_all_path}")

        # Quick cross-patient summary
        if 'r_lex_unique' in combined_rsa.columns:
            peak = (combined_rsa
                    .groupby('patient')[['r_lex_unique', 'r_vis_unique', 'r_shared']]
                    .max()
                    .round(3))
            print("\nPeak RSA r per patient:")
            print(peak.to_string())
    else:
        print("\nNo patients processed successfully.")

    print("\nDone.")


if __name__ == '__main__':
    main()
