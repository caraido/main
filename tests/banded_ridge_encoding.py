"""
tests/banded_ridge_encoding.py
===============================
Test B: Banded ridge encoding — predict neural activity from sem+phon embeddings.

Encoding direction (the Gallant-lab / Huth paradigm):

    Y_neural_bin = X_sem @ W_sem + X_phon @ W_phon + ε

Two predictor blocks (semantic embeddings, phoneme embeddings) jointly predict
neural features at each time bin.  Each block gets its OWN regularisation
weight, tuned independently via nested cross-validation on a 2-D (α_sem, α_phon)
grid.  This is the scientifically rigorous answer to the attribution question:

    "Given an optimal per-block ridge, how much neural variance does each
     block explain uniquely vs redundantly?"

Output is per-block variance partition:
    R²_sem_only, R²_phon_only, R²_joint      (raw R² per condition)
    unique_sem  = R²_joint - R²_phon_only    (marginal contribution of sem)
    unique_phon = R²_joint - R²_sem_only     (marginal contribution of phon)
    shared      = R²_sem_only + R²_phon_only - R²_joint

Banded ridge closed form:
    W = (X.T X + Λ)⁻¹ X.T Y
    where Λ = block-diag([α_sem · I_{d_sem}, α_phon · I_{d_phon}])

No visual / CLIP features.  --phon-embs and --sem-embs are CLI flags.

Usage (run from main/):
    python -m tests.banded_ridge_encoding
    python -m tests.banded_ridge_encoding --patients VB WBH \\
        --phon-embs panphon --sem-embs GloVe FastText
    python -m tests.banded_ridge_encoding --alpha-grid 0.01 0.1 1 10 100 1000

Output:
    test_results/banded_ridge_{patient}.csv
    test_results/banded_ridge_all.csv

Key columns:
    patient, phon_emb, sem_emb, bin, alpha_sem_star, alpha_phon_star,
    r2_sem_only, r2_phon_only, r2_joint, unique_sem, unique_phon, shared
"""

import os, sys, argparse, warnings, gc, time

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._phoneme_semantic_helpers import (
    load_phoneme_embeddings_for_patient, load_semantic_embeddings_for_patient,
    reformat, N_BINS_HISTORY, PHONEME_EMBEDDINGS, SEMANTIC_EMBEDDINGS_TO_USE,
    header, step, get_out_dir,
)
from semantic_regression import load_patient_data, load_shared_embedding_models


# ── Banded ridge closed-form solver ──────────────────────────────────────

def _zscore_fit(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def _zscore_apply(X, mu, sd):
    return (X - mu) / sd


def fit_banded_ridge(X_sem_tr, X_phon_tr, Y_tr, alpha_sem, alpha_phon):
    """Closed-form banded ridge.

    Returns weights split into per-block matrices.
    Solves:
        W = (X.T X + Λ)⁻¹ X.T Y
    where Λ has α_sem on first d_sem diag entries, α_phon on the rest.
    """
    d_sem = X_sem_tr.shape[1]
    d_phon = X_phon_tr.shape[1]
    X_tr = np.concatenate([X_sem_tr, X_phon_tr], axis=1)       # (n, d_sem+d_phon)

    lam = np.concatenate([np.full(d_sem,  alpha_sem),
                          np.full(d_phon, alpha_phon)])
    XtX = X_tr.T @ X_tr
    XtX[np.diag_indices_from(XtX)] += lam
    XtY = X_tr.T @ Y_tr
    W = np.linalg.solve(XtX, XtY)                              # (d_sem+d_phon, n_neural)
    return W[:d_sem], W[d_sem:]


def predict_banded(X_sem, X_phon, W_sem, W_phon):
    return X_sem @ W_sem + X_phon @ W_phon


def fit_ridge_single(X_tr, Y_tr, alpha):
    """Plain ridge for a single block (sem-only or phon-only)."""
    d = X_tr.shape[1]
    XtX = X_tr.T @ X_tr + alpha * np.eye(d)
    XtY = X_tr.T @ Y_tr
    return np.linalg.solve(XtX, XtY)


def r2_score_multioutput(Y_true, Y_pred):
    """Mean R² across output dimensions; matches sklearn's 'uniform_average'."""
    ss_res = ((Y_true - Y_pred) ** 2).sum(axis=0)
    ss_tot = ((Y_true - Y_true.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    ss_tot[ss_tot < 1e-12] = 1.0
    r2 = 1.0 - ss_res / ss_tot
    return float(r2.mean())


# ── Per-bin fit with α tuning via inner split ────────────────────────────

def fit_bin(X_sem_tr, X_phon_tr, Y_tr,
            X_sem_val, X_phon_val, Y_val,
            alpha_grid_sem, alpha_grid_phon):
    """Return (best α_sem, best α_phon, W_sem, W_phon, Y_pred_val)."""
    best = {'r2': -np.inf, 'a_sem': None, 'a_phon': None, 'W': None}
    for a_sem in alpha_grid_sem:
        for a_phon in alpha_grid_phon:
            W_sem, W_phon = fit_banded_ridge(X_sem_tr, X_phon_tr, Y_tr,
                                             a_sem, a_phon)
            Y_pred = predict_banded(X_sem_val, X_phon_val, W_sem, W_phon)
            r2 = r2_score_multioutput(Y_val, Y_pred)
            if r2 > best['r2']:
                best.update({'r2': r2, 'a_sem': a_sem, 'a_phon': a_phon,
                             'W': (W_sem, W_phon)})
    return best


def fit_single_block(X_tr, Y_tr, X_val, Y_val, alpha_grid):
    """Tune ridge α on validation for a single-block predictor."""
    best = {'r2': -np.inf, 'alpha': None, 'W': None}
    for a in alpha_grid:
        W = fit_ridge_single(X_tr, Y_tr, a)
        Y_pred = X_val @ W
        r2 = r2_score_multioutput(Y_val, Y_pred)
        if r2 > best['r2']:
            best.update({'r2': r2, 'alpha': a, 'W': W})
    return best


# ── Core per-(patient, phon, sem) runner ─────────────────────────────────

def run_combo(X_features, Y_sem, Y_phon, args):
    """Return a list of per-bin records for one (phon_emb, sem_emb) combo.

    X_features is a list of (n_trials, n_neural_features) matrices (one per bin).
    Y_sem and Y_phon are predictor blocks aligned to trials.
    """
    n_bins = len(X_features)

    # Some embedding backends can emit NaN rows for a handful of trials.
    # Drop those rows up front so phon-only and joint fits are well-defined.
    valid = np.isfinite(Y_sem).all(axis=1) & np.isfinite(Y_phon).all(axis=1)
    if not np.all(valid):
        Y_sem = Y_sem[valid]
        Y_phon = Y_phon[valid]
        X_features = [xb[valid] for xb in X_features]

    n_trials = Y_sem.shape[0]
    if n_trials < 3:
        return []

    rng = np.random.default_rng(args.seed)

    # Global split: 70/30 trainval/test; then 70/30 train/val inside trainval
    idx = rng.permutation(n_trials)
    n_test = max(int(n_trials * args.test_split), 1)
    test_idx = idx[:n_test]
    tv_idx   = idx[n_test:]
    n_val = max(int(len(tv_idx) * args.val_split), 1)
    val_idx   = tv_idx[:n_val]
    train_idx = tv_idx[n_val:]

    # Z-score predictors on train, apply to val/test
    mu_sem, sd_sem = _zscore_fit(Y_sem[train_idx])
    mu_phon, sd_phon = _zscore_fit(Y_phon[train_idx])
    Xs_tr = _zscore_apply(Y_sem[train_idx], mu_sem, sd_sem)
    Xs_va = _zscore_apply(Y_sem[val_idx],   mu_sem, sd_sem)
    Xs_te = _zscore_apply(Y_sem[test_idx],  mu_sem, sd_sem)
    Xp_tr = _zscore_apply(Y_phon[train_idx], mu_phon, sd_phon)
    Xp_va = _zscore_apply(Y_phon[val_idx],   mu_phon, sd_phon)
    Xp_te = _zscore_apply(Y_phon[test_idx],  mu_phon, sd_phon)

    records = []
    a_grid_sem  = np.asarray(args.alpha_grid, dtype=np.float64)
    a_grid_phon = np.asarray(args.alpha_grid, dtype=np.float64)

    for b in range(n_bins):
        # Center neural targets on train
        Y_full = X_features[b]                                         # (n, F)
        mu_y = Y_full[train_idx].mean(axis=0, keepdims=True)
        Y_tr  = Y_full[train_idx] - mu_y
        Y_va  = Y_full[val_idx]   - mu_y
        Y_te  = Y_full[test_idx]  - mu_y

        # Joint banded ridge (tune α_sem, α_phon on val)
        best_j = fit_bin(Xs_tr, Xp_tr, Y_tr,
                         Xs_va, Xp_va, Y_va,
                         a_grid_sem, a_grid_phon)
        W_sem_j, W_phon_j = best_j['W']
        Y_pred_j = predict_banded(Xs_te, Xp_te, W_sem_j, W_phon_j)
        r2_joint = r2_score_multioutput(Y_te, Y_pred_j)

        # Per-block partial contributions within joint model (hold the other to 0)
        r2_joint_sem_part  = r2_score_multioutput(Y_te, Xs_te @ W_sem_j)
        r2_joint_phon_part = r2_score_multioutput(Y_te, Xp_te @ W_phon_j)

        # Sem-only ridge (tune α on val)
        best_s = fit_single_block(Xs_tr, Y_tr, Xs_va, Y_va, a_grid_sem)
        r2_sem = r2_score_multioutput(Y_te, Xs_te @ best_s['W'])

        # Phon-only ridge (tune α on val)
        best_p = fit_single_block(Xp_tr, Y_tr, Xp_va, Y_va, a_grid_phon)
        r2_phon = r2_score_multioutput(Y_te, Xp_te @ best_p['W'])

        # Variance partitioning
        unique_sem  = r2_joint - r2_phon
        unique_phon = r2_joint - r2_sem
        shared      = r2_sem + r2_phon - r2_joint

        # Weight norms as auxiliary diagnostic
        sem_wnorm  = float(np.linalg.norm(W_sem_j))
        phon_wnorm = float(np.linalg.norm(W_phon_j))

        records.append({
            'bin': b,
            'alpha_sem_star':       float(best_j['a_sem']),
            'alpha_phon_star':      float(best_j['a_phon']),
            'alpha_sem_solo_star':  float(best_s['alpha']),
            'alpha_phon_solo_star': float(best_p['alpha']),
            'r2_sem_only':     float(r2_sem),
            'r2_phon_only':    float(r2_phon),
            'r2_joint':        float(r2_joint),
            'r2_joint_sem_partial':  float(r2_joint_sem_part),
            'r2_joint_phon_partial': float(r2_joint_phon_part),
            'unique_sem':      float(unique_sem),
            'unique_phon':     float(unique_phon),
            'shared':          float(shared),
            'w_norm_sem':      sem_wnorm,
            'w_norm_phon':     phon_wnorm,
            'n_train':         int(len(train_idx)),
            'n_val':           int(len(val_idx)),
            'n_test':          int(len(test_idx)),
        })
    gc.collect()
    return records


def run_patient(patient, pdata, phon_embeds, sem_embeds, args):
    out_dir = get_out_dir(args.out_dir)
    pat_csv = os.path.join(out_dir, f'banded_ridge_{patient}.csv')

    X = pdata['clean_data_binned'].swapaxes(1, 2)                  # (n, bins, ch)
    X_features = reformat(X, N_BINS_HISTORY)                       # list of (n, F)

    records = []
    for phon_name, Y_phon in phon_embeds.items():
        for sem_name, Y_sem in sem_embeds.items():
            step(f"  {phon_name} × {sem_name}  "
                 f"(d_sem={Y_sem.shape[1]}, d_phon={Y_phon.shape[1]})")
            combo_recs = run_combo(X_features, Y_sem, Y_phon, args)
            for r in combo_recs:
                r.update({'patient': patient,
                          'phon_emb': phon_name,
                          'sem_emb': sem_name})
            records.extend(combo_recs)

    df = pd.DataFrame(records)
    df.to_csv(pat_csv, index=False)
    step(f"  Saved {pat_csv}")
    return df


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Banded ridge encoding: sem+phon blocks, per-block α tuning.")
    parser.add_argument('--patients', nargs='+', default=None)
    parser.add_argument('--phon-embs', nargs='+', default=None,
                        help=f'Phoneme embeddings (default: {PHONEME_EMBEDDINGS}).')
    parser.add_argument('--sem-embs', nargs='+', default=None,
                        help='Semantic embeddings (default: GloVe). '
                             'Choices: GloVe, FastText, Word2Vec, ConceptNet.')
    parser.add_argument('--alpha-grid', nargs='+', type=float,
                        default=[1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],
                        help='Ridge α grid for both blocks (default: log-spaced).')
    parser.add_argument('--test-split', type=float, default=0.3)
    parser.add_argument('--val-split',  type=float, default=0.3,
                        help='Validation fraction of trainval (default: 0.3).')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()

    header("BANDED RIDGE ENCODING  (sem + phon, per-block α)")
    print(f"  alpha_grid={args.alpha_grid}")

    phon_list = args.phon_embs or PHONEME_EMBEDDINGS
    sem_list  = args.sem_embs  or SEMANTIC_EMBEDDINGS_TO_USE
    print(f"  phoneme embeddings: {phon_list}")
    print(f"  semantic embeddings: {sem_list}")

    from tests._phoneme_semantic_helpers import discover_patients
    patients = args.patients or discover_patients()
    print(f"  Patients: {patients}")

    step("Loading shared semantic embedding models (one-time cost)...")
    shared = load_shared_embedding_models()

    all_dfs = []
    for patient in patients:
        header(f"Patient: {patient}")
        t0 = time.time()
        pdata = load_patient_data(patient)
        phon_all = load_phoneme_embeddings_for_patient(pdata)
        sem_all  = load_semantic_embeddings_for_patient(pdata, shared, sem_list)

        phon_embeds = {k: v for k, v in phon_all.items() if k in phon_list}
        sem_embeds  = {k: v for k, v in sem_all.items()  if k in sem_list}

        df = run_patient(patient, pdata, phon_embeds, sem_embeds, args)
        all_dfs.append(df)
        step(f"  {patient} done in {time.time()-t0:.0f}s")
        del pdata, phon_embeds, sem_embeds, phon_all, sem_all
        gc.collect()

    out_dir = get_out_dir(args.out_dir)
    combined = pd.concat(all_dfs, ignore_index=True)
    combined_csv = os.path.join(out_dir, 'banded_ridge_all.csv')
    combined.to_csv(combined_csv, index=False)

    header("SUMMARY  (best-bin per (patient, phon, sem))")
    summary = (combined.sort_values('r2_joint', ascending=False)
                       .groupby(['patient', 'phon_emb', 'sem_emb'])
                       .head(1)
                       .reset_index(drop=True))
    for _, r in summary.iterrows():
        step(f"  {r['patient']}/{r['phon_emb']}×{r['sem_emb']} "
             f"bin={int(r['bin']):02d}  "
             f"r2_joint={r['r2_joint']:.3f}  "
             f"u_sem={r['unique_sem']:+.3f}  u_phon={r['unique_phon']:+.3f}  "
             f"shared={r['shared']:+.3f}")
    print("\nDone!")


if __name__ == '__main__':
    main()
