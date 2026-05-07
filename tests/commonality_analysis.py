# -*- coding: utf-8 -*-
"""
tests/commonality_analysis.py
==============================
Test C: Commonality analysis on retrieval-relevant variance.

Complementary to Test B (banded ridge encoding).  Where banded ridge
partitions *neural variance*, commonality here partitions *retrieval-relevant
variance* in the DECODING direction.

Three decoders are fit with the same kernel-PLS pipeline used elsewhere:

    M_sem:   X_neural → Y_sem            (predicts semantic embedding)
    M_phon:  X_neural → Y_phon           (predicts phoneme embedding)
    M_both:  X_neural → [Y_sem, Y_phon]  (joint, block-Frobenius-normalised)

For each decoder we score mean cosine similarity on held-out test trials.
Commonality partition (Pedhazur, Newton & Spurrell):

    unique_sem  = cos_both − cos_phon    (cosine gain unique to sem)
    unique_phon = cos_both − cos_sem     (cosine gain unique to phon)
    shared      = cos_sem + cos_phon − cos_both

Interpretation:
    • Positive unique_X     → block X carries retrieval-relevant variance
                              not in the other block.
    • Large shared          → the two blocks encode overlapping neural
                              signal (i.e., word-identity confound is
                              doing the work).
    • Negative unique       → suppression / correlated blocks; report as-is.

Also reports retrieval accuracy (word + category) under each decoder at the
best bin.  Directly answers: "How much of phon retrieval success is unique
to phon vs shared with sem?"

Usage (run from main/):
    python -m tests.commonality_analysis
    python -m tests.commonality_analysis --patients VB \\
        --phon-embs panphon token_ipa --sem-embs GloVe FastText --epochs 5

Output:
    test_results/commonality_{patient}.csv
    test_results/commonality_all.csv

Key columns:
    patient, phon_emb, sem_emb, best_bin,
    cos_sem, cos_phon, cos_joint, unique_sem, unique_phon, shared,
    word_acc_sem, word_acc_phon, word_acc_joint,
    cat_acc_sem, cat_acc_phon, cat_acc_joint
"""

import os, sys, argparse, warnings, gc, time

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._phoneme_semantic_helpers import (
    load_phoneme_embeddings_for_patient, load_semantic_embeddings_for_patient,
    reformat, build_retrieval_db, compute_retrieval_metrics,
    N_BINS_HISTORY, PHONEME_EMBEDDINGS, SEMANTIC_EMBEDDINGS_TO_USE,
    header, step, get_out_dir,
)
from semantic_regression import load_patient_data, load_shared_embedding_models


# ── Block normalisation (equal Frobenius, equal per-column variance) ──────

def fit_block_norm(Y_tr):
    """Fit per-column z-score and Frobenius-norm scalar on train data only."""
    mu = Y_tr.mean(axis=0, keepdims=True)
    sd = Y_tr.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    Y_z = (Y_tr - mu) / sd
    frob = np.linalg.norm(Y_z) + 1e-12
    return {'mu': mu, 'sd': sd, 'frob': frob}


def apply_block_norm(Y, stats):
    return (Y - stats['mu']) / stats['sd'] / stats['frob']


def inverse_block_norm(Y, stats):
    """Inverse Z + Frobenius (predictions live in normalised space)."""
    return Y * stats['frob'] * stats['sd'] + stats['mu']


# ── Pipeline ─────────────────────────────────────────────────────────────

def make_pipeline(n_components):
    return Pipeline([
        ('nystroem', Nystroem(kernel='rbf')),
        ('pls', PLSRegression(n_components=n_components, scale=False)),
    ])


# ── Cosine similarity on test set in original embedding space ───────────────

def mean_cosine_sim(Y_true, Y_pred):
    """Mean row-wise cosine similarity between Y_true and Y_pred."""
    An = Y_true / (np.linalg.norm(Y_true, axis=1, keepdims=True) + 1e-10)
    Bn = Y_pred / (np.linalg.norm(Y_pred, axis=1, keepdims=True) + 1e-10)
    return float((An * Bn).sum(axis=1).mean())


# ── Core per-(patient, phon, sem) run ────────────────────────────────────

def run_combo(X_features, Y_phon, Y_sem, labels, cats,
              n_epochs, n_components, rng_seed, split=0.3):
    """Return per-epoch per-bin metric arrays for the three decoders."""
    n_bins = len(X_features)
    n_trials = Y_phon.shape[0]

    # Retrieval databases on ORIGINAL (unnormalised) embeddings
    db_phon, u_words_p, w2c_p, u_cats_p, w2i_p = build_retrieval_db(Y_phon, labels, cats)
    db_sem,  u_words_s, w2c_s, u_cats_s, w2i_s = build_retrieval_db(Y_sem,  labels, cats)

    # Result arrays
    keys_cos = ['cos_sem', 'cos_phon', 'cos_joint']
    keys_w   = ['word_sem', 'word_phon', 'word_joint']
    keys_c   = ['cat_sem',  'cat_phon',  'cat_joint']
    all_keys = keys_cos + keys_w + keys_c
    out = {k: np.full((n_epochs, n_bins), np.nan) for k in all_keys}

    rng = np.random.default_rng(rng_seed)

    for ep in range(n_epochs):
        idx = rng.permutation(n_trials)
        n_test = max(int(n_trials * split), 1)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        # Block normalisers fit on TRAIN only
        ns = fit_block_norm(Y_sem[train_idx])
        np_ = fit_block_norm(Y_phon[train_idx])

        Y_sem_tr_n  = apply_block_norm(Y_sem[train_idx],  ns)
        Y_sem_te_n  = apply_block_norm(Y_sem[test_idx],   ns)
        Y_phon_tr_n = apply_block_norm(Y_phon[train_idx], np_)
        Y_phon_te_n = apply_block_norm(Y_phon[test_idx],  np_)

        d_sem, d_phon = Y_sem_tr_n.shape[1], Y_phon_tr_n.shape[1]
        Y_joint_tr_n = np.concatenate([Y_sem_tr_n, Y_phon_tr_n], axis=1)
        Y_joint_te_n = np.concatenate([Y_sem_te_n, Y_phon_te_n], axis=1)

        for b in range(n_bins):
            X = X_features[b]
            X_tr, X_te = X[train_idx], X[test_idx]

            # --- M_sem
            try:
                m_sem = make_pipeline(n_components).fit(X_tr, Y_sem_tr_n)
                Y_pred_sem_n = m_sem.predict(X_te)
                Y_pred_sem = inverse_block_norm(Y_pred_sem_n, ns)
                out['cos_sem'][ep, b] = mean_cosine_sim(Y_sem[test_idx], Y_pred_sem)
                m = compute_retrieval_metrics(
                    Y_pred_sem, labels[test_idx], cats[test_idx],
                    db_sem, u_words_s, w2c_s, u_cats_s, w2i_s)
                out['word_sem'][ep, b] = m['word_bal_acc']
                out['cat_sem'][ep, b]  = m['cat_indep_bal_acc']
            except Exception:
                pass

            # --- M_phon
            try:
                m_phon = make_pipeline(n_components).fit(X_tr, Y_phon_tr_n)
                Y_pred_phon_n = m_phon.predict(X_te)
                Y_pred_phon = inverse_block_norm(Y_pred_phon_n, np_)
                out['cos_phon'][ep, b] = mean_cosine_sim(Y_phon[test_idx], Y_pred_phon)
                m = compute_retrieval_metrics(
                    Y_pred_phon, labels[test_idx], cats[test_idx],
                    db_phon, u_words_p, w2c_p, u_cats_p, w2i_p)
                out['word_phon'][ep, b] = m['word_bal_acc']
                out['cat_phon'][ep, b]  = m['cat_indep_bal_acc']
            except Exception:
                pass

            # --- M_joint
            try:
                m_joint = make_pipeline(n_components).fit(X_tr, Y_joint_tr_n)
                Y_pred_joint_n = m_joint.predict(X_te)
                # Split, de-normalise, compute cosine for each block then average
                Y_pred_sem_j  = inverse_block_norm(Y_pred_joint_n[:, :d_sem],  ns)
                Y_pred_phon_j = inverse_block_norm(Y_pred_joint_n[:, d_sem:], np_)
                out['cos_joint'][ep, b] = 0.5 * (
                    mean_cosine_sim(Y_sem[test_idx],  Y_pred_sem_j) +
                    mean_cosine_sim(Y_phon[test_idx], Y_pred_phon_j))
                m_s = compute_retrieval_metrics(
                    Y_pred_sem_j, labels[test_idx], cats[test_idx],
                    db_sem, u_words_s, w2c_s, u_cats_s, w2i_s)
                m_p = compute_retrieval_metrics(
                    Y_pred_phon_j, labels[test_idx], cats[test_idx],
                    db_phon, u_words_p, w2c_p, u_cats_p, w2i_p)
                # Combined score: average of block cosines (mirrors ensemble α=0.5)
                out['word_joint'][ep, b] = 0.5 * (m_s['word_bal_acc']
                                                 + m_p['word_bal_acc'])
                out['cat_joint'][ep, b]  = 0.5 * (m_s['cat_indep_bal_acc']
                                                 + m_p['cat_indep_bal_acc'])
            except Exception:
                pass

        gc.collect()
    return out


def run_patient(patient, pdata, phon_embeds, sem_embeds, args):
    out_dir = get_out_dir(args.out_dir)
    pat_csv = os.path.join(out_dir, f'commonality_{patient}.csv')

    X = pdata['clean_data_binned'].swapaxes(1, 2)
    labels = np.asarray(pdata['target_concept'])
    cats   = np.asarray(pdata['clean_word_category'])
    X_features = reformat(X, N_BINS_HISTORY)
    n_bins = len(X_features)

    records = []
    for phon_name, Y_phon in phon_embeds.items():
        for sem_name, Y_sem in sem_embeds.items():
            step(f"  {phon_name} × {sem_name}  "
                 f"(d_sem={Y_sem.shape[1]}, d_phon={Y_phon.shape[1]})")
            res = run_combo(X_features, Y_phon, Y_sem, labels, cats,
                            n_epochs=args.epochs,
                            n_components=args.pls_components,
                            rng_seed=args.seed)

            # Best bin by joint cosine similarity
            mean_cj = np.nanmean(res['cos_joint'], axis=0)
            if np.all(np.isnan(mean_cj)):
                step(f"    [skip] all-NaN")
                continue
            best_bin = int(np.nanargmax(mean_cj))

            def _m(arr): return float(np.nanmean(arr[:, best_bin]))
            def _s(arr): return float(np.nanstd(arr[:, best_bin]))

            cos_s, cos_p, cos_j = _m(res['cos_sem']), _m(res['cos_phon']), _m(res['cos_joint'])
            records.append({
                'patient': patient,
                'phon_emb': phon_name,
                'sem_emb': sem_name,
                'best_bin': best_bin,
                'cos_sem':     cos_s, 'cos_sem_std':   _s(res['cos_sem']),
                'cos_phon':    cos_p, 'cos_phon_std':  _s(res['cos_phon']),
                'cos_joint':   cos_j, 'cos_joint_std': _s(res['cos_joint']),
                'unique_sem':  cos_j - cos_p,
                'unique_phon': cos_j - cos_s,
                'shared':      cos_s + cos_p - cos_j,
                'word_acc_sem':    _m(res['word_sem']),
                'word_acc_phon':   _m(res['word_phon']),
                'word_acc_joint':  _m(res['word_joint']),
                'cat_acc_sem':     _m(res['cat_sem']),
                'cat_acc_phon':    _m(res['cat_phon']),
                'cat_acc_joint':   _m(res['cat_joint']),
            })

    df = pd.DataFrame(records)
    df.to_csv(pat_csv, index=False)
    step(f"  Saved {pat_csv}")
    return df


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Commonality analysis: partition retrieval R² into "
                    "unique_sem, unique_phon, shared.")
    parser.add_argument('--patients', nargs='+', default=None)
    parser.add_argument('--phon-embs', nargs='+', default=None,
                        help=f'Phoneme embeddings (default: {PHONEME_EMBEDDINGS}).')
    parser.add_argument('--sem-embs', nargs='+', default=None,
                        help='Semantic embeddings (default: GloVe). '
                             'Choices: GloVe, FastText, Word2Vec, ConceptNet.')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--pls-components', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()

    header("COMMONALITY ANALYSIS  (unique_sem / unique_phon / shared cos)")
    print(f"  epochs={args.epochs}  pls_comp={args.pls_components}")

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
    combined_csv = os.path.join(out_dir, 'commonality_all.csv')
    combined.to_csv(combined_csv, index=False)

    header("SUMMARY")
    for _, r in combined.iterrows():
        step(f"  {r['patient']}/{r['phon_emb']}×{r['sem_emb']}: "
             f"cos(sem)={r['cos_sem']:.3f} cos(phon)={r['cos_phon']:.3f} "
             f"cos(joint)={r['cos_joint']:.3f} → "
             f"u_sem={r['unique_sem']:+.3f}  u_phon={r['unique_phon']:+.3f}  "
             f"shared={r['shared']:+.3f}")
    print("\nDone!")


if __name__ == '__main__':
    main()
