# -*- coding: utf-8 -*-
"""
tests/joint_embedding_pls.py
=============================
Test D: Concatenated (joint) embedding target PLS with the issues fixed.

Your proposed method: concatenate phonetic and semantic embeddings into a
single Y = [Y_sem, Y_phon], fit kernel-PLS against the joint target, then
(1) compare retrieval to vanilla, and (2) retro-track how much the model
uses each block.

Three concerns I raised earlier, addressed here:

  Problem 1 — scale/variance bias of the PLS objective
       FIX: per-column z-score on TRAIN, then equalise block Frobenius norms
            so each block contributes identical signal to cov(X, Y).
            Equal dimensionality is necessary but NOT sufficient; we also
            do the Frobenius equalisation.

  Problem 2 — retrieval metric dominated by higher-variance block
       FIX: predictions are split back into blocks, de-normalised with the
            train-time stats, and retrieval is evaluated:
              (a) in each block's ORIGINAL embedding space separately,
              (b) with an ensembled score using α=0.5 (balanced) and α*
                  tuned on an inner validation split (for a fair comparison
                  against test A's ensemble retrieval).

  Problem 3 — Y-loadings ≠ retrieval contribution
       We still compute per-block Y-loading energy (sum Q²) because you
       asked for it, but we ALSO compute a *retrieval-contribution* number:
          rc_sem  = word_acc(sem-only cosine on joint predictions)
          rc_phon = word_acc(phon-only cosine on joint predictions)
       Report BOTH so you can contrast them.  Loadings tell you where
       variance went; rc_X tells you where retrieval success came from.

Comparison to vanilla nearest-neighbour retrieval is included for each
block separately.

Usage (run from main/):
    python -m tests.joint_embedding_pls
    python -m tests.joint_embedding_pls --patients VB \\
        --phon-embs panphon --sem-embs GloVe FastText \\
        --equalize-blocks --tune-alpha

Output:
    test_results/joint_embedding_pls_{patient}.csv
    test_results/joint_embedding_pls_all.csv

Key columns:
    patient, phon_emb, sem_emb, best_bin,
    # retrieval under joint decoder:
    word_acc_joint_sem, word_acc_joint_phon,
    word_acc_joint_balanced, word_acc_joint_alpha_star, alpha_star,
    cat_acc_joint_balanced,
    # retrieval baselines:
    word_acc_vanilla_sem, word_acc_vanilla_phon,
    word_acc_sep_sem, word_acc_sep_phon,     # separate decoders
    # attribution (variance vs retrieval):
    loading_energy_sem, loading_energy_phon,
    rc_sem, rc_phon, rc_ratio_sem
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

# --- cleanup batch 1: imports added by automated migration ---
from tests.helper import make_pipeline


# ── Block normalisation ──────────────────────────────────────────────────

def fit_block_stats(Y_tr, equalize_blocks):
    """Per-column z-score + (optionally) block Frobenius norm scalar."""
    mu = Y_tr.mean(axis=0, keepdims=True)
    sd = Y_tr.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    Y_z = (Y_tr - mu) / sd
    frob = float(np.linalg.norm(Y_z) + 1e-12) if equalize_blocks else 1.0
    return {'mu': mu, 'sd': sd, 'frob': frob}


def apply_block(Y, s):
    return (Y - s['mu']) / s['sd'] / s['frob']


def invert_block(Y_norm, s):
    return Y_norm * s['frob'] * s['sd'] + s['mu']


# ── Retrieval scoring helpers ────────────────────────────────────────────

def _cosine(A, B):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return An @ Bn.T


def word_cat_acc_from_scores(S, labels, word_to_idx, word_to_cat_idx, n_cats):
    pred_word = np.argmax(S, axis=1)
    true_word = np.array([word_to_idx[w] for w in labels])
    w_acc = float(balanced_accuracy_score(true_word, pred_word))
    K = min(5, S.shape[1])
    top = np.argpartition(-S, kth=K - 1, axis=1)[:, :K]
    top_cats = word_to_cat_idx[top]
    pred_cat = np.array([np.bincount(r, minlength=n_cats).argmax() for r in top_cats])
    true_cat = word_to_cat_idx[true_word]
    c_acc = float(balanced_accuracy_score(true_cat, pred_cat))
    return w_acc, c_acc


def vanilla_nn_retrieval(Y_test, db, labels, word_to_idx, word_to_cat_idx, n_cats):
    """Upper-bound baseline: retrieve TRUE embeddings against DB."""
    S = _cosine(Y_test, db)
    return word_cat_acc_from_scores(S, labels, word_to_idx, word_to_cat_idx, n_cats)


# ── PLS pipeline ─────────────────────────────────────────────────────────


# ── Core run per (patient, phon, sem) ────────────────────────────────────

def run_combo(X_features, Y_phon, Y_sem, labels, cats, args):
    n_bins = len(X_features)
    n_trials = Y_phon.shape[0]
    rng = np.random.default_rng(args.seed)

    db_phon, uw_p, w2c_p, ucats_p, w2i_p = build_retrieval_db(Y_phon, labels, cats)
    db_sem,  uw_s, w2c_s, ucats_s, w2i_s = build_retrieval_db(Y_sem,  labels, cats)
    n_cats = len(ucats_p)

    eps = args.epochs
    # Allocate
    K = ['word_joint_sem', 'word_joint_phon',
         'word_joint_bal', 'word_joint_astar',
         'cat_joint_bal', 'alpha_star',
         'word_vanilla_sem', 'word_vanilla_phon',
         'word_sep_sem', 'word_sep_phon',
         'loading_sem', 'loading_phon',
         'rc_sem', 'rc_phon']
    A = {k: np.full((eps, n_bins), np.nan) for k in K}

    alpha_grid = np.linspace(0.0, 1.0, args.n_alpha)

    for ep in range(eps):
        idx = rng.permutation(n_trials)
        n_test = max(int(n_trials * args.test_split), 1)
        test_idx = idx[:n_test]
        tv_idx = idx[n_test:]
        n_val = max(int(len(tv_idx) * args.val_split), 1) if args.tune_alpha else 0
        val_idx   = tv_idx[:n_val] if args.tune_alpha else np.array([], dtype=int)
        train_idx = tv_idx[n_val:] if args.tune_alpha else tv_idx

        # Block normalisers on TRAIN only
        s_sem  = fit_block_stats(Y_sem[train_idx],  args.equalize_blocks)
        s_phon = fit_block_stats(Y_phon[train_idx], args.equalize_blocks)

        Y_sem_n_tr  = apply_block(Y_sem[train_idx],  s_sem)
        Y_sem_n_va  = apply_block(Y_sem[val_idx],    s_sem)   if args.tune_alpha else None
        Y_sem_n_te  = apply_block(Y_sem[test_idx],   s_sem)
        Y_phon_n_tr = apply_block(Y_phon[train_idx], s_phon)
        Y_phon_n_va = apply_block(Y_phon[val_idx],   s_phon)  if args.tune_alpha else None
        Y_phon_n_te = apply_block(Y_phon[test_idx],  s_phon)

        d_sem  = Y_sem_n_tr.shape[1]
        d_phon = Y_phon_n_tr.shape[1]
        Y_join_n_tr = np.concatenate([Y_sem_n_tr, Y_phon_n_tr], axis=1)

        for b in range(n_bins):
            X = X_features[b]
            X_tr, X_te = X[train_idx], X[test_idx]
            X_va = X[val_idx] if args.tune_alpha else None

            # ── Joint PLS decoder (the proposed method) ──
            try:
                m_join = make_pipeline(args.pls_components).fit(X_tr, Y_join_n_tr)
            except Exception:
                continue

            Y_pred_te_n = m_join.predict(X_te)
            # De-normalise each block back to its original space
            Y_pred_sem_te  = invert_block(Y_pred_te_n[:, :d_sem],  s_sem)
            Y_pred_phon_te = invert_block(Y_pred_te_n[:, d_sem:], s_phon)

            # Per-block cosine against original-space DBs
            S_sem  = _cosine(Y_pred_sem_te,  db_sem)
            S_phon = _cosine(Y_pred_phon_te, db_phon)

            w_js, _ = word_cat_acc_from_scores(
                S_sem, labels[test_idx], w2i_s, w2c_s, n_cats)
            w_jp, _ = word_cat_acc_from_scores(
                S_phon, labels[test_idx], w2i_p, w2c_p, n_cats)

            # α=0.5 balanced ensemble
            S_bal = 0.5 * S_sem + 0.5 * S_phon
            w_jb, c_jb = word_cat_acc_from_scores(
                S_bal, labels[test_idx], w2i_p, w2c_p, n_cats)

            # α* tuning on inner validation (optional)
            if args.tune_alpha:
                Y_pred_va_n = m_join.predict(X_va)
                Y_pred_sem_va  = invert_block(Y_pred_va_n[:, :d_sem],  s_sem)
                Y_pred_phon_va = invert_block(Y_pred_va_n[:, d_sem:], s_phon)
                S_sem_va  = _cosine(Y_pred_sem_va,  db_sem)
                S_phon_va = _cosine(Y_pred_phon_va, db_phon)
                best_a, best_wv = 0.5, -np.inf
                for a in alpha_grid:
                    S_v = a * S_phon_va + (1 - a) * S_sem_va
                    wv, _ = word_cat_acc_from_scores(
                        S_v, labels[val_idx], w2i_p, w2c_p, n_cats)
                    if wv > best_wv:
                        best_wv, best_a = wv, a
                S_star = best_a * S_phon + (1 - best_a) * S_sem
                w_jstar, _ = word_cat_acc_from_scores(
                    S_star, labels[test_idx], w2i_p, w2c_p, n_cats)
                A['alpha_star'][ep, b]    = best_a
                A['word_joint_astar'][ep, b] = w_jstar

            # ── Attribution: loadings vs retrieval-contribution ──
            pls = m_join.named_steps['pls']
            Q = pls.y_loadings_          # (d_sem+d_phon, n_components)
            L_sem  = float((Q[:d_sem]  ** 2).sum())
            L_phon = float((Q[d_sem:] ** 2).sum())
            tot = L_sem + L_phon + 1e-12
            A['loading_sem'][ep, b]  = L_sem / tot
            A['loading_phon'][ep, b] = L_phon / tot

            # Retrieval-contribution: word acc using ONLY that block's cosine
            # (already computed as w_js, w_jp above)
            A['rc_sem'][ep, b]  = w_js
            A['rc_phon'][ep, b] = w_jp

            A['word_joint_sem'][ep, b]  = w_js
            A['word_joint_phon'][ep, b] = w_jp
            A['word_joint_bal'][ep, b]  = w_jb
            A['cat_joint_bal'][ep, b]   = c_jb

            # ── Baseline: separate single-block decoders ──
            try:
                m_s = make_pipeline(args.pls_components).fit(X_tr, Y_sem_n_tr)
                Yp_s = invert_block(m_s.predict(X_te), s_sem)
                ws, _ = word_cat_acc_from_scores(
                    _cosine(Yp_s, db_sem), labels[test_idx],
                    w2i_s, w2c_s, n_cats)
                A['word_sep_sem'][ep, b] = ws
            except Exception:
                pass
            try:
                m_p = make_pipeline(args.pls_components).fit(X_tr, Y_phon_n_tr)
                Yp_p = invert_block(m_p.predict(X_te), s_phon)
                wp, _ = word_cat_acc_from_scores(
                    _cosine(Yp_p, db_phon), labels[test_idx],
                    w2i_p, w2c_p, n_cats)
                A['word_sep_phon'][ep, b] = wp
            except Exception:
                pass

            # ── Vanilla NN baselines (upper bound on retrieval given perfect
            #    decoding): query against the DB with the TRUE embedding
            w_vs, _ = vanilla_nn_retrieval(
                Y_sem[test_idx], db_sem, labels[test_idx],
                w2i_s, w2c_s, n_cats)
            w_vp, _ = vanilla_nn_retrieval(
                Y_phon[test_idx], db_phon, labels[test_idx],
                w2i_p, w2c_p, n_cats)
            A['word_vanilla_sem'][ep, b]  = w_vs
            A['word_vanilla_phon'][ep, b] = w_vp

        gc.collect()
    return A


def run_patient(patient, pdata, phon_embeds, sem_embeds, args):
    out_dir = get_out_dir(args.out_dir)
    pat_csv = os.path.join(out_dir, f'joint_embedding_pls_{patient}.csv')

    X = pdata['clean_data_binned'].swapaxes(1, 2)
    labels = np.asarray(pdata['target_concept'])
    cats   = np.asarray(pdata['clean_word_category'])
    X_features = reformat(X, N_BINS_HISTORY)

    records = []
    for phon_name, Y_phon in phon_embeds.items():
        for sem_name, Y_sem in sem_embeds.items():
            step(f"  {phon_name} × {sem_name}  "
                 f"(d_sem={Y_sem.shape[1]}, d_phon={Y_phon.shape[1]}, "
                 f"equalize={args.equalize_blocks}, tune_α={args.tune_alpha})")
            A = run_combo(X_features, Y_phon, Y_sem, labels, cats, args)

            # Best bin by balanced joint word acc
            mean_bal = np.nanmean(A['word_joint_bal'], axis=0)
            if np.all(np.isnan(mean_bal)):
                step(f"    [skip] all-NaN for {phon_name} × {sem_name}")
                continue
            best_bin = int(np.nanargmax(mean_bal))

            def _m(arr): return float(np.nanmean(arr[:, best_bin]))
            def _s(arr): return float(np.nanstd(arr[:, best_bin]))

            rc_s, rc_p = _m(A['rc_sem']), _m(A['rc_phon'])
            tot_rc = rc_s + rc_p + 1e-12

            records.append({
                'patient': patient,
                'phon_emb': phon_name,
                'sem_emb': sem_name,
                'best_bin': best_bin,
                'equalize_blocks': args.equalize_blocks,
                'tune_alpha':     args.tune_alpha,

                # Joint decoder retrieval
                'word_acc_joint_sem':      _m(A['word_joint_sem']),
                'word_acc_joint_phon':     _m(A['word_joint_phon']),
                'word_acc_joint_balanced': _m(A['word_joint_bal']),
                'word_acc_joint_balanced_std': _s(A['word_joint_bal']),
                'word_acc_joint_alpha_star':   _m(A['word_joint_astar']),
                'alpha_star':                  _m(A['alpha_star']),
                'cat_acc_joint_balanced':  _m(A['cat_joint_bal']),

                # Baselines
                'word_acc_vanilla_sem':  _m(A['word_vanilla_sem']),
                'word_acc_vanilla_phon': _m(A['word_vanilla_phon']),
                'word_acc_sep_sem':      _m(A['word_sep_sem']),
                'word_acc_sep_phon':     _m(A['word_sep_phon']),

                # Attribution
                'loading_energy_sem':  _m(A['loading_sem']),
                'loading_energy_phon': _m(A['loading_phon']),
                'rc_sem':              rc_s,
                'rc_phon':             rc_p,
                'rc_ratio_sem':        rc_s / tot_rc,
            })

    df = pd.DataFrame(records)
    df.to_csv(pat_csv, index=False)
    step(f"  Saved {pat_csv}")
    return df


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Joint concatenated-embedding PLS with block "
                    "Frobenius normalisation and attribution.")
    parser.add_argument('--patients', nargs='+', default=None)
    parser.add_argument('--phon-embs', nargs='+', default=None,
                        help=f'Phoneme embeddings (default: {PHONEME_EMBEDDINGS}).')
    parser.add_argument('--sem-embs', nargs='+', default=None,
                        help='Semantic embeddings (default: GloVe). '
                             'Choices: GloVe, FastText, Word2Vec, ConceptNet.')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--pls-components', type=int, default=10)
    parser.add_argument('--test-split', type=float, default=0.3)
    parser.add_argument('--val-split',  type=float, default=0.5,
                        help='Fraction of trainval used for val (α tuning). '
                             'Only used if --tune-alpha.')
    parser.add_argument('--equalize-blocks', action='store_true', default=True,
                        help='Equalise block Frobenius norms after z-scoring. '
                             'On by default.')
    parser.add_argument('--no-equalize-blocks', dest='equalize_blocks',
                        action='store_false')
    parser.add_argument('--tune-alpha', action='store_true', default=False,
                        help='In addition to α=0.5 balanced retrieval, tune α on '
                             'a held-out val split and report test @ α*.')
    parser.add_argument('--n-alpha', type=int, default=21,
                        help='α grid size for α* tuning (default: 21).')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()

    header("JOINT EMBEDDING PLS  (concatenated Y with Frobenius-equalised blocks)")
    print(f"  epochs={args.epochs}  pls_comp={args.pls_components}  "
          f"equalize_blocks={args.equalize_blocks}  tune_alpha={args.tune_alpha}")

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
    combined_csv = os.path.join(out_dir, 'joint_embedding_pls_all.csv')
    combined.to_csv(combined_csv, index=False)

    header("SUMMARY")
    for _, r in combined.iterrows():
        step(f"  {r['patient']}/{r['phon_emb']}×{r['sem_emb']}: "
             f"word(joint,bal)={r['word_acc_joint_balanced']:.3f}  "
             f"word(vanilla_sem)={r['word_acc_vanilla_sem']:.3f} "
             f"word(vanilla_phon)={r['word_acc_vanilla_phon']:.3f}  "
             f"loadings(sem/phon)={r['loading_energy_sem']:.2f}/"
             f"{r['loading_energy_phon']:.2f}  "
             f"rc(sem/phon)={r['rc_sem']:.3f}/{r['rc_phon']:.3f}")
    print("\nDone!")


if __name__ == '__main__':
    main()
