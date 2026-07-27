# -*- coding: utf-8 -*-
"""
tests/ensemble_retrieval.py
============================
Test A: Ensemble retrieval with a learned mixing weight α.

Fits TWO separate kernel-PLS decoders (one per embedding block) on the same
neural features:

    X  →  Ŷ_phon    (trained to predict phoneme embeddings)
    X  →  Ŷ_sem     (trained to predict semantic embeddings)

At retrieval time we combine per-block cosine similarities with a learned α
in [0, 1]:

    score(word w) = α · cos(Ŷ_phon, DB_phon[w]) + (1 − α) · cos(Ŷ_sem, DB_sem[w])

α is fit on an inner validation split (grid search on word retrieval accuracy).
α directly quantifies how much each block contributes to retrieval success —
this is the scientifically interpretable "attribution" number, unlike PLS
Y-loadings which reflect variance-explained, not retrieval contribution.

We report:
  • α*                       — learned mixing weight (val-best)
  • retrieval at α=0 / α=1   — sem-only / phon-only retrieval
  • retrieval at α*          — ensembled retrieval
  • retrieval vanilla        — nearest-neighbour on raw embedding DB (no model)

Usage (run from main/):
    python -m analysis.ensemble_retrieval
    python -m analysis.ensemble_retrieval --patients VB --phon-embs panphon \\
        --sem-embs GloVe FastText --epochs 10

Output:
    test_results/ensemble_retrieval_{patient}.csv   (per-patient)
    test_results/ensemble_retrieval_all.csv         (combined)

Key columns:
    patient, phon_emb, sem_emb, alpha_star,
    word_acc_alpha0, word_acc_alpha1, word_acc_star,
    cat_acc_alpha0, cat_acc_alpha1, cat_acc_star,
    best_bin
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

from analysis.helpers._phoneme_semantic_helpers import (
    load_phoneme_embeddings_for_patient, load_semantic_embeddings_for_patient,
    reformat, build_retrieval_db, cosine_sim_matrix,
    N_BINS_HISTORY, PHONEME_EMBEDDINGS, SEMANTIC_EMBEDDINGS_TO_USE,
    header, step, get_out_dir,
)
from semantic_regression import load_patient_data, load_shared_embedding_models


# ── Retrieval scoring ────────────────────────────────────────────────────

def ensemble_retrieval_scores(Y_pred_phon, Y_pred_sem,
                              db_phon, db_sem, alpha):
    """Return combined similarity matrix (n_test × n_unique_words).

    Each block's similarity is mean-centred by its own database centroid
    (canonical convention; see utils.retrieval.cosine_sim_matrix).
    """
    S_phon = cosine_sim_matrix(Y_pred_phon, db_phon)
    S_sem  = cosine_sim_matrix(Y_pred_sem,  db_sem)
    return alpha * S_phon + (1.0 - alpha) * S_sem


def retrieval_metrics_from_scores(S, true_labels, unique_words,
                                  word_to_idx, word_to_cat_idx, n_cats):
    """Compute balanced word accuracy and category-independent accuracy from
    a similarity matrix.
    """
    pred_word_idx = np.argmax(S, axis=1)
    true_word_idx = np.array([word_to_idx[w] for w in true_labels])
    word_acc = float(balanced_accuracy_score(true_word_idx, pred_word_idx))

    # Category-indep: centroid in each block's own embedding space doesn't
    # apply here because S is a similarity to individual words.  Instead we
    # derive category prediction by majority vote over the top-K nearest words
    # (K=5 by default), which is the natural generalization when scores are
    # already aggregated across blocks.
    top_k = min(5, S.shape[1])
    top_idx = np.argpartition(-S, kth=top_k - 1, axis=1)[:, :top_k]
    top_cats = word_to_cat_idx[top_idx]        # (n_test, K)
    # Majority vote
    pred_cat_idx = np.array([
        np.bincount(row, minlength=n_cats).argmax() for row in top_cats
    ])
    true_cats_by_word = word_to_cat_idx[true_word_idx]
    cat_acc = float(balanced_accuracy_score(true_cats_by_word, pred_cat_idx))

    return word_acc, cat_acc


# ── Core fit per (patient, phon_emb, sem_emb) ────────────────────────────

def fit_block_decoder(X_train, Y_train, n_components):
    """Fit one kernel-PLS decoder; returns fitted pipeline (or None on error)."""
    pipe = Pipeline([
        ('nystroem', Nystroem(kernel='rbf')),
        ('pls', PLSRegression(n_components=n_components, scale=False)),
    ])
    try:
        pipe.fit(X_train, Y_train)
        return pipe
    except Exception as e:
        return None


def run_combo(X_features, Y_phon, Y_sem, labels, cats,
              n_epochs, n_components, alpha_grid,
              rng_seed=0, split=0.3, inner_val=0.5):
    """Return dict of per-bin mean metrics across epochs for one (phon, sem) combo.

    Splits: 70% trainval / 30% test.
    trainval is further split into train / val (50/50 of trainval) for α fitting.
    """
    rng = np.random.default_rng(rng_seed)
    n_bins = len(X_features)
    n_trials = Y_phon.shape[0]

    # Retrieval databases (over all trials, training words only will be valid
    # at test-time — held-out words still get a row but may differ).
    db_phon, u_words, w2c, u_cats, w2i = build_retrieval_db(Y_phon, labels, cats)
    db_sem, _,     _,   _,      _   = build_retrieval_db(Y_sem,  labels, cats)
    n_cats = len(u_cats)

    ep_alpha_star   = np.full((n_epochs, n_bins), np.nan)
    ep_word_a0      = np.full((n_epochs, n_bins), np.nan)
    ep_word_a1      = np.full((n_epochs, n_bins), np.nan)
    ep_word_astar   = np.full((n_epochs, n_bins), np.nan)
    ep_cat_a0       = np.full((n_epochs, n_bins), np.nan)
    ep_cat_a1       = np.full((n_epochs, n_bins), np.nan)
    ep_cat_astar    = np.full((n_epochs, n_bins), np.nan)

    for ep in range(n_epochs):
        idx = rng.permutation(n_trials)
        n_test = max(int(n_trials * split), 1)
        test_idx = idx[:n_test]
        tv_idx = idx[n_test:]
        n_val = max(int(len(tv_idx) * inner_val), 1)
        val_idx = tv_idx[:n_val]
        train_idx = tv_idx[n_val:]

        for b in range(n_bins):
            X = X_features[b]

            pipe_phon = fit_block_decoder(X[train_idx], Y_phon[train_idx],
                                          n_components)
            pipe_sem  = fit_block_decoder(X[train_idx], Y_sem[train_idx],
                                          n_components)
            if pipe_phon is None or pipe_sem is None:
                continue

            # Inner val: pick α
            Y_pred_phon_val = pipe_phon.predict(X[val_idx])
            Y_pred_sem_val  = pipe_sem.predict(X[val_idx])
            best_alpha, best_val_acc = 1.0, -np.inf
            for a in alpha_grid:
                S = ensemble_retrieval_scores(Y_pred_phon_val,
                                              Y_pred_sem_val,
                                              db_phon, db_sem, a)
                w_acc, _ = retrieval_metrics_from_scores(
                    S, labels[val_idx], u_words, w2i, w2c, n_cats)
                if w_acc > best_val_acc:
                    best_val_acc, best_alpha = w_acc, a

            # Test: eval at α=0, α=1, α*
            Y_pred_phon_test = pipe_phon.predict(X[test_idx])
            Y_pred_sem_test  = pipe_sem.predict(X[test_idx])

            for a, key in [(0.0, 'a0'), (1.0, 'a1'), (best_alpha, 'astar')]:
                S_te = ensemble_retrieval_scores(Y_pred_phon_test,
                                                 Y_pred_sem_test,
                                                 db_phon, db_sem, a)
                w_acc, c_acc = retrieval_metrics_from_scores(
                    S_te, labels[test_idx], u_words, w2i, w2c, n_cats)
                if key == 'a0':
                    ep_word_a0[ep, b] = w_acc
                    ep_cat_a0[ep, b]  = c_acc
                elif key == 'a1':
                    ep_word_a1[ep, b] = w_acc
                    ep_cat_a1[ep, b]  = c_acc
                else:
                    ep_word_astar[ep, b] = w_acc
                    ep_cat_astar[ep, b]  = c_acc
                    ep_alpha_star[ep, b] = best_alpha

        gc.collect()

    return {
        'alpha_star':  ep_alpha_star,
        'word_a0':     ep_word_a0,
        'word_a1':     ep_word_a1,
        'word_astar':  ep_word_astar,
        'cat_a0':      ep_cat_a0,
        'cat_a1':      ep_cat_a1,
        'cat_astar':   ep_cat_astar,
    }


# ── Per-patient runner ───────────────────────────────────────────────────

def run_patient(patient, pdata, phon_embeds, sem_embeds, args):
    out_dir = get_out_dir(args.out_dir)
    pat_csv = os.path.join(out_dir, f'ensemble_retrieval_{patient}.csv')

    X = pdata['clean_data_binned'].swapaxes(1, 2)
    labels = np.asarray(pdata['target_concept'])
    cats   = np.asarray(pdata['clean_word_category'])
    X_features = reformat(X, N_BINS_HISTORY)

    alpha_grid = np.linspace(0.0, 1.0, args.n_alpha)
    records = []

    for phon_name, Y_phon in phon_embeds.items():
        for sem_name, Y_sem in sem_embeds.items():
            step(f"  {phon_name} × {sem_name}")
            res = run_combo(X_features, Y_phon, Y_sem, labels, cats,
                            n_epochs=args.epochs,
                            n_components=args.pls_components,
                            alpha_grid=alpha_grid,
                            rng_seed=args.seed)

            # best bin by word_astar (ensembled accuracy)
            mean_word_star = np.nanmean(res['word_astar'], axis=0)
            if np.all(np.isnan(mean_word_star)):
                step(f"    [skip] all-NaN for {phon_name} × {sem_name}")
                continue
            best_bin = int(np.nanargmax(mean_word_star))

            def _stat(arr, axis=0):
                return (float(np.nanmean(arr[:, best_bin])),
                        float(np.nanstd(arr[:, best_bin])))

            a_mean, a_std = _stat(res['alpha_star'])
            w0_m,  w0_s  = _stat(res['word_a0'])
            w1_m,  w1_s  = _stat(res['word_a1'])
            ws_m,  ws_s  = _stat(res['word_astar'])
            c0_m,  c0_s  = _stat(res['cat_a0'])
            c1_m,  c1_s  = _stat(res['cat_a1'])
            cs_m,  cs_s  = _stat(res['cat_astar'])

            records.append({
                'patient': patient,
                'phon_emb': phon_name,
                'sem_emb': sem_name,
                'best_bin': best_bin,
                'alpha_star_mean': a_mean,
                'alpha_star_std':  a_std,
                'word_acc_alpha0_mean': w0_m,  # sem-only retrieval
                'word_acc_alpha0_std':  w0_s,
                'word_acc_alpha1_mean': w1_m,  # phon-only retrieval
                'word_acc_alpha1_std':  w1_s,
                'word_acc_star_mean':   ws_m,  # ensemble
                'word_acc_star_std':    ws_s,
                'cat_acc_alpha0_mean':  c0_m,
                'cat_acc_alpha0_std':   c0_s,
                'cat_acc_alpha1_mean':  c1_m,
                'cat_acc_alpha1_std':   c1_s,
                'cat_acc_star_mean':    cs_m,
                'cat_acc_star_std':     cs_s,
            })

    df = pd.DataFrame(records)
    df.to_csv(pat_csv, index=False)
    step(f"  Saved {pat_csv}")
    return df


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ensemble retrieval with learned α mixing weight.")
    parser.add_argument('--patients', nargs='+', default=None)
    parser.add_argument('--phon-embs', nargs='+', default=None,
                        help=f'Phoneme embeddings to test '
                             f'(default: {PHONEME_EMBEDDINGS}).')
    parser.add_argument('--sem-embs', nargs='+', default=None,
                        help='Semantic embeddings to test '
                             '(default: GloVe).  Choices: GloVe, FastText, '
                             'Word2Vec, ConceptNet.')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--pls-components', type=int, default=10)
    parser.add_argument('--n-alpha', type=int, default=21,
                        help='Grid size for α ∈ [0, 1] (default: 21 → step 0.05).')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()

    header("ENSEMBLE RETRIEVAL  (learned α mixing weight)")
    print(f"  epochs={args.epochs}  pls_comp={args.pls_components}  "
          f"α grid size={args.n_alpha}")

    phon_list = args.phon_embs or PHONEME_EMBEDDINGS
    sem_list  = args.sem_embs  or SEMANTIC_EMBEDDINGS_TO_USE
    print(f"  phoneme embeddings: {phon_list}")
    print(f"  semantic embeddings: {sem_list}")

    from analysis.helpers._phoneme_semantic_helpers import discover_patients
    patients = args.patients or discover_patients()
    print(f"  Patients: {patients}")

    step("Loading shared semantic embedding models (one-time cost)...")
    shared = load_shared_embedding_models()

    all_dfs = []
    for patient in patients:
        header(f"Patient: {patient}")
        t0 = time.time()
        pdata = load_patient_data(patient)
        phon_embeds_all = load_phoneme_embeddings_for_patient(pdata)
        sem_embeds_all  = load_semantic_embeddings_for_patient(
            pdata, shared, sem_list)

        phon_embeds = {k: v for k, v in phon_embeds_all.items() if k in phon_list}
        sem_embeds  = {k: v for k, v in sem_embeds_all.items()  if k in sem_list}

        df = run_patient(patient, pdata, phon_embeds, sem_embeds, args)
        all_dfs.append(df)
        step(f"  {patient} done in {time.time()-t0:.0f}s")
        del pdata, phon_embeds, sem_embeds, phon_embeds_all, sem_embeds_all
        gc.collect()

    out_dir = get_out_dir(args.out_dir)
    combined = pd.concat(all_dfs, ignore_index=True)
    combined_csv = os.path.join(out_dir, 'ensemble_retrieval_all.csv')
    combined.to_csv(combined_csv, index=False)

    header("SUMMARY")
    for _, r in combined.iterrows():
        step(f"  {r['patient']}/{r['phon_emb']}×{r['sem_emb']}: "
             f"α*={r['alpha_star_mean']:.2f}±{r['alpha_star_std']:.2f}  "
             f"word(sem-only)={r['word_acc_alpha0_mean']:.3f}  "
             f"word(phon-only)={r['word_acc_alpha1_mean']:.3f}  "
             f"word(ens)={r['word_acc_star_mean']:.3f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
