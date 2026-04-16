"""
tests/semantic_residual_regression.py
======================================
Test 2: Does phoneme decoding survive after removing semantic neural dimensions?

Fits a semantic PLS to find the "semantic subspace" of neural activity, projects
that out, then fits phoneme PLS on the residual. Compares three conditions:

  normal       — standard phoneme regression on unmodified X
  residualized — phoneme regression on X with semantic subspace removed
  sem_only     — phoneme regression on ONLY the semantic subspace of X
                 (sanity check: this should fail for phonemes)

If category_indep_balanced_acc drops to chance under 'residualized' while cosine
similarity and word accuracy survive, you've isolated purely phonological info.

Usage (run from main/):
    python -m tests.semantic_residual_regression
    python -m tests.semantic_residual_regression --patients VB CP --epochs 20
    python -m tests.semantic_residual_regression --sem-components 4 8 12

Output:
    test_results/semantic_residual_{patient}.csv   (per-patient)
    test_results/semantic_residual_all.csv         (combined)

Key columns:
    patient, phon_emb, sem_emb, condition, sem_components,
    best_bin, cosine_mean, word_bal_acc, cat_indep_bal_acc
"""

import os, sys, argparse, warnings, gc, time

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.cross_decomposition import PLSRegression

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._phoneme_semantic_helpers import (
    load_phoneme_embeddings_for_patient, load_semantic_embeddings_for_patient,
    filter_nan_phoneme_trials,
    reformat, build_retrieval_db, compute_retrieval_metrics,
    N_BINS_HISTORY, PHONEME_EMBEDDINGS, SEMANTIC_EMBEDDINGS_TO_USE,
    header, step, get_out_dir,
)
from semantic_regression import load_patient_data, load_shared_embedding_models


# ── Semantic subspace projection ─────────────────────────────────────────

def compute_semantic_projection(X_feat, Y_sem, n_components):
    """Fit PLS on (X, Y_semantic) and return the projection matrix.

    Returns:
        proj: (n_features, n_features) orthogonal projection onto semantic subspace
        W:    (n_features, n_components) semantic directions
    """
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X_feat, Y_sem)
    W = pls.x_rotations_  # (n_features, n_components)
    # Orthogonal projection matrix onto column space of W
    proj = W @ np.linalg.pinv(W)
    return proj, W


def project_out(X_feat, proj):
    """Remove semantic subspace from X."""
    return X_feat - X_feat @ proj


def project_onto(X_feat, proj):
    """Keep only semantic subspace of X."""
    return X_feat @ proj


# ── Single condition evaluation ──────────────────────────────────────────

def evaluate_condition(X_features, Y_phon, labels, cats,
                       db_embeds, unique_words, word_to_cat_idx,
                       unique_cats, word_to_idx,
                       n_epochs, pls_components, split=0.3):
    """Run phoneme regression on the provided features.

    Returns per-epoch metrics at best bin.
    """
    n_bins = len(X_features)
    n_trials = Y_phon.shape[0]

    ep_cosine   = np.zeros((n_epochs, n_bins))
    ep_word_acc = np.zeros((n_epochs, n_bins))
    ep_cat_acc  = np.zeros((n_epochs, n_bins))

    for ep in range(n_epochs):
        # Random train/test split
        idx = np.random.permutation(n_trials)
        n_test = max(int(n_trials * split), 1)
        test_idx  = idx[:n_test]
        train_idx = idx[n_test:]

        for b in range(n_bins):
            X_feat = X_features[b]
            X_train = X_feat[train_idx]
            X_test  = X_feat[test_idx]
            Y_train = Y_phon[train_idx]

            pipe = Pipeline([
                ('nystroem', Nystroem(kernel='rbf')),
                ('pls', PLSRegression(n_components=pls_components, scale=False)),
            ])
            try:
                pipe.fit(X_train, Y_train)
                Y_pred = pipe.predict(X_test)
            except Exception:
                ep_cosine[ep, b] = np.nan
                ep_word_acc[ep, b] = np.nan
                ep_cat_acc[ep, b] = np.nan
                continue

            metrics = compute_retrieval_metrics(
                Y_pred, labels[test_idx], cats[test_idx],
                db_embeds, unique_words, word_to_cat_idx,
                unique_cats, word_to_idx,
            )
            ep_cosine[ep, b]   = metrics['cosine_mean']
            ep_word_acc[ep, b] = metrics['word_bal_acc']
            ep_cat_acc[ep, b]  = metrics['cat_indep_bal_acc']

    return ep_cosine, ep_word_acc, ep_cat_acc


# ── Per-patient runner ───────────────────────────────────────────────────

def run_patient(patient, pdata, phon_embeds, sem_embeds, args):
    out_dir = get_out_dir(args.out_dir)
    pat_csv = os.path.join(out_dir, f'semantic_residual_{patient}.csv')

    X = pdata['clean_data_binned'].swapaxes(1, 2)
    labels = np.asarray(pdata['clean_answer_labels'])
    cats   = np.asarray(pdata['clean_word_category'])
    X_features_raw = reformat(X, N_BINS_HISTORY)
    n_bins = len(X_features_raw)

    records = []

    for phon_name in PHONEME_EMBEDDINGS:
        Y_phon = phon_embeds[phon_name]
        db_embeds, unique_words, w2c, unique_cats, w2i = \
            build_retrieval_db(Y_phon, labels, cats)

        for sem_name, Y_sem in sem_embeds.items():
            for n_sem_comp in args.sem_components:
                step(f"  {phon_name} / {sem_name} / sem_comp={n_sem_comp}")

                for condition in ['normal', 'residualized', 'sem_only']:
                    step(f"    condition = {condition}")

                    if condition == 'normal':
                        X_features = X_features_raw
                    else:
                        # Compute semantic projection at each bin
                        X_features = []
                        for b in range(n_bins):
                            X_b = X_features_raw[b]
                            try:
                                proj, _ = compute_semantic_projection(
                                    X_b, Y_sem, n_sem_comp)
                            except Exception:
                                X_features.append(X_b)
                                continue
                            if condition == 'residualized':
                                X_features.append(project_out(X_b, proj))
                            else:  # sem_only
                                X_features.append(project_onto(X_b, proj))

                    ep_cos, ep_word, ep_cat = evaluate_condition(
                        X_features, Y_phon, labels, cats,
                        db_embeds, unique_words, w2c, unique_cats, w2i,
                        n_epochs=args.epochs,
                        pls_components=args.pls_components,
                    )

                    for b in range(n_bins):
                        records.append({
                            'patient': patient,
                            'phon_emb': phon_name,
                            'sem_emb': sem_name,
                            'condition': condition,
                            'sem_components': n_sem_comp,
                            'bin': b,
                            'cosine_mean': float(np.nanmean(ep_cos[:, b])),
                            'cosine_std':  float(np.nanstd(ep_cos[:, b])),
                            'word_bal_acc': float(np.nanmean(ep_word[:, b])),
                            'word_bal_acc_std': float(np.nanstd(ep_word[:, b])),
                            'cat_indep_bal_acc': float(np.nanmean(ep_cat[:, b])),
                            'cat_indep_bal_acc_std': float(np.nanstd(ep_cat[:, b])),
                        })
                    gc.collect()

    df = pd.DataFrame(records)
    df.to_csv(pat_csv, index=False)
    step(f"  Saved {pat_csv}")
    return df


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Semantic-residualized phoneme regression test")
    parser.add_argument('--patients', nargs='+', default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--pls-components', type=int, default=10)
    parser.add_argument('--sem-components', type=int, nargs='+', default=[8],
                        help='Number of semantic PLS components to project out '
                             '(default: [8])')
    parser.add_argument('--sem-embeddings', nargs='+', default=None,
                        help='Semantic embeddings to use as confounds '
                             '(default: GloVe)')
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()

    if args.sem_embeddings:
        SEMANTIC_EMBEDDINGS_TO_USE[:] = args.sem_embeddings

    header("SEMANTIC-RESIDUALIZED PHONEME REGRESSION TEST")
    print(f"  epochs={args.epochs}  pls_comp={args.pls_components}  "
          f"sem_comp={args.sem_components}")

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
        phon_embeds = load_phoneme_embeddings_for_patient(pdata)
        pdata, phon_embeds = filter_nan_phoneme_trials(pdata, phon_embeds)
        sem_embeds = load_semantic_embeddings_for_patient(
            pdata, shared, SEMANTIC_EMBEDDINGS_TO_USE)
        df = run_patient(patient, pdata, phon_embeds, sem_embeds, args)
        all_dfs.append(df)
        step(f"  {patient} done in {time.time()-t0:.0f}s")
        del pdata, phon_embeds, sem_embeds
        gc.collect()

    out_dir = get_out_dir(args.out_dir)
    combined = pd.concat(all_dfs, ignore_index=True)
    combined_csv = os.path.join(out_dir, 'semantic_residual_all.csv')
    combined.to_csv(combined_csv, index=False)

    header("SUMMARY")
    # Pick best bin (by word_bal_acc) per group before summarising
    best_bins = (combined.groupby(['patient', 'phon_emb', 'sem_emb', 'condition', 'sem_components'])
                 .apply(lambda g: g.loc[g['word_bal_acc'].idxmax()])
                 .reset_index(drop=True))
    pivot = best_bins.groupby(['patient', 'phon_emb', 'condition']).agg(
        word_acc=('word_bal_acc', 'mean'),
        cat_acc=('cat_indep_bal_acc', 'mean'),
        cosine=('cosine_mean', 'mean'),
    ).reset_index()
    for _, r in pivot.iterrows():
        step(f"  {r['patient']}/{r['phon_emb']}/{r['condition']}: "
             f"word={r['word_acc']:.4f}  cat={r['cat_acc']:.4f}  "
             f"cos={r['cosine']:.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
