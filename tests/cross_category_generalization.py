"""
tests/cross_category_generalization.py
=======================================
Test 1: Can phoneme decoding generalize across semantic categories?

Train phoneme regression (Kernel PLS) on trials from a subset of semantic
categories, test on held-out categories the model has *never seen*.  Any
above-chance accuracy on the held-out categories is purely phonological
because the semantic context is entirely novel.

Usage (run from main/):
    python -m tests.cross_category_generalization
    python -m tests.cross_category_generalization --patients VB CP AA --epochs 20
    python -m tests.cross_category_generalization --resume

Output:
    test_results/cross_cat_gen_{patient}.csv       (per-patient, incremental)
    test_results/cross_cat_gen_all.csv             (combined)

Key columns:
    patient, embedding, fold_idx, held_out_cats, n_train_trials, n_test_trials,
    best_bin, cosine_mean, word_bal_acc, cat_indep_bal_acc, word_chance, cat_chance
"""

import os, sys, argparse, warnings, gc, time
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")

# ── Ensure main/ is on the path ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._phoneme_semantic_helpers import (
    load_phoneme_embeddings_for_patient, reformat, make_kernel_pls_pipeline,
    build_retrieval_db, compute_retrieval_metrics,
    N_BINS_HISTORY, PHONEME_EMBEDDINGS, header, step, get_out_dir,
)
from semantic_regression import load_patient_data


# ── Core function ────────────────────────────────────────────────────────

def run_fold(X_features, Y, labels, cats, train_mask, test_mask,
             db_embeds, unique_words, word_to_cat_idx, unique_cats,
             word_to_idx, n_epochs=10, n_components=10):
    """Run multiple epochs on a single category-split fold.

    For each epoch, we do a random sub-split within the training categories
    (not used for evaluation — the model is re-fitted on ALL training data
    for the final retrieval).  The multiple epochs give us variance estimates.

    Returns per-epoch metrics at the best time bin (chosen on test set).
    """
    n_bins = len(X_features)
    n_cats = len(unique_cats)

    # Pre-allocate per-epoch, per-bin arrays
    epoch_cosine   = np.zeros((n_epochs, n_bins))
    epoch_word_acc = np.zeros((n_epochs, n_bins))
    epoch_cat_acc  = np.zeros((n_epochs, n_bins))

    for ep in range(n_epochs):
        for b in range(n_bins):
            X_feat = X_features[b]
            X_train = X_feat[train_mask]
            X_test  = X_feat[test_mask]
            Y_train = Y[train_mask]

            # Fit on training categories
            pipe = make_kernel_pls_pipeline(n_components)
            try:
                pipe.fit(X_train, Y_train)
                Y_pred = pipe.predict(X_test)
            except Exception:
                epoch_cosine[ep, b] = np.nan
                epoch_word_acc[ep, b] = np.nan
                epoch_cat_acc[ep, b] = np.nan
                continue

            metrics = compute_retrieval_metrics(
                Y_pred, labels[test_mask], cats[test_mask],
                db_embeds, unique_words, word_to_cat_idx,
                unique_cats, word_to_idx,
            )
            epoch_cosine[ep, b]   = metrics['cosine_mean']
            epoch_word_acc[ep, b] = metrics['word_bal_acc']
            epoch_cat_acc[ep, b]  = metrics['cat_indep_bal_acc']

        # Only need one epoch for deterministic pipeline (no random split)
        # But multiple epochs help if Nystroem sampling varies
        gc.collect()

    return epoch_cosine, epoch_word_acc, epoch_cat_acc


def run_patient(patient, pdata, phon_embeds, args):
    """Run all leave-K-categories-out folds for one patient."""
    out_dir = get_out_dir(args.out_dir)
    pat_csv = os.path.join(out_dir, f'cross_cat_gen_{patient}.csv')

    X = pdata['clean_data_binned'].swapaxes(1, 2)  # (n_trials, n_bins, n_ch)
    labels = np.asarray(pdata['target_concept'])
    cats   = np.asarray(pdata['clean_word_category'])
    unique_categories = np.unique(cats)

    # Resume support
    done_keys = set()
    if args.resume and os.path.exists(pat_csv):
        existing = pd.read_csv(pat_csv)
        done_keys = set(zip(existing['embedding'], existing['fold_idx']))
        step(f"Resuming: {len(done_keys)} fold×embedding combos already done")
    else:
        existing = None

    X_features = reformat(X, N_BINS_HISTORY)
    n_bins = len(X_features)
    n_leave_out = args.n_leave_out

    folds = list(combinations(unique_categories, n_leave_out))
    records = []

    for emb_name in PHONEME_EMBEDDINGS:
        Y = phon_embeds[emb_name]
        db_embeds, unique_words, word_to_cat_idx, unique_cats, word_to_idx = \
            build_retrieval_db(Y, labels, cats)
        n_cats = len(unique_cats)

        for fold_idx, held_out in enumerate(folds):
            if (emb_name, fold_idx) in done_keys:
                continue

            train_mask = ~np.isin(cats, list(held_out))
            test_mask  =  np.isin(cats, list(held_out))
            n_train = int(train_mask.sum())
            n_test  = int(test_mask.sum())

            if n_test < 5 or n_train < 20:
                step(f"  Skipping fold {fold_idx} ({held_out}): "
                     f"too few trials (train={n_train}, test={n_test})")
                continue

            step(f"  {emb_name} fold {fold_idx}/{len(folds)-1}  "
                 f"held_out={held_out}  train={n_train} test={n_test}")

            ep_cos, ep_word, ep_cat = run_fold(
                X_features, Y, labels, cats, train_mask, test_mask,
                db_embeds, unique_words, word_to_cat_idx, unique_cats,
                word_to_idx, n_epochs=args.epochs, n_components=args.pls_components,
            )

            # Best bin by word accuracy on test (mean over epochs)
            mean_word = np.nanmean(ep_word, axis=0)
            best_bin = int(np.nanargmax(mean_word))

            # Chance levels
            n_unique_test_words = len(np.unique(labels[test_mask]))
            n_held_cats = len(held_out)
            word_chance = 1.0 / max(n_unique_test_words, 1)
            cat_chance  = 1.0 / max(n_cats, 1)

            records.append({
                'patient': patient,
                'embedding': emb_name,
                'fold_idx': fold_idx,
                'held_out_cats': '|'.join(held_out),
                'n_leave_out': n_leave_out,
                'n_train_trials': n_train,
                'n_test_trials': n_test,
                'n_unique_test_words': n_unique_test_words,
                'best_bin': best_bin,
                'cosine_mean': float(np.nanmean(ep_cos[:, best_bin])),
                'cosine_std':  float(np.nanstd(ep_cos[:, best_bin])),
                'word_bal_acc': float(np.nanmean(ep_word[:, best_bin])),
                'word_bal_acc_std': float(np.nanstd(ep_word[:, best_bin])),
                'cat_indep_bal_acc': float(np.nanmean(ep_cat[:, best_bin])),
                'cat_indep_bal_acc_std': float(np.nanstd(ep_cat[:, best_bin])),
                'word_chance': word_chance,
                'cat_chance': cat_chance,
            })
            gc.collect()

    # Save per-patient CSV (append if resuming)
    new_df = pd.DataFrame(records)
    if existing is not None and len(new_df) > 0:
        combined = pd.concat([existing, new_df], ignore_index=True)
    elif existing is not None:
        combined = existing
    else:
        combined = new_df
    combined.to_csv(pat_csv, index=False)
    step(f"  Saved {pat_csv}  ({len(combined)} rows)")
    return combined


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-category generalization test for phoneme regression")
    parser.add_argument('--patients', nargs='+', default=None,
                        help='Patient IDs (default: all discovered)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Epochs per fold (default: 5)')
    parser.add_argument('--pls-components', type=int, default=10,
                        help='PLS n_components (default: 10)')
    parser.add_argument('--n-leave-out', type=int, default=2,
                        help='Number of categories to hold out per fold (default: 2)')
    parser.add_argument('--out-dir', default=None,
                        help='Output directory (default: test_results/)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already-computed fold×embedding combos')
    args = parser.parse_args()

    header("CROSS-CATEGORY GENERALIZATION TEST")
    print(f"  epochs={args.epochs}  pls_components={args.pls_components}  "
          f"n_leave_out={args.n_leave_out}")

    from semantic_regression import load_patient_data
    from tests._phoneme_semantic_helpers import discover_patients

    patients = args.patients or discover_patients()
    print(f"  Patients: {patients}")

    all_dfs = []
    for patient in patients:
        header(f"Patient: {patient}")
        t0 = time.time()
        pdata = load_patient_data(patient)
        phon_embeds = load_phoneme_embeddings_for_patient(pdata)
        df = run_patient(patient, pdata, phon_embeds, args)
        all_dfs.append(df)
        elapsed = time.time() - t0
        step(f"  {patient} done in {elapsed:.0f}s")
        del pdata, phon_embeds
        gc.collect()

    # Combined CSV
    out_dir = get_out_dir(args.out_dir)
    combined = pd.concat(all_dfs, ignore_index=True)
    combined_csv = os.path.join(out_dir, 'cross_cat_gen_all.csv')
    combined.to_csv(combined_csv, index=False)
    header("SUMMARY")
    print(f"  Combined CSV: {combined_csv}")
    print(f"  Total rows: {len(combined)}")

    # Quick summary: mean word accuracy across folds
    summary = combined.groupby(['patient', 'embedding']).agg(
        mean_word_acc=('word_bal_acc', 'mean'),
        mean_cat_acc=('cat_indep_bal_acc', 'mean'),
        mean_cosine=('cosine_mean', 'mean'),
        word_chance=('word_chance', 'mean'),
        cat_chance=('cat_chance', 'mean'),
    ).reset_index()

    for _, row in summary.iterrows():
        above_word = row['mean_word_acc'] > row['word_chance']
        above_cat  = row['mean_cat_acc'] > row['cat_chance']
        step(f"  {row['patient']}/{row['embedding']}: "
             f"word_acc={row['mean_word_acc']:.4f} "
             f"(chance={row['word_chance']:.4f}, "
             f"{'ABOVE' if above_word else 'below'})  "
             f"cat_acc={row['mean_cat_acc']:.4f} "
             f"(chance={row['cat_chance']:.4f}, "
             f"{'ABOVE' if above_cat else 'below'})")

    print("\nDone!")


if __name__ == '__main__':
    main()
