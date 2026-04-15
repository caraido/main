"""
tests/partial_rsa.py
=====================
Test 3: Partial RSA — what fraction of neural prediction geometry is
uniquely phonological vs uniquely semantic?

Computes three RDMs per patient at each time bin:
  - Neural prediction RDM  (from phoneme regression output)
  - Phoneme ground-truth RDM
  - Semantic ground-truth RDM

Then:
  r_pred_phon       = Spearman(neural RDM, phoneme RDM)
  r_pred_sem        = Spearman(neural RDM, semantic RDM)
  r_phon_sem        = Spearman(phoneme RDM, semantic RDM)
  r_partial_phon    = partial corr of pred~phon | sem
  r_partial_sem     = partial corr of pred~sem  | phon

If r_partial_phon is significant and r_partial_sem is not, the neural
prediction is uniquely phonological.

Usage (run from main/):
    python -m tests.partial_rsa
    python -m tests.partial_rsa --patients VB CP AA --epochs 20

Output:
    test_results/partial_rsa_{patient}.csv   (per-patient, per-bin)
    test_results/partial_rsa_all.csv         (combined)

Key columns:
    patient, phon_emb, sem_emb, bin_index, time_ms,
    r_pred_phon, r_pred_sem, r_phon_sem,
    r_partial_phon, r_partial_sem
"""

import os, sys, argparse, warnings, gc, time

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.cross_decomposition import PLSRegression

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._phoneme_semantic_helpers import (
    load_phoneme_embeddings_for_patient, load_semantic_embeddings_for_patient,
    reformat, partial_spearman, build_retrieval_db,
    N_BINS_HISTORY, PHONEME_EMBEDDINGS, SEMANTIC_EMBEDDINGS_TO_USE,
    header, step, get_out_dir,
)
from semantic_regression import load_patient_data, load_shared_embedding_models


# ── Core computation ─────────────────────────────────────────────────────

def _word_stratified_split(labels, unique_words, word_to_idx, split=0.3,
                           rng=None):
    """Split trials so every word appears in test at least once.

    Strategy:
      1. For each word, randomly pick at least 1 trial for test.
      2. From the remaining trials, randomly fill up to the target test size.
      3. Rest go to train.

    If a word has only 1 trial, that trial goes to test AND train (it must
    appear in both for the model to learn something and for the word to have
    a valid prediction).  This slight train/test contamination for singletons
    is acceptable because:
      - RSA cares about the representational geometry, not accuracy
      - The alternative (omitting singletons) would lose words from the RDM
    """
    if rng is None:
        rng = np.random.default_rng()

    n_trials = len(labels)
    n_test_target = max(int(n_trials * split), 1)

    # Group trial indices by word
    word_trials = {w: [] for w in unique_words}
    for i, lab in enumerate(labels):
        word_trials[lab].append(i)

    test_set = set()
    singleton_set = set()  # trials that must be in BOTH train and test

    # Step 1: guarantee at least 1 test trial per word
    for w in unique_words:
        trials = word_trials[w]
        if len(trials) == 1:
            # Singleton: put in test AND mark for inclusion in train too
            test_set.add(trials[0])
            singleton_set.add(trials[0])
        else:
            chosen = rng.choice(trials, size=1, replace=False)[0]
            test_set.add(chosen)

    # Step 2: fill up to target test size from remaining trials
    remaining = [i for i in range(n_trials) if i not in test_set]
    n_more = n_test_target - len(test_set)
    if n_more > 0 and len(remaining) > 0:
        extra = rng.choice(remaining, size=min(n_more, len(remaining)), replace=False)
        test_set.update(extra.tolist())

    test_idx  = np.array(sorted(test_set))
    # Train = everything NOT in test, PLUS singletons (they must be in both)
    train_idx = np.array(sorted(
        set(range(n_trials)) - test_set | singleton_set
    ))

    return train_idx, test_idx


def compute_per_word_predictions(X_features, Y, labels,
                                 n_epochs, pls_components, split=0.3):
    """Run phoneme regression and accumulate per-word mean predictions.

    For each epoch:
      - Word-stratified train/test split (every word in test at least once)
      - Fit Kernel PLS on train
      - Predict on test
      - Accumulate predictions per word

    Returns:
        pred_per_word: dict {bin_index: (n_valid_words, D)} mean predicted
                       embedding per word, averaged across epochs.
        valid_words:   array of word labels that had ≥ 1 test prediction.
        pred_counts_per_word: dict {bin_index: (n_valid_words,)} count array.
    """
    n_bins = len(X_features)
    n_trials = Y.shape[0]
    unique_words = np.unique(labels)
    n_words = len(unique_words)
    word_to_idx = {w: i for i, w in enumerate(unique_words)}
    dim = Y.shape[1]

    rng = np.random.default_rng()

    # Accumulate predictions: (n_bins, n_words, D) sums, (n_bins, n_words) counts
    pred_sums  = np.zeros((n_bins, n_words, dim), dtype=np.float64)
    pred_counts = np.zeros((n_bins, n_words), dtype=np.int64)

    for ep in range(n_epochs):
        train_idx, test_idx = _word_stratified_split(
            labels, unique_words, word_to_idx, split=split, rng=rng)

        for b in range(n_bins):
            X_feat = X_features[b]
            pipe = Pipeline([
                ('nystroem', Nystroem(kernel='rbf')),
                ('pls', PLSRegression(n_components=pls_components, scale=False)),
            ])
            try:
                pipe.fit(X_feat[train_idx], Y[train_idx])
                Y_pred = pipe.predict(X_feat[test_idx])
            except Exception:
                continue

            for j, ti in enumerate(test_idx):
                wi = word_to_idx[labels[ti]]
                pred_sums[b, wi] += Y_pred[j]
                pred_counts[b, wi] += 1

    # Average — only keep words that were actually predicted
    # (with stratified split this should be ALL words, but guard anyway)
    min_count = pred_counts.min(axis=0)  # min count across bins per word
    valid_mask = min_count > 0           # word must have predictions at ALL bins
    valid_words = unique_words[valid_mask]

    if valid_mask.sum() < n_words:
        n_dropped = n_words - valid_mask.sum()
        print(f"    WARNING: {n_dropped}/{n_words} words had zero predictions "
              f"at some bins and are excluded from RDM")

    pred_per_word = {}
    counts_per_word = {}
    for b in range(n_bins):
        valid_sums   = pred_sums[b, valid_mask]
        valid_counts = pred_counts[b, valid_mask]
        # safe divide (valid_counts > 0 guaranteed by valid_mask)
        pred_per_word[b]   = valid_sums / valid_counts[:, None]
        counts_per_word[b] = valid_counts

    return pred_per_word, valid_words, counts_per_word


def compute_partial_rsa_timecourse(pred_per_word, valid_words,
                                   Y_phon, Y_sem, labels):
    """Compute RSA and partial RSA at each time bin.

    Args:
        pred_per_word: dict {bin: (n_valid_words, D)} from compute_per_word_predictions
        valid_words:   array of word labels that have valid predictions at all bins
        Y_phon, Y_sem: per-trial embeddings (n_trials, D_phon/sem)
        labels:        per-trial word labels (n_trials,)

    Only words in valid_words are included in the RDMs.  This ensures that
    all three RDMs (neural, phoneme, semantic) have the same dimensionality
    and no all-zero vectors.

    Returns DataFrame with one row per bin.
    """
    valid_set = set(valid_words)
    word_to_idx = {w: i for i, w in enumerate(valid_words)}
    n_words = len(valid_words)
    dim_phon = Y_phon.shape[1]
    dim_sem  = Y_sem.shape[1]

    # Ground-truth per-word embeddings (only for valid words)
    phon_per_word = np.zeros((n_words, dim_phon), dtype=np.float64)
    sem_per_word  = np.zeros((n_words, dim_sem), dtype=np.float64)
    word_counts   = np.zeros(n_words, dtype=np.int64)
    for i in range(len(labels)):
        if labels[i] not in valid_set:
            continue
        wi = word_to_idx[labels[i]]
        phon_per_word[wi] += Y_phon[i]
        sem_per_word[wi]  += Y_sem[i]
        word_counts[wi]   += 1

    has_data = word_counts > 0
    phon_per_word[has_data] /= word_counts[has_data, None]
    sem_per_word[has_data]  /= word_counts[has_data, None]

    # Guard: if somehow a valid_word has zero ground-truth trials, filter it
    if not has_data.all():
        keep = has_data
        phon_per_word = phon_per_word[keep]
        sem_per_word  = sem_per_word[keep]
        # Also filter pred_per_word to match
        pred_per_word = {b: v[keep] for b, v in pred_per_word.items()}
        n_words = keep.sum()
        print(f"    WARNING: {(~keep).sum()} valid words had no ground-truth — filtered")

    if n_words < 3:
        print(f"    ERROR: only {n_words} words — cannot compute RDM")
        return pd.DataFrame()

    # Ground-truth RDMs (fixed across bins)
    rdm_phon = pdist(phon_per_word, 'cosine')
    rdm_sem  = pdist(sem_per_word,  'cosine')

    # Check for NaN in ground-truth RDMs (shouldn't happen now)
    if np.any(np.isnan(rdm_phon)) or np.any(np.isnan(rdm_sem)):
        print(f"    ERROR: NaN in ground-truth RDMs — skipping")
        return pd.DataFrame()

    r_phon_sem, p_phon_sem = spearmanr(rdm_phon, rdm_sem)

    rows = []
    for b, pred_word in sorted(pred_per_word.items()):
        rdm_pred = pdist(pred_word, 'cosine')

        # Check for NaN (from all-zero or identical prediction vectors)
        if np.any(np.isnan(rdm_pred)) or np.std(rdm_pred) < 1e-12:
            rows.append({
                'bin_index': b,
                'r_pred_phon': np.nan, 'r_pred_sem': np.nan,
                'r_phon_sem': r_phon_sem,
                'r_partial_phon': np.nan, 'r_partial_sem': np.nan,
                'p_pred_phon': np.nan, 'p_pred_sem': np.nan,
            })
            continue

        r_pred_phon, p_pred_phon = spearmanr(rdm_pred, rdm_phon)
        r_pred_sem,  p_pred_sem  = spearmanr(rdm_pred, rdm_sem)

        r_partial_phon = partial_spearman(rdm_pred, rdm_phon, rdm_sem)
        r_partial_sem  = partial_spearman(rdm_pred, rdm_sem,  rdm_phon)

        rows.append({
            'bin_index': b,
            'r_pred_phon': r_pred_phon,
            'r_pred_sem':  r_pred_sem,
            'r_phon_sem':  r_phon_sem,
            'r_partial_phon': r_partial_phon,
            'r_partial_sem':  r_partial_sem,
            'p_pred_phon': p_pred_phon,
            'p_pred_sem':  p_pred_sem,
        })

    return pd.DataFrame(rows)


# ── Per-patient runner ───────────────────────────────────────────────────

def run_patient(patient, pdata, phon_embeds, sem_embeds, args):
    out_dir = get_out_dir(args.out_dir)
    pat_csv = os.path.join(out_dir, f'partial_rsa_{patient}.csv')

    X = pdata['clean_data_binned'].swapaxes(1, 2)
    labels = np.asarray(pdata['target_concept'])
    cats   = np.asarray(pdata['clean_word_category'])
    X_features = reformat(X, N_BINS_HISTORY)

    bin_size_ms = int(pdata.get('bin_size_ms', 100))

    all_rows = []
    for phon_name in PHONEME_EMBEDDINGS:
        Y_phon = phon_embeds[phon_name]

        step(f"  Computing per-word predictions for {phon_name}...")
        pred_per_word, valid_words, pred_counts = compute_per_word_predictions(
            X_features, Y_phon, labels,
            n_epochs=args.epochs,
            pls_components=args.pls_components,
        )
        step(f"    {len(valid_words)} words with valid predictions "
             f"(out of {len(np.unique(labels))} total)")

        for sem_name, Y_sem in sem_embeds.items():
            step(f"  Partial RSA: {phon_name} vs {sem_name}")
            rsa_df = compute_partial_rsa_timecourse(
                pred_per_word, valid_words, Y_phon, Y_sem, labels)
            rsa_df['patient'] = patient
            rsa_df['phon_emb'] = phon_name
            rsa_df['sem_emb'] = sem_name
            rsa_df['time_ms'] = (rsa_df['bin_index'] - N_BINS_HISTORY) * bin_size_ms
            all_rows.append(rsa_df)

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(pat_csv, index=False)
    step(f"  Saved {pat_csv}")
    return df


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Partial RSA: phonological vs semantic representational geometry")
    parser.add_argument('--patients', nargs='+', default=None)
    parser.add_argument('--epochs', type=int, default=10,
                        help='Epochs for accumulating predictions (default: 10)')
    parser.add_argument('--pls-components', type=int, default=10)
    parser.add_argument('--sem-embeddings', nargs='+', default=None)
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()

    if args.sem_embeddings:
        SEMANTIC_EMBEDDINGS_TO_USE[:] = args.sem_embeddings

    header("PARTIAL RSA: PHONOLOGICAL VS SEMANTIC")
    print(f"  epochs={args.epochs}  pls_comp={args.pls_components}")

    from tests._phoneme_semantic_helpers import discover_patients
    patients = args.patients or discover_patients()
    print(f"  Patients: {patients}")

    step("Loading shared semantic embedding models...")
    shared = load_shared_embedding_models()

    all_dfs = []
    for patient in patients:
        header(f"Patient: {patient}")
        t0 = time.time()
        pdata = load_patient_data(patient)
        phon_embeds = load_phoneme_embeddings_for_patient(pdata)
        sem_embeds = load_semantic_embeddings_for_patient(
            pdata, shared, SEMANTIC_EMBEDDINGS_TO_USE)
        df = run_patient(patient, pdata, phon_embeds, sem_embeds, args)
        all_dfs.append(df)
        step(f"  {patient} done in {time.time()-t0:.0f}s")
        del pdata, phon_embeds, sem_embeds
        gc.collect()

    out_dir = get_out_dir(args.out_dir)
    combined = pd.concat(all_dfs, ignore_index=True)
    combined_csv = os.path.join(out_dir, 'partial_rsa_all.csv')
    combined.to_csv(combined_csv, index=False)

    header("SUMMARY — Peak-bin partial RSA")
    for patient in patients:
        pat_df = combined[combined['patient'] == patient]
        for phon in PHONEME_EMBEDDINGS:
            sub = pat_df[pat_df['phon_emb'] == phon]
            if len(sub) == 0:
                continue
            # Peak bin by r_pred_phon
            peak_row = sub.loc[sub['r_pred_phon'].idxmax()]
            step(f"  {patient}/{phon} @ bin={int(peak_row['bin_index'])} "
                 f"({peak_row['time_ms']:.0f} ms): "
                 f"r_phon={peak_row['r_pred_phon']:.3f}  "
                 f"r_sem={peak_row['r_pred_sem']:.3f}  "
                 f"r_partial_phon={peak_row['r_partial_phon']:.3f}  "
                 f"r_partial_sem={peak_row['r_partial_sem']:.3f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
