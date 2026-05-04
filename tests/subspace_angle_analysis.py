# -*- coding: utf-8 -*-
"""
tests/subspace_angle_analysis.py
=================================
Test 4: Do phonological and semantic PLS subspaces overlap in neural space?

Fits semantic PLS and phoneme PLS independently on the same neural data,
extracts the x_rotations_ (neural directions used by each model), then
computes principal angles between the two subspaces.

  Large angles (near 90 deg) → orthogonal subspaces → separable
  Small angles (near 0 deg)  → entangled → shared neural dimensions

This is a diagnostic tool that characterises the geometry of the problem.
If angles are large, approaches 1-3 should work well.  If small, the same
neural dimensions carry both information types and surgical separation
is harder.

Usage (run from main/):
    python -m tests.subspace_angle_analysis
    python -m tests.subspace_angle_analysis --patients VB CP AA --epochs 20

Output:
    test_results/subspace_angles_{patient}.csv   (per-patient)
    test_results/subspace_angles_all.csv         (combined)

Key columns:
    patient, phon_emb, sem_emb, bin_index, time_ms,
    angle_1..angle_k, mean_angle, min_angle, max_angle
"""

import os, sys, argparse, warnings, gc, time

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.cross_decomposition import PLSRegression

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._phoneme_semantic_helpers import (
    load_phoneme_embeddings_for_patient, load_semantic_embeddings_for_patient,
    filter_nan_phoneme_trials,
    reformat, N_BINS_HISTORY, PHONEME_EMBEDDINGS, SEMANTIC_EMBEDDINGS_TO_USE,
    header, step, get_out_dir,
)
from semantic_regression import load_patient_data, load_shared_embedding_models


# ── Core computation ─────────────────────────────────────────────────────

def fit_and_get_rotations(X_feat, Y, n_components):
    """Fit Kernel PLS and return x_rotations_ in the Nystroem-mapped space.

    Returns:
        W: (n_nystroem_features, n_components)  — PLS x_rotations_
    """
    nys = Nystroem(kernel='rbf')
    X_mapped = nys.fit_transform(X_feat)
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X_mapped, Y)
    return pls.x_rotations_


def compute_angles_timecourse(X_features, Y_phon, Y_sem, n_components):
    """Compute principal angles between phoneme and semantic subspaces
    at each time bin.

    Returns DataFrame with one row per bin.
    """
    n_bins = len(X_features)
    rows = []

    for b in range(n_bins):
        X_feat = X_features[b]
        try:
            W_phon = fit_and_get_rotations(X_feat, Y_phon, n_components)
            W_sem  = fit_and_get_rotations(X_feat, Y_sem,  n_components)
        except Exception as e:
            rows.append({'bin_index': b, 'error': str(e)})
            continue

        # Principal angles (returned in radians, descending order)
        angles_rad = subspace_angles(W_phon, W_sem)
        angles_deg = np.degrees(angles_rad)

        row = {
            'bin_index': b,
            'mean_angle': float(angles_deg.mean()),
            'min_angle':  float(angles_deg.min()),
            'max_angle':  float(angles_deg.max()),
            'median_angle': float(np.median(angles_deg)),
        }
        for i, a in enumerate(angles_deg):
            row[f'angle_{i+1}'] = float(a)
        rows.append(row)

    return pd.DataFrame(rows)


# ── Per-patient runner ───────────────────────────────────────────────────

def run_patient(patient, pdata, phon_embeds, sem_embeds, args):
    out_dir = get_out_dir(args.out_dir)
    pat_csv = os.path.join(out_dir, f'subspace_angles_{patient}.csv')

    X = pdata['clean_data_binned'].swapaxes(1, 2)
    labels = np.asarray(pdata['clean_answer_labels'])
    X_features = reformat(X, N_BINS_HISTORY)

    bin_size_ms = int(pdata.get('bin_size_ms', 100))

    all_rows = []
    for phon_name in PHONEME_EMBEDDINGS:
        Y_phon = phon_embeds[phon_name]

        for sem_name, Y_sem in sem_embeds.items():
            step(f"  {phon_name} vs {sem_name}")
            df = compute_angles_timecourse(
                X_features, Y_phon, Y_sem,
                n_components=args.pls_components)
            df['patient'] = patient
            df['phon_emb'] = phon_name
            df['sem_emb'] = sem_name
            df['time_ms'] = (df['bin_index'] - N_BINS_HISTORY) * bin_size_ms
            all_rows.append(df)

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(pat_csv, index=False)
    step(f"  Saved {pat_csv}")
    return df


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Subspace angle analysis: phoneme vs semantic neural directions")
    parser.add_argument('--patients', nargs='+', default=None)
    parser.add_argument('--pls-components', type=int, default=10)
    parser.add_argument('--sem-embeddings', nargs='+', default=None)
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()

    if args.sem_embeddings:
        SEMANTIC_EMBEDDINGS_TO_USE[:] = args.sem_embeddings

    header("SUBSPACE ANGLE ANALYSIS: PHONEME VS SEMANTIC")
    print(f"  pls_components={args.pls_components}")

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
    combined_csv = os.path.join(out_dir, 'subspace_angles_all.csv')
    combined.to_csv(combined_csv, index=False)

    header("SUMMARY — Mean subspace angles at post-onset bins")
    for patient in patients:
        pat = combined[(combined['patient'] == patient) &
                       (combined['bin_index'] >= N_BINS_HISTORY)]
        for phon in PHONEME_EMBEDDINGS:
            sub = pat[pat['phon_emb'] == phon]
            if len(sub) == 0 or 'mean_angle' not in sub:
                continue
            mean_a = sub['mean_angle'].mean()
            min_a  = sub['min_angle'].min()
            step(f"  {patient}/{phon}: "
                 f"mean={mean_a:.1f} deg  min={min_a:.1f} deg  "
                 f"({'ORTHOGONAL' if mean_a > 60 else 'ENTANGLED' if mean_a < 30 else 'MIXED'})")

    print("\nDone!")


if __name__ == '__main__':
    main()
