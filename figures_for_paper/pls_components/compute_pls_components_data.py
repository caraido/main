# -*- coding: utf-8 -*-
"""
figures_for_paper/pls_components/compute_pls_components_data.py
===============================================================
Aggregation step for the PLS-component-selection supplementary figure: turns the
per-participant sweep CSVs into the two **tracked** source-data tables the figure is
drawn from.

    python figures_for_paper/pls_components/compute_pls_components_data.py

Reads   results/pls_components/pls_lc_*.csv
        (produced by `python -m analysis.model_diagnostics.pls_components_sweep`)
Writes  figures_for_paper/pls_components/source_data/pls_components_grandmean.csv
        figures_for_paper/pls_components/source_data/pls_components_per_patient.csv

Why this file exists
--------------------
Both CSVs used to be produced by `pls_components_selection.ipynb` in this directory, and
they are tracked paper deliverables that downstream prose quotes. A notebook is exploratory
or demo output by repository rule -- nothing outside a notebook may depend on what one
writes -- so the *producing* half was lifted out verbatim and this module is now the
authority for those two files. The notebook keeps the plotting and is annotated to say so.

It is NOT a duplicate of `analysis/model_diagnostics/pls_components_sweep.py`. That module
does the expensive part -- ~600 model fits, `results/pls_components/pls_lc_{PATIENT}.csv`
-- and this one only aggregates its output into the paper's tracked tables. Nothing here
fits a model or touches participant data, which is also why it belongs beside the figure
rather than under `analysis/`: its destination is the tracked
`figures_for_paper/<figure>/source_data/`, matching the five other `compute_*_data.py`
steps in this tree.

Cohort
------
`--expect-patients` (default 12) is asserted, not discovered. The shipped figure and both
CSVs are N=12 -- the 2026-04 sweep's seven participants plus AP, CP, DR, MM, WBH added
2026-07-01/02 for GloVe + kernel_pls, which is exactly the slice filtered below. A missing
`pls_lc_*.csv` must fail loudly rather than quietly publishing a smaller cohort than the
caption claims. See `00_figure_caption.md`: this figure was deliberately NOT regenerated
for the 2026-08 re-run, so it is still whole-brain, 1000 ms history, N=12 and is not
comparable with the other figures in `figures_for_paper/`.

Metric note
-----------
`test_cosine` / `train_cosine` come straight from the model's predicted-vs-true cosine
(`all_cosine_sim` / `all_train_cosine_sim`). They are **not** mean-centred --
`utils.retrieval.mean_center_db` is applied on the *retrieval* path, so it affects
`cat_bal_acc` / `word_bal_acc` but not these two columns. Anything describing panel c as a
"mean-centered cosine" gap does not match what these numbers are.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS_ROOT = os.path.dirname(HERE)
MAIN_DIR = os.path.dirname(FIGS_ROOT)                       # …/main
if MAIN_DIR not in sys.path:
    sys.path.insert(0, MAIN_DIR)
if FIGS_ROOT not in sys.path:
    sys.path.insert(0, FIGS_ROOT)                           # paper_common (display IDs)

from paper_common import display_id                         # noqa: E402
from utils.paths import paper_source_data, results_dir      # noqa: E402

#: Sweep columns carried through to the figure, in the order the CSVs already have them.
#: Changing this order rewrites both tracked CSVs; diff them before committing.
METRICS = {
    'cat_bal_acc':  'Balanced category accuracy',
    'word_bal_acc': 'Balanced word accuracy',
    'test_cosine':  'Held-out cosine similarity',
    'train_cosine': 'Train cosine similarity',
}

GRANDMEAN_CSV = 'pls_components_grandmean.csv'
PER_PATIENT_CSV = 'pls_components_per_patient.csv'


def load_sweep(embedding, model, n_max):
    """Every `pls_lc_*.csv` under results/pls_components/, filtered to the figure's slice.

    `create=False`: this is a read path, and an accessor that mkdir-ed a results root it
    was only inspecting would hide a missing sweep behind an empty directory.
    """
    root = results_dir('pls_components', create=False)
    paths = sorted(root.glob('pls_lc_*.csv'))
    if not paths:
        raise FileNotFoundError(
            f"no pls_lc_*.csv under {root} -- run "
            f"`python -m analysis.model_diagnostics.pls_components_sweep` first"
        )
    raw = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

    df = raw[(raw.embedding == embedding)
             & (raw.model == model)
             & (raw.n_components <= n_max)].copy()
    # The sweep is resumable and appends, so a re-run can leave a setting twice.
    df = df.drop_duplicates(
        subset=['patient', 'embedding', 'model', 'n_components', 'epoch'])
    return df


def aggregate(df):
    """(per_pat, grand) -- hierarchical aggregation, each participant weighted equally.

    Per participant, average over the repeated train/test splits; then mean +/- SEM
    **across participants**. Averaging the pooled epochs instead would weight a
    participant by trial count, which is not what the caption claims.
    """
    comps = sorted(df.n_components.unique())

    per_pat = (df.groupby(['patient', 'n_components'])[list(METRICS)]
                 .mean().reset_index())

    grand = pd.DataFrame({'n_components': comps}).set_index('n_components')
    for col in METRICS:
        g = per_pat.groupby('n_components')[col]
        grand[col + '_mean'] = g.mean()
        grand[col + '_sem'] = g.std(ddof=1) / np.sqrt(g.count())

    grand['cosine_gap_mean'] = grand['train_cosine_mean'] - grand['test_cosine_mean']
    # The gap SEM is taken over PER-PARTICIPANT gaps, not derived from the two marginal
    # SEMs -- train and test cosine are paired within a participant, so combining the
    # marginals would ignore that pairing and overstate the spread.
    per_pat['cosine_gap'] = per_pat['train_cosine'] - per_pat['test_cosine']
    gg = per_pat.groupby('n_components')['cosine_gap']
    grand['cosine_gap_sem'] = gg.std(ddof=1) / np.sqrt(gg.count())
    return per_pat, grand.reset_index()


# VERIFIED 2026-08-11 against the committed CSVs: same 12 participants, every value
# identical to ~15 significant figures. The ONLY difference is the last ulp of the derived
# `cosine_gap_*` columns (e.g. ...737793 vs ...737792) -- a subtraction, so summation order
# decides the final bit. "git diff is empty" is therefore too strict a pass condition for
# this file; compare with a tolerance. The notebook's numbers are reproduced.
def write_source_data(per_pat, grand):
    """Both tracked CSVs. Returns their paths."""
    # Published source data identifies participants by display ID (NUEx###); the internal
    # initials are retained after it purely for traceability.
    per_pat_out = per_pat.copy()
    per_pat_out.insert(0, 'display_id', per_pat_out['patient'].map(display_id))

    # create=False because these name FILES, not directories: paper_source_data ->
    # paper_dir -> os.makedirs(path), so a filename passed with create=True is mkdir'd
    # as a directory and the write then fails with FileExistsError. The idiom matches
    # figures_for_paper/semantic_regression/within_category_null_panels.py:83.
    paper_source_data('pls_components')                 # ensure source_data/ exists
    grand_path = paper_source_data('pls_components', GRANDMEAN_CSV, create=False)
    per_path = paper_source_data('pls_components', PER_PATIENT_CSV, create=False)
    grand.to_csv(grand_path, index=False)
    per_pat_out.to_csv(per_path, index=False)
    return grand_path, per_path


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description='Aggregate the PLS n_components sweep into the figure source data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--embedding', default='GloVe',
                    help='Embedding slice the figure plots.')
    ap.add_argument('--model', default='kernel_pls',
                    help='Model slice; the paper decoder is kernel PLS.')
    ap.add_argument('--n-max', type=int, default=35, dest='n_max',
                    help='Largest n_components to carry through.')
    ap.add_argument('--chosen-n', type=int, default=10, dest='chosen_n',
                    help='The component count the figure argues for; only printed.')
    ap.add_argument('--expect-patients', type=int, default=12, dest='expect_patients',
                    help='Asserted cohort size. Raise it only alongside a re-run of the '
                         'sweep AND of the caption, which still says N=12.')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    df = load_sweep(args.embedding, args.model, args.n_max)
    patients = sorted(df.patient.unique())
    comps = sorted(df.n_components.unique())
    print(f'{len(patients)} patients: {patients}')
    print(f'n_components: {comps}')
    print(f'epochs/setting: {df.groupby(["patient", "n_components"]).size().unique()}')
    if len(patients) != args.expect_patients:
        raise AssertionError(
            f'expected {args.expect_patients} patients, found {len(patients)}: {patients}'
        )

    per_pat, grand = aggregate(df)
    grand_path, per_path = write_source_data(per_pat, grand)
    print(f'wrote {os.path.relpath(grand_path, MAIN_DIR)}  ({len(grand)} rows)')
    print(f'wrote {os.path.relpath(per_path, MAIN_DIR)}  ({len(per_pat)} rows)')

    # The values quoted in the manuscript, so they can be checked without opening a CSV.
    def at(col, n):
        return float(grand.loc[grand.n_components == n, col].iloc[0])

    if args.chosen_n in comps:
        print(f"category accuracy @ n={args.chosen_n}: "
              f"{at('cat_bal_acc_mean', args.chosen_n):.3f}")
    for n in (2, 10, 15, 20):
        if n in comps:
            print(f'train-test cosine gap @ n={n}: {at("cosine_gap_mean", n):.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
