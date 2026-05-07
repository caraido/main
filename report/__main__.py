# -*- coding: utf-8 -*-
"""
report.__main__ — CLI entry point for the report package.

Usage:
    python -m report <run_dir> [options]

Examples:
    # Full report on a single run (full path):
    python -m report results/semantic_regression/2026-03-27_14-30-00_KRR_l2_50ep

    # Bare run ID (auto-resolved to results/semantic_regression/<run_id>):
    python -m report 2026-03-27_14-30-00_KRR_l2_50ep

    # Most recent run:
    python -m report latest

    # Skip heavy analyses:
    python -m report results/semantic_regression/my_run --skip-norms --skip-bias

    # Specify a separate figures directory:
    python -m report results/semantic_regression/my_run \\
        --fig-dir figures/semantic_regression/my_run
"""

import os
import sys
import json
import argparse
import pandas as pd

from .helper.significance_testing import compute_significance
from .helper.word_bias_analysis import compute_word_bias
from .helper.metric_dissociation import compute_metric_dissociation
from .helper.embedding_norms import compute_norm_analysis
from .semantic_regression_report import generate_report
from .auditory_naming_regression_report import generate_report as generate_an_report


def _resolve_run_dir(run_dir):
    """
    Resolve run_dir, supporting:
    - Absolute or relative paths used as-is if they exist
    - Bare run IDs auto-resolved to results/semantic_regression/<run_id>
    - 'latest' → most recently modified folder under results/semantic_regression/
    """
    if run_dir == 'latest':
        base = os.path.join('results', 'semantic_regression')
        if not os.path.isdir(base):
            raise FileNotFoundError(f"results directory not found: {base}")
        candidates = [
            os.path.join(base, d) for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))
        ]
        if not candidates:
            raise FileNotFoundError(f"No run folders found in {base}")
        return max(candidates, key=os.path.getmtime)

    if os.path.isdir(run_dir):
        return run_dir

    # Try resolving as run ID under results/semantic_regression/
    candidate = os.path.join('results', 'semantic_regression', run_dir)
    if os.path.isdir(candidate):
        return candidate

    raise FileNotFoundError(
        f"Run directory not found: {run_dir!r}\n"
        f"  Tried as-is and as results/semantic_regression/{run_dir}"
    )


def main():
    parser = argparse.ArgumentParser(
        prog='python -m report',
        description='Generate a cross-patient analysis report for a single '
                    'semantic regression run.',
    )
    parser.add_argument(
        'run_dir',
        help='Path to the run results folder, bare run ID, or "latest". '
             'Examples: results/semantic_regression/2026-03-27_KRR_l2_50ep/ '
             'or just 2026-03-27_KRR_l2_50ep or latest',
    )
    parser.add_argument(
        '--fig-dir', default=None,
        help='Path to the run figures folder (default: inferred from run_dir '
             'by replacing results/ with figures/).',
    )
    parser.add_argument(
        '--out-dir', default=None,
        help='Output directory for report and CSVs (default: <run_dir>/report/).',
    )
    parser.add_argument('--skip-bias',  action='store_true', help='Skip bias analysis')
    parser.add_argument('--skip-norms', action='store_true', help='Skip norm analysis (loads PKLs)')
    parser.add_argument(
        '--data-dir', default=None,
        help='Path to the raw data folder containing patient subdirs with '
             '*_auditory_naming_df.pkl files. Used for cue-timing lines in the '
             'auditory naming report. Defaults to <main>/data/ relative to run_dir.',
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir.rstrip('/\\'))

    # Infer figure directory: results/semantic_regression/X → figures/semantic_regression/X
    if args.fig_dir:
        fig_dir = args.fig_dir
    else:
        fig_dir = run_dir.replace('results/', 'figures/', 1)
        if fig_dir == run_dir:
            fig_dir = None  # couldn't infer

    out_dir = args.out_dir or os.path.join(run_dir, 'report')

    # Infer data directory for auditory naming cue timings
    data_dir = args.data_dir
    if data_dir is None:
        # Try: <...>/main/data/ relative to run_dir ancestry
        _rd = os.path.abspath(run_dir)
        for _ in range(6):
            _cand = os.path.join(_rd, 'data')
            if os.path.isdir(_cand):
                data_dir = _cand
                break
            _rd = os.path.dirname(_rd)

    # Load meta.json if available
    meta = None
    meta_path = os.path.join(run_dir, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"Run ID   : {meta.get('run_id', '?')}")
        print(f"Pipeline : {meta.get('regressor_pipeline', '?')}")
        print(f"Closest  : {meta.get('closest', '?')}")
        print(f"Patients : {meta.get('patients', '?')}")
    else:
        print(f"[Warning] No meta.json found in {run_dir}")

    print(f"Run dir  : {run_dir}")
    print(f"Fig dir  : {fig_dir}")
    print(f"Out dir  : {out_dir}")
    print(f"Data dir : {data_dir}")
    print()

    # ── Auditory naming task: use dedicated report generator ──────────────────
    task = meta.get('task', 'picture_naming') if meta else 'picture_naming'
    if task == 'auditory_naming':
        print("Task: auditory_naming → running auditory naming report")
        report_path = generate_an_report(run_dir, out_dir, meta=meta, data_dir=data_dir)
        print()
        print("Pipeline complete!")
        if report_path:
            print(f"  Report : {report_path}")
        print(f"  Out    : {out_dir}/")
        return

    # ── Step 1: Significance ──────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: SIGNIFICANCE TESTING")
    print("=" * 60)
    sig_df = compute_significance(run_dir, fig_dir=fig_dir)
    if len(sig_df):
        os.makedirs(out_dir, exist_ok=True)
        sig_df.to_csv(os.path.join(out_dir, 'null_corrected_significance.csv'), index=False)

    # ── Step 2: Word bias ─────────────────────────────────────────────────────
    bias_df = pd.DataFrame()
    if not args.skip_bias:
        print()
        print("=" * 60)
        print("STEP 2: WORD PREDICTION BIAS")
        print("=" * 60)
        bias_df = compute_word_bias(run_dir)
        if len(bias_df):
            bias_df.to_csv(os.path.join(out_dir, 'word_prediction_bias.csv'), index=False)

    # ── Step 3: Metric dissociation ───────────────────────────────────────────
    print()
    print("=" * 60)
    print("STEP 3: METRIC DISSOCIATION")
    print("=" * 60)
    dissoc_df = compute_metric_dissociation(run_dir)
    if len(dissoc_df):
        dissoc_df.to_csv(os.path.join(out_dir, 'metric_dissociation.csv'), index=False)

    # ── Step 4: Norm analysis ─────────────────────────────────────────────────
    norm_df = pd.DataFrame()
    if not args.skip_norms:
        print()
        print("=" * 60)
        print("STEP 4: EMBEDDING NORM ANALYSIS")
        print("=" * 60)
        norm_df = compute_norm_analysis(run_dir)
        if len(norm_df):
            norm_df.to_csv(os.path.join(out_dir, 'embedding_norm_analysis.csv'), index=False)

    # ── Step 5: HTML report ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("STEP 5: GENERATE REPORT")
    print("=" * 60)
    report_path = generate_report(sig_df, bias_df, dissoc_df, norm_df, out_dir, meta=meta, run_dir=run_dir)

    print()
    print("Pipeline complete!")
    if report_path:
        print(f"  Report : {report_path}")
    print(f"  CSVs   : {out_dir}/")


if __name__ == '__main__':
    main()
