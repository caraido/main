#!/usr/bin/env python3
"""
tests.pls_learning_curve — Detect PLS overfitting by sweeping n_components.

For each (patient, embedding) pair, trains PLS models with n_components
ranging from 2 to max_components.  At each setting it records train R²
and test R² (averaged over epochs).  The gap between train and test R²
is the key overfitting diagnostic:

  - Healthy:     train ≈ test, both plateau at some n_components value.
  - Overfitting: train keeps rising but test plateaus or drops.

Outputs:
  tests/results/pls_learning_curve.csv   — full sweep data
  tests/results/pls_learning_curve.html  — interactive Plotly figure

Usage:
    python -m tests.pls_learning_curve --patients AA --embedding GloVe
    python -m tests.pls_learning_curve --patients AA AZ --epochs 20 --max-comp 30

Interpretation guide:
  - If test R² peaks early (n < 10) and then drops, PLS is overfitting
    beyond that point.  Use the peak n_components.
  - If test R² keeps rising up to max_components, you may want to extend
    the sweep.
  - Compare the kernel_pls curve to the plain pls curve: if kernel_pls
    peaks higher, nonlinearity adds value on top of PLS.
"""

import os
import sys
import argparse
import gc
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

# Add project root to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline


def run_learning_curve(patient, pdata, embeddings, emb_name,
                       n_epochs=10, comp_range=None, include_kernel=True,
                       closest='l2'):
    """
    Sweep n_components for PLS (and optionally Kernel PLS) on one
    patient × embedding pair.

    Parameters
    ----------
    patient : str
        Patient ID.
    pdata : dict
        Patient data dict from load_patient_data().
    embeddings : dict
        embedding_name → embedding_array.
    emb_name : str
        Which embedding to use (e.g. 'GloVe').
    n_epochs : int
        Random train/test splits per n_components setting.
    comp_range : list[int] or None
        n_components values to sweep.  Defaults to [2, 4, 6, 8, 10, 15, 20, 25, 30].
    include_kernel : bool
        If True, also run Kernel PLS (Nystroem + PLS) at each setting.
    closest : str
        Retrieval metric ('l2' or 'cosine').

    Returns
    -------
    pd.DataFrame
        Columns: patient, embedding, model, n_components, epoch,
        train_r2, test_r2, cat_bal_acc, word_bal_acc.
    """
    from models.model import BasicRegressor

    if comp_range is None:
        comp_range = [2, 4, 6, 8, 10, 15, 20, 25, 30]

    X = pdata['clean_data_binned'].swapaxes(1, 2)
    labels = pdata['target_concept']
    category_labels = pdata['clean_word_category']
    y = embeddings[emb_name]

    # Cap n_components at the embedding dimensionality and sample count
    n_samples = X.shape[0]
    n_features = y.shape[1] if y.ndim > 1 else 1
    max_possible = min(n_samples - 1, n_features)
    comp_range = [c for c in comp_range if c <= max_possible]

    if not comp_range:
        print(f"    [!] No valid n_components for {emb_name} "
              f"(max possible = {max_possible})")
        return pd.DataFrame()

    models_to_run = [('pls', False)]
    if include_kernel:
        models_to_run.append(('kernel_pls', True))

    records = []

    for model_name, use_kernel in models_to_run:
        for n_comp in comp_range:
            print(f"    {model_name}  n_comp={n_comp:3d} ...", end='', flush=True)

            # Build pipeline
            steps = []
            if use_kernel:
                steps.append(('nystroem', Nystroem(kernel='rbf')))
            steps.append(('pls', PLSRegression(n_components=n_comp, scale=False)))
            pipeline = Pipeline(steps)

            # PLS handles dim reduction internally — no PCA on y
            br = BasicRegressor(pipeline, y_reducer=None)
            br.load_data(
                X, y,
                n_bins_history=10,
                labels=labels,
                category_labels=category_labels,
            )

            try:
                br.fit(
                    n_epochs=n_epochs,
                    parallel=None,
                    closest=closest,
                    compute_retrieval=True,
                    save_retrieval_pairs=False,
                    compute_top_k_accuracy=False,
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            # Extract per-epoch results at best bin
            train_scores = np.array(br.all_train_score)   # (n_epochs, n_bins)
            test_scores  = np.array(br.all_test_score)    # (n_epochs, n_bins)
            cosine_sim   = np.array(br.all_cosine_sim)    # (n_epochs, n_bins)
            train_cosine = np.array(br.all_train_cosine_sim)
            cat_acc      = np.array(br.all_retrieval_category_balanced_acc)
            word_acc     = np.array(br.all_retrieval_word_balanced_acc)

            if test_scores.size == 0:
                print("  (no data)")
                continue

            # Use best test-R² bin for consistency
            best_bin = int(np.argmax(test_scores.mean(0)))

            for ep in range(test_scores.shape[0]):
                rec = {
                    'patient':      patient,
                    'embedding':    emb_name,
                    'model':        model_name,
                    'n_components': n_comp,
                    'epoch':        ep,
                    'train_r2':     float(train_scores[ep, best_bin]),
                    'test_r2':      float(test_scores[ep, best_bin]),
                    'test_cosine':  float(cosine_sim[ep, best_bin]) if cosine_sim.size > 0 else np.nan,
                    'train_cosine': float(train_cosine[ep, best_bin]) if train_cosine.size > 0 else np.nan,
                }
                if cat_acc.size > 0:
                    cat_bin = int(np.argmax(cat_acc.mean(0)))
                    rec['cat_bal_acc'] = float(cat_acc[ep, cat_bin])
                if word_acc.size > 0:
                    word_bin = int(np.argmax(word_acc.mean(0)))
                    rec['word_bal_acc'] = float(word_acc[ep, word_bin])
                records.append(rec)

            mean_train = float(train_scores[:, best_bin].mean())
            mean_test  = float(test_scores[:, best_bin].mean())
            mean_cos   = float(cosine_sim[:, best_bin].mean()) if cosine_sim.size > 0 else float('nan')
            gap = mean_train - mean_test
            print(f"  train={mean_train:.4f}  test={mean_test:.4f}  "
                  f"cos={mean_cos:.4f}  gap={gap:+.4f}")

            gc.collect()

    return pd.DataFrame(records)


def generate_html_report(df, out_path):
    """
    Generate a standalone HTML report with learning-curve plots.

    Each (patient, embedding, model) combination gets a plot showing
    train R² and test R² vs n_components, with shaded ±1 SE bands.
    """
    if df.empty:
        return None

    # Aggregate: mean ± SE over epochs
    agg = (df.groupby(['patient', 'embedding', 'model', 'n_components'])
             .agg(
                 train_mean=('train_r2', 'mean'),
                 train_se=('train_r2', lambda x: x.std() / np.sqrt(len(x))),
                 test_mean=('test_r2', 'mean'),
                 test_se=('test_r2', lambda x: x.std() / np.sqrt(len(x))),
             )
             .reset_index())

    # Find optimal n_components per group
    idx_best = agg.groupby(['patient', 'embedding', 'model'])['test_mean'].idxmax()
    best = agg.loc[idx_best, ['patient', 'embedding', 'model',
                               'n_components', 'test_mean']].copy()
    best.columns = ['patient', 'embedding', 'model',
                    'best_n_comp', 'best_test_r2']

    # Build HTML with embedded SVG charts (no JS dependencies)
    html_parts = ["""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>PLS Learning Curve — Overfitting Diagnostic</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 1100px; margin: 2rem auto; padding: 0 1rem;
         color: #1a1a1a; background: #fafafa; }
  h1 { border-bottom: 2px solid #2563eb; padding-bottom: 0.5rem; }
  h2 { color: #374151; margin-top: 2rem; }
  .summary { background: #f0f9ff; border-left: 4px solid #2563eb;
             padding: 1rem 1.5rem; margin: 1.5rem 0; border-radius: 4px; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
  th, td { border: 1px solid #d1d5db; padding: 8px 12px; text-align: center; }
  th { background: #f3f4f6; font-weight: 600; }
  tr:nth-child(even) { background: #f9fafb; }
  .overfit { color: #dc2626; font-weight: 600; }
  .healthy { color: #16a34a; font-weight: 600; }
  .chart-container { background: white; border: 1px solid #e5e7eb;
                     border-radius: 8px; padding: 1.5rem; margin: 1rem 0; }
  svg { display: block; margin: 0 auto; }
  .legend { display: flex; gap: 2rem; justify-content: center;
            margin-top: 0.5rem; font-size: 0.9rem; }
  .legend-item { display: flex; align-items: center; gap: 0.3rem; }
  .footer { color: #6b7280; font-size: 0.85rem; margin-top: 3rem;
            border-top: 1px solid #e5e7eb; padding-top: 1rem; }
</style>
</head><body>
<h1>PLS Learning Curve — Overfitting Diagnostic</h1>
"""]

    # Summary table
    html_parts.append('<div class="summary"><strong>Interpretation:</strong> '
                      'If the gap between train and test R² grows with '
                      'n_components, the model is overfitting. The optimal '
                      'n_components is where test R² peaks.</div>')

    html_parts.append('<h2>Optimal n_components per Configuration</h2>')
    html_parts.append('<table><tr><th>Patient</th><th>Embedding</th>'
                      '<th>Model</th><th>Best n_comp</th>'
                      '<th>Test R²</th><th>Verdict</th></tr>')

    for _, row in best.iterrows():
        # Check overfitting: is the gap at best_n_comp large?
        sub = agg[(agg.patient == row['patient']) &
                  (agg.embedding == row['embedding']) &
                  (agg.model == row['model'])]
        if len(sub) == 0:
            continue
        at_best = sub[sub.n_components == row['best_n_comp']].iloc[0]
        gap = at_best['train_mean'] - at_best['test_mean']

        if gap > 0.15:
            verdict = '<span class="overfit">⚠ Overfitting</span>'
        elif gap > 0.05:
            verdict = '<span class="overfit">Moderate gap</span>'
        else:
            verdict = '<span class="healthy">Healthy</span>'

        html_parts.append(
            f'<tr><td>{row["patient"]}</td><td>{row["embedding"]}</td>'
            f'<td>{row["model"]}</td><td>{int(row["best_n_comp"])}</td>'
            f'<td>{row["best_test_r2"]:.4f}</td><td>{verdict}</td></tr>')

    html_parts.append('</table>')

    # SVG charts for each (patient, embedding) pair
    for (pat, emb), grp in agg.groupby(['patient', 'embedding']):
        html_parts.append(f'<h2>{pat} — {emb}</h2>')
        html_parts.append('<div class="chart-container">')

        # Chart dimensions
        W, H = 600, 300
        PAD_L, PAD_R, PAD_T, PAD_B = 60, 20, 20, 50
        plot_w = W - PAD_L - PAD_R
        plot_h = H - PAD_T - PAD_B

        all_comps = sorted(grp['n_components'].unique())
        all_vals = list(grp['train_mean']) + list(grp['test_mean'])
        y_min = min(min(all_vals), 0)
        y_max = max(all_vals) * 1.1 if max(all_vals) > 0 else 0.1
        x_min, x_max = min(all_comps), max(all_comps)
        if x_min == x_max:
            x_max = x_min + 1

        def sx(v):
            return PAD_L + (v - x_min) / (x_max - x_min) * plot_w

        def sy(v):
            return PAD_T + plot_h - (v - y_min) / (y_max - y_min) * plot_h

        svg = [f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">']

        # Axes
        svg.append(f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" '
                   f'y2="{PAD_T + plot_h}" stroke="#9ca3af" stroke-width="1"/>')
        svg.append(f'<line x1="{PAD_L}" y1="{PAD_T + plot_h}" '
                   f'x2="{PAD_L + plot_w}" y2="{PAD_T + plot_h}" '
                   f'stroke="#9ca3af" stroke-width="1"/>')

        # Y-axis ticks
        for i in range(6):
            yv = y_min + (y_max - y_min) * i / 5
            yp = sy(yv)
            svg.append(f'<text x="{PAD_L - 8}" y="{yp + 4}" '
                       f'text-anchor="end" font-size="11" '
                       f'fill="#6b7280">{yv:.3f}</text>')
            svg.append(f'<line x1="{PAD_L}" y1="{yp}" '
                       f'x2="{PAD_L + plot_w}" y2="{yp}" '
                       f'stroke="#e5e7eb" stroke-width="0.5"/>')

        # X-axis ticks
        for c in all_comps:
            xp = sx(c)
            svg.append(f'<text x="{xp}" y="{PAD_T + plot_h + 18}" '
                       f'text-anchor="middle" font-size="11" '
                       f'fill="#6b7280">{c}</text>')

        # Axis labels
        svg.append(f'<text x="{W // 2}" y="{H - 5}" text-anchor="middle" '
                   f'font-size="12" fill="#374151">n_components</text>')
        svg.append(f'<text x="14" y="{H // 2}" text-anchor="middle" '
                   f'font-size="12" fill="#374151" '
                   f'transform="rotate(-90 14 {H // 2})">R²</text>')

        colors = {'pls': ('#2563eb', '#93c5fd'),
                  'kernel_pls': ('#dc2626', '#fca5a5')}

        for model_name in grp['model'].unique():
            mg = grp[grp.model == model_name].sort_values('n_components')
            c_main, c_light = colors.get(model_name, ('#6b7280', '#d1d5db'))

            # Train line (dashed)
            pts_train = ' '.join(f'{sx(r.n_components)},{sy(r.train_mean)}'
                                 for _, r in mg.iterrows())
            svg.append(f'<polyline points="{pts_train}" fill="none" '
                       f'stroke="{c_main}" stroke-width="1.5" '
                       f'stroke-dasharray="6,3" opacity="0.6"/>')

            # Test line (solid)
            pts_test = ' '.join(f'{sx(r.n_components)},{sy(r.test_mean)}'
                                for _, r in mg.iterrows())
            svg.append(f'<polyline points="{pts_test}" fill="none" '
                       f'stroke="{c_main}" stroke-width="2"/>')

            # SE band around test
            band_upper = ' '.join(
                f'{sx(r.n_components)},{sy(r.test_mean + r.test_se)}'
                for _, r in mg.iterrows())
            band_lower = ' '.join(
                f'{sx(r.n_components)},{sy(r.test_mean - r.test_se)}'
                for _, r in mg.iloc[::-1].iterrows())
            svg.append(f'<polygon points="{band_upper} {band_lower}" '
                       f'fill="{c_light}" opacity="0.3"/>')

            # Dots on test
            for _, r in mg.iterrows():
                svg.append(f'<circle cx="{sx(r.n_components)}" '
                           f'cy="{sy(r.test_mean)}" r="3" '
                           f'fill="{c_main}"/>')

        svg.append('</svg>')
        html_parts.append('\n'.join(svg))

        # Legend
        html_parts.append('<div class="legend">')
        for model_name in grp['model'].unique():
            c_main = colors.get(model_name, ('#6b7280',))[0]
            html_parts.append(
                f'<span class="legend-item">'
                f'<svg width="30" height="12"><line x1="0" y1="6" x2="30" '
                f'y2="6" stroke="{c_main}" stroke-width="2"/></svg>'
                f'{model_name} test</span>'
                f'<span class="legend-item">'
                f'<svg width="30" height="12"><line x1="0" y1="6" x2="30" '
                f'y2="6" stroke="{c_main}" stroke-width="1.5" '
                f'stroke-dasharray="6,3" opacity="0.6"/></svg>'
                f'{model_name} train</span>')
        html_parts.append('</div>')
        html_parts.append('</div>')

    # Footer
    html_parts.append(f'<div class="footer">Generated {datetime.now():%Y-%m-%d %H:%M} '
                      f'by tests.pls_learning_curve</div>')
    html_parts.append('</body></html>')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(html_parts))
    print(f"\nHTML report: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        prog='python -m tests.pls_learning_curve',
        description='PLS overfitting diagnostic — sweep n_components and '
                    'plot train vs test R² learning curves.',
    )
    parser.add_argument('--patients', nargs='+', default=['AA'],
                        help='Patient IDs (default: AA)')
    parser.add_argument('--embedding', nargs='+',
                        default=['GloVe', 'FastText', 'Word2Vec',
                                 'ConceptNet', 'DINOv2', 'SimCLR'],
                        help='Embeddings to test (default: all 6)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Epochs per n_components setting (default: 10)')
    parser.add_argument('--max-comp', type=int, default=30,
                        help='Maximum n_components to sweep (default: 30)')
    parser.add_argument('--comp-step', type=int, default=None,
                        help='Step size for n_components sweep '
                             '(default: auto [2,4,6,8,10,15,20,25,30])')
    parser.add_argument('--no-kernel', action='store_true',
                        help='Skip Kernel PLS (faster)')
    parser.add_argument('--closest', choices=['l2', 'cosine'], default='l2')
    parser.add_argument('--out-dir', default='tests/results')
    args = parser.parse_args()

    os.chdir(_PROJECT_DIR)
    os.makedirs(args.out_dir, exist_ok=True)

    # Build component range
    if args.comp_step:
        comp_range = list(range(2, args.max_comp + 1, args.comp_step))
    else:
        comp_range = [c for c in [2, 4, 6, 8, 10, 15, 20, 25, 30]
                      if c <= args.max_comp]

    from semantic_regression import (load_patient_data,
                                     load_shared_embedding_models,
                                     build_patient_embeddings)

    print("Loading shared embedding models...")
    shared = load_shared_embedding_models()

    all_results = []
    for patient in args.patients:
        print(f"\n{'=' * 60}")
        print(f"Patient: {patient}")
        print(f"{'=' * 60}")

        pdata = load_patient_data(patient)
        embeddings = build_patient_embeddings(pdata, shared)

        for emb_name in args.embedding:
            if emb_name not in embeddings:
                print(f"  [!] Embedding '{emb_name}' not found, skipping")
                continue

            print(f"\n  Embedding: {emb_name}")
            print(f"  {'─' * 50}")

            df = run_learning_curve(
                patient, pdata, embeddings, emb_name,
                n_epochs=args.epochs,
                comp_range=comp_range,
                include_kernel=not args.no_kernel,
                closest=args.closest,
            )
            all_results.append(df)
            gc.collect()

    results = pd.concat(all_results, ignore_index=True)
    csv_path = os.path.join(args.out_dir, 'pls_learning_curve.csv')
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    # HTML report
    html_path = os.path.join(args.out_dir, 'pls_learning_curve.html')
    generate_html_report(results, html_path)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY — Optimal n_components")
    print(f"{'=' * 60}")

    agg = (results.groupby(['patient', 'embedding', 'model', 'n_components'])
                  .agg(train_r2=('train_r2', 'mean'),
                       test_r2=('test_r2', 'mean'),
                       test_cosine=('test_cosine', 'mean'))
                  .reset_index())

    for (pat, emb, model), g in agg.groupby(['patient', 'embedding', 'model']):
        best_row = g.loc[g['test_r2'].idxmax()]
        gap = best_row['train_r2'] - best_row['test_r2']
        cos = best_row['test_cosine']
        flag = " ⚠ OVERFIT" if gap > 0.15 else ""
        print(f"  {pat}/{emb}/{model}: best n_comp={int(best_row['n_components']):3d}  "
              f"test_R²={best_row['test_r2']:.4f}  cos={cos:.4f}  "
              f"train-test gap={gap:+.4f}{flag}")


if __name__ == '__main__':
    main()
