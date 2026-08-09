#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis.model_diagnostics.pls_components_sweep — Detect PLS overfitting by
sweeping n_components.

For each (patient, embedding) pair, trains PLS models with n_components
ranging from 2 to 40.  At each setting it records train/test R², cosine
similarity, word accuracy, and category accuracy.

The key question: different metrics peak at different n_components —
R²/cosine peak early (~4), while word/cat accuracy keep rising.  This
script characterises that trade-off across patients and embeddings.

Default sweep: [2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40]
Default retrieval: cosine (use --closest l2 to override)
Default: PLS only (use --kernel to also run Kernel PLS)

Results are saved per-patient incrementally, so a crash only loses the
current patient.  Re-running with --resume skips already-done combinations.

Outputs (in --out-dir):
  pls_lc_{patient}_{embedding}_pls.csv   — per-patient per-embedding CSV
  pls_learning_curve_summary.html        — multi-metric HTML report

Usage:
    # Full sweep, all 12 patients, 5 embeddings, cosine retrieval:
    python -m analysis.model_diagnostics.pls_components_sweep \\
        --patients AA AP AZ CP DR EH EM LH MM RB VB WBH \\
        --embedding GloVe FastText Word2Vec DINOv2 SimCLR \\
        --epochs 20 --closest cosine --no-kernel \\
        --out-dir path/to/results

    # Quick test on one patient:
    python -m analysis.model_diagnostics.pls_components_sweep --patients AA --embedding GloVe --epochs 10

    # Resume interrupted run:
    python -m analysis.model_diagnostics.pls_components_sweep \\
        --patients AA AP AZ CP DR EH EM LH MM RB VB WBH \\
        --embedding GloVe FastText Word2Vec DINOv2 SimCLR \\
        --epochs 20 --resume --out-dir path/to/results

Cohort note: this sweep covers all 12 participants. The seven-patient list that
used to sit here (VB RB AA LH AZ EH EM, run 2026-04) was the cohort at the time;
AP CP DR MM WBH were swept 2026-07-01/02 for GloVe + kernel_pls only, which is
exactly the slice the figure notebook filters to. The paper figure
(figures_for_paper/pls_components/) and both its source_data CSVs are N=12;
its notebook asserts len(patients) == 12, so a dropped CSV fails loudly.

Interpretation guide:
  - R² / cosine peak early (n≈4): extra components fit noise → test quality drops.
  - Word/cat accuracy keep rising: retrieval is a ranking task, not reconstruction.
  - Sweet spot for retrieval: n=8 (most gain, moderate overfitting).
  - Use n=4 if you want best embedding geometry (cosine/R²).
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
from analysis.helpers._phoneme_semantic_helpers import get_out_dir
from utils import config as _cfg


DEFAULT_COMP_RANGE = [2, 4, 6, 8, 10, 15, 20, 25, 30, 35]


def run_learning_curve(patient, pdata, embeddings, emb_name,
                       n_epochs=10, comp_range=None, include_kernel=False,
                       closest='cosine'):
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
        n_components values to sweep.  Defaults to [2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50].
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
        comp_range = DEFAULT_COMP_RANGE

    X = pdata['clean_data_binned'].swapaxes(1, 2)
    labels = pdata['clean_answer_labels']
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

    if include_kernel:
        models_to_run = [('kernel_pls', True)]
    else:
        models_to_run = [('pls', False)]

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
                n_bins_history=_cfg.N_BINS_HISTORY,
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
    Generate standalone HTML report with 4-panel SVG plots per patient × embedding.
    Panels: Test R² (train+test), Cosine (train+test), Word Acc (test), Cat Acc (test).
    """
    if df.empty:
        return None

    se = lambda x: x.std() / np.sqrt(max(len(x), 1))

    agg_dict = {
        'train_r2_mean': ('train_r2', 'mean'), 'train_r2_se': ('train_r2', se),
        'test_r2_mean':  ('test_r2',  'mean'), 'test_r2_se':  ('test_r2',  se),
    }
    if 'test_cosine' in df.columns:
        agg_dict.update({
            'test_cos_mean':  ('test_cosine',  'mean'), 'test_cos_se':  ('test_cosine',  se),
            'train_cos_mean': ('train_cosine', 'mean'), 'train_cos_se': ('train_cosine', se),
        })
    if 'cat_bal_acc' in df.columns:
        agg_dict.update({'cat_mean': ('cat_bal_acc', 'mean'), 'cat_se': ('cat_bal_acc', se)})
    if 'word_bal_acc' in df.columns:
        agg_dict.update({'word_mean': ('word_bal_acc', 'mean'), 'word_se': ('word_bal_acc', se)})

    agg = (df.groupby(['patient', 'embedding', 'model', 'n_components'])
             .agg(**agg_dict).reset_index())

    # ── best-n summary table ─────────────────────────────────────────────────
    def best_n_for(col):
        return agg.groupby(['patient','embedding','model'])[col].idxmax()

    html = ["""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>PLS Learning Curve — All Metrics</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 1400px; margin: 2rem auto;
         padding: 0 1.5rem; color: #1a1a1a; background: #fafafa; }
  h1 { border-bottom: 3px solid #2563eb; padding-bottom: 0.5rem; }
  h2 { color: #374151; margin-top: 2.5rem; font-size: 1.1rem; }
  .note { background: #f0f9ff; border-left: 4px solid #2563eb;
          padding: 0.8rem 1.2rem; margin: 1rem 0; border-radius: 4px; font-size: 0.92rem; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.88rem; }
  th { background: #2563eb; color: #fff; padding: 7px 10px; text-align: center; }
  td { padding: 6px 10px; border-bottom: 1px solid #e5e7eb; text-align: center; }
  tr:nth-child(even) td { background: #f9fafb; }
  .of { color: #dc2626; font-weight: 600; }
  .ok { color: #16a34a; font-weight: 600; }
  .mo { color: #d97706; font-weight: 600; }
  .panels { display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 16px; }
  .panel { background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 8px; }
  svg { display: block; }
  .footer { color: #9ca3af; font-size: 0.8rem; margin-top: 2rem;
            border-top: 1px solid #e5e7eb; padding-top: 0.8rem; }
</style></head><body>
<h1>PLS Learning Curve — n_components Sweep</h1>
"""]

    html.append('<div class="note"><b>Reading the plots:</b> '
                'Solid = test, dashed = train. Shading = ±1 SE across epochs. '
                'R² and cosine peak early (~n=4) then decline as the model overfits the regression surface. '
                'Word/cat accuracy keep rising because nearest-neighbour retrieval only cares about rank order, '
                'not absolute reconstruction quality. '
                'Sweet spot: <b>n=4</b> for cosine/R², <b>n=8</b> for retrieval accuracy.</div>')

    # ── summary table ────────────────────────────────────────────────────────
    html.append('<h2>Best n_components per metric (test, no gap constraint)</h2>')
    cols = ['patient','embedding','model']
    metric_cols = []
    for col, label in [('test_r2_mean','R²'), ('test_cos_mean','Cosine'),
                       ('word_mean','Word Acc'), ('cat_mean','Cat Acc')]:
        if col in agg.columns:
            metric_cols.append((col, label))

    html.append('<table><tr>' +
                ''.join(f'<th>{c}</th>' for c in ['Patient','Embedding','Model']) +
                ''.join(f'<th>Best n ({lbl})</th><th>{lbl}@best</th>' for _, lbl in metric_cols) +
                '<th>R² gap @n=8</th><th>Verdict</th></tr>')

    for (pat, emb, model), g in agg.groupby(['patient','embedding','model']):
        row_html = f'<td>{pat}</td><td>{emb}</td><td>{model}</td>'
        for col, _ in metric_cols:
            if col in g.columns:
                idx = g[col].idxmax()
                best_n = int(g.loc[idx, 'n_components'])
                best_v = float(g.loc[idx, col])
                row_html += f'<td>{best_n}</td><td>{best_v:.3f}</td>'
        # gap at n=8
        g8 = g[g.n_components == 8]
        if len(g8):
            gap8 = float(g8['train_r2_mean'].values[0] - g8['test_r2_mean'].values[0])
            verdict = ('<span class="of">⚠ overfit</span>' if gap8 > 0.20
                       else '<span class="mo">moderate</span>' if gap8 > 0.08
                       else '<span class="ok">healthy</span>')
            row_html += f'<td>{gap8:.3f}</td><td>{verdict}</td>'
        else:
            row_html += '<td>—</td><td>—</td>'
        html.append(f'<tr>{row_html}</tr>')
    html.append('</table>')

    # ── per-patient-embedding 4-panel plots ──────────────────────────────────
    PANEL_CFG = [
        # (test_col, train_col, test_se, train_se, ylabel, test_colour, train_colour, show_zero)
        ('test_r2_mean',  'train_r2_mean', 'test_r2_se',  'train_r2_se',  'R²',       '#2563eb', '#93c5fd', True),
        ('test_cos_mean', 'train_cos_mean','test_cos_se',  'train_cos_se', 'Cosine',   '#f59e0b', '#fcd34d', False),
        ('word_mean',     None,            'word_se',      None,           'Word Acc', '#16a34a', None,      False),
        ('cat_mean',      None,            'cat_se',       None,           'Cat Acc',  '#dc2626', None,      False),
    ]

    PW, PH = 310, 210
    PL, PR, PT, PB = 46, 10, 22, 38

    def make_panel(g_model, test_col, train_col, test_se_col, train_se_col,
                   ylabel, tc, trc, show_zero, all_comps):
        """Render one SVG panel."""
        pw = PW - PL - PR
        ph = PH - PT - PB

        if test_col not in g_model.columns:
            return (f'<svg width="{PW}" height="{PH}" xmlns="http://www.w3.org/2000/svg">'
                    f'<text x="{PW//2}" y="{PH//2}" text-anchor="middle" '
                    f'font-size="11" fill="#9ca3af">no data</text></svg>')

        all_vals = list(g_model[test_col].dropna())
        if train_col and train_col in g_model.columns:
            all_vals += list(g_model[train_col].dropna())
        if not all_vals:
            return ''
        ymin = min(min(all_vals) * 1.05 if min(all_vals) < 0 else 0, 0) if show_zero else min(all_vals) * 0.97
        ymax = max(all_vals) * 1.08 if max(all_vals) > 0 else 0.05
        yr = ymax - ymin or 1
        xmin, xmax = min(all_comps), max(all_comps)
        xr = xmax - xmin or 1

        def fx(v): return PL + pw * (v - xmin) / xr
        def fy(v): return PT + ph - ph * (v - ymin) / yr

        parts = [f'<svg width="{PW}" height="{PH}" xmlns="http://www.w3.org/2000/svg" '
                 f'style="font-family:sans-serif">']
        # title
        parts.append(f'<text x="{PW/2:.0f}" y="14" text-anchor="middle" '
                     f'font-size="11" font-weight="bold" fill="#374151">{ylabel}</text>')
        # axes
        parts.append(f'<line x1="{PL}" y1="{PT}" x2="{PL}" y2="{PT+ph}" stroke="#9ca3af" stroke-width="1"/>')
        parts.append(f'<line x1="{PL}" y1="{PT+ph}" x2="{PL+pw}" y2="{PT+ph}" stroke="#9ca3af" stroke-width="1"/>')
        # zero line
        if show_zero and ymin < 0 < ymax:
            parts.append(f'<line x1="{PL}" y1="{fy(0):.1f}" x2="{PL+pw}" y2="{fy(0):.1f}" '
                         f'stroke="#d1d5db" stroke-width="0.8" stroke-dasharray="4,2"/>')
        # y ticks
        for i in range(4):
            tv = ymin + yr * i / 3
            ty = fy(tv)
            parts.append(f'<text x="{PL-4}" y="{ty+3:.1f}" text-anchor="end" '
                         f'font-size="9" fill="#6b7280">{tv:.2f}</text>')
            parts.append(f'<line x1="{PL}" y1="{ty:.1f}" x2="{PL+pw}" y2="{ty:.1f}" '
                         f'stroke="#f3f4f6" stroke-width="0.6"/>')
        # x ticks
        for c in all_comps:
            parts.append(f'<text x="{fx(c):.1f}" y="{PT+ph+13}" text-anchor="middle" '
                         f'font-size="9" fill="#6b7280">{c}</text>')
        # x label
        parts.append(f'<text x="{PL+pw/2:.0f}" y="{PH-3}" text-anchor="middle" '
                     f'font-size="9" fill="#6b7280">n_components</text>')

        # train line (dashed, lighter)
        if train_col and train_col in g_model.columns:
            mg_s = g_model.sort_values('n_components')
            pts = ' '.join(f'{fx(r.n_components):.1f},{fy(r[train_col]):.1f}' for _, r in mg_s.iterrows())
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{trc}" '
                         f'stroke-width="1.4" stroke-dasharray="5,3" opacity="0.7"/>')

        # test SE band
        mg_s = g_model.sort_values('n_components')
        if test_se_col in g_model.columns:
            up = ' '.join(f'{fx(r.n_components):.1f},{fy(r[test_col]+r[test_se_col]):.1f}' for _, r in mg_s.iterrows())
            lo = ' '.join(f'{fx(r.n_components):.1f},{fy(r[test_col]-r[test_se_col]):.1f}' for _, r in mg_s.iloc[::-1].iterrows())
            parts.append(f'<polygon points="{up} {lo}" fill="{tc}" opacity="0.15"/>')

        # test line
        pts = ' '.join(f'{fx(r.n_components):.1f},{fy(r[test_col]):.1f}' for _, r in mg_s.iterrows())
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{tc}" stroke-width="2.2"/>')

        # dots + best-n marker
        best_idx = mg_s[test_col].idxmax()
        for _, r in mg_s.iterrows():
            is_best = (r.name == best_idx)
            r_dot = 4 if is_best else 3
            sw = '2' if is_best else '0'
            parts.append(f'<circle cx="{fx(r.n_components):.1f}" cy="{fy(r[test_col]):.1f}" '
                         f'r="{r_dot}" fill="{tc}" stroke="#fff" stroke-width="{sw}"/>')
            if is_best:
                parts.append(f'<text x="{fx(r.n_components):.1f}" y="{fy(r[test_col])-7:.1f}" '
                              f'text-anchor="middle" font-size="9" font-weight="bold" fill="{tc}">'
                              f'n={int(r.n_components)}</text>')

        parts.append('</svg>')
        return '\n'.join(parts)

    for (pat, emb), grp in agg.groupby(['patient', 'embedding']):
        html.append(f'<h2>{pat} — {emb}</h2>')
        all_comps = sorted(grp['n_components'].unique())

        for model_name in sorted(grp['model'].unique()):
            g_model = grp[grp.model == model_name].copy()
            html.append(f'<div style="font-size:0.85rem;color:#6b7280;margin:4px 0 2px">{model_name}</div>')
            html.append('<div class="panels">')
            for test_col, train_col, test_se_col, train_se_col, ylabel, tc, trc, show_zero in PANEL_CFG:
                if test_col not in agg.columns:
                    continue
                panel_svg = make_panel(g_model, test_col, train_col, test_se_col,
                                       train_se_col, ylabel, tc, trc or '#ccc',
                                       show_zero, all_comps)
                html.append(f'<div class="panel">{panel_svg}</div>')
            html.append('</div>')

    html.append(f'<div class="footer">Generated {datetime.now():%Y-%m-%d %H:%M} '
                f'· tests.pls_learning_curve · {len(df)} epoch-records</div>')
    html.append('</body></html>')

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    print(f"\nHTML report: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        prog='python -m analysis.pls_components_sweep',
        description='PLS n_components sweep — train/test R², cosine, word acc, cat acc.',
    )
    parser.add_argument('--patients', nargs='+',
                        default=['AA'],
                        help='Patient IDs  (default: AA)')
    parser.add_argument('--embedding', nargs='+',
                        default=['GloVe', 'FastText', 'Word2Vec', 'DINOv2', 'SimCLR'],
                        help='Embeddings to sweep  (default: GloVe FastText Word2Vec DINOv2 SimCLR)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Epochs per n_components setting  (default: 20)')
    parser.add_argument('--comp-range', nargs='+', type=int,
                        default=DEFAULT_COMP_RANGE,
                        help='n_components values to test  '
                             '(default: 2 4 6 8 10 15 20 25 30 35 40)')
    parser.add_argument('--kernel', action='store_true',
                        help='Run Kernel PLS only (Nystroem + PLS); omit for regular PLS only')
    parser.add_argument('--closest', choices=['l2', 'cosine'], default='cosine',
                        help='Retrieval metric  (default: cosine)')
    parser.add_argument('--out-dir', default=None,
                        help='Output directory for CSVs and HTML  '
                             '(default: results/<analysis>)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip (patient, embedding, model, n_components) combos '
                             'already present in existing per-patient CSVs')
    args = parser.parse_args()

    os.chdir(_PROJECT_DIR)
    args.out_dir = get_out_dir(args.out_dir)
    comp_range = sorted(set(args.comp_range))

    from semantic_regression import (load_patient_data,
                                     load_shared_embedding_models,
                                     build_patient_embeddings)

    print(f"n_components sweep : {comp_range}")
    print(f"Embeddings         : {args.embedding}")
    print(f"Patients           : {args.patients}")
    print(f"Epochs / setting   : {args.epochs}")
    print(f"Retrieval          : {args.closest}")
    print(f"Model              : {'kernel_pls' if args.kernel else 'pls'}")
    print(f"Resume             : {args.resume}")
    total = (len(args.patients) * len(args.embedding) *
             len(comp_range) * (2 if args.kernel else 1))
    print(f"Total model fits   : ~{total}  (×{args.epochs} epochs each)\n")

    print("Loading shared embedding models...")
    shared = load_shared_embedding_models()

    all_dfs = []

    for patient in args.patients:
        print(f"\n{'=' * 60}")
        print(f"Patient: {patient}")
        print(f"{'=' * 60}")

        # Per-patient CSV: allows incremental resume
        pat_csv = os.path.join(args.out_dir, f'pls_lc_{patient}.csv')
        existing = None
        done_keys = set()
        if args.resume and os.path.exists(pat_csv):
            existing = pd.read_csv(pat_csv)
            done_keys = set(
                zip(existing.patient, existing.embedding,
                    existing.model, existing.n_components)
            )
            print(f"  Resuming: {len(done_keys)} (patient,emb,model,n) combos already done")

        pdata = load_patient_data(patient)
        embeddings = build_patient_embeddings(pdata, shared)

        pat_records = []
        for emb_name in args.embedding:
            if emb_name not in embeddings:
                print(f"  [!] '{emb_name}' not available, skipping")
                continue

            print(f"\n  Embedding: {emb_name}")
            print(f"  {'─' * 50}")

            # Build per-embedding comp_range (filter already-done)
            model_name_key = 'kernel_pls' if args.kernel else 'pls'
            if args.resume:
                emb_comp = [c for c in comp_range
                            if (patient, emb_name, model_name_key, c) not in done_keys]
                if not emb_comp:
                    print(f"    All n_components already done, skipping")
                    continue
            else:
                emb_comp = comp_range

            df = run_learning_curve(
                patient, pdata, embeddings, emb_name,
                n_epochs=args.epochs,
                comp_range=emb_comp,
                include_kernel=args.kernel,
                closest=args.closest,
            )
            if not df.empty:
                pat_records.append(df)
            gc.collect()

        # Merge with existing and save
        if pat_records:
            new_df = pd.concat(pat_records, ignore_index=True)
            if existing is not None:
                combined = pd.concat([existing, new_df], ignore_index=True)
            else:
                combined = new_df
            combined.to_csv(pat_csv, index=False)
            print(f"\n  Saved: {pat_csv}  ({len(combined)} rows)")
            all_dfs.append(combined)
        elif existing is not None:
            all_dfs.append(existing)

    if not all_dfs:
        print("No results to report.")
        return

    results = pd.concat(all_dfs, ignore_index=True)

    # Combined summary CSV
    summary_csv = os.path.join(args.out_dir, 'pls_learning_curve_all.csv')
    results.to_csv(summary_csv, index=False)
    print(f"\nCombined CSV: {summary_csv}")

    # HTML report
    html_path = os.path.join(args.out_dir, 'pls_learning_curve_summary.html')
    generate_html_report(results, html_path)

    # Terminal summary
    print(f"\n{'=' * 60}")
    print("SUMMARY — Best n per metric (test, across all patients/embeddings)")
    print(f"{'=' * 60}")

    se = lambda x: x.std() / np.sqrt(max(len(x), 1))
    agg = (results.groupby(['patient','embedding','model','n_components'])
                  .agg(train_r2=('train_r2','mean'), test_r2=('test_r2','mean'),
                       test_cos=('test_cosine','mean') if 'test_cosine' in results.columns
                                else ('train_r2','mean'),
                       word_acc=('word_bal_acc','mean') if 'word_bal_acc' in results.columns
                                else ('train_r2','mean'),
                       cat_acc=('cat_bal_acc','mean') if 'cat_bal_acc' in results.columns
                                else ('train_r2','mean'))
                  .reset_index())

    for (pat, emb, model), g in agg.groupby(['patient','embedding','model']):
        r2_n   = int(g.loc[g.test_r2.idxmax(),  'n_components'])
        cos_n  = int(g.loc[g.test_cos.idxmax(),  'n_components'])
        word_n = int(g.loc[g.word_acc.idxmax(),  'n_components'])
        cat_n  = int(g.loc[g.cat_acc.idxmax(),   'n_components'])
        g8 = g[g.n_components == 8]
        gap8 = float(g8['train_r2'].values[0] - g8['test_r2'].values[0]) if len(g8) else float('nan')
        print(f"  {pat:4s}/{emb:8s}/{model}: "
              f"R²@n={r2_n:2d}  cos@n={cos_n:2d}  word@n={word_n:2d}  cat@n={cat_n:2d}  "
              f"gap@8={gap8:+.3f}")


if __name__ == '__main__':
    main()
