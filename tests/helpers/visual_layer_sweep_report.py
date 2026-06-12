# -*- coding: utf-8 -*-
"""
tests.helper.visual_layer_sweep_report
======================================
HTML report generation for the visual model layer sweep experiment.

Key design decisions, informed by embedding statistics diagnostics:

  1. **Primary metric: word_bal_acc (word retrieval balanced accuracy).**
     This is the only cross-layer-comparable metric because it only asks
     "is the nearest-neighbour correct?" — invariant to embedding spread
     and dimensionality.  Both cosine and R² are biased across layers:

  2. **Cosine similarity is shown but flagged as layer-incomparable.**
     Even after mean-centering, cosine is inversely proportional to
     sqrt(intrinsic_dim).  Early layers (intrinsic_dim ~12) read ~0.26;
     late layers / pooled (intrinsic_dim ~55) read ~0.09 — NOT because
     they encode less signal, but because their centred targets have a
     much larger L2 norm in the denominator.

  3. **R² is shown but also flagged as layer-incomparable.**
     A PLS(4) predictor lives in a 4-D subspace.  For pooled embeddings
     with ~55 effective dimensions, R² is capped near 4/55 ≈ 0.07 even
     for a perfect predictor, while a layer with intrinsic_dim ~12 can
     achieve R² up to 4/12 ≈ 0.33 with the same mapping quality.

  4. **Layer 0 is annotated as artifactual.**
     DINOv2 layer_00 is the patch-embedding + positional-embedding
     output, which is a *constant* (same vector for every image). It
     produces R²=1, cosine=1, and chance retrieval — all meaningless.

  5. **Best-layer selection is based solely on word_bal_acc.**
     The Wilcoxon tests compare each layer vs pooled on word_bal_acc only.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime


# ── helpers ──────────────────────────────────────────────────────────────────

def _agg(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-layer means and SEs across epochs."""
    return (
        df.groupby(['patient', 'model_family', 'layer_key',
                    'layer_idx', 'layer_type', 'embed_dim'])
          .agg(
              word_mean=('word_bal_acc', 'mean'),
              word_se  =('word_bal_acc', lambda x: x.std() / np.sqrt(len(x))),
              cat_mean =('cat_bal_acc',  'mean'),
              cat_se   =('cat_bal_acc',  lambda x: x.std() / np.sqrt(len(x))),
              cos_mean =('test_cosine',  'mean'),
              cos_se   =('test_cosine',  lambda x: x.std() / np.sqrt(len(x))),
              r2_mean  =('test_r2',      'mean'),
              r2_se    =('test_r2',      lambda x: x.std() / np.sqrt(len(x))),
          )
          .reset_index()
          .sort_values(['patient', 'model_family', 'layer_idx'])
    )


_CSS = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  max-width: 1300px; margin: 2rem auto; padding: 0 1rem;
  color: #1a1a1a; background: #fafafa;
}
h1  { border-bottom: 2px solid #0369a1; padding-bottom: .5rem; }
h2  { color: #374151; margin-top: 2.5rem; }
h3  { color: #6b7280; margin: 1rem 0 .4rem; }
.summary {
  background: #f0f9ff; border-left: 4px solid #0369a1;
  padding: 1rem 1.5rem; margin: 1.5rem 0; border-radius: 4px;
}
.warn {
  background: #fff7ed; border-left: 4px solid #ea580c;
  padding: .6rem 1rem; margin: .8rem 0; border-radius: 4px;
  font-size: .9rem;
}
.note {
  background: #f0fdf4; border-left: 4px solid #16a34a;
  padding: .6rem 1rem; margin: .8rem 0; border-radius: 4px;
  font-size: .9rem;
}
table {
  border-collapse: collapse; width: 100%; margin: 1rem 0;
  font-size: .88rem;
}
th, td {
  border: 1px solid #d1d5db; padding: 5px 9px; text-align: center;
}
th    { background: #f3f4f6; font-weight: 600; }
tr:nth-child(even) { background: #f9fafb; }
.best  { background: #dcfce7 !important; font-weight: 600; }
.sig   { color: #16a34a; font-weight: 700; }
.ns    { color: #9ca3af; }
.art   { background: #fee2e2 !important; color: #991b1b; font-style: italic; }
.chart { background: white; border: 1px solid #e5e7eb;
         border-radius: 8px; padding: 1.5rem; margin: 1rem 0; }
.legend { display:flex; gap:2rem; justify-content:center;
          margin-top:.5rem; font-size:.88rem; }
.footer { color:#6b7280; font-size:.85rem; margin-top:3rem;
          border-top:1px solid #e5e7eb; padding-top:1rem; }
.metric-col-primary   { color: #0369a1; }
.metric-col-secondary { color: #9ca3af; font-size:.82rem; }
.fig-row { display:flex; gap:1rem; flex-wrap:wrap; }
.fig-row .chart { flex:1; min-width:340px; }
"""

_MODEL_COLORS = {
    'DINOv2':       '#0369a1',
    'DINOv2_Small': '#0891b2',
    'DINOv3':       '#7c3aed',
    'SimCLR':       '#16a34a',
    'MoCo':         '#dc2626',
}
_MODEL_COLOR_DEFAULT = '#888888'

_NOTES_HTML = """
<div class="summary">
  <strong>Primary metric: word retrieval balanced accuracy</strong><br>
  This is the only metric that is directly comparable across layers with
  different embedding structures.  It asks: after mean-centering, is the
  nearest vocabulary neighbour the correct word?  The answer is invariant
  to embedding spread and intrinsic dimensionality.
</div>
<div class="warn">
  <strong>⚠ Cosine similarity is NOT comparable across layers.</strong>
  Even after mean-centering, cosine ∝ 1/√(intrinsic_dim).  Early layers
  (dim ≈ 12) score ~0.26; pooled (dim ≈ 55) score ~0.09 — purely because
  the denominator ‖y_c‖ grows with dimensionality, not because the signal
  is weaker.  Use cosine only to compare models trained on the <em>same</em>
  embedding space.
</div>
<div class="warn">
  <strong>⚠ R² is NOT comparable across layers.</strong>
  A PLS(k) predictor lives in a k-dimensional subspace.  For pooled
  embeddings (intrinsic_dim ≈ 55) and k=4, R² is bounded by ~4/55 ≈ 0.07
  even for a perfect predictor.  For layer 1 (intrinsic_dim ≈ 12) the same
  model can achieve R² ≈ 4/12 ≈ 0.33.  R² therefore artificially favours
  low-dimensional early layers.
</div>
<div class="note">
  <strong>ℹ Layer 0 is always artifactual.</strong>
  DINOv2/ViT layer_00 is the raw patch + positional embedding, which is a
  constant vector identical for every image.  It yields R²=1, cosine=1, and
  chance retrieval — all meaningless.  Rows for layer_00 are highlighted in red.
</div>
"""


def _svg_chart(sub: pd.DataFrame, family: str, patient: str,
               pooled_word: float, pooled_cat: float) -> str:
    """Return an SVG layer-profile chart showing word_bal_acc and cat_bal_acc."""
    intermediates = (sub[sub.layer_type == 'intermediate']
                     .sort_values('layer_idx'))
    if len(intermediates) < 2:
        return ''

    W, H = 680, 300
    PL, PR, PT, PB = 70, 110, 25, 55
    pw, ph = W - PL - PR, H - PT - PB

    layers     = intermediates['layer_idx'].values
    word_vals  = intermediates['word_mean'].values
    cat_vals   = intermediates['cat_mean'].values
    word_se    = intermediates['word_se'].values

    all_vals = list(word_vals) + list(cat_vals) + [pooled_word, pooled_cat]
    y_min = max(min(v for v in all_vals if not np.isnan(v)) - 0.02, 0)
    y_max = max(v for v in all_vals if not np.isnan(v)) + 0.03
    x_min, x_max = int(layers[0]), int(layers[-1])
    if x_min == x_max:
        x_max += 1

    def sx(v):
        return PL + (v - x_min) / (x_max - x_min) * pw

    def sy(v):
        return PT + ph - (v - y_min) / (y_max - y_min) * ph

    svg = [f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">']

    # Grid & axes
    for i in range(6):
        yv = y_min + (y_max - y_min) * i / 5
        yp = sy(yv)
        svg.append(f'<text x="{PL-8}" y="{yp+4}" text-anchor="end" '
                   f'font-size="11" fill="#6b7280">{yv:.3f}</text>')
        svg.append(f'<line x1="{PL}" y1="{yp}" x2="{PL+pw}" y2="{yp}" '
                   f'stroke="#e5e7eb" stroke-width="0.8"/>')
    svg.append(f'<line x1="{PL}" y1="{PT}" x2="{PL}" y2="{PT+ph}" '
               f'stroke="#9ca3af" stroke-width="1.2"/>')
    svg.append(f'<line x1="{PL}" y1="{PT+ph}" x2="{PL+pw}" y2="{PT+ph}" '
               f'stroke="#9ca3af" stroke-width="1.2"/>')
    for l in layers:
        xp = sx(l)
        svg.append(f'<text x="{xp}" y="{PT+ph+16}" text-anchor="middle" '
                   f'font-size="11" fill="#6b7280">{int(l)}</text>')
    svg.append(f'<text x="{W//2}" y="{H-6}" text-anchor="middle" '
               f'font-size="12" fill="#374151">Layer Index</text>')
    svg.append(f'<text x="14" y="{PT+ph//2}" text-anchor="middle" '
               f'font-size="12" fill="#374151" '
               f'transform="rotate(-90,14,{PT+ph//2})">Balanced Accuracy</text>')

    # Pooled baselines
    for val, colour, label in [
        (pooled_word, '#0369a1', 'pooled word'),
        (pooled_cat,  '#7c3aed', 'pooled cat'),
    ]:
        if y_min < val < y_max:
            yp = sy(val)
            svg.append(f'<line x1="{PL}" y1="{yp}" x2="{PL+pw}" y2="{yp}" '
                       f'stroke="{colour}" stroke-width="1.2" '
                       f'stroke-dasharray="5,4" opacity="0.55"/>')
            svg.append(f'<text x="{PL+pw+4}" y="{yp+4}" font-size="10" '
                       f'fill="{colour}" opacity="0.8">{label}</text>')

    # SE error bands — word_bal_acc only
    band_pts_top = [(sx(l), sy(v + e))
                    for l, v, e in zip(layers, word_vals, word_se)]
    band_pts_bot = [(sx(l), sy(v - e))
                    for l, v, e in zip(layers, word_vals, word_se)]
    band_pts = band_pts_top + list(reversed(band_pts_bot))
    band_str = ' '.join(f'{x},{y}' for x, y in band_pts)
    svg.append(f'<polygon points="{band_str}" fill="#0369a1" opacity="0.12"/>')

    # cat_bal_acc line
    pts = ' '.join(f'{sx(l)},{sy(v)}' for l, v in zip(layers, cat_vals))
    svg.append(f'<polyline points="{pts}" fill="none" '
               f'stroke="#7c3aed" stroke-width="2"/>')
    for l, v in zip(layers, cat_vals):
        svg.append(f'<circle cx="{sx(l)}" cy="{sy(v)}" r="3.5" fill="#7c3aed"/>')

    # word_bal_acc line (drawn last = on top)
    pts = ' '.join(f'{sx(l)},{sy(v)}' for l, v in zip(layers, word_vals))
    svg.append(f'<polyline points="{pts}" fill="none" '
               f'stroke="#0369a1" stroke-width="2.5"/>')
    for l, v in zip(layers, word_vals):
        svg.append(f'<circle cx="{sx(l)}" cy="{sy(v)}" r="4" fill="#0369a1"/>')

    svg.append('</svg>')
    return '\n'.join(svg)


def _svg_single_line(layers, values, se, pooled_val,
                    y_label, color, W=680, H=260):
    """SVG line chart for one scalar metric across layers, with ±SE band."""
    if len(layers) < 2:
        return ''
    PL, PR, PT, PB = 70, 90, 25, 50
    pw, ph = W - PL - PR, H - PT - PB

    all_vals = list(values)
    if pooled_val is not None:
        all_vals.append(pooled_val)
    valid = [v for v in all_vals if not np.isnan(v)]
    if not valid:
        return ''
    y_min = max(min(valid) - 0.02, 0)
    y_max = max(valid) + 0.03
    x_min, x_max = int(layers[0]), int(layers[-1])
    if x_min == x_max:
        x_max += 1

    def sx(v): return PL + (v - x_min) / (x_max - x_min) * pw
    def sy(v): return PT + ph - (v - y_min) / (y_max - y_min) * ph

    svg = [f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">']

    for i in range(6):
        yv = y_min + (y_max - y_min) * i / 5
        yp = sy(yv)
        svg.append(f'<text x="{PL-8}" y="{yp+4}" text-anchor="end" '
                   f'font-size="11" fill="#6b7280">{yv:.3f}</text>')
        svg.append(f'<line x1="{PL}" y1="{yp}" x2="{PL+pw}" y2="{yp}" '
                   f'stroke="#e5e7eb" stroke-width="0.8"/>')
    svg.append(f'<line x1="{PL}" y1="{PT}" x2="{PL}" y2="{PT+ph}" '
               f'stroke="#9ca3af" stroke-width="1.2"/>')
    svg.append(f'<line x1="{PL}" y1="{PT+ph}" x2="{PL+pw}" y2="{PT+ph}" '
               f'stroke="#9ca3af" stroke-width="1.2"/>')
    for l in layers:
        xp = sx(l)
        svg.append(f'<text x="{xp}" y="{PT+ph+16}" text-anchor="middle" '
                   f'font-size="11" fill="#6b7280">{int(l)}</text>')
    svg.append(f'<text x="{W//2}" y="{H-6}" text-anchor="middle" '
               f'font-size="12" fill="#374151">Layer Index</text>')
    svg.append(f'<text x="14" y="{PT+ph//2}" text-anchor="middle" '
               f'font-size="12" fill="#374151" '
               f'transform="rotate(-90,14,{PT+ph//2})">{y_label}</text>')

    if pooled_val is not None and y_min <= pooled_val <= y_max:
        yp = sy(pooled_val)
        svg.append(f'<line x1="{PL}" y1="{yp}" x2="{PL+pw}" y2="{yp}" '
                   f'stroke="{color}" stroke-width="1.2" '
                   f'stroke-dasharray="5,4" opacity="0.55"/>')
        svg.append(f'<text x="{PL+pw+4}" y="{yp+4}" font-size="10" '
                   f'fill="{color}" opacity="0.8">pooled</text>')

    if se is not None:
        band_top = [(sx(l), sy(v + e)) for l, v, e in zip(layers, values, se)]
        band_bot = [(sx(l), sy(v - e)) for l, v, e in zip(layers, values, se)]
        band = band_top + list(reversed(band_bot))
        band_str = ' '.join(f'{x},{y}' for x, y in band)
        svg.append(f'<polygon points="{band_str}" fill="{color}" opacity="0.12"/>')

    pts = ' '.join(f'{sx(l)},{sy(v)}' for l, v in zip(layers, values))
    svg.append(f'<polyline points="{pts}" fill="none" '
               f'stroke="{color}" stroke-width="2.5"/>')
    for l, v in zip(layers, values):
        svg.append(f'<circle cx="{sx(l)}" cy="{sy(v)}" r="4" fill="{color}"/>')

    svg.append('</svg>')
    return '\n'.join(svg)


def _summary_table(sub: pd.DataFrame, stat_df: pd.DataFrame,
                   patient: str, family: str) -> str:
    """Return HTML for the per-patient, per-model summary table."""
    pooled_row = sub[sub.layer_type == 'pooled']
    pooled_word = pooled_row['word_mean'].values[0] if len(pooled_row) else 0.0
    pooled_cat  = pooled_row['cat_mean'].values[0]  if len(pooled_row) else 0.0
    pooled_r2   = pooled_row['r2_mean'].values[0]   if len(pooled_row) else 0.0
    pooled_cos  = pooled_row['cos_mean'].values[0]  if len(pooled_row) else 0.0

    best_word_idx = (sub[sub.layer_type != 'pooled']['word_mean'].idxmax()
                     if len(sub[sub.layer_type != 'pooled']) else None)

    rows = ['<table>',
            '<tr>'
            '<th>Layer</th>'
            '<th class="metric-col-primary">Word Acc ↑ (primary)</th>'
            '<th>Cat Acc ↑</th>'
            '<th class="metric-col-secondary">Cosine ⚠</th>'
            '<th class="metric-col-secondary">R² ⚠</th>'
            '<th>Δ Word vs pooled</th>'
            '</tr>']

    for _, row in sub.iterrows():
        lk = row['layer_key']
        is_layer0 = lk.endswith('_00')
        is_best   = (row.name == best_word_idx)

        if is_layer0:
            tr_cls = ' class="art"'
        elif is_best:
            tr_cls = ' class="best"'
        else:
            tr_cls = ''

        word_delta = row['word_mean'] - pooled_word

        # Significance star (word balanced acc, Bonferroni-corrected Wilcoxon)
        sig_html = ''
        if not stat_df.empty:
            s = stat_df[(stat_df.patient == patient) & (stat_df.layer_key == lk)]
            if len(s) > 0:
                wp = s.iloc[0].get('word_pval_bonf', 1.0)
                sig_html = (' <span class="sig" title="Bonferroni-corrected p&lt;0.05">*</span>'
                            if wp < 0.05 else '')

        layer0_note = ' <small>(constant — artifact)</small>' if is_layer0 else ''

        rows.append(
            f'<tr{tr_cls}>'
            f'<td>{lk}{layer0_note}</td>'
            f'<td class="metric-col-primary">'
            f'{row["word_mean"]:.4f} ± {row["word_se"]:.4f}</td>'
            f'<td>{row["cat_mean"]:.4f} ± {row["cat_se"]:.4f}</td>'
            f'<td class="metric-col-secondary">'
            f'{row["cos_mean"]:.4f} ± {row["cos_se"]:.4f}</td>'
            f'<td class="metric-col-secondary">'
            f'{row["r2_mean"]:.4f} ± {row["r2_se"]:.4f}</td>'
            f'<td>{word_delta:+.4f}{sig_html}</td>'
            f'</tr>'
        )

    rows.append('</table>')
    return '\n'.join(rows)


def _cross_model_svg(agg: pd.DataFrame, metric_col: str, y_label: str,
                     W: int = 760, H: int = 310) -> str:
    """
    Multi-model line chart averaged across patients.

    One colored line per model family; ±SE band across patients.
    Dashed horizontal line = per-model pooled baseline.
    X = layer_idx (artifact layer_00 excluded).
    """
    intermediates = agg[
        (agg.layer_type == 'intermediate') &
        ~agg.layer_key.str.endswith('_00')
    ]
    pooled_agg = agg[agg.layer_type == 'pooled']

    # Aggregate across patients per (model_family, layer_idx)
    model_data = {}
    for family, fsub in intermediates.groupby('model_family'):
        by_layer = (
            fsub.groupby('layer_idx')[metric_col]
                .agg(m='mean',
                     se=lambda x: (float(np.std(x.values, ddof=1) / np.sqrt(len(x)))
                                   if len(x) > 1 else 0.0))
                .reset_index()
                .sort_values('layer_idx')
        )
        p_vals = pooled_agg[pooled_agg.model_family == family][metric_col].values
        model_data[family] = {
            'layers': by_layer['layer_idx'].values,
            'mean':   by_layer['m'].values,
            'se':     by_layer['se'].values,
            'pooled': float(np.mean(p_vals)) if len(p_vals) else None,
        }

    if not model_data:
        return ''

    all_layers = sorted({int(l) for d in model_data.values() for l in d['layers']})
    all_vals   = ([v for d in model_data.values() for v in d['mean']] +
                  [d['pooled'] for d in model_data.values() if d['pooled'] is not None])
    valid      = [v for v in all_vals if v is not None and not np.isnan(v)]
    if not valid:
        return ''

    x_min, x_max = 0, max(all_layers)
    y_min = max(min(valid) - 0.015, 0.0)
    y_max = max(valid) + 0.025

    # Leave generous right margin for legend
    PL, PR, PT, PB = 65, 175, 25, 50
    pw, ph = W - PL - PR, H - PT - PB

    def sx(v): return PL + (v - x_min) / max(x_max - x_min, 1) * pw
    def sy(v): return PT + ph - (v - y_min) / (y_max - y_min) * ph

    svg = [f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">']

    # Grid + Y tick labels
    for i in range(6):
        yv = y_min + (y_max - y_min) * i / 5
        yp = sy(yv)
        svg.append(f'<text x="{PL-6}" y="{yp+4}" text-anchor="end" '
                   f'font-size="11" fill="#6b7280">{yv:.3f}</text>')
        svg.append(f'<line x1="{PL}" y1="{yp:.1f}" x2="{PL+pw}" y2="{yp:.1f}" '
                   f'stroke="#e5e7eb" stroke-width="0.8"/>')

    # X tick labels
    for l in all_layers:
        xp = sx(l)
        svg.append(f'<text x="{xp:.1f}" y="{PT+ph+16}" text-anchor="middle" '
                   f'font-size="11" fill="#6b7280">{l}</text>')

    # Axes
    svg.append(f'<line x1="{PL}" y1="{PT}" x2="{PL}" y2="{PT+ph}" '
               f'stroke="#9ca3af" stroke-width="1.2"/>')
    svg.append(f'<line x1="{PL}" y1="{PT+ph}" x2="{PL+pw}" y2="{PT+ph}" '
               f'stroke="#9ca3af" stroke-width="1.2"/>')
    svg.append(f'<text x="{PL+pw//2}" y="{H-6}" text-anchor="middle" '
               f'font-size="12" fill="#374151">Layer Index</text>')
    svg.append(f'<text x="13" y="{PT+ph//2}" text-anchor="middle" '
               f'font-size="12" fill="#374151" '
               f'transform="rotate(-90,13,{PT+ph//2})">{y_label}</text>')

    # Draw each model
    lx      = PL + pw + 12   # legend x-start
    leg_y   = PT + 8          # legend y-cursor

    for family in sorted(model_data):
        d     = model_data[family]
        color = _MODEL_COLORS.get(family, _MODEL_COLOR_DEFAULT)
        layers, means, ses = d['layers'], d['mean'], d['se']

        # ±SE band
        if len(layers) >= 2:
            top = [(sx(l), sy(v + e)) for l, v, e in zip(layers, means, ses)]
            bot = [(sx(l), sy(v - e)) for l, v, e in zip(layers, means, ses)]
            poly_str = ' '.join(f'{x:.1f},{y:.1f}'
                                for x, y in top + list(reversed(bot)))
            svg.append(f'<polygon points="{poly_str}" fill="{color}" opacity="0.13"/>')

        # Line
        pts = ' '.join(f'{sx(l):.1f},{sy(v):.1f}' for l, v in zip(layers, means))
        svg.append(f'<polyline points="{pts}" fill="none" '
                   f'stroke="{color}" stroke-width="2.5"/>')

        # Dots
        for l, v in zip(layers, means):
            svg.append(f'<circle cx="{sx(l):.1f}" cy="{sy(v):.1f}" '
                       f'r="4" fill="{color}"/>')

        # Legend: solid line entry
        svg.append(f'<line x1="{lx}" y1="{leg_y+5}" x2="{lx+20}" y2="{leg_y+5}" '
                   f'stroke="{color}" stroke-width="2.5"/>')
        svg.append(f'<circle cx="{lx+10}" cy="{leg_y+5}" r="4" fill="{color}"/>')
        svg.append(f'<text x="{lx+25}" y="{leg_y+9}" font-size="11" '
                   f'fill="#374151">{family}</text>')
        leg_y += 22

    svg.append('</svg>')
    return '\n'.join(svg)


def _top_summary_section(agg: pd.DataFrame, n_patients: int) -> str:
    """
    Cross-model comparison figures at the top of the report.

    Three charts (word bal acc, cat bal acc, R²) showing all model families
    on the same layer-index axis, averaged across patients with ±SE bands.
    Dashed lines mark per-model pooled baselines.
    """
    parts = [
        '<h2>Cross-Model Comparison — Averaged Across Patients</h2>',
        f'<div class="note">'
        f'Each line is the mean ± SE across {n_patients} patient(s). '
        f'Dashed lines = per-model pooled (final-layer) baselines. '
        f'Artifact layer_00 (constant patch embedding) excluded. '
        f'Word Acc and Cat Acc are comparable across layers; '
        f'R² is not (see notes below).'
        f'</div>',
    ]

    metrics = [
        ('word_mean', 'Word Bal Acc (primary)',          True),
        ('cat_mean',  'Cat Bal Acc',                     True),
        ('r2_mean',   'R² ⚠ (not layer-comparable)', False),
    ]

    for col, label, _primary in metrics:
        svg = _cross_model_svg(agg, col, label)
        if svg:
            parts.append('<div class="chart">')
            parts.append(f'<h3>{label} — all models, mean ± SE across patients</h3>')
            parts.append(svg)
            parts.append('</div>')

    return '\n'.join(parts)


def generate_html_report(df: pd.DataFrame, stat_df: pd.DataFrame,
                         out_path: str) -> None:
    """
    Write a standalone HTML layer-sweep report to *out_path*.

    Parameters
    ----------
    df : raw per-(layer, epoch) results from run_layer_sweep
    stat_df : per-layer Wilcoxon results from compute_vs_pooled_stats
    out_path : destination .html file
    """
    if df.empty:
        print("  [report] No data — skipping HTML generation.")
        return

    # Exclude the final pre-pooling intermediate layers (layer_12 for DINOv2,
    # layer_04 for SimCLR) — these are the CLS-token / final projection outputs
    # immediately before global average pooling, not true intermediate layers.
    _EXCLUDE_FINAL = {'dinov2_layer_12', 'simclr_layer_04'}

    agg = _agg(df)
    agg = agg[~agg['layer_key'].isin(_EXCLUDE_FINAL)].copy()
    if not stat_df.empty:
        stat_df = stat_df[~stat_df['layer_key'].isin(_EXCLUDE_FINAL)].copy()

    n_patients = len(df['patient'].unique())

    html = [f'<!DOCTYPE html>\n<html><head>\n<meta charset="utf-8">'
            f'\n<title>Layer Sweep Report</title>'
            f'\n<style>{_CSS}</style>\n</head><body>',
            '<h1>Layer Sweep — Visual Model Intermediate Layers</h1>',
            _top_summary_section(agg, n_patients),
            _NOTES_HTML]

    # ── Per-patient, per-model section ───────────────────────────────────
    for (patient, family), sub in agg.groupby(['patient', 'model_family']):
        html.append(f'<h2>{patient} — {family}</h2>')

        pooled_row  = sub[sub.layer_type == 'pooled']
        pooled_word = pooled_row['word_mean'].values[0] if len(pooled_row) else 0.0
        pooled_cat  = pooled_row['cat_mean'].values[0]  if len(pooled_row) else 0.0

        # Summary numbers
        best_layer = (sub[(sub.layer_type == 'intermediate') & (~sub.layer_key.str.endswith('_00'))]
                      .sort_values('word_mean', ascending=False))
        if len(best_layer) > 0:
            br = best_layer.iloc[0]
            delta = br['word_mean'] - pooled_word
            html.append(
                f'<p>Best intermediate layer: <strong>{br["layer_key"]}</strong> '
                f'— word_bal_acc = {br["word_mean"]:.4f} '
                f'({delta:+.4f} vs pooled {pooled_word:.4f})</p>'
            )

        html.append(_summary_table(sub, stat_df, patient, family))

        # SVG chart — word and cat bal acc only
        chart_svg = _svg_chart(sub, family, patient, pooled_word, pooled_cat)
        if chart_svg:
            html.append('<div class="chart">')
            html.append(f'<h3>Layer Profile — {family} ({patient}): '
                        f'Retrieval Balanced Accuracy</h3>')
            html.append(chart_svg)
            html.append(
                '<div class="legend">'
                '<span><svg width="20" height="12"><line x1="0" y1="6" x2="20" y2="6" '
                'stroke="#0369a1" stroke-width="2.5"/></svg> Word Bal Acc (primary)</span>'
                '<span><svg width="20" height="12"><line x1="0" y1="6" x2="20" y2="6" '
                'stroke="#7c3aed" stroke-width="2"/></svg> Cat Bal Acc</span>'
                '<span><svg width="24" height="12"><line x1="0" y1="6" x2="24" y2="6" '
                'stroke="#0369a1" stroke-width="1.2" stroke-dasharray="5,4" opacity="0.6"/>'
                '</svg> Pooled baselines</span>'
                '</div>'
            )
            html.append('</div>')

        # R² per-patient chart
        pooled_r2 = pooled_row['r2_mean'].values[0] if len(pooled_row) else None
        intermediates_r2 = (sub[sub.layer_type == 'intermediate']
                            .sort_values('layer_idx'))
        if len(intermediates_r2) >= 2:
            r2_svg = _svg_single_line(
                intermediates_r2['layer_idx'].values,
                intermediates_r2['r2_mean'].values,
                intermediates_r2['r2_se'].values,
                pooled_r2, 'R²', '#dc2626',
            )
            if r2_svg:
                html.append('<div class="chart">')
                html.append(
                    f'<h3>Layer Profile \u2014 {family} ({patient}): '
                    f'R² '
                    f'<small style="color:#ea580c;font-weight:normal">'
                    f'(\u26a0 not comparable across layers \u2014 '
                    f'see notes above)</small></h3>'
                )
                html.append(r2_svg)
                html.append('</div>')

    # ── Cross-patient consistency ─────────────────────────────────────────
    if len(df['patient'].unique()) > 1 and not stat_df.empty:
        html.append('<h2>Cross-Patient Consistency</h2>')
        html.append('<div class="note">Rankings are based on word_bal_acc '
                    '(Bonferroni-corrected one-sided Wilcoxon vs pooled).</div>')

        consistency = (
            stat_df[stat_df.layer_type == 'intermediate']
            .groupby(['model_family', 'layer_key', 'layer_idx'])
            .agg(
                n_patients      =('patient',       'count'),
                n_sig_word      =('word_pval_bonf', lambda x: (x < 0.05).sum()),
                mean_word_delta =('word_delta',     'mean'),
                mean_cat_delta  =('cat_delta',      'mean'),
            )
            .reset_index()
            .sort_values(['model_family', 'layer_idx'])
        )

        html.append(
            '<table><tr>'
            '<th>Model</th><th>Layer</th>'
            '<th>Patients tested</th>'
            '<th class="metric-col-primary">N sig better (word)</th>'
            '<th class="metric-col-primary">Mean Δword</th>'
            '<th>Mean Δcat</th>'
            '</tr>'
        )
        for _, row in consistency.iterrows():
            is_layer0 = str(row['layer_key']).endswith('_00')
            tr_cls = ' class="art"' if is_layer0 else ''
            html.append(
                f'<tr{tr_cls}>'
                f'<td>{row["model_family"]}</td>'
                f'<td>{row["layer_key"]}</td>'
                f'<td>{int(row["n_patients"])}</td>'
                f'<td class="metric-col-primary">{int(row["n_sig_word"])}</td>'
                f'<td class="metric-col-primary">{row["mean_word_delta"]:+.4f}</td>'
                f'<td>{row["mean_cat_delta"]:+.4f}</td>'
                f'</tr>'
            )
        html.append('</table>')

    # ── Summary — mean across subjects ─────────────────────────────────
    if n_patients > 0:
        html.append('<h2>Summary \u2014 Mean Across Subjects</h2>')
        html.append(
            '<div class="note">Each point is the mean \u00b1 SE of <em>per-subject means</em> '
            f'across {n_patients} subject(s). '
            'Word Acc and Cat Acc are directly comparable across layers. '
            'R\u00b2 is shown for completeness but is not comparable across layers '
            '(see notes above).</div>'
        )

        for family, fsub_all in agg.groupby('model_family'):
            intermediates_s = (fsub_all[fsub_all.layer_type == 'intermediate']
                               .groupby(['layer_key', 'layer_idx'])
                               .agg(
                                   word_mean=('word_mean', 'mean'),
                                   word_se  =('word_mean',
                                              lambda x: x.std(ddof=1) / np.sqrt(len(x))
                                              if len(x) > 1 else 0.0),
                                   cat_mean =('cat_mean',  'mean'),
                                   cat_se   =('cat_mean',
                                              lambda x: x.std(ddof=1) / np.sqrt(len(x))
                                              if len(x) > 1 else 0.0),
                                   r2_mean  =('r2_mean',   'mean'),
                                   r2_se    =('r2_mean',
                                              lambda x: x.std(ddof=1) / np.sqrt(len(x))
                                              if len(x) > 1 else 0.0),
                               )
                               .reset_index()
                               .sort_values('layer_idx'))

            pooled_s = fsub_all[fsub_all.layer_type == 'pooled']
            p_word = pooled_s['word_mean'].mean() if len(pooled_s) else None
            p_cat  = pooled_s['cat_mean'].mean()  if len(pooled_s) else None
            p_r2   = pooled_s['r2_mean'].mean()   if len(pooled_s) else None

            layers_s = intermediates_s['layer_idx'].values
            if len(layers_s) < 2:
                continue

            html.append(f'<h3>{family}</h3>')

            for metric_col, se_col, y_label, color, pval in [
                ('word_mean', 'word_se', 'Word Bal Acc', '#0369a1', p_word),
                ('cat_mean',  'cat_se',  'Cat Bal Acc',  '#7c3aed', p_cat),
                ('r2_mean',   'r2_se',   'R\u00b2',      '#dc2626', p_r2),
            ]:
                s_svg = _svg_single_line(
                    layers_s,
                    intermediates_s[metric_col].values,
                    intermediates_s[se_col].values,
                    pval, y_label, color,
                )
                if s_svg:
                    html.append('<div class="chart">')
                    html.append(
                        f'<h3 style="margin-top:.2rem">{y_label} '
                        f'\u2014 mean \u00b1 SE across subjects ({family})</h3>'
                    )
                    html.append(s_svg)
                    html.append('</div>')

            # Summary table
            html.append(
                '<table><tr>'
                '<th>Layer</th>'
                '<th class="metric-col-primary">Word Acc (mean \u00b1 SE)</th>'
                '<th>Cat Acc (mean \u00b1 SE)</th>'
                '<th class="metric-col-secondary">R\u00b2 \u26a0 (mean \u00b1 SE)</th>'
                '</tr>'
            )
            for _, row in intermediates_s.iterrows():
                is_layer0 = str(row['layer_key']).endswith('_00')
                tr_cls = ' class="art"' if is_layer0 else ''
                html.append(
                    f'<tr{tr_cls}>'
                    f'<td>{row["layer_key"]}</td>'
                    f'<td class="metric-col-primary">'
                    f'{row["word_mean"]:.4f} \u00b1 {row["word_se"]:.4f}</td>'
                    f'<td>{row["cat_mean"]:.4f} \u00b1 {row["cat_se"]:.4f}</td>'
                    f'<td class="metric-col-secondary">'
                    f'{row["r2_mean"]:.4f} \u00b1 {row["r2_se"]:.4f}</td>'
                    f'</tr>'
                )
            if p_word is not None:
                html.append(
                    f'<tr style="background:#f0f9ff;font-style:italic">'
                    f'<td>pooled (mean)</td>'
                    f'<td class="metric-col-primary">{p_word:.4f}</td>'
                    f'<td>{p_cat:.4f}</td>'
                    f'<td class="metric-col-secondary">{p_r2:.4f}</td>'
                    f'</tr>'
                )
            html.append('</table>')

    html.append(
        f'<div class="footer">Generated {datetime.now():%Y-%m-%d %H:%M} '
        f'by tests.visual_layer_sweep | Primary metric: word_bal_acc</div>'
    )
    html.append('</body></html>')

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    print(f"  HTML report: {out_path}")


def print_console_summary(results: pd.DataFrame) -> None:
    """Print a terminal summary — best layer vs pooled, ranked by word_bal_acc."""
    agg = (results.groupby(['patient', 'model_family', 'layer_key',
                             'layer_idx', 'layer_type'])
                  .agg(word=('word_bal_acc', 'mean'),
                       cat =('cat_bal_acc',  'mean'),
                       r2  =('test_r2',      'mean'))
                  .reset_index())

    print(f"\n{'='*65}")
    print("SUMMARY — Best Intermediate Layer vs Pooled  (primary: word_bal_acc)")
    print(f"{'='*65}")

    for (patient, family), grp in agg.groupby(['patient', 'model_family']):
        pooled = grp[grp.layer_type == 'pooled']
        if len(pooled) == 0:
            continue
        p_word = pooled.iloc[0]['word']
        p_cat  = pooled.iloc[0]['cat']

        # Exclude layer_00 (constant artifact)
        intermediates = (grp[(grp.layer_type == 'intermediate') &
                              ~grp.layer_key.str.endswith('_00')]
                         .sort_values('layer_idx'))
        if len(intermediates) == 0:
            continue

        best_word_row = intermediates.loc[intermediates['word'].idxmax()]

        print(f"\n  {patient} / {family}:")
        print(f"    Pooled:        word={p_word:.4f}  cat={p_cat:.4f}")
        print(f"    Best (word):   {best_word_row['layer_key']:28s}"
              f"  word={best_word_row['word']:.4f}"
              f"  Δ={best_word_row['word'] - p_word:+.4f}")
        print(f"    [note] Cosine and R² are not comparable across layers —"
              f" see HTML report for details.")
