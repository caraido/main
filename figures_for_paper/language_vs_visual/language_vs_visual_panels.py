# -*- coding: utf-8 -*-
"""
figures_for_paper/language_vs_visual — plotting step (CSV-only, lightweight).

Renders "Decoded picture-naming information reflects linguistic rather than visual
structure" from source_data/*.csv (compute_language_vs_visual_data.py +
compute_rsa_embedding_similarity.py). Reads no PKLs.

Panels of the combined figure (left→right, top→bottom), as relaid out 2026-08-11:
  a  01_procrustes_matrix     Procrustes similarity between the four embedding models
  b  02_r2_timecourse         R² effect, language vs vision family, + lang>vis raster + cue legend
  c  04_peak_model_comparison at the semantic peak bin: pairwise Δ(language−vision) for
                              R²/category/word, bars + per-participant dots + one-sided
                              Wilcoxon stars — three rows down the whole right column
  d  05_preference_delta      between-participant ranked Δ(language−vision), category & word,
                              side by side
  e  06_layer_sweep           DINOv3/MoCo accuracy vs layer depth (1-indexed) vs language reference

Rendered but NOT part of the combined figure:
  03_category_timecourse      category effect, language vs vision family. Dropped from the
                              combined figure 2026-08-11 and drawn with NO panel letter —
                              'c' now belongs to the peak comparison.
  00_combined, 00_legend
  S1_preference_delta_per_participant   per-participant vision→language trajectories

The panel *functions* keep their historical names (panel_d = peak comparison, panel_e =
preference delta, panel_f = layer sweep); only the letters they are given changed, so a
function name here is not a claim about which letter it carries.

Run (cwd = main/):
  python figures_for_paper/language_vs_visual/language_vs_visual_panels.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS_ROOT = os.path.dirname(HERE)
sys.path.insert(0, FIGS_ROOT)
from paper_common import (display_id, assign_colors, apply_paper_style,   # noqa: E402
                          load_cue_style, embedding_colors)

SRC = os.path.join(HERE, 'source_data')
CUE_STYLE = load_cue_style()

# Colours are the single source of truth in figures_for_paper/embedding_style.json.
MODEL_COLOR, FAMILY_COLOR, FAMILY_LABEL, D_BAR_COLOR = embedding_colors()
MODEL_ORDER = ['GloVe', 'Word2Vec', 'DINOv3', 'MoCo']
PAIR_ORDER = ['GloVe>DINOv3', 'GloVe>MoCo', 'Word2Vec>DINOv3', 'Word2Vec>MoCo']
XLIM = (-0.5, 3.5)


def _read(name):
    return pd.read_csv(os.path.join(SRC, name))


def _has(name):
    return os.path.exists(os.path.join(SRC, name))


def _cue_bands(ax):
    if not _has('cue_timing.csv'):
        return
    for _, r in _read('cue_timing.csv').iterrows():
        st = CUE_STYLE.get(r['cue'])
        if not st:
            continue
        ax.axvspan(r['mean_s'] - r['std_s'], r['mean_s'] + r['std_s'], color=st['color'],
                   alpha=0.06, lw=0, zorder=0)
        ax.axvline(r['mean_s'], color=st['color'], lw=0.9, alpha=0.5, zorder=1)


def _cue_handles():
    """Legend handles for the cue vertical lines actually present in cue_timing.csv."""
    if not _has('cue_timing.csv'):
        return []
    present = set(_read('cue_timing.csv')['cue'])
    return [mlines.Line2D([], [], color=CUE_STYLE[c]['color'], lw=1.2, alpha=0.7,
                          label=CUE_STYLE[c]['label'])
            for c in CUE_STYLE if c in present]


def _letter(ax, s, dx=-40, dy=14):
    ax.annotate(s, xy=(0, 1), xycoords='axes fraction', xytext=(dx, dy),
                textcoords='offset points', fontsize=12, fontweight='bold', va='bottom', ha='left')


# ── Panel a — Procrustes similarity matrix (no title) ──────────────────────────

def panel_a(ax, letter='a', cbar=True):
    df = _read('panel_a_procrustes_matrix.csv').set_index('model').loc[MODEL_ORDER, MODEL_ORDER]
    M = df.to_numpy()
    im = ax.imshow(M, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(MODEL_ORDER, rotation=30, ha='right', fontsize=8)
    ax.set_yticklabels(MODEL_ORDER, fontsize=8)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if M[i, j] > 0.6 else '#222', fontweight='bold' if i == j else 'normal')
    ax.axhline(1.5, color='k', lw=1.2); ax.axvline(1.5, color='k', lw=1.2)   # family separators
    if cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('Procrustes similarity', fontsize=7); cb.ax.tick_params(labelsize=7)
    if letter:
        _letter(ax, letter, dx=-46, dy=10)


# ── Time-course panels (b category, c R²) ──────────────────────────────────────

def _timecourse(ax, csv, ylabel, letter, cue_legend=False, sig_label=True):
    """`sig_label` draws the in-plot "lang > vis (FDR q<0.05)" tag beside the significance
    bar. Off for a panel whose caption explains the bar (the shipped panel **b**), on for a
    panel that travels without one — an unexplained black bar under a curve is not
    self-evident, and the tag is the only explanation such a panel carries."""
    df = _read(csv)
    _cue_bands(ax)
    ymax = float((df['mean'] + df['sem']).max())
    fam_handles = []
    for fam in ['language', 'vision']:
        d = df[df.family == fam].sort_values('bin_index')
        c = FAMILY_COLOR[fam]
        t, mu, se = d.time_s.to_numpy(), d['mean'].to_numpy(), d['sem'].to_numpy()
        ln, = ax.plot(t, mu, color=c, lw=1.8, label=FAMILY_LABEL[fam], zorder=3)
        ax.fill_between(t, mu - se, mu + se, color=c, alpha=0.18, lw=0, zorder=2)
        fam_handles.append(ln)
    sig = df[df.family == 'language'].sort_values('bin_index')
    y0 = -0.20 * ymax                                    # raster sits well below the curves near zero
    segs = [(t - 0.05, 0.1) for t, s in zip(sig.time_s, sig.significant) if s]
    if segs:
        ax.broken_barh(segs, (y0, 0.05 * ymax), facecolors='#333333', edgecolors='none', zorder=4)
        if sig_label:
            ax.text(XLIM[0] + 0.02, y0 + 0.025 * ymax, 'lang > vis\n(FDR q<0.05)', fontsize=6,
                    color='#333333', va='center', ha='left')
    ax.axvline(0, color='black', lw=0.8, ls=':'); ax.axhline(0, color='#999', lw=0.6)
    ax.set_xlim(*XLIM); ax.set_ylim(-0.26 * ymax, ymax * 1.08)
    ax.set_xlabel('Time from picture onset (s)'); ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    leg1 = ax.legend(handles=fam_handles, fontsize=6.5, frameon=False, loc='upper right')
    if cue_legend:
        ax.add_artist(leg1)
        ax.legend(handles=_cue_handles(), fontsize=6, frameon=False, loc='upper left',
                  title='cues', title_fontsize=6)
    if letter:
        _letter(ax, letter)


def panel_b(ax, letter='b', sig_label=False):   # b = R² effect
    # sig_label=False by default: this panel ships inside the combined figure, whose caption
    # defines the significance bar. Removed from the plot 2026-08-11 (Alec) — a caption and an
    # in-plot tag saying the same thing is one of them going stale unnoticed.
    _timecourse(ax, 'panel_c_r2_timecourse.csv', 'R² effect (R² − chance)', letter,
                cue_legend=True, sig_label=sig_label)


def panel_c(ax, letter='c', sig_label=True):   # c = category effect
    # Keeps its tag: this panel is standalone-only and travels without a caption.
    _timecourse(ax, 'panel_b_category_timecourse.csv', 'Category effect (acc. − chance)', letter,
                cue_legend=False, sig_label=sig_label)


# ── Panel d — peak-bin between-model comparison (bars + dots + stars) ───────────

D_METRICS = [('r2', 'R² diff'), ('category', 'Category acc. diff'), ('word', 'Word acc. diff')]


def panel_d(axes, letter='d', share_x=False):
    """`share_x` draws the pair labels under the LAST axes only — for the stacked layout in
    the combined figure, where all three axes carry the same four pairs and repeating the
    two-line labels three times costs vertical space without telling the reader anything."""
    dots = _read('panel_d_peak_pairwise.csv')
    stats = _read('panel_d_peak_pairwise_stats.csv')
    rng = np.random.default_rng(3)
    for i, (mkey, mlabel) in enumerate(D_METRICS):
        ax = axes[i]
        st = stats[stats.metric == mkey].set_index('pair').loc[PAIR_ORDER]
        dd = dots[dots.metric == mkey]
        x = np.arange(len(PAIR_ORDER))
        # blue already denotes the language family in b/c/e, so grouping bars here use purple/green
        colors = [D_BAR_COLOR[p.split('>')[0]] for p in PAIR_ORDER]
        ax.bar(x, st['mean_diff'].to_numpy(), color=colors, width=0.66, edgecolor='white',
               alpha=0.85, zorder=2)
        ax.errorbar(x, st['mean_diff'].to_numpy(), yerr=st['sem'].to_numpy(), fmt='none',
                    ecolor='#333', elinewidth=1.0, capsize=2.5, zorder=4)
        # per-participant dots
        for xi, pair in zip(x, PAIR_ORDER):
            v = dd[dd.pair == pair]['diff'].to_numpy()
            jit = (rng.random(len(v)) - 0.5) * 0.3
            ax.plot(xi + jit, v, 'o', ms=4, color='#444', mec='white', mew=0.4, alpha=0.7, zorder=3)
        # stars
        ax.axhline(0, color='#333', lw=0.7)
        top = max(float((st['mean_diff'] + st['sem']).max()), float(dd['diff'].max()))
        bot = min(0.0, float(dd['diff'].min()))
        for xi, pair in zip(x, PAIR_ORDER):
            star = st.loc[pair, 'star']
            ax.text(xi, top + 0.06 * (top - bot), star, ha='center', va='bottom',
                    fontsize=8 if star != 'n.s.' else 6.5,
                    color='#111' if star != 'n.s.' else '#888')
        ax.set_xticks(x)
        last = (i == len(D_METRICS) - 1)
        ax.set_xticklabels([p.replace('>', '\n>') for p in PAIR_ORDER] if (last or not share_x)
                           else [], fontsize=6)
        ax.set_ylabel(f'Δ {mlabel}\n(language − vision)', fontsize=7.5)
        ax.set_ylim(bot - 0.05 * (top - bot), top + 0.20 * (top - bot))
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if i == 0 and letter:
            _letter(ax, letter, dx=-52, dy=10)


# ── Panel e — between-participant ranked Δ(language − vision) ───────────────────

def panel_e(axes, letter='e'):
    df = _read('panel_e_preference_delta.csv')
    for ax, metric in zip(axes, ['category', 'word']):
        d = df[df.metric == metric].sort_values('delta')
        y = np.arange(len(d))
        colors = [FAMILY_COLOR['language'] if v > 0 else FAMILY_COLOR['vision'] for v in d.delta]
        ax.barh(y, d.delta.to_numpy(), color=colors, edgecolor='white', height=0.72)
        ax.set_yticks(y); ax.set_yticklabels(d.display_id, fontsize=6)
        ax.axvline(0, color='#333', lw=0.8)
        ax.set_xlabel(f'Δ {metric} acc. (language − vision)', fontsize=7)
        n_pref = int((d.delta > 0).sum())
        ax.text(0.97, 0.04, f'{n_pref}/{len(d)} favour language', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=6.5, color=FAMILY_COLOR['language'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    if letter:
        _letter(axes[0], letter, dx=-44, dy=10)


# ── Panel f — vision layer sweep vs language reference (1-indexed layers) ───────

def panel_f(axes, letter='f'):
    if not _has('panel_f_layer_sweep.csv'):
        for ax in axes:
            ax.text(0.5, 0.5, 'layer sweep pending', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='#999'); ax.axis('off')
        return False
    df = _read('panel_f_layer_sweep.csv')
    ref = _read('panel_f_language_reference.csv').iloc[0] if _has('panel_f_language_reference.csv') else None
    VIS = {'DINOv3': MODEL_COLOR['DINOv3'], 'MoCo': MODEL_COLOR['MoCo']}
    for j, (mean_c, sem_c, reflab, ylab) in enumerate([
            ('cat_bal_acc_mean', 'cat_bal_acc_sem', 'cat_ref', 'Category accuracy'),
            ('word_bal_acc_mean', 'word_bal_acc_sem', 'word_ref', 'Word accuracy')]):
        ax = axes[j]
        maxlayer = 1
        for fam, g in df.groupby('model_family'):
            g = g.sort_values('layer_idx')
            c = VIS.get(fam, '#777')
            xl = g.layer_idx.to_numpy() + 1                 # 1-indexed integer layers
            ml, sl = g[mean_c].to_numpy(), g[sem_c].to_numpy()
            ax.plot(xl, ml, '-o', ms=3, lw=1.3, color=c, label=fam)
            ax.fill_between(xl, ml - sl, ml + sl, color=c, alpha=0.13, lw=0)
            maxlayer = max(maxlayer, int(xl.max()))
        if ref is not None:
            m, s = float(ref[f'{reflab}_mean']), float(ref[f'{reflab}_sem'])
            ax.axhspan(m - s, m + s, color=FAMILY_COLOR['language'], alpha=0.13, lw=0)
            ax.axhline(m, color=FAMILY_COLOR['language'], lw=1.4, ls='--', label='language ref.')
        ax.set_xticks([t for t in (1, 4, 7, 10, 13) if t <= maxlayer])
        ax.tick_params(axis='x', labelsize=7)
        ax.set_xlabel('Vision-model layer'); ax.set_ylabel(ylab)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if j == 0:
            ax.legend(fontsize=6, frameon=False, loc='best')
            if letter:
                _letter(ax, letter, dx=-44, dy=12)
    return True


def _participant_colors(df):
    pats = list(dict.fromkeys(df['patient'].tolist()))
    cols = assign_colors(pats)
    return {p: cols[i] for i, p in enumerate(pats)}


def _save(fig, stem, dpi=200):
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    fig.savefig(stem + '.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main():
    apply_paper_style()

    f, ax = plt.subplots(figsize=(4.2, 3.6)); panel_a(ax); f.tight_layout()
    _save(f, os.path.join(HERE, '01_procrustes_matrix'))

    f, ax = plt.subplots(figsize=(5.4, 3.6)); panel_b(ax); f.tight_layout()
    _save(f, os.path.join(HERE, '02_r2_timecourse'))

    # Standalone only — this panel left the combined figure on 2026-08-11, so it carries no
    # letter: 'c' now belongs to the peak comparison, and two panels answering to one letter
    # is how a caption comes to describe the wrong axes.
    f, ax = plt.subplots(figsize=(5.4, 3.6)); panel_c(ax, letter=None); f.tight_layout()
    _save(f, os.path.join(HERE, '03_category_timecourse'))

    # Letters below match the combined figure, which is where they are read from.
    f, axes = plt.subplots(1, 3, figsize=(9.6, 3.2)); panel_d(axes, letter='c'); f.tight_layout()
    _save(f, os.path.join(HERE, '04_peak_model_comparison'))

    f, axes = plt.subplots(1, 2, figsize=(7.0, 4.2)); panel_e(axes, letter='d'); f.tight_layout()
    _save(f, os.path.join(HERE, '05_preference_delta'))

    f, axes = plt.subplots(1, 2, figsize=(7.2, 3.2)); panel_f(axes, letter='e'); f.tight_layout()
    _save(f, os.path.join(HERE, '06_layer_sweep'))

    _combined()
    _legend()
    _supplement()
    print(f"[panels] figures written to {HERE}")


def _combined():
    """Combined figure, relaid out 2026-08-11 (Alec):

        col 0            col 1            col 2
        a procrustes  |  b R² t-course |  c  Δ R²        (rows 0–2,
        d pref Δcat   |    pref Δword  |     Δ category   one metric
        e layer cat   |    layer word  |     Δ word       per row)

    The category-effect timecourse (``panel_c``) is no longer part of this figure; it
    still renders as the standalone ``03_category_timecourse``. Panel *functions* keep
    their historical names — panel_d is the peak comparison, panel_e the preference
    delta, panel_f the layer sweep — so only the letters passed in change. Letters still
    run left→right then top→bottom: a, b, c across the top, then d, e down the left.
    """
    fig = plt.figure(figsize=(13.0, 9.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.0], hspace=0.42, wspace=0.42)
    # row 0, cols 0–1
    panel_a(fig.add_subplot(gs[0, 0]), cbar=False)
    panel_b(fig.add_subplot(gs[0, 1]))
    # right column, full height: peak comparison, one metric per row. share_x because the
    # three axes carry identical pair labels; only the bottom one needs them.
    sub_c = gs[0:3, 2].subgridspec(3, 1, hspace=0.22)
    panel_d([fig.add_subplot(sub_c[j, 0]) for j in range(3)], letter='c', share_x=True)
    # row 1, cols 0–1: preference delta, side by side
    sub_d = gs[1, 0:2].subgridspec(1, 2, wspace=0.38)
    panel_e([fig.add_subplot(sub_d[0, j]) for j in range(2)], letter='d')
    # row 2, cols 0–1: layer sweep
    panel_f([fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])], letter='e')
    _save(fig, os.path.join(HERE, '00_combined'), dpi=250)


def _legend():
    fig, ax = plt.subplots(figsize=(8.0, 0.7)); ax.axis('off')
    handles = [mlines.Line2D([], [], color=MODEL_COLOR[e], lw=2.4, label=e) for e in MODEL_ORDER]
    handles += _cue_handles()
    ax.legend(handles=handles, ncol=len(handles), loc='center', fontsize=8, frameon=False)
    fig.tight_layout(); _save(fig, os.path.join(HERE, '00_legend'))


def _supplement():
    df = _read('panel_e_preference_delta.csv')
    colors = _participant_colors(df)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    for ax, m in zip(axes, ['category', 'word']):
        d = df[df.metric == m]
        for _, r in d.iterrows():
            ax.plot([0, 1], [r['vision'], r['language']], '-o', ms=4, lw=1.0,
                    color=colors[r['patient']], mec='white', mew=0.4, alpha=0.9, label=display_id(r['patient']))
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Vision', 'Language'])
        ax.set_title(m.capitalize(), fontsize=9); ax.set_ylabel('Post-stim. accuracy − chance')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    axes[1].legend(fontsize=5.5, frameon=False, ncol=2, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout(); _save(fig, os.path.join(HERE, 'S1_preference_delta_per_participant'))


if __name__ == '__main__':
    main()
