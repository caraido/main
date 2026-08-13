# -*- coding: utf-8 -*-
"""
figures_for_paper/extendability — Open-vocabulary / zero-shot extendability panels.

Paper-figure generator for the *extendability* of the regression-and-retrieval
decoder. The kernel-PLS decoder predicts a GloVe embedding per trial; the
predicted vector is ranked by cosine similarity against an OPEN word gallery (the
stimulus words plus thousands of never-presented, POS/frequency-matched
distractors). Because the decoder outputs a point in a linguistic space rather
than a class label, new words can be added to the retrieval gallery without any
retraining, and words never seen in training can still be retrieved.

Five panels (one topic — "extendability"):
  a  median percentile rank vs gallery size N (200-5000), distribution over
     participants (box + points) with the mean trend                (evidence 1)
  b  top-k retrieval accuracy vs k at N=5000, distribution over participants
     (box + points)                                                 (evidence 1)
  c  in-vocab vs held-out (zero-shot) median percentile rank, each tested against
     chance — zero-shot words are retrieved far above chance         (evidence 2)
  d  nDCG@100 of the neural ranking vs its permutation null (whole-list semantic
     organisation, independent WordNet grade)                       (evidence 3)
  e  TWO 2D MDS (cosine) maps, the two participants highest on top-10 accuracy:
     predicted words land beside the ground-truth word and its near-synonyms
     (illustration).  One panel letter, two square axes — the right-hand map is
     identified by its own title, not by a letter of its own.

A Wu-Palmer neighbour-similarity panel sat at d until 2026-08-12 and was cut as
editorially redundant with nDCG, which carries the same claim under the same
independent WordNet grade.  The metric itself is untouched upstream and its group
result is still written to source_data/group_inference.csv.

Supplements (NOT in the combined main figure):
  S1  per-participant held-out trial percentile distributions across N (14 panels)
  S2  qualitative best-case retrievals per participant (HTML + CSV)
  S3  MDS showcase for NUE031; S4 the same for NUE036

Inputs (already computed; this script only re-plots — it does NOT re-run the heavy
permutation pipeline):
  main/figures/open_vocab_retrieval/source_data/
    per_patient_metrics_picture_naming.csv   (N=5000 headline metrics per patient)
    sweep_picture_naming.csv                 (per-patient N x variant sweep)
    group_inference_picture_naming.json      (group Wilcoxon vs chance + CIs)
  figures_for_paper/extendability/source_data/   (from compute_extendability_data.py)
    cache_heldout_trial_percentile_by_N.csv  (supp S1)
    cache_panelf_mds.csv                     (panel e, left map)
    cache_panelf_AA.csv                      (panel e, right map)
    cache_panelf_{RB,WBH}.csv                (supps S3, S4)
    cache_qualitative_bestcases.csv          (supp S2)

Reproduce (any env with numpy/pandas/matplotlib/scipy; reads CSVs, not project pkls):
  # (once, in the Speech env) build the caches for panels e/f + supplements:
  python figures_for_paper/extendability/compute_extendability_data.py
  # then render:
  python figures_for_paper/extendability/extendability_panels.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
from matplotlib.ticker import NullLocator

# Editable-text vector output (house rule)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS_ROOT = os.path.dirname(HERE)                          # …/figures_for_paper
MAIN_DIR = os.path.dirname(FIGS_ROOT)                      # …/main
sys.path.insert(0, FIGS_ROOT)                              # shared figure conventions
from paper_common import display_id, assign_colors, apply_paper_style  # noqa: E402
from utils.config import ALPHA, p_stars                    # noqa: E402  (repo-wide cutoff)

OPENVOCAB_SRC = os.path.join(MAIN_DIR, 'figures', 'open_vocab_retrieval', 'source_data')
FIG_DIR = HERE
SRC_DIR = os.path.join(HERE, 'source_data')

TASK = 'picture_naming'
HEADLINE_N = 5000
HEADLINE_VARIANT = 'matched'
KS = [1, 5, 10, 50, 100]
NS = [200, 500, 1000, 2000, 5000]
CHANCE_PCT = 0.5
SIG_ALPHA = ALPHA         # threshold for a per-participant effect drawn as "significant"
# The two MDS showcases are ONE panel (e) holding two maps, so only the left one carries a
# letter — the right is identified by its own title.  NUE027 and NUE041 are the two highest
# on top-10 retrieval accuracy; see SHOWCASE_LEFT/RIGHT in compute_extendability_data.py for
# why that criterion is named rather than a bare "best participant".
PANELF_MAIN = [('cache_panelf_mds.csv', 'e'), ('cache_panelf_AA.csv', None)]
# Supplementary showcases (not in the combined figure): (patient, supp label).
PANELF_SUPP = [('RB', 'S3'), ('WBH', 'S4')]

BLUE = 'tab:blue'
GREY = '#888888'
BOX_FACE = '#e8e8e8'

# ── Print geometry ──────────────────────────────────────────────────────────────
# The combined figure is drawn at its FINAL printed width, so nothing is reduced on the
# way to the page and every size below is the size a reader actually gets.  It used to be
# drawn 13.6 in wide for a 7.2 in slot, i.e. reduced to 0.53x, which silently took the base
# 8 pt text to 4.2 pt and the MDS neighbour words to 2.8 pt — under Nature's ~5 pt floor
# for every element on the figure.  Drawing 1:1 is what keeps utils.config's type scale
# meaningful; do not enlarge this canvas without scaling the type with it.
FIG_W_DOUBLE = 7.2          # Nature double-column, 183 mm
MDS_WORD_SIZE = 5.2         # grey neighbour words — kept small on purpose, see draw_mds
MDS_BOLD_SIZE = 7.5         # ground-truth / predicted words
MDS_TITLE_SIZE = 8.0
ANNOT_SIZE = 6.5            # "chance" and similar in-axes annotations
STAR_SIZE = 8.0
LETTER_SIZE = 9.0           # panel letters

apply_paper_style()


# ── Significance helpers ──────────────────────────────────────────────────────

def _stars(p):
    """p-value -> significance string (utils.config.p_stars; n.s. spelled out)."""
    return p_stars(p)


def _wilcoxon(values, chance, alternative):
    """One-sided Wilcoxon signed-rank of `values` vs a chance constant.
    Returns (p_value, n). Mirrors stats.wilcoxon_vs_chance without importing the
    heavy pipeline package."""
    from scipy.stats import wilcoxon
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    diffs = v - chance
    if len(v) < 1 or np.allclose(diffs, 0):
        return np.nan, len(v)
    try:
        _, p = wilcoxon(diffs, alternative=alternative)
    except ValueError:
        return np.nan, len(v)
    return float(p), len(v)


def _sig_bracket(ax, x0, x1, y, text, color='#222222', fs=8):
    """Draw a significance bracket spanning [x0,x1] with a centred label, working on
    both linear and log y-axes (tick height is multiplicative on a log axis)."""
    if ax.get_yscale() == 'log':
        y2 = y * 1.10
    else:
        yl = ax.get_ylim(); y2 = y + 0.03 * (yl[1] - yl[0])
    ax.plot([x0, x0, x1, x1], [y, y2, y2, y], lw=1.0, color=color, clip_on=False)
    ax.text((x0 + x1) / 2, y2, text, ha='center', va='bottom', fontsize=fs,
            color=color, clip_on=False)


def _box_points(ax, positions, data_by_pos, patients, colors, width=0.55, seed=0):
    """Boxplot (IQR + median) per position with jittered per-participant points
    (fixed colour per participant) and a black across-participant mean line.
    Shared by panels a, b, c, d. ``data_by_pos[i]`` is aligned to ``patients``."""
    color_of = {p: colors[i] for i, p in enumerate(patients)}
    ax.boxplot(data_by_pos, positions=positions, widths=width, showfliers=False,
               patch_artist=True, zorder=2,
               medianprops=dict(color='#333333', lw=1.3),
               boxprops=dict(facecolor=BOX_FACE, edgecolor='#999999', lw=0.8),
               whiskerprops=dict(color='#999999', lw=0.8),
               capprops=dict(color='#999999', lw=0.8))
    rng = _rng(seed)
    for xi, arr in zip(positions, data_by_pos):
        arr = np.asarray(arr, dtype=float)
        for pi, p in enumerate(patients):
            jx = xi + (rng.random() - 0.5) * 0.28
            ax.plot(jx, arr[pi], 'o', ms=3.0, color=color_of[p], alpha=0.85, zorder=3, mew=0)
    means = [float(np.nanmean(np.asarray(a, dtype=float))) for a in data_by_pos]
    ax.plot(positions, means, color='black', lw=2.0, marker='o', ms=4, zorder=4)
    return means


# ── Data loading ────────────────────────────────────────────────────────────────

def load_inputs():
    perp = pd.read_csv(os.path.join(OPENVOCAB_SRC, f'per_patient_metrics_{TASK}.csv'))
    sweep = pd.read_csv(os.path.join(OPENVOCAB_SRC, f'sweep_{TASK}.csv'))
    with open(os.path.join(OPENVOCAB_SRC, f'group_inference_{TASK}.json'), encoding='utf-8') as f:
        ginf = json.load(f)
    patients = list(perp['patient'])
    return perp, sweep, ginf, patients


def _cache(name):
    path = os.path.join(SRC_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def _sem(a):
    a = np.asarray(a, dtype=float)
    return float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else np.nan


def _rng(seed=0):
    return np.random.default_rng(seed)


# ── Panel a: median percentile rank vs N (box + points) ──────────────────────────

def draw_scaling(ax, sweep, patients, colors, panel_letter=None):
    """a — distribution of median percentile rank (rank/N; lower=better) over
    participants at each gallery size N, with the across-participant mean trend."""
    sub = sweep[sweep['variant'] == HEADLINE_VARIANT]
    piv = sub.pivot(index='patient', columns='N', values='median_percentile').reindex(patients)
    xpos = np.arange(len(NS))
    data = [piv[n].to_numpy(dtype=float) for n in NS]
    _box_points(ax, xpos, data, patients, colors, seed=0)
    ax.axhline(CHANCE_PCT, ls='--', color=GREY, lw=1.0, zorder=1)
    # Above the line, not centred on it: va='center' put the text across the dashed rule.
    ax.text(xpos[-1], CHANCE_PCT, 'chance ', color=GREY, fontsize=ANNOT_SIZE,
            va='bottom', ha='right')
    # significance vs chance (Wilcoxon, one-sided less) at y≈0.4
    for xi, d in enumerate(data):
        p, _ = _wilcoxon(d, CHANCE_PCT, 'less')
        ax.text(xi, 0.4, _stars(p), ha='center', va='center', fontsize=STAR_SIZE, color='#222222')
    ax.set_xticks(xpos); ax.set_xticklabels([str(n) for n in NS])
    ax.set_xlim(-0.6, len(NS) - 0.4)
    ax.set_ylim(0, 0.55)
    ax.set_xlabel('Gallery size $N$ (words)')
    ax.set_ylabel('Median percentile rank')
    _letter(ax, panel_letter)


# ── Panel b: top-k accuracy vs k (box + points) ─────────────────────────────────

def draw_cmc(ax, perp, patients, colors, panel_letter=None):
    """b — distribution of top-k retrieval accuracy over participants at each k
    (N=5000), matching panel a's box+points fashion."""
    xpos = np.arange(len(KS))
    data = [perp[f'top{k}_all'].to_numpy(dtype=float) for k in KS]
    ax.set_yscale('log')
    _box_points(ax, xpos, data, patients, colors, seed=1)
    # chance = k / N
    chance = [k / float(HEADLINE_N) for k in KS]
    ax.plot(xpos, chance, ls='--', color=GREY, lw=1.0, marker='.', zorder=1)
    # Right-anchored inside the axes.  Hanging off the right edge with ha='left' put this
    # label over panel c's y-axis once the figure was saved at a fixed width.
    ax.text(xpos[-1], chance[-1] * 1.35, 'chance ($k/N$) ', color=GREY, fontsize=ANNOT_SIZE,
            va='bottom', ha='right')
    # significance vs chance k/N (Wilcoxon greater), above each box
    for xi, (d, k) in enumerate(zip(data, KS)):
        p, _ = _wilcoxon(d, k / float(HEADLINE_N), 'greater')
        ax.text(xi, np.nanmax(d) * 1.35, _stars(p), ha='center', va='bottom', fontsize=STAR_SIZE, color='#222222')
    ax.set_xticks(xpos); ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlim(-0.6, len(KS) - 0.4)
    ax.set_xlabel('Rank $k$ (gallery $N$=5000)')
    ax.set_ylabel('Top-$k$ retrieval accuracy')
    _letter(ax, panel_letter)


# ── Panel c: in-vocab vs held-out ───────────────────────────────────────────────

def draw_zeroshot(ax, perp, ginf, patients, colors, panel_letter=None):
    """c — in-vocab vs held-out (zero-shot) median percentile rank; box + points.

    The claim this panel carries is that ZERO-SHOT RETRIEVAL BEATS CHANCE, so each
    box is tested against chance and neither is tested against the other.  An earlier
    version drew a paired in-vocab-vs-held-out bracket instead, which made the panel
    read as "zero-shot is worse" — true, and beside the point: the decoder is not
    supposed to match in-vocab on words it never trained on, it is supposed to beat
    chance on them.  In-vocab stays on the panel as the reference (and remains the
    split's sanity check: in-vocab BELOW held-out is what a correct hold-out looks
    like), it just is not the comparison being drawn."""
    inv = perp['median_percentile_invocab'].to_numpy(dtype=float)
    hld = perp['median_percentile_heldout'].to_numpy(dtype=float)
    ax.set_yscale('log')
    _box_points(ax, [0, 1], [inv, hld], patients, colors, width=0.5, seed=3)
    ax.axhline(CHANCE_PCT, ls='--', color=GREY, lw=1.0, zorder=1)
    # Anchored to the axes' right edge, not hanging past it: this panel is the last in its
    # row, so an overhanging label runs straight off the canvas once the figure is saved at
    # a fixed print width rather than with a tight bounding box.
    ax.text(1.5, CHANCE_PCT, 'chance ', color=GREY, fontsize=ANNOT_SIZE, va='bottom', ha='right')
    ax.set_xticks([0, 1])
    # Set a size explicitly rather than inheriting TICK_SIZE: this panel is the narrow one
    # in a 5:5:2 row (~0.76 in), and two category labels at the shared 7 pt collide.
    ax.set_xticklabels(['In-vocab', 'Held-out\n(zero-shot)'], fontsize=6.0)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0.005, 0.9)
    ax.set_yticks([0.01, 0.03, 0.1, 0.5])
    ax.set_yticklabels(['0.01', '0.03', '0.1', '0.5'])
    ax.get_yaxis().set_minor_locator(NullLocator())
    ax.set_ylabel('Median percentile rank')
    # Stars vs chance, one per box, read straight off the group inference rather than
    # recomputed here so the figure cannot drift from group_inference.csv.
    for xi, key in [(0, 'median_percentile_invocab'), (1, 'median_percentile_heldout')]:
        p = ginf.get(key, {}).get('p_value', np.nan)
        ax.text(xi, 0.20, _stars(p), ha='center', va='center', fontsize=STAR_SIZE, color='#222222')
    _letter(ax, panel_letter)


# ── Panel d: nDCG@100 vs its permutation null ────────────────────────────────────

def draw_ndcg(ax, perp, ginf, patients, colors, panel_letter=None):
    """d — nDCG@100 of the neural ranking vs its permutation null: two boxes with
    per-participant points, black mean line, and a group-level bracket + stars
    (Wilcoxon of observed-minus-null).

    The null is the only reference that makes this panel readable: absolute nDCG@100
    is ~0.65 but CHANCE nDCG is ~0.59-0.64, not 0, so the observed value alone says
    nothing.  Never plot the observed column without it."""
    obs = perp['graded_ndcg_mean'].to_numpy(dtype=float)
    null = perp['ndcg_null_mean'].to_numpy(dtype=float)
    _box_points(ax, [0, 1], [null, obs], patients, colors, width=0.5, seed=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Null', 'Neural\nretrieval'])
    ax.set_xlim(-0.5, 1.5)
    lo = min(null.min(), obs.min()); hi = max(null.max(), obs.max())
    pad = 0.06 * (hi - lo + 1e-9)
    ax.set_ylim(lo - pad, hi + 4.0 * pad)
    ax.set_ylabel('nDCG@100\n(independent WordNet grade)')
    gp = ginf.get('ndcg_vs_null', {}).get('p_value', np.nan)
    _sig_bracket(ax, 0, 1, hi + 1.4 * pad, _stars(gp))
    _letter(ax, panel_letter)


# ── Panels e/f: 2D MDS (cosine) semantic-neighbourhood showcases ─────────────────

def draw_mds(ax, mds, panel_letter=None):
    """e/f — MDS (cosine) of a showcase participant: each predicted word (blue, bold,
    placed at its own GloVe vector) sits beside the ground-truth word (black, bold)
    and their shared near-synonym neighbours (grey).

    Pairs are selected upstream to minimise the connector length drawn here, and
    homonyms are excluded from the bold words — see ``compute_extendability_data.py``.
    The per-pair distances are carried in the cache as ``pair_dist_2d`` /
    ``pair_cos_dist``; this function only draws them."""
    if mds is None:
        ax.text(0.5, 0.5, 'MDS cache missing\n(run compute_extendability_data.py)',
                ha='center', va='center', fontsize=7, color='#999999', transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([]); _letter(ax, panel_letter)
        return
    did = mds['display_id'].iloc[0]
    # Limits and box shape FIRST: label placement below measures rendered text boxes, and a
    # later change to either would invalidate every measurement.
    _square_limits(ax, mds)
    # faint predicted -> truth connector per showcase group
    for grp, g in mds.groupby('trial_group'):
        pr = g[g['role'] == 'predicted']; tr = g[g['role'] == 'truth']
        if len(pr) and len(tr):
            ax.plot([pr['x'].iloc[0], tr['x'].iloc[0]], [pr['y'].iloc[0], tr['y'].iloc[0]],
                    color='#cccccc', lw=0.8, zorder=1)
    # peripheral neighbours (grey text)
    for _, r in mds[mds['role'] == 'neighbor'].iterrows():
        ax.text(r['x'], r['y'], r['label'], fontsize=MDS_WORD_SIZE, color='#9a9a9a',
                ha='center', va='center', zorder=2)
    # Bold labels sit on top of the grey neighbour cloud, and pairs are now deliberately
    # close together, so both collide with surrounding text more than they used to.  A
    # white halo keeps them readable without moving any point or hiding a neighbour.
    halo = [pe.withStroke(linewidth=2.0, foreground='white')]

    # Each label is pushed directly AWAY from its partner.  The old fixed convention
    # (truth above, prediction below) collides whenever the truth sits below its own
    # prediction: the two labels then meet in the gap between the points.  That was rare
    # when pairs were chosen by similarity and is common now that they are chosen to be
    # close, e.g. watermelon/peach at 5.7% of the layout span.
    partner = {}
    for grp, g in mds.groupby('trial_group'):
        tr = g[g['role'] == 'truth']
        pr = g[g['role'] == 'predicted']
        if len(tr) and len(pr):
            partner[(grp, 'truth')] = (pr['x'].iloc[0], pr['y'].iloc[0])
            partner[(grp, 'predicted')] = (tr['x'].iloc[0], tr['y'].iloc[0])

    def _dir(r):
        p = partner.get((r['trial_group'], r['role']))
        dx, dy = (r['x'] - p[0], r['y'] - p[1]) if p else (0.0, 1.0)
        return (dx, dy) if (dx or dy) else (0.0, 1.0)

    def _place(dx, dy, pad=7.0):
        n = float(np.hypot(dx, dy))
        ox, oy = pad * dx / n, pad * dy / n
        ha = 'left' if ox > 0.4 * pad else ('right' if ox < -0.4 * pad else 'center')
        return (ox, oy), ha, ('bottom' if oy >= 0 else 'top')

    # Pushing each label away from its own partner still leaves labels of DIFFERENT pairs
    # free to land on each other — `apple` and `pear` did exactly that in NUE041 and read
    # as the single word "appear".  So each bold label is measured once placed, and if it
    # overlaps one already down, the offset is rotated around its anchor until it clears.
    # Deterministic and greedy: first free angle wins, and if none is free the preferred
    # direction is kept rather than the label being dropped.
    ax.figure.canvas.draw()
    rend = ax.figure.canvas.get_renderer()
    placed = []
    # Markers sit ABOVE the labels (zorder 8/9 vs 6/7).  A bold label carries a 2 pt white
    # halo, and with pairs this close a label routinely passes over a neighbouring pair's
    # dot — the halo then erases it.  That is how NUE027 came to show two blue words and
    # one blue dot: `peach`'s marker was under the `lime` label.  The marker is the datum;
    # the label is an annotation of it, so the marker wins.
    for role, color, zdot, ztxt in [('truth', 'black', 8, 6), ('predicted', BLUE, 9, 7)]:
        for _, r in mds[mds['role'] == role].iterrows():
            ax.plot(r['x'], r['y'], 'o', ms=3.5, color=color, zorder=zdot)
            dx, dy = _dir(r)
            base = np.arctan2(dy, dx)
            best = None
            for da in (0, 40, -40, 80, -80, 120, -120, 180):
                a = base + np.radians(da)
                xy, ha, va = _place(np.cos(a), np.sin(a))
                t = ax.annotate(r['label'], (r['x'], r['y']), textcoords='offset points',
                                xytext=xy, fontsize=MDS_BOLD_SIZE, color=color,
                                fontweight='bold', ha=ha, va=va, zorder=ztxt,
                                path_effects=halo)
                bb = t.get_window_extent(rend)
                if best is None:
                    best = t                      # fall back to the preferred direction
                if not any(bb.overlaps(q) for q in placed):
                    if best is not t:
                        best.remove()
                    placed.append(bb)
                    best = t
                    break
                if best is not t:
                    t.remove()
            else:
                placed.append(best.get_window_extent(rend))
    _corner_axes(ax)
    # Just the participant ID. apply_paper_style sets axes.titleweight='bold' globally, so
    # normal weight has to be asked for explicitly.
    ax.set_title(did, fontsize=MDS_TITLE_SIZE, fontweight='normal')
    _letter(ax, panel_letter)


def _square_limits(ax, mds):
    """Square box AND equal data ranges on both axes.

    The two MDS dimensions carry the same units, so stretching one against the other
    distorts exactly the distances this panel exists to show — a connector's drawn length
    would stop being proportional to the embedding distance it represents.  Limits are
    padded 10% so edge labels are not clipped, then centred so the equal ranges do not
    shift the cloud off-centre."""
    cx = 0.5 * (mds['x'].max() + mds['x'].min())
    cy = 0.5 * (mds['y'].max() + mds['y'].min())
    half = 0.55 * max(mds['x'].max() - mds['x'].min(), mds['y'].max() - mds['y'].min())
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_box_aspect(1.0)


def _corner_axes(ax, frac=0.15, pad=-0.035, size=5.0, color='#555555'):
    """Replace the axes frame with a small L-shaped direction marker, lower-left.

    Two reasons, one cosmetic and one not.  The MDS coordinates have no meaningful origin,
    unit or zero — only relative distance is interpretable — so a full frame with a spine
    running the length of the panel invites a reader to read positions off it that do not
    exist.  And at print size the neighbour cloud reaches the panel edge, where the frame
    and its axis labels collided with the words.  A corner marker states the two directions
    and then gets out of the way.

    The arms are equal fractions of the axes box, which is square and carries equal data
    ranges on both dimensions, so the marker is a true right angle at the data's own scale.
    ``pad`` is negative so the marker sits just OUTSIDE the point cloud: placed inside, it
    landed in a dense corner of NUE027 and had to fight `watermelon` and `pineapple` for
    the same space.
    """
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(''); ax.set_ylabel('')
    halo = [pe.withStroke(linewidth=1.8, foreground='white')]
    kw = dict(transform=ax.transAxes, color=color, clip_on=False, zorder=10)
    ax.plot([pad, pad + frac], [pad, pad], lw=0.9, solid_capstyle='round',
            path_effects=halo, **kw)
    ax.plot([pad, pad], [pad, pad + frac], lw=0.9, solid_capstyle='round',
            path_effects=halo, **kw)
    ax.text(pad + frac / 2, pad - 0.022, 'MDS dim 1', ha='center', va='top',
            fontsize=size, path_effects=halo, **kw)
    ax.text(pad - 0.022, pad + frac / 2, 'MDS dim 2', ha='center', va='bottom',
            rotation=90, fontsize=size, path_effects=halo, **kw)


def _letter(ax, letter):
    if letter is None:
        return
    ax.annotate(letter, xy=(0, 1), xycoords='axes fraction',
                xytext=(-24, 6), textcoords='offset points',
                fontsize=LETTER_SIZE, fontweight='bold', va='bottom', ha='left')


def _legend_handles(patients, colors):
    h = [mlines.Line2D([], [], color=colors[i], marker='o', ls='', ms=5, label=display_id(p))
         for i, p in enumerate(patients)]
    h.append(mlines.Line2D([], [], color='black', lw=2.5, marker='o', ms=5, label='mean'))
    h.append(mlines.Line2D([], [], color=GREY, lw=1.2, ls='--', label='chance'))
    return h


# ── Supplement 1: per-participant held-out trial distributions across N ──────────

def supp_heldout_distributions(heldout, patients, colors):
    if heldout is None:
        print("[extendability] S1 skipped — cache_heldout_trial_percentile_by_N.csv missing")
        return
    color_of = {p: colors[i] for i, p in enumerate(patients)}
    ncol = 4
    nrow = int(np.ceil(len(patients) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.6 * nrow), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    xpos = np.arange(len(NS))
    rng = _rng(2)
    for ai, p in enumerate(patients):
        ax = axes[ai]
        sub = heldout[heldout['patient'] == p]
        data = [sub[sub['N'] == n]['percentile'].to_numpy(dtype=float) for n in NS]
        if any(len(d) for d in data):
            ax.boxplot([d if len(d) else [np.nan] for d in data], positions=xpos, widths=0.6,
                       showfliers=False, patch_artist=True,
                       medianprops=dict(color='#333333', lw=1.2),
                       boxprops=dict(facecolor=BOX_FACE, edgecolor='#aaaaaa', lw=0.7),
                       whiskerprops=dict(color='#aaaaaa', lw=0.7),
                       capprops=dict(color='#aaaaaa', lw=0.7))
            for xi, d in enumerate(data):
                jx = xi + (rng.random(len(d)) - 0.5) * 0.3
                ax.plot(jx, d, 'o', ms=2.2, color=color_of[p], alpha=0.55, mew=0, zorder=3)
        ax.axhline(CHANCE_PCT, ls='--', color=GREY, lw=0.8)
        ax.set_title(display_id(p), fontsize=8, fontweight='bold')
        ax.set_xticks(xpos); ax.set_xticklabels([str(n) for n in NS], fontsize=6, rotation=45)
        ax.set_ylim(-0.02, 0.62)
    for ai in range(len(patients), len(axes)):
        axes[ai].axis('off')
    fig.text(0.5, 0.015, 'Gallery size $N$ (words)', ha='center', fontsize=9)
    fig.text(0.005, 0.5, 'Per-trial percentile rank (held-out trials)', va='center',
             rotation='vertical', fontsize=9)
    fig.suptitle('Held-out (zero-shot) per-trial retrieval distributions by participant',
                 fontsize=10, fontweight='bold')
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.97))
    _save(fig, os.path.join(FIG_DIR, 'S1_heldout_trial_distributions'))


# ── Supplement 2: qualitative best-case table (HTML + CSV) ───────────────────────

def supp_bestcase_table(best):
    if best is None:
        print("[extendability] S2 skipped — cache_qualitative_bestcases.csv missing")
        return
    b = best.copy()
    b.to_csv(os.path.join(SRC_DIR, 'S2_qualitative_bestcases.csv'), index=False)

    def _row_html(r):
        tops = []
        grades = str(r['grades']).split(';')
        for j, col in enumerate(['top1', 'top2', 'top3', 'top4', 'top5']):
            w = r[col]
            g = grades[j] if j < len(grades) else ''
            hit = (str(w).lower() == str(r['true_word']).lower())
            style = 'font-weight:bold;color:#1f6f1f;' if hit else 'color:#333;'
            tops.append(f'<td style="{style}">{w}<br><span style="color:#999;font-size:11px">{g}</span></td>')
        return (f"<tr><td>{r['display_id']}</td>"
                f"<td style='font-weight:bold'>{r['true_word']}</td>"
                f"<td style='color:#666'>{r['category']}</td>"
                f"<td style='text-align:right'>{r['rank']}</td>"
                f"<td style='text-align:right'>{r['near_miss_sim']:.3f}</td>"
                f"<td style='text-align:right'>{r['ndcg']:.3f}</td>"
                + ''.join(tops) + "</tr>")

    head = ("<th>ID</th><th>true word</th><th>category</th><th>rank</th>"
            "<th>near-miss<br>sim</th><th>nDCG@100</th>"
            "<th>top-1</th><th>top-2</th><th>top-3</th><th>top-4</th><th>top-5</th>")
    rows = '\n'.join(_row_html(r) for _, r in b.iterrows())
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Extendability — qualitative best cases</title>
<style>
 body{{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#222}}
 h2{{font-size:18px}} p{{color:#555;max-width:820px;font-size:13px}}
 table{{border-collapse:collapse;font-size:13px}}
 th,td{{border:1px solid #ddd;padding:5px 8px;vertical-align:top}}
 th{{background:#f3f3f3;text-align:left}}
</style></head><body>
<h2>Qualitative best-case retrievals (picture naming)</h2>
<p>Per participant, the words whose mean predicted embedding retrieved the most
semantically related neighbourhood (highest top-10 Wu–Palmer near-miss similarity),
one per semantic category. Each cell shows a retrieved word and its independent
WordNet Wu–Palmer similarity to the true word; green bold marks the exact true word.</p>
<table><thead><tr>{head}</tr></thead><tbody>
{rows}
</tbody></table></body></html>"""
    with open(os.path.join(FIG_DIR, 'S2_qualitative_bestcases.html'), 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[extendability] S2 -> {len(b)} best-case rows")


# ── Source data ─────────────────────────────────────────────────────────────────

def write_source_data(perp, sweep, ginf, patients):
    os.makedirs(SRC_DIR, exist_ok=True)
    did = {p: display_id(p) for p in patients}

    # Panel a + robustness: full N x variant sweep, per participant.
    s = sweep.copy(); s.insert(0, 'display_id', s['patient'].map(did))
    keep = ['display_id', 'patient', 'task', 'variant', 'N',
            'median_percentile', 'top1', 'top5', 'top10', 'top50', 'top100',
            'median_rank', 'chance_median_percentile']
    s[[c for c in keep if c in s.columns]].to_csv(
        os.path.join(SRC_DIR, 'panel_a_sweep_per_participant.csv'), index=False)

    # Panel a group mean±sem + per-N Wilcoxon vs chance.
    rows = []
    sub = sweep[sweep['variant'] == HEADLINE_VARIANT]
    for n in NS:
        vals = sub[sub['N'] == n]['median_percentile'].values.astype(float)
        pv, _ = _wilcoxon(vals, CHANCE_PCT, 'less')
        rows.append(dict(N=n, variant=HEADLINE_VARIANT,
                         median_percentile_mean=float(np.mean(vals)),
                         median_percentile_sem=_sem(vals),
                         wilcoxon_p_vs_chance=pv, sig=_stars(pv),
                         chance_median_percentile=CHANCE_PCT))
    pd.DataFrame(rows).to_csv(os.path.join(SRC_DIR, 'panel_a_group_mean_sem.csv'), index=False)

    # Panel b: CMC at N=5000 — per-participant top-k + chance + per-k Wilcoxon.
    rows = []
    for p in patients:
        r = perp[perp['patient'] == p].iloc[0]
        d = dict(display_id=did[p], patient=p, N=HEADLINE_N, n_trials=int(r['n_trials']),
                 median_rank=float(r['median_rank_all']))
        for k in KS:
            d[f'top{k}'] = float(r[f'top{k}_all']); d[f'chance_top{k}'] = k / float(HEADLINE_N)
        rows.append(d)
    dfb = pd.DataFrame(rows)
    dfb.to_csv(os.path.join(SRC_DIR, 'panel_b_cmc_N5000.csv'), index=False)
    sig_rows = []
    for k in KS:
        pv, _ = _wilcoxon(dfb[f'top{k}'].values, k / float(HEADLINE_N), 'greater')
        sig_rows.append(dict(k=k, chance=k / float(HEADLINE_N),
                             topk_mean=float(dfb[f'top{k}'].mean()), wilcoxon_p=pv, sig=_stars(pv)))
    pd.DataFrame(sig_rows).to_csv(os.path.join(SRC_DIR, 'panel_b_significance.csv'), index=False)

    # Panel c: zero-shot — in-vocab vs held-out + counts + perm p + paired test.
    c = perp[['patient', 'n_trials', 'n_held_out',
              'median_percentile_all', 'median_percentile_invocab', 'median_percentile_heldout',
              'perm_p_median_percentile_all']].copy()
    c.insert(0, 'display_id', c['patient'].map(did))
    c['chance_median_percentile'] = CHANCE_PCT
    c.rename(columns={'perm_p_median_percentile_all': 'perm_p_all'}, inplace=True)
    c.to_csv(os.path.join(SRC_DIR, 'panel_c_zeroshot.csv'), index=False)

    # Panel d: nDCG@100 — obs vs null, within-participant perm p.  (The Wu-Palmer
    # neighbour-similarity panel that used to sit here was cut 2026-08-12; nothing
    # plots that metric now, so it no longer gets a source-data file.  Its group
    # result survives in group_inference.csv below, and the full per-participant
    # record is untouched upstream in
    # figures/open_vocab_retrieval/source_data/per_patient_metrics_picture_naming.csv.)
    d = perp[['patient', 'graded_ndcg_mean', 'ndcg_null_mean', 'perm_p_ndcg']].copy()
    d.insert(0, 'display_id', d['patient'].map(did))
    d['ndcg_delta'] = d['graded_ndcg_mean'] - d['ndcg_null_mean']
    d.rename(columns={'graded_ndcg_mean': 'ndcg_obs', 'ndcg_null_mean': 'ndcg_null'}, inplace=True)
    d.to_csv(os.path.join(SRC_DIR, 'panel_d_ndcg.csv'), index=False)

    # Group-level inference (tidy from the pipeline JSON) — Results text.
    grows = []
    for key in ['median_percentile_all', 'median_percentile_invocab', 'median_percentile_heldout']:
        g = ginf[key]
        grows.append(dict(metric=key, n=g['n'], median=g['median'], chance=g['chance'],
                          ci_mean=g['ci_mean'], ci_lo=g['ci_lo'], ci_hi=g['ci_hi'],
                          wilcoxon_p=g['p_value'], test='Wilcoxon signed-rank vs chance (one-sided)'))
    for key, lab in [('near_miss_vs_null', 'near_miss_obs_minus_null'),
                     ('ndcg_vs_null', 'ndcg_obs_minus_null')]:
        if key in ginf:
            nm = ginf[key]
            grows.append(dict(metric=lab, n=nm['n'], median=nm['median'], chance=nm['chance'],
                              ci_mean=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                              wilcoxon_p=nm['p_value'], test='Wilcoxon signed-rank vs 0 (one-sided)'))
    nd = ginf['ndcg']
    grows.append(dict(metric='ndcg_at_100', n=nd['n'], median=np.nan, chance=np.nan,
                      ci_mean=nd['mean'], ci_lo=nd['lo'], ci_hi=nd['hi'],
                      wilcoxon_p=np.nan, test='bootstrap 95% CI (descriptive)'))
    pd.DataFrame(grows).to_csv(os.path.join(SRC_DIR, 'group_inference.csv'), index=False)


# ── Caption ─────────────────────────────────────────────────────────────────────

CAPTION = """# Figure caption — Extendability of the regression-and-retrieval decoder

The paragraph below is the caption as it should appear in the manuscript — copy it whole.
Everything under "Notes" is provenance for this repository and is not part of the caption.
Generated by `extendability_panels.py`; keep it in the Nature legend style recorded in
`figures_for_paper/README.md` §4.

**Figure | Extendability of the regression-and-retrieval decoder to an open vocabulary.**
Open-vocabulary word retrieval during picture naming (N = {N} participants). The decoder predicts a
GloVe embedding per trial; that vector is ranked by cosine similarity against a gallery of {HNC}
words — the stimulus words plus part-of-speech- and frequency-matched distractors never presented to
any participant — and the rank of the true word is the score. **a** Median percentile rank (rank/N;
lower is better) versus gallery size N. **b** Top-k retrieval accuracy versus rank k at N = {HNC}.
**c** Median percentile rank at N = {HNC} for words seen in training (in-vocab) and words held
entirely out of training (held-out, zero-shot; 30% of unique words withheld per cross-validation
split). **d** nDCG@100 of the neural ranking against a permutation null in which each trial keeps its
retrieved ranking but is graded against a permuted true word; absolute nDCG is uninterpretable
without this null, because chance nDCG is approximately 0.6 rather than 0. **e** Two-dimensional
metric multidimensional scaling of cosine distances between word embeddings, shown for the two
participants with the highest top-10 retrieval accuracy: ground-truth word (black, bold), the word
the decoder retrieved first (blue, bold, drawn at its own embedding), their nearest gallery
neighbours (grey), and a line joining each pair. Pairs are selected to minimise that line's length;
homonyms are excluded from bold words. Both maps use equal ranges on the two dimensions, so a
line's length is proportional to the embedding distance. In **a**–**d**: box,
interquartile range and median across participants; coloured points, individual participants, one
fixed colour throughout; black line and markers, across-participant mean; dashed grey line, chance
(median percentile rank 0.5; top-k accuracy k/N). Asterisks: one-sided Wilcoxon signed-rank against
chance in **a** and **b**, against chance for each group separately in **c**, and of the
observed-minus-null difference in **d**. {PSENT} Channels: the 13-region temporal-parietal
whitelist on `nmm_roi`. Participants are identified by display ID.

## Notes — not part of the caption

- Figure: `00_extendability_combined.{{png,pdf}}`, rendered by `extendability_panels.py` from
  `source_data/*.csv` only. Per-panel standalones: `01_scaling_median_percentile` (**a**),
  `02_cmc_N5000` (**b**), `03_zeroshot_invocab_vs_heldout` (**c**), `04_ndcg_vs_null` (**d**),
  `05_mds_neighbourhood_left` and `06_mds_neighbourhood_right` (the two maps of **e**).
- Plotted values: `panel_a_sweep_per_participant.csv` + `panel_a_group_mean_sem.csv`,
  `panel_b_cmc_N5000.csv` + `panel_b_significance.csv`, `panel_c_zeroshot.csv`,
  `panel_d_ndcg.csv`, `cache_panelf_mds.csv` and `cache_panelf_AA.csv` (**e**, left and
  right). Group-level tests are tidied into `group_inference.csv`.
- **e** is NUE027 (left) and NUE041 (right), the top two on `top10_all` (0.238 and 0.266).
  The criterion is stated in the caption because the ranking is **not stable across
  metrics**: on median percentile rank and top-100 accuracy the order is NUE041, then
  NUE050, with NUE027 only third. A reader should not have to guess which sense of "best"
  is meant. NUE041 was promoted out of S3 on 2026-08-13, and NUE031 returned to it.
- Pair selection minimises the 2D connector length, which does not exist until a layout does.
  It is resolved by iterating select → lay out → re-select, keeping the best configuration
  that was actually measured feasible. A pair may not exceed 25% of the layout diagonal, so
  the ten-pair cap does not act as a quota — each map carries the pairs that qualify and no
  filler. Per-pair distances are written to the caches as
  `pair_dist_2d`, `pair_dist_frac` and `pair_cos_dist` so the claim can be checked rather
  than taken on trust.
- Pairs are also capped at two per semantic category. Ranking by closeness favours the
  densest region of the embedding, and without that cap NUE027 drew six fruit words into one
  corner where the labels became unreadable.
- **A Wu–Palmer neighbour-similarity panel sat at d until 2026-08-12** and was cut as
  editorially redundant with nDCG. The metric is untouched upstream; its group result is
  still the `near_miss_obs_minus_null` row of `group_inference.csv`.
- Homonyms excluded from the bold pairs are `bat, mouse, nail, nut, fan` — the `homonym`
  column of `data_archive/wordset picture naming expanded.xlsx`, read literally. That column
  is not a complete inventory of ambiguous words (it leaves `spring`, `park` and `date`
  unflagged despite splitting them into senses); taking it at face value is a deliberate
  choice, not an oversight.
- Supplements: S1, per-participant held-out per-trial percentile distributions across N;
  S2, qualitative best-case retrievals; S3 and S4, further MDS showcases (NUE031, NUE036).
  None has its own caption yet.
"""


def _caption_p_values(perp, sweep, ginf, patients):
    """Every group P value the caption speaks for, so the text cannot outlive the run.

    Covers the per-N tests of **a**, the per-k tests of **b**, the two vs-chance tests of
    **c** and the null contrast of **d** — i.e. exactly the asterisks drawn on the figure."""
    ps = []
    sub = sweep[sweep['variant'] == HEADLINE_VARIANT]
    for n in NS:
        p, _ = _wilcoxon(sub[sub['N'] == n]['median_percentile'].values.astype(float),
                         CHANCE_PCT, 'less')
        ps.append(p)
    for k in KS:
        p, _ = _wilcoxon(perp[f'top{k}_all'].to_numpy(dtype=float),
                         k / float(HEADLINE_N), 'greater')
        ps.append(p)
    for key in ['median_percentile_invocab', 'median_percentile_heldout', 'ndcg_vs_null']:
        ps.append(ginf.get(key, {}).get('p_value', np.nan))
    return [p for p in ps if not np.isnan(p)]


def _fmt_p(p):
    """P value in the manuscript's style: 6.1 x 10^-5 as '6.1 × 10⁻⁵'."""
    sup = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')
    mant, exp = f'{p:.1e}'.split('e')
    return f'{mant} × 10{str(int(exp)).translate(sup)}'


def write_caption(patients, perp, sweep, ginf):
    ps = _caption_p_values(perp, sweep, ginf, patients)
    pmax, pmin = max(ps), min(ps)
    at_floor = np.allclose(ps, pmin)
    # Say plainly when every test has bottomed out.  A one-sided Wilcoxon at n = 14 cannot
    # return anything below 2^-14, so quoting that number without saying it is the floor
    # would imply a graded effect the test is incapable of expressing.
    psent = (f'Every one of these tests returns P = {_fmt_p(pmin)}, the smallest value '
             f'attainable at n = {len(patients)}, so the asterisks mark a floor rather than '
             f'a graded effect.' if at_floor else
             f'All P ≤ {_fmt_p(pmax)}; the smallest value attainable at n = {len(patients)} '
             f'is {_fmt_p(2.0 ** -len(patients))}.')
    txt = CAPTION.format(
        N=len(patients), HN=HEADLINE_N, HNC=f'{HEADLINE_N:,}', PSENT=psent)
    with open(os.path.join(FIG_DIR, 'caption.md'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(txt)
    words = len(txt.split('## Notes')[0].split('**Figure |')[1].split())
    print(f"[extendability] caption {words} words (Nature cap 350); "
          f"group P {'all at floor' if at_floor else 'range'} "
          f"{_fmt_p(pmin)}..{_fmt_p(pmax)}")


# ── Orchestration ───────────────────────────────────────────────────────────────

def _save(fig, stem, dpi=200, tight=True):
    """Write both formats.  ``tight=False`` keeps the canvas at exactly ``figsize``.

    bbox_inches='tight' grows the saved canvas to enclose anything drawn outside the axes
    — here the panel letters and the corner direction markers — so the combined figure came
    out 194 mm wide from a 183 mm figsize.  A journal then scales it 0.94x to fit the
    column, which silently drops the 5.2 pt neighbour words below the ~5 pt floor.  The
    combined figure therefore saves untight, with margins reserved by subplots_adjust; the
    standalones keep 'tight' because they are working views with no fixed print width.
    """
    kw = dict(bbox_inches='tight') if tight else {}
    fig.savefig(stem + '.pdf', **kw)
    fig.savefig(stem + '.png', dpi=dpi, **kw)
    plt.close(fig)


def generate():
    os.makedirs(SRC_DIR, exist_ok=True)
    perp, sweep, ginf, patients = load_inputs()
    colors = assign_colors(patients)
    mds_main = [_cache(name) for name, _ in PANELF_MAIN]
    heldout = _cache('cache_heldout_trial_percentile_by_N.csv')
    best = _cache('cache_qualitative_bestcases.csv')

    # Standalone panels, numbered by panel order (house rule).
    specs = [
        ('01_scaling_median_percentile', (4.3, 3.4), lambda ax: draw_scaling(ax, sweep, patients, colors)),
        ('02_cmc_N5000', (4.3, 3.4), lambda ax: draw_cmc(ax, perp, patients, colors)),
        ('03_zeroshot_invocab_vs_heldout', (4.0, 3.4), lambda ax: draw_zeroshot(ax, perp, ginf, patients, colors)),
        ('04_ndcg_vs_null', (4.0, 3.4), lambda ax: draw_ndcg(ax, perp, ginf, patients, colors)),
        ('05_mds_neighbourhood_left', (5.0, 5.0), lambda ax: draw_mds(ax, mds_main[0])),
        ('06_mds_neighbourhood_right', (5.0, 5.0), lambda ax: draw_mds(ax, mds_main[1])),
    ]
    for stem, size, fn in specs:
        fig, ax = plt.subplots(figsize=size); fn(ax)
        fig.tight_layout(); _save(fig, os.path.join(FIG_DIR, stem))

    # Combined layout.  Row 1 = a, b, c at 5:5:2 — the width each panel needs is set by how
    # many positions it plots (five gallery sizes, five ranks, two vocabulary conditions),
    # so a fixed equal split starved a and b to pad c.  Row 2 = d (two boxes, so the same
    # narrow 2) + panel e's two SQUARE maps.  The row heights follow from that: the maps are
    # squares whose side is their column width, so row 2 has to be tall enough to hold one.
    fig = plt.figure(figsize=(FIG_W_DOUBLE, 5.8))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.35], hspace=0.34)
    top = outer[0].subgridspec(1, 3, width_ratios=[5, 5, 2], wspace=0.32)
    bot = outer[1].subgridspec(1, 3, width_ratios=[2, 5, 5], wspace=0.30)
    draw_scaling(fig.add_subplot(top[0, 0]), sweep, patients, colors, panel_letter='a')
    draw_cmc(fig.add_subplot(top[0, 1]), perp, patients, colors, panel_letter='b')
    draw_zeroshot(fig.add_subplot(top[0, 2]), perp, ginf, patients, colors, panel_letter='c')
    draw_ndcg(fig.add_subplot(bot[0, 0]), perp, ginf, patients, colors, panel_letter='d')
    for col, (df, (_, letter)) in enumerate(zip(mds_main, PANELF_MAIN), start=1):
        draw_mds(fig.add_subplot(bot[0, col]), df, panel_letter=letter)
    fig.legend(handles=_legend_handles(patients, colors), ncol=8, loc='lower center',
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.005))
    # Fixed-aspect axes are not compatible with tight_layout, so place the grid explicitly.
    # These margins are load-bearing: the figure saves at a fixed print width (no tight
    # bounding box), so anything drawn outside an axes — d's two-line y-label, c's chance
    # annotation, a bold MDS word pushed to the right edge — is clipped by the canvas
    # rather than accommodated. Re-check the edges after changing any of them.
    fig.subplots_adjust(left=0.092, right=0.955, top=0.945, bottom=0.115)
    _save(fig, os.path.join(FIG_DIR, '00_extendability_combined'), dpi=300, tight=False)

    # Supplements (not in the combined figure)
    supp_heldout_distributions(heldout, patients, colors)
    supp_bestcase_table(best)
    # Supplementary panel-f showcases for the next-best participants
    for pat, slabel in PANELF_SUPP:
        df = _cache(f'cache_panelf_{pat}.csv')
        if df is None:
            print(f"[extendability] {slabel} {pat} skipped — cache_panelf_{pat}.csv missing")
            continue
        figx, axx = plt.subplots(figsize=(5.0, 5.0)); draw_mds(axx, df)
        figx.tight_layout(); _save(figx, os.path.join(FIG_DIR, f'{slabel}_mds_neighbourhood_{pat}'))

    write_source_data(perp, sweep, ginf, patients)
    write_caption(patients, perp, sweep, ginf)

    # Results-text numbers
    def gmean(col):
        v = perp[col].values.astype(float); return float(np.mean(v)), _sem(v)
    print("[extendability] figures + caption ->", FIG_DIR)
    print("[extendability] source data       ->", SRC_DIR)
    print(f"  participants: {[display_id(p) for p in patients]}")
    print(f"  median %rank  all   median={ginf['median_percentile_all']['median']:.4f} "
          f"Wilcoxon p={ginf['median_percentile_all']['p_value']:.2e}")
    print(f"  median %rank  invoc median={ginf['median_percentile_invocab']['median']:.4f} "
          f"heldout median={ginf['median_percentile_heldout']['median']:.4f} "
          f"(p={ginf['median_percentile_heldout']['p_value']:.2e})")
    m, s = gmean('top10_all'); print(f"  top-10 (N=5000) mean={m:.3f} +/- {s:.3f}")
    m, s = gmean('top100_all'); print(f"  top-100(N=5000) mean={m:.3f} +/- {s:.3f}")
    print(f"  nDCG@100 mean={ginf['ndcg']['mean']:.3f} CI[{ginf['ndcg']['lo']:.3f},{ginf['ndcg']['hi']:.3f}]")
    print(f"  near-miss delta Wilcoxon p={ginf['near_miss_vs_null']['p_value']:.2e}")
    if 'ndcg_vs_null' in ginf:
        print(f"  nDCG delta Wilcoxon p={ginf['ndcg_vs_null']['p_value']:.2e}")


if __name__ == '__main__':
    generate()
