# -*- coding: utf-8 -*-
"""
figures_for_paper/extendability/compute_extendability_data.py
=============================================================
Heavy compute step for the extendability figure (panels e/f + supplements).
Rebuilds the matched galleries from GloVe and the already-computed per-trial
predicted embeddings, then writes three cache CSVs that the (lightweight,
CSV-only) ``extendability_panels.py`` reads.  Run once, in the Speech conda env:

    C:/Users/Owner/miniconda3/envs/Speech/python.exe \
        figures_for_paper/extendability/compute_extendability_data.py

Outputs -> figures_for_paper/extendability/source_data/
  1. cache_heldout_trial_percentile_by_N.csv   (supp-1: per held-out trial, per N)
  2. cache_panelf_mds.csv                       (panel e: 2D MDS-cosine showcase, best patient)
     cache_panelf_RB.csv                        (panel f: same, second-best patient)
     cache_panelf_{AA,WBH}.csv                  (supps S3/S4)
  3. cache_qualitative_bestcases.csv            (supp-2: best cases per patient)

This re-scores predictions against freshly built galleries; it does NOT re-fit
any model.  Reuses the open_vocab_retrieval package (gallery / retrieval /
metrics / relevance) so the gallery construction is identical to the shipped run.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS_ROOT = os.path.dirname(HERE)
MAIN_DIR = os.path.dirname(FIGS_ROOT)                       # …/main
if MAIN_DIR not in sys.path:
    sys.path.insert(0, MAIN_DIR)
sys.path.insert(0, FIGS_ROOT)                              # paper_common (display IDs)

from paper_common import display_id                        # noqa: E402
from analysis.open_vocab_retrieval import gallery as gallery_mod          # noqa: E402
from analysis.open_vocab_retrieval import retrieval, metrics, relevance   # noqa: E402

OPENVOCAB_SRC = os.path.join(MAIN_DIR, 'figures', 'open_vocab_retrieval', 'source_data')
SRC_DIR = os.path.join(HERE, 'source_data')

TASK = 'picture_naming'
NS = [200, 500, 1000, 2000, 5000]
HEADLINE_N = 5000
# The two showcase participants of panel e, chosen by TOP-10 RETRIEVAL ACCURACY at N=5000
# (`top10_all`): AA 0.266 and VB 0.238 are the top two.  Named by POSITION, not rank — AA
# outranks VB on this metric and sits on the right — because "best participant" is not a
# well-defined phrase here and an earlier constant called BEST_PATIENT made a claim the
# data did not support.  The ranking is not stable across metrics: on median percentile
# rank and top-100 accuracy it is AA, then PV (0.0136 / 0.569), with VB only third.  A
# still earlier revision ranked by delta-vs-null, which put VB first and RB second; RB is
# now a supplementary showcase.  Whichever is used, the caption must say which.
SHOWCASE_LEFT = 'VB'         # -> cache_panelf_mds.csv (historical file name, kept)
SHOWCASE_RIGHT = 'AA'        # -> cache_panelf_AA.csv
PANELF_EXTRA = ['RB', 'WBH']        # supplementary showcases only (S3, S4)
N_SHOWCASE = 10              # max showcase words per panel (a cap, not a quota)
N_NEIGHBORS = 22             # peripheral neighbours per showcase word
# Cap showcase pairs per semantic category.  Lowered 3 -> 2 on 2026-08-12: ranking by
# closeness systematically favours the densest region of the embedding, so a cap of 3 put
# six fruit words in one corner of NUE027 and the labels became unreadable.  The cap is
# what keeps "diverse semantic category" true of the panel, and it matters more under a
# distance-first ranking than it did under the old similarity-first one.
MAX_PER_CAT = 2
MIN_NEAR_MISS_SIM = 0.55     # a showcase must have a genuinely coherent neighbourhood
MAX_SELECT_ITERS = 8         # select -> lay out -> re-select, at most this many rounds
MIN_SHOWCASE = 3             # never strip a panel below this, even to honour the cutoff
# A showcase pair must land within this fraction of the layout's own diagonal.  Without a
# hard cutoff the greedy fill treats N_SHOWCASE as a quota and pads the panel with whatever
# is left once the close pairs run out — which is how a first pass produced sport->park at
# 38% of the span and knight->lock at 60%, connectors long enough to argue the opposite of
# the panel's point.  Distances are kept as a FRACTION of the span, not in layout units,
# because MDS scale is arbitrary: a scale-free criterion is stable under the re-layout that
# every drop triggers, an absolute one chases its own tail.
MAX_PAIR_DIST_FRAC = 0.25

# Homonymous stimulus words are excluded from the bold showcase pairs: with two
# unrelated senses a "near-miss" is ambiguous by construction (is `nail` -> `match`
# a hit on the fastener sense or a miss on the finger sense?).  Source = the
# `homonym` column of `data_archive/wordset picture naming expanded.xlsx`, which
# flags the numbered sense keys bat1/bat2/mouse1/mouse2/nail1/nail2/nut1/nut2/fan2
# -> these five lemmas.  Pinned here rather than read at build time because
# `data_archive/` is git-ignored (.gitignore pattern `data*`), so the xlsx cannot be
# a dependency of a tracked figure.  NB the column is not a complete inventory of
# ambiguous items: the same file splits spring1/spring2, park1/park2 and date1/date2
# into senses but leaves their homonym cell blank.  Taking the column literally is a
# deliberate choice (Alec, 2026-08-12), not an oversight.
HOMONYM_WORDS = frozenset({'bat', 'mouse', 'nail', 'nut', 'fan'})
N_BEST_PER_PATIENT = 4       # rows per patient in the qualitative best-case table
BESTCASE_RANK_MAX = 50       # a "best case" must also retrieve the true word well
CENTER = True                # canonical mean-centred cosine (matches the run)
PE_COLS = [f'pe{j}' for j in range(300)]


# ── Data loading ──────────────────────────────────────────────────────────

def load_trials():
    """Per-patient arrays from the shipped per-trial predictions CSV."""
    df = pd.read_csv(os.path.join(OPENVOCAB_SRC, f'trial_predictions_{TASK}.csv'))
    per = {}
    for pat, g in df.groupby('patient', sort=False):
        per[pat] = dict(
            pred_emb=g[PE_COLS].to_numpy(dtype=np.float64),
            true_word=g['true_word'].astype(str).to_numpy(),
            category=g['category'].astype(str).to_numpy(),
            is_held_out=g['is_held_out'].to_numpy(dtype=bool),
        )
    patients = list(dict.fromkeys(df['patient']))   # first-appearance order
    stim_words = sorted(set(df['true_word'].astype(str)))
    return per, patients, stim_words


def build_matched_gallery(glove, stim_words, n):
    """Matched gallery reproducing the shipped run (concreteness/subtlex=None)."""
    return gallery_mod.build_gallery(glove, stim_words, n=n, variant='matched',
                                     concreteness=None, subtlex=None)


# ── 1. Held-out per-trial percentile across gallery sizes N ────────────────

def compute_heldout_percentiles(per, patients, stim_words, glove):
    rows = []
    for n in NS:
        gal = build_matched_gallery(glove, stim_words, n)
        for pat in patients:
            d = per[pat]
            ho = d['is_held_out']
            if not ho.any():
                continue
            sims = retrieval.similarity_matrix(d['pred_emb'][ho], gal.emb, center=CENTER)
            tidx = retrieval.true_indices(d['true_word'][ho], gal.word_to_index)
            rank = retrieval.compute_ranks(sims, tidx)
            valid = rank > 0
            pct = rank[valid] / float(gal.N)
            words = d['true_word'][ho][valid]
            cats = d['category'][ho][valid]
            did = display_id(pat)
            for w, c, p in zip(words, cats, pct):
                rows.append(dict(display_id=did, patient=pat, N=int(n),
                                 true_word=w, category=c, percentile=float(p)))
        print(f"  [heldout%] N={n}: {sum(1 for r in rows if r['N']==n)} trials", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(SRC_DIR, 'cache_heldout_trial_percentile_by_N.csv'), index=False)
    print(f"  wrote cache_heldout_trial_percentile_by_N.csv ({len(out)} rows)", flush=True)
    return out


# ── per-word mean predictions (denoised query) ────────────────────────────

def per_word_mean_predictions(d):
    """Mean predicted embedding per unique true word for one patient.

    Returns (words, emb (W,300), category-per-word). Denoises the single-trial
    query the way the qualitative table does, so a showcase word is representative
    rather than the noisiest trial."""
    words = d['true_word']
    uniq = list(dict.fromkeys(words))
    emb = np.zeros((len(uniq), 300), dtype=np.float64)
    cat = []
    for i, w in enumerate(uniq):
        m = words == w
        emb[i] = d['pred_emb'][m].mean(axis=0)
        cat.append(d['category'][m][0])
    return uniq, emb, np.array(cat)


def word_retrieval_grades(words, emb, gal, rel_fn, want_ndcg=False):
    """For each per-word mean prediction: nearest gallery word (predicted word),
    true-word rank, top-10 Wu-Palmer near-miss similarity, and (optionally)
    nDCG@100.  Returns a list of dicts (order == input words)."""
    sims = retrieval.similarity_matrix(emb, gal.emb, center=CENTER)
    order = retrieval.ranked_indices(sims)              # (W, N) desc
    tidx = retrieval.true_indices(words, gal.word_to_index)
    rank = retrieval.compute_ranks(sims, tidx)
    recs = []
    for i, w in enumerate(words):
        order_row = order[i]
        pred_word = gal.words[int(order_row[0])]
        nm = metrics.near_miss_similarity(order_row, w, gal.words, rel_fn, k=10)
        rec = dict(true_word=w, pred_word=pred_word, rank=int(rank[i]),
                   in_gallery=bool(tidx[i] >= 0), near_miss_sim=float(nm),
                   order_row=order_row)
        if want_ndcg:
            rec['ndcg'] = float(metrics.ndcg_independent(order_row, w, gal.words, rel_fn, k=100))
        recs.append(rec)
    return recs


# ── 2. MDS showcase panels (best + second-best patient, and supplements) ──

def _layout(rows, glove, n_init=4):
    """Metric MDS (cosine) over the GloVe vectors of ``rows`` -> (n, 2) coords.

    ``n_init=1`` is used for the throwaway candidate-pool layout, which only has to
    rank pairs; the rendered layout keeps the full 4 restarts."""
    from sklearn.manifold import MDS
    from sklearn.metrics.pairwise import cosine_distances
    X = np.vstack([gallery_mod.glove_vector(glove, r['label']) for r in rows])
    D = cosine_distances(X)
    return MDS(n_components=2, dissimilarity='precomputed', random_state=0,
               n_init=n_init, max_iter=400).fit_transform(D)


def _showcase_points(chosen, gal, word_cat):
    """Unique real-word points for a set of pairs: truth + predicted (both bold)
    + the top-N retrieved neighbours (grey), deduped globally so no word appears
    twice.  Group key is ``{index}:{true_word}``, matching the plotter."""
    bold = {w for r in chosen for w in (r['true_word'], r['pred_word'])}
    rows, seen = [], set()

    def add(word, role, group, cat_):
        if word in seen:
            return
        seen.add(word)
        rows.append(dict(label=word, role=role, trial_group=group, category=cat_))

    for gi, r in enumerate(chosen):
        tw, pw = r['true_word'], r['pred_word']
        grp = f"{gi}:{tw}"
        add(tw, 'truth', grp, word_cat[tw])
        add(pw, 'predicted', grp, word_cat[tw])
        cnt = 0
        for j in r['order_row']:
            gw = gal.words[int(j)]
            if gw in bold:                # skip words that are a showcase truth/pred
                continue
            add(gw, 'neighbor', grp, '')
            cnt += 1
            if cnt >= N_NEIGHBORS:
                break
    return rows


def _span(xy):
    """Diagonal of the layout's bounding box — the scale a connector is judged against."""
    return float(np.hypot(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))


def _pair_dists_2d(chosen, rows, xy):
    """Euclidean distance in the 2D layout between each pair's truth and predicted
    point — the quantity the showcase is selected to minimise, and the one a reader
    actually sees as the length of the connector line.  Returned as (absolute,
    fraction-of-span); selection uses the fraction, the caption quotes the fraction,
    and the absolute value is kept only because it is what you would measure off the
    axes."""
    pos = {r['label']: xy[i] for i, r in enumerate(rows)}
    absd = [float(np.linalg.norm(pos[r['true_word']] - pos[r['pred_word']]))
            for r in chosen]
    span = _span(xy)
    return absd, [d / span for d in absd]


def _pair_dists_cos(chosen, glove):
    """Cosine distance between the two GloVe vectors of each pair.  This is what the
    MDS is approximating, so it is recorded alongside the 2D distance: if the two
    disagree badly, the projection — not the retrieval — is what moved the words."""
    out = []
    for r in chosen:
        a = gallery_mod.glove_vector(glove, r['true_word'])
        b = gallery_mod.glove_vector(glove, r['pred_word'])
        out.append(float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    return out


def _select(pool, word_cat, rank_by):
    """Indices into ``pool``, greedily ascending ``rank_by`` (fraction of layout span),
    under the diversity constraints: bold words mutually distinct (no word is a truth in
    one pair and a predicted in another), at most MAX_PER_CAT pairs per semantic
    category, at most N_SHOWCASE pairs.

    Pairs beyond MAX_PAIR_DIST_FRAC are refused outright.  N_SHOWCASE is therefore a cap
    and not a quota: a participant whose pool runs thin yields fewer pairs rather than
    having distant ones padded in to reach ten."""
    order = [i for i in sorted(range(len(pool)), key=lambda i: rank_by[i])
             if rank_by[i] <= MAX_PAIR_DIST_FRAC]

    def greedy(first=None):
        seq = order if first is None else [first] + [j for j in order if j != first]
        used, cat_count, out = set(), {}, []
        for i in seq:
            tw, pw = pool[i]['true_word'], pool[i]['pred_word']
            c = word_cat[tw]
            if tw in used or pw in used or cat_count.get(c, 0) >= MAX_PER_CAT:
                continue
            out.append(i); used.add(tw); used.add(pw)
            cat_count[c] = cat_count.get(c, 0) + 1
            if len(out) >= N_SHOWCASE:
                break
        return out

    # Plain greedy is myopic about the mutual-distinctness rule: taking the single
    # closest pair can burn two words that would otherwise have carried two pairs each,
    # leaving a panel with fewer examples than the data supports.  (Measured: it cost
    # NUE031 four of seven pairs, because peach->pear blocked both lime->peach and
    # mango->pear.)  Restart it once per candidate first pick and keep the selection
    # with the MOST pairs, breaking ties on mean distance — so the panel stays as full
    # as the constraints allow without ever admitting a pair over the cutoff.
    best_key, best = None, []
    for first in [None] + order:
        cand = greedy(first)
        key = (-len(cand), float(np.mean([rank_by[i] for i in cand])) if cand else 9e9)
        if best_key is None or key < best_key:
            best_key, best = key, cand
    return sorted(best, key=lambda i: rank_by[i])


def compute_panelf_mds(per, glove, gal, rel_fn, patient, out_name):
    """2D semantic-neighbourhood showcase for one participant (panels e/f, supps S3/S4).

    Every plotted point is a REAL gallery word laid out by word-to-word cosine
    distance (metric MDS) — including the "predicted" word, which is the top-1
    retrieved gallery word placed at ITS OWN GloVe vector, not the (mean-shrunk,
    centrally-clustered) predicted embedding.  This keeps the retrieval geometry
    faithful: a near-synonym prediction sits next to the true word and shared
    neighbours rather than collapsing to the middle.

    Pairs are chosen to MINIMISE their distance in the rendered 2D layout, so the
    panel shows what it claims to show — a near-miss landing next door.  That is
    circular by nature (the distance does not exist until the layout does), so it is
    resolved by iteration: rank the whole eligible pool on a throwaway pool layout,
    select, lay out the selection, replace each selected pair's estimate with its
    measured distance, and re-select.  At most MAX_SELECT_ITERS rounds; it stops
    early once the selection is stable.  The distances that survive are written to
    the cache so the claim is auditable rather than asserted.

    Homonyms are excluded from the bold words (see HOMONYM_WORDS); the grey
    neighbour cloud is drawn from the full gallery and is not filtered."""
    d = per[patient]
    words, emb, cat = per_word_mean_predictions(d)
    recs = word_retrieval_grades(words, emb, gal, rel_fn, want_ndcg=False)
    word_cat = {w: cat[i] for i, w in enumerate(words)}

    # Pass 1 — eligibility.  MIN_NEAR_MISS_SIM stays a QUALITY GATE (a showcase must
    # have a genuinely coherent neighbourhood) but no longer decides the ranking.
    pool = [r for r in recs
            if r['in_gallery'] and r['pred_word'] != r['true_word']
            and r['near_miss_sim'] >= MIN_NEAR_MISS_SIM
            and r['true_word'] not in HOMONYM_WORDS
            and r['pred_word'] not in HOMONYM_WORDS]
    n_homonym_blocked = sum(
        1 for r in recs
        if r['in_gallery'] and r['pred_word'] != r['true_word']
        and r['near_miss_sim'] >= MIN_NEAR_MISS_SIM
        and (r['true_word'] in HOMONYM_WORDS or r['pred_word'] in HOMONYM_WORDS))
    if not pool:
        raise RuntimeError(f'{patient}: no eligible showcase pairs after filtering')

    # Pass 2 — lay the whole pool out once so every candidate has a 2D distance.
    pool_rows = _showcase_points(pool, gal, word_cat)
    _, rank_by = _pair_dists_2d(pool, pool_rows, _layout(pool_rows, glove, n_init=1))

    # Passes 3-5 — select, lay out, re-select against the layout actually rendered.  The
    # pool layout and the rendered layout differ because the point set differs, so the
    # first ranking is only an estimate; each round replaces the estimate for the pairs
    # currently selected with what was measured.  A pair measured over the cutoff thereby
    # becomes ineligible and the next round re-picks from the whole pool, so the selector
    # RECOVERS rather than merely losing a slot.
    #
    # Only configurations that have been laid out and verified feasible are eligible to
    # ship, and the best of those is kept.  Shipping whatever the last round happened to
    # hold is what let a 38%-of-span connector into NUE036 on an earlier pass: the loop
    # ran out of iterations mid-oscillation and the drop pass could only subtract from a
    # bad set, never re-pick a good one.
    sel = _select(pool, word_cat, rank_by)
    best = None
    for _ in range(MAX_SELECT_ITERS):
        chosen = [pool[i] for i in sel]
        rows = _showcase_points(chosen, gal, word_cat)
        xy = _layout(rows, glove)
        dists, fracs = _pair_dists_2d(chosen, rows, xy)
        for i, fv in zip(sel, fracs):
            rank_by[i] = fv                 # measured beats estimated
        if fracs and max(fracs) <= MAX_PAIR_DIST_FRAC:
            key = (-len(sel), float(np.mean(fracs)))
            if best is None or key < best[0]:
                best = (key, chosen, rows, xy, dists, fracs)
        nxt = _select(pool, word_cat, rank_by)
        if nxt == sel:
            break
        sel = nxt

    if best is not None:
        _, chosen, rows, xy, dists, fracs = best
    else:
        # Nothing feasible was found in the iteration budget.  Ship the last measured
        # configuration minus its violators rather than silently keeping them, and say so.
        over = {j for j, f in enumerate(fracs) if f > MAX_PAIR_DIST_FRAC}
        if len(chosen) - len(over) >= MIN_SHOWCASE:
            chosen = [r for j, r in enumerate(chosen) if j not in over]
            rows = _showcase_points(chosen, gal, word_cat)
            xy = _layout(rows, glove)
            dists, fracs = _pair_dists_2d(chosen, rows, xy)
        print(f"    {patient}: WARNING no configuration met the {MAX_PAIR_DIST_FRAC:.0%} "
              f"cutoff within {MAX_SELECT_ITERS} rounds; shipping {len(chosen)} pairs, "
              f"worst {max(fracs):.1%} of span", flush=True)
    coss = _pair_dists_cos(chosen, glove)

    # Carry each pair's distances on its two bold rows (blank for neighbours), so the
    # figure and the numbers behind it stay one unit.
    dist_of = {f"{gi}:{r['true_word']}": (dists[gi], fracs[gi], coss[gi])
               for gi, r in enumerate(chosen)}
    out = pd.DataFrame(rows)
    out.insert(0, 'patient', patient)
    out.insert(1, 'display_id', display_id(patient))
    out['x'] = xy[:, 0]; out['y'] = xy[:, 1]
    nan3 = (np.nan, np.nan, np.nan)
    bold_mask = out['role'].isin(['truth', 'predicted'])
    for col, k in [('pair_dist_2d', 0), ('pair_dist_frac', 1), ('pair_cos_dist', 2)]:
        out[col] = np.where(bold_mask,
                            out['trial_group'].map(lambda g, k=k: dist_of.get(g, nan3)[k]),
                            np.nan)
    out.to_csv(os.path.join(SRC_DIR, out_name), index=False)

    print(f"  wrote {out_name} ({len(out)} pts, {len(chosen)} showcase pairs, MDS/cosine)",
          flush=True)
    print(f"    {patient} pool={len(pool)} eligible "
          f"({n_homonym_blocked} pairs blocked as homonym), layout span={_span(xy):.3f}, "
          f"cutoff={MAX_PAIR_DIST_FRAC:.0%} of span", flush=True)
    for gi, r in enumerate(chosen):
        print(f"      {r['true_word']:>12s} -> {r['pred_word']:<12s} "
              f"2d={dists[gi]:.3f} ({fracs[gi]:.1%} of span)  "
              f"cos={coss[gi]:.3f}  nm={r['near_miss_sim']:.3f}", flush=True)
    return out


# ── 3. Qualitative best-case table (per patient) ──────────────────────────

def compute_bestcases(per, patients, glove, gal, rel_fn):
    rows = []
    for pat in patients:
        d = per[pat]
        words, emb, cat = per_word_mean_predictions(d)
        word_cat = {w: cat[i] for i, w in enumerate(words)}
        recs = word_retrieval_grades(words, emb, gal, rel_fn, want_ndcg=False)
        # A "best case" must retrieve the true word well AND land on related words;
        # rank by near-miss similarity among the well-retrieved (rank<=BESTCASE_RANK_MAX).
        pool = [r for r in recs if r['in_gallery'] and r['rank'] <= BESTCASE_RANK_MAX]
        if len(pool) < N_BEST_PER_PATIENT:      # relax if a patient has too few
            pool = [r for r in recs if r['in_gallery']]
        cand = sorted(pool, key=lambda r: r['near_miss_sim'], reverse=True)
        chosen, used_cat = [], set()
        for r in cand:
            c = word_cat[r['true_word']]
            if c in used_cat:
                continue
            chosen.append(r); used_cat.add(c)
            if len(chosen) >= N_BEST_PER_PATIENT:
                break
        for r in chosen[:N_BEST_PER_PATIENT]:
            order_row = r['order_row']
            top5 = [gal.words[int(j)] for j in order_row[:5]]
            grades = [f"{rel_fn(r['true_word'], w):.2f}" for w in top5]
            ndcg = metrics.ndcg_independent(order_row, r['true_word'], gal.words, rel_fn, k=100)
            rows.append(dict(
                display_id=display_id(pat), patient=pat, true_word=r['true_word'],
                category=word_cat[r['true_word']], rank=r['rank'],
                near_miss_sim=round(r['near_miss_sim'], 4), ndcg=round(float(ndcg), 4),
                top1=top5[0], top2=top5[1], top3=top5[2], top4=top5[3], top5=top5[4],
                grades=';'.join(grades)))
        print(f"  [bestcases] {pat}: {len(chosen)} cases", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(SRC_DIR, 'cache_qualitative_bestcases.csv'), index=False)
    print(f"  wrote cache_qualitative_bestcases.csv ({len(out)} rows)", flush=True)
    return out


def main():
    os.makedirs(SRC_DIR, exist_ok=True)
    print("[compute] loading trial predictions ...", flush=True)
    per, patients, stim_words = load_trials()
    print(f"  {len(patients)} patients, {len(stim_words)} stimulus words", flush=True)

    print("[compute] loading GloVe (torchtext, ~10 s) ...", flush=True)
    glove = gallery_mod.load_glove()
    rel_fn = relevance.make_relevance_fn('wup')

    print("[1/3] held-out per-trial percentiles across N ...", flush=True)
    compute_heldout_percentiles(per, patients, stim_words, glove)

    showcase = [SHOWCASE_LEFT, SHOWCASE_RIGHT] + PANELF_EXTRA
    print(f"[2/3] MDS showcases (patients {showcase}) ...", flush=True)
    gal5000 = build_matched_gallery(glove, stim_words, HEADLINE_N)
    compute_panelf_mds(per, glove, gal5000, rel_fn, SHOWCASE_LEFT, 'cache_panelf_mds.csv')
    for pat in [SHOWCASE_RIGHT] + PANELF_EXTRA:
        compute_panelf_mds(per, glove, gal5000, rel_fn, pat, f'cache_panelf_{pat}.csv')

    print("[3/3] qualitative best cases per patient ...", flush=True)
    compute_bestcases(per, patients, glove, gal5000, rel_fn)

    print("[compute] done ->", SRC_DIR, flush=True)


if __name__ == '__main__':
    main()
