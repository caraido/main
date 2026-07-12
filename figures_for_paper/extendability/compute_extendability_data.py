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
  2. cache_panelf_mds.csv                       (panel f: 2D MDS-cosine showcase, best patient)
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
from tests.open_vocab_retrieval import gallery as gallery_mod          # noqa: E402
from tests.open_vocab_retrieval import retrieval, metrics, relevance   # noqa: E402

OPENVOCAB_SRC = os.path.join(MAIN_DIR, 'figures', 'open_vocab_retrieval', 'source_data')
SRC_DIR = os.path.join(HERE, 'source_data')

TASK = 'picture_naming'
NS = [200, 500, 1000, 2000, 5000]
HEADLINE_N = 5000
BEST_PATIENT = 'VB'          # combined-figure panel f (highest top-10 acc; largest null delta)
PANELF_EXTRA = ['RB', 'AA', 'WBH']  # supplementary panel-f patients (largest null-vs-neural delta)
N_SHOWCASE = 10              # panel-f showcase words (diverse category)
N_NEIGHBORS = 22             # peripheral neighbours per showcase word
MAX_PER_CAT = 3              # cap showcase words per semantic category (diversity)
MIN_NEAR_MISS_SIM = 0.55     # a showcase must have a genuinely coherent neighbourhood
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


# ── 2. Panel-f MDS showcase (best patient) ────────────────────────────────

def compute_panelf_mds(per, glove, gal, rel_fn, patient, out_name):
    """Panel-f 2D semantic-neighbourhood layout for one participant.

    Every plotted point is a REAL gallery word laid out by word-to-word cosine
    distance (metric MDS) — including the "predicted" word, which is the top-1
    retrieved gallery word placed at ITS OWN GloVe vector, not the (mean-shrunk,
    centrally-clustered) predicted embedding.  This keeps the retrieval geometry
    faithful: a near-synonym prediction sits next to the true word and shared
    neighbours rather than collapsing to the middle."""
    from sklearn.manifold import MDS
    from sklearn.metrics.pairwise import cosine_distances
    d = per[patient]
    words, emb, cat = per_word_mean_predictions(d)
    recs = word_retrieval_grades(words, emb, gal, rel_fn, want_ndcg=False)
    word_cat = {w: cat[i] for i, w in enumerate(words)}

    # Select showcase near-misses (pred != true) by neighbour similarity, with all
    # bold words mutually distinct (no word is a truth in one pair and a predicted
    # in another) and at most MAX_PER_CAT per semantic category.
    used_words, cat_count, chosen = set(), {}, []
    for r in sorted(recs, key=lambda r: r['near_miss_sim'], reverse=True):
        tw, pw = r['true_word'], r['pred_word']
        c = word_cat[tw]
        if (not r['in_gallery'] or pw == tw or r['near_miss_sim'] < MIN_NEAR_MISS_SIM
                or tw in used_words or pw in used_words
                or cat_count.get(c, 0) >= MAX_PER_CAT):
            continue
        chosen.append(r); used_words.add(tw); used_words.add(pw)
        cat_count[c] = cat_count.get(c, 0) + 1
        if len(chosen) >= N_SHOWCASE:
            break

    bold_words = used_words
    # Assemble unique real-word points: truth + predicted (both bold) + top-N
    # retrieved neighbours (grey), deduped globally so no word appears twice.
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
            if gw in bold_words:          # skip words that are a showcase truth/pred
                continue
            add(gw, 'neighbor', grp, '')
            cnt += 1
            if cnt >= N_NEIGHBORS:
                break

    X = np.vstack([gallery_mod.glove_vector(glove, r['label']) for r in rows])
    D = cosine_distances(X)
    xy = MDS(n_components=2, dissimilarity='precomputed', random_state=0,
             n_init=4, max_iter=400).fit_transform(D)
    out = pd.DataFrame(rows)
    out.insert(0, 'patient', patient)
    out.insert(1, 'display_id', display_id(patient))
    out['x'] = xy[:, 0]; out['y'] = xy[:, 1]
    out.to_csv(os.path.join(SRC_DIR, out_name), index=False)
    print(f"  wrote {out_name} ({len(out)} pts, {len(chosen)} showcase words, MDS/cosine)",
          flush=True)
    print(f"    {patient} showcase: {[(r['true_word'], r['pred_word'], round(r['near_miss_sim'],3)) for r in chosen]}",
          flush=True)
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

    print(f"[2/3] panel-f MDS showcase (patients {[BEST_PATIENT] + PANELF_EXTRA}) ...", flush=True)
    gal5000 = build_matched_gallery(glove, stim_words, HEADLINE_N)
    compute_panelf_mds(per, glove, gal5000, rel_fn, BEST_PATIENT, 'cache_panelf_mds.csv')
    for pat in PANELF_EXTRA:
        compute_panelf_mds(per, glove, gal5000, rel_fn, pat, f'cache_panelf_{pat}.csv')

    print("[3/3] qualitative best cases per patient ...", flush=True)
    compute_bestcases(per, patients, glove, gal5000, rel_fn)

    print("[compute] done ->", SRC_DIR, flush=True)


if __name__ == '__main__':
    main()
