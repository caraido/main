# -*- coding: utf-8 -*-
"""
figures_for_paper/extendability_co_trained/compute_extendability_data.py
========================================================================
Heavy compute step for the CO-TRAINED extendability figure (panels f + supplements),
run for BOTH tasks (picture_naming, auditory_naming).  Co-trained analogue of
``figures_for_paper/extendability/compute_extendability_data.py``.

Rebuilds the matched galleries from GloVe and the already-computed co-trained
per-trial predicted embeddings (written by ``run_co_trained_retrieval.py``), then
writes task-suffixed cache CSVs that the CSV-only ``extendability_panels.py`` reads.
Re-scores predictions against fresh galleries; does NOT re-fit any model.  Run once,
in the Speech conda env:

    C:/Users/Owner/miniconda3/envs/Speech/python.exe \
        figures_for_paper/extendability_co_trained/compute_extendability_data.py

Inputs  <- figures_for_paper/extendability_co_trained/source_data/
    trial_predictions_{task}.csv          (co-trained per-trial predicted GloVe)
    per_patient_metrics_{task}.csv        (used to pick showcase participants)
Outputs -> figures_for_paper/extendability_co_trained/source_data/
    cache_heldout_trial_percentile_by_N_{task}.csv   (supp S1)
    cache_panelf_mds_{task}.csv                        (panel f, best participant)
    cache_panelf_{pat}_{task}.csv                      (supp S3/S4 participants)
    cache_qualitative_bestcases_{task}.csv             (supp S2)
    panelf_showcase_{task}.json                        (best + extra participants; read by panels)
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS_ROOT = os.path.dirname(HERE)                          # …/figures_for_paper
MAIN_DIR = os.path.dirname(FIGS_ROOT)                      # …/main
if MAIN_DIR not in sys.path:
    sys.path.insert(0, MAIN_DIR)
sys.path.insert(0, FIGS_ROOT)                              # paper_common (display IDs)

from paper_common import display_id                        # noqa: E402
from tests.open_vocab_retrieval import gallery as gallery_mod          # noqa: E402
from tests.open_vocab_retrieval import retrieval, metrics, relevance   # noqa: E402

SRC_DIR = os.path.join(HERE, 'source_data')

TASKS = ['picture_naming', 'auditory_naming']
NS = [200, 500, 1000, 2000, 5000]
HEADLINE_N = 5000
N_PANELF_EXTRA = 3           # supplementary panel-f participants (best after the top one)
N_SHOWCASE = 10              # panel-f showcase words (diverse category)
N_NEIGHBORS = 22             # peripheral neighbours per showcase word
MAX_PER_CAT = 3              # cap showcase words per semantic category (diversity)
MIN_NEAR_MISS_SIM = 0.55     # a showcase must have a genuinely coherent neighbourhood
N_BEST_PER_PATIENT = 4       # rows per patient in the qualitative best-case table
BESTCASE_RANK_MAX = 50       # a "best case" must also retrieve the true word well
CENTER = True                # canonical mean-centred cosine (matches the run)
PE_COLS = [f'pe{j}' for j in range(300)]


# ── Data loading ──────────────────────────────────────────────────────────

def load_trials(task):
    """Per-patient arrays from the co-trained per-trial predictions CSV."""
    df = pd.read_csv(os.path.join(SRC_DIR, f'trial_predictions_{task}.csv'))
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


def select_showcase_patients(task, patients):
    """Best panel-f participant + extras, ranked by retrieval quality (lowest
    median percentile rank).  Falls back to first-appearance order if the
    per-patient metrics CSV is unavailable."""
    path = os.path.join(SRC_DIR, f'per_patient_metrics_{task}.csv')
    ranked = list(patients)
    if os.path.exists(path):
        m = pd.read_csv(path)
        if 'median_percentile_all' in m.columns:
            m = m[m['patient'].isin(patients)].sort_values('median_percentile_all')
            ranked = list(m['patient'])
            ranked += [p for p in patients if p not in ranked]   # any missing, appended
    best = ranked[0]
    extras = ranked[1:1 + N_PANELF_EXTRA]
    with open(os.path.join(SRC_DIR, f'panelf_showcase_{task}.json'), 'w', encoding='utf-8') as f:
        json.dump({'best': best, 'extras': extras}, f, indent=2)
    return best, extras


def build_matched_gallery(glove, stim_words, n):
    """Matched gallery reproducing the shipped run (concreteness/subtlex=None)."""
    return gallery_mod.build_gallery(glove, stim_words, n=n, variant='matched',
                                     concreteness=None, subtlex=None)


# ── 1. Held-out per-trial percentile across gallery sizes N ────────────────

def compute_heldout_percentiles(per, patients, stim_words, glove, task):
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
        print(f"  [{task} heldout%] N={n}: {sum(1 for r in rows if r['N']==n)} trials", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(SRC_DIR, f'cache_heldout_trial_percentile_by_N_{task}.csv'), index=False)
    print(f"  wrote cache_heldout_trial_percentile_by_N_{task}.csv ({len(out)} rows)", flush=True)
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
    centrally-clustered) predicted embedding."""
    from sklearn.manifold import MDS
    from sklearn.metrics.pairwise import cosine_distances
    d = per[patient]
    words, emb, cat = per_word_mean_predictions(d)
    recs = word_retrieval_grades(words, emb, gal, rel_fn, want_ndcg=False)
    word_cat = {w: cat[i] for i, w in enumerate(words)}

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

    if not chosen:
        print(f"  [panel-f] {patient}: no coherent showcase words (min sim {MIN_NEAR_MISS_SIM}) — skipped",
              flush=True)
        return None

    bold_words = used_words
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
            if gw in bold_words:
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
    print(f"  wrote {out_name} ({len(out)} pts, {len(chosen)} showcase words, MDS/cosine)", flush=True)
    print(f"    {patient} showcase: {[(r['true_word'], r['pred_word'], round(r['near_miss_sim'],3)) for r in chosen]}",
          flush=True)
    return out


# ── 3. Qualitative best-case table (per patient) ──────────────────────────

def compute_bestcases(per, patients, glove, gal, rel_fn, task):
    rows = []
    for pat in patients:
        d = per[pat]
        words, emb, cat = per_word_mean_predictions(d)
        word_cat = {w: cat[i] for i, w in enumerate(words)}
        recs = word_retrieval_grades(words, emb, gal, rel_fn, want_ndcg=False)
        pool = [r for r in recs if r['in_gallery'] and r['rank'] <= BESTCASE_RANK_MAX]
        if len(pool) < N_BEST_PER_PATIENT:
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
        print(f"  [{task} bestcases] {pat}: {len(chosen)} cases", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(SRC_DIR, f'cache_qualitative_bestcases_{task}.csv'), index=False)
    print(f"  wrote cache_qualitative_bestcases_{task}.csv ({len(out)} rows)", flush=True)
    return out


def compute_task(task, glove, rel_fn):
    print(f"\n========== {task} ==========", flush=True)
    per, patients, stim_words = load_trials(task)
    print(f"  {len(patients)} patients, {len(stim_words)} stimulus words", flush=True)
    best, extras = select_showcase_patients(task, patients)
    print(f"  panel-f best={best} extras={extras}", flush=True)

    print(f"[1/3] held-out per-trial percentiles across N ...", flush=True)
    compute_heldout_percentiles(per, patients, stim_words, glove, task)

    print(f"[2/3] panel-f MDS showcase (patients {[best] + extras}) ...", flush=True)
    gal5000 = build_matched_gallery(glove, stim_words, HEADLINE_N)
    compute_panelf_mds(per, glove, gal5000, rel_fn, best, f'cache_panelf_mds_{task}.csv')
    for pat in extras:
        compute_panelf_mds(per, glove, gal5000, rel_fn, pat, f'cache_panelf_{pat}_{task}.csv')

    print(f"[3/3] qualitative best cases per patient ...", flush=True)
    compute_bestcases(per, patients, glove, gal5000, rel_fn, task)


def main():
    os.makedirs(SRC_DIR, exist_ok=True)
    print("[compute] loading GloVe (torchtext, ~10 s) ...", flush=True)
    glove = gallery_mod.load_glove()
    rel_fn = relevance.make_relevance_fn('wup')
    for task in TASKS:
        compute_task(task, glove, rel_fn)
    print("\n[compute] done ->", SRC_DIR, flush=True)


if __name__ == '__main__':
    main()
