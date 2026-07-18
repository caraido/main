# -*- coding: utf-8 -*-
"""
figures_for_paper/semantic_regression/within_category_null.py
=============================================================
Does word-level retrieval carry information BEYOND semantic category — or is it
category decoding dressed up as a word read-out?  This is the single sharpest
attack on the word-decoding claim (an ML-literate committee member WILL pose it):
top-1 word accuracy (~4%) is usually shown against the naive uniform 1/W null,
but it sits only ~2x above a *category-preserving* null.  If word retrieval does
not clear that null, "we decode words" collapses to "we decode category".

This script benchmarks top-k word retrieval against the correct null:

  * uniform null        — the substituted word is any stimulus word (chance 1/W).
  * within-category null — the substituted word is a random word OF THE SAME
    CATEGORY.  This is exactly the distribution a decoder that knows ONLY the
    category (and then guesses uniformly within it) would produce.  Observed
    top-k above THIS null == genuine sub-category word identity.

Mechanism (exact; no re-fitting, no re-ranking):
  For each trial we rank that patient's W unique stimulus words by mean-centred
  cosine to the trial's predicted embedding — the canonical retrieval convention
  (utils.retrieval.mean_center_db).  top-k for any candidate word = its rank < k.
  Observed top-k reads the rank of the TRUE word; each null draw substitutes a
  random word (uniform, or same-category) and reads ITS rank off the SAME fixed
  ranking, so a permutation is a cheap membership check (10k draws is trivial).

Prediction (what "the decoder knows more than category" looks like): the excess
of observed over the within-category null should GROW with k, because top-1 is
where category and word identity are hardest to separate and top-3/5 is where
true within-category discrimination shows through.
  Analytic category-preserving nulls (cat.acc 0.29, ~11 words/category):
      top-1 ~ 0.29*(1/11) ~ 2.6% ; top-3 ~ 7.7% ; top-5 ~ 12.8%
  vs observed ~ 4.2 / 11.4 / 17.8%.  This script computes the *empirical* version
  of that null per participant (no uniform-confusion assumption) with a p-value.

Reads the same per-trial predictions the extendability / semantic-regression
figures use:
  figures/open_vocab_retrieval/source_data/trial_predictions_<task>.csv
  (columns: patient, true_word, category, is_held_out, pe0..pe299)

Run (Speech env):
    C:/Users/Owner/miniconda3/envs/Speech/python.exe ^
        figures_for_paper/semantic_regression/within_category_null.py

Outputs -> figures_for_paper/semantic_regression/source_data/
    within_category_null_topk.csv   (per-participant + GROUP: obs / uniform / within-cat / p)
and a figure  12_within_category_null.(png|pdf)  beside the other panels.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS_ROOT = os.path.dirname(HERE)                          # figures_for_paper/
MAIN_DIR = os.path.dirname(FIGS_ROOT)                      # …/main
for pth in (MAIN_DIR, FIGS_ROOT):
    if pth not in sys.path:
        sys.path.insert(0, pth)

from paper_common import display_id, assign_colors, apply_paper_style   # noqa: E402
from analysis.open_vocab_retrieval import gallery as gallery_mod           # noqa: E402

# canonical mean-centre; fall back to a local copy if utils isn't importable
try:
    from utils.retrieval import mean_center_db, normalize_rows          # noqa: E402
except Exception:                                                       # pragma: no cover
    def normalize_rows(M, eps=1e-10):
        return M / (np.linalg.norm(M, axis=1, keepdims=True) + eps)
    def mean_center_db(db, q):
        m = db.mean(axis=0)
        return db - m, q - m, m

# ── config ────────────────────────────────────────────────────────────────
TASK = 'picture_naming'
KS = (1, 3, 5)
N_PERM = 10_000
SEED = 0
CENTER = True                     # canonical mean-centred cosine (matches the run)
PE_COLS = [f'pe{j}' for j in range(300)]
OPENVOCAB_SRC = os.path.join(MAIN_DIR, 'figures', 'open_vocab_retrieval', 'source_data')
SRC_DIR = os.path.join(HERE, 'source_data')


def load_trials():
    df = pd.read_csv(os.path.join(OPENVOCAB_SRC, f'trial_predictions_{TASK}.csv'))
    per = {}
    for pat, g in df.groupby('patient', sort=False):
        per[pat] = dict(
            pred_emb=g[PE_COLS].to_numpy(dtype=np.float64),
            true_word=g['true_word'].astype(str).str.strip().str.lower().to_numpy(),
            category=g['category'].astype(str).str.strip().str.lower().to_numpy(),
        )
    patients = list(dict.fromkeys(df['patient']))          # first-appearance order
    return per, patients


def rank_matrix_for_patient(d, glove):
    """(T, W) rank of every unique stimulus word per trial (0 = nearest).

    DB = GloVe target vector of each unique word (the vector the regressor was
    trained to predict); ranking is mean-centred cosine of the prediction vs DB.
    Returns rank_mat, words (W,), word_cat_idx (W,), true_word_idx (T,)."""
    words = np.array(list(dict.fromkeys(d['true_word'])))               # unique, stable order
    w2i = {w: i for i, w in enumerate(words)}
    db = np.vstack([gallery_mod.glove_vector(glove, w) for w in words]) # (W, 300)
    # one category per word (first trial's category for that word)
    word_cat = np.array([d['category'][np.where(d['true_word'] == w)[0][0]] for w in words])
    cats, cat_idx = np.unique(word_cat, return_inverse=True)

    q = d['pred_emb']
    if CENTER:
        db_c, q_c, _ = mean_center_db(db, q)
    else:
        db_c, q_c = db, q
    sims = normalize_rows(q_c) @ normalize_rows(db_c).T                  # (T, W)
    # rank of each word per trial: argsort(argsort(-sims))  → 0 = nearest
    order = np.argsort(-sims, axis=1, kind='stable')
    rank_mat = np.empty_like(order)
    rows = np.arange(sims.shape[0])[:, None]
    rank_mat[rows, order] = np.arange(sims.shape[1])[None, :]
    tw_idx = np.array([w2i[w] for w in d['true_word']], dtype=np.int64)
    return rank_mat, words, cat_idx, tw_idx


def topk_from_ranks(ranks_1d, k):
    """Fraction of trials whose candidate word ranked in the top-k."""
    return float(np.mean(ranks_1d < k))


def run_patient(d, glove, rng):
    rank_mat, words, cat_idx, tw_idx = rank_matrix_for_patient(d, glove)
    T, W = rank_mat.shape
    trial_rows = np.arange(T)
    # per-trial pool of same-category word indices (for the within-cat null)
    cat_to_words = {c: np.where(cat_idx == c)[0] for c in np.unique(cat_idx)}
    trial_cat = cat_idx[tw_idx]

    # pre-sample null substitutions once: (N_PERM, T)
    unif = rng.integers(0, W, size=(N_PERM, T))
    wcat = np.empty((N_PERM, T), dtype=np.int64)
    for t in range(T):
        pool = cat_to_words[trial_cat[t]]
        wcat[:, t] = rng.choice(pool, size=N_PERM, replace=True)

    out = {}
    for k in KS:
        obs = topk_from_ranks(rank_mat[trial_rows, tw_idx], k)
        # null top-k distributions: read the substituted word's rank off rank_mat
        unif_ranks = rank_mat[trial_rows[None, :], unif]                 # (N_PERM, T)
        wcat_ranks = rank_mat[trial_rows[None, :], wcat]
        unif_dist = np.mean(unif_ranks < k, axis=1)
        wcat_dist = np.mean(wcat_ranks < k, axis=1)
        # one-sided p: P(null >= observed) under the within-category null
        p = (1 + np.sum(wcat_dist >= obs)) / (N_PERM + 1)
        out[k] = dict(obs=obs,
                      unif_mean=float(unif_dist.mean()),
                      wcat_mean=float(wcat_dist.mean()),
                      wcat_lo=float(np.percentile(wcat_dist, 2.5)),
                      wcat_hi=float(np.percentile(wcat_dist, 97.5)),
                      excess=obs - float(wcat_dist.mean()),
                      p_within_cat=float(p))
    return out, W, len(np.unique(cat_idx))


def main():
    os.makedirs(SRC_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    print(f"[within-cat null] loading predictions ({TASK}) ...", flush=True)
    per, patients = load_trials()
    print(f"  {len(patients)} patients", flush=True)
    print("[within-cat null] loading GloVe (~10 s) ...", flush=True)
    glove = gallery_mod.load_glove()

    rows = []
    for pat in patients:
        res, W, C = run_patient(per[pat], glove, rng)
        did = display_id(pat)
        for k in KS:
            r = res[k]
            rows.append(dict(display_id=did, patient=pat, k=k, n_words=W, n_cats=C, **r))
        msg = "  ".join(f"top{k}: obs={res[k]['obs']:.3f} vs cat-null={res[k]['wcat_mean']:.3f} "
                        f"(Δ={res[k]['excess']:+.3f}, p={res[k]['p_within_cat']:.3g})" for k in KS)
        print(f"  {did}: {msg}", flush=True)

    df = pd.DataFrame(rows)
    csv = os.path.join(SRC_DIR, 'within_category_null_topk.csv')
    df.to_csv(csv, index=False)

    # ── group summary ─────────────────────────────────────────────────────
    print("\n===== GROUP (mean across participants) =====", flush=True)
    for k in KS:
        g = df[df.k == k]
        n_sig = int((g.p_within_cat < 0.05).sum())
        print(f"  top-{k}:  observed {g.obs.mean():.3f} | uniform null {g.unif_mean.mean():.3f} | "
              f"within-category null {g.wcat_mean.mean():.3f} | excess {g.excess.mean():+.3f} | "
              f"{n_sig}/{len(g)} participants p<0.05", flush=True)
    print(f"\n  wrote {csv}", flush=True)

    _make_figure(df)


def _make_figure(df):
    """Grouped bars per k: observed vs within-category null (+uniform line), dots=participants."""
    import matplotlib.pyplot as plt
    apply_paper_style()
    AMBER, GREY = '#C2670F', '#9AA4AE'
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ks = list(KS)
    x = np.arange(len(ks)); wdt = 0.36
    obs = [df[df.k == k].obs.mean() for k in ks]
    cat = [df[df.k == k].wcat_mean.mean() for k in ks]
    obs_e = [df[df.k == k].obs.sem() for k in ks]
    ax.bar(x - wdt/2, obs, wdt, yerr=obs_e, color=AMBER, label='observed (true word)',
           capsize=2, error_kw=dict(lw=0.8))
    ax.bar(x + wdt/2, cat, wdt, color=GREY, alpha=0.85, label='within-category null')
    # uniform-null reference points
    unif = [df[df.k == k].unif_mean.mean() for k in ks]
    ax.plot(x, unif, ls='--', lw=0.9, color='k', marker='_', label='uniform null (1/W)')
    # participant dots on the observed bars + significance stars
    for xi, k in zip(x, ks):
        g = df[df.k == k]
        ax.scatter(np.full(len(g), xi - wdt/2) + np.random.uniform(-0.05, 0.05, len(g)),
                   g.obs, s=6, color='k', alpha=0.35, zorder=5)
        p = g.p_within_cat.median()
        star = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'n.s.'
        top = max(g.obs.max(), g.wcat_mean.max())
        ax.text(xi, top + 0.012, star, ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f'top-{k}' for k in ks])
    ax.set_ylabel('word-retrieval accuracy'); ax.set_ylim(0, None)
    ax.set_title('a', loc='left', fontweight='bold')
    ax.legend(loc='upper left', fontsize=6.5)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(HERE, f'12_within_category_null.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote 12_within_category_null.png/.pdf", flush=True)


if __name__ == '__main__':
    main()
