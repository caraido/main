# -*- coding: utf-8 -*-
"""
figures_for_paper/semantic_regression/compute_within_category_null.py
====================================================================
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

BOTH TASKS.  Picture naming and auditory naming are computed by the same code
and land in one CSV keyed by a ``task`` column, because the Holm correction
spans them: six tests (3 k x 2 tasks) are one family, and a family cannot be
corrected from a file holding half of it.  The two arms are NOT a matched
contrast — cohort (15 vs 10), vocabulary size and category inventory all differ,
and the auditory arm spans two stimulus sets (5-7 categories per participant).

Reads the same per-trial predictions the extendability / semantic-regression
figures use:
  figures/open_vocab_retrieval/source_data/trial_predictions_<task>.csv
  (columns: patient, task, true_word, true_label, category, is_held_out,
   cv_fold, pe0..pe299 — selected by name, so extra columns are harmless)

Run (Speech env, from main/):
    conda run -n Speech python ^
        figures_for_paper/semantic_regression/compute_within_category_null.py --task both

Outputs -> figures_for_paper/semantic_regression/source_data/
    within_category_null_topk.csv    per participant x task x k
    within_category_null_group.csv   cohort mean/SEM + Wilcoxon + Holm, 6 rows
Rendered by within_category_null_panels.py, which computes nothing.
"""

from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS_ROOT = os.path.dirname(HERE)                          # figures_for_paper/
MAIN_DIR = os.path.dirname(FIGS_ROOT)                      # …/main
for pth in (MAIN_DIR, FIGS_ROOT):
    if pth not in sys.path:
        sys.path.insert(0, pth)

from paper_common import display_id, p_stars                              # noqa: E402
from analysis.open_vocab_retrieval import gallery as gallery_mod           # noqa: E402
from utils.paths import figures_dir, paper_source_data                     # noqa: E402

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
TASKS = ('picture_naming', 'auditory_naming')
#: Which arms actually ship in the supplementary figure.  Auditory naming is computed and
#: kept in source_data but not plotted (Alec, 2026-08-11: the auditory result needs a team
#: discussion first).  This also fixes the Holm family -- see group_stats.
SHIPPED_TASKS = ('picture_naming',)
KS = (1, 3, 5)
N_PERM = 10_000
SEED = 0
CENTER = True                     # canonical mean-centred cosine (matches the run)
PE_COLS = [f'pe{j}' for j in range(300)]

#: The sanctioned accessors.  ``create=False`` on the file forms is mandatory, not
#: stylistic: ``paper_dir`` runs makedirs on the fully joined path, so ``create=True``
#: would produce a *directory* named ``within_category_null_topk.csv``.  ``create=False``
#: on the input is deliberate too — silently mkdir-ing a gitignored input directory turns
#: "the input is missing" into "the input is empty", which fails later and less clearly.
OPENVOCAB_SRC = figures_dir('open_vocab_retrieval', 'source_data', create=False)
TOPK_CSV = paper_source_data('semantic_regression',
                             'within_category_null_topk.csv', create=False)
GROUP_CSV = paper_source_data('semantic_regression',
                              'within_category_null_group.csv', create=False)

#: Per-participant series, and the group-CSV stem each aggregates into.
SERIES = (('obs', 'obs'), ('unif_mean', 'unif'), ('wcat_mean', 'wcat'))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description='Category-preserving null for word-level retrieval, both tasks.')
    ap.add_argument('--task', choices=(*TASKS, 'both'), default='both',
                    help='which arm to (re)compute; the default runs both, because '
                         'one process pays the ~10 s / ~5 GB GloVe load only once')
    ap.add_argument('--n-perm', type=int, default=N_PERM)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--group-only', action='store_true',
                    help='skip the permutations entirely; re-derive only the group CSV '
                         'from the tracked per-participant CSV')
    return ap.parse_args(argv)


def load_trials(task: str):
    """Per-patient predicted embeddings / true words / categories for one task.

    Columns are selected by NAME, so the two extra leading columns the open-vocab
    writer emits (``task``, ``true_label``, ``cv_fold``) are harmless here.
    """
    df = pd.read_csv(os.path.join(OPENVOCAB_SRC, f'trial_predictions_{task}.csv'))
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


def run_patient(d, glove, rng, n_perm=N_PERM):
    rank_mat, words, cat_idx, tw_idx = rank_matrix_for_patient(d, glove)
    T, W = rank_mat.shape
    trial_rows = np.arange(T)
    # per-trial pool of same-category word indices (for the within-cat null)
    cat_to_words = {c: np.where(cat_idx == c)[0] for c in np.unique(cat_idx)}
    trial_cat = cat_idx[tw_idx]

    # pre-sample null substitutions once: (n_perm, T)
    unif = rng.integers(0, W, size=(n_perm, T))
    wcat = np.empty((n_perm, T), dtype=np.int64)
    for t in range(T):
        pool = cat_to_words[trial_cat[t]]
        wcat[:, t] = rng.choice(pool, size=n_perm, replace=True)

    out = {}
    for k in KS:
        obs = topk_from_ranks(rank_mat[trial_rows, tw_idx], k)
        # null top-k distributions: read the substituted word's rank off rank_mat
        unif_ranks = rank_mat[trial_rows[None, :], unif]                 # (n_perm, T)
        wcat_ranks = rank_mat[trial_rows[None, :], wcat]
        unif_dist = np.mean(unif_ranks < k, axis=1)
        wcat_dist = np.mean(wcat_ranks < k, axis=1)
        # one-sided p: P(null >= observed) under the within-category null
        p = (1 + np.sum(wcat_dist >= obs)) / (n_perm + 1)
        out[k] = dict(obs=obs,
                      unif_mean=float(unif_dist.mean()),
                      wcat_mean=float(wcat_dist.mean()),
                      wcat_lo=float(np.percentile(wcat_dist, 2.5)),
                      wcat_hi=float(np.percentile(wcat_dist, 97.5)),
                      excess=obs - float(wcat_dist.mean()),
                      p_within_cat=float(p))
    return out, W, len(np.unique(cat_idx))


def run_task(task: str, glove, *, n_perm: int, seed: int) -> pd.DataFrame:
    """Per-participant rows for one task.

    The RNG is created FRESH here, per task, so an arm's numbers are a function of
    (seed, that task's patient order) alone.  Running one task does not move the
    other, and ``--task picture_naming`` reproduces the picture rows of a
    ``--task both`` run bit-for-bit — which is what makes "the picture arm is
    unchanged" a checkable claim rather than an eyeball.
    """
    print(f"[within-cat null] loading predictions ({task}) ...", flush=True)
    per, patients = load_trials(task)
    print(f"  {len(patients)} patients", flush=True)

    rng = np.random.default_rng(seed)
    rows = []
    for pat in patients:
        res, W, C = run_patient(per[pat], glove, rng, n_perm=n_perm)
        did = display_id(pat)
        for k in KS:
            rows.append(dict(display_id=did, patient=pat, task=task, k=k,
                             n_words=W, n_cats=C, **res[k]))
        msg = "  ".join(f"top{k}: obs={res[k]['obs']:.3f} vs cat-null={res[k]['wcat_mean']:.3f} "
                        f"(d={res[k]['excess']:+.3f}, p={res[k]['p_within_cat']:.3g})" for k in KS)
        print(f"  {did}: {msg}", flush=True)
    return pd.DataFrame(rows)


def merge_tasks(new_by_task: dict) -> pd.DataFrame:
    """Fold freshly computed task frames into the tracked CSV.

    Rewriting one arm must not silently discard the other: the Holm correction is a
    single family spanning both, so a file holding one task cannot be corrected.
    Reads the existing CSV if present, drops the rows of every task just recomputed,
    concatenates, sorts.
    """
    old = None
    if os.path.exists(TOPK_CSV):
        old = pd.read_csv(TOPK_CSV)
        if 'task' not in old.columns:      # pre-rewrite CSV: picture-only by construction
            old['task'] = 'picture_naming'

    # Task order from TASKS; WITHIN a task, row order is left exactly as it came --
    # first-appearance patient order for a fresh frame, untouched for a retained one.
    # Sorting here (by display_id, say) would reorder every existing row and destroy
    # the "the picture arm is unchanged" diff, which is the only real check this
    # refactor has.
    frames = []
    for task in TASKS:
        if task in new_by_task:
            frames.append(new_by_task[task])
        elif old is not None:
            keep = old[old['task'] == task]
            if len(keep):
                frames.append(keep)
    df = pd.concat(frames, ignore_index=True)
    return df[['display_id', 'patient', 'task', 'k', 'n_words', 'n_cats',
               'obs', 'unif_mean', 'wcat_mean', 'wcat_lo', 'wcat_hi',
               'excess', 'p_within_cat']]


def group_stats(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, k): cohort mean/SEM of each series, plus the group test.

    The test is a one-sided Wilcoxon signed-rank across participants on
    excess = obs - wcat_mean.  scipy uses the exact signed-rank distribution at these
    n, which is what makes the 1/2**n floor exact.

    **Holm is applied WITHIN a task, over that task's three k.**  The correction family
    is the set of tests a figure reports, and each task's panel reports three.  A
    six-test family was right while both tasks shipped in one figure; since 2026-08-11
    only picture naming does (auditory is retained here but unshipped, pending Alec's
    discussion with the team), so correcting the picture p-values for auditory tests no
    reader can see would be conservative for no stated reason.  Changing what ships
    therefore changes this function -- that coupling is deliberate.
    """
    frames = []
    for task in TASKS:
        rows = []
        for k in KS:
            s = df[(df.task == task) & (df.k == k)]
            if not len(s):
                continue
            n = len(s)
            row = dict(task=task, k=k, n=n)
            for col, stem in SERIES:
                row[f'{stem}_mean'] = s[col].mean()
                row[f'{stem}_sem'] = s[col].std(ddof=1) / np.sqrt(n)
            row['excess_mean'] = s.excess.mean()
            row['excess_sem'] = s.excess.std(ddof=1) / np.sqrt(n)
            row['p_wilcoxon'] = stats.wilcoxon(s.excess.values, alternative='greater').pvalue
            rows.append(row)
        if not rows:
            continue
        g = pd.DataFrame(rows)
        g['p_holm'] = multipletests(g['p_wilcoxon'].values, method='holm')[1]
        g['n_tests'] = len(g)
        frames.append(g)
    grp = pd.concat(frames, ignore_index=True)
    grp['stars'] = grp['p_holm'].map(p_stars)
    grp['shipped'] = grp['task'].isin(SHIPPED_TASKS)
    return grp[['task', 'k', 'n', 'unif_mean', 'unif_sem', 'wcat_mean', 'wcat_sem',
                'obs_mean', 'obs_sem', 'excess_mean', 'excess_sem',
                'p_wilcoxon', 'p_holm', 'n_tests', 'stars', 'shipped']]


def _fmt_p(p: float) -> str:
    return f"p<0.001" if p < 0.001 else f"p={p:.3f}"


def print_group_summary(grp: pd.DataFrame, df: pd.DataFrame) -> None:
    for task in grp.task.unique():
        g = grp[grp.task == task]
        tag = "" if task in SHIPPED_TASKS else "  [NOT SHIPPED -- computed, not plotted]"
        print(f"\n===== GROUP: {task} (N={int(g.n.iloc[0])}, "
              f"Holm over {int(g.n_tests.iloc[0])} tests){tag} =====", flush=True)
        for _, r in g.iterrows():
            s = df[(df.task == task) & (df.k == r.k)]
            n_sig = int((s.p_within_cat < 0.05).sum())
            print(f"  top-{int(r.k)}:  observed {r.obs_mean:.3f} | uniform null "
                  f"{r.unif_mean:.3f} | within-category null {r.wcat_mean:.3f} | "
                  f"excess {r.excess_mean:+.3f} | Wilcoxon p={r.p_wilcoxon:.3g} -> "
                  f"Holm p={r.p_holm:.3g} {r.stars}", flush=True)
            print(f"           {n_sig}/{len(s)} participants exceed their own "
                  f"category-only null (uncorrected, per-participant permutation; "
                  f"diagnostic, not reported)", flush=True)

    # Caption-ready block.  The caption transcribes these by hand; printing them in
    # caption form is the cheapest defence against transcription drift.
    print("\n===== caption-ready (Holm-adjusted within task) =====", flush=True)
    for task in grp.task.unique():
        g = grp[grp.task == task]
        body = "; ".join(f"top-{int(r.k)} {_fmt_p(r.p_holm)}" for _, r in g.iterrows())
        tag = "" if task in SHIPPED_TASKS else "   (not shipped)"
        print(f"  {task.replace('_', ' ')}: {body}{tag}", flush=True)


def main(argv=None) -> None:
    args = parse_args(argv)

    if args.group_only:
        if not os.path.exists(TOPK_CSV):
            raise FileNotFoundError(f"{TOPK_CSV} not found -- run without --group-only first.")
        df = pd.read_csv(TOPK_CSV)
        print(f"[within-cat null] --group-only: re-deriving from {TOPK_CSV}", flush=True)
    else:
        tasks = TASKS if args.task == 'both' else (args.task,)
        print("[within-cat null] loading GloVe (~10 s) ...", flush=True)
        glove = gallery_mod.load_glove()      # memoised singleton; one process pays once
        df = merge_tasks({t: run_task(t, glove, n_perm=args.n_perm, seed=args.seed)
                          for t in tasks})
        os.makedirs(os.path.dirname(TOPK_CSV), exist_ok=True)
        df.to_csv(TOPK_CSV, index=False)
        print(f"\n  wrote {TOPK_CSV}", flush=True)

    grp = group_stats(df)
    grp.to_csv(GROUP_CSV, index=False)
    print_group_summary(grp, df)
    print(f"\n  wrote {GROUP_CSV}", flush=True)


if __name__ == '__main__':
    main()
