# -*- coding: utf-8 -*-
"""Out-of-fold co-trained fit, its latent space, and the three candidate 2-D views.

One fit per fold serves all three views, so they describe the *same* model and the choice
between them is about presentation rather than about which decoder happened to be trained.

The three views (Alec, 2026-08-13 — "work up all three, then pick"):

  1. ``latent``   the co-trained PLS latent space itself. Literally "the low-D neural space
                  trained with the co-trainer": both tasks' trials are pushed through one
                  Nystroem-RBF map and one PLS projection, so they land in one space by
                  construction. Two of the ten components are plotted -- see
                  ``component_diagnostics`` for how they are chosen, which is the whole
                  question this view turns on.
  2. ``lda``      category-discriminant axes fitted on PICTURE trials in that latent space,
                  with AUDITORY trials projected into them. The strongest claim available
                  here -- axes defined by one task organising the other -- and the one most
                  likely to fail, since naive PN->AN transfer is at chance in this data.
  3. ``glove``    metric MDS on the cosine distances of the co-trained model's PREDICTED
                  GloVe vectors. A shared space by construction too, but an output space
                  rather than a neural one; this is the existing panel-a method with one
                  co-trained decoder in place of two separate ones.

Everything is computed on **out-of-fold** predictions with word-grouped folds, so no trial
is embedded by a model that saw it and no word appears in both sides of a fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression   # noqa: F401  (documents the stack)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import GroupKFold

from analysis.cross_task.cross_task_cotrain import (
    load_patient, make_model, _balance_pooled, N_PLS_COMPONENTS,
)

TASKS = ("picture", "auditory")

# One truncated category label ("object/too") predates the fix upstream and still appears in
# older runs; normalise so one category is never split across two colours or two centroids.
_CATEGORY_FIX = {"object/too": "object/tool"}


def _fix_cats(cats):
    return np.array([_CATEGORY_FIX.get(str(c), str(c)) for c in cats])


def pooled_arrays(patient, pic_run, aud_run, balance, seed):
    """Both tasks' peak-bin features stacked into one training set.

    Channels are already intersected and identically ordered by ``load_patient``, which is
    what makes the two tasks poolable at all. ``balance`` reuses the co-trainer's own
    ``_balance_pooled`` rather than a second implementation, so this pilot's pooled set is
    formed exactly the way the shipped figure's is (``downsample`` there).
    """
    pic, aud = load_patient(patient, pic_run, aud_run)
    rng = np.random.default_rng(seed)
    ip, ia = _balance_pooled(np.arange(len(pic["y"])), np.arange(len(aud["y"])),
                             balance, rng)
    X = np.vstack([pic["X"][ip], aud["X"][ia]])
    Y = np.vstack([pic["y"][ip], aud["y"][ia]])
    words = np.concatenate([pic["words"][ip], aud["words"][ia]]).astype(str)
    cats = _fix_cats(np.concatenate([pic["cats"][ip], aud["cats"][ia]]))
    task = np.array(["picture"] * len(ip) + ["auditory"] * len(ia))
    return X, Y, words, cats, task, pic["n_channels"]


def oof_latent_and_prediction(X, Y, words, n_folds, seed):
    """Word-grouped out-of-fold ``(latent scores, predicted GloVe)`` from ONE co-trained fit
    per fold.

    Grouping by word rather than by trial is deliberate and costs accuracy: a held-out trial
    whose word was in training is an easier target, and the question here is whether the two
    tasks share structure, not how high accuracy can go. The latent scores come from
    ``pls.transform(nys.transform(X))`` -- ``x_scores_`` is training-set only and would
    silently give in-sample positions for held-out trials.
    """
    uniq = np.unique(words)
    k = int(max(2, min(n_folds, len(uniq))))
    rng = np.random.default_rng(seed)
    order = {w: i for i, w in enumerate(rng.permutation(uniq))}
    groups = np.array([order[w] for w in words], dtype=int)

    n_comp = min(N_PLS_COMPONENTS, X.shape[0] - 1)
    Z = np.full((X.shape[0], n_comp), np.nan)
    Yhat = np.full(Y.shape, np.nan)
    for tr, te in GroupKFold(n_splits=k).split(X, Y, groups=groups):
        model = make_model("kernel_pls", len(tr))
        model.fit(X[tr], Y[tr])
        nys, pls = model.named_steps["nys"], model.named_steps["pls"]
        z = pls.transform(nys.transform(X[te]))
        Z[te, :z.shape[1]] = z
        Yhat[te] = model.predict(X[te])
    ok = np.isfinite(Z).all(axis=1) & np.isfinite(Yhat).all(axis=1)
    return Z, Yhat, ok


def _anova_f(v, labels):
    """One-way ANOVA F of a single component across category labels. NaN if degenerate."""
    groups = [v[labels == c] for c in np.unique(labels)]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return np.nan
    grand = np.concatenate(groups).mean()
    n_tot = sum(len(g) for g in groups)
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df_b, df_w = len(groups) - 1, n_tot - len(groups)
    if df_w <= 0 or ssw <= 0:
        return np.nan
    return float((ssb / df_b) / (ssw / df_w))


def _auc(v, positive):
    """Rank AUC of a single component separating ``positive`` from the rest, folded to
    [0.5, 1] because direction is arbitrary. 0.5 = the component says nothing about task."""
    pos, neg = v[positive], v[~positive]
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    a = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(max(a, 1 - a))


def component_diagnostics(Z, cats, task):
    """Per-component table: does it carry CATEGORY, and does it leak TASK?

    This exists because of the obvious objection to view 1. The co-trained PLS maximises
    covariance with GloVe, and task identity is not a target -- but the two tasks differ in
    trial count, amplitude and timing, so a nuisance direction that separates them can still
    dominate the leading components. If it does, plotting components 1-2 would show the
    tasks pulling apart, which is the opposite of the point the figure exists to make.

    ``cat_f_min`` is the ANOVA F across categories computed WITHIN each task and then
    minimised over the two. Minimum, not mean: a component that separates categories in
    picture only is not evidence of a shared space, and averaging would let a strong picture
    effect carry a null auditory one.

    ``task_auc`` is how well the component alone separates picture from auditory. 0.5 is
    clean; near 1.0 means the component is largely a task axis.

    Neither is a test, and the selection below is made on the same trials that are then
    plotted -- so this is a presentation choice, disclosed, not an inference. The full table
    ships so a reader can see what the unselected components looked like.
    """
    rows = []
    is_pic = task == "picture"
    for k in range(Z.shape[1]):
        v = Z[:, k]
        f_pic = _anova_f(v[is_pic], cats[is_pic])
        f_aud = _anova_f(v[~is_pic], cats[~is_pic])
        rows.append(dict(component=k + 1, cat_f_picture=f_pic, cat_f_auditory=f_aud,
                         cat_f_min=np.nanmin([f_pic, f_aud]),
                         task_auc=_auc(v, is_pic),
                         var=float(np.var(v))))
    return pd.DataFrame(rows)


def pick_components(diag, max_task_auc=0.70):
    """The two components to plot: highest ``cat_f_min`` among those that are not mainly
    task axes. Falls back to the top two by ``cat_f_min`` if the filter empties, and the
    caller records which happened -- a view built from task axes must not be presented as
    though it were built from category ones.
    """
    ok = diag[diag["task_auc"] <= max_task_auc].dropna(subset=["cat_f_min"])
    rule = "cat_f_min, task_auc <= {:.2f}".format(max_task_auc)
    if len(ok) < 2:
        ok = diag.dropna(subset=["cat_f_min"])
        rule = "cat_f_min only (task_auc filter left <2 components)"
    top = ok.sort_values("cat_f_min", ascending=False).head(2)["component"].tolist()
    return sorted(int(c) for c in top), rule


def word_means(task, words, cats, *arrays):
    """Collapse trials to one point per (task, word). Returns (task, words, cats, *arrays).

    The single-trial clouds overlap almost completely in every view -- that is the finding,
    not a plotting failure -- so the readable unit is the word, not the trial. Averaging the
    repeats of a word within a task removes the per-trial noise while keeping the thing the
    decoder is actually asked about: a word's position in semantic space. It also equalises
    words, which matters because repeat counts are very uneven (auditory naming has few
    repeats), so a trial-level centroid is silently weighted by how often a word happened to
    be presented.

    NB this is an average of held-out predictions, so it does not leak: every trial entering
    the mean was predicted by a model that never saw it.
    """
    key = pd.DataFrame({"task": task, "word": words, "cat": cats})
    grp = key.groupby(["task", "word"], sort=True)
    idx = list(grp.indices.values())
    out_task = np.array([task[i[0]] for i in idx])
    out_word = np.array([words[i[0]] for i in idx])
    out_cat = np.array([cats[i[0]] for i in idx])
    reduced = [np.vstack([a[i].mean(axis=0) for i in idx]) for a in arrays]
    return (out_task, out_word, out_cat, *reduced)


def per_category_cosine(E, cats, task, n_shuffle, seed):
    """Observed cross-task cosine per category, plus its category-shuffle null.

    ``category_centroid_alignment`` returns the cohort statistic and one p-value; this
    returns the PER-CATEGORY value and the null it should be read against, because the
    question the panel has to answer is "is *this* category's shift meaningful", and a single
    figure-level p cannot answer it. Same construction as the shared statistic: each task's
    points are mean-centred first (kernel-PLS shrinks predictions toward the mean, so raw
    centroids all point the same way), then the two tasks' centroid directions are compared.
    """
    is_pic = task == "picture"
    shared = sorted(set(cats[is_pic]) & set(cats[~is_pic]))
    if len(shared) < 2:
        return pd.DataFrame()
    pc = E[is_pic] - E[is_pic].mean(axis=0, keepdims=True)
    ac = E[~is_pic] - E[~is_pic].mean(axis=0, keepdims=True)
    cp, ca = cats[is_pic], cats[~is_pic]

    def _cent(M, lab, c):
        v = M[lab == c].mean(axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    obs = {c: float(np.dot(_cent(pc, cp, c), _cent(ac, ca, c))) for c in shared}
    rng = np.random.default_rng(seed + 11)
    null = {c: [] for c in shared}
    for _ in range(n_shuffle):
        perm = rng.permutation(ca)
        for c in shared:
            if (perm == c).sum() >= 1:
                null[c].append(float(np.dot(_cent(pc, cp, c), _cent(ac, perm, c))))
    rows = []
    for c in shared:
        d = np.asarray(null[c], dtype=float)
        rows.append(dict(category=c, cosine=obs[c],
                         null_p95=float(np.percentile(d, 95)) if d.size else np.nan,
                         null_mean=float(d.mean()) if d.size else np.nan,
                         p=float((1 + (d >= obs[c]).sum()) / (d.size + 1)) if d.size else np.nan,
                         n_picture=int((cp == c).sum()), n_auditory=int((ca == c).sum())))
    return pd.DataFrame(rows)


# ── the three views ────────────────────────────────────────────────────────

def view_latent(Z, comps):
    """Two chosen columns of the co-trained latent space, as-is. No further fitting."""
    return Z[:, [c - 1 for c in comps]]


def view_lda(Z, cats, task, seed):
    """Category-discriminant axes fitted on PICTURE trials only, auditory projected in.

    The axes never see an auditory trial, so auditory structure along them cannot be fitted
    into existence. Needs >= 3 picture categories with >= 2 trials each for two discriminants;
    returns None when that is not met rather than silently dropping to one axis.
    """
    is_pic = task == "picture"
    zc, cc = Z[is_pic], cats[is_pic]
    keep = np.isin(cc, [c for c, n in zip(*np.unique(cc, return_counts=True)) if n >= 2])
    zc, cc = zc[keep], cc[keep]
    if len(np.unique(cc)) < 3:
        return None
    lda = LinearDiscriminantAnalysis(n_components=2, solver="eigen", shrinkage="auto")
    lda.fit(zc, cc)
    return lda.transform(Z)


def view_glove(Yhat, seed):
    """Metric MDS on cosine distances of the predicted GloVe vectors, both tasks jointly.

    Run on the WORD-MEAN predictions, so the subsampling the trial-level version needed
    (MDS is O(n^2)) is gone: a participant has at most a few hundred (task, word) rows.
    That also removes a subtle problem with the trial-level version -- MDS placed each trial
    independently, so a word with many repeats got many nearly-identical points and pulled
    the layout toward itself.
    """
    D = cosine_distances(Yhat)
    return MDS(n_components=2, dissimilarity="precomputed", random_state=seed,
               normalized_stress=False).fit_transform(D)
