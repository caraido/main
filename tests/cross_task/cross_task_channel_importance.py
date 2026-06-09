"""Per-channel importance for the co-trained (pooled pic+aud) semantic model.

Answers: which channels drive *both* tasks' retrieval accuracy, which are
picture-only, and which are auditory-only — for the SAME pooled kernel-PLS
model used in ``cross_task_cotrain.py``.

Two complementary attributions are produced per channel, evaluated on the
picture test set and the auditory test set separately:

  1. Permutation importance  (Δ word_bal_acc when the channel's full history
     block is shuffled across trials).  This is the "contributes to accuracy"
     measure and is robust to the RBF non-linearity.  Significance: a per-
     bootstrap label-shuffle null gives the noise floor of Δacc; one-sided
     p-values are pooled across bootstraps and BH-FDR corrected.

  2. Analytic Jacobian sensitivity  (mean ‖∂ŷ/∂x‖ back-propagated through the
     Nystroem-RBF map and the PLS affine map, aggregated over the channel's
     history columns).  This is the faithful local "back-projection through the
     kernel"; it scores sensitivity of the predicted GloVe embedding, not
     accuracy, so it is reported as a cross-check rather than for the grouping.

Grouping (permutation-null significance):
    both        : sig. positive Δacc in BOTH tasks
    picture_only: sig. positive Δacc in pic only
    auditory_only: sig. positive Δacc in aud only
    neither     : sig. in neither

Memory note: this loads the per-patient semantic_regression_results.pkl
(100 MB – 2.6 GB each) via cross_task_cotrain.load_patient, so run it on a
machine with enough RAM (the project README recommends 16 GB+).  Run e.g.:

    python -m main.tests.cross_task.cross_task_channel_importance
    python -m main.tests.cross_task.cross_task_channel_importance --patient RB
    python -m main.tests.cross_task.cross_task_channel_importance \
        --n-bootstrap 20 --n-perm-repeats 5 --null-shuffles 3
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path setup (mirror cross_task_cotrain so `tests`/`utils` resolve when run
#    either as a module or as a script) ──────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_THIS_DIR)
_MAIN_DIR = os.path.dirname(_TESTS_DIR)
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

# Reuse the co-training pipeline so the model + data are identical.
from tests.cross_task.cross_task_cotrain import (
    load_patient, make_model, _build_db, _score, _norm,
    _stratified_word_split, _balance_pooled,
    PIC_RUN_DEFAULT, AUD_RUN_DEFAULT, SHARED_PATIENTS, OUT_ROOT,
)
from utils.retrieval import compute_retrieval_metrics


# ── helpers ────────────────────────────────────────────────────────────────
METRICS = ("cat_indep_bal_acc", "word_bal_acc")   # default first (more robust)
_METRIC_TAG = {"cat_indep_bal_acc": "catindep", "word_bal_acc": "word"}


def _metric_value(Y_pred: np.ndarray, words: np.ndarray, cats: np.ndarray,
                  db: tuple, metric: str) -> float:
    m = compute_retrieval_metrics(Y_pred, words, cats, *db)
    return float(m[metric])


def _channel_columns(c: int, n_ch: int, n_hist: int) -> np.ndarray:
    """Column indices of channel *c* across all history bins (X layout is
    ``channels + b*n_ch`` — see _word_means in cross_task_cotrain)."""
    return c + n_ch * np.arange(n_hist)


def permutation_importance(model, X_te, words_te, cats_te, db,
                           n_ch: int, n_hist: int, n_repeats: int,
                           rng: np.random.Generator, metric: str) -> np.ndarray:
    """Δ <metric> per channel (baseline − permuted), averaged over repeats."""
    base = _metric_value(model.predict(X_te), words_te, cats_te, db, metric)
    n_te = X_te.shape[0]
    drops = np.zeros(n_ch)
    for c in range(n_ch):
        cols = _channel_columns(c, n_ch, n_hist)
        acc = 0.0
        for _ in range(n_repeats):
            Xp = X_te.copy()
            perm = rng.permutation(n_te)
            Xp[:, cols] = Xp[perm][:, cols]      # same row-perm across the block
            acc += _metric_value(model.predict(Xp), words_te, cats_te, db, metric)
        drops[c] = base - acc / n_repeats
    return drops


def null_importance(model, X_te, words_te, cats_te, db,
                    n_ch: int, n_hist: int, n_shuffles: int,
                    rng: np.random.Generator, metric: str) -> np.ndarray:
    """Pooled null of Δacc: under shuffled trial labels every channel is
    irrelevant, so its Δacc reflects only sampling noise.  Returns a flat
    array of null Δacc values (n_shuffles × n_ch)."""
    nulls: List[float] = []
    n_te = X_te.shape[0]
    base_pred = model.predict(X_te)
    for _ in range(n_shuffles):
        sh = rng.permutation(n_te)            # break Y_pred <-> label alignment
        w_sh, c_sh = words_te[sh], cats_te[sh]
        base = _metric_value(base_pred, w_sh, c_sh, db, metric)
        for c in range(n_ch):
            cols = _channel_columns(c, n_ch, n_hist)
            Xp = X_te.copy()
            Xp[:, cols] = Xp[rng.permutation(n_te)][:, cols]
            nulls.append(base - _metric_value(model.predict(Xp), w_sh, c_sh, db, metric))
    return np.asarray(nulls)


def _pls_affine(model, n_targets: int) -> Tuple[np.ndarray, np.ndarray]:
    """Recover the PLS affine map  ŷ = φ @ A + b  empirically (orientation-
    agnostic w.r.t. sklearn's coef_ convention)."""
    pls = model.named_steps["pls"]
    n_feat = model.named_steps["nys"].n_components
    b = pls.predict(np.zeros((1, n_feat)))[0]            # (n_targets,)
    A = pls.predict(np.eye(n_feat))                       # (n_feat, n_targets)
    A = A - b[None, :]
    return A, b


def jacobian_sensitivity(model, X_te, n_ch: int, n_hist: int) -> np.ndarray:
    """Mean ‖∂ŷ/∂x‖₂ per input feature, aggregated to per-channel.

    φ(x) = k(x) @ N^T  with  k_j(x) = exp(-γ‖x - z_j‖²),  z_j = landmarks.
    ∂φ_m/∂x = Σ_j N_{mj} k_j (-2γ)(x - z_j);   ∂ŷ/∂x = Jφ^T @ A.
    """
    nys = model.named_steps["nys"]
    Z = nys.components_                                   # (L, d)
    N = nys.normalization_                                # (L, L)
    gamma = nys.gamma if nys.gamma is not None else 1.0 / nys.n_features_in_
    A, _ = _pls_affine(model, n_targets=None)             # (L, n_targets)

    d = X_te.shape[1]
    sens = np.zeros(d)
    for x in X_te:
        diff = x[None, :] - Z                             # (L, d)
        k = np.exp(-gamma * np.sum(diff * diff, axis=1))  # (L,)
        dk = (-2.0 * gamma) * (k[:, None] * diff)         # ∂k_j/∂x  (L, d)
        Jphi = N @ dk                                     # (L, d)
        J = Jphi.T @ A                                    # (d, n_targets)
        sens += np.linalg.norm(J, axis=1)
    sens /= X_te.shape[0]
    return sens.reshape(n_hist, n_ch).sum(axis=0)         # per-channel


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        adj[i] = prev
    return adj


# ── per-patient analysis ─────────────────────────────────────────────────
def analyze_patient(patient: str, pic_run: str, aud_run: str,
                    n_bootstrap: int, test_frac: float, zero_shot_frac: float,
                    balance: str, n_perm_repeats: int, null_shuffles: int,
                    alpha: float, rng_seed: int, metric: str) -> pd.DataFrame:
    pic, aud = load_patient(patient, pic_run, aud_run)
    n_ch, n_hist = pic["n_channels"], pic["n_hist"]
    chan_names = pic["chan_names"]
    db_pic, db_aud = _build_db(pic), _build_db(aud)
    shared = np.array(sorted(set(pic["words"]) & set(aud["words"])))
    rng = np.random.default_rng(rng_seed)

    imp_pic = np.zeros((n_bootstrap, n_ch))
    imp_aud = np.zeros((n_bootstrap, n_ch))
    jac_pic = np.zeros((n_bootstrap, n_ch))
    jac_aud = np.zeros((n_bootstrap, n_ch))
    null_pic, null_aud = [], []
    used = 0

    for b in range(n_bootstrap):
        n_zs = int(round(len(shared) * zero_shot_frac))
        unseen = (set(rng.choice(shared, n_zs, replace=False).tolist())
                  if n_zs > 0 else set())
        p_tr, p_te = _stratified_word_split(pic["words"], unseen, test_frac, rng)
        a_tr, a_te = _stratified_word_split(aud["words"], unseen, test_frac, rng)
        if min(len(p_tr), len(a_tr), len(p_te), len(a_te)) < 3:
            continue
        bp, ba = _balance_pooled(p_tr, a_tr, balance, rng)
        X_pool = np.vstack([pic["X"][bp], aud["X"][ba]])
        y_pool = np.vstack([pic["y"][bp], aud["y"][ba]])

        model = make_model("kernel_pls", X_pool.shape[0], None)
        model.fit(X_pool, y_pool)

        imp_pic[used] = permutation_importance(
            model, pic["X"][p_te], pic["words"][p_te], pic["cats"][p_te],
            db_pic, n_ch, n_hist, n_perm_repeats, rng, metric)
        imp_aud[used] = permutation_importance(
            model, aud["X"][a_te], aud["words"][a_te], aud["cats"][a_te],
            db_aud, n_ch, n_hist, n_perm_repeats, rng, metric)
        jac_pic[used] = jacobian_sensitivity(model, pic["X"][p_te], n_ch, n_hist)
        jac_aud[used] = jacobian_sensitivity(model, aud["X"][a_te], n_ch, n_hist)
        if null_shuffles > 0:
            null_pic.append(null_importance(
                model, pic["X"][p_te], pic["words"][p_te], pic["cats"][p_te],
                db_pic, n_ch, n_hist, null_shuffles, rng, metric))
            null_aud.append(null_importance(
                model, aud["X"][a_te], aud["words"][a_te], aud["cats"][a_te],
                db_aud, n_ch, n_hist, null_shuffles, rng, metric))
        used += 1
        print(f"  {patient}: bootstrap {used}/{n_bootstrap}")

    imp_pic, imp_aud = imp_pic[:used], imp_aud[:used]
    jac_pic, jac_aud = jac_pic[:used], jac_aud[:used]
    obs_pic, obs_aud = imp_pic.mean(0), imp_aud.mean(0)

    # one-sided p: compare each bootstrap's observed Δacc against that bootstrap's
    # null so both are on the same scale (mean-of-bootstraps vs. individual-bootstrap
    # null inflates the null by sqrt(n_bootstrap), making nothing significant).
    if null_pic:
        p_pic_boots = np.zeros((used, n_ch))
        p_aud_boots = np.zeros((used, n_ch))
        for b in range(used):
            nl_p, nl_a = null_pic[b], null_aud[b]
            for c in range(n_ch):
                p_pic_boots[b, c] = (1 + np.sum(nl_p >= imp_pic[b, c])) / (1 + len(nl_p))
                p_aud_boots[b, c] = (1 + np.sum(nl_a >= imp_aud[b, c])) / (1 + len(nl_a))
        p_pic = p_pic_boots.mean(0)
        p_aud = p_aud_boots.mean(0)
    else:
        p_pic = np.ones(n_ch)
        p_aud = np.ones(n_ch)
    q_pic, q_aud = _bh_fdr(p_pic), _bh_fdr(p_aud)

    sig_pic = (q_pic < alpha) & (obs_pic > 0)
    sig_aud = (q_aud < alpha) & (obs_aud > 0)
    group = np.where(sig_pic & sig_aud, "both",
             np.where(sig_pic, "picture_only",
             np.where(sig_aud, "auditory_only", "neither")))

    return pd.DataFrame({
        "patient": patient,
        "metric": metric,
        "channel": chan_names[:n_ch],
        "perm_imp_pic": obs_pic, "perm_imp_aud": obs_aud,
        "p_pic": p_pic, "p_aud": p_aud, "q_pic": q_pic, "q_aud": q_aud,
        "jac_sens_pic": jac_pic.mean(0), "jac_sens_aud": jac_aud.mean(0),
        "group": group,
    }).sort_values(["group", "perm_imp_pic"], ascending=[True, False])


# ── plotting ───────────────────────────────────────────────────────────────
_GCOL = {"both": "#2ca02c", "picture_only": "#1f77b4",
         "auditory_only": "#d62728", "neither": "#bbbbbb"}


def _scatter(df: pd.DataFrame, xcol: str, ycol: str, title: str,
             xlab: str, ylab: str, out: Path, annotate_top: int = 5) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    for g, sub in df.groupby("group"):
        ax.scatter(sub[xcol], sub[ycol], s=26, c=_GCOL.get(g, "#777"),
                   label=f"{g} (n={len(sub)})", alpha=0.8, edgecolors="none")
    top = df.assign(_m=df[[xcol, ycol]].min(axis=1)).nlargest(annotate_top, "_m")
    for _, r in top.iterrows():
        ax.annotate(str(r["channel"]), (r[xcol], r[ycol]), fontsize=8,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient", nargs="*", default=SHARED_PATIENTS)
    ap.add_argument("--pic-run", default=PIC_RUN_DEFAULT)
    ap.add_argument("--aud-run", default=AUD_RUN_DEFAULT)
    ap.add_argument("--n-bootstrap", type=int, default=20)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--zero-shot-frac", type=float, default=0.3)
    ap.add_argument("--balance", default="none",
                    choices=["none", "downsample", "upsample"])
    ap.add_argument("--n-perm-repeats", type=int, default=5)
    ap.add_argument("--null-shuffles", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metric", nargs="+", default=["cat_indep_bal_acc"],
                    choices=list(METRICS),
                    help="Retrieval metric(s) driving the permutation importance "
                         "+ grouping. cat_indep_bal_acc (default) is more robust "
                         "than word_bal_acc; pass both to run each.")
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for metric in args.metric:
        tag = _METRIC_TAG[metric]
        for pat in args.patient:
            print(f"[{pat}] analysing channel importance (metric={metric}) …")
            try:
                df = analyze_patient(
                    pat, args.pic_run, args.aud_run, args.n_bootstrap,
                    args.test_frac, args.zero_shot_frac, args.balance,
                    args.n_perm_repeats, args.null_shuffles, args.alpha,
                    args.seed, metric)
            except Exception as exc:
                print(f"  [{pat}] FAILED: {type(exc).__name__}: {exc}")
                continue
            pdir = out_root / pat
            pdir.mkdir(parents=True, exist_ok=True)
            df.to_csv(pdir / f"channel_importance_{pat}_{tag}.csv", index=False)
            _scatter(df, "perm_imp_pic", "perm_imp_aud",
                     f"{pat} · permutation importance (Δ {metric})",
                     f"Δ{tag} picture", f"Δ{tag} auditory",
                     pdir / f"channel_importance_{pat}_{tag}.png")
            _scatter(df, "jac_sens_pic", "jac_sens_aud",
                     f"{pat} - Jacobian sensitivity (|grad-y / grad-x|)",
                     "sensitivity picture", "sensitivity auditory",
                     pdir / f"channel_jacobian_{pat}_{tag}.png")
            all_rows.append(df)
            print(f"  [{pat}/{tag}] groups: " +
                  ", ".join(f"{g}={int((df['group']==g).sum())}"
                            for g in ["both", "picture_only", "auditory_only", "neither"]))
            gc.collect()

    if all_rows:
        alldf = pd.concat(all_rows, ignore_index=True)
        alldf.to_csv(out_root / "channel_importance_all.csv", index=False)
        summary = (alldf.groupby(["patient", "metric", "group"]).size()
                   .unstack(fill_value=0))
        summary.to_csv(out_root / "channel_importance_group_counts.csv")
        print("\nGroup counts per patient:\n", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
