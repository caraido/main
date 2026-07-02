# -*- coding: utf-8 -*-
"""
tests/open_vocab_retrieval/figures.py
=====================================
Step 8 + §6 of the guide: publication figures.

  1. metric-vs-N curve      — median percentile rank (+ top-1/5/10) vs gallery
                              size, chance line, matched vs raw.
  2. CMC curve              — top-k accuracy vs k at the headline N (chance k/N).
  3. in-vocab vs held-out   — the Claim-2 panel (in-vocab > held-out > chance).
  4. near-miss / nDCG       — observed nDCG / near-miss similarity vs matched null.
  5. qualitative table      — true -> top-5 retrieved with the independent grade
                              (illustrative, written as CSV/HTML).

All plotting functions take already-computed DataFrames and an output PNG path;
the caller (run.py) is responsible for writing the source-data CSVs under
``figures/open_vocab_retrieval/source_data/`` per the repo convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Fig 1: metric vs N ────────────────────────────────────────────────────

def plot_metric_vs_N(summary_df: pd.DataFrame, out_path: Path,
                     ks: Sequence[int] = (1, 5, 10)) -> None:
    """Median percentile rank and top-k vs gallery size, per variant."""
    variants = sorted(summary_df["variant"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    colors = {"matched": "#1f77b4", "raw": "#d62728"}

    ax = axes[0]
    for v in variants:
        sub = summary_df[summary_df["variant"] == v].sort_values("N")
        ax.errorbar(sub["N"].to_numpy(), sub["median_percentile_mean"].to_numpy(),
                    yerr=sub["median_percentile_sem"].to_numpy(), marker="o",
                    capsize=3, color=colors.get(v, None), label=f"{v}")
    ax.axhline(0.5, ls="--", color="grey", label="chance (0.5)")
    ax.set_xscale("log"); ax.set_xlabel("Gallery size N (log)")
    ax.set_ylabel("Median percentile rank (lower = better)")
    ax.set_title("N-robust headline metric")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    linestyles = ["-", "--", ":", "-."]
    for v in variants:
        sub = summary_df[summary_df["variant"] == v].sort_values("N")
        for ki, k in enumerate(ks):
            col = f"top{k}_mean"
            if col in sub.columns:
                ax.plot(sub["N"].to_numpy(), sub[col].to_numpy(), marker="o",
                        color=colors.get(v, None),
                        ls=linestyles[ki % len(linestyles)],
                        label=f"{v} top{k}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Gallery size N (log)")
    ax.set_ylabel("Top-k accuracy (log)")
    ax.set_title("Top-k vs N (chance = k/N)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    fig.suptitle("Open-vocabulary retrieval: metric vs gallery size", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── Fig 2: CMC curve ──────────────────────────────────────────────────────

def plot_cmc(sweep_df: pd.DataFrame, N: int, variant: str, out_path: Path,
             ks: Sequence[int] = (1, 5, 10, 50, 100)) -> None:
    """CMC (top-k vs k) at a fixed headline N, cross-patient mean, with chance."""
    sub = sweep_df[(sweep_df["N"] == N) & (sweep_df["variant"] == variant)]
    if sub.empty:
        return
    ks = [k for k in ks if f"top{k}" in sub.columns]
    means = np.array([sub[f"top{k}"].mean() for k in ks], dtype=float)
    sems = np.array([sub[f"top{k}"].sem() for k in ks], dtype=float)
    N_eff = float(sub["N_effective"].mean())
    ks_arr = np.array(ks, dtype=float)
    chance = ks_arr / N_eff

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.errorbar(ks_arr, means, yerr=sems, marker="o", capsize=3, color="#1f77b4",
                label="observed")
    ax.plot(ks_arr, chance, ls="--", color="grey", label="chance (k/N)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("k"); ax.set_ylabel("CMC(k) = top-k accuracy")
    ax.set_title(f"CMC curve — {variant} gallery, N={N}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── Fig 3: in-vocab vs held-out vs chance ─────────────────────────────────

def plot_invocab_vs_heldout(group_df: pd.DataFrame, out_path: Path) -> None:
    """Per-patient median percentile rank for in-vocab vs held-out, with the
    chance line — the Claim-2 panel (expect in-vocab < held-out < 0.5)."""
    patients = group_df["patient"].tolist()
    x = np.arange(len(patients))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(patients)), 4.2))
    ax.bar(x - w / 2, group_df["median_percentile_invocab"].to_numpy(), w,
           color="#1f77b4", label="in-vocab")
    ax.bar(x + w / 2, group_df["median_percentile_heldout"].to_numpy(), w,
           color="#ff7f0e", label="held-out (zero-shot)")
    ax.axhline(0.5, ls="--", color="grey", label="chance (0.5)")
    ax.set_xticks(x); ax.set_xticklabels(patients)
    ax.set_ylabel("Median percentile rank (lower = better)")
    ax.set_title("Zero-shot generalization: in-vocab vs held-out vs chance")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── Fig 4: near-miss / nDCG vs null ───────────────────────────────────────

def plot_near_miss(group_df: pd.DataFrame, out_path: Path) -> None:
    """Observed nDCG and near-miss similarity vs their permutation null means,
    per patient."""
    patients = group_df["patient"].tolist()
    x = np.arange(len(patients))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    ax.bar(x, group_df["ndcg_mean"].to_numpy(), color="#2ca02c", label="observed nDCG")
    if "ndcg_null_mean" in group_df.columns:
        ax.plot(x, group_df["ndcg_null_mean"].to_numpy(), "kx", label="null mean")
    ax.set_xticks(x); ax.set_xticklabels(patients)
    ax.set_ylabel("nDCG (independent WordNet relevance)")
    ax.set_title("Graded near-miss: nDCG vs null"); ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x, group_df["near_miss_sim_mean"].to_numpy(), color="#9467bd",
           label="observed top-k similarity")
    if "near_miss_null_mean" in group_df.columns:
        ax.plot(x, group_df["near_miss_null_mean"].to_numpy(), "kx", label="matched null mean")
    ax.set_xticks(x); ax.set_xticklabels(patients)
    ax.set_ylabel("Mean WordNet similarity of top-k neighbours")
    ax.set_title("Near-miss similarity vs null"); ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── Fig 6: qualitative true -> top-5 table ────────────────────────────────

def qualitative_table(predictions, gallery, rel_fn, out_csv: Path,
                      out_html: Optional[Path] = None,
                      n_per_patient: int = 6, center: bool = True,
                      seed: int = 0) -> pd.DataFrame:
    """Emit a ``true -> top-5 retrieved`` table with the independent grade per
    retrieved word.  ILLUSTRATIVE, not evidentiary (the evidence is Steps 4-6).

    Two deliberate choices so the table is representative rather than a cherry of
    the noisiest trials:

      * **per-word mean prediction** — the query is the mean predicted embedding
        across all of a word's trials (the same denoising the mean-per-word
        retrieval DB uses), not a single high-variance trial;
      * **spread across patients** — up to ``n_per_patient`` distinct words are
        shown for EVERY patient (a mix of in-vocab and held-out), so the table is
        not dominated by whoever happens to be first.

    The true word's own gallery entry is excluded from its top-5 so a correct
    rank-1 does not display as trivially retrieving itself.
    """
    from .retrieval import similarity_matrix, ranked_indices

    rng = np.random.default_rng(seed)
    recs = []
    for tp in predictions:
        words = np.asarray(tp.true_word)
        uniq = np.unique(words)
        # per-word mean predicted embedding + held-out flag
        means, held = [], []
        for w in uniq:
            m = words == w
            means.append(tp.pred_emb[m].mean(axis=0))
            held.append(bool(tp.is_held_out[m][0]))
        means = np.vstack(means)
        held = np.array(held)

        sims = similarity_matrix(means, gallery.emb, center=center)
        order = ranked_indices(sims)

        # sample a mix: prefer some held-out and some in-vocab words
        chosen = []
        for pool in (np.where(held)[0], np.where(~held)[0]):
            pool = pool.copy(); rng.shuffle(pool)
            chosen.extend(pool[: max(1, n_per_patient // 2)].tolist())
        chosen = list(dict.fromkeys(chosen))[:n_per_patient]

        for wi in chosen:
            tw = uniq[wi]
            top5 = [gallery.words[j] for j in order[wi] if gallery.words[j] != tw][:5]
            grades = [round(rel_fn(tw, w), 3) for w in top5]
            recs.append({
                "patient": tp.patient, "true_word": tw,
                "is_held_out": bool(held[wi]),
                "top1": top5[0], "top2": top5[1], "top3": top5[2],
                "top4": top5[3], "top5": top5[4],
                "grades": ";".join(str(g) for g in grades),
            })

    df = pd.DataFrame(recs).sort_values(["patient", "is_held_out"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    if out_html is not None:
        html = ("<h3>Illustrative: decoded true word &rarr; top-5 retrieved "
                "(open gallery, per-word mean prediction)</h3>"
                "<p><i>Illustrative only — the evidence is the rank / near-miss "
                "metrics with permutation significance. Query = mean predicted "
                "embedding across a word's trials; the true word is excluded from "
                "its own top-5. Grades are the independent WordNet relevance.</i></p>"
                + df.to_html(index=False))
        out_html.write_text(html, encoding="utf-8")
    return df
