"""
Supplementary figure — within-category (category-preserving) null for word-level decoding.

Tests whether the semantic decoder resolves WORD IDENTITY beyond CATEGORY. The
category-preserving null permutes the word<->trial correspondence *within* each
semantic category (category structure preserved, sub-category identity destroyed)
and recomputes top-k. Any excess of observed over this null is word-level information.

Produces three panels and one combined supplementary figure:
    (a) decomposition   uniform chance -> category-only null -> observed  (cohort mean +/- SEM)
    (b) excess vs k     word-level excess (obs - category-null) across k, per patient + cohort
    (c) forest (top-5)  per-patient observed vs the category-only 95% null band

Lives beside the figure it supplements. It used to sit in its own
figures_for_paper/within_category_null/ folder, a half-finished split that never produced
anything; moved here 2026-08-11 on Alec's call that this belongs as a supplementary figure
under semantic_regression.

Two modules share this analysis, and the division is the compute/render seam:
    within_category_null.py         computes within_category_null_topk.csv and renders the
                                    headline 12_within_category_null.{png,pdf}
    within_category_null_panels.py  this file -- renders the supplementary breakdown from
                                    that CSV, computing nothing

Run (VSCode "Run Python File", or terminal):
    python figures_for_paper/semantic_regression/within_category_null_panels.py
Reads the tracked source CSV written by within_category_null.py and writes
S5_within_category_null_{decomposition,excess_vs_k,forest_topk} plus the combined
S5_within_category_null.{png,pdf} into this folder. S5 because 00-12 are taken by the main
panels and the repo numbers supplementary figures S1, S2, ....
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # remove this line to show interactively in VSCode
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                      # figures_for_paper/
from paper_common import (apply_paper_style, ALPHA,       # noqa: E402
                          DPI_PANEL, DPI_COMBINED)
from utils.paths import paper_source_data                 # noqa: E402  (paper_common adds main/)

# ---------------------------------------------------------------- style
# House style first: editable vector text (fonttype 42) and the repo-wide type sizes.
# This module used to set its own rcParams and skip apply_paper_style entirely, which
# figures_for_paper/README.md §3 forbids -- the visible cost is PDF text that is no longer
# selectable, which only shows up at submission.
apply_paper_style()

# These are ROLE colours (which null, which observation), not participant identity.
# Participant identity comes from participants.json via paper_common and must never be
# hard-coded; nothing here assigns a colour per participant.
INK   = "#16202A"; TEAL  = "#0F6E6A"; TEALD = "#0B4F4C"
AMBER = "#C2670F"; AMBERD= "#9A5109"
GRAY  = "#6B7580"; LGRAY = "#AEB6BF"
KS = [1, 3, 5]

#: Rendered as "p<0.05" etc. Derived from utils.config.ALPHA so the annotation cannot
#: outlive the threshold it describes -- README §5.
ALPHA_LABEL = f"p<{ALPHA:g}"


# ---------------------------------------------------------------- data
def find_csv() -> Path:
    """The one location this figure's input lives in.

    The producer is figures_for_paper/semantic_regression/within_category_null.py, which
    writes into ITS OWN source_data/ -- the CSV is tracked there, beside the shipped
    12_within_category_null figure. This used to probe four candidate locations, three of
    which never held the file; a four-way probe is a way of not knowing where your input
    is. One sanctioned accessor instead.
    """
    csv = paper_source_data("semantic_regression", "within_category_null_topk.csv",
                            create=False)
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found -- run "
            f"figures_for_paper/semantic_regression/within_category_null.py first."
        )
    return csv


def load() -> pd.DataFrame:
    df = pd.read_csv(find_csv())
    df["excess"] = df["obs"] - df["wcat_mean"]        # recompute defensively
    return df


def cohort_stats(df: pd.DataFrame) -> dict:
    """Per-k cohort means, SEM across patients, and one-sided Wilcoxon on excess>0."""
    out = {}
    for k in KS:
        s = df[df.k == k]
        n = len(s)
        p = (stats.wilcoxon(s.excess.values, alternative="greater").pvalue
             if HAVE_SCIPY and n > 0 else np.nan)
        out[k] = dict(
            unif=s.unif_mean.mean(), unif_se=s.unif_mean.std(ddof=1) / np.sqrt(n),
            wcat=s.wcat_mean.mean(), wcat_se=s.wcat_mean.std(ddof=1) / np.sqrt(n),
            obs=s.obs.mean(),       obs_se=s.obs.std(ddof=1) / np.sqrt(n),
            p=p,
        )
    return out


def pstr(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


# ---------------------------------------------------------------- panels (draw into a given Axes)
def panel_decomposition(ax, df, coh):
    x = np.arange(3); w = 0.26
    u  = [coh[k]["unif"] for k in KS]; c  = [coh[k]["wcat"] for k in KS]; o  = [coh[k]["obs"] for k in KS]
    ue = [coh[k]["unif_se"] for k in KS]; ce = [coh[k]["wcat_se"] for k in KS]; oe = [coh[k]["obs_se"] for k in KS]
    ax.bar(x - w, u, w, yerr=ue, color=LGRAY, capsize=3, ec="white", label="uniform chance (no information)")
    ax.bar(x,     c, w, yerr=ce, color=TEAL,  capsize=3, ec="white", label="category-only null")
    ax.bar(x + w, o, w, yerr=oe, color=AMBER, capsize=3, ec="white", label="observed decoder")
    for i, k in enumerate(KS):                       # word-level excess bracket + p per group
        xx = x[i] + w
        ax.annotate("", xy=(xx, o[i]), xytext=(xx, c[i]),
                    arrowprops=dict(arrowstyle="<->", color=AMBERD, lw=1.4))
        ax.text(xx + 0.05, (c[i] + o[i]) / 2, f"word\nexcess\n{pstr(coh[k]['p'])}",
                fontsize=8.6, color=AMBERD, va="center", ha="left", fontweight="bold")
    ax.annotate("", xy=(x[2], c[2]), xytext=(x[2], u[2]),  # category component on k=5
                arrowprops=dict(arrowstyle="<->", color=TEALD, lw=1.4))
    ax.text(x[2] - 0.04, (u[2] + c[2]) / 2, "category\ncomponent",
            fontsize=8.6, color=TEALD, va="center", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(["top-1", "top-3", "top-5"])
    ax.set_ylabel("retrieval accuracy (cohort mean \u00B1 SEM)")
    ax.set_title("Retrieval decomposes into category + word components",
                 fontsize=12.5, fontweight="bold", color=INK, pad=8)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_ylim(0, max(o) * 1.28); ax.margins(x=0.06)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def panel_excess_vs_k(ax, df, coh):
    piv = df.pivot_table(index="display_id", columns="k", values="excess")
    sig5 = set(df[(df.k == 5) & (df.p_within_cat < ALPHA)].display_id)
    n_total = len(piv)
    for pid, row in piv.iterrows():
        on = pid in sig5
        ax.plot(KS, [row[k] for k in KS], color=AMBER if on else LGRAY,
                lw=1.6, alpha=0.85 if on else 0.6, marker="o", ms=4, zorder=4 if on else 2)
    cm = [df[df.k == k].excess.mean() for k in KS]
    ax.plot(KS, cm, color=INK, lw=3, marker="o", ms=8, zorder=6)
    ax.axhline(0, color=TEALD, lw=1.2, ls=(0, (4, 3)))
    ax.text(5.02, 0, "category-only\nexpectation", fontsize=8.6, color=TEALD, va="center", ha="left")
    for i, k in enumerate(KS):
        ax.text(k, cm[i] + 0.006, pstr(coh[k]["p"]), ha="center", fontsize=8.2,
                color=INK, fontweight="bold")
    ax.set_xticks(KS); ax.set_xlabel("k (top-k)")
    ax.set_ylabel("excess over category-only null")
    ax.set_title("Word-level excess grows with k", fontsize=12.5, fontweight="bold", color=INK, pad=8)
    n_on = len(sig5)
    handles = [Line2D([0], [0], color=INK, lw=3, marker="o", ms=7, label="cohort mean"),
               Line2D([0], [0], color=AMBER, lw=1.6, marker="o", ms=4, label=f"sig. at k=5 (n={n_on})"),
               Line2D([0], [0], color=LGRAY, lw=1.6, marker="o", ms=4,
                      label=f"n.s. at k=5 (n={n_total - n_on})")]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left")
    ax.set_xlim(0.8, 5.9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def panel_forest(ax, df, k=5):
    s = df[df.k == k].sort_values("excess", ascending=False).reset_index(drop=True)
    n = len(s)
    for i, row in s.iterrows():
        y = n - 1 - i
        sig = row.p_within_cat < ALPHA
        ax.plot([row.wcat_lo, row.wcat_hi], [y, y], color=TEAL, lw=6, alpha=0.28,
                solid_capstyle="round", zorder=2)                       # category-only 95% band
        ax.plot([row.wcat_mean] * 2, [y - 0.22, y + 0.22], color=TEALD, lw=1.8, zorder=3)  # null mean
        ax.plot([row.unif_mean] * 2, [y - 0.14, y + 0.14], color=LGRAY, lw=1.4, zorder=3)  # uniform
        ax.scatter([row.obs], [y], s=95, zorder=5,
                   facecolor=AMBER if sig else "white",
                   edgecolor=AMBERD if sig else GRAY, linewidths=1.8)   # observed
        ax.text(-0.004, y, row.display_id, ha="right", va="center", fontsize=9.5,
                color=INK if sig else GRAY, fontweight="bold" if sig else "normal")
        if sig:
            ax.text(row.obs + 0.006, y, f"+{row.excess:.3f}", va="center",
                    fontsize=8, color=AMBERD, fontweight="bold")
    n_sig = int((s.p_within_cat < ALPHA).sum())
    ax.set_yticks([]); ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(-0.055, s.obs.max() * 1.12)
    ax.set_xlabel(f"top-{k} retrieval accuracy")
    ax.set_title(f"Per patient: observed vs category-only null (top-{k})   \u2014   "
                 f"{n_sig}/{n} exceed the null band ({ALPHA_LABEL})",
                 fontsize=12.5, fontweight="bold", color=INK, pad=8)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=AMBER, markeredgecolor=AMBERD,
               markersize=10, label=f"observed \u2014 significant ({ALPHA_LABEL})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor=GRAY,
               markersize=10, label="observed \u2014 n.s."),
        Line2D([0], [0], color=TEAL, lw=6, alpha=0.28, label="category-only null (95%)"),
        Line2D([0], [0], color=TEALD, lw=1.8, label="category-only mean"),
        Line2D([0], [0], color=LGRAY, lw=1.4, label="uniform chance"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.6, loc="lower right")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


# ---------------------------------------------------------------- drivers
def make_individual(df, coh, outdir: Path):
    # S5_* rather than 01_/02_/03_: this module now writes into semantic_regression/, where
    # 00-12 already belong to the main panels. Colliding would have overwritten
    # 01_picture_category_indep and two others.
    specs = [("S5_within_category_null_decomposition",
              lambda a: panel_decomposition(a, df, coh), (8.2, 5.0)),
             ("S5_within_category_null_excess_vs_k",
              lambda a: panel_excess_vs_k(a, df, coh),  (7.6, 5.0)),
             ("S5_within_category_null_forest_topk",
              lambda a: panel_forest(a, df, 5),         (8.6, 5.4))]
    for name, fn, size in specs:
        fig, ax = plt.subplots(figsize=size)
        fn(ax)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(outdir / f"{name}.{ext}", dpi=DPI_PANEL)
        plt.close(fig)
        print("wrote", outdir / f"{name}.png")


def make_combined(df, coh, outdir: Path):
    fig = plt.figure(figsize=(13, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], hspace=0.34, wspace=0.22,
                          left=0.075, right=0.975, top=0.90, bottom=0.07)
    ax_a = fig.add_subplot(gs[0, 0]); panel_decomposition(ax_a, df, coh)
    ax_b = fig.add_subplot(gs[0, 1]); panel_excess_vs_k(ax_b, df, coh)
    ax_c = fig.add_subplot(gs[1, :]); panel_forest(ax_c, df, 5)
    for ax, lab in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c")]:
        ax.text(-0.04, 1.06, lab, transform=ax.transAxes, fontsize=17,
                fontweight="bold", color=INK, va="top", ha="right")
    fig.suptitle("Word identity is decoded beyond category (category-preserving null)",
                 fontsize=15, fontweight="bold", color=INK, y=0.965)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"S5_within_category_null.{ext}", dpi=DPI_COMBINED)
        print("wrote", outdir / f"S5_within_category_null.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    df = load()
    coh = cohort_stats(df)
    if not HAVE_SCIPY:
        print("NOTE: scipy not found - Wilcoxon p-values shown as 'n/a'. `pip install scipy` to enable.")
    make_individual(df, coh, HERE)
    make_combined(df, coh, HERE)
    print("done.")
