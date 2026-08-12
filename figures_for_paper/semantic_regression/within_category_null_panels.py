"""
Supplementary figure — within-category (category-preserving) null, both tasks.

Tests whether the semantic decoder resolves WORD IDENTITY beyond CATEGORY. The
category-preserving null permutes the word<->trial correspondence *within* each
semantic category (category structure preserved, sub-category identity destroyed)
and recomputes top-k. Any excess of observed over this null is word-level information.

One panel, picture naming. It decomposes retrieval into chance -> category-only ->
category+word identity at top-1/3/5, with per-participant points on all three bars and a
bracket carrying the group star between the two compared bars.

**Auditory naming is computed but not shipped** (Alec, 2026-08-11: that arm is a null and
needs a team discussion before it goes in the paper). Its rows stay in both source CSVs and
`--task auditory_naming` renders it on demand as an unshipped diagnostic, which is why this
module is still task-parameterised for a figure that currently has one panel.

This module RENDERS AND COMPUTES NOTHING. Bar heights, SEMs and stars come from
within_category_null_group.csv; the points come from within_category_null_topk.csv.
Both are written by compute_within_category_null.py, which owns the Wilcoxon and the
Holm correction. A renderer that recomputes its own statistics is how a figure comes
to disagree with its own source data.

Lives beside the figure it supplements. It used to sit in its own
figures_for_paper/within_category_null/ folder, a half-finished split that never produced
anything; moved here 2026-08-11 on Alec's call that this belongs as a supplementary figure
under semantic_regression.

Run (VSCode "Run Python File", or terminal):
    python figures_for_paper/semantic_regression/within_category_null_panels.py
Writes S5_within_category_null.{png,pdf} into this folder. S5 because 00-11 are taken by
the main panels and the repo numbers supplementary figures S1, S2, ....
Caption: S5_within_category_null_caption.md, beside it -- the main figure's caption.md is
a separate file.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # remove this line to show interactively in VSCode
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                      # figures_for_paper/
from paper_common import apply_paper_style, DPI_PANEL     # noqa: E402
from utils.paths import paper_source_data                 # noqa: E402  (paper_common adds main/)

# ---------------------------------------------------------------- style
# House style first: editable vector text (fonttype 42) and the repo-wide type sizes.
# This module used to set its own rcParams and skip apply_paper_style entirely, which
# figures_for_paper/README.md §3 forbids -- the visible cost is PDF text that is no longer
# selectable, which only shows up at submission.
apply_paper_style()

# These are ROLE colours (which null, which observation), not participant identity.
# The per-participant points are plain black by design (Alec, 2026-08-11): with three bars
# per group they carry position, not identity, and a 15-colour scatter over three coloured
# bars read as a second, competing encoding. Nothing here assigns a colour per participant,
# so participants.json is not consulted -- if points ever need identity again, take the
# colours from there via paper_common.assign_colors, never from a local palette.
INK = "#16202A"
TEAL = "#0F6E6A"
AMBER = "#C2670F"
LGRAY = "#AEB6BF"
DOT = "#111111"

KS = (1, 3, 5)
TASKS = ("picture_naming", "auditory_naming")
#: Only picture naming ships; auditory renders on demand as an unshipped diagnostic.
SHIPPED_TASK = "picture_naming"

#: (group-CSV stem, colour, legend label, per-participant column).  One tuple per bar, so a
#: bar, its per-participant points and its legend entry cannot drift apart.
SERIES = (("unif", LGRAY, "chance", "unif_mean"),
          ("wcat", TEAL, "category-only", "wcat_mean"),
          ("obs", AMBER, "category+word identity", "obs"))
#: The bracket spans these two bars -- the word-level comparison the figure is about.
BRACKET_PAIR = (1, 2)

W_BAR = 0.26
JITTER_SEED = 7


# ---------------------------------------------------------------- data
def _csv(name: str) -> Path:
    """The one location this figure's inputs live in.

    The producer is compute_within_category_null.py, which writes into ITS OWN
    source_data/ -- both CSVs are tracked there. This used to probe four candidate
    locations, three of which never held the file; a four-way probe is a way of not
    knowing where your input is. One sanctioned accessor instead.
    """
    path = paper_source_data("semantic_regression", name, create=False)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run "
            f"figures_for_paper/semantic_regression/compute_within_category_null.py first."
        )
    return path


def load_topk(task: str) -> pd.DataFrame:
    """Per-participant rows for one task. Raises rather than rendering half a figure."""
    df = pd.read_csv(_csv("within_category_null_topk.csv"))
    if "task" not in df.columns:
        raise ValueError("within_category_null_topk.csv has no `task` column -- it predates "
                         "the two-task rewrite. Re-run compute_within_category_null.py.")
    d = df[df.task == task]
    if d.empty:
        raise ValueError(f"within_category_null_topk.csv has no {task} rows. "
                         f"Run compute_within_category_null.py --task {task}.")
    return d


def load_group(task: str) -> pd.DataFrame:
    """This task's group tests.

    Asserts the Holm family is exactly the three k drawn in the panel. That is the check
    that keeps the stars honest: if the correction ever spans tests the panel does not
    show, or fewer than it does, the figure refuses to render rather than shipping a star
    whose family the caption misdescribes.
    """
    grp = pd.read_csv(_csv("within_category_null_group.csv"))
    g = grp[grp.task == task]
    if set(g.k) != set(KS):
        raise ValueError(f"within_category_null_group.csv covers k={sorted(g.k)} for {task}, "
                         f"expected {sorted(KS)}.")
    if set(g.n_tests) != {len(KS)}:
        raise ValueError(f"n_tests is {sorted(set(g.n_tests))} for {task}, expected {len(KS)} "
                         f"-- the Holm family is not this panel's three tests.")
    return g


# ---------------------------------------------------------------- drawing helpers
def _sig_bracket(ax, x0, x1, y, text, color=INK, fs=8):
    """Significance bracket spanning [x0,x1] with a centred label.

    Reads ax.get_ylim() to size the tick, so the caller MUST set y-limits first --
    called earlier it sizes against matplotlib's autoscale and the brackets land at
    inconsistent heights. Body from extendability/extendability_panels.py.
    """
    yl = ax.get_ylim()
    y2 = y + 0.03 * (yl[1] - yl[0])
    ax.plot([x0, x0, x1, x1], [y, y2, y2, y], lw=1.0, color=color, clip_on=False)
    ax.text((x0 + x1) / 2, y2, text, ha="center", va="bottom",
            fontsize=fs if text != "n.s." else 6.5,
            color=color if text != "n.s." else "#888888", clip_on=False)


def _jitter(n: int, task_idx: int) -> np.ndarray:
    """Seeded horizontal offsets, ONE vector per panel, reused across all three bars so a
    participant's three points share an x-offset and the eye can trace them. Seeded on
    (JITTER_SEED, task_idx) so a task's panel renders identically on every run."""
    rng = np.random.default_rng([JITTER_SEED, task_idx])
    return (rng.random(n) - 0.5) * (0.62 * W_BAR)


def _ymax(d: pd.DataFrame, g: pd.DataFrame) -> float:
    """Highest thing that will be drawn: a bar+SEM, or a participant point."""
    bars = max((g[f"{stem}_mean"] + g[f"{stem}_sem"]).max() for stem, _, _, _ in SERIES)
    pts = max(d[col].max() for _, _, _, col in SERIES)
    return float(max(bars, pts))


# ---------------------------------------------------------------- panel
def panel_decomposition(ax, d, g, task, *, legend=True, ylabel=True):
    """chance / category-only / category+word identity at each k, one task.

    Bar heights, SEMs and stars are READ from `g`; the points are READ from `d`.
    Nothing is computed here.
    """
    ymax = _ymax(d, g)
    g = g.set_index("k")
    pats = list(dict.fromkeys(d.patient))          # first-appearance order
    jit = _jitter(len(pats), TASKS.index(task))
    x = np.arange(len(KS))

    pair_top = np.zeros(len(KS))                   # top of whatever the bracket must clear
    for j, (stem, colour, label, col) in enumerate(SERIES):
        xs = x + (j - 1) * W_BAR
        h = [g.loc[k, f"{stem}_mean"] for k in KS]
        se = [g.loc[k, f"{stem}_sem"] for k in KS]
        ax.bar(xs, h, W_BAR, yerr=se, color=colour, ec="white", lw=0.5, capsize=2,
               error_kw=dict(lw=0.8), label=label, zorder=2)
        for i, k in enumerate(KS):
            v = d[d.k == k].set_index("patient").loc[pats, col].to_numpy()
            # Thin white ring: the points are black and two of the three bars are dark, so
            # this is what separates a point from the bar it sits on and from its overlaps.
            # It is a legibility aid, not a second colour encoding.
            ax.scatter(xs[i] + jit, v, s=9, c=DOT, edgecolors="white", linewidths=0.3,
                       alpha=0.8, zorder=3)
            if j in BRACKET_PAIR:
                pair_top[i] = max(pair_top[i], h[i] + se[i], float(v.max()))

    # y-limits BEFORE any bracket -- _sig_bracket sizes its tick off get_ylim().
    # 1.20 is what the tallest group needs and no more: the k=5 bracket sits at
    # ymax + 0.03*ymax, its tick adds 0.03 of the axis range, and the star sits above
    # that.  Anything larger is white space that makes the bars look shorter.
    ax.set_ylim(0, ymax * 1.20)
    for i, k in enumerate(KS):
        j0, j1 = BRACKET_PAIR
        _sig_bracket(ax, x[i] + (j0 - 1) * W_BAR, x[i] + (j1 - 1) * W_BAR,
                     pair_top[i] + 0.03 * ymax, g.loc[k, "stars"])

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in KS])
    # No title and no suptitle anywhere -- the task, the model and the test all live in
    # S5_within_category_null_caption.md.
    ax.set_xlabel("Top-$k$ retrieval")
    if ylabel:
        ax.set_ylabel("retrieval accuracy (cohort mean ± SEM)")
    if legend:
        ax.legend(loc="upper left")            # frameon=False comes from apply_paper_style
    ax.margins(x=0.06)


# ---------------------------------------------------------------- drivers
def _save(fig, stem: Path, dpi: int):
    """Both formats, every time (README §2)."""
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")
    print("wrote", f"{stem}.png/.pdf")


def make_figure(d, g, task: str, outdir: Path) -> Path:
    """The single-panel supplementary figure. No panel letter -- there is one panel."""
    fig, ax = plt.subplots(figsize=(4.3, 3.4))
    panel_decomposition(ax, d, g, task)
    fig.tight_layout()
    # The shipped arm owns the plain stem; anything else is a task-suffixed diagnostic, so
    # rendering the unshipped auditory panel can never overwrite the figure that ships.
    stem = outdir / ("S5_within_category_null" if task == SHIPPED_TASK
                     else f"S5_within_category_null_{task}")
    _save(fig, stem, DPI_PANEL)
    plt.close(fig)
    return stem


def print_drawn(g, task: str):
    """Every number the figure puts on paper, so it can be diffed against the group CSV."""
    print("task,k,unif_mean,wcat_mean,obs_mean,p_holm,n_tests,stars")
    for k in KS:
        r = g[g.k == k].iloc[0]
        print(f"{task},{int(k)},{r.unif_mean:.6f},{r.wcat_mean:.6f},"
              f"{r.obs_mean:.6f},{r.p_holm:.6f},{int(r.n_tests)},{r.stars}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--task", choices=TASKS, default=SHIPPED_TASK,
                    help="picture_naming ships; auditory_naming renders a task-suffixed "
                         "diagnostic that is NOT part of the paper figure")
    args = ap.parse_args(argv)
    d, g = load_topk(args.task), load_group(args.task)
    make_figure(d, g, args.task, HERE)
    if args.task != SHIPPED_TASK:
        print(f"NOTE: {args.task} is a diagnostic render -- not shipped, no caption.")
    print_drawn(g, args.task)
    print("done.")


if __name__ == "__main__":
    main()
