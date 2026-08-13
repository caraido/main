# -*- coding: utf-8 -*-
"""Do both speech tasks land in one 2-D space under a single co-trained decoder?

Pilot for the cross-task figure's retired MDS panel. The retired panel compared TWO
separately trained decoders and could not show clear category clusters that also mapped to
the same place. This asks the question the other way round: fit ONE co-trained decoder and
look at the space it actually builds, so "both tasks project into the same space" is true by
construction and the only open question is whether CATEGORY is visible in it.

Three views are produced from the same out-of-fold fit and compared on the same statistic
(see latent.py). Nothing here is a paper figure; the winner gets rebuilt in
figures_for_paper/ under that folder's conventions.

    python -m tests.cotrain_latent_space.run --patients AA --why "..."
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_MAIN_DIR = Path(__file__).resolve().parents[2]
if str(_MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_MAIN_DIR))

from utils.run_context import open_run                                   # noqa: E402
from utils.config import (CROSS_TASK_FIGURE_ROI_DIR, TPM_LADDER_RUNS)    # noqa: E402
from analysis.cross_task.cross_task_prediction_mds import (              # noqa: E402
    category_centroid_alignment)
from tests.cotrain_latent_space import latent, figures                   # noqa: E402

ANALYSIS = Path(__file__).resolve().parent.name

#: The `tpm`/h10 pair the cross-task FIGURE is built on, not the repo-wide PIC_RUN/AUD_RUN.
#: Those are the `tp`/h5 pins and using them here produced a 250-feature (5-bin) latent space
#: while the figure's decoder sees 10 bins -- a different model, so a view built on it could
#: not be promoted into that figure without re-running everything. Read from
#: utils.config.TPM_LADDER_RUNS, never typed.
DEFAULT_PIC_RUN, DEFAULT_AUD_RUN = TPM_LADDER_RUNS[0], TPM_LADDER_RUNS[1]

VIEWS = [("latent", "1. co-trained PLS latent space"),
         ("lda", "2. picture-defined discriminants, auditory projected"),
         ("glove", "3. predicted-GloVe MDS (co-trained)")]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--patients", nargs="+", default=["AA"],
                    help="Participant ids. Takes a LIST -- pass them all in one invocation.")
    ap.add_argument("--folds", type=int, default=5,
                    help="Word-grouped out-of-fold splits. One co-trained fit per fold.")
    ap.add_argument("--balance", default="downsample",
                    choices=["none", "downsample", "upsample"],
                    help="Pooled-set resampling. Default matches the shipped cross-task "
                         "figure, so the latent space is the one the paper's decoder builds.")
    ap.add_argument("--n-shuffle", type=int, default=500,
                    help="Category-label shuffles for the alignment p-value.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pic-run", default=DEFAULT_PIC_RUN)
    ap.add_argument("--aud-run", default=DEFAULT_AUD_RUN)
    ap.add_argument("--why", required=True,
                    help="One line: the question this run answers.")
    ap.add_argument("--supersedes", default=None)
    return ap.parse_args(argv)


def compute_one(patient, args, run):
    """Every view for one participant. Returns (summary rows, diagnostics rows)."""
    X, Y, words, cats, task, n_ch = latent.pooled_arrays(
        patient, args.pic_run, args.aud_run, args.balance, args.seed)
    print("  [{}] {} trials ({} pic / {} aud), {} channels, {} features".format(
        patient, len(Y), int((task == "picture").sum()), int((task == "auditory").sum()),
        n_ch, X.shape[1]))

    Z, Yhat, ok = latent.oof_latent_and_prediction(X, Y, words, args.folds, args.seed)
    Z, Yhat, cats, task, words = Z[ok], Yhat[ok], cats[ok], task[ok], words[ok]

    # Component diagnostics stay at the TRIAL level: they are an F and an AUC, and both want
    # every sample they can get. Everything plotted is at the word level.
    diag = latent.component_diagnostics(Z, cats, task)
    diag.insert(0, "patient", patient)
    comps, rule = latent.pick_components(diag)

    wtask, wwords, wcats, Zw, Yhatw = latent.word_means(task, words, cats, Z, Yhat)
    print("    components plotted: {} ({}); {} word-level points".format(
        comps, rule, len(wwords)))

    embeddings = {
        "latent": latent.view_latent(Zw, comps),
        # LDA is FITTED on picture trials (all of them, for the fit) and applied to the word
        # means. Fitting on the 6-or-so word means per category would be badly underdetermined.
        "lda": None,
        "glove": latent.view_glove(Yhatw, args.seed),
    }
    lda_trial = latent.view_lda(Z, cats, task, args.seed)
    if lda_trial is not None:
        embeddings["lda"] = latent.word_means(task, words, cats, lda_trial)[3]

    rows, summary, drawn, percats = [], {}, [], {}
    for key, title in VIEWS:
        E = embeddings[key]
        if E is None:
            rows.append(dict(patient=patient, view=key, n_points=0,
                             alignment=np.nan, alignment_p=np.nan, n_shared_categories=0,
                             note="not computable: <3 picture categories with >=2 trials"))
            drawn.append((key, title, None))
            continue
        # The alignment is computed IN THE VIEW'S OWN 2-D SPACE, on the word-level points
        # that are actually drawn, because the question the pilot asks is about what a reader
        # can see. That makes the three numbers comparable to one another and NOT comparable
        # to the 300-D value quoted for the retired panel.
        is_pic = wtask == "picture"
        a, p, n_shared, _ = category_centroid_alignment(
            E[is_pic], wcats[is_pic], E[~is_pic], wcats[~is_pic], args.n_shuffle, args.seed)
        rows.append(dict(patient=patient, view=key, n_points=int(len(E)),
                         alignment=a, alignment_p=p, n_shared_categories=n_shared,
                         note=("components {}; {}".format(comps, rule)
                               if key == "latent" else "")))
        summary[key] = dict(alignment=a, alignment_p=p)
        drawn.append((key, title, E))
        pc = latent.per_category_cosine(E, wcats, wtask, args.n_shuffle, args.seed)
        if not pc.empty:
            pc.insert(0, "view", key)
            pc.insert(0, "patient", patient)
            percats[key] = pc
        pd.DataFrame({"task": wtask, "category": wcats, "word": wwords,
                      "dim1": E[:, 0], "dim2": E[:, 1]}).to_csv(
            run.table("points_{}_{}".format(patient, key)), index=False)

    # The per-category panel shows the BEST view for this participant, named in its title,
    # so the reader is not left comparing three bar charts to decide which one to believe.
    best = max(summary, key=lambda k: summary[k]["alignment"]) if summary else None
    figures.patient_figure(drawn, wcats, wtask, patient, summary,
                           run.figure("views_{}".format(patient)),
                           percat=percats.get(best), seed=args.seed)
    figures.diagnostics_figure(diag, comps, patient,
                               run.figure("components_{}".format(patient)))
    pc_all = (pd.concat(percats.values(), ignore_index=True) if percats
              else pd.DataFrame())
    return rows, diag, pc_all


def main(argv=None):
    args = parse_args(argv)
    run_id = "{:%Y-%m-%d_%H-%M-%S}_{}_{}_k{}".format(
        datetime.now(), ANALYSIS, args.balance, args.folds)
    with open_run(ANALYSIS, run_id, why=args.why, supersedes=args.supersedes,
                  meta={"patients": list(args.patients), "folds": args.folds,
                        "balance": args.balance, "seed": args.seed,
                        "n_shuffle": args.n_shuffle,
                        "pic_run": args.pic_run, "aud_run": args.aud_run,
                        "roi_dir": CROSS_TASK_FIGURE_ROI_DIR}) as run:
        rows, diags, percats = [], [], []
        for patient in args.patients:
            try:
                r, d, pc = compute_one(patient, args, run)
            except Exception as exc:                                    # noqa: BLE001
                run.fail(patient, exc)
                print("  [{}] FAILED: {}".format(patient, exc))
                continue
            rows.extend(r)
            diags.append(d)
            if not pc.empty:
                percats.append(pc)
            run.succeed(patient)

        if not rows:
            return 1
        df = pd.DataFrame(rows)
        df.to_csv(run.table("summary"), index=False)
        pd.concat(diags, ignore_index=True).to_csv(run.table("components"), index=False)
        if percats:
            pd.concat(percats, ignore_index=True).to_csv(
                run.table("per_category"), index=False)

        print("\n", df.to_string(index=False), sep="")
        best = (df.dropna(subset=["alignment"])
                  .groupby("view")["alignment"].mean().sort_values(ascending=False))
        for view, val in best.items():
            print("  mean alignment  {:<8s} {:.3f}".format(view, val))
        run.headline(n_participants=int(df["patient"].nunique()),
                     **{"mean_alignment_" + v: round(float(x), 4)
                        for v, x in best.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
