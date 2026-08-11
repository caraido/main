# -*- coding: utf-8 -*-
"""
figures_for_paper/language_vs_visual — null-mean extraction step.

Builds ``cache_null_means_100ep.csv``: per participant x embedding x time bin, the
observed and shuffled-null balanced accuracy for category and word retrieval, averaged
over epochs. ``compute_language_vs_visual_data.py`` subtracts them to get the
above-chance effect that panels a-e plot.

WHY THIS SCRIPT EXISTS
----------------------
This extraction used to live only in a cell of ``notebooks/language_vs_visual.ipynb``,
which made it the last analysis step in the repository that could not be reproduced from
the command line. The notebook also disagreed with the rest of the repo in three ways
that a reader of the figure could not see:

  * it selected runs by NEWEST MATCHING GLOB rather than the pinned ``utils.config.PIC_RUN``,
    so re-running it after any new run silently retargeted the figure;
  * it referenced ``main/tests/results/``, a path the 2026-07 reorg deleted;
  * it carried its own ``ALPHA = 0.001`` (and a second, shadowing ``0.025``) instead of the
    repo-wide cutoff.

None of those affect the cache itself -- it holds no p-values and no run selection beyond
the folder it is pointed at -- but they are why the notebook is not a dependable producer.
This script takes the pinned run and an explicit participant list, and nothing else.

The cache is a CACHE: every number in it is derived from the run's own PKLs, so it can be
deleted and rebuilt. It exists because the alternative is loading ~19 GB of PKLs every time
the figure is drawn.

Run (Speech conda env; cwd = main/):
    python figures_for_paper/language_vs_visual/compute_null_means.py
    python figures_for_paper/language_vs_visual/compute_null_means.py --run <run_id> --patients AA AZ
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS_ROOT = os.path.dirname(HERE)
MAIN_DIR = os.path.dirname(FIGS_ROOT)
sys.path.insert(0, FIGS_ROOT)
sys.path.insert(0, MAIN_DIR)

from utils import config as _cfg                              # noqa: E402
from utils.paths import figures_dir, results_dir              # noqa: E402
from utils.run_meta import read_window, read_meta             # noqa: E402
from report.helper.results_loader import load_patient_from_pkl  # noqa: E402

#: The embeddings panels a-e contrast. Kept here rather than imported from the compute
#: step so this script has no dependency on it -- the cache is an input to that script.
EMBEDDINGS = ["GloVe", "Word2Vec", "ConceptNet", "DINOv2Small", "DINOv3", "MoCo"]

#: NB this filename is also written by notebooks/language_vs_visual.ipynb and read by
#: compute_language_vs_visual_data.py -- three references to one string in a gitignored
#: directory, with a paper figure downstream. The directory is now single-sourced through
#: utils.paths.figures_dir; consolidating the *filename* to one constant, and stopping the
#: notebook writing it at all, is queued as its own change.
OUT = str(figures_dir("language_vs_visual", "source_data", create=False)
          / "cache_null_means_100ep.csv")


def extract(run_id, patients, embeddings, bin_ms):
    """One row per (patient, embedding, bin): observed and null means over epochs."""
    run_dir = str(results_dir("semantic_regression", run_id, create=False))
    rows, missing = [], []
    for patient in patients:
        pkl = os.path.join(run_dir, patient, "semantic_regression_results.pkl")
        if not os.path.exists(pkl):
            missing.append(patient)
            print(f"  [{patient}] no PKL in this run -- skipped", flush=True)
            continue
        print(f"  [{patient}] loading …", end=" ", flush=True)
        data = load_patient_from_pkl(pkl)
        if data is None:
            missing.append(patient)
            print("FAILED to load", flush=True)
            continue
        print(f"ok ({len(data)} embeddings)", flush=True)

        for emb in embeddings:
            if emb not in data:
                continue
            d = data[emb]
            cat_obs = np.asarray(d["cat_obs"])          # (n_epochs, n_bins)
            cat_null = np.asarray(d["cat_null"])
            word_obs = np.asarray(d["word_obs"])
            word_null = np.asarray(d["word_null"])
            for b in range(cat_obs.shape[1]):
                rows.append({
                    "run_id": run_id,
                    "patient": patient,
                    "embedding": emb,
                    "bin_index": b,
                    "time_ms": b * bin_ms,
                    "cat_obs_mean": float(cat_obs[:, b].mean()),
                    "cat_null_mean": float(cat_null[:, b].mean()),
                    "word_obs_mean": float(word_obs[:, b].mean()),
                    "word_null_mean": float(word_null[:, b].mean()),
                })
    return pd.DataFrame(rows), missing


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--run", default=_cfg.PIC_RUN,
                    help="Run id under results/semantic_regression/ "
                         "(default: utils.config.PIC_RUN -- the pinned picture run)")
    ap.add_argument("--patients", nargs="+", default=list(_cfg.PICTURE_PATIENTS),
                    help="Participants to extract (default: utils.config.PICTURE_PATIENTS)")
    ap.add_argument("--embeddings", nargs="+", default=EMBEDDINGS)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    run_dir = os.path.join(MAIN_DIR, "results", "semantic_regression", args.run)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"run not found: {run_dir}")

    # Bin width comes from the run, never from a constant here: the figure's time axis is
    # (bin_index - n_bins_history) * bin_size, and a wrong width silently rescales it.
    n_hist, bin_ms = read_window(run_dir)
    meta = read_meta(run_dir)
    print(f"[null-means] run   : {args.run}")
    print(f"[null-means] window: {n_hist} bins x {bin_ms:g} ms"
          f"  |  roi_atlas: {meta.get('roi_atlas')}")
    print(f"[null-means] cohort: {len(args.patients)} requested")

    df, missing = extract(args.run, args.patients, args.embeddings, bin_ms)
    if df.empty:
        raise SystemExit("no rows extracted -- nothing written.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    got = sorted(df.patient.unique())
    print(f"\n[null-means] wrote {len(df):,} rows, {len(got)} participants -> {args.out}")
    print(f"[null-means] embeddings: {sorted(df.embedding.unique())}")
    print(f"[null-means] bins/participant: "
          f"{df.groupby('patient').bin_index.max().add(1).to_dict()}")
    if missing:
        # Loud, because a silently short cohort is how this figure came to be N=12 while
        # its neighbours were N=13.
        print(f"[null-means] WARNING: {len(missing)} participant(s) produced no rows: "
              f"{', '.join(missing)}")


if __name__ == "__main__":
    main()
