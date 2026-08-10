# -*- coding: utf-8 -*-
"""CLI orchestrator for the auditory_alignment pilot: compute all (cue, patient) cells,
then build the HTML report.

Run from main/ in the Speech conda env
(C:\\Users\\Owner\\miniconda3\\envs\\Speech\\python.exe):

  # smoke test (one cheap cell)
  python -m tests.auditory_alignment.run --patients AZ --cues stim_on --epochs 5
  # fast pilot (all 6 patients x 4 cues, ~20 epochs) + report
  python -m tests.auditory_alignment.run --epochs 20
  # full run (convention 50 epochs; recompute)
  python -m tests.auditory_alignment.run --epochs 50 --overwrite
  # rebuild figures + HTML from cached cells only
  python -m tests.auditory_alignment.run --report-only
"""

import os
import sys
import argparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAIN_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

from tests.auditory_alignment import config          # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Auditory alignment comparison: which cue triggers semantic info?",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--patients", nargs="+", default=None,
                    help="Patient IDs (default: every auditory participant in "
                         "utils.config.AUDITORY_PATIENTS). Takes a LIST -- pass them all "
                         "in one invocation: the report writer aggregates and overwrites, "
                         "so looping this flag leaves a report describing only the last "
                         "participant. Use --no-report per chunk plus one --report-only "
                         "if a split run is ever needed.")
    ap.add_argument("--cues", nargs="+", default=None, choices=list(config.CUES),
                    help="Cue keys to align to (default: all four).")
    ap.add_argument("--epochs", type=int, default=config.DEFAULTS["epochs"],
                    help="Regression epochs per cell.")
    ap.add_argument("--embedding", default=config.DEFAULTS["embedding"],
                    help="Embedding target (default GloVe).")
    ap.add_argument("--bin-size", type=int, default=config.DEFAULTS["bin_size"],
                    dest="bin_size", help="Bin size (ms).")
    ap.add_argument("--history-bins", type=int, default=config.DEFAULTS["n_bins_history"],
                    dest="n_bins_history", help="Feature-lag history bins.")
    ap.add_argument("--roi-atlas", default=config.DEFAULTS["roi_atlas"],
                    dest="roi_atlas", choices=["nmm", "dk", "none"],
                    help="Channel gate applied to semantic_regression. Recorded in each "
                         "cell's meta.json; cells computed before 2026-08-10 are "
                         "whole-brain and are not comparable with gated ones.")
    ap.add_argument("--pctile", type=float, default=config.DEFAULTS["pctile"],
                    help="Per-bin permutation null percentile (per patient).")
    ap.add_argument("--alpha", type=float, default=0.05, help="BH-FDR level for the report.")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="Patients to exclude from the group aggregate (e.g. RB).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute cells even if a cached npz exists.")
    ap.add_argument("--report-only", action="store_true", dest="report_only",
                    help="Skip compute; rebuild figures + HTML from cached cells.")
    ap.add_argument("--no-report", action="store_true", dest="no_report",
                    help="Compute only; do not build the report.")
    args = ap.parse_args(argv)

    patients = args.patients or list(config.AUD_PATIENTS)
    cues = args.cues or list(config.CUES.keys())

    if not args.report_only:
        from tests.auditory_alignment import align_runner
        align_runner.run_all(
            cues, patients, epochs=args.epochs, embedding=args.embedding,
            bin_size=args.bin_size, n_bins_history=args.n_bins_history,
            roi_atlas=args.roi_atlas, overwrite=args.overwrite,
        )

    if not args.no_report:
        from tests.auditory_alignment import report
        report.build_report(
            patients=patients, cues=cues, alpha=args.alpha, pctile=args.pctile,
            exclude=tuple(args.exclude),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
