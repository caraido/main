# -*- coding: utf-8 -*-
"""Skeleton for a new pilot. Copy the directory; do not edit it in place.

    cp -r tests/_template tests/<slug>
    python -m tests.<slug>.run --why "the one question this pilot answers"

Everything here that is not obvious is the point of the file. There is no
``os.path.join``, no ``MAIN_DIR / "results"`` and no bare filename resolved against the
working directory: every path comes from the ``RunContext`` that ``open_run`` hands out,
which is also what writes ``meta.json`` before any work starts. The pilot contract those
two facts serve is in README.md.

Replace ``compute_one`` with the actual analysis and delete the rest of this docstring.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# main/ on sys.path, so the module runs both as `python -m tests.<slug>.run` and as a
# plain script. Same three lines every pilot here uses; parents[2] because this file sits
# at main/tests/<slug>/run.py.
_MAIN_DIR = Path(__file__).resolve().parents[2]
if str(_MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_MAIN_DIR))

from utils.run_context import open_run          # noqa: E402

#: The results key, read off this directory's name so a copy needs no edit here. It is
#: also what makes ``utils.paths.stage_of`` report the analysis as ``pilot`` for free --
#: stage is derived from which lifecycle folder owns the name, never stored.
ANALYSIS = Path(__file__).resolve().parent.name


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--patients", nargs="+", default=["AZ"],
                    help="Participant ids. Takes a LIST -- pass them all in one "
                         "invocation rather than looping the flag, or the aggregate "
                         "describes only the last one.")
    ap.add_argument("--epochs", type=int, default=5,
                    help="Train/test splits per participant. Keep it small while piloting.")
    ap.add_argument("--why", required=True,
                    help="One line: the question this run answers. Stored in meta.json "
                         "and shown by `python -m utils.audit_runs --status`.")
    ap.add_argument("--supersedes", default=None,
                    help="The run_id this replaces, so a chain of re-runs is "
                         "reconstructable from the manifests alone.")
    return ap.parse_args(argv)


def compute_one(patient, epochs):
    """The pilot. Returns rows to be tabulated, or raises.

    Raising is fine: ``main`` records the failure against the participant and carries on,
    so one bad participant does not lose the rest of the run.
    """
    raise NotImplementedError("replace compute_one with the actual analysis")


def main(argv=None):
    args = parse_args(argv)
    if ANALYSIS == "_template":
        print("tests/_template is a skeleton, not a pilot: copy it to tests/<slug>/ "
              "first, or this writes an unnamed results/_template/ run.", file=sys.stderr)
        return 2

    run_id = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{ANALYSIS}_ep{args.epochs}"

    # meta= carries the run's PARAMETERS. The lifecycle fields (status, started_at,
    # duration, stage) are owned by open_run and passing one raises -- a manifest must not
    # be able to assert success before any work has run.
    with open_run(ANALYSIS, run_id,
                  why=args.why,
                  supersedes=args.supersedes,
                  meta={"patients": list(args.patients), "epochs": args.epochs}) as run:
        rows = []
        for patient in args.patients:
            try:
                rows.extend(compute_one(patient, args.epochs))
            except Exception as exc:                                   # noqa: BLE001
                run.fail(patient, exc)      # recorded in meta.json; the run continues
                continue
            run.succeed(patient)

        if not rows:
            return 1

        # Every destination this pilot may write to:
        #   run.table("<name>")   -> results/<slug>/<run_id>/source_data/<name>.csv
        #   run.figure("<name>")  -> results/<slug>/<run_id>/figures/<name>.png
        #   run.report("<slug>")  -> results/<slug>/<run_id>/report/<slug>_<run_id>.html
        #   run.path("a", "b.npz")-> results/<slug>/<run_id>/a/b.npz
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(run.table("summary"), index=False)

        # The few numbers this run exists to produce, so --status and a docs/experiments/
        # entry can quote them without opening anything. Keep it small.
        run.headline(n_rows=len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
