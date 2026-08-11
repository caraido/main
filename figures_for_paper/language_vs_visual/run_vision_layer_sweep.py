# -*- coding: utf-8 -*-
"""
Restricted vision layer sweep for the language-vs-visual figure (panel f).

Reuses tests/embedding_sweeps/visual_layer_sweep.py but restricts the layer sweep
to the two 2-vs-2 vision models the figure uses — **DINOv3** and **MoCo** — instead
of every visual backbone. Writes the same layer_sweep.csv schema to the script's
canonical output path (main/results/layer_sweep/layer_sweep.csv), concatenating
across patients so parallel shards can each append their slice.

Run (Speech env; cwd = main/):
  python figures_for_paper/language_vs_visual/run_vision_layer_sweep.py --patients AA AP AZ
  # Re-running for a subset MERGES into the existing layer_sweep.csv: those patients are
  # replaced, everyone else is kept. Pass --overwrite to start the file fresh.
  #
  # This is what results/layer_sweep_KAW/ works around -- before 2026-08-08 this script
  # overwrote, so KAW could only be added by writing to a separate --out-dir, and
  # compute_language_vs_visual_data.py globs layer_sweep_*/ to pick that shard back up.
  # Once the full cohort has been swept into results/layer_sweep/, the shard and the glob
  # can both go.
"""

import os
import sys
import argparse
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAIN = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _MAIN)

from analysis.embedding_sweeps.visual_layer_sweep import (   # noqa: E402
    load_layerwise_embeddings, run_layer_sweep)
import semantic_regression as sr                           # noqa: E402
from semantic_regression import load_patient_data          # noqa: E402
from utils import config as _cfg                           # noqa: E402
from utils.paths import results_dir                        # noqa: E402

KEEP_PREFIXES = ('dinov3', 'moco')                          # the 2-vs-2 vision family


def sweep_patient(patient, epochs=10, model='kernel_pls'):
    pdata = load_patient_data(patient)
    layer_embeds = load_layerwise_embeddings(pdata)
    kept = {k: v for k, v in layer_embeds.items() if k.startswith(KEEP_PREFIXES)}
    print(f"  [{patient}] {len(kept)}/{len(layer_embeds)} embeddings kept "
          f"({sorted(kept.keys())})", flush=True)
    df = run_layer_sweep(patient, pdata, kept, n_epochs=epochs, model_mode=model)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patients', nargs='+', required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--model', default='kernel_pls')
    # Same destination as analysis/embedding_sweeps/visual_layer_sweep.py, which is the
    # point: both used to hand-compose it separately, so the two could drift apart.
    ap.add_argument('--out-dir', default=str(results_dir('layer_sweep', create=False)))
    ap.add_argument('--overwrite', action='store_true',
                    help='Replace layer_sweep.csv instead of merging into it. Use when '
                         'redoing the whole cohort; without it, patients already in the '
                         'file are updated in place and the rest are kept.')
    ap.add_argument('--roi-atlas', choices=list(_cfg.ROI_ATLAS_CHOICES),
                    default=_cfg.ROI_ATLAS_DEFAULT, dest='roi_atlas',
                    help='Atlas gating channel selection. The sweep builds its features '
                         'through semantic_regression.load_patient_data, so it inherits '
                         'the gate; this makes the choice explicit and recorded.')
    args = ap.parse_args()
    os.chdir(_MAIN)
    os.makedirs(args.out_dir, exist_ok=True)

    # The sweep calls semantic_regression.load_patient_data directly rather than reading a
    # finished run, so the gate has to be set on that module. Without this the sweep would
    # silently use the module default while the rest of the pass used something else.
    sr.ROI_ATLAS = args.roi_atlas
    print(f"ROI atlas: {args.roi_atlas}  |  history: {sr.N_BINS_HISTORY} bins", flush=True)

    frames = []
    for p in args.patients:
        print(f"\n{'='*56}\nPatient: {p}\n{'='*56}", flush=True)
        try:
            frames.append(sweep_patient(p, epochs=args.epochs, model=args.model))
        except Exception as e:
            print(f"  [{p}] FAILED: {e}", flush=True)
    if not frames:
        print("No results.")
        return
    out = pd.concat(frames, ignore_index=True)
    # Stamp the gate into the rows. Without this an NMM sweep and a DK sweep are
    # indistinguishable once written, and the merge below (and the figure's own
    # drop_duplicates) would silently mix or replace one with the other.
    out['roi_atlas'] = args.roi_atlas
    path = os.path.join(args.out_dir, 'layer_sweep.csv')

    # MERGE, do not clobber. The docstring above has always claimed this script
    # "concatenates across patients", but it wrote with a plain to_csv -- so running it
    # for one patient silently discarded every other patient's rows. That is why
    # results/layer_sweep_KAW/ exists as a separate shard, and why the figure has to glob
    # layer_sweep_*/ to find it. Merging here is what lets that shard be retired.
    #
    # The replace key is (patient, roi_atlas): re-sweeping a patient under one atlas must
    # not evict that patient's rows under the other.
    if os.path.exists(path) and not args.overwrite:
        prev = pd.read_csv(path)
        if 'roi_atlas' not in prev.columns:
            # Pre-2026-08-08 rows: whole-brain, 10-bin. Labelled so they can be told apart
            # rather than silently inheriting the current atlas.
            prev['roi_atlas'] = 'legacy'
        done = set(zip(out['patient'], out['roi_atlas']))
        keep_mask = [k not in done for k in zip(prev['patient'], prev['roi_atlas'])]
        kept = prev[keep_mask]
        out = pd.concat([kept, out], ignore_index=True)
        print(f"  merged into existing file: kept {len(kept)} row(s) from "
              f"{kept.patient.nunique()} patient(s), replaced {len(done)} "
              f"(patient, atlas) pair(s)", flush=True)

    out.to_csv(path, index=False)
    print(f"\nWrote {len(out)} rows ({out.patient.nunique()} patients) -> {path}", flush=True)


if __name__ == '__main__':
    main()
