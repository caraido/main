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
  # shard across patients into separate out-dirs, then this script's --merge concatenates.
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
from semantic_regression import load_patient_data          # noqa: E402

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
    ap.add_argument('--out-dir', default=os.path.join(_MAIN, 'results', 'layer_sweep'))
    args = ap.parse_args()
    os.chdir(_MAIN)
    os.makedirs(args.out_dir, exist_ok=True)

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
    path = os.path.join(args.out_dir, 'layer_sweep.csv')
    out.to_csv(path, index=False)
    print(f"\nWrote {len(out)} rows ({out.patient.nunique()} patients) -> {path}", flush=True)


if __name__ == '__main__':
    main()
