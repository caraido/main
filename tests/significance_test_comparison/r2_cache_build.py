# -*- coding: utf-8 -*-
"""Throwaway: extract the R2 obs/null arrays from the pinned PN/AN run PKLs.

The panels cache (figures_for_paper/semantic_regression/panels_cache_{task}_GloVe.npz)
carries category_indep and word_top1/3/5 only. R2 (`all_test_score` vs `all_chance`) is
the one *continuous-fit* metric with a real matched null, so it stands in for cosine,
which has no stored null anywhere (models/model.py never scores a shuffled cosine).

Mirrors semantic_regression_panels.build_cache: same key scheme ({pat}__{metric}__{obs,null}),
same JSON sidecar carrying `run_dir` so the cache self-invalidates when the pins move.

Run from main/:   python -m tests.significance_test_comparison.r2_cache_build
Outputs:          results/significance_test_comparison/r2_cache_{task}_{embedding}.npz
                  (+ .npz.json sidecar)
"""

import os
import sys
import json
import time

import numpy as np

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if MAIN_DIR not in sys.path:
    sys.path.insert(0, MAIN_DIR)

from utils.config import PIC_RUN, AUD_RUN, run_dir          # noqa: E402
from utils.paths import results_dir                          # noqa: E402
from report.helper.results_loader import load_pkl_raw        # noqa: E402

#: This pilot's one output destination. Never a hand-composed path -- the cache used to
#: live beside the code in tmp/, which is the anti-pattern the 2026-08 migration removed.
ANALYSIS = 'significance_test_comparison'
OUT_DIR = str(results_dir(ANALYSIS))

EMBEDDING = 'GloVe'
# (key, obs attr, null attr) — the only metric this script adds.
R2_SPEC = ('r2', 'all_test_score', 'all_chance')

TASKS = {'picture': PIC_RUN, 'auditory': AUD_RUN}


def cache_path_for(task, embedding=EMBEDDING):
    return os.path.join(OUT_DIR, f'r2_cache_{task}_{embedding}.npz')


def _patient_dirs(rdir):
    """Same rule as semantic_regression_panels._patient_dirs."""
    return sorted(
        d for d in os.listdir(rdir)
        if os.path.isdir(os.path.join(rdir, d))
        and not d.endswith('.json') and d not in ('report', '__pycache__')
    )


def load_cache(cache_path, rdir):
    """Cached arrays, or None if absent / built from a different run."""
    if not (os.path.exists(cache_path) and os.path.exists(cache_path + '.json')):
        return None
    side = json.load(open(cache_path + '.json'))
    cached = side.get('run_dir')
    if cached is None or os.path.abspath(cached) != os.path.abspath(rdir):
        print(f"  [cache] {os.path.basename(cache_path)} built from "
              f"{cached or 'an unrecorded run'} — stale, rebuilding", flush=True)
        return None
    return {'arrays': dict(np.load(cache_path)), 'side': side}


def build_cache(rdir, cache_path, embedding=EMBEDDING):
    key, obs_attr, null_attr = R2_SPEC
    arrays, kept = {}, []
    for p in _patient_dirs(rdir):
        pkl_path = os.path.join(rdir, p, 'semantic_regression_results.pkl')
        if not os.path.exists(pkl_path):
            continue
        t0 = time.time()
        size_mb = os.path.getsize(pkl_path) / 1e6
        print(f"  [cache] loading {p} ({size_mb:.0f} MB) ...", end='', flush=True)
        try:
            data = load_pkl_raw(pkl_path)
        except Exception as e:                                   # noqa: BLE001
            print(f" FAILED ({e})", flush=True)
            continue
        if data is None or embedding not in data.get('regressors', {}):
            print(f" no '{embedding}' regressor — skipped", flush=True)
            continue
        br = data['regressors'][embedding]
        if not (hasattr(br, obs_attr) and hasattr(br, null_attr)):
            print(f" missing {obs_attr}/{null_attr} — skipped", flush=True)
            del data
            continue
        obs = np.asarray(getattr(br, obs_attr), dtype=np.float32)
        null = np.asarray(getattr(br, null_attr), dtype=np.float32)
        if obs.ndim != 2 or null.shape != obs.shape:
            print(f" shape mismatch obs{obs.shape} null{null.shape} — skipped", flush=True)
            del data, br
            continue
        arrays[f'{p}__{key}__obs'] = obs
        arrays[f'{p}__{key}__null'] = null
        kept.append(p)
        print(f" ok {obs.shape} [{time.time() - t0:.0f}s]", flush=True)
        del data, br

    if not kept:
        raise RuntimeError(f"no patients cached from {rdir} (embedding {embedding!r})")

    side = {'patients': kept, 'embedding': embedding, 'metric': key,
            'obs_attr': obs_attr, 'null_attr': null_attr,
            'run_dir': os.path.abspath(rdir)}
    np.savez_compressed(cache_path, **arrays)
    with open(cache_path + '.json', 'w') as f:
        json.dump(side, f, indent=2)
    print(f"  [cache] saved {len(kept)} patients -> {os.path.basename(cache_path)}", flush=True)
    return {'arrays': dict(np.load(cache_path)), 'side': side}


def get(task):
    """Cached r2 arrays for `task`, building them if needed."""
    rdir = run_dir(TASKS[task])
    cp = cache_path_for(task)
    c = load_cache(cp, rdir)
    if c is None:
        c = build_cache(rdir, cp)
    return c


def main():
    for task in TASKS:
        print(f"[{task}] {TASKS[task]}", flush=True)
        c = get(task)
        print(f"[{task}] {len(c['side']['patients'])} patients: "
              f"{', '.join(c['side']['patients'])}\n", flush=True)


if __name__ == '__main__':
    main()
