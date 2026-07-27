# -*- coding: utf-8 -*-
"""In-process driver: run main/semantic_regression.py once per (cue, patient) with
--warp none --align {cue}, extract the compact per-bin metric arrays, cache them.

The reconstruction the user asked for is exactly what the align block of
semantic_regression.py already does (slice raw hg_data around the cue, floor the window
to whole bins so the cue lands on a bin boundary, trim to the min window across trials,
bin, then run the full clean+fit pipeline). We drive it by setting the module globals the
way main() does and calling its functions — so we reuse the tested pipeline verbatim and
only add per-cell output keying + metric extraction.
"""

import os
import sys
import gc
import traceback

import numpy as np

# ── Path bootstrap: put main/ on sys.path so `import semantic_regression`, `utils.*` work.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_THIS_DIR)
_MAIN_DIR = os.path.dirname(_TESTS_DIR)
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

from utils.paths import results_dir                       # noqa: E402
from tests.auditory_alignment import config               # noqa: E402
from tests.auditory_alignment import metrics as M         # noqa: E402


def cell_dir(cue_key, patient, create=True):
    """results/auditory_alignment/{cue_key}/{patient}/ ."""
    return str(results_dir(config.ANALYSIS, cue_key, patient, create=create))


def _set_globals(sr, align_value, *, embedding, bin_size, n_bins_history):
    """Replicate what semantic_regression.main() sets for `--task auditory_naming
    --warp none --align {align_value} --embedding {embedding}`. Every global the load/
    embed/regress path reads is set explicitly (we bypass main(), so nothing else sets
    them)."""
    sr.TASK = "auditory_naming"
    sr.AUDITORY_WARP = "none"                # no time-warp — align only
    sr.AUDITORY_WARP_SCOPE = "group"         # unused when warp == none, kept consistent
    sr.AUDITORY_WARP_TARGET_SEC = None
    sr.ALIGN_CUE = align_value               # e.g. 'aud_stim_onset'
    sr.ALIGN_BACK = None                     # full available window (min across trials)
    sr.ALIGN_FORWARD = None
    sr.EMBEDDING_NAMES = [embedding]         # main() auto-defaults auditory to text-only;
    sr.BIN_SIZE = bin_size                   # we bypass main(), so set it ourselves
    sr.N_BINS_HISTORY = n_bins_history


def run_one(sr, cue_key, patient, shared, *, epochs, embedding, bin_size,
            n_bins_history, overwrite=False):
    """Compute one (cue, patient) cell and cache perbin.npz + meta.json. Returns the dir."""
    out_dir = cell_dir(cue_key, patient)
    if not overwrite and M.is_done(out_dir):
        print(f"  [skip] {cue_key}/{patient}: already computed", flush=True)
        return out_dir

    align_value = config.CUES[cue_key]
    _set_globals(sr, align_value, embedding=embedding, bin_size=bin_size,
                 n_bins_history=n_bins_history)

    pdata = sr.load_patient_data(patient)
    embeddings = sr.build_patient_embeddings(pdata, shared, embedding_names=[embedding])
    regressors = sr.run_regressions(
        pdata, embeddings, n_epochs=epochs,
        closest=config.DEFAULTS["closest"], model_mode=config.DEFAULTS["model"],
        embedding_names=[embedding],
    )
    br = regressors[embedding]

    arrays = M.extract_perbin(br, config.METRICS)
    n_bins = int(np.asarray(br.all_test_score).shape[1])
    k, t_center = M.bin_offset_axis(n_bins, pdata.get("actual_back_sec"), bin_size)

    meta = dict(
        patient=patient,
        cue_key=cue_key,
        align_cue=align_value,
        embedding=embedding,
        epochs=int(epochs),
        bin_size_ms=int(bin_size),
        n_bins_history=int(n_bins_history),
        n_bins=int(n_bins),
        n_trials=int(np.asarray(pdata["clean_data_binned"]).shape[0]),
        n_channels=int(np.asarray(pdata["clean_data_binned"]).shape[1]),
        n_unique_words=int(len({str(w) for w in pdata["target_concept"]})),
        actual_back_sec=(None if pdata.get("actual_back_sec") is None
                         else float(pdata["actual_back_sec"])),
        actual_forward_sec=(None if pdata.get("actual_forward_sec") is None
                            else float(pdata["actual_forward_sec"])),
        rel_cues=pdata.get("rel_cues"),
        rel_cues_reference=pdata.get("rel_cues_reference"),
        k=[int(x) for x in k],
        t_center_s=[float(x) for x in t_center],
    )
    M.save_perbin(out_dir, arrays, meta)
    print(f"  [done] {cue_key}/{patient}: n_bins={n_bins}, n_trials={meta['n_trials']}, "
          f"back={meta['actual_back_sec']}s, words={meta['n_unique_words']}", flush=True)

    del regressors, br, pdata, embeddings, arrays
    gc.collect()
    return out_dir


def _make_console_unicode_safe():
    """semantic_regression's logging prints box-drawing chars; the Windows cp1252 console
    raises on them when we bypass its main() (which sets up a UTF-8 tee). Mirror the
    project's fix (semantic_regression_panels.py) so the driver's stdout tolerates them."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                try:
                    stream.reconfigure(errors="replace")
                except Exception:
                    pass


def run_all(cues, patients, *, epochs, embedding, bin_size, n_bins_history, overwrite=False):
    """Compute all (cue, patient) cells. Loads GloVe (+ the shared text models) ONCE and
    reuses it across cells. Per-cell try/except so one failure doesn't abort the batch."""
    _make_console_unicode_safe()
    import semantic_regression as sr

    # semantic_regression uses RELATIVE data paths ('data/...'); main() chdirs to main/
    # before loading, and we bypass main(), so do it here. Our own outputs use absolute
    # paths (utils.paths), so this doesn't affect where results are written.
    os.chdir(os.path.dirname(os.path.abspath(sr.__file__)))

    todo = [(c, p) for c in cues for p in patients
            if overwrite or not M.is_done(cell_dir(c, p, create=False))]
    if not todo:
        print("All requested (cue, patient) cells already computed — nothing to run.", flush=True)
        return
    print(f"Computing {len(todo)} / {len(cues) * len(patients)} cells "
          f"(epochs={epochs}, embedding={embedding}).", flush=True)

    shared = sr.load_shared_embedding_models()

    n_ok, n_fail = 0, 0
    for cue_key in cues:
        for patient in patients:
            try:
                run_one(sr, cue_key, patient, shared, epochs=epochs, embedding=embedding,
                        bin_size=bin_size, n_bins_history=n_bins_history, overwrite=overwrite)
                n_ok += 1
            except Exception:
                n_fail += 1
                print(f"  [ERROR] {cue_key}/{patient} — continuing", flush=True)
                traceback.print_exc()
    print(f"Compute complete: {n_ok} ok, {n_fail} failed.", flush=True)
