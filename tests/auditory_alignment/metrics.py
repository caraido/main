# -*- coding: utf-8 -*-
"""Per-bin metric extraction from a fitted BasicRegressor, the cue-relative time axis,
and compact (npz + json) caching of one (cue, patient) cell.

The heavy result pkl (full regressor objects) is deliberately NOT saved — only the small
(n_epochs, n_bins) obs/null arrays per metric plus a meta.json. Tens of KB per cell.
"""

import os
import json
import numpy as np


def extract_perbin(br, metrics):
    """Pull per-bin obs (and null, if any) arrays from a fitted BasicRegressor.

    Parameters
    ----------
    br : models.model.BasicRegressor  (already .fit())
    metrics : iterable of (key, label, obs_attr, null_attr_or_None, family)

    Returns
    -------
    dict[key] -> {'obs': (n_epochs, n_bins) float array, 'null': (n_epochs, n_bins) or None}
    """
    out = {}
    for key, _label, obs_attr, null_attr, _fam in metrics:
        if not hasattr(br, obs_attr):
            raise AttributeError(f"BasicRegressor has no attribute {obs_attr!r} for metric {key!r}")
        obs = np.asarray(getattr(br, obs_attr), dtype=np.float64)
        null = None
        if null_attr is not None and hasattr(br, null_attr):
            nv = getattr(br, null_attr)
            if nv is not None and np.size(nv) > 0:
                null = np.asarray(nv, dtype=np.float64)
        out[key] = {"obs": obs, "null": null}
    return out


def bin_offset_axis(n_bins, actual_back_sec, bin_size_ms):
    """Cue-relative bin offset and bin-center time (seconds), x=0 exactly at the cue.

    The align block slices [cue - back, cue + fwd] with `back` floored to whole bins, so
    the cue sits on the boundary at input-bin index cue_bin = round(back / bin). Output
    bin b (semantic_regression fits one regressor per input bin, anchored at the current
    bin) therefore has:
        k(b)        = b - cue_bin           integer offset from the cue (common grid)
        t_center(b) = (k + 0.5) * bin_s     bin-center time; left edge k*bin_s, so the
                                            k=0 bin's left edge is exactly the cue (t=0).
    Aggregation across patients/cues joins on the integer k (exact — no interpolation).
    """
    bin_s = bin_size_ms / 1000.0
    back = 0.0 if actual_back_sec is None else float(actual_back_sec)
    cue_bin = int(round(back / bin_s))
    k = np.arange(n_bins) - cue_bin
    t_center = (k + 0.5) * bin_s
    return k, t_center


def save_perbin(out_dir, arrays, meta):
    """Write perbin.npz ({key}__obs / {key}__null float32) + meta.json to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    flat = {}
    for key, d in arrays.items():
        flat[f"{key}__obs"] = np.asarray(d["obs"], dtype=np.float32)
        if d.get("null") is not None:
            flat[f"{key}__null"] = np.asarray(d["null"], dtype=np.float32)
    np.savez_compressed(os.path.join(out_dir, "perbin.npz"), **flat)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=_json_default)


def load_perbin(cell_dir):
    """Inverse of save_perbin -> (arrays, meta). arrays[key] = {'obs':.., 'null':.. or None}."""
    npz = np.load(os.path.join(cell_dir, "perbin.npz"))
    with open(os.path.join(cell_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    arrays = {}
    for name in npz.files:
        key, kind = name.split("__")
        arrays.setdefault(key, {"obs": None, "null": None})[kind] = np.asarray(npz[name], dtype=np.float64)
    return arrays, meta


def is_done(cell_dir):
    return (os.path.exists(os.path.join(cell_dir, "perbin.npz"))
            and os.path.exists(os.path.join(cell_dir, "meta.json")))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
