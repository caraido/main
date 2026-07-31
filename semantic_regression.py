 # -*- coding: utf-8 -*-
"""semantic_regression.py
------------------------
Batch script: neural activity → word embeddings (semantic regression).

Runs the full pipeline (data loading, preprocessing, regression, figure saving,
and source-data export) for every patient that has a picture_naming_df.pkl file
under data/{patient}/.

Each invocation creates a unique **run** identified by a datetime stamp.
A ``meta.json`` is written alongside the outputs so every run is fully
reproducible (hyperparameters, versions, command line, git hash, …).

Output layout (relative to main/):
    figures/semantic_regression/{run_id}/{patient}/
        r2_over_time.html
        word_retrieval_balanced_acc.html
        category_retrieval_balanced_acc.html
        confusion_word.png
        confusion_category.png
        count_vs_accuracy.png
        count_vs_f1.png
    figures/semantic_regression/{run_id}/meta.json

    results/semantic_regression/{run_id}/{patient}/
        semantic_regression_results.pkl   – all BasicRegressor objects + metadata
        top1_decoding_source_data.csv     – true/predicted word+category at best bin
        per_time_scores.csv               – R², balanced-acc, F1 over all time bins
    results/semantic_regression/{run_id}/meta.json

    logs/semantic_regression_{run_id}.log

Usage (from main/):
    python semantic_regression.py
    python semantic_regression.py --patients AZ VB
    python semantic_regression.py --epochs 30 --closest cosine
"""

import argparse
import collections
import gc
import gzip
import json
import math
import os
import platform
import subprocess
import sys
import pickle as pk
import traceback
import warnings
warnings.filterwarnings('ignore')  # suppress all warnings before any library loads
from datetime import datetime
from urllib.request import urlretrieve

import dill
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')   # non-interactive backend – safe for terminal/batch runs
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from torchtext.vocab import GloVe, FastText

# ── project imports ───────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from utils.utils import remove_number, plot_accuracy_plotly
from models.model import BasicRegressor

# --- cleanup batch 1: imports added by automated migration ---
from utils.logging import _sep, _header, _section, _progress
from utils.confusion_matrices import _best_bin_from_top1, _collect_pairs_at_bin, _make_cm, _normalize_col, _rank_labels_by_f1, _plot_cm_grid, _per_word_stats, _per_word_f1_stats

# --- cleanup batch 2: re-import previously-local helpers from utils ---
from utils.run_meta import (
    git_hash as _git_hash,
    git_dirty as _git_dirty,
    write_meta as _write_meta,
)
# Aliased: `results_dir` is also a local variable and a parameter name below.
from utils.paths import (
    results_dir as _results_dir,
    figures_dir as _figures_dir,
)
from utils.patient_data import (
    INVALID_ANSWER_SET as _INVALID_ANSWER_SET,
    find_df_path as _find_df_path,
    is_valid_answer as _is_valid_answer,
    extract_col as _extract_col,
    discover_patients as _discover_patients,
)
from utils.confusion_matrices import _plot_count_vs_metric


# --- cleanup batch 2: backward-compatibility wrapper ---------------------
# Re-expose discover_patients as a no-arg function so external callers
# (tests/, notebooks/) that import this name from the script can keep doing
# `from {module} import discover_patients` without change.
def discover_patients():
    """Discover patient IDs that have a {patient}_{task}_df.pkl in DATA_FOLDER."""
    return _discover_patients(DATA_FOLDER, TASK)


# ─────────────────────────────────────────────────────────────────────────────
#  Constants / defaults
# ─────────────────────────────────────────────────────────────────────────────
DATA_FOLDER        = 'data'
TASK               = 'picture_naming'
BIN_SIZE           = 100       # ms
N_BINS_HISTORY     = 10
N_EPOCHS           = 50
Y_PCA_COMPONENTS   = 10
KRR_ALPHA          = 1.5
PARALLEL_WORKERS   = 10
PLS_COMPONENTS     = 10        # n_components for PLS regression

# Embeddings are loaded in this order; the same order is used for all plots.
EMBEDDING_NAMES = ['GloVe', 'FastText', 'Word2Vec', 'ConceptNet', 'DINOv2', 'DINOv2Small', 'DINOv3', 'MoCo', 'SimCLR']

IMAGE_FOLDER_NAME  = 'pictureNaming extended all'
EMBEDDINGS_FOLDER  = os.path.join('embeddings', IMAGE_FOLDER_NAME)

CONCEPTNET_URL   = (
    'https://conceptnet.s3.amazonaws.com/downloads/2019/'
    'numberbatch/numberbatch-en-19.08.txt.gz'
)
CONCEPTNET_CACHE = os.path.join(DATA_FOLDER, 'conceptnet-en-19.08.txt.gz')

TASK_TO_XLSX = {
    'picture_naming': os.path.join(
        'data_archive', 'wordset picture naming expanded.xlsx'
    ),
}

# ── Time-warp settings (picture & auditory naming) ────────────────────────────
# AUDITORY_WARP selects which segment of every trial is linearly stretched to a common
# duration before binning, so a chosen event lands at the same time across trials (and,
# under scope='group', across patients).  Applies to BOTH tasks:
#   'none'  = no warping (raw timeline).
#   'stim'  = warp the stimulus segment [stim_onset → stim_offset].
#             picture_naming : trial_onset   → go_cue_onset      (picture-viewing window)
#             auditory_naming: aud_stim_onset → aud_stim_offset  (prompt-word span)
#   'voice' = warp the pre-speech segment [stim_onset → voice_onset] (single segment), so
#             voice onset lands at a common time.  stim_onset is trial_onset (picture) or
#             aud_stim_onset (auditory).
# (Name kept as AUDITORY_WARP for backward compatibility with meta.json / report readers.)
AUDITORY_WARP = 'none'

# Warp scope — what the target segment duration is:
#   'group'   = the median over the pooled trials of EVERY patient in the run. Every
#               patient's segment is stretched to the same duration, so the seg-end event
#               (stim offset / voice onset) falls at the same time for all of them and
#               group figures can mark it with a single line.
#   'patient' = each patient's own median (the original behaviour). The event is fixed
#               within a patient but still differs BETWEEN patients (e.g. 3.23 s for LH
#               vs 4.64 s for RB), which shows up as spurious across-participant spread
#               on any group plot.
AUDITORY_WARP_SCOPE = 'group'

# Resolved at run time when scope='group' (seconds); None = warp to each patient's median.
AUDITORY_WARP_TARGET_SEC = None

# How AUDITORY_WARP_TARGET_SEC was obtained: 'computed' = the pooled median over this run's
# patients, 'pinned' = supplied verbatim via --warp-target-sec.  Recorded in meta.json,
# because the two mean different things when reading a run back: under 'computed' the target
# is a property OF this run's cohort, under 'pinned' it came from somewhere else and the
# cohort had no say in it.  See --warp-target-sec for why that distinction is load-bearing.
AUDITORY_WARP_TARGET_SOURCE = None

# Cache of per-trial warp-segment durations, so the group median does not require
# re-reading the multi-GB trial pkls on every run (invalidated by size + mtime + task).
STIM_DURATION_CACHE = os.path.join(DATA_FOLDER, '_warp_segment_durations.json')

# ── Behavioral-cue alignment ──────────────────────────────────────────────────
# ALIGN_CUE selects which event each trial is sliced around before binning.
# 'none'            — no alignment; full trial length kept (default behaviour)
# 'trial_onset'     — align to trial-start
# 'go_cue'          — align to go-cue / green-screen onset
# 'voice_onset'     — align to speech onset
# 'voice_offset'    — align to speech offset
# 'aud_stim_onset'  — align to auditory-stimulus onset  (auditory_naming only)
# 'aud_stim_offset' — align to auditory-stimulus offset (auditory_naming only)
ALIGN_CUE     = 'none'   # see choices above
ALIGN_BACK    = None     # seconds before cue; None = use full available window (shortest across trials)
ALIGN_FORWARD = None     # seconds after cue;  None = use full available window (shortest across trials)

# Default embeddings for auditory naming: text-only (no picture stimulus). Image
# embeddings are still SUPPORTED for auditory when explicitly requested via --embedding
# (they map the answered word to its picture-stimulus vector) — they are just excluded
# from the default list.
AUDITORY_EMBEDDING_NAMES = ['GloVe', 'FastText', 'Word2Vec', 'ConceptNet']

# Image/vision embedding names (the complement of the text embeddings in EMBEDDING_NAMES).
VISION_EMBEDDING_NAMES = ['DINOv2', 'DINOv2Small', 'DINOv3', 'MoCo', 'SimCLR']

# Answered-word values that indicate an invalid / missing response.

# ─────────────────────────────────────────────────────────────────────────────
#  Loose-accuracy helpers
# ─────────────────────────────────────────────────────────────────────────────
# A predicted word is a "loose" match if it equals the true word after
# article-stripping + lemmatisation, OR shares a WordNet noun synset.

_LOOSE_ARTICLES = ('a ', 'an ', 'the ')

try:
    import nltk
    from nltk.corpus import wordnet as wn
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4',  quiet=True)
    _loose_lemmatizer = WordNetLemmatizer()

    def _normalize_loose(w):
        s = str(w).strip().lower()
        for art in _LOOSE_ARTICLES:
            if s.startswith(art):
                s = s[len(art):]
                break
        base = ''.join(c for c in s if c.isalpha())
        return _loose_lemmatizer.lemmatize(base, pos='n')

    def _share_synset_loose(w1, w2):
        s1 = set(wn.synsets(w1, pos=wn.NOUN))
        s2 = set(wn.synsets(w2, pos=wn.NOUN))
        return bool(s1 & s2)

    def _is_loose_match(true_word, pred_word):
        tn = _normalize_loose(true_word)
        pn = _normalize_loose(pred_word)
        return tn == pn or _share_synset_loose(tn, pn)

except Exception:
    _loose_lemmatizer = WordNetLemmatizer()

    def _normalize_loose(w):
        s = str(w).strip().lower()
        for art in _LOOSE_ARTICLES:
            if s.startswith(art):
                s = s[len(art):]
                break
        return ''.join(c for c in s if c.isalpha())

    def _is_loose_match(true_word, pred_word):
        return _normalize_loose(true_word) == _normalize_loose(pred_word)


# ─────────────────────────────────────────────────────────────────────────────
#  Terminal progress helpers
# ─────────────────────────────────────────────────────────────────────────────

def _step(msg):
    print(f'     ▸  {msg}')

def _ok(msg=''):
    print(f'        ✓  {msg}')

def _warn(msg):
    print(f'        ⚠  {msg}')

def _progress_done():
    print()  # newline after progress bar


class _Tee:
    """Duplicate writes to both the original stream and a log file."""
    def __init__(self, log_file, original_stream):
        self._log    = log_file
        self._term   = original_stream

    def write(self, data):
        self._term.write(data)
        self._term.flush()
        # Replace carriage-returns with newlines so the log file stays readable
        self._log.write(data.replace('\r', '\n'))
        self._log.flush()

    def flush(self):
        self._term.flush()
        self._log.flush()

    def isatty(self):
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Small utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_pkl(path):
    with open(path, 'rb') as f:
        return dill.load(f)


def _normalize_tokens(tokens):
    return np.array([str(t).strip().lower() for t in tokens])


# ── Homonym sense disambiguation (auditory_naming → vision embeddings) ─────────
# The picture gallery splits each homonym into two visually-distinct senses keyed by a
# meaning-number: e.g. gallery bases 'bat1' (baseball bat, category object/tool) and
# 'bat2' (the flying animal). An auditory trial's stimulus word is the bare lemma ('bat'),
# which matches neither, so we resolve the intended sense from the spoken prompt/definition
# and rewrite the vision-lookup label to the sense key — the existing base-match branch in
# _map_to_target then averages only that sense's images. Keyword sets are grounded in the
# {sense_key → category} map verified across all *_picture_naming_labels.pkl:
#   bat1=object/tool  bat2=animal      mouse1=animal     mouse2=object/tool
#   nail1=body part   nail2=object/tool nut1=food/fruit  nut2=object/tool
# (the digit is NOT a consistent sense indicator — must use this table).
HOMONYM_SENSE_KEYWORDS = {
    'bat':   [('bat1',   ('baseball', 'swung', 'swing', 'hit', 'ball', 'wooden', 'sport')),
              ('bat2',   ('animal', 'fly', 'flies', 'flying', 'mammal', 'cave', 'wings', 'nocturnal'))],
    'mouse': [('mouse2', ('computer', 'cursor', 'click', 'control', 'device', 'pointer')),
              ('mouse1', ('animal', 'rodent', 'rat', 'cheese', 'tail', 'furry'))],
    'nail':  [('nail2',  ('hammer', 'hammers', 'pound', 'metal', 'wood', 'carpenter')),
              ('nail1',  ('finger', 'toe', 'body', 'hand', 'manicure', 'grows', 'cut'))],
    'nut':   [('nut1',   ('food', 'almond', 'almonds', 'cashew', 'cashews', 'pecan', 'shell', 'eat', 'snack')),
              ('nut2',   ('bolt', 'screw', 'metal', 'hardware', 'fasten', 'threaded'))],
}


def _resolve_homonym_sense(base_word, prompt_text):
    """Return the gallery sense key (e.g. 'bat1') for a homonym given its spoken prompt,
    or None when it cannot be disambiguated (no keyword hits, or a tie) — in which case the
    caller keeps the bare word and _map_to_target zero-fills it, preserving trial alignment.
    """
    senses = HOMONYM_SENSE_KEYWORDS.get(base_word)
    if not senses or not prompt_text:
        return None
    p = str(prompt_text).lower()
    scored = [(sum(kw in p for kw in kws), key) for key, kws in senses]
    best_n, best_key = max(scored)
    if best_n == 0 or sum(1 for n, _ in scored if n == best_n) > 1:
        return None
    return best_key


def _map_to_target(sources, key, target_labels):
    """Return [N_samples, D] array aligned to target_labels.

    *sources* is a list of ``(results_dict, words_arr)`` pairs tried in
    priority order.  For each label the first source that contains it wins
    (exact match first; then stripping trailing picture-numbers and averaging
    all matching variants, e.g. 'seal' ← mean of seal1/2/3).
    """
    labels_norm = _normalize_tokens(target_labels)

    # Pre-build exact + base lookups for every source once
    lookups = []
    fill_shape = None
    for d, words_arr in sources:
        words_norm = _normalize_tokens(np.asarray(words_arr))
        exact: dict = {w: i for i, w in enumerate(words_norm)}
        base: dict  = {}
        for i, w in enumerate(words_norm):
            base.setdefault(remove_number(w), []).append(i)
        lookups.append((d, exact, base))
        if fill_shape is None and len(words_norm) > 0:
            fill_shape = np.asarray(d[key][0]).squeeze().shape

    # One row per target label, in order.  A label absent from every source is
    # zero-filled (NOT dropped) so the returned array stays aligned with the neural
    # data and label arrays — mirrors the OOV handling for Word2Vec/ConceptNet, and is
    # what lets auditory_naming decode to image embeddings (its answered words are a
    # subset of the picture vocabulary, but need not cover it exactly).
    out, missing = [], []
    for t_raw, t_norm in zip(target_labels, labels_norm):
        vec = None
        for d, exact, base in lookups:
            if t_norm in exact:
                vec = np.asarray(d[key][exact[t_norm]]).squeeze()
                break
            elif t_norm in base:
                variants = [np.asarray(d[key][i]).squeeze() for i in base[t_norm]]
                vec = np.mean(variants, axis=0)
                break
        out.append(vec)
        if vec is None:
            missing.append(t_raw)
        elif fill_shape is None:
            fill_shape = np.asarray(vec).shape
    if missing:
        _warn(f'{key}: {len(missing)}/{len(target_labels)} missing label(s) after all '
              f'fallbacks → zero-filled to keep trial alignment; e.g. {missing[:5]}')
        if fill_shape is None:
            return np.array([])           # no source vectors at all — nothing to map
        zero = np.zeros(fill_shape, dtype=float)
        out = [zero if v is None else v for v in out]
    return np.array(out)


def _layer_keys(d, prefix):
    keys = [k for k in d if k.startswith(prefix)]
    return sorted(keys, key=lambda x: int(x.split('_')[-1]))


def _visual_embed_folders(patient):
    """Priority-ordered list of embedding source folders for *patient*.

    The patient-specific folder (``embeddings/pictureNaming {patient}``) is
    placed first when it exists, then every other subfolder of ``embeddings/``
    is appended as a fallback (sorted for determinism).  This ensures that any
    label not found in the primary folder is looked up in the remaining ones.
    """
    base = 'embeddings'
    specific = os.path.join(base, f'pictureNaming {patient}')
    all_folders = sorted(
        os.path.join(base, name)
        for name in os.listdir(base)
        if os.path.isdir(os.path.join(base, name))
    ) if os.path.isdir(base) else []
    ordered = []
    if os.path.isdir(specific):
        ordered.append(specific)
    for f in all_folders:
        if f not in ordered:
            ordered.append(f)
    return ordered





def _aud_stim_times(trial_df):
    """(aud_stim_onset, aud_stim_offset) per trial, in seconds.

    For auditory naming the stimulus bounds are derived from the prompt word times:
    prompt_word_onsets[i][0] is the first word's onset, prompt_word_offsets[i][-1] the
    last word's offset.  Falls back to explicit stimulus columns (picture naming has
    none, so this is all-NaN there and ignored downstream).
    """
    if TASK == 'auditory_naming' and 'prompt_word_onsets' in trial_df.columns:
        def _edge(v, i):
            a = np.asarray(v, dtype=float).ravel()
            return float(a[i]) if len(a) > 0 else np.nan
        onset = np.array([_edge(v, 0) for v in trial_df['prompt_word_onsets']])
        offset = np.array([_edge(v, -1) for v in trial_df['prompt_word_offsets']])
        return onset, offset
    return (_extract_col(trial_df, 'aud_stim_onset', 'auditory_stimulus_onset',
                         'stimulus_onset'),
            _extract_col(trial_df, 'aud_stim_offset', 'auditory_stimulus_offset',
                         'stimulus_offset'))


def _warp_segment_bounds(trial_df, mode):
    """Per-trial (seg_start, seg_end) warp boundaries in seconds, task-aware.

    seg_start (stimulus onset): trial_onset (picture_naming) / aud_stim_onset (auditory).
    seg_end:
        mode 'stim'  → go_cue_onset (picture_naming) / aud_stim_offset (auditory).
        mode 'voice' → voice_onset (both tasks).
    Any missing column comes back as NaN and is filtered/skipped by the caller.  Single
    source of truth for the task→event mapping used by both the group-target computation
    and the per-patient warp.
    """
    voice = np.asarray(trial_df['voice_onset'].values, dtype=float)
    if TASK == 'auditory_naming':
        seg_start, stim_off = _aud_stim_times(trial_df)
        seg_start = np.asarray(seg_start, dtype=float)
        stim_off  = np.asarray(stim_off,  dtype=float)
    else:  # picture_naming: picture onset == trial onset; go-cue marks the stim offset
        seg_start = np.asarray(trial_df['trial_onset'].values, dtype=float)
        stim_off  = np.asarray(
            _extract_col(trial_df, 'go_cue_onset', 'green_screen_onset'), dtype=float)
    seg_end = voice if mode == 'voice' else stim_off
    return seg_start, seg_end


def _stim_duration_cache_key(df_path):
    st = os.stat(df_path)
    return {'size': st.st_size, 'mtime': int(st.st_mtime)}


def load_segment_durations(patients, refresh=False):
    """Per-trial warp-segment durations (seconds) per patient, for BOTH warp modes.

    Returns ``{patient: {'stim': np.ndarray, 'voice': np.ndarray}}`` where 'stim' is the
    [stim_onset → stim_offset] duration and 'voice' the [stim_onset → voice_onset]
    duration (task-aware, see ``_warp_segment_bounds``).  Reading these means loading the
    trial pkls, which are 0.9–4.5 GB apiece, so results are cached in STIM_DURATION_CACHE
    and re-read only when a pkl changes (size+mtime), the task differs, or ``refresh=True``.
    Patients are loaded one at a time and freed immediately.
    """
    cache = {}
    if os.path.exists(STIM_DURATION_CACHE) and not refresh:
        try:
            with open(STIM_DURATION_CACHE) as f:
                cache = json.load(f)
        except Exception as e:
            _warn(f'Could not read {STIM_DURATION_CACHE} ({e}); rebuilding')
            cache = {}

    out, dirty = {}, False
    for patient in patients:
        df_path = _find_df_path(os.path.join(DATA_FOLDER, patient), patient, TASK)
        if df_path is None or not os.path.exists(df_path):
            _warn(f'{patient}: no {TASK} df — excluded from the group warp target')
            continue
        key = _stim_duration_cache_key(df_path)
        hit = cache.get(patient)
        if hit and not refresh and hit.get('size') == key['size'] \
                and hit.get('mtime') == key['mtime'] and hit.get('task') == TASK \
                and 'stim_dur_s' in hit and 'voice_dur_s' in hit:
            out[patient] = {'stim':  np.asarray(hit['stim_dur_s'],  dtype=float),
                            'voice': np.asarray(hit['voice_dur_s'], dtype=float)}
            continue
        _step(f'{patient}: reading warp-segment durations from {os.path.basename(df_path)} '
              f'({key["size"] / 1e9:.1f} GB, one-time) …')
        trial_df = load_pkl(df_path)
        if isinstance(trial_df, dict):
            trial_df = pd.DataFrame(trial_df)
        seg_start, stim_end  = _warp_segment_bounds(trial_df, 'stim')
        _,         voice_end = _warp_segment_bounds(trial_df, 'voice')
        del trial_df
        stim_dur  = stim_end  - seg_start
        voice_dur = voice_end - seg_start
        stim_dur  = stim_dur[np.isfinite(stim_dur)   & (stim_dur  > 0)]
        voice_dur = voice_dur[np.isfinite(voice_dur) & (voice_dur > 0)]
        out[patient] = {'stim': stim_dur, 'voice': voice_dur}
        cache[patient] = {**key, 'task': TASK,
                          'stim_dur_s':  [float(d) for d in stim_dur],
                          'voice_dur_s': [float(d) for d in voice_dur]}
        dirty = True

    if dirty:
        try:
            with open(STIM_DURATION_CACHE, 'w') as f:
                json.dump(cache, f)
            _ok(f'Warp-segment-duration cache updated → {STIM_DURATION_CACHE}')
        except Exception as e:
            _warn(f'Could not write {STIM_DURATION_CACHE} ({e})')
    return out


def compute_group_segment_duration(patients, mode, refresh=False):
    """Median warp-segment duration (seconds) over the pooled trials of all `patients`,
    for the warp `mode` ('stim' or 'voice').

    This is the warp target under ``--warp-scope group``: every patient's segment is
    stretched to this one duration, so the seg-end event (stim offset / voice onset) lands
    at the same time for everybody.  Returns (median_sec, per_patient_medians) — or
    (None, {}) if no patient has a usable segment, in which case the caller falls back to
    per-patient medians.
    """
    per_all = load_segment_durations(patients, refresh=refresh)
    per_patient = {p: d[mode] for p, d in per_all.items() if len(d[mode]) > 0}
    if not per_patient:
        return None, {}
    pooled = np.concatenate([per_patient[p] for p in sorted(per_patient)])
    group_median = float(np.median(pooled))
    medians = {p: float(np.median(d)) for p, d in sorted(per_patient.items())}
    _section(f'Group {mode}-segment duration (warp target)')
    for p, m in medians.items():
        print(f'    {p:6s}  n={len(per_patient[p]):3d} trials  median={m:.3f} s  '
              f'[{per_patient[p].min():.3f}, {per_patient[p].max():.3f}]')
    print(f'    {"GROUP":6s}  n={len(pooled):3d} trials  median={group_median:.3f} s  '
          f'← every patient warped to this')
    return group_median, medians


def _linear_time_warp(data, fs, seg_start, seg_end, timing_arrays, target_dur_sec=None):
    """Linearly warp the [seg_start, seg_end] segment of each trial to a common duration.

    The pre-segment (before seg_start) is left identical and the post-segment (after
    seg_end) is kept intact and rigidly shifted, so only the chosen segment is
    stretched/compressed.  Every array in ``timing_arrays`` is remapped onto the warped
    timeline (seg_start/seg_end themselves, if present, remap consistently).

    Parameters
    ----------
    data : list/array of (n_channels, n_time) arrays
        Raw (unbinned) neural data, one entry per trial.
    fs : int
        Sampling rate in Hz.
    seg_start, seg_end : np.ndarray, shape (n_trials,)
        Per-trial warp-segment boundaries (seconds).  seg_start is the stimulus onset
        (trial_onset / aud_stim_onset); seg_end is the stimulus offset (go_cue /
        aud_stim_offset) for a 'stim' warp or voice_onset for a 'voice' warp.
    timing_arrays : dict[str, np.ndarray]
        Timing-cue arrays (length n_trials) whose values are remapped onto the warped
        timeline.
    target_dur_sec : float or None
        Duration to warp every segment to, in seconds (``--warp-scope group``: the median
        over all patients, so the seg-end event lands at the same time in every patient).
        None warps to THIS patient's own median segment duration (``--warp-scope patient``).

    A trial whose segment is invalid (NaN bound, seg_end <= seg_start, seg_start < 0, or
    seg_end past the trial's end) is left unwarped (identity) and counted.

    Returns
    -------
    data_warped : np.ndarray, shape (n_trials, n_channels, n_time_warped)
        Warped data truncated to the shortest resulting trial.
    timing_arrays_w : dict  – updated copies of all timing arrays.
    """
    from scipy.interpolate import interp1d

    n = len(data)
    seg_start = np.asarray(seg_start, dtype=float)
    seg_end   = np.asarray(seg_end,   dtype=float)
    lengths   = np.array([data[i].shape[1] for i in range(n)])

    start_idx = np.array([int(np.round(s * fs)) if np.isfinite(s) else -1 for s in seg_start])
    end_idx   = np.array([int(np.round(e * fs)) if np.isfinite(e) else -1 for e in seg_end])
    valid = (np.isfinite(seg_start) & np.isfinite(seg_end)
             & (start_idx >= 0) & (end_idx > start_idx) & (end_idx <= lengths))

    timing_arrays_w = {k: v.copy() for k, v in timing_arrays.items()}
    if not np.any(valid):
        _warn('No trial has a valid warp segment → returning data unwarped')
        return np.array([d[:, :int(lengths.min())] for d in data]), timing_arrays_w

    durations = (end_idx - start_idx)[valid]
    if target_dur_sec is not None:
        median_seg = max(int(round(target_dur_sec * fs)), 1)
        scope = f'group target {target_dur_sec:.3f} s'
    else:
        median_seg = int(np.median(durations))
        scope = "this patient's median"
    _step(f'Time-warp: segment durations min={durations.min()} max={durations.max()} '
          f'→ {median_seg} samples ({median_seg/fs:.3f} s, {scope})')

    def _resample(seg, n_out):
        # Per-channel linear resample; a 0/1-sample segment (interp1d needs >=2 nodes)
        # is constant-held.
        if seg.shape[1] < 2:
            return np.repeat(seg.astype(float), n_out, axis=1)
        orig_t = np.arange(seg.shape[1])
        warp_t = np.linspace(0, seg.shape[1] - 1, n_out)
        out = np.zeros((seg.shape[0], n_out))
        for ch in range(seg.shape[0]):
            f = interp1d(orig_t, seg[ch], kind='linear', fill_value='extrapolate')
            out[ch] = f(warp_t)
        return out

    def _warp_cue(cue_time, s_idx, e_idx, median_seg, fs):
        if np.isnan(cue_time):
            return cue_time
        cue_idx = cue_time * fs
        if cue_idx < s_idx:                       # pre-segment: identity
            return cue_time
        orig = e_idx - s_idx
        if cue_idx <= e_idx:                      # within segment: proportional stretch
            if orig <= 0:
                return cue_time
            rel = (cue_idx - s_idx) / orig
            return (s_idx + rel * median_seg) / fs
        return cue_time + (median_seg - orig) / fs   # past segment: rigid shift

    data_warped = []
    n_skip = 0
    for i in range(n):
        trial = data[i]  # (n_channels, n_time)
        if not valid[i]:
            data_warped.append(trial)            # leave unwarped, cues unchanged
            n_skip += 1
            continue
        s, e = int(start_idx[i]), int(end_idx[i])
        pre, during, post = trial[:, :s], trial[:, s:e], trial[:, e:]
        data_warped.append(np.concatenate(
            [pre, _resample(during, median_seg), post], axis=1))
        for k in timing_arrays_w:
            timing_arrays_w[k][i] = _warp_cue(timing_arrays[k][i], s, e, median_seg, fs)

    if n_skip:
        _warn(f'{n_skip}/{n} trials had an invalid warp segment → left unwarped')

    shortest = min(d.shape[1] for d in data_warped)
    data_warped = np.array([d[:, :shortest] for d in data_warped])
    _ok(f'Warped data shape: {data_warped.shape}')
    return data_warped, timing_arrays_w


# ─────────────────────────────────────────────────────────────────────────────
#  Shared embedding model loading  (done ONCE for the whole batch run)
# ─────────────────────────────────────────────────────────────────────────────

def load_shared_embedding_models():
    """Load all heavyweight shared embedding models and pickle files once."""
    _header('Loading shared embedding models  (one-time cost)')
    shared = {}

    # ── GloVe ────────────────────────────────────────────────────────────────
    _step('GloVe 840B 300-D …')
    shared['glove'] = GloVe(dim=300, name='840B')
    _ok(f'GloVe  ({len(shared["glove"].stoi):,} vocab tokens)')

    # ── FastText ─────────────────────────────────────────────────────────────
    _step('FastText (simple wiki) 300-D …')
    shared['fasttext'] = FastText(language='simple')
    _ok('FastText loaded')

    # ── Word2Vec ─────────────────────────────────────────────────────────────
    _step('Word2Vec google-news-300 …')
    import gensim.downloader as gensim_api
    shared['word2vec'] = gensim_api.load('word2vec-google-news-300')
    _ok('Word2Vec loaded')

    # ── ConceptNet Numberbatch ────────────────────────────────────────────────
    _step('ConceptNet Numberbatch …')
    if not os.path.exists(CONCEPTNET_CACHE):
        _step(f'Downloading from {CONCEPTNET_URL}')
        urlretrieve(CONCEPTNET_URL, CONCEPTNET_CACHE)
        _ok(f'Downloaded → {CONCEPTNET_CACHE}')
    cn_embed = {}
    with gzip.open(CONCEPTNET_CACHE, 'rt', encoding='utf-8') as f:
        header = f.readline().strip().split()
        n_words, emb_dim = int(header[0]), int(header[1])
        for i, line in enumerate(f):
            if i % 100_000 == 0:
                _progress(i, n_words, 'ConceptNet entries')
            parts = line.strip().split(' ')
            word  = parts[0][6:] if parts[0].startswith('/c/en/') else parts[0]
            vec   = np.array(parts[1:], dtype=np.float32)
            if len(vec) == emb_dim:
                cn_embed[word] = vec
    _progress_done()
    shared['conceptnet'] = cn_embed
    _ok(f'ConceptNet  ({len(cn_embed):,} entries, dim={emb_dim})')

    # ── Pre-computed multimodal pickles (shared / not patient-specific) ───────
    for key, fname in [
        ('clip_layerwise',   'clip_layerwise_embeddings.pk'),
        ('vit',              'vit_imagenet_layerwise_embeddings.pk'),
    ]:
        _step(f'Loading {fname} …')
        with open(os.path.join(EMBEDDINGS_FOLDER, fname), 'rb') as f:
            shared[key] = pk.load(f)
        _ok(f'{key} loaded  ({len(shared[key].get("words", []))} words)')

    # DINOv2 / SimCLR are loaded per-patient (patient-specific folders may exist),
    # so we only store their default path here as a fallback marker.
    shared['_dinov2_default_folder']       = EMBEDDINGS_FOLDER
    shared['_dinov2_small_default_folder']  = EMBEDDINGS_FOLDER
    shared['_dinov3_default_folder']        = EMBEDDINGS_FOLDER
    shared['_moco_default_folder']          = EMBEDDINGS_FOLDER
    shared['_simclr_default_folder']        = EMBEDDINGS_FOLDER

    _section('All shared models ready')
    return shared


# ─────────────────────────────────────────────────────────────────────────────
#  Per-patient data loading & preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def load_patient_data(patient):
    """Load, bin, and clean neural data for one patient."""
    patient_folder = os.path.join(DATA_FOLDER, patient)

    df_path     = _find_df_path(patient_folder, patient, TASK)
    labels_path = os.path.join(patient_folder, f'{patient}_{TASK}_labels.pkl')
    if df_path is None or not os.path.exists(labels_path):
        raise FileNotFoundError(
            f'Missing data for {patient}: df_path={df_path}, labels_path={labels_path}'
        )

    # Channels: task-specific > patient-level > picture_naming (shared montage) > None
    #
    # Auditory-naming exports historically shipped without a channels file, so the
    # channel set fell back to bare integer indices with no `clean` / shank-prefix
    # mask.  Downstream (cross_task_cotrain.load_patient) then could not name-match
    # picture vs auditory channels: the intersection came back empty and it fell
    # back to positional `ch{i}` pairing, misaligning electrodes wherever the two
    # runs dropped different channels.
    #
    # The auditory and picture recordings are the same implant: the channel_name
    # arrays are identical in count and order (verified for AA/AZ/DR).  So reuse
    # the picture_naming channels file for any task lacking its own, giving both
    # tasks matching anatomical names AND the same clean/exclusion mask.
    for ch_path in [
        os.path.join(patient_folder, f'{patient}_{TASK}_channels.pkl'),
        os.path.join(patient_folder, f'{patient}_channels.pkl'),
        os.path.join(patient_folder, f'{patient}_picture_naming_channels.pkl'),
    ]:
        if os.path.exists(ch_path):
            channels_path = ch_path
            break
    else:
        channels_path = None

    # ── Load files ───────────────────────────────────────────────────────────
    _step(f'Loading {os.path.basename(df_path)} …')
    trial_df   = load_pkl(df_path)
    labels_df  = load_pkl(labels_path)
    channels_df = load_pkl(channels_path) if channels_path else None
    for obj in (trial_df, labels_df, channels_df):
        if isinstance(obj, dict):
            obj = pd.DataFrame(obj)
    if isinstance(trial_df,   dict): trial_df   = pd.DataFrame(trial_df)
    if isinstance(labels_df,  dict): labels_df  = pd.DataFrame(labels_df)
    if isinstance(channels_df, dict) and channels_df is not None:
        channels_df = pd.DataFrame(channels_df)
    _ok(f'trial_df {trial_df.shape},  labels_df {labels_df.shape}')

    # ── Extract timing & labels ───────────────────────────────────────────────
    fs              = int(trial_df['fs'].iloc[0])
    n_samp_per_bin  = fs * BIN_SIZE // 1000
    data_list       = list(trial_df['hg_data'].values)
    trial_onset     = trial_df['trial_onset'].values.astype(float)
    go_cue_onset    = _extract_col(trial_df, 'go_cue_onset', 'green_screen_onset')
    trial_offset    = trial_df['trial_offset'].values.astype(float)
    voice_onset     = trial_df['voice_onset'].values.astype(float)
    voice_offset    = trial_df['voice_offset'].values.astype(float)
    target_labels   = trial_df['target_word'].values.astype(str)
    answer_labels   = trial_df['answered_word'].values.astype(str)
    # Auditory naming: the spoken prompt/definition per trial (tokens joined), used to
    # disambiguate homonym senses for vision-embedding targets. Empty for picture naming.
    if TASK == 'auditory_naming' and 'prompt_words' in trial_df.columns:
        prompt_texts = np.array(
            [' '.join(np.asarray(pw).ravel().astype(str)).lower() if pw is not None else ''
             for pw in trial_df['prompt_words'].values],
            dtype=object)
    else:
        prompt_texts = np.array([''] * len(trial_df), dtype=object)
    bad_trials      = (trial_df['bad_trials'].values.astype(bool)
                       if 'bad_trials' in trial_df.columns
                       else np.ones(len(trial_df), dtype=bool))
    # Auditory naming: stimulus bounds from the prompt word times (see _aud_stim_times).
    aud_stim_onset, aud_stim_offset = _aud_stim_times(trial_df)
    if TASK == 'auditory_naming' and np.any(np.isfinite(aud_stim_onset)):
        _ok(f'aud_stim_onset range:  [{np.nanmin(aud_stim_onset):.3f}, '
            f'{np.nanmax(aud_stim_onset):.3f}] s')
        _ok(f'aud_stim_offset range: [{np.nanmin(aud_stim_offset):.3f}, '
            f'{np.nanmax(aud_stim_offset):.3f}] s')
    _ok(f'fs={fs} Hz  |  {len(data_list)} trials  |  '
        f'data shape[0]: {data_list[0].shape}')

    # ── Optional linear time warp on raw data (picture & auditory naming) ──────
    # Warp is applied before any binning/alignment so timing updates are at the
    # native sampling resolution (typically 1 ms when fs=1000).  'stim' warps
    # [stim_onset → stim_offset]; 'voice' warps [stim_onset → voice_onset]; the
    # task→event mapping lives in _warp_segment_bounds.
    if AUDITORY_WARP != 'none':
        seg_start, seg_end = _warp_segment_bounds(trial_df, AUDITORY_WARP)
        if not np.any(np.isfinite(seg_start) & np.isfinite(seg_end)
                      & (seg_end > seg_start)):
            _warn(f'{AUDITORY_WARP!r} warp requested but no trial has a valid '
                  f'[{AUDITORY_WARP}] segment; skipping warp')
        else:
            _step(f'Applying {AUDITORY_WARP!r} time warp to raw data …')
            data_w, t_w = _linear_time_warp(
                data_list, fs=fs,
                seg_start=seg_start,
                seg_end=seg_end,
                timing_arrays={
                    'trial_onset':     trial_onset,
                    'trial_offset':    trial_offset,
                    'go_cue':          go_cue_onset,
                    'voice_onset':     voice_onset,
                    'voice_offset':    voice_offset,
                    'aud_stim_onset':  aud_stim_onset,
                    'aud_stim_offset': aud_stim_offset,
                },
                # None under --warp-scope patient → each patient's own median
                target_dur_sec=AUDITORY_WARP_TARGET_SEC,
            )
            data_list        = list(data_w)
            trial_onset      = t_w['trial_onset']
            trial_offset     = t_w['trial_offset']
            go_cue_onset     = t_w['go_cue']
            voice_onset      = t_w['voice_onset']
            voice_offset     = t_w['voice_offset']
            aud_stim_onset   = t_w['aud_stim_onset']
            aud_stim_offset  = t_w['aud_stim_offset']
            _ok('Warp applied before binning at native sampling rate')

    # ── Channel mask ──────────────────────────────────────────────────────────
    # Guard: a reused channels_df (e.g. picture_naming reused for auditory) must
    # have one row per recorded channel, in the same order, or positional indexing
    # into hg_data would be wrong.  If counts disagree (rare montage mismatch),
    # fall back to integer names rather than silently mis-indexing.
    if channels_df is not None and len(channels_df) != data_list[0].shape[0]:
        _warn(f'{patient}/{TASK}: channels_df has {len(channels_df)} rows but '
              f'neural data has {data_list[0].shape[0]} channels - ignoring '
              f'channels_df and using integer channel indices.')
        channels_df = None

    if channels_df is not None:
        channel_names_all = channels_df['channel_name'].values.astype(str)
        bad_channels = (np.where(~channels_df['clean'].values.astype(bool))[0]
                        if 'clean' in channels_df.columns
                        else np.array([], dtype=int))
    else:
        n_ch              = data_list[0].shape[0]
        channel_names_all = np.array([str(i) for i in range(n_ch)])
        bad_channels      = np.array([], dtype=int)

    if 'bad_channels' in trial_df.columns:
        for bc in trial_df['bad_channels'].values:
            if bc is not None and len(bc) > 0:
                for ch in np.asarray(bc).ravel():
                    if (isinstance(ch, (int, float, np.integer, np.floating))
                            and not np.isnan(float(ch))):
                        bad_channels = np.union1d(bad_channels, [int(ch)])

    remaining_ch_idx = np.delete(np.arange(len(channel_names_all)), bad_channels)
    channel_names    = channel_names_all[remaining_ch_idx]

    # ── Patient-specific channel exclusions ──────────────────────────────────
    # EDIT HERE to change which shank prefixes are excluded per patient.
    _PATIENT_EXCLUDE_PREFIXES = {
        'LH': ('O', 'V', 'P', 'Q', 'R'),   # non-language shanks
        'RB': ('V',),                        # non-language shank
    }
    if patient in _PATIENT_EXCLUDE_PREFIXES:
        _prefixes = _PATIENT_EXCLUDE_PREFIXES[patient]
        _ex = np.array(
            [i for i, cn in enumerate(channel_names)
             if str(cn).startswith(_prefixes)],
            dtype=int,
        )
        if len(_ex) > 0:
            bad_channels     = np.union1d(bad_channels, remaining_ch_idx[_ex]).astype(int)
            channel_names    = np.delete(channel_names, _ex, axis=0)
            remaining_ch_idx = np.delete(np.arange(len(channel_names_all)), bad_channels)
            _ok(f'{patient}: removed {_prefixes} shank(s) ({len(_ex)} channels)')

    _ok(f'{bad_trials.sum()} good trials  |  {len(channel_names)} good channels')

    # ── Bin neural data ───────────────────────────────────────────────────────
    _step('Binning neural data …')
    adjusted_fs        = int(1000 / BIN_SIZE)
    actual_back_sec    = None
    actual_forward_sec = None

    if ALIGN_CUE == 'none':
        shortest_trial = min(d.shape[1] for d in data_list)
        data           = np.array([d[:, :shortest_trial] for d in data_list])
        min_length     = data.shape[2] // n_samp_per_bin * n_samp_per_bin
        data           = data[:, :, :min_length]
        data_binned    = data.reshape(data.shape[0], data.shape[1], -1, n_samp_per_bin).mean(axis=3)
        del data
        gc.collect()
        _ok(f'data_binned: {data_binned.shape}  (n_trials, n_channels, n_bins)')
    else:
        _cue_arrays = {
            'trial_onset':     trial_onset,
            'go_cue':          go_cue_onset,
            'voice_onset':     voice_onset,
            'voice_offset':    voice_offset,
            'aud_stim_onset':  aud_stim_onset,
            'aud_stim_offset': aud_stim_offset,
        }
        if ALIGN_CUE not in _cue_arrays:
            raise ValueError(f'Unknown ALIGN_CUE: {ALIGN_CUE!r}')
        cue_arr = _cue_arrays[ALIGN_CUE]
        _back_str = f'{ALIGN_BACK}s' if ALIGN_BACK   is not None else 'full'
        _fwd_str  = f'{ALIGN_FORWARD}s' if ALIGN_FORWARD is not None else 'full'
        _step(f'Cue-alignment enabled  (cue={ALIGN_CUE!r}, '
              f'requested back={_back_str}, fwd={_fwd_str}) …')
        cue_samp = np.array([
            int(round(c * fs)) if np.isfinite(c) else -1
            for c in cue_arr
        ])
        good_mask = bad_trials & (cue_samp >= 0)
        if good_mask.sum() == 0:
            raise ValueError(
                f'No good trials with finite {ALIGN_CUE!r} for cue alignment'
            )
        if ALIGN_BACK is None:
            avail_backs = np.array([
                cue_samp[i]
                for i in range(len(data_list)) if good_mask[i]
            ])
        else:
            back_samp_req = int(round(ALIGN_BACK * fs))
            avail_backs = np.array([
                min(back_samp_req, cue_samp[i])
                for i in range(len(data_list)) if good_mask[i]
            ])
        if ALIGN_FORWARD is None:
            avail_fwds = np.array([
                data_list[i].shape[1] - cue_samp[i]
                for i in range(len(data_list)) if good_mask[i]
            ])
        else:
            fwd_samp_req = int(round(ALIGN_FORWARD * fs))
            avail_fwds = np.array([
                min(fwd_samp_req, data_list[i].shape[1] - cue_samp[i])
                for i in range(len(data_list)) if good_mask[i]
            ])
        global_back_samp = (int(avail_backs.min()) // n_samp_per_bin) * n_samp_per_bin
        global_fwd_samp  = (int(avail_fwds.min())  // n_samp_per_bin) * n_samp_per_bin
        total_samp = global_back_samp + global_fwd_samp
        if total_samp < n_samp_per_bin:
            raise ValueError(
                f'Cue-aligned window too short: back={global_back_samp}, '
                f'fwd={global_fwd_samp} samples (need >= {n_samp_per_bin})'
            )
        actual_back_sec    = global_back_samp / fs
        actual_forward_sec = global_fwd_samp  / fs
        _ok(f'Global window: back={global_back_samp} samp ({actual_back_sec:.3f}s), '
            f'fwd={global_fwd_samp} samp ({actual_forward_sec:.3f}s)')
        n_ch_raw = data_list[0].shape[0]
        aligned = []
        for i in range(len(data_list)):
            if cue_samp[i] >= 0:
                start = cue_samp[i] - global_back_samp
                end   = cue_samp[i] + global_fwd_samp
                aligned.append(data_list[i][:, start:end])
            else:
                aligned.append(np.zeros((n_ch_raw, total_samp), dtype=data_list[i].dtype))
        data        = np.array(aligned)
        del aligned
        data_binned = data.reshape(data.shape[0], data.shape[1], -1, n_samp_per_bin).mean(axis=3)
        del data
        gc.collect()
        _ok(f'data_binned (cue-aligned to {ALIGN_CUE!r}): {data_binned.shape}  '
            f'(n_trials, n_channels, n_bins)')

    # ── Remove bad channels / bad trials ─────────────────────────────────────
    clean_data_binned   = np.delete(data_binned, bad_channels, axis=1)[bad_trials]
    del data_binned
    gc.collect()
    clean_voice_onset   = voice_onset[bad_trials]
    clean_voice_offset  = voice_offset[bad_trials]
    clean_go_cue_onset  = go_cue_onset[bad_trials]
    clean_trial_onset   = trial_onset[bad_trials]
    clean_aud_stim_onset  = aud_stim_onset[bad_trials]
    clean_aud_stim_offset = aud_stim_offset[bad_trials]
    clean_target_labels = target_labels[bad_trials]
    clean_answer_labels = answer_labels[bad_trials]
    clean_prompt_texts  = prompt_texts[bad_trials]
    _ok(f'clean_data_binned: {clean_data_binned.shape}')

    # ── Auditory naming: remove trials with invalid answered words ─────────────
    if TASK == 'auditory_naming':
        valid_mask = np.array([_is_valid_answer(w) for w in clean_answer_labels])
        n_invalid  = int((~valid_mask).sum())
        if n_invalid > 0:
            _warn(f'Removing {n_invalid} trials with invalid answered words '
                  f'(e.g. {clean_answer_labels[~valid_mask][:5].tolist()})')
            clean_data_binned     = clean_data_binned[valid_mask]
            clean_voice_onset     = clean_voice_onset[valid_mask]
            clean_voice_offset    = clean_voice_offset[valid_mask]
            clean_go_cue_onset    = clean_go_cue_onset[valid_mask]
            clean_trial_onset     = clean_trial_onset[valid_mask]
            clean_aud_stim_onset  = clean_aud_stim_onset[valid_mask]
            clean_aud_stim_offset = clean_aud_stim_offset[valid_mask]
            clean_target_labels   = clean_target_labels[valid_mask]
            clean_answer_labels   = clean_answer_labels[valid_mask]
            clean_prompt_texts    = clean_prompt_texts[valid_mask]
        _ok(f'{valid_mask.sum()} trials kept after invalid-answer filter')

    # ── Vision decoding target label: disambiguate homonym senses ──────────────
    # For a homonym trial (mouse/bat/nail/nut) the bare stimulus word matches neither
    # gallery sense key ('bat1'/'bat2'), so rewrite it to the sense implied by the spoken
    # prompt; _map_to_target then averages only that sense's images. Non-homonym words keep
    # their bare form (→ mean of all variants). Unresolved → bare word (→ zero-fill).
    if TASK == 'auditory_naming':
        clean_vision_label = np.array(clean_target_labels, dtype=object)
        _n_unresolved = 0
        for _i, (_tw, _ptxt) in enumerate(zip(clean_target_labels, clean_prompt_texts)):
            _base = ''.join(c for c in str(_tw).lower() if c.isalpha())
            if _base in HOMONYM_SENSE_KEYWORDS:
                _sk = _resolve_homonym_sense(_base, _ptxt)
                if _sk is not None:
                    clean_vision_label[_i] = _sk
                else:
                    _n_unresolved += 1
        if _n_unresolved:
            _warn(f'{_n_unresolved} homonym trial(s) could not be sense-resolved from the '
                  f'prompt; left as bare word (vision → zero-fill)')
        clean_vision_label = clean_vision_label.astype(str)
    else:
        clean_vision_label = clean_target_labels

    # ── Semantic categories ───────────────────────────────────────────────────
    _step('Assigning semantic categories …')
    # For auditory naming use answered words for category lookup; fall back to
    # target word if the answered word is not found in the labels dict.
    _primary_labels   = clean_answer_labels if TASK == 'auditory_naming' else clean_target_labels
    _secondary_labels = clean_target_labels  # fallback for auditory naming
    if 'class' in labels_df.columns:
        w2c = dict(zip(
            labels_df['target_word'].astype(str),
            labels_df['class'].astype(str),
        ))
        # Size the array from the FULL class vocabulary, not just the labels that
        # happen to resolve on the first pass.  A numpy unicode array is fixed-width
        # and truncates silently on assignment, so when 'object/tool' (11 chars) is
        # absent at construction the dtype is sized by 'food/fruit' (10) and every
        # label the fallback loop below resolves becomes 'object/too'.  That is not
        # hypothetical: it fired for both CP and RB (their auditory answers reach
        # 'object/tool' only via the fallback), which is where the downstream
        # _CATEGORY_FIX patch in analysis/cross_task/cross_task_prediction_mds.py
        # came from.  Fix it here so consumers do not each need that patch.
        _cat_width = max([len(c) for c in w2c.values()] + [len('unknown')])
        word_category = np.array([w2c.get(w, 'unknown') for w in _primary_labels],
                                 dtype=f'<U{_cat_width}')
        n_unk = (word_category == 'unknown').sum()
        if n_unk > 0:
            base2cat = {
                remove_number(str(lbl)).lower(): cat
                for lbl, cat in w2c.items()
            }
            for i, (wp, wt, cat) in enumerate(
                zip(_primary_labels, _secondary_labels, word_category)
            ):
                if cat == 'unknown':
                    # Try base form of primary label
                    word_category[i] = base2cat.get(
                        remove_number(str(wp)).lower(), 'unknown'
                    )
                if word_category[i] == 'unknown' and TASK == 'auditory_naming':
                    # Fall back to target (stimulus) word
                    word_category[i] = base2cat.get(
                        remove_number(str(wt)).lower(), 'unknown'
                    )
            n_resolved = n_unk - (word_category == 'unknown').sum()
            _ok(f'Resolved {n_resolved}/{n_unk} unknown categories')
    elif TASK in TASK_TO_XLSX and os.path.exists(TASK_TO_XLSX[TASK]):
        df_xlsx   = pd.read_excel(TASK_TO_XLSX[TASK])
        wcol      = df_xlsx.columns[0]
        df_xlsx.set_index(wcol, inplace=True)
        cat_sr    = df_xlsx.fillna(0).apply(pd.to_numeric).idxmax(axis=1).reset_index()
        cat_sr.columns = [wcol, 'Category']
        w2c       = dict(zip(cat_sr[wcol], cat_sr['Category']))
        lex_tmp   = np.array([remove_number(t).lower() for t in _primary_labels])
        word_category = np.array([w2c.get(w, 'unknown') for w in lex_tmp])
        word_category = np.array([
            'food and fruit' if w in ('fruit', 'food (exclude fruit)') else w
            for w in word_category
        ])
        _ok(f'Categories from xlsx')
    else:
        word_category = np.array(['unknown'] * len(clean_target_labels))
        _warn('No category source found; all categories = "unknown"')

    clean_word_category = word_category
    _ok(str(dict(collections.Counter(clean_word_category))))

    # ── Lemmatise labels ──────────────────────────────────────────────────────
    # For auditory naming, embeddings are looked up by the answered word.
    _step('Lemmatising target labels …')
    lemmatizer    = WordNetLemmatizer()
    _embed_source = clean_answer_labels if TASK == 'auditory_naming' else clean_target_labels
    if any(kw in TASK for kw in ('Flashing', 'auditory', 'picture')):
        target_lexeme = np.array([remove_number(t).lower() for t in _embed_source])
    else:
        target_lexeme = np.array([str(w).lower() for w in _embed_source])

    target_lemma = np.array([
        lemmatizer.lemmatize(''.join(c for c in w if c.isalpha()), pos='n')
        for w in target_lexeme
    ])

    # Build target_concept (disambiguates homonyms)
    base_of_lex = np.array([''.join(c for c in w if c.isalpha()) for w in target_lexeme])
    _b2v, _b2c  = {}, {}
    for lex in np.unique(target_lexeme):
        base = ''.join(c for c in lex if c.isalpha())
        _b2v.setdefault(base, set()).add(lex)
    for base, cat in zip(base_of_lex, clean_word_category):
        _b2c.setdefault(base, set()).add(cat)
    ambig = {b for b in _b2v if len(_b2v[b]) > 1 or len(_b2c.get(b, set())) > 1}
    target_concept = np.array([
        f'{base}({cat})' if base in ambig else base
        for base, cat in zip(base_of_lex, clean_word_category)
    ])
    _ok(f'{len(np.unique(target_concept))} unique concepts, '
        f'{len(ambig)} homonym base(s)')

    # ── Compute cue times relative to the final (possibly warped) timeline ───
    _ref_cue = ALIGN_CUE if ALIGN_CUE != 'none' else 'trial_onset'
    _step(f'Computing cue times relative to {_ref_cue!r} …')
    _all_clean_cues = {
        'trial_onset':     clean_trial_onset,
        'go_cue':          clean_go_cue_onset,
        'voice_onset':     clean_voice_onset,
        'voice_offset':    clean_voice_offset,
        'aud_stim_onset':  clean_aud_stim_onset,
        'aud_stim_offset': clean_aud_stim_offset,
    }
    _ref_arr = _all_clean_cues[_ref_cue]
    rel_cues = {}
    for cue_name, cue_vals in _all_clean_cues.items():
        if cue_name == _ref_cue:
            rel_cues[cue_name] = {'mean': 0.0, 'std': 0.0}
        else:
            diff = cue_vals - _ref_arr
            rel_cues[cue_name] = {
                'mean': float(np.nanmean(diff)),
                'std':  float(np.nanstd(diff)),
            }
        _ok(f'{cue_name:>20s}:  '
            f'mean={rel_cues[cue_name]["mean"]:+.3f}s  '
            f'std={rel_cues[cue_name]["std"]:.3f}s')

    return dict(
        patient               = patient,
        fs                    = fs,
        adjusted_fs           = adjusted_fs,
        clean_data_binned     = clean_data_binned,
        clean_target_labels   = clean_target_labels,
        clean_answer_labels   = clean_answer_labels,
        vision_label          = clean_vision_label,
        clean_channel_names   = np.array(channel_names),
        clean_word_category   = clean_word_category,
        clean_voice_onset     = clean_voice_onset,
        clean_voice_offset    = clean_voice_offset,
        clean_go_cue_onset    = clean_go_cue_onset,
        clean_trial_onset     = clean_trial_onset,
        clean_aud_stim_onset  = clean_aud_stim_onset,
        clean_aud_stim_offset = clean_aud_stim_offset,
        trial_onset           = trial_onset,
        go_cue_onset          = go_cue_onset,
        trial_offset          = trial_offset,
        voice_onset           = voice_onset,
        target_lexeme         = target_lexeme,
        target_lemma          = target_lemma,
        target_concept        = target_concept,
        labels_df             = labels_df,
        warp                  = AUDITORY_WARP,
        align_cue             = ALIGN_CUE,
        rel_cues_reference    = _ref_cue,
        actual_back_sec       = actual_back_sec,
        actual_forward_sec    = actual_forward_sec,
        rel_cues              = rel_cues,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Per-patient embedding array building
# ─────────────────────────────────────────────────────────────────────────────

def build_patient_embeddings(pdata, shared, embedding_names=None):
    """Look up patient-specific embedding arrays from the shared models.

    Text embeddings (GloVe, FastText, Word2Vec, ConceptNet) are always built.
    Image embeddings (DINOv2, DINOv2Small, DINOv3, MoCo, SimCLR) are built for
    picture_naming, and for auditory_naming ONLY when one of them is requested via
    ``embedding_names`` (the default auditory list is text-only, so a plain auditory
    run skips the heavy image loads).  For auditory naming an image embedding maps the
    answered word to its picture-stimulus vector via ``_map_to_target``.

    Parameters
    ----------
    embedding_names : list[str] or None
        The embeddings this run will actually fit.  Defaults to the global
        EMBEDDING_NAMES.  Used only to decide whether to load image embeddings.
    """
    _step('Building embedding arrays for this patient …')
    lemma  = pdata['target_lemma']
    # Vision lookups use the homonym-sense-resolved label (bare word for non-homonyms,
    # 'bat1'/'bat2' etc. for homonyms); text embeddings use `lemma` and are unaffected.
    labels = pdata.get('vision_label', pdata['clean_target_labels'])

    embed = {}

    embed['GloVe'] = np.array([shared['glove'][w].numpy() for w in lemma])
    _ok(f'GloVe:  {embed["GloVe"].shape}')

    embed['FastText'] = np.array([shared['fasttext'][w].numpy() for w in lemma])
    _ok(f'FastText: {embed["FastText"].shape}')

    embed['Word2Vec'] = np.array([
        shared['word2vec'][w] if w in shared['word2vec'] else np.zeros(300)
        for w in lemma
    ])
    _ok(f'Word2Vec: {embed["Word2Vec"].shape}')

    embed['ConceptNet'] = np.array([
        shared['conceptnet'].get(w, np.zeros(300, dtype=np.float32))
        for w in lemma
    ], dtype=np.float32)
    n_found_cn = sum(1 for w in lemma if w in shared['conceptnet'])
    _ok(f'ConceptNet: {embed["ConceptNet"].shape}  '
        f'({n_found_cn}/{len(lemma)} words found)')

    # Image embeddings – built for picture_naming, and for auditory_naming only when a
    # vision embedding is explicitly requested (default auditory list is text-only, so a
    # plain auditory run skips the heavy image loads).
    _active = embedding_names if embedding_names is not None else EMBEDDING_NAMES
    _want_vision = any(n in VISION_EMBEDDING_NAMES for n in _active)
    if TASK == 'auditory_naming' and not _want_vision:
        _step('Skipping image embeddings (none requested; default auditory list is text-only)')
        return embed

    # DINOv2 / SimCLR – try patient-specific folder first, then all other
    # available embedding folders as fallbacks so that no label is lost.
    embed_folders = _visual_embed_folders(pdata['patient'])
    _step(f'Visual embeddings folders (priority order): '
          f'{[os.path.basename(f) for f in embed_folders]}')

    # DINOv2 pooled
    _dinov2_sources = []
    for _folder in embed_folders:
        _fpath = os.path.join(_folder, 'dinov2_layerwise_embeddings.pk')
        if os.path.exists(_fpath):
            with open(_fpath, 'rb') as _f:
                _d = pk.load(_f)
            _dinov2_sources.append((_d, np.array(_d['words'])))
    embed['DINOv2'] = _map_to_target(_dinov2_sources, 'dinov2_pooled', labels)
    _ok(f'DINOv2: {embed["DINOv2"].shape}')

    # SimCLR pooled
    _simclr_sources = []
    for _folder in embed_folders:
        _fpath = os.path.join(_folder, 'simclr_layerwise_embeddings.pk')
        if os.path.exists(_fpath):
            with open(_fpath, 'rb') as _f:
                _d = pk.load(_f)
            _simclr_sources.append((_d, np.array(_d['words'])))
    embed['SimCLR'] = _map_to_target(_simclr_sources, 'simclr_pooled', labels)
    _ok(f'SimCLR: {embed["SimCLR"].shape}')

    # DINOv2-Small pooled  [384-dim]
    _dinov2_small_sources = []
    for _folder in embed_folders:
        _fpath = os.path.join(_folder, 'dinov2_small_layerwise_embeddings.pk')
        if os.path.exists(_fpath):
            with open(_fpath, 'rb') as _f:
                _d = pk.load(_f)
            _dinov2_small_sources.append((_d, np.array(_d['words'])))
    embed['DINOv2Small'] = _map_to_target(_dinov2_small_sources, 'dinov2_small_pooled', labels)
    _ok(f'DINOv2Small: {embed["DINOv2Small"].shape}')

    # DINOv3 pooled  [384-dim]
    _dinov3_sources = []
    for _folder in embed_folders:
        _fpath = os.path.join(_folder, 'dinov3_layerwise_embeddings.pk')
        if os.path.exists(_fpath):
            with open(_fpath, 'rb') as _f:
                _d = pk.load(_f)
            _dinov3_sources.append((_d, np.array(_d['words'])))
    embed['DINOv3'] = _map_to_target(_dinov3_sources, 'dinov3_pooled', labels)
    _ok(f'DINOv3: {embed["DINOv3"].shape}')

    # MoCo (SSL ResNet-18) pooled  [512-dim]
    _moco_sources = []
    for _folder in embed_folders:
        _fpath = os.path.join(_folder, 'moco_ssl_resnet18_layerwise_embeddings.pk')
        if os.path.exists(_fpath):
            with open(_fpath, 'rb') as _f:
                _d = pk.load(_f)
            _moco_sources.append((_d, np.array(_d['words'])))
    embed['MoCo'] = _map_to_target(_moco_sources, 'moco_pooled', labels)
    _ok(f'MoCo: {embed["MoCo"].shape}')

    return embed


# ─────────────────────────────────────────────────────────────────────────────
#  Regression
# ─────────────────────────────────────────────────────────────────────────────

def _make_regressor_pipeline(mode='krr'):
    """
    Build the regression pipeline.

    Parameters
    ----------
    mode : str
        One of:
          'krr'          — Nystroem(RBF) + Ridge (current default, nonlinear + regularized)
          'linear_ridge' — Ridge only (linear + regularized)
          'pls'          — PLSRegression (linear, implicit regularization via n_components)
          'kernel_pls'   — Nystroem(RBF) + PLSRegression (nonlinear + implicit regularization)

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    if mode == 'krr':
        return Pipeline([
            ('nystroem', Nystroem(kernel='rbf')),
            ('ridge',    Ridge(alpha=KRR_ALPHA)),
        ])
    elif mode == 'linear_ridge':
        return Pipeline([
            ('ridge', Ridge(alpha=KRR_ALPHA)),
        ])
    elif mode == 'pls':
        return Pipeline([
            ('pls', PLSRegression(n_components=PLS_COMPONENTS, scale=False)),
        ])
    elif mode == 'kernel_pls':
        return Pipeline([
            ('nystroem', Nystroem(kernel='rbf')),
            ('pls',      PLSRegression(n_components=PLS_COMPONENTS, scale=False)),
        ])
    else:
        raise ValueError(f"Unknown model mode: {mode!r}. "
                         f"Choose from: krr, linear_ridge, pls, kernel_pls")


def run_regressions(pdata, embeddings, n_epochs, closest='l2', model_mode='krr',
                    embedding_names=None):
    """Fit one BasicRegressor per embedding type; return dict name→regressor.

    Parameters
    ----------
    embedding_names : list[str] or None
        Which embeddings to run.  Defaults to the global EMBEDDING_NAMES.
        Pass AUDITORY_EMBEDDING_NAMES (text-only) for auditory_naming.
    """
    X               = pdata['clean_data_binned'].swapaxes(1, 2)
    labels          = pdata['target_concept']
    category_labels = pdata['clean_word_category']
    regressors      = {}
    active_names    = embedding_names if embedding_names is not None else EMBEDDING_NAMES
    n_total         = len(active_names)

    for idx, emb_name in enumerate(active_names, start=1):
        _step(f'[{idx}/{n_total}]  {emb_name} regression  (epochs={n_epochs}, '
              f'parallel={PARALLEL_WORKERS}, closest={closest}) …')
        # PLS handles dimensionality reduction internally — skip PCA
        use_pca = model_mode not in ('pls', 'kernel_pls')
        br = BasicRegressor(
            _make_regressor_pipeline(mode=model_mode),
            y_reducer=PCA(Y_PCA_COMPONENTS) if use_pca else None,
        )
        br.load_data(
            X, embeddings[emb_name],
            n_bins_history=N_BINS_HISTORY,
            labels=labels,
            category_labels=category_labels,
        )
        br.fit(
            n_epochs=n_epochs,
            parallel=PARALLEL_WORKERS,
            closest=closest,
            compute_retrieval=True,
            save_retrieval_pairs=True,
            compute_top_k_accuracy=False,
        )
        regressors[emb_name] = br
        best = int(np.nanargmax(np.nanmean(br.all_retrieval_top1, axis=0)))
        top1_at_best = float(np.nanmean(br.all_retrieval_top1, axis=0)[best])
        _ok(f'{emb_name} done  |  best bin={best}  |  top-1 word acc={top1_at_best:.3f}')
        gc.collect()

    return regressors


# ─────────────────────────────────────────────────────────────────────────────
#  Confusion-matrix helpers  (adapted from notebook cell #VSC-1e7c1c97)
# ─────────────────────────────────────────────────────────────────────────────







# ─────────────────────────────────────────────────────────────────────────────
#  Per-word count vs metric plots  (adapted from notebook cell #VSC-d7531eb5)
# ─────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────
#  Save figures
# ─────────────────────────────────────────────────────────────────────────────

def save_figures(patient, pdata, regressors, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    _section(f'Saving figures  →  {fig_dir}')

    active_names = [n for n in EMBEDDING_NAMES if n in regressors]
    model_map    = {name: regressors[name] for name in active_names}
    adj_fs       = pdata['adjusted_fs']
    v_on         = pdata['clean_voice_onset']
    v_off        = pdata['clean_voice_offset']
    n_bins       = pdata['clean_data_binned'].shape[2]

    align_cue = pdata.get('align_cue', 'none')
    if align_cue != 'none' and pdata.get('actual_back_sec') is not None:
        # ── Cue-aligned: use stored window and rel_cues for line positions ────
        back    = pdata['actual_back_sec']
        forward = pdata['actual_forward_sec']
        rel     = pdata.get('rel_cues') or {}
        # Fixed display order for all possible vertical cue lines
        _cue_display_order = [
            ('trial_onset',     'trial onset'),
            ('go_cue',          'go cue'),
            ('aud_stim_onset',  'aud stim on'),
            ('aud_stim_offset', 'aud stim off'),
            ('voice_onset',     'voice on'),
            ('voice_offset',    'voice off'),
        ]
        common_lines = []
        line_labels  = []
        for cue_key, cue_label in _cue_display_order:
            stats = rel.get(cue_key)
            if stats is None:
                continue
            m = stats['mean']
            if not np.isfinite(m):
                continue
            common_lines.append(m)
            line_labels.append(f'{cue_label} (ref)' if cue_key == align_cue else cue_label)
    elif TASK == 'auditory_naming':
        # ── Legacy: auditory_naming without explicit alignment ─────────────────
        ref          = pdata['clean_aud_stim_onset']
        t_onset_arr  = pdata['clean_trial_onset']
        aud_off_arr  = pdata['clean_aud_stim_offset']
        ref_mean     = float(np.nanmean(ref))
        if not np.isfinite(ref_mean):
            _warn('clean_aud_stim_onset is NaN; falling back to trial_onset alignment')
            ref_mean = float(np.nanmean(pdata.get('clean_trial_onset', np.array([0.0]))))
        back         = ref_mean
        forward      = float(n_bins / adj_fs) - back
        if not (np.isfinite(back) and np.isfinite(forward) and forward > 0):
            back    = float(n_bins / adj_fs) / 2
            forward = float(n_bins / adj_fs) / 2
        common_lines = [
            float(np.nanmean(t_onset_arr) - ref_mean),
            0.0,
            float(np.nanmean(aud_off_arr) - ref_mean),
            float(np.nanmean(v_on)        - ref_mean),
            float(np.nanmean(v_off)       - ref_mean),
        ]
        line_labels = ['trial onset', 'aud stim on', 'aud stim off',
                       'voice on', 'voice off']
    else:
        # ── Legacy: picture_naming without alignment ───────────────────────────
        t_onset = pdata['trial_onset']
        go_cue  = pdata['go_cue_onset']
        back    = float(np.nanmean(t_onset))
        forward = float(n_bins / adj_fs - np.nanmean(t_onset))
        common_lines = [
            0 - np.nanmean(t_onset),
            float(np.nanmean(go_cue) - np.nanmean(t_onset)),
            float(np.nanmean(v_on)   - np.nanmean(t_onset)),
            float(np.nanmean(v_off)  - np.nanmean(t_onset)),
        ]
        line_labels = ['trial onset', 'go cue', 'voice on', 'voice off']

    data_labels  = active_names + ['chance']
    zero_stds    = [0] * len(active_names)
    br0          = regressors[active_names[0]]

    # common plotly kwargs
    plotly_kw = dict(
        lines       = common_lines,
        line_labels = line_labels,
        data_labels = data_labels,
        back        = back,
        forward     = forward,
        tick_interval = 1,
    )

    # ── 1.  R² over time ──────────────────────────────────────────────────────
    _step('R² over time …')
    fig_r2, _ = plot_accuracy_plotly(
        *[regressors[n].all_test_score.mean(0) for n in active_names],
        br0.all_chance.mean(0),
        data_std = zero_stds + [br0.all_chance.std(0)],
        ylabel   = 'R²',
        title    = f'{patient}: R² over Time (Trial-Onset Aligned)',
        **plotly_kw,
    )
    fig_r2.write_html(os.path.join(fig_dir, 'r2_over_time.html'))
    _ok('r2_over_time.html')

    # ── 2.  Word retrieval balanced accuracy ──────────────────────────────────
    _step('Word retrieval balanced accuracy …')
    fig_wb, _ = plot_accuracy_plotly(
        *[np.mean(regressors[n].all_retrieval_word_balanced_acc, axis=0)
          for n in active_names],
        np.mean(br0.all_retrieval_chance_word_balanced_acc, axis=0),
        data_std = zero_stds + [np.std(br0.all_retrieval_chance_word_balanced_acc, axis=0)],
        ylabel   = 'Balanced Accuracy',
        title    = f'{patient}: Word Retrieval Balanced Accuracy',
        **plotly_kw,
    )
    fig_wb.write_html(os.path.join(fig_dir, 'word_retrieval_balanced_acc.html'))
    _ok('word_retrieval_balanced_acc.html')

    # ── 3.  Category retrieval balanced accuracy ──────────────────────────────
    _step('Category retrieval balanced accuracy …')
    fig_cb, _ = plot_accuracy_plotly(
        *[np.mean(regressors[n].all_retrieval_category_balanced_acc, axis=0)
          for n in active_names],
        np.mean(br0.all_retrieval_category_chance_balanced_acc, axis=0),
        data_std = zero_stds + [
            np.std(br0.all_retrieval_category_chance_balanced_acc, axis=0)
        ],
        ylabel   = 'Balanced Accuracy',
        title    = f'{patient}: Category Retrieval Balanced Accuracy',
        **plotly_kw,
    )
    fig_cb.write_html(os.path.join(fig_dir, 'category_retrieval_balanced_acc.html'))
    _ok('category_retrieval_balanced_acc.html')

    # ── 4.  Confusion matrices – word (top-10 by F1) ──────────────────────────
    _step('Confusion matrix (word, top-10 by F1) …')
    fig_cw = _plot_cm_grid(model_map, mode='word', normalize=True,
                           cmap='viridis', top_k_words_by_f1=10)
    fig_cw.savefig(os.path.join(fig_dir, 'confusion_word.png'),
                   dpi=150, bbox_inches='tight')
    plt.close(fig_cw)
    _ok('confusion_word.png')

    # ── 5.  Confusion matrices – category ────────────────────────────────────
    _step('Confusion matrix (category) …')
    fig_cc = _plot_cm_grid(model_map, mode='category', normalize=True, cmap='viridis')
    fig_cc.savefig(os.path.join(fig_dir, 'confusion_category.png'),
                   dpi=150, bbox_inches='tight')
    plt.close(fig_cc)
    _ok('confusion_category.png')

    # ── 6.  Per-word count vs accuracy ───────────────────────────────────────
    _step('Per-word count vs. accuracy …')
    fig_ca = _plot_count_vs_metric(model_map, metric='accuracy')
    fig_ca.savefig(os.path.join(fig_dir, 'count_vs_accuracy.png'),
                   dpi=150, bbox_inches='tight')
    plt.close(fig_ca)
    _ok('count_vs_accuracy.png')

    # ── 7.  Per-word count vs F1 ──────────────────────────────────────────────
    _step('Per-word count vs. F1 …')
    fig_cf = _plot_count_vs_metric(model_map, metric='f1')
    fig_cf.savefig(os.path.join(fig_dir, 'count_vs_f1.png'),
                   dpi=150, bbox_inches='tight')
    plt.close(fig_cf)
    _ok('count_vs_f1.png')


# ─────────────────────────────────────────────────────────────────────────────
#  Save source data
# ─────────────────────────────────────────────────────────────────────────────

def save_source_data(patient, pdata, regressors, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    _section(f'Saving source data  →  {results_dir}')

    # ── 1.  Full regressor objects (pk) ───────────────────────────────────────
    _step('semantic_regression_results.pkl …')
    reg_path = os.path.join(results_dir, 'semantic_regression_results.pkl')
    with open(reg_path, 'wb') as f:
        pk.dump({
            'patient':              patient,
            'embedding_names':      EMBEDDING_NAMES,
            'regressors':           regressors,
            'target_concept':       pdata['target_concept'],
            'clean_word_category':  pdata['clean_word_category'],
            'clean_target_labels':  pdata['clean_target_labels'],
            'clean_answer_labels':  pdata['clean_answer_labels'],
            'clean_channel_names':  pdata['clean_channel_names'],
            'bin_size_ms':          BIN_SIZE,
            'n_bins_history':       N_BINS_HISTORY,
            'actual_back_sec':      pdata.get('actual_back_sec'),
            'actual_forward_sec':   pdata.get('actual_forward_sec'),
            'rel_cues':             pdata.get('rel_cues'),
            'rel_cues_reference':   pdata.get('rel_cues_reference'),
        }, f, protocol=4)
    _ok(f'semantic_regression_results.pkl  ({os.path.getsize(reg_path) / 1e6:.1f} MB)')

    # ── 1b. Cue statistics JSON (standalone, easy to load for plotting) ────────
    _step('cue_stats.json …')
    cue_stats = {
        'patient':            patient,
        'align_cue':          pdata.get('align_cue', 'none'),
        'rel_cues_reference': pdata.get('rel_cues_reference'),
        'actual_back_sec':    pdata.get('actual_back_sec'),
        'actual_forward_sec': pdata.get('actual_forward_sec'),
        'rel_cues':           pdata.get('rel_cues'),
    }
    cue_stats_path = os.path.join(results_dir, 'cue_stats.json')
    with open(cue_stats_path, 'w', encoding='utf-8') as f:
        json.dump(cue_stats, f, indent=2, ensure_ascii=False, default=str)
    _ok(f'cue_stats.json')

    # ── 2.  Top-1 decoding source data CSV  (all test pairs, ALL time bins) ───
    # This captures true/predicted word+category for every test trial,
    # every epoch, and every time bin.  A flag marks the best-bin rows.
    _step('top1_decoding_source_data.csv  (all retrieval pairs across time) …')
    rows = []
    for emb_name, br in regressors.items():
        best_bin_word = _best_bin_from_top1(br, mode='word')
        best_bin_cat  = _best_bin_from_top1(br, mode='category')

        for epoch_idx, rec in enumerate(br.all_retrieval_pairs):
            bin_idx  = int(rec['bin_index'])
            true_wi  = np.asarray(rec['true_word_idx'], dtype=np.int64)
            pred_wi  = np.asarray(rec['pred_word_idx'], dtype=np.int64)
            pred_ci_indep = np.asarray(rec['pred_category_idx_indep'], dtype=np.int64) if 'pred_category_idx_indep' in rec else None
            for t, (tw, pw) in enumerate(zip(true_wi, pred_wi)):
                true_word = br.index_to_word[tw]
                pred_word = br.index_to_word[pw]
                if br.word_index_to_category_index is not None:
                    true_cat = br.index_to_category[br.word_index_to_category_index[tw]]
                    pred_cat = br.index_to_category[br.word_index_to_category_index[pw]]
                else:
                    true_cat = pred_cat = 'N/A'
                if pred_ci_indep is not None and br.word_index_to_category_index is not None:
                    pred_cat_indep = br.index_to_category[pred_ci_indep[t]]
                    true_cat_indep = br.index_to_category[br.word_index_to_category_index[tw]]
                    cat_correct_indep = pred_cat_indep == true_cat_indep
                else:
                    pred_cat_indep = 'N/A'
                    cat_correct_indep = 'N/A'
                word_loose = _is_loose_match(true_word, pred_word)
                # Loose category: word is loose-correct (implies correct category)
                # OR the independent category predictor matched.
                # cat_correct_indep is True/False/"N/A"; treat N/A as False.
                cat_loose  = word_loose or (cat_correct_indep is True)
                rows.append({
                    'patient':           patient,
                    'embedding':         emb_name,
                    'epoch':             epoch_idx,
                    'bin_index':         bin_idx,
                    'is_best_word_bin':  bin_idx == best_bin_word,
                    'is_best_cat_bin':   bin_idx == best_bin_cat,
                    'true_word':         true_word,
                    'pred_word':         pred_word,
                    'true_category':     true_cat,
                    'pred_category':     pred_cat,
                    'pred_category_indep': pred_cat_indep,
                    'word_correct':           true_word == pred_word,
                    'category_correct':       true_cat  == pred_cat,
                    'category_correct_indep': cat_correct_indep,
                    'word_correct_loose':     word_loose,
                    'category_correct_loose': cat_loose,
                })

    df_pairs = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, 'top1_decoding_source_data.csv')
    df_pairs.to_csv(csv_path, index=False)
    _ok(f'top1_decoding_source_data.csv  '
        f'({len(df_pairs):,} rows, '
        f'{df_pairs["bin_index"].nunique()} bins, '
        f'{df_pairs["embedding"].nunique()} embeddings)')

    # Pre-compute per-bin loose accuracy for merging into the summary CSV.
    if len(df_pairs) > 0:
        _loose_by_bin = (
            df_pairs
            .groupby(['embedding', 'bin_index'])[
                ['word_correct_loose', 'category_correct_loose']
            ]
            .mean()
            .rename(columns={
                'word_correct_loose':     'word_loose_acc',
                'category_correct_loose': 'category_loose_acc',
            })
            .reset_index()
        )
    else:
        _loose_by_bin = pd.DataFrame(
            columns=['embedding', 'bin_index', 'word_loose_acc', 'category_loose_acc']
        )

    # ── 3.  Per-time-bin summary scores CSV ──────────────────────────────────
    _step('per_time_scores.csv …')
    score_rows = []
    for emb_name, br in regressors.items():
        n_bins = br.all_test_score.shape[1]
        r2_mean    = br.all_test_score.mean(0)
        r2_std     = br.all_test_score.std(0)
        chance_mean = br.all_chance.mean(0)
        wbal_mean  = np.mean(br.all_retrieval_word_balanced_acc,     axis=0)
        cbal_mean  = np.mean(br.all_retrieval_category_balanced_acc, axis=0)
        cbal_indep_mean = np.mean(br.all_retrieval_category_indep_balanced_acc, axis=0) if hasattr(br, 'all_retrieval_category_indep_balanced_acc') and br.all_retrieval_category_indep_balanced_acc.size > 0 else np.full(n_bins, np.nan)
        wf1_mean   = np.mean(br.all_retrieval_word_f1,               axis=0)
        cf1_mean   = np.mean(br.all_retrieval_category_f1,           axis=0)
        top3_mean  = np.mean(br.all_retrieval_top3,  axis=0)
        top5_mean  = np.mean(br.all_retrieval_top5,  axis=0)
        # Cosine similarity (direction-only fit metric, always available)
        cos_mean   = br.all_cosine_sim.mean(0)   if hasattr(br, 'all_cosine_sim')   and br.all_cosine_sim.size   > 0 else np.zeros(n_bins)
        cos_std    = br.all_cosine_sim.std(0)     if hasattr(br, 'all_cosine_sim')   and br.all_cosine_sim.size   > 0 else np.zeros(n_bins)
        for b in range(n_bins):
            score_rows.append({
                'patient':              patient,
                'embedding':            emb_name,
                'bin_index':            b,
                'r2_mean':              r2_mean[b],
                'r2_std':               r2_std[b],
                'cosine_mean':          cos_mean[b],
                'cosine_std':           cos_std[b],
                'chance_mean':          chance_mean[b],
                'word_balanced_acc':    wbal_mean[b],
                'category_balanced_acc': cbal_mean[b],
                'category_balanced_acc_indep': cbal_indep_mean[b],
                'word_f1':              wf1_mean[b],
                'category_f1':          cf1_mean[b],
                'word_top3_acc':        top3_mean[b],
                'word_top5_acc':        top5_mean[b],
            })
    df_scores  = pd.DataFrame(score_rows)
    df_scores  = df_scores.merge(_loose_by_bin, on=['embedding', 'bin_index'], how='left')
    scores_path = os.path.join(results_dir, 'per_time_scores.csv')
    df_scores.to_csv(scores_path, index=False)
    _ok(f'per_time_scores.csv  ({len(df_scores):,} rows)')


# ─────────────────────────────────────────────────────────────────────────────
#  Patient discovery
# ─────────────────────────────────────────────────────────────────────────────

def check_auditory_naming_availability():
    """Print a table of auditory naming data availability across all patients."""
    _section('Auditory naming data availability check')
    if not os.path.isdir(DATA_FOLDER):
        _warn(f'DATA_FOLDER "{DATA_FOLDER}" not found')
        return
    rows = []
    for name in sorted(os.listdir(DATA_FOLDER)):
        folder = os.path.join(DATA_FOLDER, name)
        if not os.path.isdir(folder):
            continue
        df_path  = _find_df_path(folder, name, 'auditory_naming')
        lbl_path = os.path.join(folder, f'{name}_auditory_naming_labels.pkl')
        ch_candidates = [
            os.path.join(folder, f'{name}_auditory_naming_channels.pkl'),
            os.path.join(folder, f'{name}_channels.pkl'),
            os.path.join(folder, f'{name}_picture_naming_channels.pkl'),
        ]
        ch_found = next((p for p in ch_candidates if os.path.exists(p)), None)
        has_df  = df_path is not None
        has_lbl = os.path.exists(lbl_path)
        # Check for stimulus timing columns
        stim_col_info = 'N/A'
        if has_df:
            try:
                df_tmp = load_pkl(df_path)
                if isinstance(df_tmp, dict):
                    df_tmp = pd.DataFrame(df_tmp)
                stim_cols = [c for c in df_tmp.columns
                             if 'stim' in c.lower() or 'stimulus' in c.lower()]
                stim_col_info = ', '.join(stim_cols[:5]) if stim_cols else 'none found'
                del df_tmp
            except Exception as e:
                stim_col_info = f'ERROR: {e}'
        rows.append((name, has_df, has_lbl, ch_found, stim_col_info))

    # Only show patients that have at least one auditory naming file
    rows = [r for r in rows if r[1] or r[2]]
    if not rows:
        _warn('No auditory naming data found under DATA_FOLDER.')
        return
    print(f'\n  {"Patient":8}  {"df":4}  {"labels":7}  '
          f'{"channels_file":38}  stim_cols')
    print('  ' + '-' * 90)
    for name, hd, hl, ch, sc in rows:
        ch_name = os.path.basename(ch) if ch else 'none'
        print(f'  {name:8}  {"✓" if hd else "✗":4}  {"✓" if hl else "✗":7}  '
              f'{ch_name:38}  {sc}')
    print()



# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────



def _build_meta(args, patients, run_id, log_path, warp_patient_medians=None):
    """Build a metadata dict that captures everything needed to reproduce a run."""
    import sklearn
    import torch

    _warping = AUDITORY_WARP != 'none'

    return {
        # ── Run identification ────────────────────────────────────────────
        'run_id':               run_id,
        'timestamp_utc':        datetime.utcnow().isoformat() + 'Z',
        'timestamp_local':      datetime.now().isoformat(),
        'command_line':         sys.argv,
        'script_path':          os.path.abspath(__file__),
        'log_path':             log_path,

        # ── Version control ───────────────────────────────────────────────
        'git_commit':           _git_hash(),
        'git_dirty':            _git_dirty(),

        # ── Task & data ───────────────────────────────────────────────────
        'task':                 TASK,
        'align_cue':            ALIGN_CUE,
        'align_back_sec':       ALIGN_BACK,
        'align_forward_sec':    ALIGN_FORWARD,
        # Warp mode: 'none' | 'stim' ([stim_onset → stim_offset]) | 'voice'
        # ([stim_onset → voice_onset]).  Applies to both tasks (key name kept as
        # 'auditory_warp' for backward compatibility with report readers).
        'auditory_warp':        AUDITORY_WARP,
        # What the warped segment was stretched TO. 'group' = one duration for every
        # patient (the pooled median over auditory_warp_target_patients), so the seg-end
        # event lands at the same time in all of them; 'patient' = each patient's own
        # median, which leaves the event differing between patients.
        'auditory_warp_scope':  AUDITORY_WARP_SCOPE if _warping else 'N/A',
        'auditory_warp_target_sec':      AUDITORY_WARP_TARGET_SEC if _warping else None,
        # 'computed' = the pooled median over auditory_warp_target_patients below;
        # 'pinned'   = supplied via --warp-target-sec, i.e. it came from somewhere else.
        'auditory_warp_target_source':   AUDITORY_WARP_TARGET_SOURCE if _warping else None,
        # WHO DEFINED THE TARGET — deliberately not "who was run under it". Under a pinned
        # target this run's patients did not set it, so recording them here would assert a
        # provenance that is false; None means "look at auditory_warp_target_source".
        'auditory_warp_target_patients': patients if (_warping and AUDITORY_WARP_SCOPE == 'group'
                                                      and AUDITORY_WARP_TARGET_SOURCE
                                                      == 'computed') else None,
        'auditory_warp_patient_medians': warp_patient_medians or None,
        'data_folder':          os.path.abspath(DATA_FOLDER),
        'patients':             patients,

        # ── Hyperparameters ───────────────────────────────────────────────
        'n_epochs':             args.epochs,
        'bin_size_ms':          BIN_SIZE,
        'n_bins_history':       N_BINS_HISTORY,
        'y_pca_components':     Y_PCA_COMPONENTS,
        'krr_alpha':            KRR_ALPHA,
        'parallel_workers':     PARALLEL_WORKERS,

        # ── Retrieval / similarity ────────────────────────────────────────
        'closest':              args.closest,
        'model_mode':           args.model,

        # ── Embeddings ────────────────────────────────────────────────────
        'embedding_names':      EMBEDDING_NAMES,
        'embeddings_folder':    os.path.abspath(EMBEDDINGS_FOLDER),

        # ── Model / pipeline ──────────────────────────────────────────────
        'regressor_pipeline':   f'{args.model}: ' + {
            'krr':          f'Nystroem(rbf) → Ridge(α={KRR_ALPHA})',
            'linear_ridge': f'Ridge(α={KRR_ALPHA})',
            'pls':          f'PLSRegression(n={PLS_COMPONENTS})',
            'kernel_pls':   f'Nystroem(rbf) → PLSRegression(n={PLS_COMPONENTS})',
        }.get(args.model, '?'),
        'y_reducer':            'PCA(n_components={})'.format(Y_PCA_COMPONENTS),
        'split_strategy':       'random train_test_split',
        'split_fraction':       0.3,

        # ── Environment ───────────────────────────────────────────────────
        'python_version':       platform.python_version(),
        'platform':             platform.platform(),
        'numpy_version':        np.__version__,
        'pandas_version':       pd.__version__,
        'sklearn_version':      sklearn.__version__,
        'torch_version':        torch.__version__,
    }



def main():
    global EMBEDDING_NAMES, BIN_SIZE, N_BINS_HISTORY, TASK, AUDITORY_WARP, ALIGN_CUE, ALIGN_BACK, ALIGN_FORWARD
    global AUDITORY_WARP_SCOPE, AUDITORY_WARP_TARGET_SEC, AUDITORY_WARP_TARGET_SOURCE
    parser = argparse.ArgumentParser(
        description='Batch semantic regression: neural activity → word embeddings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--patients', nargs='*', default=None,
        help='Patient IDs to process (omit to auto-discover all)',
    )
    parser.add_argument(
        '--epochs', type=int, default=N_EPOCHS,
        help='Number of regression epochs per embedding',
    )
    parser.add_argument(
        '--closest', choices=['l2', 'cosine'], default='cosine',
        help='Retrieval similarity metric (l2 = Euclidean, cosine = cosine similarity)',
    )
    parser.add_argument(
        '--model', choices=['krr', 'linear_ridge', 'pls', 'kernel_pls'],
        default='kernel_pls',
        help='Regression model: krr (Nystroem+Ridge, default), linear_ridge, '
             'pls (Partial Least Squares), kernel_pls (Nystroem+PLS)',
    )
    parser.add_argument(
        '--embedding', nargs='+', default=None,
        metavar='EMB',
        help='Embeddings to run (default: all). '
             'Choices: GloVe FastText Word2Vec ConceptNet DINOv2 DINOv2Small DINOv3 MoCo SimCLR',
    )
    parser.add_argument(
        '--bin-size', type=int, default=BIN_SIZE,
        help='Bin size in ms  (default: 100)',
    )
    parser.add_argument(
        '--history-bins', type=int, default=N_BINS_HISTORY,
        dest='history_bins',
        help='Number of preceding time bins fed to the model as history '
             '(feature lag).',
    )
    parser.add_argument(
        '--task',
        choices=['picture_naming', 'auditory_naming'],
        default='picture_naming',
        help='Task type to process. Use "auditory_naming" for the auditory '
             'naming paradigm (text-only embeddings, answered-word labels).',
    )
    parser.add_argument(
        '--warp',
        choices=['none', 'stim', 'voice', 'linear'],
        default='none',
        dest='warp',
        help='Time-warping mode (applies to both picture_naming and auditory_naming). '
             '"stim" linearly warps the stimulus segment [stim_onset -> stim_offset] to a '
             'common duration (picture: trial_onset -> go_cue_onset; auditory: '
             'aud_stim_onset -> aud_stim_offset). "voice" warps the pre-speech segment '
             '[stim_onset -> voice_onset] instead, so voice onset lands at a common time. '
             'See --warp-scope for what the common duration is. '
             '"linear" is a DEPRECATED ALIAS kept so archived command lines reproduce: '
             'before commit 1aca186 the only warp mode was --warp linear, which warped '
             '[aud_stim_onset, aud_stim_offset] to THAT PATIENT\'S own median stimulus '
             'duration. It is rewritten to "--warp stim --warp-scope patient", which is '
             'exactly equivalent. Do not use it in new runs.',
    )
    parser.add_argument(
        '--warp-scope',
        choices=['group', 'patient'],
        default='group',
        dest='warp_scope',
        help='Which median segment duration "--warp" stretches to. '
             '"group" (default) uses the median over the pooled trials of ALL patients '
             'in the run, so the seg-end event (stim offset / voice onset) lands at the '
             'same time in every patient (required for group-level figures). "patient" '
             'uses each patient\'s own median, which leaves the event differing between '
             'patients. No effect unless --warp is stim or voice.',
    )
    parser.add_argument(
        '--warp-target-sec',
        type=float,
        default=None,
        dest='warp_target_sec',
        help='Warp every trial to THIS duration (seconds) instead of computing the target '
             'from the run\'s own patients. Only meaningful with --warp-scope group. '
             'Use it to add a patient to an existing group-warped cohort without re-warping '
             'the patients already in it: --warp-scope group makes the target the pooled '
             'median over the run\'s patients, so a new patient shifts it and silently '
             'changes everyone else. Pinning the existing target removes that coupling — '
             'the new patient depends on the constant, nobody depends on the new patient. '
             'Read the pin off the existing run\'s meta.json (auditory_warp_target_sec); '
             'it is recorded there as auditory_warp_target_source=pinned.',
    )
    parser.add_argument(
        '--align',
        choices=['none', 'trial_onset', 'go_cue', 'voice_onset',
                 'voice_offset', 'aud_stim_onset', 'aud_stim_offset'],
        default='none',
        dest='align',
        help='Behavioral cue to align each trial around before binning. '
             '"none" keeps the raw trial-onset-anchored timeline (default). '
             '"voice_onset", "go_cue", etc. slice a fixed window around that event.',
    )
    parser.add_argument(
        '--align-back', type=float, default=None,
        dest='align_back',
        help='Seconds before the alignment cue to include. '
             'Omit (or pass nothing) to use the full available window '
             '(shortest back-distance across trials).  (default: full window)',
    )
    parser.add_argument(
        '--align-forward', type=float, default=None,
        dest='align_forward',
        help='Seconds after the alignment cue to include. '
             'Omit (or pass nothing) to use the full available window '
             '(shortest forward-distance across trials).  (default: full window)',
    )
    args = parser.parse_args()

    # ── Deprecated --warp linear -> --warp stim --warp-scope patient ─────────
    # Runs archived before commit 1aca186 record "--warp linear" in their
    # meta.json command_line (e.g. the auditory run used by cross_task_*). That
    # flag was later generalized to {none, stim, voice} + --warp-scope, so those
    # command lines stopped parsing. Old "linear" warped the auditory stimulus
    # segment to each PATIENT'S own median duration == stim + scope patient, so
    # the rewrite is exact rather than approximate.
    if args.warp == 'linear':
        explicit_scope = any(a == '--warp-scope' or a.startswith('--warp-scope=')
                             for a in sys.argv[1:])
        args.warp = 'stim'
        if not explicit_scope:
            args.warp_scope = 'patient'
        print("[deprecation] '--warp linear' -> '--warp stim --warp-scope "
              f"{args.warp_scope}' (exact equivalent of the pre-1aca186 behaviour).")

    # Always run relative to this script's directory (main/)
    os.chdir(_SCRIPT_DIR)

    # ── Override global constants from CLI ──────────────────────────────────
    TASK          = args.task
    AUDITORY_WARP = args.warp
    AUDITORY_WARP_SCOPE = args.warp_scope
    ALIGN_CUE     = args.align
    ALIGN_BACK    = args.align_back
    ALIGN_FORWARD = args.align_forward
    if TASK == 'auditory_naming':
        # Default to text-only embeddings; CLI --embedding can still override
        EMBEDDING_NAMES = AUDITORY_EMBEDDING_NAMES
    if args.embedding is not None:
        EMBEDDING_NAMES = args.embedding
    BIN_SIZE = args.bin_size
    N_BINS_HISTORY = args.history_bins

    # ── Unique run identifier ─────────────────────────────────────────────────
    timestamp   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # Warp now applies to both tasks; scope distinguishes group vs patient targets.
    warp_part   = f'_warp-{args.warp}-{args.warp_scope}' if args.warp != 'none' else ''
    align_part  = f'_align-{ALIGN_CUE}' if ALIGN_CUE != 'none' else ''
    run_id      = f'{timestamp}_{TASK}{warp_part}{align_part}_{args.model}_{args.closest}_{args.epochs}ep'

    # ── Set up log file (tee stdout → terminal + file) ────────────────────────
    log_dir  = os.path.join(_SCRIPT_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'semantic_regression_{run_id}.log')
    _log_fh  = open(log_path, 'w', encoding='utf-8', buffering=1)
    sys.stdout = _Tee(_log_fh, sys.__stdout__)
    sys.stderr = _Tee(_log_fh, sys.__stderr__)

    patients = args.patients if args.patients else _discover_patients(DATA_FOLDER, TASK)

    _header('Semantic Regression  –  Batch Pipeline')
    print(f'  Run ID       : {run_id}')
    print(f'  Task         : {TASK}')
    if AUDITORY_WARP != 'none':
        print(f'  Warp mode    : {AUDITORY_WARP}  (scope: {AUDITORY_WARP_SCOPE})')
    if ALIGN_CUE != 'none':
        _ab = f'{ALIGN_BACK}s' if ALIGN_BACK   is not None else 'full'
        _af = f'{ALIGN_FORWARD}s' if ALIGN_FORWARD is not None else 'full'
        print(f'  Align cue    : {ALIGN_CUE}  (back={_ab}, fwd={_af})')
    print(f'  Embeddings   : {EMBEDDING_NAMES}')
    print(f'  Epochs       : {args.epochs}')
    print(f'  Closest      : {args.closest}')
    print(f'  Bin size     : {BIN_SIZE} ms  |  history: {N_BINS_HISTORY} bins')
    print(f'  KRR alpha    : {KRR_ALPHA}  |  PCA components: {Y_PCA_COMPONENTS}')
    print(f'  Patients     : {patients}')
    print(f'  Log file     : {log_path}')

    if TASK == 'auditory_naming':
        check_auditory_naming_availability()

    if not patients:
        print('\n  No patients to process. Exiting.')
        return

    # ── Resolve the warp target ───────────────────────────────────────────────
    # Under scope='group' every patient is warped to ONE duration — the median (over the
    # pooled trials of all patients in this run) of the active warp segment — so the
    # seg-end event (stim offset / voice onset) lands at the same time for everybody and
    # group figures can mark it with a single line. It must be resolved here: before the
    # first patient is warped, and before _build_meta records it. Note it depends on
    # `patients`, so a --patients subset shifts it.
    # --warp-target-sec short-circuits the computation: the target is supplied, so it does
    # NOT depend on `patients` and adding a patient cannot move it. That is the whole point
    # of the flag — see its help text. The per-patient medians are still computed and
    # recorded, because they are the evidence for whether the pin is a sane target for the
    # patients actually being run.
    warp_patient_medians = {}
    if AUDITORY_WARP != 'none' and AUDITORY_WARP_SCOPE == 'group':
        if args.warp_target_sec is not None:
            AUDITORY_WARP_TARGET_SEC = float(args.warp_target_sec)
            AUDITORY_WARP_TARGET_SOURCE = 'pinned'
            _, warp_patient_medians = compute_group_segment_duration(patients, AUDITORY_WARP)
            print(f'    {"PINNED":6s}  --warp-target-sec {AUDITORY_WARP_TARGET_SEC:.4f} s  '
                  f'← every patient warped to this (run cohort did NOT set it)')
        else:
            AUDITORY_WARP_TARGET_SEC, warp_patient_medians = \
                compute_group_segment_duration(patients, AUDITORY_WARP)
            AUDITORY_WARP_TARGET_SOURCE = 'computed'
        if AUDITORY_WARP_TARGET_SEC is None:
            _warn('No usable warp-segment durations across patients — falling back to '
                  'per-patient median warp (scope=patient)')
            AUDITORY_WARP_SCOPE = 'patient'
            AUDITORY_WARP_TARGET_SOURCE = None
    elif args.warp_target_sec is not None:
        _warn('--warp-target-sec ignored: it only applies with --warp != none and '
              f'--warp-scope group (got --warp {AUDITORY_WARP} --warp-scope '
              f'{AUDITORY_WARP_SCOPE})')

    # ── Run output directories ────────────────────────────────────────────────
    # Absolute, via utils.paths — never relative to the working directory. A relative
    # path here put output outside the repository whenever this was launched from the
    # project root instead of main/. create=False leaves directory creation where it
    # already happens, in _write_meta a few lines down.
    fig_run_dir     = _figures_dir('semantic_regression', run_id, create=False)
    results_run_dir = _results_dir('semantic_regression', run_id, create=False)

    # ── 1.  Load shared models (once) ─────────────────────────────────────────
    shared = load_shared_embedding_models()

    # ── 2.  Write run metadata ────────────────────────────────────────────────
    meta = _build_meta(args, patients, run_id, log_path,
                       warp_patient_medians=warp_patient_medians)
    _write_meta(meta, fig_run_dir, results_run_dir)
    _step(f'meta.json written → {fig_run_dir}  &  {results_run_dir}')

    # ── 3.  Process each patient ──────────────────────────────────────────────
    n_total  = len(patients)
    n_ok     = 0
    n_failed = 0
    succeeded_patients = []
    failed_patients    = []

    for idx, patient in enumerate(patients, start=1):
        _header(f'Patient {idx}/{n_total}:  {patient}')
        fig_dir     = os.path.join(fig_run_dir,     patient)
        results_dir = os.path.join(results_run_dir, patient)
        try:
            pdata      = load_patient_data(patient)
            embeddings = build_patient_embeddings(pdata, shared,
                                                  embedding_names=EMBEDDING_NAMES)
            regressors = run_regressions(
                pdata, embeddings,
                n_epochs=args.epochs,
                closest=args.closest,
                model_mode=args.model,
                embedding_names=EMBEDDING_NAMES,
            )
            save_figures(patient, pdata, regressors, fig_dir)
            save_source_data(patient, pdata, regressors, results_dir)
            _section(f'Patient {patient}  COMPLETE')
            print(f'  Figures : {fig_dir}')
            print(f'  Results : {results_dir}')
            n_ok += 1
            succeeded_patients.append(patient)
        except Exception:
            n_failed += 1
            failed_patients.append(patient)
            _sep('━')
            print(f'  ERROR – patient {patient}')
            traceback.print_exc()
            _sep('━')
            print('  Continuing to next patient …')

    # ── 4.  Update meta.json with outcome ─────────────────────────────────────
    meta['succeeded_patients'] = succeeded_patients
    meta['failed_patients']    = failed_patients
    meta['n_succeeded']        = n_ok
    meta['n_failed']           = n_failed
    _write_meta(meta, fig_run_dir, results_run_dir)

    _header(f'Batch complete  –  {n_ok} succeeded, {n_failed} failed')

    # Restore stdout/stderr and close log
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    _log_fh.close()
    print(f'\n  Log saved → {log_path}')


if __name__ == '__main__':
    main()
 