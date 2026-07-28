# -*- coding: utf-8 -*-
"""phoneme_regression.py
-----------------------
Batch script: neural activity → phoneme embeddings (phoneme regression).

Mirrors semantic_regression.py but uses PWESuite phoneme embeddings instead
of lexical-semantic embeddings.  Two embedding types are supported:

  panphon    — articulatory feature vectors (pwesuite_panphon_embeddings.pk)
  token_ipa  — IPA token-sequence vectors   (pwesuite_token_ipa_embeddings.pk)

Both produce 300-dimensional embeddings per word.

Output layout (relative to main/):
    figures/phoneme_regression/{run_id}/{patient}/
        r2_over_time.html
        word_retrieval_balanced_acc.html
        category_retrieval_balanced_acc.html
        confusion_word.png
        confusion_category.png
        count_vs_accuracy.png
        count_vs_f1.png
    figures/phoneme_regression/{run_id}/meta.json

    results/phoneme_regression/{run_id}/{patient}/
        phoneme_regression_results.pkl
        top1_decoding_source_data.csv
        per_time_scores.csv
    results/phoneme_regression/{run_id}/meta.json

    logs/phoneme_regression_{run_id}.log

Usage (from main/):
    python phoneme_regression.py
    python phoneme_regression.py --patients AZ VB
    python phoneme_regression.py --embedding panphon token_ipa
    python phoneme_regression.py --epochs 30 --closest cosine --model kernel_pls
    python phoneme_regression.py --bin-size 20
"""

import argparse
import collections
import gc
import json
import math
import os
import platform
import subprocess
import sys
import pickle as pk
import traceback
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

import dill
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

# ── project imports ───────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from utils.utils import remove_number, plot_accuracy_plotly
from models.model import BasicRegressor

# --- cleanup batch 1: imports added by automated migration ---
from utils.logging import _sep, _header, _section
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
PLS_COMPONENTS     = 10
ALIGN_VOICE        = False
VOICE_BACK         = 2.5
VOICE_FORWARD      = 1.5

# Auditory naming
AUDITORY_WARP = 'none'
ALIGN_CUE     = 'none'
ALIGN_BACK    = None
ALIGN_FORWARD = None

EMBEDDING_NAMES = ['panphon', 'token_ipa']
IMAGE_FOLDER_NAME  = 'pictureNaming extended all'
EMBEDDINGS_FOLDER  = os.path.join('embeddings', IMAGE_FOLDER_NAME)
_PWESUITE_FILES = {
    'panphon':   'pwesuite_panphon_embeddings.pk',
    'token_ipa': 'pwesuite_token_ipa_embeddings.pk',
}
TASK_TO_XLSX = {
    'picture_naming': os.path.join('data_archive', 'wordset picture naming expanded.xlsx'),
    'auditory_naming': os.path.join('data_archive', 'wordset picture naming expanded.xlsx'),
}
_ANSWER_ARTICLES = ('a ', 'an ', 'the ')


def _normalize_answer_to_head_noun(w):
    """Strip leading article + collapse multi-word to head noun. Verbatim — no
    lemma/synset matching, so 'taxi' and 'cab' stay distinct."""
    s = str(w).strip().lower()
    for art in _ANSWER_ARTICLES:
        if s.startswith(art):
            s = s[len(art):].lstrip(); break
    if ' ' in s:
        s = s.split()[-1]
    return s


# ─────────────────────────────────────────────────────────────────────────────
#  Terminal progress helpers  (identical to semantic_regression.py)
# ─────────────────────────────────────────────────────────────────────────────

def _step(msg):
    print(f'     ▸  {msg}')

def _ok(msg=''):
    print(f'        ✓  {msg}')

def _warn(msg):
    print(f'        ⚠  {msg}')


class _Tee:
    """Duplicate writes to both the original stream and a log file."""
    def __init__(self, log_file, original_stream):
        self._log  = log_file
        self._term = original_stream

    def write(self, data):
        self._term.write(data)
        self._term.flush()
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


def _map_phoneme_embed(embed_dict, target_labels):
    """Return [N_samples, D] array aligned to target_labels.

    Lookup order:
      1. Exact match on normalized word.
      2. Strip pic-number via remove_number (e.g. 'bat1' → 'bat') and average
         all matching vocab variants.
      3. Strip ALL trailing digits (e.g. 'nut21' → 'nut') and average variants.
    Missing words get a zero vector and trigger a warning.
    """
    words_norm   = _normalize_tokens(np.asarray(embed_dict['words']))
    embeddings   = np.asarray(embed_dict['phoneme_embedding'])   # (N_vocab, D)
    dim          = embeddings.shape[1]

    exact  = {w: i for i, w in enumerate(words_norm)}
    base   = {}
    for i, w in enumerate(words_norm):
        base.setdefault(remove_number(w), []).append(i)

    labels_norm = _normalize_tokens(target_labels)
    out, missing = [], []
    for t_raw, t_norm in zip(target_labels, labels_norm):
        if t_norm in exact:
            out.append(embeddings[exact[t_norm]])
        elif t_norm in base:
            variants = [embeddings[i] for i in base[t_norm]]
            out.append(np.mean(variants, axis=0))
        elif remove_number(t_norm) in base:
            variants = [embeddings[i] for i in base[remove_number(t_norm)]]
            out.append(np.mean(variants, axis=0))
        else:
            # Strip ALL trailing digits (e.g. 'nut21' → 'nut', 'bat22' → 'bat')
            bare = ''.join(c for c in t_norm if c.isalpha())
            if bare and bare in base:
                variants = [embeddings[i] for i in base[bare]]
                out.append(np.mean(variants, axis=0))
            elif bare and bare in exact:
                out.append(embeddings[exact[bare]])
            else:
                out.append(np.full(dim, np.nan, dtype=np.float32))
                missing.append(t_raw)
    if missing:
        _warn(f'{len(missing)} labels not found in phoneme vocab '
              f'(NaN assigned, trial will be dropped): {missing[:5]}')
    return np.array(out, dtype=np.float32)





def _linear_time_warp(data, fs, aud_stim_onset, aud_stim_offset, timing_arrays):
    """Linearly warp [stim_onset, stim_offset] of each trial to median stim duration."""
    from scipy.interpolate import interp1d
    durations = np.array([
        int(np.round(aud_stim_offset[i] * fs)) - int(np.round(aud_stim_onset[i] * fs))
        for i in range(len(data))
    ])
    median_dur = int(np.median(durations))
    _step(f'Time-warp: stim durations min={durations.min()} max={durations.max()} '
          f'median={median_dur} samples ({median_dur/fs:.3f} s)')
    data_warped = []
    aud_stim_offset_w = np.empty_like(aud_stim_offset)
    timing_arrays_w = {k: v.copy() for k, v in timing_arrays.items()}
    def _warp_cue(cue_time, onset_idx, offset_idx, median_dur, fs):
        cue_idx = cue_time * fs
        if np.isnan(cue_time): return cue_time
        if cue_idx < onset_idx: return cue_time
        if cue_idx > offset_idx:
            shift = (median_dur - (offset_idx - onset_idx)) / fs
            return cue_time + shift
        orig_dur = offset_idx - onset_idx
        if orig_dur <= 0: return cue_time
        rel = (cue_idx - onset_idx) / orig_dur
        return (onset_idx + rel * median_dur) / fs
    for i in range(len(data)):
        trial = data[i]
        onset_idx  = int(np.round(aud_stim_onset[i]  * fs))
        offset_idx = int(np.round(aud_stim_offset[i] * fs))
        offset_idx = max(offset_idx, onset_idx + 1)
        pre, during, post = trial[:, :onset_idx], trial[:, onset_idx:offset_idx], trial[:, offset_idx:]
        orig_t = np.arange(during.shape[1])
        warp_t = np.linspace(0, during.shape[1] - 1, median_dur)
        warped = np.zeros((trial.shape[0], median_dur))
        for ch in range(trial.shape[0]):
            f_ = interp1d(orig_t, during[ch], kind='linear', fill_value='extrapolate')
            warped[ch] = f_(warp_t)
        data_warped.append(np.concatenate([pre, warped, post], axis=1))
        aud_stim_offset_w[i] = (onset_idx + median_dur) / fs
        for k in timing_arrays_w:
            timing_arrays_w[k][i] = _warp_cue(timing_arrays[k][i], onset_idx, offset_idx, median_dur, fs)
    shortest = min(d.shape[1] for d in data_warped)
    data_warped = np.array([d[:, :shortest] for d in data_warped])
    _ok(f'Warped data shape: {data_warped.shape}')
    return data_warped, aud_stim_onset.copy(), aud_stim_offset_w, timing_arrays_w


# ─────────────────────────────────────────────────────────────────────────────
#  Shared phoneme embedding loading  (done ONCE)
# ─────────────────────────────────────────────────────────────────────────────

def load_shared_embedding_models():
    """Load PWESuite pickle files for the requested embedding types."""
    _header('Loading PWESuite phoneme embedding files  (one-time cost)')
    shared = {}
    for name in EMBEDDING_NAMES:
        fname = _PWESUITE_FILES[name]
        fpath = os.path.join(EMBEDDINGS_FOLDER, fname)
        _step(f'{name}  ←  {fpath} …')
        with open(fpath, 'rb') as f:
            shared[name] = pk.load(f)
        vocab_size = len(shared[name]['words'])
        dim        = np.asarray(shared[name]['phoneme_embedding']).shape[1]
        mode       = str(shared[name].get('feature_mode', name))
        _ok(f'{name}: {vocab_size} words, dim={dim}, feature_mode={mode}')
    _section('All phoneme embedding files loaded')
    return shared


# ─────────────────────────────────────────────────────────────────────────────
#  Per-patient data loading & preprocessing  (identical to semantic_regression)
# ─────────────────────────────────────────────────────────────────────────────

def load_patient_data(patient):
    """Load, bin, and clean neural data for one patient."""
    patient_folder = os.path.join(DATA_FOLDER, patient)
    df_path        = _find_df_path(patient_folder, patient, TASK)
    labels_path    = os.path.join(patient_folder, f'{patient}_{TASK}_labels.pkl')
    if df_path is None or not os.path.exists(labels_path):
        raise FileNotFoundError(
            f'Missing data for {patient}: df_path={df_path}, labels_path={labels_path}'
        )

    for ch_path in [
        os.path.join(patient_folder, f'{patient}_{TASK}_channels.pkl'),
        os.path.join(patient_folder, f'{patient}_channels.pkl'),
    ]:
        if os.path.exists(ch_path):
            channels_path = ch_path
            break
    else:
        channels_path = None

    _step(f'Loading {os.path.basename(df_path)} …')
    trial_df    = load_pkl(df_path)
    labels_df   = load_pkl(labels_path)
    channels_df = load_pkl(channels_path) if channels_path else None
    if isinstance(trial_df,   dict): trial_df   = pd.DataFrame(trial_df)
    if isinstance(labels_df,  dict): labels_df  = pd.DataFrame(labels_df)
    if isinstance(channels_df, dict) and channels_df is not None:
        channels_df = pd.DataFrame(channels_df)
    _ok(f'trial_df {trial_df.shape},  labels_df {labels_df.shape}')

    fs             = int(trial_df['fs'].iloc[0])
    n_samp_per_bin = fs * BIN_SIZE // 1000
    data_list      = list(trial_df['hg_data'].values)
    trial_onset    = trial_df['trial_onset'].values.astype(float)
    go_cue_onset   = (trial_df['go_cue_onset'].values.astype(float)
                     if 'go_cue_onset' in trial_df.columns
                     else _extract_col(trial_df, 'green_screen_onset'))
    trial_offset   = trial_df['trial_offset'].values.astype(float)
    voice_onset    = trial_df['voice_onset'].values.astype(float)
    voice_offset   = trial_df['voice_offset'].values.astype(float)
    target_labels  = trial_df['target_word'].values.astype(str)
    answer_labels  = trial_df['answered_word'].values.astype(str)
    bad_trials     = (trial_df['bad_trials'].values.astype(bool)
                      if 'bad_trials' in trial_df.columns
                      else np.ones(len(trial_df), dtype=bool))

    # Auditory naming: derive aud_stim onset/offset from prompt_word_onsets/offsets
    if TASK == 'auditory_naming' and 'prompt_word_onsets' in trial_df.columns:
        def _first(v):
            a = np.asarray(v, dtype=float).ravel()
            return float(a[0]) if len(a) > 0 else np.nan
        def _last(v):
            a = np.asarray(v, dtype=float).ravel()
            return float(a[-1]) if len(a) > 0 else np.nan
        aud_stim_onset  = np.array([_first(v) for v in trial_df['prompt_word_onsets']])
        aud_stim_offset = np.array([_last(v)  for v in trial_df['prompt_word_offsets']])
        _ok(f'aud_stim_onset range:  [{np.nanmin(aud_stim_onset):.3f}, '
            f'{np.nanmax(aud_stim_onset):.3f}] s')
        _ok(f'aud_stim_offset range: [{np.nanmin(aud_stim_offset):.3f}, '
            f'{np.nanmax(aud_stim_offset):.3f}] s')
    else:
        aud_stim_onset  = _extract_col(trial_df, 'aud_stim_onset',
                                        'auditory_stimulus_onset', 'stimulus_onset')
        aud_stim_offset = _extract_col(trial_df, 'aud_stim_offset',
                                        'auditory_stimulus_offset', 'stimulus_offset')
    _ok(f'fs={fs} Hz  |  {len(data_list)} trials  |  data shape[0]: {data_list[0].shape}')

    # Auditory naming: optional linear time-warp BEFORE binning
    if TASK == 'auditory_naming' and AUDITORY_WARP == 'linear':
        valid_warp = (np.isfinite(aud_stim_onset) & np.isfinite(aud_stim_offset)
                      & (aud_stim_offset > aud_stim_onset))
        if not np.all(valid_warp):
            _warn('Linear warp requested but some aud_stim onset/offset values '
                  'are invalid; skipping warp')
        else:
            _step('Applying linear time warp to raw stimulus segment ...')
            data_w, ao_w, aoff_w, t_w = _linear_time_warp(
                data_list, fs=fs,
                aud_stim_onset=aud_stim_onset,
                aud_stim_offset=aud_stim_offset,
                timing_arrays={
                    'trial_onset': trial_onset,  'trial_offset': trial_offset,
                    'go_cue':      go_cue_onset, 'voice_onset': voice_onset,
                    'voice_offset': voice_offset,
                },
            )
            data_list, aud_stim_onset, aud_stim_offset = list(data_w), ao_w, aoff_w
            trial_onset, trial_offset = t_w['trial_onset'], t_w['trial_offset']
            go_cue_onset = t_w['go_cue']
            voice_onset, voice_offset = t_w['voice_onset'], t_w['voice_offset']
            _ok('Warp applied before binning at native sampling rate')

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

    _PATIENT_EXCLUDE_PREFIXES = {
        'LH': ('O', 'V', 'P', 'Q', 'R'),
        'RB': ('V',),
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

    _step('Binning neural data …')
    adjusted_fs = int(1000 / BIN_SIZE)
    actual_back_sec = None
    actual_forward_sec = None

    _effective_align = ALIGN_CUE if ALIGN_CUE != 'none' else ('voice_onset' if ALIGN_VOICE else 'none')
    _effective_back  = (VOICE_BACK if (ALIGN_VOICE and ALIGN_CUE == 'none' and ALIGN_BACK is None)
                        else ALIGN_BACK)
    _effective_fwd   = (VOICE_FORWARD if (ALIGN_VOICE and ALIGN_CUE == 'none' and ALIGN_FORWARD is None)
                        else ALIGN_FORWARD)

    if _effective_align != 'none':
        _cue_arrays = {
            'trial_onset':     trial_onset,
            'go_cue':          go_cue_onset,
            'voice_onset':     voice_onset,
            'voice_offset':    voice_offset,
            'aud_stim_onset':  aud_stim_onset,
            'aud_stim_offset': aud_stim_offset,
        }
        if _effective_align not in _cue_arrays:
            raise ValueError(f'Unknown ALIGN_CUE: {_effective_align!r}')
        cue_arr = _cue_arrays[_effective_align]
        _back_str = f'{_effective_back}s' if _effective_back is not None else 'full'
        _fwd_str  = f'{_effective_fwd}s'  if _effective_fwd  is not None else 'full'
        _step(f'Cue-alignment enabled  (cue={_effective_align!r}, '
              f'requested back={_back_str}, fwd={_fwd_str}) ...')
        cue_samp = np.array([
            int(round(c * fs)) if np.isfinite(c) else -1
            for c in cue_arr
        ])
        good_mask = bad_trials & (cue_samp >= 0)
        if good_mask.sum() == 0:
            raise ValueError(f'No good trials with finite {_effective_align!r} for alignment')
        if _effective_back is None:
            avail_backs = np.array([cue_samp[i] for i in range(len(data_list)) if good_mask[i]])
        else:
            back_samp_req = int(round(_effective_back * fs))
            avail_backs = np.array([min(back_samp_req, cue_samp[i])
                                    for i in range(len(data_list)) if good_mask[i]])
        if _effective_fwd is None:
            avail_fwds = np.array([data_list[i].shape[1] - cue_samp[i]
                                   for i in range(len(data_list)) if good_mask[i]])
        else:
            fwd_samp_req = int(round(_effective_fwd * fs))
            avail_fwds = np.array([min(fwd_samp_req, data_list[i].shape[1] - cue_samp[i])
                                   for i in range(len(data_list)) if good_mask[i]])
        global_back_samp = (int(avail_backs.min()) // n_samp_per_bin) * n_samp_per_bin
        global_fwd_samp  = (int(avail_fwds.min())  // n_samp_per_bin) * n_samp_per_bin
        total_samp = global_back_samp + global_fwd_samp
        if total_samp < n_samp_per_bin:
            raise ValueError(f'Cue-aligned window too short')
        actual_back_sec    = global_back_samp / fs
        actual_forward_sec = global_fwd_samp  / fs
        _ok(f'Global window: back={global_back_samp} samp ({actual_back_sec:.3f}s), '
            f'fwd={global_fwd_samp} samp ({actual_forward_sec:.3f}s)')
        n_ch_raw = data_list[0].shape[0]
        aligned = []
        for i in range(len(data_list)):
            if cue_samp[i] >= 0:
                start = cue_samp[i] - global_back_samp
                end_  = cue_samp[i] + global_fwd_samp
                aligned.append(data_list[i][:, start:end_])
            else:
                aligned.append(np.zeros((n_ch_raw, total_samp), dtype=data_list[i].dtype))
        data = np.array(aligned)
        del aligned
        data_binned = data.reshape(data.shape[0], data.shape[1], -1, n_samp_per_bin).mean(axis=3)
        del data
        gc.collect()
        _ok(f'data_binned (cue-aligned to {_effective_align!r}): {data_binned.shape}')
    else:
        shortest_trial = min(d.shape[1] for d in data_list)
        data           = np.array([d[:, :shortest_trial] for d in data_list])
        min_length     = data.shape[2] // n_samp_per_bin * n_samp_per_bin
        data           = data[:, :, :min_length]
        data_binned    = data.reshape(data.shape[0], data.shape[1], -1, n_samp_per_bin).mean(axis=3)
        del data
        gc.collect()
        _ok(f'data_binned: {data_binned.shape}  (n_trials, n_channels, n_bins)')

    clean_data_binned   = np.delete(data_binned, bad_channels, axis=1)[bad_trials]
    del data_binned
    gc.collect()
    clean_voice_onset   = voice_onset[bad_trials]
    clean_voice_offset  = voice_offset[bad_trials]
    clean_target_labels = target_labels[bad_trials]
    clean_answer_labels = answer_labels[bad_trials]
    clean_aud_stim_onset  = aud_stim_onset[bad_trials]
    clean_aud_stim_offset = aud_stim_offset[bad_trials]
    _ok(f'clean_data_binned: {clean_data_binned.shape}')

    # Auditory naming: drop trials with invalid answered words; then normalize
    # (strip leading article + reduce multi-word phrases to head noun, verbatim).
    if TASK == 'auditory_naming':
        valid_mask = np.array([_is_valid_answer(w) for w in clean_answer_labels])
        n_invalid  = int((~valid_mask).sum())
        if n_invalid > 0:
            _warn(f'Removing {n_invalid} trials with invalid answered words')
            clean_data_binned     = clean_data_binned[valid_mask]
            clean_voice_onset     = clean_voice_onset[valid_mask]
            clean_voice_offset    = clean_voice_offset[valid_mask]
            clean_target_labels   = clean_target_labels[valid_mask]
            clean_answer_labels   = clean_answer_labels[valid_mask]
            clean_aud_stim_onset  = clean_aud_stim_onset[valid_mask]
            clean_aud_stim_offset = clean_aud_stim_offset[valid_mask]
        _ok(f'{valid_mask.sum()} trials kept after invalid-answer filter')

        # Normalize to single-word, article-stripped form (verbatim — no loose match)
        raw_answers = np.array(clean_answer_labels, dtype=object)
        normalized  = np.array([_normalize_answer_to_head_noun(w) for w in raw_answers])
        n_stripped = int(sum(1 for r in raw_answers
                              if any(str(r).strip().lower().startswith(a)
                                     for a in _ANSWER_ARTICLES)))
        n_multi    = int(sum(1 for r in raw_answers if ' ' in str(r).strip()))
        n_changed  = int((raw_answers != normalized).sum())
        clean_answer_labels = normalized
        if n_changed:
            _ok(f'Normalized {n_changed} answered word(s): '
                f'{n_stripped} article(s) stripped, '
                f'{n_multi} multi-word phrase(s) -> head noun')

        # Re-filter: drop trials that collapsed to empty
        post_valid = np.array([len(str(w)) > 0 and any(c.isalpha() for c in str(w))
                                for w in clean_answer_labels])
        n_drop = int((~post_valid).sum())
        if n_drop > 0:
            _warn(f'Dropping {n_drop} trials whose answered word collapsed to empty')
            clean_data_binned     = clean_data_binned[post_valid]
            clean_voice_onset     = clean_voice_onset[post_valid]
            clean_voice_offset    = clean_voice_offset[post_valid]
            clean_target_labels   = clean_target_labels[post_valid]
            clean_answer_labels   = clean_answer_labels[post_valid]
            clean_aud_stim_onset  = clean_aud_stim_onset[post_valid]
            clean_aud_stim_offset = clean_aud_stim_offset[post_valid]
        _ok(f'Final auditory trial count: {len(clean_answer_labels)}')

    # ── Relative cue statistics (voice-onset alignment) ───────────────────
    rel_cues = None
    if ALIGN_VOICE:
        _step('Computing cue times relative to voice onset …')
        clean_trial_onset  = trial_onset[bad_trials]
        clean_go_cue       = go_cue_onset[bad_trials]
        clean_trial_offset = trial_offset[bad_trials]
        rel_cues = {
            'trial_onset':  {'mean': float(np.nanmean(clean_trial_onset  - clean_voice_onset)),
                             'std':  float(np.nanstd( clean_trial_onset  - clean_voice_onset))},
            'go_cue':       {'mean': float(np.nanmean(clean_go_cue       - clean_voice_onset)),
                             'std':  float(np.nanstd( clean_go_cue       - clean_voice_onset))},
            'voice_onset':  {'mean': 0.0, 'std': 0.0},
            'voice_offset': {'mean': float(np.nanmean(clean_voice_offset - clean_voice_onset)),
                             'std':  float(np.nanstd( clean_voice_offset - clean_voice_onset))},
            'trial_offset': {'mean': float(np.nanmean(clean_trial_offset - clean_voice_onset)),
                             'std':  float(np.nanstd( clean_trial_offset - clean_voice_onset))},
        }
        for cue_name, stats in rel_cues.items():
            _ok(f'{cue_name:>15s}:  mean={stats["mean"]:+.3f}s  std={stats["std"]:.3f}s')

    _step('Assigning semantic categories …')
    if 'class' in labels_df.columns:
        w2c = dict(zip(
            labels_df['target_word'].astype(str),
            labels_df['class'].astype(str),
        ))
        word_category = np.array([w2c.get(w, 'unknown') for w in clean_target_labels])
        n_unk = (word_category == 'unknown').sum()
        if n_unk > 0:
            base2cat = {
                remove_number(str(lbl)).lower(): cat
                for lbl, cat in w2c.items()
            }
            for i, (w, cat) in enumerate(zip(clean_target_labels, word_category)):
                if cat == 'unknown':
                    word_category[i] = base2cat.get(
                        remove_number(str(w)).lower(), 'unknown'
                    )
            n_resolved = n_unk - (word_category == 'unknown').sum()
            _ok(f'Resolved {n_resolved}/{n_unk} unknown categories via base-word')
    elif TASK in TASK_TO_XLSX and os.path.exists(TASK_TO_XLSX[TASK]):
        df_xlsx   = pd.read_excel(TASK_TO_XLSX[TASK])
        wcol      = df_xlsx.columns[0]
        df_xlsx.set_index(wcol, inplace=True)
        cat_sr    = df_xlsx.fillna(0).apply(pd.to_numeric).idxmax(axis=1).reset_index()
        cat_sr.columns = [wcol, 'Category']
        w2c       = dict(zip(cat_sr[wcol], cat_sr['Category']))
        lex_tmp   = np.array([remove_number(t).lower() for t in clean_target_labels])
        word_category = np.array([w2c.get(w, 'unknown') for w in lex_tmp])
        word_category = np.array([
            'food and fruit' if w in ('fruit', 'food (exclude fruit)') else w
            for w in word_category
        ])
        _ok('Categories from xlsx')
    else:
        word_category = np.array(['unknown'] * len(clean_target_labels))
        _warn('No category source found; all categories = "unknown"')

    clean_word_category = word_category
    _ok(str(dict(collections.Counter(clean_word_category))))

    _step('Lemmatising target labels …')
    lemmatizer = WordNetLemmatizer()
    if any(kw in TASK for kw in ('Flashing', 'auditory', 'picture')):
        target_lexeme = np.array([remove_number(t).lower() for t in clean_target_labels])
    else:
        target_lexeme = np.array([str(w).lower() for w in clean_target_labels])

    target_lemma = np.array([
        lemmatizer.lemmatize(''.join(c for c in w if c.isalpha()), pos='n')
        for w in target_lexeme
    ])

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
    _ok(f'{len(np.unique(target_concept))} unique concepts, {len(ambig)} homonym base(s)')

    return dict(
        patient             = patient,
        task                = TASK,
        fs                  = fs,
        adjusted_fs         = adjusted_fs,
        clean_data_binned   = clean_data_binned,
        clean_target_labels = clean_target_labels,
        clean_answer_labels = clean_answer_labels,
        clean_channel_names = np.array(channel_names),
        clean_word_category = clean_word_category,
        clean_voice_onset   = clean_voice_onset,
        clean_voice_offset  = clean_voice_offset,
        clean_aud_stim_onset  = clean_aud_stim_onset,
        clean_aud_stim_offset = clean_aud_stim_offset,
        trial_onset         = trial_onset,
        go_cue_onset        = go_cue_onset,
        trial_offset        = trial_offset,
        voice_onset         = voice_onset,
        target_lexeme       = target_lexeme,
        target_lemma        = target_lemma,
        target_concept      = target_concept,
        labels_df           = labels_df,
        align_voice         = ALIGN_VOICE,
        align_cue           = ALIGN_CUE,
        align_back_sec      = ALIGN_BACK,
        align_forward_sec   = ALIGN_FORWARD,
        auditory_warp       = AUDITORY_WARP if TASK == 'auditory_naming' else 'N/A',
        actual_back_sec     = actual_back_sec,
        actual_forward_sec  = actual_forward_sec,
        rel_cues            = rel_cues,
    )


def check_auditory_naming_availability():
    """Print availability of auditory_naming data across patient folders."""
    _section('Auditory naming data availability check')
    if not os.path.isdir(DATA_FOLDER):
        _warn(f'DATA_FOLDER "{DATA_FOLDER}" not found'); return
    rows = []
    for name in sorted(os.listdir(DATA_FOLDER)):
        folder = os.path.join(DATA_FOLDER, name)
        if not os.path.isdir(folder): continue
        df_path  = _find_df_path(folder, name, 'auditory_naming')
        lbl_path = os.path.join(folder, f'{name}_auditory_naming_labels.pkl')
        ch_found = next((p for p in [
            os.path.join(folder, f'{name}_auditory_naming_channels.pkl'),
            os.path.join(folder, f'{name}_channels.pkl'),
            os.path.join(folder, f'{name}_picture_naming_channels.pkl'),
        ] if os.path.exists(p)), None)
        if df_path is not None or os.path.exists(lbl_path):
            rows.append((name, df_path is not None, os.path.exists(lbl_path), ch_found))
    if not rows:
        _warn('No auditory_naming data found.'); return
    print(f'\n  {"Patient":8}  {"df":4}  {"labels":7}  channels')
    print('  ' + '-' * 80)
    for name, hd, hl, ch in rows:
        ch_name = os.path.basename(ch) if ch else 'none'
        print(f'  {name:8}  {"OK" if hd else "--":4}  {"OK" if hl else "--":7}  {ch_name}')
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Per-patient embedding array building
# ─────────────────────────────────────────────────────────────────────────────

def build_patient_embeddings(pdata, shared):
    """Look up phoneme embedding arrays aligned to this patient's trial labels.

    Returns (pdata, embed) where NaN-embedding trials (ambiguous answered
    words) have been removed from both.
    """
    _step('Building phoneme embedding arrays for this patient …')
    # Use clean_answer_labels (what the patient actually said) for vocab match
    labels = pdata['clean_answer_labels']
    embed  = {}
    for name in EMBEDDING_NAMES:
        embed[name] = _map_phoneme_embed(shared[name], labels)
        _ok(f'{name}: {embed[name].shape}')

    # Drop trials whose answered word could not be embedded (NaN rows).
    n_trials = len(labels)
    valid = np.ones(n_trials, dtype=bool)
    for Y in embed.values():
        valid &= ~np.isnan(Y).any(axis=1)
    n_removed = int((~valid).sum())
    if n_removed > 0:
        _warn(f'Dropping {n_removed} trial(s) with NaN phoneme embeddings '
              f'(ambiguous answered word)')
        pdata = dict(pdata)
        for key, val in list(pdata.items()):
            if isinstance(val, np.ndarray) and val.shape[0] == n_trials:
                pdata[key] = val[valid]
        embed = {k: v[valid] for k, v in embed.items()}

    return pdata, embed


# ─────────────────────────────────────────────────────────────────────────────
#  Regression pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _make_regressor_pipeline(mode='krr'):
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


def run_regressions(pdata, embeddings, n_epochs, closest='l2', model_mode='krr'):
    """Fit one BasicRegressor per phoneme embedding type."""
    X               = pdata['clean_data_binned'].swapaxes(1, 2)
    labels          = pdata['clean_answer_labels']
    category_labels = pdata['clean_word_category']
    regressors      = {}
    n_total         = len(EMBEDDING_NAMES)

    for idx, emb_name in enumerate(EMBEDDING_NAMES, start=1):
        _step(f'[{idx}/{n_total}]  {emb_name} regression  '
              f'(epochs={n_epochs}, parallel={PARALLEL_WORKERS}, closest={closest}) …')
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
        best     = int(np.nanargmax(np.nanmean(br.all_retrieval_top1, axis=0)))
        top1     = float(np.nanmean(br.all_retrieval_top1, axis=0)[best])
        _ok(f'{emb_name} done  |  best bin={best}  |  top-1 word acc={top1:.3f}')
        gc.collect()

    return regressors


# ─────────────────────────────────────────────────────────────────────────────
#  Confusion-matrix helpers
# ─────────────────────────────────────────────────────────────────────────────







# ─────────────────────────────────────────────────────────────────────────────
#  Per-word count vs metric plots
# ─────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────
#  Save figures
# ─────────────────────────────────────────────────────────────────────────────

def save_figures(patient, pdata, regressors, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    _section(f'Saving figures  →  {fig_dir}')

    model_map = {name: regressors[name] for name in EMBEDDING_NAMES}
    adj_fs    = pdata['adjusted_fs']
    n_bins    = pdata['clean_data_binned'].shape[2]

    if pdata.get('align_voice', False):
        back    = pdata['actual_back_sec']
        forward = pdata['actual_forward_sec']
        rc      = pdata['rel_cues']
        common_lines = [
            rc['trial_onset']['mean'],
            rc['go_cue']['mean'],
            0.0,                          # voice onset is the reference
            rc['voice_offset']['mean'],
        ]
    else:
        t_onset = pdata['trial_onset']
        go_cue  = pdata['go_cue_onset']
        v_on    = pdata['clean_voice_onset']
        v_off   = pdata['clean_voice_offset']
        back    = float(np.nanmean(t_onset))
        forward = float(n_bins / adj_fs - np.nanmean(t_onset))
        common_lines = [
            0 - np.nanmean(t_onset),
            float(np.nanmean(go_cue) - np.nanmean(t_onset)),
            float(np.nanmean(v_on)   - np.nanmean(t_onset)),
            float(np.nanmean(v_off)  - np.nanmean(t_onset)),
        ]
    line_labels = ['trial onset', 'go cue', 'voice on', 'voice off']
    data_labels = EMBEDDING_NAMES + ['chance']
    zero_stds   = [0] * len(EMBEDDING_NAMES)
    br0         = regressors[EMBEDDING_NAMES[0]]

    plotly_kw = dict(
        lines         = common_lines,
        line_labels   = line_labels,
        data_labels   = data_labels,
        back          = back,
        forward       = forward,
        tick_interval = 1,
    )

    _step('R² over time …')
    fig_r2, _ = plot_accuracy_plotly(
        *[regressors[n].all_test_score.mean(0) for n in EMBEDDING_NAMES],
        br0.all_chance.mean(0),
        data_std = zero_stds + [br0.all_chance.std(0)],
        ylabel   = 'R²',
        title    = f'{patient}: R² over Time (Phoneme Regression)',
        **plotly_kw,
    )
    fig_r2.write_html(os.path.join(fig_dir, 'r2_over_time.html'))
    _ok('r2_over_time.html')

    _step('Word retrieval balanced accuracy …')
    fig_wb, _ = plot_accuracy_plotly(
        *[np.mean(regressors[n].all_retrieval_word_balanced_acc, axis=0)
          for n in EMBEDDING_NAMES],
        np.mean(br0.all_retrieval_chance_word_balanced_acc, axis=0),
        data_std = zero_stds + [np.std(br0.all_retrieval_chance_word_balanced_acc, axis=0)],
        ylabel   = 'Balanced Accuracy',
        title    = f'{patient}: Word Retrieval Balanced Accuracy (Phoneme)',
        **plotly_kw,
    )
    fig_wb.write_html(os.path.join(fig_dir, 'word_retrieval_balanced_acc.html'))
    _ok('word_retrieval_balanced_acc.html')

    _step('Category retrieval balanced accuracy …')
    fig_cb, _ = plot_accuracy_plotly(
        *[np.mean(regressors[n].all_retrieval_category_balanced_acc, axis=0)
          for n in EMBEDDING_NAMES],
        np.mean(br0.all_retrieval_category_chance_balanced_acc, axis=0),
        data_std = zero_stds + [
            np.std(br0.all_retrieval_category_chance_balanced_acc, axis=0)
        ],
        ylabel   = 'Balanced Accuracy',
        title    = f'{patient}: Category Retrieval Balanced Accuracy (Phoneme)',
        **plotly_kw,
    )
    fig_cb.write_html(os.path.join(fig_dir, 'category_retrieval_balanced_acc.html'))
    _ok('category_retrieval_balanced_acc.html')

    _step('Confusion matrix (word, top-10 by F1) …')
    fig_cw = _plot_cm_grid(model_map, mode='word', normalize=True,
                           cmap='viridis', top_k_words_by_f1=10)
    fig_cw.savefig(os.path.join(fig_dir, 'confusion_word.png'),
                   dpi=150, bbox_inches='tight')
    plt.close(fig_cw)
    _ok('confusion_word.png')

    _step('Confusion matrix (category) …')
    fig_cc = _plot_cm_grid(model_map, mode='category', normalize=True, cmap='viridis')
    fig_cc.savefig(os.path.join(fig_dir, 'confusion_category.png'),
                   dpi=150, bbox_inches='tight')
    plt.close(fig_cc)
    _ok('confusion_category.png')

    _step('Per-word count vs. accuracy …')
    fig_ca = _plot_count_vs_metric(model_map, metric='accuracy')
    fig_ca.savefig(os.path.join(fig_dir, 'count_vs_accuracy.png'),
                   dpi=150, bbox_inches='tight')
    plt.close(fig_ca)
    _ok('count_vs_accuracy.png')

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

    _step('phoneme_regression_results.pkl …')
    reg_path = os.path.join(results_dir, 'phoneme_regression_results.pkl')
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
            'align_voice':          pdata.get('align_voice', False),
            'actual_back_sec':      pdata.get('actual_back_sec'),
            'actual_forward_sec':   pdata.get('actual_forward_sec'),
            'rel_cues':             pdata.get('rel_cues'),
        }, f, protocol=4)
    _ok(f'phoneme_regression_results.pkl  ({os.path.getsize(reg_path) / 1e6:.1f} MB)')

    _step('top1_decoding_source_data.csv …')
    rows = []
    for emb_name, br in regressors.items():
        best_bin_word = _best_bin_from_top1(br, mode='word')
        best_bin_cat  = _best_bin_from_top1(br, mode='category')
        for rec in br.all_retrieval_pairs:
            bin_idx  = int(rec['bin_index'])
            true_wi  = np.asarray(rec['true_word_idx'], dtype=np.int64)
            pred_wi  = np.asarray(rec['pred_word_idx'], dtype=np.int64)
            pred_ci_indep = np.asarray(rec['pred_category_idx_indep'], dtype=np.int64) if 'pred_category_idx_indep' in rec else None
            fold_idx = int(rec.get('fold_index', 0))
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
                rows.append({
                    'patient':          patient,
                    'embedding':        emb_name,
                    'epoch':            fold_idx,
                    'bin_index':        bin_idx,
                    'is_best_word_bin': bin_idx == best_bin_word,
                    'is_best_cat_bin':  bin_idx == best_bin_cat,
                    'true_word':        true_word,
                    'pred_word':        pred_word,
                    'true_category':    true_cat,
                    'pred_category':    pred_cat,
                    'pred_category_indep': pred_cat_indep,
                    'word_correct':     true_word == pred_word,
                    'category_correct': true_cat  == pred_cat,
                    'category_correct_indep': cat_correct_indep,
                })
    df_pairs = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, 'top1_decoding_source_data.csv')
    df_pairs.to_csv(csv_path, index=False)
    _ok(f'top1_decoding_source_data.csv  ({len(df_pairs):,} rows)')

    _step('per_time_scores.csv …')
    score_rows = []
    for emb_name, br in regressors.items():
        n_bins     = br.all_test_score.shape[1]
        r2_mean    = br.all_test_score.mean(0)
        r2_std     = br.all_test_score.std(0)
        chance_mean  = br.all_chance.mean(0)
        wbal_mean    = np.mean(br.all_retrieval_word_balanced_acc,              axis=0)
        wchance_mean = np.mean(br.all_retrieval_chance_word_balanced_acc,       axis=0)
        cbal_mean    = np.mean(br.all_retrieval_category_balanced_acc,          axis=0)
        cchance_mean = np.mean(br.all_retrieval_category_chance_balanced_acc,   axis=0)
        cbal_indep_mean    = np.mean(br.all_retrieval_category_indep_balanced_acc, axis=0) if hasattr(br, 'all_retrieval_category_indep_balanced_acc') and br.all_retrieval_category_indep_balanced_acc.size > 0 else np.full(n_bins, np.nan)
        cchance_indep_mean = np.mean(br.all_retrieval_category_indep_chance_balanced_acc, axis=0) if hasattr(br, 'all_retrieval_category_indep_chance_balanced_acc') and br.all_retrieval_category_indep_chance_balanced_acc.size > 0 else np.full(n_bins, np.nan)
        wf1_mean   = np.mean(br.all_retrieval_word_f1,               axis=0)
        cf1_mean   = np.mean(br.all_retrieval_category_f1,           axis=0)
        top3_mean  = np.mean(br.all_retrieval_top3,  axis=0)
        top5_mean  = np.mean(br.all_retrieval_top5,  axis=0)
        cos_mean   = (br.all_cosine_sim.mean(0)
                      if hasattr(br, 'all_cosine_sim') and br.all_cosine_sim.size > 0
                      else np.zeros(n_bins))
        cos_std    = (br.all_cosine_sim.std(0)
                      if hasattr(br, 'all_cosine_sim') and br.all_cosine_sim.size > 0
                      else np.zeros(n_bins))
        for b in range(n_bins):
            score_rows.append({
                'patient':               patient,
                'embedding':             emb_name,
                'bin_index':             b,
                'r2_mean':               r2_mean[b],
                'r2_std':                r2_std[b],
                'cosine_mean':           cos_mean[b],
                'cosine_std':            cos_std[b],
                'chance_mean':           chance_mean[b],
                'word_balanced_acc':     wbal_mean[b],
                'word_chance_mean':      wchance_mean[b],
                'category_balanced_acc': cbal_mean[b],
                'cat_chance_mean':       cchance_mean[b],
                'category_balanced_acc_indep': cbal_indep_mean[b],
                'cat_indep_chance_mean':       cchance_indep_mean[b],
                'word_f1':               wf1_mean[b],
                'category_f1':           cf1_mean[b],
                'word_top3_acc':         top3_mean[b],
                'word_top5_acc':         top5_mean[b],
            })
    df_scores   = pd.DataFrame(score_rows)
    scores_path = os.path.join(results_dir, 'per_time_scores.csv')
    df_scores.to_csv(scores_path, index=False)
    _ok(f'per_time_scores.csv  ({len(df_scores):,} rows)')


# ─────────────────────────────────────────────────────────────────────────────
#  Patient discovery
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────



def _build_meta(args, patients, run_id, log_path):
    import sklearn
    import torch
    return {
        'run_id':               run_id,
        'timestamp_utc':        datetime.utcnow().isoformat() + 'Z',
        'timestamp_local':      datetime.now().isoformat(),
        'command_line':         sys.argv,
        'script_path':          os.path.abspath(__file__),
        'log_path':             log_path,
        'git_commit':           _git_hash(),
        'git_dirty':            _git_dirty(),
        'task':                 TASK,
        'align_cue':            ALIGN_CUE,
        'align_back_sec':       ALIGN_BACK,
        'align_forward_sec':    ALIGN_FORWARD,
        'auditory_warp':        AUDITORY_WARP if TASK == 'auditory_naming' else 'N/A',
        'auditory_word_norm':   ('strip_article+head_noun'
                                  if TASK == 'auditory_naming' else 'N/A'),
        'data_folder':          os.path.abspath(DATA_FOLDER),
        'patients':             patients,
        'n_epochs':             args.epochs,
        'bin_size_ms':          BIN_SIZE,
        'n_bins_history':       N_BINS_HISTORY,
        'y_pca_components':     Y_PCA_COMPONENTS,
        'krr_alpha':            KRR_ALPHA,
        'parallel_workers':     PARALLEL_WORKERS,
        'closest':              args.closest,
        'model_mode':           args.model,
        'embedding_names':      EMBEDDING_NAMES,
        'embeddings_folder':    os.path.abspath(EMBEDDINGS_FOLDER),
        'regressor_pipeline':   f'{args.model}: ' + {
            'krr':          f'Nystroem(rbf) → Ridge(α={KRR_ALPHA})',
            'linear_ridge': f'Ridge(α={KRR_ALPHA})',
            'pls':          f'PLSRegression(n={PLS_COMPONENTS})',
            'kernel_pls':   f'Nystroem(rbf) → PLSRegression(n={PLS_COMPONENTS})',
        }.get(args.model, '?'),
        'y_reducer':            'PCA(n_components={})'.format(Y_PCA_COMPONENTS),
        'split_strategy':       'random train_test_split',
        'split_fraction':       0.3,
        'python_version':       platform.python_version(),
        'platform':             platform.platform(),
        'numpy_version':        np.__version__,
        'pandas_version':       pd.__version__,
        'sklearn_version':      sklearn.__version__,
        'torch_version':        torch.__version__,
        'align_voice':          ALIGN_VOICE,
        'voice_back':           VOICE_BACK,
        'voice_forward':        VOICE_FORWARD,
    }



def main():
    global EMBEDDING_NAMES, BIN_SIZE, PLS_COMPONENTS, ALIGN_VOICE, VOICE_BACK, VOICE_FORWARD
    global TASK, AUDITORY_WARP, ALIGN_CUE, ALIGN_BACK, ALIGN_FORWARD

    parser = argparse.ArgumentParser(
        description='Batch phoneme regression: neural activity → PWESuite phoneme embeddings',
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
        help='Retrieval similarity metric',
    )
    parser.add_argument(
        '--model', choices=['krr', 'linear_ridge', 'pls', 'kernel_pls'],
        default='kernel_pls',
        help='Regression model',
    )
    parser.add_argument(
        '--embedding', nargs='+', default=None,
        choices=['panphon', 'token_ipa'],
        metavar='EMB',
        help='Phoneme embedding type(s) to run (default: both panphon and token_ipa)',
    )
    parser.add_argument(
        '--bin-size', type=int, default=BIN_SIZE,
        help='Bin size in ms  (default: 100)',
    )
    parser.add_argument(
        '--n-components', type=int, default=PLS_COMPONENTS,
        help='n_components for PLS/Kernel-PLS  (default: 10)',
    )
    parser.add_argument(
        '--align-voice', action='store_true', default=False,
        help='Align each trial to voice onset instead of trial start',
    )
    parser.add_argument(
        '--voice-back', type=float, default=VOICE_BACK,
        help='Seconds before voice onset to include  (default: 2.5)',
    )
    parser.add_argument(
        '--voice-forward', type=float, default=VOICE_FORWARD,
        help='Seconds after voice onset to include  (default: 1.5)',
    )
    parser.add_argument(
        '--task', choices=['picture_naming', 'auditory_naming'],
        default='picture_naming',
        help='Task type. Use "auditory_naming" for auditory paradigm.',
    )
    parser.add_argument(
        '--warp', choices=['none', 'linear'], default='none', dest='warp',
        help='Time-warp mode (auditory_naming only): "linear" warps '
             '[aud_stim_onset, aud_stim_offset] to median stimulus duration.',
    )
    parser.add_argument(
        '--align', choices=['none', 'trial_onset', 'go_cue', 'voice_onset',
                            'voice_offset', 'aud_stim_onset', 'aud_stim_offset'],
        default='none', dest='align',
        help='Behavioral cue to align each trial around before binning.',
    )
    parser.add_argument(
        '--align-back', type=float, default=None, dest='align_back',
        help='Seconds before the alignment cue (default: full available).',
    )
    parser.add_argument(
        '--align-forward', type=float, default=None, dest='align_forward',
        help='Seconds after the alignment cue (default: full available).',
    )
    args = parser.parse_args()

    os.chdir(_SCRIPT_DIR)

    # ── Override globals from CLI ─────────────────────────────────────────────
    if args.embedding is not None:
        EMBEDDING_NAMES = args.embedding
    BIN_SIZE       = args.bin_size
    PLS_COMPONENTS = args.n_components
    ALIGN_VOICE    = args.align_voice
    VOICE_BACK     = args.voice_back
    VOICE_FORWARD  = args.voice_forward
    TASK           = args.task
    AUDITORY_WARP  = args.warp
    ALIGN_CUE      = args.align
    ALIGN_BACK     = args.align_back
    ALIGN_FORWARD  = args.align_forward

    timestamp   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    task_part   = f'_{TASK}' if TASK != 'picture_naming' else ''
    warp_part   = f'_warp-{AUDITORY_WARP}' if TASK == 'auditory_naming' else ''
    align_part  = f'_align-{ALIGN_CUE}' if ALIGN_CUE != 'none' else ''
    run_id      = (f'{timestamp}{task_part}{warp_part}{align_part}'
                   f'_{args.model}_{args.closest}_{args.epochs}ep')
    if ALIGN_VOICE and ALIGN_CUE == 'none':
        run_id += '_voicealign'

    log_dir  = os.path.join(_SCRIPT_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'phoneme_regression_{run_id}.log')
    _log_fh  = open(log_path, 'w', encoding='utf-8', buffering=1)
    sys.stdout = _Tee(_log_fh, sys.__stdout__)
    sys.stderr = _Tee(_log_fh, sys.__stderr__)

    patients = args.patients if args.patients else _discover_patients(DATA_FOLDER, TASK)

    _header('Phoneme Regression  –  Batch Pipeline')
    print(f'  Run ID       : {run_id}')
    print(f'  Task         : {TASK}')
    if TASK == 'auditory_naming':
        print(f'  Warp mode    : {AUDITORY_WARP}')
    if ALIGN_CUE != 'none':
        _ab = f'{ALIGN_BACK}s' if ALIGN_BACK   is not None else 'full'
        _af = f'{ALIGN_FORWARD}s' if ALIGN_FORWARD is not None else 'full'
        print(f'  Align cue    : {ALIGN_CUE}  (back={_ab}, fwd={_af})')
    print(f'  Embeddings   : {EMBEDDING_NAMES}')
    print(f'  Epochs       : {args.epochs}')
    print(f'  Closest      : {args.closest}')
    print(f'  Model        : {args.model}')
    print(f'  n_components : {PLS_COMPONENTS}')
    print(f'  Bin size     : {BIN_SIZE} ms  |  history: {N_BINS_HISTORY} bins')
    print(f'  Align voice  : {ALIGN_VOICE}' +
          (f'  (back={VOICE_BACK}s, fwd={VOICE_FORWARD}s)' if ALIGN_VOICE else ''))
    print(f'  Patients     : {patients}')
    print(f'  Log file     : {log_path}')

    if TASK == 'auditory_naming':
        check_auditory_naming_availability()

    if not patients:
        print('\n  No patients to process. Exiting.')
        return

    # Absolute, via utils.paths — never relative to the working directory. A relative
    # path here put output outside the repository whenever this was launched from the
    # project root instead of main/. create=False leaves directory creation where it
    # already happens, in _write_meta a few lines down.
    fig_run_dir     = _figures_dir('phoneme_regression', run_id, create=False)
    results_run_dir = _results_dir('phoneme_regression', run_id, create=False)

    shared = load_shared_embedding_models()

    meta = _build_meta(args, patients, run_id, log_path)
    _write_meta(meta, fig_run_dir, results_run_dir)

    n_ok, n_failed      = 0, 0
    succeeded_patients  = []
    failed_patients     = []
    n_total             = len(patients)

    for idx, patient in enumerate(patients, start=1):
        _header(f'Patient {idx}/{n_total}:  {patient}')
        fig_dir     = os.path.join(fig_run_dir,     patient)
        results_dir = os.path.join(results_run_dir, patient)
        try:
            pdata      = load_patient_data(patient)
            pdata, embeddings = build_patient_embeddings(pdata, shared)
            regressors = run_regressions(
                pdata, embeddings,
                n_epochs=args.epochs,
                closest=args.closest,
                model_mode=args.model,
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

    meta['succeeded_patients'] = succeeded_patients
    meta['failed_patients']    = failed_patients
    meta['n_succeeded']        = n_ok
    meta['n_failed']           = n_failed
    _write_meta(meta, fig_run_dir, results_run_dir)

    _header(f'Batch complete  –  {n_ok} succeeded, {n_failed} failed')

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    _log_fh.close()
    print(f'\n  Log saved → {log_path}')


if __name__ == '__main__':
    main()
