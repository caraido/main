"""semantic_vanilla_retrieval.py
--------------------------------
Batch script: leave-one-out nearest-centroid retrieval in neural feature space.

No regression model, no embeddings.  For every trial (leave-one-out), the
per-word centroid of all *other* trials' neural features at each time bin is
used as the retrieval database.  Word & category decoding accuracies are
computed directly against these centroids.

A chance distribution (default: 50 label-permutation shuffles) is built so
that all retrieval metrics are accompanied by an empirical null distribution.

Output layout (relative to main/):
    figures/semantic_vanilla_retrieval/{run_id}/{patient}/
        word_retrieval_balanced_acc.html
        category_retrieval_balanced_acc.html
        confusion_word.png
        confusion_category.png
        count_vs_accuracy.png
        count_vs_f1.png
    figures/semantic_vanilla_retrieval/{run_id}/meta.json

    results/semantic_vanilla_retrieval/{run_id}/{patient}/
        vanilla_retrieval_results.pkl
        top1_decoding_source_data.csv
        per_time_scores.csv          – same columns as semantic_regression;
                                       r2_mean/r2_std/cosine_mean/cosine_std
                                       are NaN (not applicable)
    results/semantic_vanilla_retrieval/{run_id}/meta.json

    logs/semantic_vanilla_retrieval_{run_id}.log

Usage (from main/):
    python semantic_vanilla_retrieval.py
    python semantic_vanilla_retrieval.py --patients AZ VB
    python semantic_vanilla_retrieval.py --shuffles 100 --closest l2
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
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score

# ── project imports ───────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from utils.utils import remove_number, plot_accuracy_plotly, reformat

# ─────────────────────────────────────────────────────────────────────────────
#  Constants / defaults
# ─────────────────────────────────────────────────────────────────────────────
DATA_FOLDER    = 'data'
TASK           = 'picture_naming'
BIN_SIZE       = 100        # ms
N_BINS_HISTORY = 10
N_SHUFFLES     = 50         # label permutations for chance distribution

TASK_TO_XLSX = {
    'picture_naming': os.path.join(
        'data_archive', 'wordset picture naming expanded.xlsx'
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
#  Terminal progress helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sep(char='─', width=72):
    print(char * width)

def _header(msg):
    print()
    _sep('═')
    print(f'  {msg}')
    _sep('═')

def _section(msg):
    print()
    _sep()
    print(f'  >> {msg}')
    _sep()

def _step(msg):
    print(f'     ▸  {msg}')

def _ok(msg=''):
    print(f'        ✓  {msg}')

def _warn(msg):
    print(f'        ⚠  {msg}')

def _progress(current, total, label=''):
    bar_len = 40
    filled  = int(bar_len * current / total) if total else 0
    bar     = '█' * filled + '░' * (bar_len - filled)
    print(f'\r        [{bar}] {current}/{total}  {label}', end='', flush=True)

def _progress_done():
    print()


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
#  NeuralRetriever
# ─────────────────────────────────────────────────────────────────────────────

class NeuralRetriever:
    """
    Leave-one-out nearest-centroid retrieval in neural feature space.

    For each time bin and each trial *i*, the per-word centroid of all *other*
    trials' neural features is computed (LOO) and used as the retrieval
    database.  No regression or dimensionality reduction is applied.

    All ``all_retrieval_*`` attribute names match those of ``BasicRegressor``
    so that all downstream figure- and CSV-saving helpers work unchanged.

    Real-data output shapes : ``(1, n_bins)``  — wrapped as if one epoch.
    Chance output shapes    : ``(n_shuffles, n_bins)``.

    Trials whose word has only one occurrence cannot form an unbiased LOO
    centroid and are silently excluded from metrics.
    """

    def __init__(self):
        self.n_bins_history = 0
        self.n_shuffles     = 0
        self._closest       = None
        self.labels         = None
        self.X_to_use       = None
        self.n_bins         = None

        # Label bookkeeping — identical interface to BasicRegressor
        self.word_to_index              = {}
        self.index_to_word              = np.array([], dtype=object)
        self.category_to_index          = {}
        self.index_to_category          = np.array([], dtype=object)
        self.word_index_to_category_index = None
        self._index_dtype               = np.uint32

        # Real retrieval — shape (1, n_bins)
        self.all_retrieval_top1                  = np.array([[]])
        self.all_retrieval_top3                  = np.array([[]])
        self.all_retrieval_top5                  = np.array([[]])
        self.all_retrieval_word_balanced_acc     = np.array([[]])
        self.all_retrieval_word_f1               = np.array([[]])
        self.all_retrieval_category_top1         = np.array([[]])
        self.all_retrieval_category_balanced_acc = np.array([[]])
        self.all_retrieval_category_f1           = np.array([[]])

        # Chance — shape (n_shuffles, n_bins)
        self.all_retrieval_chance_top1                  = np.array([[]])
        self.all_retrieval_chance_top3                  = np.array([[]])
        self.all_retrieval_chance_top5                  = np.array([[]])
        self.all_retrieval_chance_word_balanced_acc     = np.array([[]])
        self.all_retrieval_chance_word_f1               = np.array([[]])
        self.all_retrieval_category_chance_top1         = np.array([[]])
        self.all_retrieval_category_chance_balanced_acc = np.array([[]])
        self.all_retrieval_category_chance_f1           = np.array([[]])

        # Pair records — same schema as BasicRegressor.all_retrieval_pairs
        self.all_retrieval_pairs = []

    # ------------------------------------------------------------------
    def load_data(self, data, n_bins_history=10, labels=None, category_labels=None):
        """
        Parameters
        ----------
        data : ndarray, shape (n_trials, n_bins, n_channels)
            Cleaned, binned neural activity (axes swapped from clean_data_binned).
        n_bins_history : int
            Consecutive bins to concatenate per feature vector (passed to ``reformat``).
        labels : array-like of str
            Word identity per trial (required).
        category_labels : array-like of str
            Semantic category per trial (optional).
        """
        if labels is None:
            raise ValueError('labels must be provided to NeuralRetriever.load_data')

        self.n_bins_history = n_bins_history
        self.labels         = np.asarray(labels)

        unique_words, sample_word_idx = np.unique(self.labels, return_inverse=True)
        self.index_to_word = np.asarray(unique_words)
        self.word_to_index = {w: i for i, w in enumerate(self.index_to_word)}
        n_words = len(self.index_to_word)

        if n_words <= np.iinfo(np.uint16).max:
            self._index_dtype = np.uint16
        elif n_words <= np.iinfo(np.uint32).max:
            self._index_dtype = np.uint32
        else:
            self._index_dtype = np.uint64

        if category_labels is not None:
            category_labels = np.asarray(category_labels)
            word_categories = np.empty(n_words, dtype=object)
            assigned        = np.zeros(n_words, dtype=bool)
            for wi, cat in zip(sample_word_idx, category_labels):
                if not assigned[wi]:
                    word_categories[wi] = cat
                    assigned[wi]        = True
                elif word_categories[wi] != cat:
                    raise ValueError(
                        f"Word '{self.index_to_word[wi]}' maps to multiple categories: "
                        f"'{word_categories[wi]}' and '{cat}'"
                    )
            unique_categories = np.unique(word_categories)
            self.index_to_category = np.asarray(unique_categories)
            self.category_to_index = {c: i for i, c in enumerate(self.index_to_category)}
            self.word_index_to_category_index = np.array(
                [self.category_to_index[c] for c in word_categories], dtype=np.int32
            )
        else:
            self.index_to_category            = np.array([], dtype=object)
            self.category_to_index            = {}
            self.word_index_to_category_index = None

        self.X_to_use = reformat(data, n_bins_history)
        self.n_bins   = len(self.X_to_use)

    # ------------------------------------------------------------------
    def fit(self, n_shuffles=50, closest='cosine', save_retrieval_pairs=True):
        """
        Run LOO nearest-centroid retrieval on real data, then build a
        chance distribution via label permutation.

        Parameters
        ----------
        n_shuffles : int
            Number of label permutations for the chance distribution.
        closest : str
            Distance metric: ``'cosine'`` (default) or ``'l2'``.
        save_retrieval_pairs : bool
            Store per-bin (true, pred) index pairs for confusion matrices and CSVs.
        """
        self._closest   = closest
        self.n_shuffles = n_shuffles

        word_idx_all = np.array(
            [self.word_to_index[lbl] for lbl in self.labels], dtype=np.int64
        )

        # ── Real retrieval (one LOO pass) ──────────────────────────────────
        _step('LOO retrieval (real data) …')
        real = self._run_loo_pass(
            self.X_to_use, word_idx_all, self.labels, closest, save_retrieval_pairs
        )
        self.all_retrieval_top1                  = np.array([real['top1']])
        self.all_retrieval_top3                  = np.array([real['top3']])
        self.all_retrieval_top5                  = np.array([real['top5']])
        self.all_retrieval_word_balanced_acc     = np.array([real['word_bal_acc']])
        self.all_retrieval_word_f1               = np.array([real['word_f1']])
        self.all_retrieval_category_top1         = np.array([real['cat_top1']])
        self.all_retrieval_category_balanced_acc = np.array([real['cat_bal_acc']])
        self.all_retrieval_category_f1           = np.array([real['cat_f1']])
        if save_retrieval_pairs:
            self.all_retrieval_pairs = real['pairs']

        best_word = int(np.nanargmax(np.nanmean(self.all_retrieval_top1, axis=0)))
        _ok(f'best bin={best_word}  |  word top-1={float(np.nanmax(real["top1"])):.3f}'
            f'  |  word bal-acc={float(np.nanmax(real["word_bal_acc"])):.3f}')

        # ── Chance (n_shuffles permuted-label passes) ──────────────────────
        _step(f'Chance distribution ({n_shuffles} shuffles) …')
        ch_top1     = []
        ch_top3     = []
        ch_top5     = []
        ch_wbal     = []
        ch_wf1      = []
        ch_cat_top1 = []
        ch_cat_bal  = []
        ch_cat_f1   = []

        for sh in range(n_shuffles):
            _progress(sh + 1, n_shuffles)
            perm_labels   = np.random.permutation(self.labels)
            perm_word_idx = np.array(
                [self.word_to_index[lbl] for lbl in perm_labels], dtype=np.int64
            )
            ch = self._run_loo_pass(
                self.X_to_use, perm_word_idx, perm_labels,
                closest, save_pairs=False
            )
            ch_top1.append(ch['top1'])
            ch_top3.append(ch['top3'])
            ch_top5.append(ch['top5'])
            ch_wbal.append(ch['word_bal_acc'])
            ch_wf1.append(ch['word_f1'])
            ch_cat_top1.append(ch['cat_top1'])
            ch_cat_bal.append(ch['cat_bal_acc'])
            ch_cat_f1.append(ch['cat_f1'])

        _progress_done()

        self.all_retrieval_chance_top1                  = np.array(ch_top1)
        self.all_retrieval_chance_top3                  = np.array(ch_top3)
        self.all_retrieval_chance_top5                  = np.array(ch_top5)
        self.all_retrieval_chance_word_balanced_acc     = np.array(ch_wbal)
        self.all_retrieval_chance_word_f1               = np.array(ch_wf1)
        self.all_retrieval_category_chance_top1         = np.array(ch_cat_top1)
        self.all_retrieval_category_chance_balanced_acc = np.array(ch_cat_bal)
        self.all_retrieval_category_chance_f1           = np.array(ch_cat_f1)

        _ok(f'chance mean word top-1 @ best bin: '
            f'{float(np.nanmean(self.all_retrieval_chance_top1, axis=0)[best_word]):.3f}'
            f' ± {float(np.nanstd(self.all_retrieval_chance_top1, axis=0)[best_word]):.3f}')

    # ------------------------------------------------------------------
    def _run_loo_pass(self, X_to_use, word_idx_all, labels_all,
                      closest, save_pairs):
        """
        LOO centroid retrieval over all time bins for one label assignment.

        For each bin:
          1. Build per-word sums / counts from all trials.
          2. For each trial *i*: patch word w_i's centroid to exclude trial *i*,
             then find the nearest centroid.  Trials with count_w == 1 are skipped.

        Top-3 / top-5 are computed by checking whether the true word appears
        among the 3 / 5 nearest centroids.
        """
        n_words       = len(self.index_to_word)
        top1_bins     = []
        top3_bins     = []
        top5_bins     = []
        wbal_bins     = []
        wf1_bins      = []
        cat_top1_bins = []
        cat_bal_bins  = []
        cat_f1_bins   = []
        pairs_list    = []

        for bin_idx, X_bin in enumerate(X_to_use):
            n_trials, n_feat = X_bin.shape

            # Per-word sums and counts (vectorised)
            word_sums   = np.zeros((n_words, n_feat), dtype=np.float64)
            word_counts = np.zeros(n_words, dtype=np.int64)
            np.add.at(word_sums,   word_idx_all, X_bin)
            np.add.at(word_counts, word_idx_all, 1)

            # Global centroids
            valid_w   = word_counts > 0
            centroids = np.zeros_like(word_sums)
            centroids[valid_w] = (
                word_sums[valid_w] / word_counts[valid_w, np.newaxis]
            )

            # Mean-centre the DB (constant across all queries in this bin)
            db_mean = centroids.mean(axis=0)
            db_c    = centroids - db_mean   # (n_words, n_feat)

            # Pre-normalise for cosine distance
            if closest == 'cosine':
                db_norms  = np.linalg.norm(db_c, axis=1, keepdims=True) + 1e-10
                db_normed = db_c / db_norms   # (n_words, n_feat)
  
            true_wi_list   = []
            pred_wi_list   = []
            top3_wi_list   = []
            top5_wi_list   = []
            trial_idx_list = []

            for i in range(n_trials):
                wi = int(word_idx_all[i])
                if word_counts[wi] <= 1:
                    # Cannot form an unbiased LOO centroid
                    continue

                # LOO centroid for word wi (mean-centred)
                loo_c = (word_sums[wi] - X_bin[i]) / (word_counts[wi] - 1) - db_mean

                if closest == 'cosine':
                    # Temporarily patch the normalised DB row for word wi
                    saved_row       = db_normed[wi].copy()
                    loo_norm        = np.linalg.norm(loo_c) + 1e-10
                    db_normed[wi]   = loo_c / loo_norm

                    q_c    = X_bin[i] - db_mean
                    q_norm = np.linalg.norm(q_c) + 1e-10
                    dist   = 1.0 - db_normed @ (q_c / q_norm)   # (n_words,)

                    db_normed[wi] = saved_row   # restore

                else:  # L2
                    saved_row = db_c[wi].copy()
                    db_c[wi]  = loo_c

                    q_c  = X_bin[i] - db_mean
                    diff = db_c - q_c[np.newaxis, :]   # (n_words, n_feat)
                    dist = np.sum(diff ** 2, axis=1)    # (n_words,)

                    db_c[wi] = saved_row   # restore

                sorted_idx = np.argsort(dist)
                true_wi_list.append(wi)
                pred_wi_list.append(int(sorted_idx[0]))
                top3_wi_list.append(sorted_idx[:3].astype(self._index_dtype))
                top5_wi_list.append(sorted_idx[:5].astype(self._index_dtype))
                trial_idx_list.append(i)

            if not true_wi_list:
                # All words had only one trial — fill with NaN
                nan = float('nan')
                for lst in (top1_bins, top3_bins, top5_bins,
                            wbal_bins, wf1_bins,
                            cat_top1_bins, cat_bal_bins, cat_f1_bins):
                    lst.append(nan)
                if save_pairs:
                    pairs_list.append({
                        'bin_index':     int(bin_idx),
                        'fold_index':    -1,
                        'test_indices':  np.array([], dtype=np.int32),
                        'true_word_idx': np.array([], dtype=self._index_dtype),
                        'pred_word_idx': np.array([], dtype=self._index_dtype),
                    })
                continue

            true_wi_arr = np.array(true_wi_list, dtype=self._index_dtype)
            pred_wi_arr = np.array(pred_wi_list, dtype=self._index_dtype)
            top3_wi_arr = np.stack(top3_wi_list, axis=0)   # (n_valid, 3)
            top5_wi_arr = np.stack(top5_wi_list, axis=0)   # (n_valid, 5)

            top1_acc = float(np.mean(pred_wi_arr == true_wi_arr))
            top3_acc = float(np.mean(
                np.any(top3_wi_arr == true_wi_arr[:, np.newaxis], axis=1)
            ))
            top5_acc = float(np.mean(
                np.any(top5_wi_arr == true_wi_arr[:, np.newaxis], axis=1)
            ))

            _word_labels = np.unique(true_wi_arr).tolist()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                wbal_acc = float(balanced_accuracy_score(true_wi_arr, pred_wi_arr))
            wf1 = float(f1_score(true_wi_arr, pred_wi_arr, average='macro',
                                  labels=_word_labels, zero_division=0))

            cat_top1_acc = cat_bal_acc = cat_f1 = float('nan')
            if self.word_index_to_category_index is not None:
                pred_cat    = self.word_index_to_category_index[pred_wi_arr]
                true_cat    = self.word_index_to_category_index[true_wi_arr]
                _cat_labels = np.unique(true_cat).tolist()
                cat_top1_acc = float(np.mean(pred_cat == true_cat))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    cat_bal_acc = float(balanced_accuracy_score(true_cat, pred_cat))
                cat_f1 = float(f1_score(true_cat, pred_cat, average='macro',
                                         labels=_cat_labels, zero_division=0))

            top1_bins.append(top1_acc)
            top3_bins.append(top3_acc)
            top5_bins.append(top5_acc)
            wbal_bins.append(wbal_acc)
            wf1_bins.append(wf1)
            cat_top1_bins.append(cat_top1_acc)
            cat_bal_bins.append(cat_bal_acc)
            cat_f1_bins.append(cat_f1)

            if save_pairs:
                pairs_list.append({
                    'bin_index':     int(bin_idx),
                    'fold_index':    -1,
                    'test_indices':  np.array(trial_idx_list, dtype=np.int32),
                    'true_word_idx': true_wi_arr,
                    'pred_word_idx': pred_wi_arr,
                })

        return {
            'top1':        top1_bins,
            'top3':        top3_bins,
            'top5':        top5_bins,
            'word_bal_acc': wbal_bins,
            'word_f1':     wf1_bins,
            'cat_top1':    cat_top1_bins,
            'cat_bal_acc': cat_bal_bins,
            'cat_f1':      cat_f1_bins,
            'pairs':       pairs_list,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Small utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_pkl(path):
    with open(path, 'rb') as f:
        return dill.load(f)


def _find_df_path(patient_folder, patient, task):
    """Try the standard path first, then a 'combined' variant."""
    std = os.path.join(patient_folder, f'{patient}_{task}_df.pkl')
    if os.path.exists(std):
        return std
    combined = os.path.join(patient_folder, f'{patient}_{task}_combined_df.pkl')
    if os.path.exists(combined):
        return combined
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Per-patient data loading & preprocessing  (same as semantic_regression.py)
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

    fs              = int(trial_df['fs'].iloc[0])
    n_samp_per_bin  = fs * BIN_SIZE // 1000
    data_list       = list(trial_df['hg_data'].values)
    trial_onset     = trial_df['trial_onset'].values.astype(float)
    go_cue_onset    = trial_df['go_cue_onset'].values.astype(float)
    trial_offset    = trial_df['trial_offset'].values.astype(float)
    voice_onset     = trial_df['voice_onset'].values.astype(float)
    voice_offset    = trial_df['voice_offset'].values.astype(float)
    target_labels   = trial_df['target_word'].values.astype(str)
    answer_labels   = trial_df['answered_word'].values.astype(str)
    bad_trials      = (trial_df['bad_trials'].values.astype(bool)
                       if 'bad_trials' in trial_df.columns
                       else np.ones(len(trial_df), dtype=bool))
    _ok(f'fs={fs} Hz  |  {len(data_list)} trials  |  '
        f'data shape[0]: {data_list[0].shape}')

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
    shortest_trial = min(d.shape[1] for d in data_list)
    data           = np.array([d[:, :shortest_trial] for d in data_list])
    min_length     = data.shape[2] // n_samp_per_bin * n_samp_per_bin
    data           = data[:, :, :min_length]
    data_binned    = data.reshape(data.shape[0], data.shape[1], -1, n_samp_per_bin).mean(axis=3)
    del data
    gc.collect()
    adjusted_fs = int(1000 / BIN_SIZE)
    _ok(f'data_binned: {data_binned.shape}  (n_trials, n_channels, n_bins)')

    clean_data_binned   = np.delete(data_binned, bad_channels, axis=1)[bad_trials]
    del data_binned
    gc.collect()
    clean_voice_onset   = voice_onset[bad_trials]
    clean_voice_offset  = voice_offset[bad_trials]
    clean_target_labels = target_labels[bad_trials]
    clean_answer_labels = answer_labels[bad_trials]
    _ok(f'clean_data_binned: {clean_data_binned.shape}')

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
    _ok(f'{len(np.unique(target_concept))} unique concepts, '
        f'{len(ambig)} homonym base(s)')

    return dict(
        patient             = patient,
        fs                  = fs,
        adjusted_fs         = adjusted_fs,
        clean_data_binned   = clean_data_binned,
        clean_target_labels = clean_target_labels,
        clean_answer_labels = clean_answer_labels,
        clean_channel_names = np.array(channel_names),
        clean_word_category = clean_word_category,
        clean_voice_onset   = clean_voice_onset,
        clean_voice_offset  = clean_voice_offset,
        trial_onset         = trial_onset,
        go_cue_onset        = go_cue_onset,
        trial_offset        = trial_offset,
        voice_onset         = voice_onset,
        target_lexeme       = target_lexeme,
        target_lemma        = target_lemma,
        target_concept      = target_concept,
        labels_df           = labels_df,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Run retrieval for one patient
# ─────────────────────────────────────────────────────────────────────────────

def run_retrieval(pdata, n_shuffles=N_SHUFFLES, closest='cosine'):
    """Instantiate a NeuralRetriever, load data, and fit."""
    X = pdata['clean_data_binned'].swapaxes(1, 2)   # → (n_trials, n_bins, n_ch)
    nr = NeuralRetriever()
    nr.load_data(
        X,
        n_bins_history  = N_BINS_HISTORY,
        labels          = pdata['target_concept'],
        category_labels = pdata['clean_word_category'],
    )
    nr.fit(n_shuffles=n_shuffles, closest=closest, save_retrieval_pairs=True)
    return nr


# ─────────────────────────────────────────────────────────────────────────────
#  Confusion-matrix helpers
# ─────────────────────────────────────────────────────────────────────────────

def _best_bin_from_top1(br, mode='word'):
    if mode == 'category':
        cat_top1 = np.asarray(br.all_retrieval_category_top1)
        if cat_top1.ndim == 2 and cat_top1.size > 0:
            return int(np.nanargmax(np.nanmean(cat_top1, axis=0)))
    top1 = np.asarray(br.all_retrieval_top1)
    return int(np.nanargmax(np.nanmean(top1, axis=0)))


def _collect_pairs_at_bin(br, bin_index):
    true_idx, pred_idx = [], []
    for rec in br.all_retrieval_pairs:
        if int(rec['bin_index']) == int(bin_index):
            true_idx.append(np.asarray(rec['true_word_idx'], dtype=np.int64))
            pred_idx.append(np.asarray(rec['pred_word_idx'], dtype=np.int64))
    if not true_idx:
        raise ValueError(f'No retrieval pairs found for bin {bin_index}')
    return np.concatenate(true_idx), np.concatenate(pred_idx)


def _make_cm(br, bin_index=None, mode='word'):
    if bin_index is None:
        bin_index = _best_bin_from_top1(br, mode=mode)
    y_true_w, y_pred_w = _collect_pairs_at_bin(br, bin_index)
    if mode == 'word':
        int_labels = np.arange(len(br.index_to_word), dtype=np.int64)
        y_true, y_pred, names = y_true_w, y_pred_w, np.asarray(br.index_to_word)
    else:
        int_labels = np.arange(len(br.index_to_category), dtype=np.int64)
        y_true = br.word_index_to_category_index[y_true_w].astype(np.int64)
        y_pred = br.word_index_to_category_index[y_pred_w].astype(np.int64)
        names  = np.asarray(br.index_to_category)
    cm = confusion_matrix(y_true, y_pred, labels=int_labels)
    return cm, names, bin_index


def _normalize_col(cm):
    cm      = cm.astype(float)
    col_sum = cm.sum(axis=0, keepdims=True)
    col_sum[col_sum == 0] = 1.0
    return cm / col_sum


def _rank_labels_by_f1(cm):
    tp      = np.diag(cm).astype(float)
    fp      = cm.sum(axis=0).astype(float) - tp
    fn      = cm.sum(axis=1).astype(float) - tp
    denom   = 2.0 * tp + fp + fn
    f1      = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    support = cm.sum(axis=1).astype(float)
    return np.lexsort((np.arange(len(f1)), -support, -f1))


def _plot_cm_grid(model_map, mode='word', normalize=True, cmap='viridis',
                  top_k_words_by_f1=None):
    n_models = len(model_map)
    n_cols   = min(3, n_models)
    n_rows   = math.ceil(n_models / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 7 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, (model_name, br) in zip(axes_flat, model_map.items()):
        cm, word_names, best_bin = _make_cm(br, mode=mode)
        total_n = int(cm.sum())
        if mode == 'word' and top_k_words_by_f1 is not None:
            k    = max(1, min(int(top_k_words_by_f1), len(word_names)))
            keep = _rank_labels_by_f1(cm)[:k]
            cm   = cm[np.ix_(keep, keep)]
            word_names = word_names[keep]
        shown_n  = int(cm.sum())
        cm_plot  = _normalize_col(cm) if normalize else cm.astype(float)
        vmin, vmax = (0.0, 1.0) if normalize else (0.0, max(float(cm_plot.max()), 1.0))
        im = ax.imshow(cm_plot, aspect='auto', cmap=cmap, origin='lower',
                       vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        n_ticks = len(word_names)
        ax.set_xticks(np.arange(n_ticks))
        ax.set_yticks(np.arange(n_ticks))
        ax.set_xticklabels(word_names, rotation=90, fontsize=9)
        ax.set_yticklabels(word_names, fontsize=9)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        title = f'{model_name} | best bin={best_bin}'
        title += f' | N={shown_n}' if shown_n == total_n else f' | shown={shown_n}/{total_n}'
        ax.set_title(title)

    for ax in axes_flat[n_models:]:
        ax.set_visible(False)

    mode_str = 'Single-word' if mode == 'word' else 'Category'
    fig.suptitle(
        f'{mode_str} retrieval confusion matrices (column-normalised)',
        fontsize=14,
    )
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Per-word count vs metric plots
# ─────────────────────────────────────────────────────────────────────────────

def _per_word_stats(br):
    bin_idx = _best_bin_from_top1(br, mode='word')
    y_true, y_pred = _collect_pairs_at_bin(br, bin_idx)
    n_words = len(br.index_to_word)
    counts  = np.zeros(n_words, dtype=int)
    correct = np.zeros(n_words, dtype=int)
    for wi in range(n_words):
        mask         = y_true == wi
        counts[wi]   = mask.sum()
        correct[wi]  = (y_pred[mask] == wi).sum()
    accuracy = np.where(counts > 0, correct / counts, np.nan)
    return br.index_to_word, counts, accuracy, bin_idx


def _per_word_f1_stats(br):
    cm, names, bin_idx = _make_cm(br, mode='word')
    tp      = np.diag(cm).astype(float)
    fp      = cm.sum(axis=0).astype(float) - tp
    fn      = cm.sum(axis=1).astype(float) - tp
    denom   = 2.0 * tp + fp + fn
    f1      = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    counts  = cm.sum(axis=1).astype(int)
    f1      = np.where(counts > 0, f1, np.nan)
    return names, counts, f1, bin_idx


def _plot_count_vs_metric(model_map, metric='accuracy'):
    n_models = len(model_map)
    n_cols   = min(3, n_models)
    n_rows   = math.ceil(n_models / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 5 * n_rows),
                             squeeze=False)
    axes_flat = axes.ravel()

    for ax, (model_name, br) in zip(axes_flat, model_map.items()):
        if metric == 'accuracy':
            words, counts, vals, best_bin = _per_word_stats(br)
            ylabel = 'Top-1 word accuracy'
        else:
            words, counts, vals, best_bin = _per_word_f1_stats(br)
            ylabel = 'Per-class F1'

        valid = ~np.isnan(vals)
        ax.scatter(counts[valid], vals[valid], s=60, alpha=0.75, zorder=3)
        for w, c, v in zip(words[valid], counts[valid], vals[valid]):
            if v > 0:
                ax.annotate(w, (c, v), textcoords='offset points',
                            xytext=(4, 3), fontsize=9, alpha=0.85)
        nonzero = valid & (vals > 0)
        if nonzero.sum() >= 2:
            r = np.corrcoef(counts[nonzero].astype(float), vals[nonzero])[0, 1]
            ax.set_title(f'{model_name}  |  best bin={best_bin}  |  r={r:.2f}')
        else:
            ax.set_title(f'{model_name}  |  best bin={best_bin}')
        ax.set_xlabel('Number of trials')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n_models:]:
        ax.set_visible(False)

    fig.suptitle(f'Per-word: trial count vs. {ylabel}', fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Save figures
# ─────────────────────────────────────────────────────────────────────────────

def save_figures(patient, pdata, retriever, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    _section(f'Saving figures  →  {fig_dir}')

    model_map = {'Neural': retriever}
    adj_fs    = pdata['adjusted_fs']
    t_onset   = pdata['trial_onset']
    go_cue    = pdata['go_cue_onset']
    v_on      = pdata['clean_voice_onset']
    v_off     = pdata['clean_voice_offset']
    n_bins    = pdata['clean_data_binned'].shape[2]

    back    = float(np.nanmean(t_onset))
    forward = float(n_bins / adj_fs - np.nanmean(t_onset))

    common_lines = [
        0 - np.nanmean(t_onset),
        float(np.nanmean(go_cue) - np.nanmean(t_onset)),
        float(np.nanmean(v_on)   - np.nanmean(t_onset)),
        float(np.nanmean(v_off)  - np.nanmean(t_onset)),
    ]
    line_labels = ['trial onset', 'go cue', 'voice on', 'voice off']

    plotly_kw = dict(
        lines         = common_lines,
        line_labels   = line_labels,
        data_labels   = ['Neural', 'chance'],
        back          = back,
        forward       = forward,
        tick_interval = 1,
    )

    real_word_bal  = np.nanmean(retriever.all_retrieval_word_balanced_acc,     axis=0)
    real_cat_bal   = np.nanmean(retriever.all_retrieval_category_balanced_acc,  axis=0)
    ch_word_bal    = np.nanmean(retriever.all_retrieval_chance_word_balanced_acc, axis=0)
    ch_cat_bal     = np.nanmean(retriever.all_retrieval_category_chance_balanced_acc, axis=0)
    ch_word_bal_std = np.nanstd(retriever.all_retrieval_chance_word_balanced_acc, axis=0)
    ch_cat_bal_std  = np.nanstd(retriever.all_retrieval_category_chance_balanced_acc, axis=0)
    zero_std        = np.zeros(n_bins)

    # ── 1.  Word retrieval balanced accuracy ───────────────────────────────
    _step('Word retrieval balanced accuracy …')
    fig_wb, _ = plot_accuracy_plotly(
        real_word_bal,
        ch_word_bal,
        data_std = [zero_std, ch_word_bal_std],
        ylabel   = 'Balanced Accuracy',
        title    = f'{patient}: Word Retrieval Balanced Accuracy (Vanilla)',
        **plotly_kw,
    )
    fig_wb.write_html(os.path.join(fig_dir, 'word_retrieval_balanced_acc.html'))
    _ok('word_retrieval_balanced_acc.html')

    # ── 2.  Category retrieval balanced accuracy ───────────────────────────
    _step('Category retrieval balanced accuracy …')
    fig_cb, _ = plot_accuracy_plotly(
        real_cat_bal,
        ch_cat_bal,
        data_std = [zero_std, ch_cat_bal_std],
        ylabel   = 'Balanced Accuracy',
        title    = f'{patient}: Category Retrieval Balanced Accuracy (Vanilla)',
        **plotly_kw,
    )
    fig_cb.write_html(os.path.join(fig_dir, 'category_retrieval_balanced_acc.html'))
    _ok('category_retrieval_balanced_acc.html')

    # ── 3.  Confusion matrices – word (top-10 by F1) ──────────────────────
    _step('Confusion matrix (word, top-10 by F1) …')
    fig_cw = _plot_cm_grid(model_map, mode='word', normalize=True,
                           cmap='viridis', top_k_words_by_f1=10)
    fig_cw.savefig(os.path.join(fig_dir, 'confusion_word.png'),
                   dpi=150, bbox_inches='tight')
    plt.close(fig_cw)
    _ok('confusion_word.png')

    # ── 4.  Confusion matrices – category ─────────────────────────────────
    _step('Confusion matrix (category) …')
    fig_cc = _plot_cm_grid(model_map, mode='category', normalize=True, cmap='viridis')
    fig_cc.savefig(os.path.join(fig_dir, 'confusion_category.png'),
                   dpi=150, bbox_inches='tight')
    plt.close(fig_cc)
    _ok('confusion_category.png')

    # ── 5.  Per-word count vs accuracy ────────────────────────────────────
    _step('Per-word count vs. accuracy …')
    fig_ca = _plot_count_vs_metric(model_map, metric='accuracy')
    fig_ca.savefig(os.path.join(fig_dir, 'count_vs_accuracy.png'),
                   dpi=150, bbox_inches='tight')
    plt.close(fig_ca)
    _ok('count_vs_accuracy.png')

    # ── 6.  Per-word count vs F1 ──────────────────────────────────────────
    _step('Per-word count vs. F1 …')
    fig_cf = _plot_count_vs_metric(model_map, metric='f1')
    fig_cf.savefig(os.path.join(fig_dir, 'count_vs_f1.png'),
                   dpi=150, bbox_inches='tight')
    plt.close(fig_cf)
    _ok('count_vs_f1.png')


# ─────────────────────────────────────────────────────────────────────────────
#  Save source data
# ─────────────────────────────────────────────────────────────────────────────

def save_source_data(patient, pdata, retriever, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    _section(f'Saving source data  →  {results_dir}')

    # ── 1.  Full retriever object (pkl) ───────────────────────────────────
    _step('vanilla_retrieval_results.pkl …')
    reg_path = os.path.join(results_dir, 'vanilla_retrieval_results.pkl')
    with open(reg_path, 'wb') as f:
        pk.dump({
            'patient':              patient,
            'retriever':            retriever,
            'target_concept':       pdata['target_concept'],
            'clean_word_category':  pdata['clean_word_category'],
            'clean_target_labels':  pdata['clean_target_labels'],
            'clean_answer_labels':  pdata['clean_answer_labels'],
            'clean_channel_names':  pdata['clean_channel_names'],
            'bin_size_ms':          BIN_SIZE,
            'n_bins_history':       N_BINS_HISTORY,
            'n_shuffles':           retriever.n_shuffles,
            'closest':              retriever._closest,
        }, f, protocol=4)
    _ok(f'vanilla_retrieval_results.pkl  ({os.path.getsize(reg_path) / 1e6:.1f} MB)')

    # ── 2.  Top-1 decoding source data CSV ────────────────────────────────
    _step('top1_decoding_source_data.csv …')
    best_bin_word = _best_bin_from_top1(retriever, mode='word')
    best_bin_cat  = _best_bin_from_top1(retriever, mode='category')
    rows = []
    for rec in retriever.all_retrieval_pairs:
        bin_idx = int(rec['bin_index'])
        true_wi = np.asarray(rec['true_word_idx'], dtype=np.int64)
        pred_wi = np.asarray(rec['pred_word_idx'], dtype=np.int64)
        for tw, pw in zip(true_wi, pred_wi):
            true_word = retriever.index_to_word[tw]
            pred_word = retriever.index_to_word[pw]
            if retriever.word_index_to_category_index is not None:
                true_cat = retriever.index_to_category[
                    retriever.word_index_to_category_index[tw]]
                pred_cat = retriever.index_to_category[
                    retriever.word_index_to_category_index[pw]]
            else:
                true_cat = pred_cat = 'N/A'
            rows.append({
                'patient':           patient,
                'bin_index':         bin_idx,
                'is_best_word_bin':  bin_idx == best_bin_word,
                'is_best_cat_bin':   bin_idx == best_bin_cat,
                'true_word':         true_word,
                'pred_word':         pred_word,
                'true_category':     true_cat,
                'pred_category':     pred_cat,
                'word_correct':      true_word == pred_word,
                'category_correct':  true_cat  == pred_cat,
            })

    df_pairs = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, 'top1_decoding_source_data.csv')
    df_pairs.to_csv(csv_path, index=False)
    _ok(f'top1_decoding_source_data.csv  '
        f'({len(df_pairs):,} rows, '
        f'{df_pairs["bin_index"].nunique()} bins)')

    # ── 3.  Per-time-bin summary scores CSV ───────────────────────────────
    _step('per_time_scores.csv …')
    n_bins          = retriever.n_bins
    wbal_mean       = np.nanmean(retriever.all_retrieval_word_balanced_acc,     axis=0)
    cbal_mean       = np.nanmean(retriever.all_retrieval_category_balanced_acc,  axis=0)
    wf1_mean        = np.nanmean(retriever.all_retrieval_word_f1,                axis=0)
    cf1_mean        = np.nanmean(retriever.all_retrieval_category_f1,            axis=0)
    top3_mean       = np.nanmean(retriever.all_retrieval_top3,                   axis=0)
    top5_mean       = np.nanmean(retriever.all_retrieval_top5,                   axis=0)
    ch_wbal_mean    = np.nanmean(retriever.all_retrieval_chance_word_balanced_acc,     axis=0)
    ch_cbal_mean    = np.nanmean(retriever.all_retrieval_category_chance_balanced_acc, axis=0)
    ch_wbal_std     = np.nanstd( retriever.all_retrieval_chance_word_balanced_acc,     axis=0)
    ch_cbal_std     = np.nanstd( retriever.all_retrieval_category_chance_balanced_acc, axis=0)

    score_rows = []
    for b in range(n_bins):
        score_rows.append({
            'patient':                          patient,
            'bin_index':                        b,
            # R² and cosine not applicable for vanilla retrieval — kept as NaN
            # so downstream code reading the same columns still works
            'r2_mean':                          float('nan'),
            'r2_std':                           float('nan'),
            'cosine_mean':                      float('nan'),
            'cosine_std':                       float('nan'),
            'chance_mean':                      float('nan'),   # R² chance N/A
            'word_balanced_acc':                wbal_mean[b],
            'category_balanced_acc':            cbal_mean[b],
            'word_f1':                          wf1_mean[b],
            'category_f1':                      cf1_mean[b],
            'word_top3_acc':                    top3_mean[b],
            'word_top5_acc':                    top5_mean[b],
            # Additional chance columns specific to vanilla retrieval
            'chance_word_balanced_acc':         ch_wbal_mean[b],
            'chance_category_balanced_acc':     ch_cbal_mean[b],
            'chance_word_balanced_acc_std':     ch_wbal_std[b],
            'chance_category_balanced_acc_std': ch_cbal_std[b],
        })

    df_scores  = pd.DataFrame(score_rows)
    scores_path = os.path.join(results_dir, 'per_time_scores.csv')
    df_scores.to_csv(scores_path, index=False)
    _ok(f'per_time_scores.csv  ({len(df_scores):,} rows)')


# ─────────────────────────────────────────────────────────────────────────────
#  Patient discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_patients():
    """Return sorted list of patient IDs that have a picture_naming_df.pkl."""
    patients = []
    if not os.path.isdir(DATA_FOLDER):
        return patients
    for name in sorted(os.listdir(DATA_FOLDER)):
        folder = os.path.join(DATA_FOLDER, name)
        if not os.path.isdir(folder):
            continue
        if _find_df_path(folder, name, TASK) is not None:
            patients.append(name)
    return patients


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _git_hash():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=_SCRIPT_DIR, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _git_dirty():
    try:
        out = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            cwd=_SCRIPT_DIR, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return len(out) > 0
    except Exception:
        return None


def _build_meta(args, patients, run_id, log_path):
    import sklearn
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = 'N/A'

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
        'data_folder':          os.path.abspath(DATA_FOLDER),
        'patients':             patients,
        'model_mode':           'vanilla_retrieval',
        'n_shuffles':           args.shuffles,
        'bin_size_ms':          BIN_SIZE,
        'n_bins_history':       N_BINS_HISTORY,
        'closest':              args.closest,
        'python_version':       platform.python_version(),
        'platform':             platform.platform(),
        'numpy_version':        np.__version__,
        'pandas_version':       pd.__version__,
        'sklearn_version':      sklearn.__version__,
        'torch_version':        torch_version,
    }


def _write_meta(meta, *dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)


def main():
    global BIN_SIZE

    parser = argparse.ArgumentParser(
        description='Batch vanilla retrieval: LOO nearest-centroid in neural feature space',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--patients', nargs='*', default=None,
        help='Patient IDs to process (omit to auto-discover all)',
    )
    parser.add_argument(
        '--shuffles', type=int, default=N_SHUFFLES,
        help='Number of label-permutation shuffles for the chance distribution',
    )
    parser.add_argument(
        '--closest', choices=['l2', 'cosine'], default='cosine',
        help='Retrieval distance metric',
    )
    parser.add_argument(
        '--bin-size', type=int, default=BIN_SIZE,
        help='Bin size in ms',
    )
    args = parser.parse_args()

    os.chdir(_SCRIPT_DIR)
    BIN_SIZE = args.bin_size

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_id    = f'{timestamp}_vanilla_{args.closest}_{args.shuffles}sh'
  
    log_dir  = os.path.join(_SCRIPT_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'semantic_vanilla_retrieval_{run_id}.log')
    _log_fh  = open(log_path, 'w', encoding='utf-8', buffering=1)
    sys.stdout = _Tee(_log_fh, sys.__stdout__)
    sys.stderr = _Tee(_log_fh, sys.__stderr__)

    patients = args.patients if args.patients else discover_patients()

    _header('Vanilla Neural Retrieval  –  Batch Pipeline')
    print(f'  Run ID        : {run_id}')
    print(f'  Task          : {TASK}')
    print(f'  Method        : LOO nearest-centroid (no model, no embeddings)')
    print(f'  Shuffles      : {args.shuffles}')
    print(f'  Closest       : {args.closest}')
    print(f'  Bin size      : {BIN_SIZE} ms  |  history: {N_BINS_HISTORY} bins')
    print(f'  Patients      : {patients}')
    print(f'  Log file      : {log_path}')

    if not patients:
        print('\n  No patients to process. Exiting.')
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        _log_fh.close()
        return

    fig_run_dir     = os.path.join('figures',  'semantic_vanilla_retrieval', run_id)
    results_run_dir = os.path.join('results',  'semantic_vanilla_retrieval', run_id)

    meta = _build_meta(args, patients, run_id, log_path)
    _write_meta(meta, fig_run_dir, results_run_dir)
    _step(f'meta.json written → {fig_run_dir}  &  {results_run_dir}')

    n_total           = len(patients)
    n_ok              = 0
    n_failed          = 0
    succeeded_patients = []
    failed_patients    = []

    for idx, patient in enumerate(patients, start=1):
        _header(f'Patient {idx}/{n_total}:  {patient}')
        fig_dir     = os.path.join(fig_run_dir,     patient)
        results_dir = os.path.join(results_run_dir, patient)
        try:
            pdata     = load_patient_data(patient)
            retriever = run_retrieval(
                pdata,
                n_shuffles = args.shuffles,
                closest    = args.closest,
            )
            save_figures(patient, pdata, retriever, fig_dir)
            save_source_data(patient, pdata, retriever, results_dir)
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
