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

# Phoneme embedding types (keys also used as plot labels)
EMBEDDING_NAMES = ['panphon', 'token_ipa']

IMAGE_FOLDER_NAME  = 'pictureNaming extended all'
EMBEDDINGS_FOLDER  = os.path.join('embeddings', IMAGE_FOLDER_NAME)

# Mapping: embedding name → pickle filename
_PWESUITE_FILES = {
    'panphon':   'pwesuite_panphon_embeddings.pk',
    'token_ipa': 'pwesuite_token_ipa_embeddings.pk',
}

TASK_TO_XLSX = {
    'picture_naming': os.path.join(
        'data_archive', 'wordset picture naming expanded.xlsx'
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
#  Terminal progress helpers  (identical to semantic_regression.py)
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


def _find_df_path(patient_folder, patient, task):
    std = os.path.join(patient_folder, f'{patient}_{task}_df.pkl')
    if os.path.exists(std):
        return std
    combined = os.path.join(patient_folder, f'{patient}_{task}_combined_df.pkl')
    if os.path.exists(combined):
        return combined
    return None


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
    go_cue_onset   = trial_df['go_cue_onset'].values.astype(float)
    trial_offset   = trial_df['trial_offset'].values.astype(float)
    voice_onset    = trial_df['voice_onset'].values.astype(float)
    voice_offset   = trial_df['voice_offset'].values.astype(float)
    target_labels  = trial_df['target_word'].values.astype(str)
    answer_labels  = trial_df['answered_word'].values.astype(str)
    bad_trials     = (trial_df['bad_trials'].values.astype(bool)
                      if 'bad_trials' in trial_df.columns
                      else np.ones(len(trial_df), dtype=bool))
    _ok(f'fs={fs} Hz  |  {len(data_list)} trials  |  data shape[0]: {data_list[0].shape}')

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
    _ok(f'{len(np.unique(target_concept))} unique concepts, {len(ambig)} homonym base(s)')

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
        ax.set_xlabel('Number of test samples')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n_models:]:
        ax.set_visible(False)

    fig.suptitle(f'Per-word: sample count vs. {ylabel}', fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Save figures
# ─────────────────────────────────────────────────────────────────────────────

def save_figures(patient, pdata, regressors, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    _section(f'Saving figures  →  {fig_dir}')

    model_map = {name: regressors[name] for name in EMBEDDING_NAMES}
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

def discover_patients():
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
    }


def _write_meta(meta, *dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)


def main():
    global EMBEDDING_NAMES, BIN_SIZE, PLS_COMPONENTS

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
        default='krr',
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
    args = parser.parse_args()

    os.chdir(_SCRIPT_DIR)

    # ── Override globals from CLI ─────────────────────────────────────────────
    if args.embedding is not None:
        EMBEDDING_NAMES = args.embedding
    BIN_SIZE       = args.bin_size
    PLS_COMPONENTS = args.n_components

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_id    = f'{timestamp}_{args.model}_{args.closest}_{args.epochs}ep'

    log_dir  = os.path.join(_SCRIPT_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'phoneme_regression_{run_id}.log')
    _log_fh  = open(log_path, 'w', encoding='utf-8', buffering=1)
    sys.stdout = _Tee(_log_fh, sys.__stdout__)
    sys.stderr = _Tee(_log_fh, sys.__stderr__)

    patients = args.patients if args.patients else discover_patients()

    _header('Phoneme Regression  –  Batch Pipeline')
    print(f'  Run ID       : {run_id}')
    print(f'  Task         : {TASK}')
    print(f'  Embeddings   : {EMBEDDING_NAMES}')
    print(f'  Epochs       : {args.epochs}')
    print(f'  Closest      : {args.closest}')
    print(f'  Model        : {args.model}')
    print(f'  n_components : {PLS_COMPONENTS}')
    print(f'  Bin size     : {BIN_SIZE} ms  |  history: {N_BINS_HISTORY} bins')
    print(f'  Patients     : {patients}')
    print(f'  Log file     : {log_path}')

    if not patients:
        print('\n  No patients to process. Exiting.')
        return

    fig_run_dir     = os.path.join('figures',  'phoneme_regression', run_id)
    results_run_dir = os.path.join('results',  'phoneme_regression', run_id)

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
