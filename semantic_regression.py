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
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from torchtext.vocab import GloVe, FastText

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

# Embeddings are loaded in this order; the same order is used for all plots.
EMBEDDING_NAMES = ['GloVe', 'FastText', 'Word2Vec', 'ConceptNet', 'DINOv2', 'SimCLR']

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
    filled = int(bar_len * current / total) if total else 0
    bar = '█' * filled + '░' * (bar_len - filled)
    print(f'\r        [{bar}] {current}/{total}  {label}', end='', flush=True)

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
    for d, words_arr in sources:
        words_norm = _normalize_tokens(np.asarray(words_arr))
        exact: dict = {w: i for i, w in enumerate(words_norm)}
        base: dict  = {}
        for i, w in enumerate(words_norm):
            base.setdefault(remove_number(w), []).append(i)
        lookups.append((d, exact, base))

    out, missing = [], []
    for t_raw, t_norm in zip(target_labels, labels_norm):
        found = False
        for d, exact, base in lookups:
            if t_norm in exact:
                out.append(np.asarray(d[key][exact[t_norm]]).squeeze())
                found = True
                break
            elif t_norm in base:
                variants = [np.asarray(d[key][i]).squeeze() for i in base[t_norm]]
                out.append(np.mean(variants, axis=0))
                found = True
                break
        if not found:
            missing.append(t_raw)
    if missing:
        _warn(f'{key}: {len(missing)} missing label(s) after all fallbacks '
              f'e.g. {missing[:5]}')
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
    shared['_dinov2_default_folder'] = EMBEDDINGS_FOLDER
    shared['_simclr_default_folder'] = EMBEDDINGS_FOLDER

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

    # Channels: task-specific > patient-level > None
    for ch_path in [
        os.path.join(patient_folder, f'{patient}_{TASK}_channels.pkl'),
        os.path.join(patient_folder, f'{patient}_channels.pkl'),
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

    # ── Channel mask ──────────────────────────────────────────────────────────
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
    shortest_trial = min(d.shape[1] for d in data_list)
    data           = np.array([d[:, :shortest_trial] for d in data_list])
    min_length     = data.shape[2] // n_samp_per_bin * n_samp_per_bin
    data           = data[:, :, :min_length]
    data_binned    = data.reshape(data.shape[0], data.shape[1], -1, n_samp_per_bin).mean(axis=3)
    del data
    gc.collect()
    adjusted_fs = int(1000 / BIN_SIZE)
    _ok(f'data_binned: {data_binned.shape}  (n_trials, n_channels, n_bins)')

    # ── Remove bad channels / bad trials ─────────────────────────────────────
    clean_data_binned   = np.delete(data_binned, bad_channels, axis=1)[bad_trials]
    del data_binned
    gc.collect()
    clean_voice_onset   = voice_onset[bad_trials]
    clean_voice_offset  = voice_offset[bad_trials]
    clean_target_labels = target_labels[bad_trials]
    clean_answer_labels = answer_labels[bad_trials]
    _ok(f'clean_data_binned: {clean_data_binned.shape}')

    # ── Semantic categories ───────────────────────────────────────────────────
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
        _ok(f'Categories from xlsx')
    else:
        word_category = np.array(['unknown'] * len(clean_target_labels))
        _warn('No category source found; all categories = "unknown"')

    clean_word_category = word_category
    _ok(str(dict(collections.Counter(clean_word_category))))

    # ── Lemmatise labels ──────────────────────────────────────────────────────
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
    """Look up patient-specific embedding arrays from the shared models."""
    _step('Building embedding arrays for this patient …')
    lemma  = pdata['target_lemma']
    labels = pdata['clean_target_labels']

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

    return embed


# ─────────────────────────────────────────────────────────────────────────────
#  Regression
# ─────────────────────────────────────────────────────────────────────────────

def _make_regressor_pipeline():
    return Pipeline([
        ('nystroem', Nystroem(kernel='rbf')),
        ('ridge',    Ridge(alpha=KRR_ALPHA)),
    ])


def run_regressions(pdata, embeddings, n_epochs, closest='l2'):
    """Fit one BasicRegressor per embedding type; return dict name→regressor."""
    X              = pdata['clean_data_binned'].swapaxes(1, 2)   # (n_trials, n_bins, n_ch)
    labels         = pdata['target_concept']
    category_labels = pdata['clean_word_category']
    regressors     = {}
    n_total        = len(EMBEDDING_NAMES)

    for idx, emb_name in enumerate(EMBEDDING_NAMES, start=1):
        _step(f'[{idx}/{n_total}]  {emb_name} regression  (epochs={n_epochs}, '
              f'parallel={PARALLEL_WORKERS}, closest={closest}) …')
        br = BasicRegressor(_make_regressor_pipeline(), y_reducer=PCA(Y_PCA_COMPONENTS))
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

def _best_bin_from_top1(br, mode='word'):
    if mode == 'category':
        cat_top1 = np.asarray(br.all_retrieval_category_top1)
        if cat_top1.ndim == 2 and cat_top1.size > 0:
            return int(np.nanargmax(np.nanmean(cat_top1, axis=0)))
    top1 = np.asarray(br.all_retrieval_top1)
    return int(np.nanargmax(np.nanmean(top1, axis=0)))


def _collect_pairs_at_bin(br, bin_index):
    true_idx, pred_idx = [], []
    for epoch_pairs in br.all_retrieval_pairs:
        for rec in epoch_pairs:
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
#  Per-word count vs metric plots  (adapted from notebook cell #VSC-d7531eb5)
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

    model_map   = {name: regressors[name] for name in EMBEDDING_NAMES}
    adj_fs      = pdata['adjusted_fs']
    t_onset     = pdata['trial_onset']
    go_cue      = pdata['go_cue_onset']
    v_on        = pdata['clean_voice_onset']
    v_off       = pdata['clean_voice_offset']
    n_bins      = pdata['clean_data_binned'].shape[2]

    back    = float(np.nanmean(t_onset))
    forward = float(n_bins / adj_fs - np.nanmean(t_onset))

    common_lines = [
        0 - np.nanmean(t_onset),
        float(np.nanmean(go_cue) - np.nanmean(t_onset)),
        float(np.nanmean(v_on)   - np.nanmean(t_onset)),
        float(np.nanmean(v_off)  - np.nanmean(t_onset)),
    ]
    line_labels  = ['trial onset', 'go cue', 'voice on', 'voice off']
    data_labels  = EMBEDDING_NAMES + ['chance']
    zero_stds    = [0] * len(EMBEDDING_NAMES)
    br0          = regressors[EMBEDDING_NAMES[0]]

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
        *[regressors[n].all_test_score.mean(0) for n in EMBEDDING_NAMES],
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
          for n in EMBEDDING_NAMES],
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
          for n in EMBEDDING_NAMES],
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
        }, f, protocol=4)
    _ok(f'semantic_regression_results.pkl  ({os.path.getsize(reg_path) / 1e6:.1f} MB)')

    # ── 2.  Top-1 decoding source data CSV  (all test pairs, ALL time bins) ───
    # This captures true/predicted word+category for every test trial,
    # every epoch, and every time bin.  A flag marks the best-bin rows.
    _step('top1_decoding_source_data.csv  (all retrieval pairs across time) …')
    rows = []
    for emb_name, br in regressors.items():
        best_bin_word = _best_bin_from_top1(br, mode='word')
        best_bin_cat  = _best_bin_from_top1(br, mode='category')

        for epoch_idx, epoch_pairs in enumerate(br.all_retrieval_pairs):
            for rec in epoch_pairs:
                bin_idx  = int(rec['bin_index'])
                true_wi  = np.asarray(rec['true_word_idx'], dtype=np.int64)
                pred_wi  = np.asarray(rec['pred_word_idx'], dtype=np.int64)
                for tw, pw in zip(true_wi, pred_wi):
                    true_word = br.index_to_word[tw]
                    pred_word = br.index_to_word[pw]
                    if br.word_index_to_category_index is not None:
                        true_cat = br.index_to_category[br.word_index_to_category_index[tw]]
                        pred_cat = br.index_to_category[br.word_index_to_category_index[pw]]
                    else:
                        true_cat = pred_cat = 'N/A'
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
                        'word_correct':      true_word == pred_word,
                        'category_correct':  true_cat  == pred_cat,
                    })

    df_pairs = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, 'top1_decoding_source_data.csv')
    df_pairs.to_csv(csv_path, index=False)
    _ok(f'top1_decoding_source_data.csv  '
        f'({len(df_pairs):,} rows, '
        f'{df_pairs["bin_index"].nunique()} bins, '
        f'{df_pairs["embedding"].nunique()} embeddings)')

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
        wf1_mean   = np.mean(br.all_retrieval_word_f1,               axis=0)
        cf1_mean   = np.mean(br.all_retrieval_category_f1,           axis=0)
        for b in range(n_bins):
            score_rows.append({
                'patient':              patient,
                'embedding':            emb_name,
                'bin_index':            b,
                'r2_mean':              r2_mean[b],
                'r2_std':               r2_std[b],
                'chance_mean':          chance_mean[b],
                'word_balanced_acc':    wbal_mean[b],
                'category_balanced_acc': cbal_mean[b],
                'word_f1':              wf1_mean[b],
                'category_f1':          cf1_mean[b],
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
    """Return the current short git commit hash, or None if unavailable."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=_SCRIPT_DIR, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _git_dirty():
    """Return True if the working tree has uncommitted changes."""
    try:
        out = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            cwd=_SCRIPT_DIR, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return len(out) > 0
    except Exception:
        return None


def _build_meta(args, patients, run_id, log_path):
    """Build a metadata dict that captures everything needed to reproduce a run."""
    import sklearn
    import torch

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

        # ── Embeddings ────────────────────────────────────────────────────
        'embedding_names':      EMBEDDING_NAMES,
        'embeddings_folder':    os.path.abspath(EMBEDDINGS_FOLDER),

        # ── Model / pipeline ──────────────────────────────────────────────
        'regressor_pipeline':   'Nystroem(kernel="rbf") → Ridge(alpha={})'.format(KRR_ALPHA),
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


def _write_meta(meta, *dirs):
    """Write meta.json into each of the given directories."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)


def main():
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
        '--closest', choices=['l2', 'cosine'], default='l2',
        help='Retrieval similarity metric (l2 = Euclidean, cosine = cosine similarity)',
    )
    args = parser.parse_args()

    # Always run relative to this script's directory (main/)
    os.chdir(_SCRIPT_DIR)

    # ── Unique run identifier (includes model, retrieval metric, epochs) ───────
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_id = f'{timestamp}_KRR_{args.closest}_{args.epochs}ep'

    # ── Set up log file (tee stdout → terminal + file) ────────────────────────
    log_dir  = os.path.join(_SCRIPT_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'semantic_regression_{run_id}.log')
    _log_fh  = open(log_path, 'w', encoding='utf-8', buffering=1)
    sys.stdout = _Tee(_log_fh, sys.__stdout__)
    sys.stderr = _Tee(_log_fh, sys.__stderr__)

    patients = args.patients if args.patients else discover_patients()

    _header('Semantic Regression  –  Batch Pipeline')
    print(f'  Run ID       : {run_id}')
    print(f'  Task         : {TASK}')
    print(f'  Embeddings   : {EMBEDDING_NAMES}')
    print(f'  Epochs       : {args.epochs}')
    print(f'  Closest      : {args.closest}')
    print(f'  Bin size     : {BIN_SIZE} ms  |  history: {N_BINS_HISTORY} bins')
    print(f'  KRR alpha    : {KRR_ALPHA}  |  PCA components: {Y_PCA_COMPONENTS}')
    print(f'  Patients     : {patients}')
    print(f'  Log file     : {log_path}')

    if not patients:
        print('\n  No patients to process. Exiting.')
        return

    # ── Run output directories ────────────────────────────────────────────────
    fig_run_dir     = os.path.join('figures',  'semantic_regression', run_id)
    results_run_dir = os.path.join('results',  'semantic_regression', run_id)

    # ── 1.  Load shared models (once) ─────────────────────────────────────────
    shared = load_shared_embedding_models()

    # ── 2.  Write run metadata ────────────────────────────────────────────────
    meta = _build_meta(args, patients, run_id, log_path)
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
            embeddings = build_patient_embeddings(pdata, shared)
            regressors = run_regressions(
                pdata, embeddings,
                n_epochs=args.epochs,
                closest=args.closest,
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
