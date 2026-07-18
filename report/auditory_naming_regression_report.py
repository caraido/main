"""
report.auditory_naming_regression_report — HTML analysis report for auditory naming runs.

Generates a self-contained HTML report with:
  1. Run configuration (from meta.json)
  2. Per-patient time-series figures (5 rows per patient):
       cosine similarity | word verbatim | word loose | category verbatim | category loose
       + shuffled-null 95% CI shading + per-bin significance tick marks
       + vertical lines for all behavioral cues (aud_stim_onset, aud_stim_offset,
         go_cue_onset, voice_onset, voice_offset)
  3. Verbatim vs. loose accuracy comparison (per patient)
  4. Per-bin significance summary table
  5. Peak-timing analysis: peak bin relative to all preceding cues (per patient)
  6. Cross-patient summary (group-level curves + significance table)

Bonferroni correction: n_bins × n_embeddings × n_patients (most conservative).

Output: auditory_naming_report_<run_id>.html in out_dir.
"""

import os
import io
import re
import json
import base64
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as _scipy_stats

warnings.filterwarnings('ignore')

try:
    from .helper.config import EMBEDDING_NAMES
    from .helper.results_loader import load_patient_from_pkl, _install_stubs, load_pkl_raw
except ImportError:
    try:
        from helper.config import EMBEDDING_NAMES
        from helper.results_loader import load_patient_from_pkl, _install_stubs, load_pkl_raw
    except ImportError:
        from report.helper.config import EMBEDDING_NAMES
        from report.helper.results_loader import load_patient_from_pkl, _install_stubs, load_pkl_raw

# ─── Constants ─────────────────────────────────────────────────────────────────

# Only text embeddings for auditory naming (no visual models)
AN_EMBEDDING_NAMES = ['GloVe', 'FastText', 'Word2Vec', 'ConceptNet']

EMB_COLORS = {
    'GloVe':      '#1565C0',
    'FastText':   '#0288D1',
    'Word2Vec':   '#00838F',
    'ConceptNet': '#2E7D32',
}

CUE_STYLES = {
    'aud_stim_onset':  {'color': '#E65100', 'ls': '-',  'lw': 1.4, 'label': 'Stim onset'},
    'aud_stim_offset': {'color': '#AD1457', 'ls': '--', 'lw': 1.4, 'label': 'Stim offset'},
    'go_cue_onset':    {'color': '#1B5E20', 'ls': '-',  'lw': 1.2, 'label': 'Go cue'},
    'voice_onset':     {'color': '#1A237E', 'ls': '-',  'lw': 1.4, 'label': 'Voice onset'},
    'voice_offset':    {'color': '#4A148C', 'ls': '--', 'lw': 1.2, 'label': 'Voice offset'},
}

SIG_ALPHA_DISPLAY = 0.05   # Bonferroni-corrected threshold displayed on plots
ROW_LABELS = [
    'Cosine Similarity',
    'Word Acc (%)',
    'Word Loose\nBal. Acc (%)',
    'Cat Acc (%)',
    'Cat Loose\nBal. Acc (%)',
]

PLOTLY_JS = "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>"

PLOTLY_NULL_ALPHA = 0.10
PLOTLY_SIG_ALPHA = 0.95
PLOTLY_SIG_STRIP = 0.035
PLOTLY_EMBEDDING_ORDER = {name: idx for idx, name in enumerate(AN_EMBEDDING_NAMES)}

AXIS_LABELS = {
    'cosine': 'Cosine Similarity',
    'word': 'Word Acc (%)',
    'word_loose': 'Word Loose Acc (%)',
    'cat': 'Cat Acc (%)',
    'cat_loose': 'Cat Loose Acc (%)',
}

PANEL_TITLES = {
    'cosine': 'Cosine Similarity',
    'word': 'Word Accuracy',
    'word_loose': 'Word Loose Accuracy',
    'cat': 'Category Accuracy',
    'cat_loose': 'Category Loose Accuracy',
}


# ─── Per-patient actual window helpers ────────────────────────────────────────

_LOG_PATIENT_RE = re.compile(r'Patient \d+/\d+:\s+(\S+)')
_LOG_WINDOW_RE  = re.compile(r'Global window: back=\d+ samp \((\d+\.\d+)s\)')


def _parse_log_back_sec(log_path, patient):
    """Scan a run log for this patient's actual back window in seconds."""
    current = None
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as fh:
            for line in fh:
                m = _LOG_PATIENT_RE.search(line)
                if m:
                    current = m.group(1)
                if current == patient:
                    m2 = _LOG_WINDOW_RE.search(line)
                    if m2:
                        return float(m2.group(1))
    except Exception:
        pass
    return None


def _load_actual_back_sec(run_dir, patient, meta):
    """
    Return actual_back_sec for a patient.
    Priority: PKL 'actual_back_sec' field → log file parsing → None.
    """
    pkl_path = os.path.join(run_dir, patient, 'semantic_regression_results.pkl')
    if os.path.exists(pkl_path):
        try:
            try:
                import dill as _dill
            except ImportError:
                import pickle as _dill
            with open(pkl_path, 'rb') as fh:
                d = _dill.load(fh)
            val = d.get('actual_back_sec')
            if val is not None:
                return float(val)
        except Exception:
            pass
    if meta:
        log_path = meta.get('log_path')
        if log_path and os.path.exists(str(log_path)):
            val = _parse_log_back_sec(str(log_path), patient)
            if val is not None:
                return val
    return None


# ─── Cue timing loader ─────────────────────────────────────────────────────────

# Map from rel_cues keys (semantic_regression.py) to CUE_STYLES keys (report).
_REL_CUES_TO_REPORT_KEY = {
    'aud_stim_onset':  'aud_stim_onset',
    'aud_stim_offset': 'aud_stim_offset',
    'go_cue':          'go_cue_onset',   # name differs between processing and report
    'voice_onset':     'voice_onset',
    'voice_offset':    'voice_offset',
}


def _load_cue_info_from_stats_json(run_dir, patient):
    """
    Load pre-computed (post-warp) cue stats from cue_stats.json.

    Returns dict compatible with _load_cue_timings:
        {cue_name: {'mean_s': float, 'std_s': float}}
    or None if the file is absent, unreadable, or empty.
    """
    json_path = os.path.join(run_dir, patient, 'cue_stats.json')
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
    except Exception:
        return None

    rel_cues = data.get('rel_cues')
    if not rel_cues:
        return None

    result = {}
    for src_key, dst_key in _REL_CUES_TO_REPORT_KEY.items():
        if dst_key not in CUE_STYLES:
            continue
        entry = rel_cues.get(src_key)
        if entry is None:
            continue
        # Support both 'mean'/'std' (from semantic_regression.py) and 'mean_s'/'std_s'
        mean_v = entry.get('mean', entry.get('mean_s'))
        std_v  = entry.get('std',  entry.get('std_s'))
        if mean_v is None or std_v is None:
            continue
        try:
            m, s = float(mean_v), float(std_v)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(m) and np.isfinite(s)):
            continue
        result[dst_key] = {'mean_s': m, 'std_s': s}

    return result if result else None


def _load_cue_timings(patient, data_dir, align_cue='none', ref_bin_s=0.0, warp='none',
                      target_sec=None):
    """
    Load per-trial cue timings from the patient's auditory naming DataFrame PKL.

    Returns
    -------
    dict[str, dict] or None
        Keys: cue names ('aud_stim_onset', 'aud_stim_offset', 'go_cue_onset',
              'voice_onset', 'voice_offset').
        Values: {'mean_s': float, 'std_s': float} relative to the alignment
                reference event (in seconds).  Reference = aud_stim_onset when align_cue='none'.
        Returns None if the data file is not found.
    """
    try:
        import dill
    except ImportError:
        import pickle as dill

    df_path = os.path.join(data_dir, patient, f'{patient}_auditory_naming_df.pkl')
    if not os.path.exists(df_path):
        print(f"  [cues] {patient}: data PKL not found ({df_path})", flush=True)
        return None

    try:
        with open(df_path, 'rb') as f:
            trial_df = dill.load(f)
    except Exception as e:
        print(f"  [cues] {patient}: failed to load df ({e})", flush=True)
        return None

    # Keep good trials (bad_trials == True means keep, matching semantic_regression.py convention)
    if 'bad_trials' in trial_df.columns:
        trial_df = trial_df[trial_df['bad_trials'].astype(bool)].reset_index(drop=True)

    def _col(df, *names):
        for n in names:
            if n in df.columns:
                return df[n].values.astype(float)
        return np.full(len(df), np.nan)

    def _first(v):
        try:
            a = np.asarray(v, dtype=float).ravel()
            return float(a[0]) if len(a) > 0 else np.nan
        except Exception:
            return np.nan

    def _last(v):
        try:
            a = np.asarray(v, dtype=float).ravel()
            return float(a[-1]) if len(a) > 0 else np.nan
        except Exception:
            return np.nan

    # Derive cue arrays (in seconds, absolute)
    if 'prompt_word_onsets' in trial_df.columns:
        aud_stim_onset  = np.array([_first(v) for v in trial_df['prompt_word_onsets']])
        aud_stim_offset = np.array([_last(v)  for v in trial_df['prompt_word_offsets']])
    else:
        aud_stim_onset  = _col(trial_df, 'aud_stim_onset', 'stimulus_onset')
        aud_stim_offset = _col(trial_df, 'aud_stim_offset', 'stimulus_offset')

    go_cue_onset   = _col(trial_df, 'go_cue_onset', 'green_screen_onset')
    voice_onset    = _col(trial_df, 'voice_onset')
    voice_offset   = _col(trial_df, 'voice_offset')
    trial_onset    = _col(trial_df, 'trial_onset')

    # ── Apply linear time warp to cue times if requested ─────────────────────
    # Mirrors semantic_regression._linear_time_warp / _warp_cue (single segment):
    #   'stim'  warps [aud_stim_onset → aud_stim_offset] to the common stim duration.
    #   'voice' warps [aud_stim_onset → voice_onset]     to the common onset→voice duration.
    # Prefers the pipeline's group target (target_sec from meta) over this patient's own
    # median, so the drawn cue lines match the warped data.  ('linear' is a legacy alias
    # for 'stim' — normalized to 'stim' by the caller.)
    if warp in ('stim', 'voice'):
        seg_start = np.asarray(aud_stim_onset, dtype=float)
        seg_end   = np.asarray(voice_onset if warp == 'voice' else aud_stim_offset,
                               dtype=float)
        seg_durs  = seg_end - seg_start
        valid_durs = seg_durs[np.isfinite(seg_durs) & (seg_durs > 0)]
        if len(valid_durs) > 0:
            median_seg_s = float(target_sec) if target_sec else float(np.median(valid_durs))

            def _warp_arr(cue_arr):
                out = np.empty_like(cue_arr, dtype=float)
                for _i, _t in enumerate(cue_arr):
                    _s, _e = seg_start[_i], seg_end[_i]
                    if not (np.isfinite(_t) and np.isfinite(_s) and np.isfinite(_e)
                            and _e > _s):
                        out[_i] = _t
                    elif _t < _s:                                 # pre-segment: unchanged
                        out[_i] = _t
                    elif _t <= _e:                                # within segment: proportional
                        out[_i] = _s + (_t - _s) / (_e - _s) * median_seg_s
                    else:                                         # past segment: rigid shift
                        out[_i] = _t + (median_seg_s - (_e - _s))
                return out

            # seg_start/seg_end are snapshots of the raw arrays, so reassignment order
            # below does not corrupt the breakpoints. aud_stim_onset stays identity.
            aud_stim_offset = _warp_arr(aud_stim_offset)
            go_cue_onset    = _warp_arr(go_cue_onset)
            voice_offset    = _warp_arr(voice_offset)
            voice_onset     = _warp_arr(voice_onset)

    # Reference point for expressing cues relative to alignment.
    # For 'none' runs the data starts at trial_onset; t=0 in the plot is
    # ref_bin_s seconds into the trial (i.e. trial_onset + ref_bin_s).
    cue_arrays = {
        'none':            trial_onset + ref_bin_s,
        'trial_onset':     trial_onset,
        'go_cue':          go_cue_onset,
        'aud_stim_onset':  aud_stim_onset,
        'aud_stim_offset': aud_stim_offset,
        'voice_onset':     voice_onset,
        'voice_offset':    voice_offset,
    }
    ref_arr = cue_arrays.get(align_cue, aud_stim_onset)

    result = {}
    targets = {
        'aud_stim_onset':  aud_stim_onset,
        'aud_stim_offset': aud_stim_offset,
        'go_cue_onset':    go_cue_onset,
        'voice_onset':     voice_onset,
        'voice_offset':    voice_offset,
    }
    for name, arr in targets.items():
        diff_s = (arr - ref_arr)             # keep in seconds
        valid = diff_s[np.isfinite(diff_s) & np.isfinite(ref_arr)]
        if len(valid) == 0:
            continue
        result[name] = {
            'mean_s': float(np.mean(valid)),
            'std_s':  float(np.std(valid)),
        }
    return result


# ─── Per-epoch null array loaders ─────────────────────────────────────────────


def _load_loose_epoch_arrays(patient_dir):
    """
    Load per-epoch mean loose accuracy per bin from top1_decoding_source_data.csv.

    Returns dict[emb] = {'word_loose_obs': (n_epochs, n_bins) float32,
                          'cat_loose_obs':  (n_epochs, n_bins) float32}
    or None if the CSV is absent or unreadable.
    """
    csv_path = os.path.join(patient_dir, 'top1_decoding_source_data.csv')
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, usecols=[
            'embedding', 'epoch', 'bin_index',
            'word_correct_loose', 'category_correct_loose',
        ])
    except Exception:
        return None

    records = {}
    for emb in AN_EMBEDDING_NAMES:
        sub = df[df['embedding'] == emb]
        if len(sub) == 0:
            continue
        try:
            wlo = (sub.groupby(['epoch', 'bin_index'])['word_correct_loose']
                   .mean().unstack('bin_index').sort_index(axis=1))
            clo = (sub.groupby(['epoch', 'bin_index'])['category_correct_loose']
                   .mean().unstack('bin_index').sort_index(axis=1))
            records[emb] = {
                'word_loose_obs': wlo.values.astype(np.float32),
                'cat_loose_obs':  clo.values.astype(np.float32),
            }
        except Exception:
            pass
    return records if records else None


def _load_null_arrays(pkl_path, patient_dir=None):
    """
    Load per-epoch observed and null arrays from a patient's PKL.
    If patient_dir is given, also loads per-epoch loose accuracy from
    top1_decoding_source_data.csv for Wilcoxon vs verbatim null.

    Returns
    -------
    dict[str, dict] or None
        Keys: embedding names.
        Values: {
            'cat_obs':        (n_epochs, n_bins),
            'cat_null':       (n_epochs, n_bins),
            'word_obs':       (n_epochs, n_bins),
            'word_null':      (n_epochs, n_bins),
            'cosine':         (n_epochs, n_bins),  # may be absent
            'word_loose_obs': (n_epochs, n_bins),  # if CSV available
            'cat_loose_obs':  (n_epochs, n_bins),  # if CSV available
        }
    """
    data = load_pkl_raw(pkl_path)
    if data is None:
        return None

    records = {}
    for emb in AN_EMBEDDING_NAMES:
        if emb not in data.get('regressors', {}):
            continue
        br = data['regressors'][emb]
        rec = {
            'cat_obs':   np.array(br.all_retrieval_category_balanced_acc),
            'cat_null':  np.array(br.all_retrieval_category_chance_balanced_acc),
            'word_obs':  np.array(br.all_retrieval_word_balanced_acc),
            'word_null': np.array(br.all_retrieval_chance_word_balanced_acc),
        }
        if hasattr(br, 'all_cosine_sim'):
            rec['cosine'] = np.array(br.all_cosine_sim)
        records[emb] = rec

    # Also load per-epoch loose accuracy (for Wilcoxon vs verbatim null)
    if patient_dir is not None:
        loose = _load_loose_epoch_arrays(patient_dir)
        if loose:
            for emb in records:
                if emb in loose:
                    records[emb].update(loose[emb])

    return records


# ─── Per-bin significance ──────────────────────────────────────────────────────

def _compute_perbin_sig(run_dir, patients, n_bins, n_emb, patient_ref_bins=None, sig_alpha=SIG_ALPHA_DISPLAY):
    """
    Compute per-bin significance for each patient × embedding.

    Bonferroni correction: n_bins × n_emb × n_patients tests.

    Returns
    -------
    perbin_sig : dict[patient][embedding] →
        {
          'cosine':       bool[n_bins],  # pre-onset threshold
          'word_verb':    bool[n_bins],  # Wilcoxon obs vs null
          'word_loose':   bool[n_bins],  # pre-onset threshold (no per-epoch loose null)
          'cat_verb':     bool[n_bins],  # Wilcoxon obs vs null
          'cat_loose':    bool[n_bins],  # pre-onset threshold
        }
    pkl_failed : list[str]
    """
    # Bonferroni: correct only for number of time bins (per plot)
    perbin_sig = {}
    pkl_failed = []

    for patient in patients:
        pkl_path = os.path.join(run_dir, patient, 'semantic_regression_results.pkl')
        csv_path = os.path.join(run_dir, patient, 'per_time_scores.csv')

        if not os.path.exists(csv_path):
            continue

        df_csv = pd.read_csv(csv_path)
        n_b = int(df_csv['bin_index'].max()) + 1
        alpha_corr = sig_alpha / max(n_b, 1)   # Bonferroni: n_bins per plot

        pkl_data = None
        if os.path.exists(pkl_path):
            try:
                pkl_data = _load_null_arrays(pkl_path,
                               patient_dir=os.path.join(run_dir, patient))
            except Exception as e:
                print(f"  [sig] {patient}: PKL failed ({e})", flush=True)
                pkl_failed.append(patient)

        p_ref = (patient_ref_bins or {}).get(patient, 10)
        perbin_sig[patient] = {}

        for emb in AN_EMBEDDING_NAMES:
            sub = _embedding_bin_sorted(df_csv, emb)
            if len(sub) == 0:
                continue

            # ── cosine: pre-onset threshold ──────────────────────────────────
            cos_sig = _presonset_sig(sub['cosine_mean'].values if 'cosine_mean' in sub.columns else np.full(n_b, np.nan), n_pre_bins=p_ref)

            # ── verbatim + loose: Wilcoxon vs shuffled null (PKL) ────────────
            if pkl_data is not None and emb in pkl_data:
                wrd_obs  = pkl_data[emb]['word_obs'].astype(np.float32)
                wrd_null = pkl_data[emb]['word_null'].astype(np.float32)
                cat_obs  = pkl_data[emb]['cat_obs'].astype(np.float32)
                cat_null = pkl_data[emb]['cat_null'].astype(np.float32)
                wlo = pkl_data[emb].get('word_loose_obs')   # (n_ep, n_bins) or None
                clo = pkl_data[emb].get('cat_loose_obs')

                nb = wrd_obs.shape[1]
                wv_sig = np.zeros(nb, dtype=bool)
                cv_sig = np.zeros(nb, dtype=bool)
                wl_sig = np.zeros(nb, dtype=bool)
                cl_sig = np.zeros(nb, dtype=bool)

                for b in range(nb):
                    # verbatim: Wilcoxon obs vs null
                    for sig_arr, obs_b, null_b in [
                        (wv_sig, wrd_obs[:, b], wrd_null[:, b]),
                        (cv_sig, cat_obs[:, b], cat_null[:, b]),
                    ]:
                        d = obs_b - null_b
                        if np.any(d != 0):
                            try:
                                sig_arr[b] = bool(
                                    _wilcoxon_pvalue(d) < alpha_corr
                                )
                            except Exception:
                                pass

                    # loose: Wilcoxon loose_obs vs verbatim null when available;
                    # else compare loose CSV mean vs 95th percentile of null
                    for sig_arr, lo_arr, null_b, loose_col in [
                        (wl_sig, wlo,
                         wrd_null[:, b] if b < wrd_null.shape[1] else None,
                         'word_loose_acc'),
                        (cl_sig, clo,
                         cat_null[:, b] if b < cat_null.shape[1] else None,
                         'category_loose_acc'),
                    ]:
                        if null_b is None:
                            continue
                        if lo_arr is not None and b < lo_arr.shape[1]:
                            n_ep = min(lo_arr.shape[0], len(null_b))
                            d = lo_arr[:n_ep, b] - null_b[:n_ep]
                            if np.any(d != 0):
                                try:
                                    sig_arr[b] = bool(
                                        _wilcoxon_pvalue(d) < alpha_corr
                                    )
                                except Exception:
                                    pass
                        else:
                            # Fallback: loose mean vs 95th pct of verbatim null
                            if loose_col in sub.columns and b < len(sub):
                                sig_arr[b] = bool(
                                    float(sub[loose_col].values[b]) > np.percentile(null_b, 95)
                                )
            else:
                # No PKL: pre-onset threshold for all metrics
                wv_sig = _presonset_sig(sub['word_balanced_acc'].values     if 'word_balanced_acc'     in sub.columns else np.full(n_b, np.nan), n_pre_bins=p_ref)
                cv_sig = _presonset_sig(sub['category_balanced_acc'].values if 'category_balanced_acc' in sub.columns else np.full(n_b, np.nan), n_pre_bins=p_ref)
                wl_sig = _presonset_sig(sub['word_loose_acc'].values        if 'word_loose_acc'        in sub.columns else np.full(n_b, np.nan), n_pre_bins=p_ref)
                cl_sig = _presonset_sig(sub['category_loose_acc'].values    if 'category_loose_acc'    in sub.columns else np.full(n_b, np.nan), n_pre_bins=p_ref)

            perbin_sig[patient][emb] = {
                'cosine':     cos_sig,
                'word_verb':  wv_sig,
                'word_loose': wl_sig,
                'cat_verb':   cv_sig,
                'cat_loose':  cl_sig,
            }

    return perbin_sig, pkl_failed


def _presonset_sig(vals, n_pre_bins=10):
    """Return bool mask: value > pre-onset mean + 1 SEM."""
    arr   = np.asarray(vals, dtype=np.float32)
    pre   = arr[:n_pre_bins]
    valid = pre[~np.isnan(pre)]
    if len(valid) == 0:
        return np.zeros(len(arr), dtype=bool)
    mu  = float(np.mean(valid))
    sem = float(np.std(valid) / max(np.sqrt(len(valid)), 1))
    return arr > (mu + sem)


def _wilcoxon_pvalue(diff):
    """Return a float p-value from scipy.stats.wilcoxon across SciPy versions."""
    result = _scipy_stats.wilcoxon(diff, alternative='greater')
    pvalue = getattr(result, 'pvalue', None)
    if pvalue is not None:
        return float(pvalue)
    arr = np.asarray(result, dtype=np.float64).ravel()
    return float(arr[-1]) if len(arr) else float('nan')


def _embedding_bin_sorted(df, emb):
    """Return one embedding's rows ordered by bin index without pandas sort_values."""
    sub = df[df['embedding'] == emb].copy()
    if len(sub) == 0 or 'bin_index' not in sub.columns:
        return sub.reset_index(drop=True)
    order = np.argsort(sub['bin_index'].to_numpy(dtype=np.float64, copy=False), kind='mergesort')
    return sub.iloc[order].reset_index(drop=True)


def _embedding_patient_bin_sorted(df, emb):
    """Return one embedding's rows ordered by patient then bin index."""
    sub = df[df['embedding'] == emb].copy()
    if len(sub) == 0:
        return sub.reset_index(drop=True)
    if 'patient' not in sub.columns or 'bin_index' not in sub.columns:
        return sub.reset_index(drop=True)
    patient_keys = sub['patient'].astype(str).to_numpy()
    bin_keys = sub['bin_index'].to_numpy(dtype=np.float64, copy=False)
    order = np.lexsort((bin_keys, patient_keys))
    return sub.iloc[order].reset_index(drop=True)


# ─── Figure helpers ────────────────────────────────────────────────────────────

def _mark_sig_bins(ax, time_ms, sig_mask, color, row=0, n_rows=4):
    """Draw short colored tick marks at top edge of ax for significant bins."""
    strip = 0.045 / max(n_rows, 1)
    ymax  = 1.0 - row * strip
    ymin  = ymax - strip
    for b, is_sig in enumerate(sig_mask):
        if is_sig and b < len(time_ms):
            ax.axvline(time_ms[b], ymin=ymin, ymax=ymax,
                       color=color, lw=1.8, alpha=0.9, zorder=5)


def _draw_cues(ax, cue_info, alpha_line=0.6, alpha_fill=0.10):
    """Draw vertical cue lines with ±1 std shading on an axis."""
    if not cue_info:
        return
    for cue_name, cinfo in cue_info.items():
        st = CUE_STYLES.get(cue_name, {'color': '#777777', 'ls': '-', 'lw': 1.0, 'label': cue_name})
        mu  = cinfo['mean_s']
        std = cinfo['std_s']
        ymin_ax, ymax_ax = ax.get_ylim()
        ax.axvline(mu, color=st['color'], ls=st['ls'], lw=st['lw'], alpha=alpha_line, zorder=4)
        ax.axvspan(mu - std, mu + std, color=st['color'], alpha=alpha_fill, zorder=3)


def _null_band(ax, time_ms, null_arr, color, pct=95):
    """Draw null mean ± CI band. null_arr shape: (n_epochs, n_bins) or (n_bins,)."""
    null = np.asarray(null_arr, dtype=np.float32)
    if null.ndim == 2:
        null_mean = null.mean(axis=0)
        # Bootstrap-style CI: mean ± z * std / sqrt(n)
        z = _scipy_stats.norm.ppf((1 + pct / 100) / 2)
        null_sem  = null.std(axis=0) / max(np.sqrt(null.shape[0]), 1)
        lo = null_mean - z * null_sem
        hi = null_mean + z * null_sem
    else:
        null_mean = null
        lo = hi = null

    ax.axhline(null_mean.mean(), color=color, lw=0.8, ls=':', alpha=0.5)
    if not np.allclose(lo, hi):
        ax.fill_between(time_ms, lo, hi, color=color, alpha=0.08)


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _safe_html_id(*parts):
    """Build a stable DOM id from arbitrary text fragments."""
    joined = '_'.join(str(p) for p in parts if p is not None)
    return re.sub(r'[^0-9A-Za-z_]+', '_', joined).strip('_') or 'plot'


def _plotly_json(value):
    """Serialize numpy-heavy payloads into browser-safe JSON."""
    def _default(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')

    return json.dumps(value, default=_default)


def _rgba(hex_color, alpha):
    """Convert #RRGGBB to rgba(r, g, b, alpha)."""
    color = str(hex_color).lstrip('#')
    if len(color) != 6:
        return hex_color
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'


def _series_values(sub, name, n_bins, scale=1.0):
    """Return a fixed-length float array for a plot column."""
    if name not in sub.columns:
        return np.full(n_bins, np.nan, dtype=np.float32)
    vals = sub[name].values.astype(np.float32)
    if len(vals) != n_bins:
        out = np.full(n_bins, np.nan, dtype=np.float32)
        out[:min(len(vals), n_bins)] = vals[:min(len(vals), n_bins)]
        vals = out
    return vals * scale


def _null_summary(null_arr, scale=1.0, pct=95):
    """Return null mean and CI arrays for Plotly rendering."""
    null = np.asarray(null_arr, dtype=np.float32)
    if null.ndim == 2:
        mean = null.mean(axis=0)
        z = _scipy_stats.norm.ppf((1 + pct / 100) / 2)
        sem = null.std(axis=0) / max(np.sqrt(null.shape[0]), 1)
        lo = mean - z * sem
        hi = mean + z * sem
    else:
        mean = null
        lo = null
        hi = null
    return mean * scale, lo * scale, hi * scale


def _sig_segment_trace(time_s, sig_mask, color, y0, y1, xaxis, yaxis, name):
    """Return a Plotly scatter trace for significance tick segments."""
    x_vals = []
    y_vals = []
    for idx, is_sig in enumerate(np.asarray(sig_mask, dtype=bool)):
        if is_sig and idx < len(time_s):
            x = float(time_s[idx])
            x_vals.extend([x, x, None])
            y_vals.extend([float(y0), float(y1), None])
    return {
        'type': 'scatter',
        'mode': 'lines',
        'x': x_vals,
        'y': y_vals,
        'xaxis': xaxis,
        'yaxis': yaxis,
        'line': {'color': color, 'width': 2},
        'name': name,
        'hoverinfo': 'skip',
        'showlegend': False,
        'visible': True,
    }


def _embedding_toggle_html(div_id):
    """Return a shared embedding-toggle toolbar for a Plotly figure."""
    chips = []
    for emb in AN_EMBEDDING_NAMES:
        color = EMB_COLORS.get(emb, '#455A64')
        chips.append(
            f'<button type="button" class="embedding-toggle active" '
            f'data-plot="{div_id}" data-embedding="{emb}" '
            f'style="--emb-color: {color}">{emb}</button>'
        )
    return (
        '<div class="embedding-toggle-bar">'
        '<span class="embedding-toggle-label">Embeddings:</span>'
        + ''.join(chips) +
        '</div>'
    )


def _x_axis_dict(domain, x_lo, x_hi, anchor, show_tick_labels=True, title=None):
    """Return a Plotly x-axis config dict with integer-second ticks."""
    tick0 = int(np.floor(x_lo))
    d = {
        'domain': domain,
        'range': [x_lo, x_hi],
        'anchor': anchor,
        'gridcolor': '#E0E6ED',
        'zeroline': False,
        'tickmode': 'linear',
        'tick0': tick0,
        'dtick': 1,
        'tickformat': '.0f',
        'showticklabels': show_tick_labels,
    }
    if title:
        d['title'] = title
    return d


def _panel_range(arrays, floor=None, ceiling=None, pad_frac=0.12,
                 min_span=1.0, extra_top=0.0):
    """Infer a padded axis range from a list of arrays."""
    finite = []
    for arr in arrays:
        if arr is None:
            continue
        vals = np.asarray(arr, dtype=np.float32).ravel()
        vals = vals[np.isfinite(vals)]
        if len(vals):
            finite.append(vals)

    if finite:
        cat = np.concatenate(finite)
        lo = float(np.min(cat))
        hi = float(np.max(cat))
    else:
        lo = float(floor) if floor is not None else 0.0
        hi = float(ceiling) if ceiling is not None else lo + min_span

    if floor is not None:
        lo = min(lo, float(floor))
    if ceiling is not None:
        hi = max(hi, float(ceiling))

    span = max(hi - lo, float(min_span))
    lo_pad = lo - span * pad_frac
    hi_pad = hi + span * (pad_frac + extra_top)
    if floor is not None:
        lo_pad = max(float(floor), lo_pad)
    return [float(lo_pad), float(hi_pad)]


def _band_traces(time_s, lo, hi, color, xaxis, yaxis, alpha, name):
    """Return Plotly traces for a shaded band between lo and hi."""
    lo_arr = np.asarray(lo, dtype=np.float32)
    hi_arr = np.asarray(hi, dtype=np.float32)
    if len(lo_arr) == 0 or len(hi_arr) == 0:
        return []
    if not (np.isfinite(lo_arr).any() and np.isfinite(hi_arr).any()):
        return []
    clear = 'rgba(0,0,0,0)'
    return [
        {
            'type': 'scatter',
            'mode': 'lines',
            'x': time_s,
            'y': hi_arr,
            'xaxis': xaxis,
            'yaxis': yaxis,
            'line': {'color': clear, 'width': 0},
            'hoverinfo': 'skip',
            'showlegend': False,
            'name': f'{name} upper',
        },
        {
            'type': 'scatter',
            'mode': 'lines',
            'x': time_s,
            'y': lo_arr,
            'xaxis': xaxis,
            'yaxis': yaxis,
            'line': {'color': clear, 'width': 0},
            'fill': 'tonexty',
            'fillcolor': _rgba(color, alpha),
            'hoverinfo': 'skip',
            'showlegend': False,
            'name': f'{name} band',
        },
    ]


def _line_trace(time_s, values, color, xaxis, yaxis, name,
                dash='solid', width=2.0, opacity=1.0, hovertemplate=None):
    """Return a Plotly line trace."""
    return {
        'type': 'scatter',
        'mode': 'lines',
        'x': time_s,
        'y': np.asarray(values, dtype=np.float32),
        'xaxis': xaxis,
        'yaxis': yaxis,
        'line': {'color': color, 'width': width, 'dash': dash},
        'name': name,
        'showlegend': False,
        'opacity': opacity,
        'hovertemplate': hovertemplate or '%{y}<extra></extra>',
    }


def _domain_axis_ref(axis_ref):
    """Map Plotly axis refs to domain refs for shapes."""
    if axis_ref == 'y':
        return 'y domain'
    if axis_ref.startswith('y'):
        return f'y{axis_ref[1:]} domain'
    if axis_ref == 'x':
        return 'x domain'
    if axis_ref.startswith('x'):
        return f'x{axis_ref[1:]} domain'
    return axis_ref


def _cue_shapes(cue_info, panel_axes, line_alpha=0.65, fill_alpha=0.07):
    """Return Plotly shapes for cue means and std envelopes on each panel."""
    shapes = []
    if not cue_info:
        return shapes
    for axes in panel_axes.values():
        xref = axes['x']
        yref = _domain_axis_ref(axes['y'])
        for cue_name, cinfo in cue_info.items():
            st = CUE_STYLES.get(cue_name, {'color': '#777777', 'ls': '-', 'lw': 1.0})
            mu = float(cinfo['mean_s'])
            std = float(cinfo['std_s'])
            shapes.append({
                'type': 'rect',
                'xref': xref,
                'yref': yref,
                'x0': mu - std,
                'x1': mu + std,
                'y0': 0,
                'y1': 1,
                'fillcolor': _rgba(st['color'], fill_alpha),
                'line': {'width': 0},
                'layer': 'below',
            })
            shapes.append({
                'type': 'line',
                'xref': xref,
                'yref': yref,
                'x0': mu,
                'x1': mu,
                'y0': 0,
                'y1': 1,
                'line': {
                    'color': _rgba(st['color'], line_alpha),
                    'width': st.get('lw', 1.0),
                    'dash': 'dash' if st.get('ls') == '--' else 'solid',
                },
                'layer': 'above',
            })
    return shapes


def _reference_shapes(panel_axes, x_value=0.0):
    """Return alignment-reference lines for each panel."""
    shapes = []
    for axes in panel_axes.values():
        shapes.append({
            'type': 'line',
            'xref': axes['x'],
            'yref': _domain_axis_ref(axes['y']),
            'x0': x_value,
            'x1': x_value,
            'y0': 0,
            'y1': 1,
            'line': {'color': 'rgba(0,0,0,0.35)', 'width': 1, 'dash': 'dot'},
            'layer': 'above',
        })
    return shapes


def _interactive_plot_html(div_id, traces, layout, trace_groups,
                           note=None, min_height=860):
    """Render a Plotly plot with per-embedding visibility toggles."""
    payload = _plotly_json({
        'traces': traces,
        'layout': layout,
        'trace_groups': trace_groups,
    })
    note_html = f'<p class="plotly-note">{note}</p>' if note else ''
    return f"""
{_embedding_toggle_html(div_id)}
<div id="{div_id}" class="plotly-an" style="min-height:{int(min_height)}px"></div>
{note_html}
<script>
(function() {{
    var payload = {payload};
    var div = document.getElementById('{div_id}');
    if (!div || typeof Plotly === 'undefined') {{ return; }}

    var buttons = document.querySelectorAll('.embedding-toggle[data-plot="{div_id}"]');
    var state = {{}};
    Object.keys(payload.trace_groups).forEach(function(emb) {{ state[emb] = true; }});

    function syncButtons() {{
        buttons.forEach(function(btn) {{
            var emb = btn.getAttribute('data-embedding');
            var on = !!state[emb];
            btn.classList.toggle('active', on);
            btn.classList.toggle('inactive', !on);
        }});
    }}

    function applyEmbedding(emb) {{
        var indices = payload.trace_groups[emb] || [];
        if (!indices.length) {{ return; }}
        var visible = indices.map(function() {{ return state[emb]; }});
        Plotly.restyle(div, {{visible: visible}}, indices);
    }}

    Plotly.newPlot(div, payload.traces, payload.layout, {{
        responsive: true,
        displaylogo: false
    }});

    buttons.forEach(function(btn) {{
        btn.addEventListener('click', function() {{
            var emb = btn.getAttribute('data-embedding');
            state[emb] = !state[emb];
            applyEmbedding(emb);
            syncButtons();
        }});
    }});

    syncButtons();
}})();
</script>
"""


# ─── Per-patient figure (2-column layout) ────────────────────────────────────

def make_figure(patient, run_dir, ref_bin, bin_size_ms,
                cue_info=None, sig_bins=None, pkl_data=None):
    """
    Interactive per-patient time-series figure with embedding toggles.

    Cosine occupies the left half of the top row. Accuracy panels fill the
    lower two rows. Each embedding toggle hides/shows its observed lines,
    shuffle/null reference, and significance overlays together.
    """
    csv_path = os.path.join(run_dir, patient, 'per_time_scores.csv')
    df = pd.read_csv(csv_path)

    n_bins = int(df['bin_index'].max()) + 1
    time_s = np.array(
        [(b - ref_bin) * bin_size_ms / 1000.0 for b in range(n_bins)],
        dtype=np.float32,
    )

    panel_axes = {
        'cosine': {'x': 'x',  'y': 'y'},
        'word': {'x': 'x2', 'y': 'y2'},
        'word_loose': {'x': 'x3', 'y': 'y3'},
        'cat': {'x': 'x4', 'y': 'y4'},
        'cat_loose': {'x': 'x5', 'y': 'y5'},
    }
    panel_defs = [
        {'key': 'cosine', 'col': 'cosine_mean', 'scale': 1.0, 'std_col': 'cosine_std', 'null_key': 'cosine', 'sig_key': None},
        {'key': 'word', 'col': 'word_balanced_acc', 'scale': 100.0, 'std_col': None, 'null_key': 'word_null', 'sig_key': 'word_verb'},
        {'key': 'word_loose', 'col': 'word_loose_acc', 'scale': 100.0, 'std_col': None, 'null_key': 'word_null', 'sig_key': 'word_loose'},
        {'key': 'cat', 'col': 'category_balanced_acc', 'scale': 100.0, 'std_col': None, 'null_key': 'cat_null', 'sig_key': 'cat_verb'},
        {'key': 'cat_loose', 'col': 'category_loose_acc', 'scale': 100.0, 'std_col': None, 'null_key': 'cat_null', 'sig_key': 'cat_loose'},
    ]

    panel_arrays = {spec['key']: [] for spec in panel_defs}
    panel_payload = {}

    for emb in AN_EMBEDDING_NAMES:
        sub = _embedding_bin_sorted(df, emb)
        if len(sub) == 0:
            continue

        emb_payload = {}
        for spec in panel_defs:
            vals = _series_values(sub, spec['col'], n_bins, scale=spec['scale'])
            record = {'values': vals}
            panel_arrays[spec['key']].append(vals)

            if spec['std_col']:
                spread = _series_values(sub, spec['std_col'], n_bins)
                record['spread_lo'] = vals - spread
                record['spread_hi'] = vals + spread
                panel_arrays[spec['key']].extend([record['spread_lo'], record['spread_hi']])

            null_mean = null_lo = null_hi = None
            if pkl_data and emb in pkl_data and spec['null_key'] in pkl_data[emb]:
                null_mean, null_lo, null_hi = _null_summary(
                    pkl_data[emb][spec['null_key']],
                    scale=spec['scale'],
                )
            else:
                pre_vals = vals[:ref_bin]
                valid = pre_vals[np.isfinite(pre_vals)]
                if len(valid):
                    mu = float(np.mean(valid))
                    sem = float(np.std(valid) / max(np.sqrt(len(valid)), 1))
                    z = _scipy_stats.norm.ppf(0.975)
                    null_mean = np.full(n_bins, mu, dtype=np.float32)
                    null_lo = np.full(n_bins, mu - z * sem, dtype=np.float32)
                    null_hi = np.full(n_bins, mu + z * sem, dtype=np.float32)

            record['null_mean'] = null_mean
            record['null_lo'] = null_lo
            record['null_hi'] = null_hi
            if null_mean is not None:
                panel_arrays[spec['key']].extend([null_mean, null_lo, null_hi])

            if spec['sig_key'] is not None:
                sig_map = sig_bins.get(emb, {}) if sig_bins else {}
                record['sig_mask'] = np.asarray(sig_map.get(spec['sig_key'], np.zeros(n_bins, dtype=bool)), dtype=bool)

            emb_payload[spec['key']] = record

        panel_payload[emb] = emb_payload

    if not panel_payload:
        return '<p><em>No per-time data available.</em></p>'

    x_lo = float(time_s[0]) if len(time_s) else -1.0
    x_hi = float(time_s[-1]) if len(time_s) else 1.0
    if cue_info:
        for cinfo in cue_info.values():
            x_lo = min(x_lo, float(cinfo['mean_s']) - float(cinfo['std_s']))
            x_hi = max(x_hi, float(cinfo['mean_s']) + float(cinfo['std_s']))

    sig_headroom = PLOTLY_SIG_STRIP * (len(AN_EMBEDDING_NAMES) + 1)
    panel_ranges = {
        'cosine': _panel_range(panel_arrays['cosine'], min_span=0.08),
        'word': _panel_range(panel_arrays['word'], min_span=6.0, extra_top=sig_headroom),
        'word_loose': _panel_range(panel_arrays['word_loose'], min_span=6.0, extra_top=sig_headroom),
        'cat': _panel_range(panel_arrays['cat'], min_span=6.0, extra_top=sig_headroom),
        'cat_loose': _panel_range(panel_arrays['cat_loose'], min_span=6.0, extra_top=sig_headroom),
    }

    traces = []
    trace_groups = {emb: [] for emb in AN_EMBEDDING_NAMES}

    for emb in AN_EMBEDDING_NAMES:
        if emb not in panel_payload:
            continue
        color = EMB_COLORS.get(emb, '#455A64')
        row_idx = PLOTLY_EMBEDDING_ORDER.get(emb, 0)
        emb_indices = []

        cos = panel_payload[emb]['cosine']
        axes = panel_axes['cosine']
        for trace in _band_traces(time_s, cos.get('spread_lo'), cos.get('spread_hi'), color,
                                  axes['x'], axes['y'], 0.12, f'{emb} cosine spread'):
            emb_indices.append(len(traces))
            traces.append(trace)
        traces.append(_line_trace(
            time_s,
            cos['values'],
            color,
            axes['x'],
            axes['y'],
            f'{emb} cosine',
            width=2.2,
            hovertemplate=f'{emb}<br>t=%{{x:.2f}} s<br>cosine=%{{y:.3f}}<extra></extra>',
        ))
        emb_indices.append(len(traces) - 1)
        if cos.get('null_mean') is not None:
            for trace in _band_traces(time_s, cos['null_lo'], cos['null_hi'], color,
                                      axes['x'], axes['y'], PLOTLY_NULL_ALPHA,
                                      f'{emb} cosine null'):
                emb_indices.append(len(traces))
                traces.append(trace)
            traces.append(_line_trace(
                time_s,
                cos['null_mean'],
                _rgba(color, 0.9),
                axes['x'],
                axes['y'],
                f'{emb} cosine null mean',
                dash='dot',
                width=1.2,
                hovertemplate=f'{emb}<br>shuffle chance=%{{y:.3f}}<extra></extra>',
            ))
            emb_indices.append(len(traces) - 1)

        for panel_key in ['word', 'word_loose', 'cat', 'cat_loose']:
            panel = panel_payload[emb][panel_key]
            axes = panel_axes[panel_key]
            traces.append(_line_trace(
                time_s,
                panel['values'],
                color,
                axes['x'],
                axes['y'],
                f'{emb} {panel_key}',
                width=2.0,
                hovertemplate=f'{emb}<br>t=%{{x:.2f}} s<br>%{{y:.1f}}%<extra></extra>',
            ))
            emb_indices.append(len(traces) - 1)

            if panel.get('null_mean') is not None:
                for trace in _band_traces(time_s, panel['null_lo'], panel['null_hi'], color,
                                          axes['x'], axes['y'], PLOTLY_NULL_ALPHA,
                                          f'{emb} {panel_key} null'):
                    emb_indices.append(len(traces))
                    traces.append(trace)
                traces.append(_line_trace(
                    time_s,
                    panel['null_mean'],
                    _rgba(color, 0.9),
                    axes['x'],
                    axes['y'],
                    f'{emb} {panel_key} null mean',
                    dash='dot',
                    width=1.2,
                    hovertemplate=f'{emb}<br>shuffle chance=%{{y:.1f}}%<extra></extra>',
                ))
                emb_indices.append(len(traces) - 1)

            sig_mask = panel.get('sig_mask')
            if sig_mask is not None and len(sig_mask) == n_bins:
                y_lo, y_hi = panel_ranges[panel_key]
                span = max(y_hi - y_lo, 1.0)
                tick_top = y_hi - span * (0.02 + row_idx * PLOTLY_SIG_STRIP)
                tick_bottom = tick_top - span * (PLOTLY_SIG_STRIP * 0.7)
                traces.append(_sig_segment_trace(
                    time_s,
                    sig_mask,
                    color,
                    tick_bottom,
                    tick_top,
                    axes['x'],
                    axes['y'],
                    f'{emb} {panel_key} sig',
                ))
                emb_indices.append(len(traces) - 1)

        trace_groups[emb] = emb_indices

    layout = {
        'paper_bgcolor': 'white',
        'plot_bgcolor': 'white',
        'margin': {'l': 62, 'r': 24, 't': 56, 'b': 58},
        'hovermode': 'x unified',
        'showlegend': False,
        'annotations': [
            {'text': PANEL_TITLES['cosine'], 'x': 0.23, 'y': 1.05, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 12}},
            {'text': PANEL_TITLES['word'], 'x': 0.23, 'y': 0.69, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 12}},
            {'text': PANEL_TITLES['word_loose'], 'x': 0.77, 'y': 0.69, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 12}},
            {'text': PANEL_TITLES['cat'], 'x': 0.23, 'y': 0.31, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 12}},
            {'text': PANEL_TITLES['cat_loose'], 'x': 0.77, 'y': 0.31, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 12}},
        ],
        'shapes': _reference_shapes(panel_axes) + _cue_shapes(cue_info, panel_axes),
        'xaxis': _x_axis_dict([0.0, 0.46], x_lo, x_hi, 'y', show_tick_labels=False),
        'yaxis': {
            'domain': [0.74, 1.0], 'range': panel_ranges['cosine'], 'anchor': 'x',
            'title': AXIS_LABELS['cosine'], 'gridcolor': '#E0E6ED',
        },
        'xaxis2': _x_axis_dict([0.0, 0.46], x_lo, x_hi, 'y2', show_tick_labels=False),
        'yaxis2': {
            'domain': [0.37, 0.63], 'range': panel_ranges['word'], 'anchor': 'x2',
            'title': AXIS_LABELS['word'], 'gridcolor': '#E0E6ED',
        },
        'xaxis3': _x_axis_dict([0.54, 1.0], x_lo, x_hi, 'y3', show_tick_labels=False),
        'yaxis3': {
            'domain': [0.37, 0.63], 'range': panel_ranges['word_loose'], 'anchor': 'x3',
            'title': AXIS_LABELS['word_loose'], 'gridcolor': '#E0E6ED',
        },
        'xaxis4': _x_axis_dict([0.0, 0.46], x_lo, x_hi, 'y4', title='Time from alignment reference (s)'),
        'yaxis4': {
            'domain': [0.0, 0.26], 'range': panel_ranges['cat'], 'anchor': 'x4',
            'title': AXIS_LABELS['cat'], 'gridcolor': '#E0E6ED',
        },
        'xaxis5': _x_axis_dict([0.54, 1.0], x_lo, x_hi, 'y5', title='Time from alignment reference (s)'),
        'yaxis5': {
            'domain': [0.0, 0.26], 'range': panel_ranges['cat_loose'], 'anchor': 'x5',
            'title': AXIS_LABELS['cat_loose'], 'gridcolor': '#E0E6ED',
        },
    }

    div_id = _safe_html_id('auditory_naming', patient, 'main_plot')
    return _interactive_plot_html(
        div_id,
        traces,
        layout,
        trace_groups,
        note='Embedding buttons hide each embedding\'s observed curve, shuffle/null reference, and significance overlays together. Cosine significance markers are intentionally omitted.',
        min_height=930,
    )


# ─── Verbatim vs Loose comparison figure ──────────────────────────────────────

def make_comparison_figure(patient, run_dir, ref_bin, bin_size_ms):
    """
    Interactive 2-row figure: exact (solid) vs loose (dashed) per embedding.
    """
    csv_path = os.path.join(run_dir, patient, 'per_time_scores.csv')
    df = pd.read_csv(csv_path)

    n_bins  = int(df['bin_index'].max()) + 1
    time_s = np.array([(b - ref_bin) * bin_size_ms / 1000.0 for b in range(n_bins)], dtype=np.float32)
    x_lo = float(time_s[0]) if len(time_s) else -1.0
    x_hi = float(time_s[-1]) if len(time_s) else 1.0

    payload = {}
    word_arrays = []
    cat_arrays = []
    for emb in AN_EMBEDDING_NAMES:
        sub = _embedding_bin_sorted(df, emb)
        if len(sub) == 0:
            continue
        wv = _series_values(sub, 'word_balanced_acc', n_bins, scale=100.0)
        wl = _series_values(sub, 'word_loose_acc', n_bins, scale=100.0)
        cv = _series_values(sub, 'category_balanced_acc', n_bins, scale=100.0)
        cl = _series_values(sub, 'category_loose_acc', n_bins, scale=100.0)
        payload[emb] = {'word': wv, 'word_loose': wl, 'cat': cv, 'cat_loose': cl}
        word_arrays.extend([wv, wl])
        cat_arrays.extend([cv, cl])

    if not payload:
        return '<p><em>No comparison data available.</em></p>'

    traces = []
    trace_groups = {emb: [] for emb in AN_EMBEDDING_NAMES}
    for emb in AN_EMBEDDING_NAMES:
        if emb not in payload:
            continue
        color = EMB_COLORS.get(emb, '#455A64')
        emb_indices = []
        for panel_key, axis_ref, dash in [
            ('word', {'x': 'x', 'y': 'y'}, 'solid'),
            ('word_loose', {'x': 'x', 'y': 'y'}, 'dash'),
            ('cat', {'x': 'x2', 'y': 'y2'}, 'solid'),
            ('cat_loose', {'x': 'x2', 'y': 'y2'}, 'dash'),
        ]:
            traces.append(_line_trace(
                time_s,
                payload[emb][panel_key],
                color,
                axis_ref['x'],
                axis_ref['y'],
                f'{emb} {panel_key}',
                dash=dash,
                width=2.0,
                opacity=0.95 if dash == 'solid' else 0.85,
                hovertemplate=f'{emb}<br>t=%{{x:.2f}} s<br>%{{y:.1f}}%<extra></extra>',
            ))
            emb_indices.append(len(traces) - 1)
        trace_groups[emb] = emb_indices

    layout = {
        'paper_bgcolor': 'white',
        'plot_bgcolor': 'white',
        'margin': {'l': 62, 'r': 24, 't': 54, 'b': 54},
        'hovermode': 'x unified',
        'showlegend': False,
        'annotations': [
            {'text': 'Word Accuracy vs Loose', 'x': 0.5, 'y': 1.05, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 12}},
            {'text': 'Category Accuracy vs Loose', 'x': 0.5, 'y': 0.49, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 12}},
        ],
        'shapes': _reference_shapes({'word': {'x': 'x', 'y': 'y'}, 'cat': {'x': 'x2', 'y': 'y2'}}),
        'xaxis': _x_axis_dict([0.0, 1.0], x_lo, x_hi, 'y', show_tick_labels=False),
        'yaxis': {
            'domain': [0.57, 1.0],
            'range': _panel_range(word_arrays, min_span=6.0),
            'anchor': 'x', 'title': AXIS_LABELS['word'], 'gridcolor': '#E0E6ED',
        },
        'xaxis2': _x_axis_dict([0.0, 1.0], x_lo, x_hi, 'y2', title='Time from alignment reference (s)'),
        'yaxis2': {
            'domain': [0.0, 0.43],
            'range': _panel_range(cat_arrays, min_span=6.0),
            'anchor': 'x2', 'title': AXIS_LABELS['cat'], 'gridcolor': '#E0E6ED',
        },
    }

    div_id = _safe_html_id('auditory_naming', patient, 'comparison_plot')
    return _interactive_plot_html(
        div_id,
        traces,
        layout,
        trace_groups,
        note='Solid lines show exact accuracy. Dashed lines show loose accuracy.',
        min_height=620,
    )


# ─── Cross-patient group figure ────────────────────────────────────────────────

def make_group_figure(run_dir, patients, patient_ref_bins, bin_size_ms):
    """
    Interactive group-level average across patients (mean ± SEM).

    Each patient's bin_index is first converted to time_s using their own
    actual ref_bin so all patients share a common t=0 reference.
    """
    bin_s = bin_size_ms / 1000.0
    all_dfs = []
    for p in patients:
        csv_path = os.path.join(run_dir, p, 'per_time_scores.csv')
        if os.path.exists(csv_path):
            tmp = pd.read_csv(csv_path)
            rb = (patient_ref_bins or {}).get(p, 10)
            tmp['rel_bin'] = (tmp['bin_index'] - rb).astype(int)
            all_dfs.append(tmp)
    if not all_dfs:
        return None

    df_all = pd.concat(all_dfs, ignore_index=True)
    rel_bins = np.sort(df_all['rel_bin'].unique()).astype(int)
    time_s = rel_bins.astype(np.float32) * bin_s

    metric_defs = [
        {'key': 'cosine', 'col': 'cosine_mean', 'scale': 1.0, 'axis': {'x': 'x', 'y': 'y'}},
        {'key': 'word', 'col': 'word_balanced_acc', 'scale': 100.0, 'axis': {'x': 'x2', 'y': 'y2'}},
        {'key': 'word_loose', 'col': 'word_loose_acc', 'scale': 100.0, 'axis': {'x': 'x3', 'y': 'y3'}},
        {'key': 'cat', 'col': 'category_balanced_acc', 'scale': 100.0, 'axis': {'x': 'x4', 'y': 'y4'}},
    ]

    payload = {}
    panel_arrays = {spec['key']: [] for spec in metric_defs}
    for emb in AN_EMBEDDING_NAMES:
        sub_all = _embedding_patient_bin_sorted(df_all, emb)
        if len(sub_all) == 0:
            continue
        emb_payload = {}
        for spec in metric_defs:
            if spec['col'] not in sub_all.columns:
                continue
            grp = sub_all.groupby('rel_bin')[spec['col']]
            mu = grp.mean().reindex(rel_bins)
            sem = grp.sem().reindex(rel_bins)
            scale = spec['scale']
            vals = mu.values.astype(np.float32) * scale
            lo = (mu - sem).values.astype(np.float32) * scale
            hi = (mu + sem).values.astype(np.float32) * scale
            emb_payload[spec['key']] = {'values': vals, 'lo': lo, 'hi': hi}
            panel_arrays[spec['key']].extend([vals, lo, hi])
        payload[emb] = emb_payload

    if not payload:
        return None

    x_lo = float(time_s[0]) if len(time_s) else -1.0
    x_hi = float(time_s[-1]) if len(time_s) else 1.0
    traces = []
    trace_groups = {emb: [] for emb in AN_EMBEDDING_NAMES}
    for emb in AN_EMBEDDING_NAMES:
        emb_payload = payload.get(emb)
        if not emb_payload:
            continue
        color = EMB_COLORS.get(emb, '#455A64')
        emb_indices = []
        for spec in metric_defs:
            if spec['key'] not in emb_payload:
                continue
            axes = spec['axis']
            rec = emb_payload[spec['key']]
            for trace in _band_traces(time_s, rec['lo'], rec['hi'], color,
                                      axes['x'], axes['y'], 0.14,
                                      f'{emb} {spec["key"]} sem'):
                emb_indices.append(len(traces))
                traces.append(trace)
            hover = (
                f'{emb}<br>t=%{{x:.2f}} s<br>%{{y:.3f}}<extra></extra>'
                if spec['key'] == 'cosine'
                else f'{emb}<br>t=%{{x:.2f}} s<br>%{{y:.1f}}%<extra></extra>'
            )
            traces.append(_line_trace(
                time_s,
                rec['values'],
                color,
                axes['x'],
                axes['y'],
                f'{emb} {spec["key"]}',
                width=2.1,
                hovertemplate=hover,
            ))
            emb_indices.append(len(traces) - 1)
        trace_groups[emb] = emb_indices

    panel_axes = {spec['key']: spec['axis'] for spec in metric_defs}
    layout = {
        'paper_bgcolor': 'white',
        'plot_bgcolor': 'white',
        'margin': {'l': 64, 'r': 24, 't': 56, 'b': 56},
        'hovermode': 'x unified',
        'showlegend': False,
        'annotations': [
            {'text': PANEL_TITLES['cosine'], 'x': 0.23, 'y': 1.05, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 12}},
            {'text': PANEL_TITLES['word'], 'x': 0.5, 'y': 0.75, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 12}},
            {'text': PANEL_TITLES['word_loose'], 'x': 0.5, 'y': 0.49, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 12}},
            {'text': PANEL_TITLES['cat'], 'x': 0.5, 'y': 0.23, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 12}},
        ],
        'shapes': _reference_shapes(panel_axes),
        'xaxis': _x_axis_dict([0.0, 0.46], x_lo, x_hi, 'y', show_tick_labels=False),
        'yaxis': {
            'domain': [0.80, 1.0], 'range': _panel_range(panel_arrays['cosine'], min_span=0.08),
            'anchor': 'x', 'title': AXIS_LABELS['cosine'], 'gridcolor': '#E0E6ED',
        },
        'xaxis2': _x_axis_dict([0.0, 1.0], x_lo, x_hi, 'y2', show_tick_labels=False),
        'yaxis2': {
            'domain': [0.53, 0.73], 'range': _panel_range(panel_arrays['word'], min_span=6.0),
            'anchor': 'x2', 'title': AXIS_LABELS['word'], 'gridcolor': '#E0E6ED',
        },
        'xaxis3': _x_axis_dict([0.0, 1.0], x_lo, x_hi, 'y3', show_tick_labels=False),
        'yaxis3': {
            'domain': [0.26, 0.46], 'range': _panel_range(panel_arrays['word_loose'], min_span=6.0),
            'anchor': 'x3', 'title': AXIS_LABELS['word_loose'], 'gridcolor': '#E0E6ED',
        },
        'xaxis4': _x_axis_dict([0.0, 1.0], x_lo, x_hi, 'y4', title='Time from alignment reference (s)'),
        'yaxis4': {
            'domain': [0.0, 0.20], 'range': _panel_range(panel_arrays['cat'], min_span=6.0),
            'anchor': 'x4', 'title': AXIS_LABELS['cat'], 'gridcolor': '#E0E6ED',
        },
    }

    div_id = _safe_html_id('auditory_naming', 'group_plot')
    return _interactive_plot_html(
        div_id,
        traces,
        layout,
        trace_groups,
        note='Shaded regions show group mean ± SEM across patients. The cosine panel is intentionally narrower than the accuracy panels.',
        min_height=900,
    )


# ─── Peak timing analysis ──────────────────────────────────────────────────────

def _peak_timing_analysis(run_dir, patients, data_dir, patient_ref_bins, bin_size_ms, align_cue, n_bh=10, warp='none',
                          target_sec=None):
    """
    Per patient × embedding: find peak bin for cosine, word verbatim, cat verbatim.
    Report peak time (s) and latency from each preceding cue.

    Returns
    -------
    rows : list[dict]
        Keys: patient, embedding, metric, peak_bin, peak_time_s,
              peak_value, cue_<name>_s (latency from that cue to peak in seconds, positive = after cue)
    cue_col_names : list[str]
    """
    rows = []
    cue_col_names = [f'to_{c}' for c in CUE_STYLES]

    for patient in patients:
        csv_path = os.path.join(run_dir, patient, 'per_time_scores.csv')
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)

        cue_info = _load_cue_info_from_stats_json(run_dir, patient)
        if cue_info is None and data_dir:
            _ref_bin_s = n_bh * (bin_size_ms / 1000.0)
            cue_info = _load_cue_timings(patient, data_dir, align_cue,
                                         ref_bin_s=_ref_bin_s, warp=warp,
                                         target_sec=target_sec)

        n_bins  = int(df['bin_index'].max()) + 1
        p_ref   = (patient_ref_bins or {}).get(patient, 10)
        time_s = np.array([(b - p_ref) * bin_size_ms / 1000.0 for b in range(n_bins)])

        for emb in AN_EMBEDDING_NAMES:
            sub = _embedding_bin_sorted(df, emb)
            if len(sub) == 0:
                continue

            for metric, col_name, scale in [
                ('cosine',    'cosine_mean',          1.0),
                ('word_verb', 'word_balanced_acc',    100.0),
                ('cat_verb',  'category_balanced_acc', 100.0),
            ]:
                if col_name not in sub.columns:
                    continue
                vals = sub[col_name].values.astype(np.float32) * scale
                # Only post-onset bins (ref_bin may equal or exceed n_bins if
                # align_back_sec extends beyond the recorded window — fall back
                # to full-range argmax in that case)
                post = vals[p_ref:] if p_ref < len(vals) else vals
                post_offset = p_ref if p_ref < len(vals) else 0
                peak_b = int(np.argmax(post) + post_offset)
                peak_t = float(time_s[peak_b])
                peak_v = float(vals[peak_b])

                row = {
                    'patient':       patient,
                    'embedding':     emb,
                    'metric':        metric,
                    'peak_bin':      peak_b,
                    'peak_time_s':   peak_t,
                    'peak_value':    peak_v,
                }

                # Latency from each cue to peak (positive = peak is after cue)
                for cue_name in CUE_STYLES:
                    key = f'to_{cue_name}'
                    if cue_info and cue_name in cue_info:
                        row[key] = peak_t - cue_info[cue_name]['mean_s']
                    else:
                        row[key] = float('nan')

                rows.append(row)

    return rows, cue_col_names


# ─── HTML helpers ──────────────────────────────────────────────────────────────

def _meta_table_html(meta):
    if not meta:
        return ''
    labels = {
        'run_id': 'Run ID', 'timestamp_utc': 'Timestamp (UTC)',
        'task': 'Task', 'auditory_warp': 'Auditory Warp',
        'align_cue': 'Alignment Cue', 'align_back_sec': 'Align Back (s)',
        'align_forward_sec': 'Align Forward (s)', 'patients': 'Patients',
        'n_epochs': 'Epochs', 'bin_size_ms': 'Bin Size (ms)',
        'n_bins_history': 'History Bins', 'closest': 'Retrieval Distance',
        'model_mode': 'Model Mode', 'embedding_names': 'Embeddings',
        'regressor_pipeline': 'Regressor Pipeline', 'y_reducer': 'Y Reducer',
        'git_commit': 'Git Commit', 'python_version': 'Python Version',
        'succeeded_patients': 'Succeeded', 'failed_patients': 'Failed',
        'command_line': 'Command Line',
    }
    rows = ''
    for key, val in meta.items():
        label = labels.get(key, key)
        val_str = ', '.join(str(v) for v in val) if isinstance(val, list) else str(val)
        rows += f'<tr><td><strong>{label}</strong></td><td><code>{val_str}</code></td></tr>\n'
    return f'<table class="meta-table">{rows}</table>'


def _sig_summary_html(perbin_sig, patients, n_bins_history, bin_size_ms, n_tests):
    """HTML table: per patient × embedding, count of significant bins per metric."""
    header = ('<table><tr><th>Patient</th><th>Embedding</th>'
              '<th>Cosine sig bins</th><th>Word verb sig</th><th>Word loose sig</th>'
              '<th>Cat verb sig</th><th>Cat loose sig</th></tr>\n')
    body = ''
    for patient in patients:
        if patient not in perbin_sig:
            continue
        for emb in AN_EMBEDDING_NAMES:
            if emb not in perbin_sig[patient]:
                continue
            sb = perbin_sig[patient][emb]
            counts = {k: int(np.sum(v)) for k, v in sb.items()}
            body += (f'<tr><td>{patient}</td><td><strong>{emb}</strong></td>'
                     f'<td>{counts.get("cosine",0)}</td>'
                     f'<td>{counts.get("word_verb",0)}</td>'
                     f'<td>{counts.get("word_loose",0)}</td>'
                     f'<td>{counts.get("cat_verb",0)}</td>'
                     f'<td>{counts.get("cat_loose",0)}</td></tr>\n')
    return header + body + '</table>'


def _peak_table_html(peak_rows, cue_col_names):
    if not peak_rows:
        return '<p><em>No peak data.</em></p>'
    metric_labels = {'cosine': 'Cosine', 'word_verb': 'Word Verbatim', 'cat_verb': 'Cat Verbatim'}
    cue_display = {
        'to_aud_stim_onset':  'Δ Stim onset',
        'to_aud_stim_offset': 'Δ Stim offset',
        'to_go_cue_onset':    'Δ Go cue',
        'to_voice_onset':     'Δ Voice onset',
        'to_voice_offset':    'Δ Voice offset',
    }

    html = ('<table style="font-size:11px"><tr>'
            '<th>Patient</th><th>Embedding</th><th>Metric</th>'
            '<th>Peak time (s)</th><th>Peak value</th>')
    for c in cue_col_names:
        html += f'<th>{cue_display.get(c, c)}</th>'
    html += '</tr>\n'

    for row in peak_rows:
        metric_str = metric_labels.get(row['metric'], row['metric'])
        peak_val_str = f"{row['peak_value']:.3f}" if row['metric'] == 'cosine' else f"{row['peak_value']:.1f}%"
        html += (f"<tr><td>{row['patient']}</td><td><strong>{row['embedding']}</strong></td>"
                 f"<td>{metric_str}</td>"
                 f"<td>{row['peak_time_s']:.2f}</td>"
                 f"<td>{peak_val_str}</td>")
        for c in cue_col_names:
            v = row.get(c, float('nan'))
            if np.isnan(v):
                html += '<td>—</td>'
            else:
                cls = ' style="color:#27ae60"' if v > 0 else ' style="color:#c62828"'
                html += f'<td{cls}>{v:+.2f}</td>'
        html += '</tr>\n'
    html += '</table>'
    return html


def _cross_patient_table_html(run_dir, patients, perbin_sig, patient_ref_bins=None, bin_size_ms=100):
    """Summary: n patients significant per embedding, mean peak time."""
    rows_html = ''
    for emb in AN_EMBEDDING_NAMES:
        n_wv = n_cv = 0
        peak_times_w = []
        peak_times_c = []

        for p in patients:
            csv_path = os.path.join(run_dir, p, 'per_time_scores.csv')
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            sub = _embedding_bin_sorted(df, emb)
            if len(sub) == 0:
                continue

            n_bins  = int(sub['bin_index'].max()) + 1
            p_ref   = (patient_ref_bins or {}).get(p, 10)
            time_s = np.array([(b - p_ref) * bin_size_ms / 1000.0 for b in range(n_bins)])

            _post_off = p_ref if p_ref < n_bins else 0
            if perbin_sig and p in perbin_sig and emb in perbin_sig[p]:
                if np.any(perbin_sig[p][emb].get('word_verb', [])):
                    n_wv += 1
                    w_vals = sub['word_balanced_acc'].values
                    peak_b = int(np.argmax(w_vals[_post_off:]) + _post_off)
                    peak_times_w.append(float(time_s[peak_b]))
                if np.any(perbin_sig[p][emb].get('cat_verb', [])):
                    n_cv += 1
                    c_vals = sub['category_balanced_acc'].values
                    peak_b = int(np.argmax(c_vals[_post_off:]) + _post_off)
                    peak_times_c.append(float(time_s[peak_b]))

        n_p = len(patients)
        mean_pw = f'{np.mean(peak_times_w):.2f} s' if peak_times_w else '—'
        mean_pc = f'{np.mean(peak_times_c):.2f} s' if peak_times_c else '—'
        rows_html += (f'<tr><td><strong>{emb}</strong></td>'
                      f'<td>{n_wv}/{n_p}</td><td>{mean_pw}</td>'
                      f'<td>{n_cv}/{n_p}</td><td>{mean_pc}</td></tr>\n')

    return ('<table><tr><th>Embedding</th>'
            '<th>Word sig (n patients)</th><th>Word peak time (mean)</th>'
            '<th>Cat sig (n patients)</th><th>Cat peak time (mean)</th></tr>\n'
            + rows_html + '</table>')


# ─── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; color: #333; line-height: 1.6; }
h1 { color: #1a5276; border-bottom: 3px solid #2980b9; padding-bottom: 10px; }
h2 { color: #2471a3; margin-top: 40px; border-bottom: 1px solid #d4e6f1; padding-bottom: 5px; }
h3 { color: #2e86c1; }
.summary-box { background: #eaf2f8; border-left: 4px solid #2980b9; padding: 15px; margin: 20px 0; border-radius: 4px; }
.method-box  { background: #f3e5f5; border-left: 4px solid #8e24aa; padding: 15px; margin: 15px 0; border-radius: 4px; }
.warning     { background: #fdedec; border-left: 4px solid #e74c3c; padding: 15px; margin: 15px 0; border-radius: 4px; }
.finding     { background: #fef9e7; border-left: 4px solid #f39c12; padding: 15px; margin: 15px 0; border-radius: 4px; }
.meta-box    { background: #f9f9f9; border: 1px solid #ddd; border-radius: 4px; padding: 10px 15px; margin: 15px 0; }
.meta-box summary { cursor: pointer; font-weight: bold; color: #2471a3; padding: 5px 0; }
table        { border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }
.meta-table  { font-size: 12px; }
.meta-table td { padding: 4px 10px; border-bottom: 1px solid #eee; }
.meta-table tr:nth-child(even) { background: #f8f9fa; }
th { background: #2980b9; color: white; padding: 8px 10px; text-align: left; }
td { padding: 6px 10px; border-bottom: 1px solid #ddd; }
tr:nth-child(even) { background: #f8f9fa; }
code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }
.fig-grid  { display: flex; flex-wrap: wrap; gap: 18px; margin: 20px 0; align-items: flex-start; }
.fig-card  { border: 1px solid #d4e6f1; border-radius: 6px; padding: 10px 12px; background: #fafcff; flex: 1 1 520px; min-width: 520px; }
.plotly-an { width: 100%; min-height: 620px; border: 1px solid #d4e6f1; border-radius: 4px; background: white; }
.embedding-toggle-bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 4px 0 10px 0; }
.embedding-toggle-label { font-size: 12px; font-weight: 600; color: #455A64; }
.embedding-toggle { padding: 6px 10px; border: 1px solid var(--emb-color); background: white; color: #263238; cursor: pointer; border-radius: 999px; font-size: 12px; transition: all 120ms ease; }
.embedding-toggle.active { background: var(--emb-color); color: white; }
.embedding-toggle.inactive { background: white; color: #607D8B; opacity: 0.78; }
.embedding-toggle:hover { box-shadow: 0 0 0 2px rgba(21, 101, 192, 0.08); }
.plotly-note { font-size: 12px; color: #546E7A; margin: 10px 2px 0; }
"""

_CUE_LEGEND_HTML = """
<p style="font-size:11.5px;margin-top:6px;">
  <strong>Cue lines:</strong>
  <span style="color:#E65100;">&#9473;</span> Stim onset &nbsp;
  <span style="color:#AD1457;">&#x2504;</span> Stim offset &nbsp;
  <span style="color:#1B5E20;">&#9473;</span> Go cue &nbsp;
  <span style="color:#1A237E;">&#9473;</span> Voice onset &nbsp;
  <span style="color:#4A148C;">&#x2504;</span> Voice offset &nbsp;
  &emsp;
  <strong>Embeddings:</strong>
  <span style="color:#1565C0;">&#9632;</span> GloVe &nbsp;
  <span style="color:#0288D1;">&#9632;</span> FastText &nbsp;
  <span style="color:#00838F;">&#9632;</span> Word2Vec &nbsp;
  <span style="color:#2E7D32;">&#9632;</span> ConceptNet
</p>
"""


# ─── Main entry point ──────────────────────────────────────────────────────────

def generate_report(run_dir, out_dir, meta=None, data_dir=None):
    """
    Generate the full auditory naming HTML report for a single run.

    Parameters
    ----------
    run_dir  : str  — path to the run results folder
    out_dir  : str  — output directory for the HTML file
    meta     : dict or None — contents of meta.json
    data_dir : str or None — path to the raw data folder (for cue timings)
                             e.g. .../main/data/

    Returns
    -------
    str — path to the generated HTML file, or None on failure.
    """
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    run_id       = meta.get('run_id', 'unknown')    if meta else 'unknown'
    n_bh         = meta.get('n_bins_history', 10)   if meta else 10
    bin_size_ms  = meta.get('bin_size_ms', 100)     if meta else 100
    align_cue    = meta.get('align_cue', 'none')    if meta else 'none'
    warp         = meta.get('auditory_warp', 'none') if meta else 'none'
    if warp == 'linear':          # legacy alias for the stim-segment warp
        warp = 'stim'
    pipeline_str = meta.get('regressor_pipeline', '?') if meta else '?'
    align_back_sec = meta.get('align_back_sec') if meta else None
    # Group warp target (used only by the raw-df cue fallback; cue_stats.json is preferred).
    warp_target_sec = meta.get('auditory_warp_target_sec') if meta else None

    # Patients: from meta or directory scan
    if meta and meta.get('succeeded_patients'):
        patients = sorted(meta['succeeded_patients'])
    else:
        patients = sorted([
            d for d in os.listdir(run_dir)
            if os.path.isdir(os.path.join(run_dir, d))
            and os.path.exists(os.path.join(run_dir, d, 'per_time_scores.csv'))
        ])

    if not patients:
        print("[Report] No patient data found — aborting")
        return None

    # ── Determine n_bins from first patient ──────────────────────────────────
    _first_csv = os.path.join(run_dir, patients[0], 'per_time_scores.csv')
    _df0 = pd.read_csv(_first_csv)
    n_bins = int(_df0['bin_index'].max()) + 1
    n_emb  = len(AN_EMBEDDING_NAMES)

    # ── Build per-patient ref_bins ────────────────────────────────────────────
    patient_ref_bins = {}
    for _p in patients:
        if align_cue != 'none':
            _abs = _load_actual_back_sec(run_dir, _p, meta)
            if _abs is not None:
                patient_ref_bins[_p] = int(round(_abs / (bin_size_ms / 1000.0)))
            elif align_back_sec is not None:
                # Fallback: total bins − forward bins
                _csv_p = os.path.join(run_dir, _p, 'per_time_scores.csv')
                if os.path.exists(_csv_p):
                    _n_b = int(pd.read_csv(_csv_p)['bin_index'].max()) + 1
                    _fwd = int(round(float((meta or {}).get('align_forward_sec', 2.0)) / (bin_size_ms / 1000.0)))
                    patient_ref_bins[_p] = max(0, _n_b - _fwd)
                else:
                    patient_ref_bins[_p] = int(round(float(align_back_sec) / (bin_size_ms / 1000.0)))
            else:
                patient_ref_bins[_p] = n_bh
        else:
            patient_ref_bins[_p] = n_bh
    # Scalar ref_bin: only used for HTML text (not for axes)
    ref_bin = patient_ref_bins.get(patients[0], n_bh)
    print(f"  [report] patient_ref_bins: {patient_ref_bins}", flush=True)

    print(f"  [report] {len(patients)} patients, {n_bins} bins, {n_emb} embeddings", flush=True)
    print(f"  [report] Bonferroni n_tests = {n_bins} × {n_emb} × {len(patients)} = {n_bins * n_emb * len(patients)}", flush=True)

    # ── Per-bin significance ──────────────────────────────────────────────────
    print("  [report] Computing per-bin significance...", flush=True)
    perbin_sig, pkl_failed = _compute_perbin_sig(run_dir, patients, n_bins, n_emb, patient_ref_bins=patient_ref_bins)

    # ── Load cue timings ──────────────────────────────────────────────────────
    # Priority: cue_stats.json (pre-computed, post-warp) > raw df with warp applied.
    all_cue_info = {}
    print("  [report] Loading cue timings...", flush=True)
    _ref_bin_s = n_bh * (bin_size_ms / 1000.0)
    for p in patients:
        ci = _load_cue_info_from_stats_json(run_dir, p)
        if ci:
            print(f"    {p}: cue_stats.json ({list(ci.keys())})", flush=True)
        elif data_dir:
            ci = _load_cue_timings(p, data_dir, align_cue,
                                   ref_bin_s=_ref_bin_s, warp=warp,
                                   target_sec=warp_target_sec)
            src = 'raw df' + ('+warp' if warp != 'none' else '')
            print(f"    {p}: {src} ({list(ci.keys()) if ci else 'unavailable'})",
                  flush=True)
        else:
            print(f"    {p}: cue timings unavailable", flush=True)
        if ci:
            all_cue_info[p] = ci

    # ── Load PKL data (cosine per-epoch + null arrays) ────────────────────────
    all_pkl_data = {}
    for p in patients:
        pkl_path = os.path.join(run_dir, p, 'semantic_regression_results.pkl')
        if os.path.exists(pkl_path):
            try:
                all_pkl_data[p] = _load_null_arrays(pkl_path,
                                      patient_dir=os.path.join(run_dir, p))
                print(f"  [pkl] {p}: OK", flush=True)
            except Exception as e:
                print(f"  [pkl] {p}: failed ({e})", flush=True)

    # ── Per-patient figures ───────────────────────────────────────────────────
    figures_main = {}
    figures_comp = {}
    for p in patients:
        try:
            figures_main[p] = make_figure(
                p, run_dir, patient_ref_bins.get(p, n_bh), bin_size_ms,
                cue_info=all_cue_info.get(p),
                sig_bins=perbin_sig.get(p),
                pkl_data=all_pkl_data.get(p),
            )
            print(f"  [figure-main] {p}: OK", flush=True)
        except Exception as e:
            print(f"  [figure-main] {p}: FAILED ({e})", flush=True)
        try:
            figures_comp[p] = make_comparison_figure(p, run_dir, patient_ref_bins.get(p, n_bh), bin_size_ms)
            print(f"  [figure-comp] {p}: OK", flush=True)
        except Exception as e:
            print(f"  [figure-comp] {p}: FAILED ({e})", flush=True)

    # ── Group figure ──────────────────────────────────────────────────────────
    group_fig_html = None
    try:
        group_fig_html = make_group_figure(run_dir, patients, patient_ref_bins, bin_size_ms)
        print("  [group figure] OK", flush=True)
    except Exception as e:
        print(f"  [group figure] FAILED ({e})", flush=True)

    # ── Peak timing analysis ──────────────────────────────────────────────────
    peak_rows, cue_col_names = _peak_timing_analysis(
        run_dir, patients, data_dir, patient_ref_bins, bin_size_ms, align_cue,
        n_bh=n_bh, warp=warp,
        target_sec=warp_target_sec,
    )

    # ── HTML assembly ─────────────────────────────────────────────────────────
    n_tests = n_bins * n_emb * len(patients)
    meta_table = _meta_table_html(meta)
    align_label = align_cue if align_cue != 'none' else 'trial_onset (none — fixed history window)'

    # Section 2 — per-patient figures
    fig_grid_html = '<div class="fig-grid">\n'
    for p in patients:
        if p in figures_main:
            fig_grid_html += (
                f'<div class="fig-card"><h4 style="margin:4px 0">{p}</h4>'
                f'{figures_main[p]}</div>\n'
            )
    fig_grid_html += '</div>\n'

    # Section 3 — verbatim vs loose
    comp_grid_html = '<div class="fig-grid">\n'
    for p in patients:
        if p in figures_comp:
            comp_grid_html += (
                f'<div class="fig-card"><h4 style="margin:4px 0">{p}</h4>'
                f'{figures_comp[p]}</div>\n'
            )
    comp_grid_html += '</div>\n'

    # PKL failure note
    pkl_note = (
        f'<div class="warning"><strong>PKL load failed for:</strong> '
        f'{", ".join(pkl_failed)}. Accuracy significance falls back to the report\'s non-PKL heuristics for those patients.</div>\n'
        if pkl_failed else ''
    )

    # Section 6 — group figure
    group_section_html = ''
    if group_fig_html:
        group_section_html = (
            f'<div class="fig-card" style="display:block">'
            f'{group_fig_html}</div>\n'
        )

    html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Auditory Naming Report — {run_id}</title>
{PLOTLY_JS}
<style>{_CSS}</style>
</head><body>

<h1>Auditory Naming: Semantic Regression Report</h1>
<p>
  <strong>Run:</strong> <code>{run_id}</code> &nbsp;|&nbsp;
  <strong>Pipeline:</strong> <code>{pipeline_str}</code> &nbsp;|&nbsp;
  <strong>Alignment:</strong> {align_label} &nbsp;|&nbsp;
  <strong>Bonferroni n:</strong> n_bins per plot (corrected per time-series subplot)
</p>

<div class="summary-box">
<h3>Executive Summary</h3>
<p>Task: <strong>auditory naming</strong> &nbsp;|&nbsp;
   Patients: <strong>{", ".join(patients)}</strong> &nbsp;|&nbsp;
   Embeddings: GloVe, FastText, Word2Vec, ConceptNet &nbsp;|&nbsp;
   Bins: {n_bins} &times; {bin_size_ms}&nbsp;ms &nbsp;|&nbsp;
   Alignment reference: {align_label}
</p>
</div>

<h2>1. Run Configuration</h2>
<details class="meta-box" open>
  <summary>meta.json — all run parameters</summary>
  {meta_table if meta_table else '<p><em>No meta.json found.</em></p>'}
</details>

<h2>2. Per-Patient Time-Series</h2>
<div class="method-box">
    <strong>Rows (top to bottom):</strong> Cosine similarity (half-width) &middot; Word accuracy &middot;
    Word loose accuracy &middot; Category accuracy &middot; Category loose accuracy.<br>
    <strong>Null:</strong> Dotted line = null / shuffle mean; shaded band = null 95% CI (from {50} shuffled epochs).<br>
  <strong>Significance ticks</strong> at top of each panel = Bonferroni-corrected
    (p&nbsp;&lt;&nbsp;0.05&nbsp;/&nbsp;n_bins) per-bin Wilcoxon for exact &amp; loose accuracy.
    Cosine significance ticks are intentionally omitted.<br>
    <strong>Embedding buttons</strong> hide/show the corresponding curves, null references, and significance overlays together.<br>
  Vertical lines = behavioral cue events (mean &plusmn; 1&nbsp;SD across trials).
</div>
{_CUE_LEGEND_HTML}
{pkl_note}
{fig_grid_html}

<h2>3. Exact vs. Loose Accuracy</h2>
<p style="font-size:12px;">
    Solid = exact match accuracy.
  Dashed = loose (lemma / WordNet-flexible match).
  Both as percentage. Plotted per embedding per patient.
</p>
{comp_grid_html}

<h2>4. Per-Bin Significance Summary</h2>
<p style="font-size:12px;">
  Count of time bins passing Bonferroni threshold (p&nbsp;&lt;&nbsp;0.05&nbsp;/&nbsp;n_bins)
  per patient &times; embedding &times; metric.
  Verbatim &amp; loose use Wilcoxon (obs&nbsp;&minus;&nbsp;null&nbsp;&gt;&nbsp;0).
  Cosine uses pre-onset threshold.
</p>
{_sig_summary_html(perbin_sig, patients, n_bh, bin_size_ms, n_tests)}

<h2>5. Peak Timing Analysis</h2>
<p style="font-size:12px;">
  Peak bin is the post-onset maximum per patient &times; embedding &times; metric.
  <em>&Delta;&nbsp;Cue</em> = peak time &minus; cue mean time (positive = peak is <em>after</em> cue).
  Only applies to runs where cue timings could be loaded from the data folder.
</p>
{_peak_table_html(peak_rows, cue_col_names)}

<h2>6. Cross-Patient Summary</h2>
<h3>Group-Level Time Courses (mean &plusmn; SEM across patients)</h3>
{_CUE_LEGEND_HTML}
{group_section_html if group_section_html else '<p><em>Group figure not generated.</em></p>'}

<h3>Significance Summary Across Patients</h3>
{_cross_patient_table_html(run_dir, patients, perbin_sig, patient_ref_bins=patient_ref_bins, bin_size_ms=bin_size_ms)}

</body></html>'''

    out_path = os.path.join(out_dir, f'auditory_naming_report.html')
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(html)
    print(f"[Report] Saved: {out_path} ({len(html) // 1024} KB)", flush=True)
    return out_path
