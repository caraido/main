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
    'Word Verbatim\nBal. Acc (%)',
    'Word Loose\nBal. Acc (%)',
    'Cat Verbatim\nBal. Acc (%)',
    'Cat Loose\nBal. Acc (%)',
]


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


def _load_cue_timings(patient, data_dir, align_cue='none', ref_bin_s=0.0, warp='none'):
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
    # Mirrors the warp logic in semantic_regression._linear_time_warp / _warp_cue.
    # Cues before aud_stim_onset : unchanged.
    # Cues after  aud_stim_offset: shifted by (median_dur − original_stim_dur).
    # Cues within [onset, offset]: linearly interpolated to the warped timeline.
    # After warping, aud_stim_offset is identical across trials (onset + median_dur).
    if warp == 'linear':
        stim_durs = aud_stim_offset - aud_stim_onset
        valid_durs = stim_durs[np.isfinite(stim_durs) & (stim_durs > 0)]
        if len(valid_durs) > 0:
            median_dur_s = float(np.median(valid_durs))

            def _warp_arr(cue_arr):
                out = np.empty_like(cue_arr)
                for _i, _t in enumerate(cue_arr):
                    _on  = aud_stim_onset[_i]
                    _off = aud_stim_offset[_i]
                    if not (np.isfinite(_t) and np.isfinite(_on) and np.isfinite(_off)):
                        out[_i] = _t
                    elif _t < _on:
                        out[_i] = _t                              # pre-stim: unchanged
                    elif _t > _off:
                        out[_i] = _t + (median_dur_s - (_off - _on))  # post-stim: shift
                    else:
                        _od = _off - _on
                        out[_i] = (_on + (_t - _on) / _od * median_dur_s
                                   if _od > 0 else _t)            # within-stim: interpolate
                return out

            aud_stim_offset = aud_stim_onset + median_dur_s  # uniform post-warp
            voice_onset     = _warp_arr(voice_onset)
            voice_offset    = _warp_arr(voice_offset)
            go_cue_onset    = _warp_arr(go_cue_onset)

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
            sub = df_csv[df_csv['embedding'] == emb].sort_values('bin_index').reset_index(drop=True)
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
                                    _scipy_stats.wilcoxon(d, alternative='greater')[1] < alpha_corr
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
                                        _scipy_stats.wilcoxon(d, alternative='greater')[1] < alpha_corr
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


# ─── Per-patient figure (2-column layout) ────────────────────────────────────

def make_figure(patient, run_dir, ref_bin, bin_size_ms,
                cue_info=None, sig_bins=None, pkl_data=None):
    """
    Per-patient time-series figure — 2-column layout.

    Row 0 (full width): Cosine similarity
    Row 1 left/right:   Word verbatim | Word loose (both vs shuffled null band)
    Row 2 left/right:   Cat verbatim  | Cat loose  (both vs shuffled null band)
    xlim expands beyond data window to show all cue lines.
    """
    csv_path = os.path.join(run_dir, patient, 'per_time_scores.csv')
    df = pd.read_csv(csv_path)

    n_bins = int(df['bin_index'].max()) + 1
    time_s = np.array([(b - ref_bin) * bin_size_ms / 1000.0 for b in range(n_bins)])

    # ── 2-column gridspec ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    gs  = matplotlib.gridspec.GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.30)
    ax_cos = fig.add_subplot(gs[0, :])
    ax_wv  = fig.add_subplot(gs[1, 0], sharex=ax_cos)
    ax_wl  = fig.add_subplot(gs[1, 1], sharex=ax_cos)
    ax_cv  = fig.add_subplot(gs[2, 0], sharex=ax_cos)
    ax_cl  = fig.add_subplot(gs[2, 1], sharex=ax_cos)
    axes   = [ax_cos, ax_wv, ax_wl, ax_cv, ax_cl]
    fig.suptitle(f'Patient {patient}', fontsize=12, fontweight='bold')

    emb_row    = {e: i for i, e in enumerate(AN_EMBEDDING_NAMES)}
    n_rows_sig = len(AN_EMBEDDING_NAMES)

    for emb in AN_EMBEDDING_NAMES:
        sub = df[df['embedding'] == emb].sort_values('bin_index').reset_index(drop=True)
        if len(sub) == 0:
            continue
        col = EMB_COLORS.get(emb, '#333333')
        row = emb_row[emb]

        def _col(name, scale=1.0):
            if name in sub.columns:
                return sub[name].values.astype(np.float32) * scale
            return np.full(n_bins, np.nan)

        # ── Row 0: cosine ─────────────────────────────────────────────────────
        cos     = _col('cosine_mean')
        cos_std = _col('cosine_std')
        axes[0].plot(time_s, cos, color=col, lw=1.5, label=emb)
        axes[0].fill_between(time_s, cos - cos_std, cos + cos_std, color=col, alpha=0.10)
        if pkl_data and emb in pkl_data and 'cosine' in pkl_data[emb]:
            _null_band(axes[0], time_s, pkl_data[emb]['cosine'], col)
        else:
            pre = cos[:ref_bin]; valid = pre[~np.isnan(pre)]
            if len(valid):
                mu_pre  = float(np.mean(valid))
                sem_pre = float(np.std(valid) / max(np.sqrt(len(valid)), 1))
                axes[0].axhline(mu_pre, color=col, lw=0.8, ls=':', alpha=0.45)
                axes[0].fill_between(time_s, mu_pre - sem_pre, mu_pre + sem_pre,
                                     color=col, alpha=0.06)

        # ── Accuracy subplots: loose uses verbatim null band ──────────────────
        metrics = [
            ('word_balanced_acc',     1, 'word_null'),
            ('word_loose_acc',        2, 'word_null'),
            ('category_balanced_acc', 3, 'cat_null'),
            ('category_loose_acc',    4, 'cat_null'),
        ]
        for col_name, ax_idx, null_key in metrics:
            vals = _col(col_name, scale=100.0)
            axes[ax_idx].plot(time_s, vals, color=col, lw=1.5, label=emb)
            if pkl_data and emb in pkl_data and null_key in pkl_data[emb]:
                _null_band(axes[ax_idx], time_s, pkl_data[emb][null_key] * 100.0, col)
            else:
                pre_v  = (vals / 100.0)[:ref_bin]; valid_v = pre_v[~np.isnan(pre_v)]
                if len(valid_v):
                    mn = float(np.mean(valid_v)) * 100
                    sn = float(np.std(valid_v) / max(np.sqrt(len(valid_v)), 1)) * 100
                    axes[ax_idx].axhline(mn, color=col, lw=0.8, ls=':', alpha=0.45)
                    axes[ax_idx].fill_between(time_s, mn - sn * 1.96, mn + sn * 1.96,
                                              color=col, alpha=0.06)

        # ── Significance ticks ────────────────────────────────────────────────
        if sig_bins and emb in sig_bins:
            sb        = sig_bins[emb]
            keys_axes = [('cosine', 0), ('word_verb', 1), ('word_loose', 2),
                         ('cat_verb', 3), ('cat_loose', 4)]
            for sig_key, ax_idx in keys_axes:
                mask = sb.get(sig_key, np.zeros(n_bins, dtype=bool))
                if len(mask) == n_bins:
                    _mark_sig_bins(axes[ax_idx], time_s, mask, col,
                                   row=row, n_rows=n_rows_sig)

    # ── Decorations ──────────────────────────────────────────────────────────
    axes[0].axhline(0, color='gray', lw=0.6, ls='--', alpha=0.35)
    for ax in axes:
        ax.axvline(0, color='black', lw=0.7, ls=':', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)

    # ── xlim: expand to show all cue lines even beyond data window ────────────
    xl = float(time_s[0])
    xr = float(time_s[-1])
    if cue_info:
        for cinfo in cue_info.values():
            xl = min(xl, cinfo['mean_s'] - cinfo['std_s'])
            xr = max(xr, cinfo['mean_s'] + cinfo['std_s'])
        for ax in axes:
            for cue_name, cinfo in cue_info.items():
                st = CUE_STYLES.get(cue_name, {'color': '#777777', 'ls': '-', 'lw': 1.0})
                mu, std = cinfo['mean_s'], cinfo['std_s']
                ax.axvline(mu, color=st['color'], ls=st['ls'], lw=st['lw'],
                           alpha=0.65, zorder=4)
                ax.axvspan(mu - std, mu + std, color=st['color'], alpha=0.07, zorder=3)
    for ax in axes:
        ax.set_xlim(xl, xr)

    # ── Labels ────────────────────────────────────────────────────────────────
    axes[0].set_ylabel('Cosine Similarity', fontsize=8.5)
    axes[1].set_ylabel('Word Verbatim\nBal. Acc (%)', fontsize=8.5)
    axes[2].set_ylabel('Word Loose\nBal. Acc (%)', fontsize=8.5)
    axes[3].set_ylabel('Cat Verbatim\nBal. Acc (%)', fontsize=8.5)
    axes[4].set_ylabel('Cat Loose\nBal. Acc (%)', fontsize=8.5)
    axes[3].set_xlabel('Time from alignment reference (s)', fontsize=8.5)
    axes[4].set_xlabel('Time from alignment reference (s)', fontsize=8.5)

    # ── Legend (cosine panel) ─────────────────────────────────────────────────
    axes[0].legend(fontsize=7.5, loc='upper left', ncol=4)
    if cue_info:
        from matplotlib.lines import Line2D
        cue_handles = [
            Line2D([0], [0],
                   color=CUE_STYLES.get(c, {'color': '#777'})['color'],
                   ls=CUE_STYLES.get(c, {'ls': '-'})['ls'], lw=1.5,
                   label=CUE_STYLES.get(c, {'label': c})['label'])
            for c in cue_info
        ]
        existing_h, existing_l = axes[0].get_legend_handles_labels()
        axes[0].legend(
            handles=existing_h + cue_handles,
            labels=existing_l + [CUE_STYLES.get(c, {'label': c})['label'] for c in cue_info],
            fontsize=7, loc='upper left', ncol=3,
        )

    plt.tight_layout()
    return _fig_to_b64(fig)


# ─── Verbatim vs Loose comparison figure ──────────────────────────────────────

def make_comparison_figure(patient, run_dir, ref_bin, bin_size_ms):
    """
    2-row figure: verbatim (solid) vs loose (dashed) per embedding.
    Row 0 = word, Row 1 = category.
    """
    csv_path = os.path.join(run_dir, patient, 'per_time_scores.csv')
    df = pd.read_csv(csv_path)

    n_bins  = int(df['bin_index'].max()) + 1
    time_s = np.array([(b - ref_bin) * bin_size_ms / 1000.0 for b in range(n_bins)])

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    fig.suptitle(f'{patient} — Verbatim vs. Loose', fontsize=11, fontweight='bold')

    for emb in AN_EMBEDDING_NAMES:
        sub = df[df['embedding'] == emb].sort_values('bin_index').reset_index(drop=True)
        if len(sub) == 0:
            continue
        col = EMB_COLORS[emb]

        def _c(name):
            return sub[name].values.astype(np.float32) * 100.0 if name in sub.columns else np.full(n_bins, np.nan)

        wv = _c('word_balanced_acc');      wl = _c('word_loose_acc')
        cv = _c('category_balanced_acc');  cl = _c('category_loose_acc')

        axes[0].plot(time_s, wv, color=col, lw=1.5,  ls='-',  label=f'{emb} verbatim')
        axes[0].plot(time_s, wl, color=col, lw=1.5,  ls='--', label=f'{emb} loose',    alpha=0.7)
        axes[1].plot(time_s, cv, color=col, lw=1.5,  ls='-')
        axes[1].plot(time_s, cl, color=col, lw=1.5,  ls='--', alpha=0.7)

    for ax in axes:
        ax.axvline(0, color='black', lw=0.7, ls=':', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel('Word Bal. Acc (%)', fontsize=9)
    axes[0].legend(fontsize=7, loc='upper left', ncol=4)
    axes[1].set_ylabel('Category Bal. Acc (%)', fontsize=9)
    axes[1].set_xlabel('Time from alignment reference (s)', fontsize=9)

    plt.tight_layout()
    return _fig_to_b64(fig)


# ─── Cross-patient group figure ────────────────────────────────────────────────

def make_group_figure(run_dir, patients, patient_ref_bins, bin_size_ms):
    """
    Group-level average across patients (mean ± SEM).
    Each patient's bin_index is first converted to time_s using their own
    actual ref_bin so all patients share a common t=0 reference.
    4 rows: cosine | word verbatim | word loose | category verbatim.
    """
    bin_s = bin_size_ms / 1000.0
    all_dfs = []
    for p in patients:
        csv_path = os.path.join(run_dir, p, 'per_time_scores.csv')
        if os.path.exists(csv_path):
            tmp = pd.read_csv(csv_path)
            rb = (patient_ref_bins or {}).get(p, 10)
            tmp['time_s'] = (tmp['bin_index'] - rb) * bin_s
            all_dfs.append(tmp)
    if not all_dfs:
        return None

    df_all = pd.concat(all_dfs, ignore_index=True)
    # Round to bin resolution to align across patients with different actual windows
    df_all['time_s'] = (df_all['time_s'] / bin_s).round().astype(int) * bin_s
    time_s = np.sort(df_all['time_s'].unique())

    metrics = [
        ('cosine_mean',          False, 'Cosine Similarity'),
        ('word_balanced_acc',    True,  'Word Verbatim Bal. Acc (%)'),
        ('word_loose_acc',       True,  'Word Loose Bal. Acc (%)'),
        ('category_balanced_acc',True,  'Cat Verbatim Bal. Acc (%)'),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 9), sharex=True)
    fig.suptitle('Cross-Patient Average', fontsize=11, fontweight='bold')

    for emb in AN_EMBEDDING_NAMES:
        col = EMB_COLORS[emb]
        sub_all = df_all[df_all['embedding'] == emb].sort_values(['patient', 'bin_index'])
        if len(sub_all) == 0:
            continue

        for ax_idx, (col_name, pct_scale, _) in enumerate(metrics):
            if col_name not in sub_all.columns:
                continue
            grp = sub_all.groupby('time_s')[col_name]
            mu  = grp.mean().reindex(time_s)
            sem = grp.sem().reindex(time_s)
            scale = 100.0 if pct_scale else 1.0
            axes[ax_idx].plot(time_s, mu.values * scale, color=col, lw=1.5, label=emb)
            axes[ax_idx].fill_between(
                time_s,
                (mu - sem).values * scale,
                (mu + sem).values * scale,
                color=col, alpha=0.12,
            )

    for ax_idx, (_, _, ylabel) in enumerate(metrics):
        axes[ax_idx].axvline(0, color='black', lw=0.7, ls=':', alpha=0.5)
        axes[ax_idx].spines['top'].set_visible(False)
        axes[ax_idx].spines['right'].set_visible(False)
        axes[ax_idx].set_ylabel(ylabel, fontsize=8.5)
        axes[ax_idx].tick_params(labelsize=8)

    axes[0].legend(fontsize=7.5, loc='upper left', ncol=4)
    axes[-1].set_xlabel('Time from alignment reference (s)', fontsize=9)
    plt.tight_layout()
    return _fig_to_b64(fig)


# ─── Peak timing analysis ──────────────────────────────────────────────────────

def _peak_timing_analysis(run_dir, patients, data_dir, patient_ref_bins, bin_size_ms, align_cue, n_bh=10, warp='none'):
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
                                         ref_bin_s=_ref_bin_s, warp=warp)

        n_bins  = int(df['bin_index'].max()) + 1
        p_ref   = (patient_ref_bins or {}).get(patient, 10)
        time_s = np.array([(b - p_ref) * bin_size_ms / 1000.0 for b in range(n_bins)])

        for emb in AN_EMBEDDING_NAMES:
            sub = df[df['embedding'] == emb].sort_values('bin_index').reset_index(drop=True)
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
            sub = df[df['embedding'] == emb].sort_values('bin_index').reset_index(drop=True)
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
.fig-grid  { display: flex; flex-wrap: wrap; gap: 18px; margin: 20px 0; }
.fig-card  { border: 1px solid #d4e6f1; border-radius: 6px; padding: 8px; background: #fafcff; }
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
    pipeline_str = meta.get('regressor_pipeline', '?') if meta else '?'
    align_back_sec = meta.get('align_back_sec') if meta else None

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
                                   ref_bin_s=_ref_bin_s, warp=warp)
            src = 'raw df' + ('+warp' if warp == 'linear' else '')
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
    group_fig_b64 = None
    try:
        group_fig_b64 = make_group_figure(run_dir, patients, patient_ref_bins, bin_size_ms)
        print("  [group figure] OK", flush=True)
    except Exception as e:
        print(f"  [group figure] FAILED ({e})", flush=True)

    # ── Peak timing analysis ──────────────────────────────────────────────────
    peak_rows, cue_col_names = _peak_timing_analysis(
        run_dir, patients, data_dir, patient_ref_bins, bin_size_ms, align_cue,
        n_bh=n_bh, warp=warp,
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
                f'<img src="data:image/png;base64,{figures_main[p]}" alt="{p}-main" style="width:580px;"></div>\n'
            )
    fig_grid_html += '</div>\n'

    # Section 3 — verbatim vs loose
    comp_grid_html = '<div class="fig-grid">\n'
    for p in patients:
        if p in figures_comp:
            comp_grid_html += (
                f'<div class="fig-card"><h4 style="margin:4px 0">{p}</h4>'
                f'<img src="data:image/png;base64,{figures_comp[p]}" alt="{p}-comp" style="width:500px;"></div>\n'
            )
    comp_grid_html += '</div>\n'

    # PKL failure note
    pkl_note = (
        f'<div class="warning"><strong>PKL load failed for:</strong> '
        f'{", ".join(pkl_failed)}. Verbatim significance uses pre-onset threshold fallback.</div>\n'
        if pkl_failed else ''
    )

    # Section 6 — group figure
    group_fig_html = ''
    if group_fig_b64:
        group_fig_html = (
            f'<div class="fig-card" style="display:inline-block">'
            f'<img src="data:image/png;base64,{group_fig_b64}" alt="group" style="width:750px;"></div>\n'
        )

    html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Auditory Naming Report — {run_id}</title>
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
  <strong>Rows (top to bottom):</strong> Cosine similarity &middot; Word verbatim balanced acc &middot;
  Word loose balanced acc &middot; Category verbatim balanced acc &middot; Category loose balanced acc.<br>
  <strong>Null:</strong> Dotted line = null mean; shaded band = null 95% CI (from {50} shuffled epochs).<br>
  <strong>Significance ticks</strong> at top of each panel = Bonferroni-corrected
  (p&nbsp;&lt;&nbsp;0.05&nbsp;/&nbsp;n_bins) per-bin Wilcoxon for verbatim &amp; loose;
  pre-onset threshold (&gt;&nbsp;mean&nbsp;+&nbsp;1&nbsp;SEM) for cosine.<br>
  Vertical lines = behavioral cue events (mean &plusmn; 1&nbsp;SD across trials).
</div>
{_CUE_LEGEND_HTML}
{pkl_note}
{fig_grid_html}

<h2>3. Verbatim vs. Loose Accuracy</h2>
<p style="font-size:12px;">
  Solid = verbatim (exact match balanced accuracy).
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
{group_fig_html if group_fig_html else '<p><em>Group figure not generated.</em></p>'}

<h3>Significance Summary Across Patients</h3>
{_cross_patient_table_html(run_dir, patients, perbin_sig, patient_ref_bins=patient_ref_bins, bin_size_ms=bin_size_ms)}

</body></html>'''

    out_path = os.path.join(out_dir, f'auditory_naming_report.html')
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(html)
    print(f"[Report] Saved: {out_path} ({len(html) // 1024} KB)", flush=True)
    return out_path
