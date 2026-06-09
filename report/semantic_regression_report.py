# -*- coding: utf-8 -*-
"""
report.semantic_regression_report — Assemble the full HTML analysis report.

Takes DataFrames produced by the helper modules (significance_testing,
word_bias_analysis, metric_dissociation, embedding_norms) and generates a
self-contained HTML report with:
  - Run configuration (from meta.json)
  - Executive summary
  - Significance tables (category + word, with Bonferroni stars)
  - Word prediction bias analysis
  - Embedding norm analysis
  - Metric dissociation
  - Semantic vs. visual comparison

Output filename: semantic_regression_report_<run_id>.html
"""

import os
import io
import json
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as _scipy_stats

try:
    # Package execution: python -m report.semantic_regression_report
    from .helper.config import EMBEDDING_NAMES, SEM_MODELS, VIS_MODELS
    from .helper.results_loader import load_patient_from_pkl
except ImportError:
    try:
        # Script execution from report/: python semantic_regression_report.py ...
        from helper.config import EMBEDDING_NAMES, SEM_MODELS, VIS_MODELS
        from helper.results_loader import load_patient_from_pkl
    except ImportError:
        # Script execution from main/: python report/semantic_regression_report.py ...
        from report.helper.config import EMBEDDING_NAMES, SEM_MODELS, VIS_MODELS
        from report.helper.results_loader import load_patient_from_pkl


# ─── Plotting constants ───────────────────────────────────────────────────────

EMB_COLORS = {
    'GloVe':       '#1565C0',   # dark blue
    'FastText':    '#0288D1',   # sky blue
    'Word2Vec':    '#00838F',   # teal
    'ConceptNet':  '#2E7D32',   # green
    'DINOv2':      '#E65100',   # burnt orange
    'SimCLR':      '#AD1457',   # deep pink
}

# Manual threshold for per-bin Wilcoxon significance ticks.
PERBIN_SIG_ALPHA = 0.01


def _mark_sig_bins(ax, time_ms, sig_mask, color, row=0):
    """
    Draw short colored tick marks at the top edge of ax for each True bin in sig_mask.
    row offsets multiple embeddings so they don't fully overlap.
    """
    n_rows = len(EMBEDDING_NAMES)
    strip  = 0.05 / max(n_rows, 1)
    ymax   = 1.0 - row * strip
    ymin   = ymax - strip
    for b, is_sig in enumerate(sig_mask):
        if is_sig and b < len(time_ms):
            ax.axvline(time_ms[b], ymin=ymin, ymax=ymax,
                       color=color, lw=2.0, alpha=0.85, zorder=5)


def _compute_perbin_sig(run_dir, patients, n_bins_history, sig_alpha=PERBIN_SIG_ALPHA):
    """
    Compute per-bin significance masks for each patient × embedding:
    cat / word : Wilcoxon signed-rank (obs − null > 0), p < sig_alpha uncorrected, from PKL
      cosine     : pre-onset threshold (value > mean + 1 SEM), from CSV

    Returns
    -------
    perbin_sig : dict[patient][embedding] → {"cat": bool[], "word": bool[], "cosine": bool[]}
    pkl_failed : list[str]  patients where PKL could not be loaded
    """
    perbin_sig = {}
    pkl_failed = []

    for patient in patients:
        pkl_path = os.path.join(run_dir, patient, 'semantic_regression_results.pkl')
        csv_path = os.path.join(run_dir, patient, 'per_time_scores.csv')

        if not os.path.exists(csv_path):
            continue

        df_csv = pd.read_csv(csv_path)
        n_bins = int(df_csv['bin_index'].max()) + 1

        pkl_data = None
        if os.path.exists(pkl_path):
            try:
                pkl_data = load_patient_from_pkl(pkl_path)
            except Exception as e:
                print(f"  [perbin-sig] {patient}: PKL failed ({e})", flush=True)
        if pkl_data is None and os.path.exists(pkl_path):
            pkl_failed.append(patient)

        perbin_sig[patient] = {}

        for emb in EMBEDDING_NAMES:
            sub = df_csv[df_csv['embedding'] == emb].sort_values('bin_index').reset_index(drop=True)
            if len(sub) == 0:
                continue

            # cosine: pre-onset threshold (no per-epoch arrays available)
            if 'cosine_mean' in sub.columns and not sub['cosine_mean'].isna().all():
                cos = sub['cosine_mean'].values.astype(np.float32)
                # _presonset_null defined below; forward reference is fine in Python
                pre  = cos[:n_bins_history]
                valid = pre[~np.isnan(pre)]
                c_mean = float(np.mean(valid)) if len(valid) else 0.0
                c_sem  = float(np.std(valid) / max(np.sqrt(len(valid)), 1))
                cos_sig = cos > (c_mean + c_sem)
            else:
                cos_sig = np.zeros(len(sub), dtype=bool)

            # cat / word: per-bin Wilcoxon from PKL.
            # Category uses _indep arrays (independent centroid lookup).
            if pkl_data is not None and emb in pkl_data:
                d_emb    = pkl_data[emb]
                cat_obs  = np.array(d_emb.get('cat_indep_obs',  d_emb['cat_obs']),  dtype=np.float32)
                cat_null = np.array(d_emb.get('cat_indep_null', d_emb['cat_null']), dtype=np.float32)
                wrd_obs  = np.array(d_emb['word_obs'],  dtype=np.float32)
                wrd_null = np.array(d_emb['word_null'], dtype=np.float32)

                n_b      = cat_obs.shape[1]
                cat_sig  = np.zeros(n_b, dtype=bool)
                wrd_sig  = np.zeros(n_b, dtype=bool)

                for b in range(n_b):
                    for sig_arr, obs_b, null_b in [
                        (cat_sig, cat_obs[:, b], cat_null[:, b]),
                        (wrd_sig, wrd_obs[:, b], wrd_null[:, b]),
                    ]:
                        d = obs_b - null_b
                        if np.any(d != 0):
                            try:
                                _pval = _scipy_stats.wilcoxon(
                                    d, alternative='greater')[1]
                                sig_arr[b] = bool(_pval < sig_alpha)  # type: ignore[operator]
                            except Exception:
                                pass
            else:
                cat_sig = np.zeros(n_bins, dtype=bool)
                wrd_sig = np.zeros(n_bins, dtype=bool)

            perbin_sig[patient][emb] = {
                'cosine': cos_sig,
                'cat':    cat_sig,
                'word':   wrd_sig,
            }

    return perbin_sig, pkl_failed


def make_figure(patient, run_dir, n_bins_history, bin_size_ms, sig_bins=None,
                timing_events=None, align_cue='aud_stim_onset'):
    """
    Three-row figure per patient using per_time_scores.csv + PKL null arrays:
      Row 1 — cosine similarity (mean ± std); chance from CSV chance_mean column
      Row 2 — category balanced accuracy (indep); chance from PKL shuffled null
      Row 3 — word balanced accuracy; chance from PKL shuffled null

    Chance curves are the actual shuffled-pipeline null (mean ± 1 SEM across
    50 null epochs), not a pre-onset flat-line approximation. Falls back to
    pre-onset baseline if PKL cannot be loaded.

    sig_bins : dict[embedding] → {"cosine": bool[], "cat": bool[], "word": bool[]}
        Per-bin significance masks.  Where True, short colored tick marks are
        drawn at the top edge of the corresponding panel.

    timing_events : dict or None
        Event name → time in ms relative to t=0 (alignment point).
        Keys: 'aud_stim_onset', 'aud_stim_offset', 'go_cue', 'voice_onset',
        'voice_offset'.  Special keys 'note' and 'align_cue' are ignored.
        Each event is drawn as a labelled vertical line across all panels.

    align_cue : str
        The event that t=0 corresponds to.  Used for the x-axis label.

    Returns a base64-encoded PNG string for embedding in HTML.
    """
    csv_path = os.path.join(run_dir, patient, 'per_time_scores.csv')
    df = pd.read_csv(csv_path)

    n_bins  = int(df['bin_index'].max()) + 1
    time_ms = np.array([(b - n_bins_history) * bin_size_ms for b in range(n_bins)])

    # ── Per-patient timing: try patient subfolder first, fall back to run-level ─
    _patient_timing_path = os.path.join(run_dir, patient, 'timing_events.json')
    if os.path.exists(_patient_timing_path):
        try:
            with open(_patient_timing_path) as _ptf:
                _active_timing = json.load(_ptf)
        except Exception:
            _active_timing = timing_events
    else:
        _active_timing = timing_events

    # ── Load PKL null arrays (cat_indep + word) ───────────────────────────────
    pkl_nulls = {}   # emb → {'cat_mean', 'cat_sem', 'word_mean', 'word_sem'} all (n_bins,)
    pkl_path = os.path.join(run_dir, patient, 'semantic_regression_results.pkl')
    if os.path.exists(pkl_path):
        try:
            pkl_data = load_patient_from_pkl(pkl_path)
            if pkl_data:
                for emb_k, d in pkl_data.items():
                    cat_null  = np.array(d.get('cat_indep_null', d['cat_null']), dtype=np.float32)
                    word_null = np.array(d['word_null'], dtype=np.float32)
                    n_ep = cat_null.shape[0]
                    pkl_nulls[emb_k] = {
                        'cat_mean':  cat_null.mean(0)  * 100,
                        'cat_sem':   cat_null.std(0)   / np.sqrt(n_ep) * 100,
                        'word_mean': word_null.mean(0) * 100,
                        'word_sem':  word_null.std(0)  / np.sqrt(n_ep) * 100,
                    }
        except Exception as e:
            print(f"  [figure] {patient}: PKL null load failed ({e})", flush=True)

    fig, axes = plt.subplots(3, 1, figsize=(13, 7.5), sharex=True)
    fig.suptitle(f'Patient {patient}', fontsize=12, fontweight='bold')

    emb_row = {e: i for i, e in enumerate(EMBEDDING_NAMES)}

    for emb in EMBEDDING_NAMES:
        sub = df[df['embedding'] == emb].sort_values('bin_index').reset_index(drop=True)
        if len(sub) == 0:
            continue
        col = EMB_COLORS.get(emb, '#333333')
        row = emb_row[emb]

        # ── Row 0: cosine similarity ──────────────────────────────────────────
        if 'cosine_mean' in sub.columns and not sub['cosine_mean'].isna().all():
            cos = sub['cosine_mean'].values.astype(np.float32)
            axes[0].plot(time_ms, cos, color=col, lw=1.5, label=emb)
            if 'cosine_std' in sub.columns:
                cos_std = sub['cosine_std'].values.astype(np.float32)
                axes[0].fill_between(time_ms, cos - cos_std, cos + cos_std,
                                     color=col, alpha=0.10)
            # Chance for cosine: use per-bin shuffled null from CSV if available,
            # else fall back to pre-onset mean.
            if 'chance_mean' in sub.columns and not sub['chance_mean'].isna().all():
                cos_chance = sub['chance_mean'].values.astype(np.float32)
                axes[0].plot(time_ms, cos_chance, color=col, lw=0.9, ls='--', alpha=0.5)
            else:
                pre   = cos[:n_bins_history]
                valid = pre[~np.isnan(pre)]
                c_mean = float(np.mean(valid)) if len(valid) else 0.0
                c_sem  = float(np.std(valid) / max(np.sqrt(len(valid)), 1))
                axes[0].axhline(c_mean, color=col, lw=0.9, ls='--', alpha=0.5)
                axes[0].fill_between(time_ms, c_mean - c_sem, c_mean + c_sem,
                                     color=col, alpha=0.06)

        # ── Rows 1–2: accuracy (% scale) ──────────────────────────────────────
        # Use loose accuracy if available; fall back to balanced accuracy.
        # Note: significance tick marks are still based on per-epoch balanced
        # accuracy from PKL (loose accuracy arrays not stored there).
        cat_col = (
            'category_loose_acc'       if 'category_loose_acc'       in sub.columns else
            'category_balanced_acc_indep' if 'category_balanced_acc_indep' in sub.columns else
            'category_balanced_acc'
        )
        word_col = 'word_loose_acc' if 'word_loose_acc' in sub.columns else 'word_balanced_acc'
        cat_acc  = sub[cat_col].values.astype(np.float32)
        word_acc = sub[word_col].values.astype(np.float32)

        axes[1].plot(time_ms, cat_acc  * 100, color=col, lw=1.5, label=emb)
        axes[2].plot(time_ms, word_acc * 100, color=col, lw=1.5, label=emb)

        if emb in pkl_nulls:
            # Real shuffled-pipeline null: mean ± 1 SEM across 50 null epochs.
            pn = pkl_nulls[emb]
            for ax, mn, se in [
                (axes[1], pn['cat_mean'],  pn['cat_sem']),
                (axes[2], pn['word_mean'], pn['word_sem']),
            ]:
                ax.plot(time_ms, mn, color=col, lw=0.9, ls='--', alpha=0.5)
                ax.fill_between(time_ms, mn - se, mn + se, color=col, alpha=0.06)
        else:
            # Fallback: pre-onset baseline.
            for ax, acc in [(axes[1], cat_acc), (axes[2], word_acc)]:
                pre       = acc[:n_bins_history]
                null_mean = float(pre.mean())
                null_sem  = float(pre.std() / max(np.sqrt(len(pre)), 1))
                ax.axhline(null_mean * 100, color=col, lw=0.9, ls='--', alpha=0.5)
                ax.fill_between(
                    time_ms,
                    (null_mean - null_sem) * 100,
                    (null_mean + null_sem) * 100,
                    color=col, alpha=0.06,
                )

        # ── Significance tick marks ────────────────────────────────────────────
        if sig_bins and emb in sig_bins:
            sb = sig_bins[emb]
            _mark_sig_bins(axes[0], time_ms, sb['cosine'], col, row)
            _mark_sig_bins(axes[1], time_ms, sb['cat'],    col, row)
            _mark_sig_bins(axes[2], time_ms, sb['word'],   col, row)

    axes[0].axhline(0, color='gray', lw=0.7, ls='--', alpha=0.4)
    for ax in axes:
        ax.axvline(0, color='black', lw=0.8, ls=':')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)

    # ── Timing event vertical markers ─────────────────────────────────────────
    _EVENT_STYLE = {
        'aud_stim_onset':  dict(color='#222222', ls='-',  lw=1.3, alpha=0.75, label='Stim onset'),
        'aud_stim_offset': dict(color='#990000', ls='-',  lw=1.3, alpha=0.75, label='Stim offset'),
        'go_cue':          dict(color='#003388', ls='-',  lw=1.3, alpha=0.75, label='Go cue'),
        'voice_onset':     dict(color='#006600', ls='-',  lw=1.3, alpha=0.75, label='Voice onset'),
        'voice_offset':    dict(color='#006600', ls='--', lw=1.3, alpha=0.75, label='Voice offset'),
    }
    _timing_handles = []
    if _active_timing:
        import matplotlib.lines as _mlines
        for ev_name, ev_ms in _active_timing.items():
            if ev_name in ('note', 'align_cue') or ev_ms is None:
                continue
            style = _EVENT_STYLE.get(
                ev_name,
                dict(color='gray', ls=':', lw=1.0, alpha=0.6, label=ev_name),
            )
            kw = {k: v for k, v in style.items() if k != 'label'}
            for ax in axes:
                ax.axvline(float(ev_ms), **kw)
            _timing_handles.append(
                _mlines.Line2D([], [], **kw,
                               label=f"{style['label']} ({ev_ms:+.0f} ms)")
            )

    axes[0].set_ylabel('Cosine Similarity', fontsize=9)
    axes[0].legend(fontsize=7.5, loc='upper left', ncol=3)
    axes[1].set_ylabel('Category Loose Acc. (%)', fontsize=9)
    axes[1].legend(fontsize=7.5, loc='upper left', ncol=3)
    axes[2].set_ylabel('Word Loose Acc. (%)', fontsize=9)

    # X-axis label reflects the alignment cue
    _align_label_map = {
        'aud_stim_onset':  'Time from aud. stim. onset (ms)',
        'aud_stim_offset': 'Time from aud. stim. offset (ms)',
        'trial_onset':     'Time from trial onset (ms)',
        'go_cue_onset':    'Time from go cue (ms)',
        'voice_onset':     'Time from voice onset (ms)',
    }
    x_label = _align_label_map.get(align_cue, f'Time from {align_cue} (ms)')
    axes[2].set_xlabel(x_label, fontsize=9)

    if _timing_handles:
        axes[2].legend(handles=_timing_handles, fontsize=6.5,
                       loc='upper right', ncol=min(len(_timing_handles), 3))
    else:
        axes[2].legend(fontsize=7.5, loc='upper left', ncol=3)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _presonset_null(acc, n_bins_history):
    """Return (null_mean, null_sem) from pre-onset bins."""
    pre = acc[:n_bins_history]
    valid = pre[~np.isnan(pre)]
    null_mean = float(np.mean(valid)) if len(valid) else 0.0
    null_sem  = float(np.std(valid) / max(np.sqrt(len(valid)), 1))
    return null_mean, null_sem


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _sig_class(s):
    """CSS class for a significance star string."""
    return {'***': 'star-three', '**': 'star-two', '*': 'star-one'}.get(s, 'star-ns')


def _patient_tier(p, sig_df):
    """Classify patient signal strength by mean category fold-over-null."""
    sub = sig_df[sig_df.patient == p]
    fold = (sub['mean_cat_obs'] / sub['mean_cat_null']).mean()
    if fold > 1.3:
        return 'patient-high'
    if fold > 1.1:
        return 'patient-moderate'
    return 'patient-low'


def _adaptive_col(df, preferred, fallback):
    """Pick the column name that exists in df."""
    return preferred if preferred in df.columns else fallback


def _meta_table_html(meta):
    """Build an HTML table of all meta.json key-value pairs."""
    if not meta:
        return ''
    # Human-readable label map
    labels = {
        'run_id':              'Run ID',
        'timestamp_utc':       'Timestamp (UTC)',
        'command_line':        'Command Line',
        'task':                'Task',
        'patients':            'Patients',
        'n_epochs':            'Epochs',
        'bin_size_ms':         'Bin Size (ms)',
        'n_bins_history':      'History Bins',
        'closest':             'Retrieval Distance',
        'model_mode':          'Model Mode',
        'embedding_names':     'Embeddings',
        'regressor_pipeline':  'Regressor Pipeline',
        'y_reducer':           'Y Reducer',
        'git_commit':          'Git Commit',
        'git_dirty':           'Git Dirty',
        'python_version':      'Python Version',
        'sklearn_version':     'scikit-learn Version',
        'torch_version':       'PyTorch Version',
        'succeeded_patients':  'Succeeded Patients',
        'failed_patients':     'Failed Patients',
    }
    rows = ''
    for key, val in meta.items():
        label = labels.get(key, key)
        if isinstance(val, list):
            val_str = ', '.join(str(v) for v in val)
        else:
            val_str = str(val)
        rows += f'<tr><td><strong>{label}</strong></td><td><code>{val_str}</code></td></tr>\n'
    return f'<table class="meta-table">{rows}</table>'


# ─── Main report generator ────────────────────────────────────────────────────

def generate_report(sig_df, bias_df, dissoc_df, norm_df, out_dir, meta=None, run_dir=None, perbin_sig_alpha=PERBIN_SIG_ALPHA):
    """
    Generate the full HTML analysis report.

    Parameters
    ----------
    sig_df : pd.DataFrame
        Output of ``significance_testing.compute_significance()``.
    bias_df : pd.DataFrame
        Output of ``word_bias_analysis.compute_word_bias()``.
    dissoc_df : pd.DataFrame
        Output of ``metric_dissociation.compute_metric_dissociation()``.
    norm_df : pd.DataFrame
        Output of ``embedding_norms.compute_norm_analysis()``.
    out_dir : str
        Directory to write the HTML report and CSV files.
    meta : dict or None
        Run metadata (from meta.json) for display in the report header.

    Returns
    -------
    str
        Path to the generated HTML report.
    """
    os.makedirs(out_dir, exist_ok=True)

    if len(sig_df) == 0:
        print("[Report] No significance data — aborting")
        return None

    n_pairs    = len(sig_df)
    n_bins_rep = int(sig_df['n_bins'].median()) if 'n_bins' in sig_df.columns else '?'
    n_tests    = n_pairs * (n_bins_rep if isinstance(n_bins_rep, int) else 1)
    n_patients = sig_df['patient'].nunique()
    patients_sorted = sorted(
        sig_df['patient'].unique(),
        key=lambda p: sig_df[sig_df.patient == p]['mean_cat_obs'].mean(),
        reverse=True,
    )
    n_cat_sig  = (sig_df['cat_sig']  != 'NS').sum()
    n_word_sig = (sig_df['word_sig'] != 'NS').sum()

    # Run info from meta.json
    run_id       = meta.get('run_id', 'unknown')      if meta else 'unknown'
    closest_mode = meta.get('closest', 'l2')           if meta else 'l2'
    pipeline_str = meta.get('regressor_pipeline', '?') if meta else '?'
    n_bh         = meta.get('n_bins_history', 10)      if meta else 10
    bin_size_ms  = meta.get('bin_size_ms', 100)        if meta else 100

    # Alignment cue for axis label and timing event context
    align_cue_raw = (meta.get('align_cue') if meta else None) or 'aud_stim_onset'
    if align_cue_raw in (None, 'none', ''):
        align_cue_raw = 'aud_stim_onset'

    # Human-readable run descriptor for titles and filenames
    _task_str  = (meta.get('task') or 'unknown').replace('_', ' ').title() if meta else 'unknown'
    _warp_str  = meta.get('auditory_warp') or 'none' if meta else 'none'
    _align_display = align_cue_raw.replace('_', ' ')
    _run_label = f'{_task_str}  |  warp: {_warp_str}  |  aligned to: {_align_display}'

    # Short descriptor for filename: task + warp + alignment, no timestamps
    _warp_slug  = (_warp_str or 'none').replace(' ', '-').replace('/', '-')
    _align_slug = align_cue_raw.replace('_', '-')
    _task_slug  = (meta.get('task') or 'run').replace('_', '-') if meta else 'run'
    _short_desc = f'{_task_slug}_{_warp_slug}_{_align_slug}'

    # Optional timing events (ms relative to t=0) from timing_events.json in run dir
    timing_events = None
    if run_dir is not None:
        _timing_path = os.path.join(run_dir, 'timing_events.json')
        if os.path.exists(_timing_path):
            try:
                with open(_timing_path) as _tf:
                    timing_events = json.load(_tf)
                _ev_names = [k for k in timing_events if k not in ('note', 'align_cue')]
                print(f"  [timing] Loaded {len(_ev_names)} events: {_ev_names}", flush=True)
            except Exception as _te:
                print(f"  [timing] Could not load timing_events.json: {_te}", flush=True)

    # ── Per-bin significance masks (Wilcoxon per bin from PKL) ─────────────────
    perbin_sig = {}
    pkl_failed = []
    if run_dir is not None:
        print("  [perbin-sig] Computing per-bin significance...", flush=True)
        perbin_sig, pkl_failed = _compute_perbin_sig(
            run_dir, patients_sorted, n_bh, sig_alpha=perbin_sig_alpha
        )
        if pkl_failed:
            print(f"  [perbin-sig] PKL not loaded for: {', '.join(pkl_failed)}", flush=True)

    # ── Per-patient figures (from per_time_scores.csv) ────────────────────────
    figures = {}
    if run_dir is not None:
        for p in patients_sorted:
            try:
                figures[p] = make_figure(p, run_dir, n_bh, bin_size_ms,
                                         sig_bins=perbin_sig.get(p),
                                         timing_events=timing_events,
                                         align_cue=align_cue_raw)
                print(f"  [figure] {p}: OK", flush=True)
            except Exception as e:
                print(f"  [figure] {p}: FAILED ({e})", flush=True)

    # ── Per-model significance counts ─────────────────────────────────────────
    sig_counts = {emb: {'cat': 0, 'word': 0} for emb in EMBEDDING_NAMES}
    for emb in EMBEDDING_NAMES:
        sub = sig_df[sig_df.embedding == emb]
        sig_counts[emb]['cat']  = (sub['cat_sig']  != 'NS').sum()
        sig_counts[emb]['word'] = (sub['word_sig'] != 'NS').sum()

    # Active embeddings: only those with at least one row in sig_df
    active_embs = [e for e in EMBEDDING_NAMES if sig_df['embedding'].eq(e).any()]
    active_sem  = [e for e in active_embs if e in SEM_MODELS]
    active_vis  = [e for e in active_embs if e in VIS_MODELS]
    has_visual  = len(active_vis) > 0

    # ── Word bias summary ─────────────────────────────────────────────────────
    bias_summary = []
    if len(bias_df) > 0:
        ent_col = _adaptive_col(bias_df, 'pred_entropy_norm', 'pred_entropy')
        for emb in EMBEDDING_NAMES:
            sub = bias_df[bias_df.embedding == emb]
            if len(sub) == 0:
                continue
            top = sub.groupby('top1_word').size().sort_values(ascending=False)
            fav = top.index[0]
            n_fav = top.iloc[0]
            mean_pct = sub[sub.top1_word == fav]['top1_frac'].mean()
            mean_ent = sub[ent_col].mean()
            bias_summary.append({
                'emb': emb, 'fav_word': fav,
                'n_patients': f'{n_fav}/{n_patients}',
                'mean_pct':   f'{mean_pct*100:.1f}%',
                'mean_ent':   f'{mean_ent:.3f}',
            })

    # ── Norm-bias summary ─────────────────────────────────────────────────────
    norm_html = ''
    if len(norm_df) > 0:
        rank_col = _adaptive_col(norm_df, 'norm_rank', 'raw_norm_rank')
        word_col = _adaptive_col(norm_df, 'word', 'raw_norm_word')
        norm_col = _adaptive_col(norm_df, 'pca_norm', 'raw_norm')

        norm_html += '<h3>Embedding Norm vs. Predicted Words</h3>\n'
        norm_html += ('<p>Words with the smallest L2 norm in PCA-reduced embedding space '
                      'per model. Ridge regression is biased toward predicting these words.</p>\n')
        norm_html += '<table><tr><th>Model</th>'
        for r in range(5):
            norm_html += f'<th>Rank {r+1}</th>'
        norm_html += '</tr>\n'

        for emb in EMBEDDING_NAMES:
            sub = norm_df[(norm_df.embedding == emb) & (norm_df[rank_col] < 5)]
            if len(sub) == 0:
                continue
            cells = []
            for rank in range(5):
                rank_sub = sub[sub[rank_col] == rank]
                if len(rank_sub) == 0:
                    cells.append('—')
                else:
                    top_word = rank_sub.groupby(word_col).size().sort_values(ascending=False).index[0]
                    med_norm = rank_sub[rank_sub[word_col] == top_word][norm_col].median()
                    cells.append(f'{top_word} <small>(‖e‖={med_norm:.3f})</small>')
            norm_html += (f'<tr><td><strong>{emb}</strong></td>'
                          + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>\n')
        norm_html += '</table>\n'

        # Norm–bias match rate
        if len(bias_df) > 0:
            match_count = total_count = 0
            for emb in EMBEDDING_NAMES:
                for p in sig_df.patient.unique():
                    bias_row = bias_df[(bias_df.patient == p) & (bias_df.embedding == emb)]
                    norm_row = norm_df[(norm_df.patient == p) & (norm_df.embedding == emb)
                                       & (norm_df[rank_col] == 0)]
                    if len(bias_row) > 0 and len(norm_row) > 0:
                        total_count += 1
                        if bias_row.iloc[0]['top1_word'] == norm_row.iloc[0][word_col]:
                            match_count += 1
            if total_count > 0:
                pct = match_count / total_count
                norm_html += (
                    f'<div class="finding"><strong>Norm–bias correlation:</strong> '
                    f'{match_count}/{total_count} ({pct*100:.0f}%) match. ')
                if pct > 0.7:
                    norm_html += 'Ridge shrinkage is the dominant cause.</div>\n'
                elif pct > 0.3:
                    norm_html += 'Partial — shrinkage is one factor among several.</div>\n'
                else:
                    norm_html += ('Low — bias not primarily driven by norm proximity. '
                                  'Other embedding geometry factors dominate.</div>\n')

    # ── Build table rows ──────────────────────────────────────────────────────
    def _build_table_rows(metric='cat'):
        rows = []
        for p in patients_sorted:
            sub  = sig_df[sig_df.patient == p]
            tier = _patient_tier(p, sig_df)
            n_cats  = round(1 / sub['mean_cat_null'].mean()) if sub['mean_cat_null'].mean() > 0 else '?'
            n_words = round(1 / sub['mean_word_null'].mean()) if sub['mean_word_null'].mean() > 0 else '?'
            null_col = sub[f'mean_{metric}_null'].mean()
            cells = []
            for emb in active_embs:
                row = sub[sub.embedding == emb]
                if len(row) == 0:
                    cells.append('<td>—</td>')
                    continue
                r   = row.iloc[0]
                acc = r[f'mean_{metric}_obs']
                null = r[f'mean_{metric}_null']
                fc  = acc / null if null > 0 else 0
                sig = r[f'{metric}_sig']
                fmt = f'{acc*100:.1f}%' if metric == 'cat' else f'{acc*100:.2f}%'
                cells.append(
                    f'<td class="data-cell">{fmt} ({fc:.1f}×) '
                    f'<span class="{_sig_class(sig)}">{sig}</span></td>')
            fmt_null = f'{null_col*100:.1f}%' if metric == 'cat' else f'{null_col*100:.2f}%'
            rows.append(
                f'<tr class="{tier}"><td><strong>{p}</strong></td>'
                f'<td>{n_words} / {n_cats}</td>'
                + ''.join(cells)
                + f'<td class="chance-cell">{fmt_null}</td></tr>')
        return '\n'.join(rows)

    cat_rows  = _build_table_rows('cat')
    word_rows = _build_table_rows('word')

    # ── Overview table ────────────────────────────────────────────────────────
    overview_rows = ''
    for emb in active_embs:
        mtype = 'Semantic' if emb in SEM_MODELS else 'Visual'
        c = sig_counts[emb]['cat']
        w = sig_counts[emb]['word']
        c_cls = 'sig' if c >= 10 else ('ns' if c < 6 else '')
        w_cls = 'sig' if w >= 10 else ('ns' if w < 6 else '')
        overview_rows += (f'<tr><td><strong>{emb}</strong></td>'
                          f'<td class="{c_cls}">{c}/{n_patients}</td>'
                          f'<td class="{w_cls}">{w}/{n_patients}</td>'
                          f'<td>{mtype}</td></tr>\n')

    # ── Bias table ────────────────────────────────────────────────────────────
    bias_table = ''
    if bias_summary:
        bias_table = ('<table><tr><th>Model</th><th>Favorite Word</th>'
                      '<th>Patients</th><th>Mean % Predictions</th>'
                      '<th>Entropy (norm)</th></tr>\n')
        for b in bias_summary:
            bias_table += (f'<tr><td>{b["emb"]}</td><td><strong>"{b["fav_word"]}"</strong></td>'
                           f'<td>{b["n_patients"]}</td><td>{b["mean_pct"]}</td>'
                           f'<td>{b["mean_ent"]}</td></tr>\n')
        bias_table += '</table>'

    # ── Dissociation HTML ─────────────────────────────────────────────────────
    dissoc_html = ''
    if len(dissoc_df) > 0:
        consistent = 0
        total = dissoc_df.patient.nunique()
        for p in dissoc_df.patient.unique():
            sub = dissoc_df[dissoc_df.patient == p]
            if (sub.loc[sub.best_r2.idxmax(), 'embedding'] ==
                sub.loc[sub.best_cat_acc.idxmax(), 'embedding'] ==
                sub.loc[sub.best_word_acc.idxmax(), 'embedding']):
                consistent += 1
        dissoc_html = (f'<p><strong>{consistent}/{total}</strong> patients have the same '
                       f'model winning all three metrics.</p>')
        d2 = dissoc_df.copy()
        d2['r2_cat_gap']   = np.abs(d2.r2_best_bin - d2.cat_best_bin)
        d2['r2_word_gap']  = np.abs(d2.r2_best_bin - d2.word_best_bin)
        d2['cat_word_gap'] = np.abs(d2.cat_best_bin - d2.word_best_bin)
        dissoc_html += (f'<p>Mean bin gap: R²↔Cat = {d2.r2_cat_gap.mean():.1f}, '
                        f'R²↔Word = {d2.r2_word_gap.mean():.1f}, '
                        f'Cat↔Word = {d2.cat_word_gap.mean():.1f} bins.</p>')

    # ── Dynamic embedding header cells (only active embeddings) ──────────────
    _emb_header_cells = ''.join(
        f'<th class="{"sem-header" if e in SEM_MODELS else "vis-header"}">{e}</th>'
        for e in active_embs
    )

    # ── Semantic vs visual ────────────────────────────────────────────────────
    sem_cat = sum(sig_counts[e]['cat'] for e in active_sem)
    vis_cat = sum(sig_counts[e]['cat'] for e in active_vis)

    if has_visual:
        sem_vis_section = f'''
<h2>6. Semantic vs. Visual</h2>
<table><tr><th>Group</th><th>Cat Sig</th><th>Per Model</th></tr>
<tr><td>Semantic</td><td>{sem_cat}/{n_patients*len(active_sem)}</td>
<td>{"  |  ".join(f"{e}: {sig_counts[e]['cat']}/{n_patients}" for e in active_sem)}</td></tr>
<tr><td>Visual</td><td>{vis_cat}/{n_patients*len(active_vis)}</td>
<td>{"  |  ".join(f"{e}: {sig_counts[e]['cat']}/{n_patients}" for e in active_vis)}</td></tr>
</table>'''
    else:
        sem_vis_section = ''

    # ── Meta table for "Run Configuration" section ────────────────────────────
    meta_table = _meta_table_html(meta)

    # ── Figure grid HTML ──────────────────────────────────────────────────────
    if figures:
        fig_html = '<div class="fig-grid">\n'
        for p in patients_sorted:
            if p in figures:
                fig_html += (
                    f'<div class="fig-card"><img src="data:image/png;base64,{figures[p]}" '
                    f'alt="{p}" style="width:560px;"></div>\n'
                )
        fig_html += '</div>\n'
        _pkl_note = (
            '<br><em style="color:#c62828;">PKL not loaded for: '
            + ', '.join(pkl_failed)
            + ' &mdash; cat/word significance marks omitted.</em>'
            if pkl_failed else ''
        )
        _emb_legend = '  '.join(
            f'<span style="color:{EMB_COLORS[e]};">&#9632;</span> {e}'
            for e in active_embs
        )
        fig_section = f'''
<h2>2. Time-Series ({bin_size_ms} ms bins)</h2>
<p style="font-size:11px;">
  Row 1 = cosine similarity (mean &plusmn; std). Row 2 = category loose accuracy.
  Row 3 = word loose accuracy. All from <code>per_time_scores.csv</code>.<br>
  Dashed = shuffled-pipeline null mean (50 epochs); shaded = &plusmn;1&nbsp;SEM across epochs.
  Cosine chance from CSV <code>chance_mean</code>; cat/word chance from PKL null arrays.<br>
    <strong>Tick marks at top of each panel</strong> = p&nbsp;&lt;&nbsp;{perbin_sig_alpha:.3g} uncorrected
  per-bin Wilcoxon (obs vs. shuffled null) for cat/word;
  pre-onset threshold (&gt;&nbsp;mean&nbsp;+&nbsp;1&nbsp;SEM) for cosine.<br>
  Dotted vertical = trial onset (t&nbsp;=&nbsp;0&nbsp;ms).
  {_emb_legend}
  {_pkl_note}
</p>
{fig_html}'''
    else:
        fig_section = '<p style="font-size:11px;"><em>Time-series figures not generated (run_dir not provided).</em></p>'

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Semantic Regression — {_run_label}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; color: #333; line-height: 1.6; }}
  h1 {{ color: #1a5276; border-bottom: 3px solid #2980b9; padding-bottom: 10px; }}
  h2 {{ color: #2471a3; margin-top: 40px; border-bottom: 1px solid #d4e6f1; padding-bottom: 5px; }}
  h3 {{ color: #2e86c1; }}
  .summary-box {{ background: #eaf2f8; border-left: 4px solid #2980b9; padding: 15px; margin: 20px 0; border-radius: 4px; }}
  .finding {{ background: #fef9e7; border-left: 4px solid #f39c12; padding: 15px; margin: 15px 0; border-radius: 4px; }}
  .warning {{ background: #fdedec; border-left: 4px solid #e74c3c; padding: 15px; margin: 15px 0; border-radius: 4px; }}
  .method-box {{ background: #f3e5f5; border-left: 4px solid #8e24aa; padding: 15px; margin: 15px 0; border-radius: 4px; }}
  .meta-box {{ background: #f9f9f9; border: 1px solid #ddd; border-radius: 4px; padding: 10px 15px; margin: 15px 0; }}
  .meta-box summary {{ cursor: pointer; font-weight: bold; color: #2471a3; padding: 5px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }}
  .meta-table {{ font-size: 12px; }}
  .meta-table td {{ padding: 4px 10px; border-bottom: 1px solid #eee; }}
  .meta-table tr:nth-child(even) {{ background: #f8f9fa; }}
  th {{ background: #2980b9; color: white; padding: 8px 10px; text-align: left; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #ddd; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  .sig {{ color: #27ae60; font-weight: bold; }}
  .ns  {{ color: #e74c3c; }}
  code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
  small {{ color: #888; }}
  .data-cell {{ font-variant-numeric: tabular-nums; text-align: center; }}
  .chance-cell {{ background: #f0f0f0; font-weight: bold; text-align: center; }}
  .star-three {{ color: #1b5e20; font-weight: bold; }}
  .star-two   {{ color: #2e7d32; font-weight: bold; }}
  .star-one   {{ color: #388e3c; }}
  .star-ns    {{ color: #c62828; }}
  .patient-high     td:first-child {{ background: #e8f5e9; font-weight: bold; }}
  .patient-moderate td:first-child {{ background: #fff8e1; }}
  .patient-low      td:first-child {{ background: #ffebee; }}
  .sem-header {{ background: #1565C0; color: white; }}
  .vis-header {{ background: #E65100; color: white; }}
  #cat-table, #word-table {{ font-size: 12px; table-layout: fixed; }}
  #cat-table th, #word-table th {{ padding: 6px 5px; text-align: center; font-size: 11px; }}
  #cat-table td, #word-table td {{ padding: 5px; text-align: center; font-size: 11.5px; }}
  .fig-grid {{ display: flex; flex-wrap: wrap; gap: 18px; margin: 20px 0; }}
  .fig-card {{ border: 1px solid #d4e6f1; border-radius: 6px; padding: 8px; background: #fafcff; }}
</style></head><body>

<h1>Semantic Regression: {_run_label}</h1>
<p><strong>Run:</strong> <code>{run_id}</code> &nbsp;|&nbsp;
   <strong>Pipeline:</strong> <code>{pipeline_str}</code> &nbsp;|&nbsp;
   <strong>Retrieval:</strong> {closest_mode} &nbsp;|&nbsp;
   <strong>Test:</strong> Wilcoxon vs. shuffled null, Bonferroni ({n_tests} tests)</p>

<div class="summary-box">
<h3>Executive Summary</h3>
<p><strong>Category: {n_cat_sig}/{n_tests} ({n_cat_sig*100//n_tests}%) significant</strong> after
Bonferroni correction. Word: {n_word_sig}/{n_tests} ({n_word_sig*100//n_tests}%).
Strongest: {", ".join(patients_sorted[:3])}.</p>
</div>

<h2>1. Run Configuration</h2>
<details class="meta-box" open>
  <summary>meta.json — all run parameters</summary>
  {meta_table if meta_table else '<p><em>No meta.json found for this run.</em></p>'}
</details>

{fig_section}

<h2>3. Significance Testing</h2>
<div class="method-box">
<strong>Method:</strong> Internal shuffled null preserves all pipeline biases (Nystroem, PLS, PCA, retrieval).
At each patient &times; embedding's peak time bin (argmax of mean obs&minus;null), 50 obs vs 50 null
epoch accuracies are compared via one-sided Wilcoxon signed-rank.<br>
<strong>Bonferroni correction:</strong> {n_tests} tests = {n_pairs} patient&times;embedding pairs &times; {n_bins_rep} time bins,
correcting for both cross-patient/embedding and peak-bin-selection multiple comparisons.<br>
<strong>Category accuracy:</strong> independent centroid lookup (<code>category_balanced_acc_indep</code>) —
predicted category is determined by a separate nearest-centroid step in category space,
independent of word retrieval. This avoids the algebraic tautology in the confounded version.
</div>

<h3>Per-Model Significance</h3>
<table><tr><th>Model</th><th>Cat Sig</th><th>Word Sig</th><th>Type</th></tr>
{overview_rows}</table>

<h3>Category Decoding</h3>
<p style="font-size:12px;">
<span class="star-three">*** p&lt;0.001</span> &nbsp;
<span class="star-two">** p&lt;0.01</span> &nbsp;
<span class="star-one">* p&lt;0.05</span> &nbsp;
<span class="star-ns">NS</span> (Bonferroni)</p>
<table id="cat-table">
<tr><th>Patient</th><th>N words/cats</th>
{_emb_header_cells}
<th>Null</th></tr>
{cat_rows}</table>

<h3>Word Decoding</h3>
<div class="warning"><strong>Interpret with caution</strong> — word predictions may be
dominated by prediction bias (see Section 5).</div>
<table id="word-table">
<tr><th>Patient</th><th>N words/cats</th>
{_emb_header_cells}
<th>Null</th></tr>
{word_rows}</table>

<h2>4. Word Prediction Bias</h2>
{bias_table if bias_table else '<p><em>Bias analysis skipped.</em></p>'}

{norm_html}

<h2>5. Metric Dissociation</h2>
{dissoc_html if dissoc_html else '<p><em>No data.</em></p>'}

{sem_vis_section}

</body></html>'''

    out_path = os.path.join(out_dir, f'semantic_regression_report_{_short_desc}.html')
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(html)
    print(f"[Report] Saved: {out_path} ({len(html)//1024} KB)")
    return out_path
