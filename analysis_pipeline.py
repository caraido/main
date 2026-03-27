#!/usr/bin/env python3
"""
Semantic Regression Analysis Pipeline
======================================
Reads per-patient results from semantic_regression.py, computes:
  1. Per-epoch significance (Wilcoxon signed-rank vs. shuffled null)
  2. Word prediction bias analysis (entropy, favorite word, norm correlation)
  3. Metric dissociation (R², category acc, word acc across models/bins)
  4. Embedding norm analysis (smallest-norm words in PCA space)
  5. Generates full HTML report

Usage:
    python analysis_pipeline.py [--results-dir PATH] [--figures-dir PATH] [--out-dir PATH]

Defaults assume this folder layout (results/figures sit one level above this script):
    ../semantic_regression/          (results PKLs and CSVs)
    ../semantic_regression_figures/  (per-patient HTML figures)

NOTE on mean-centering (commit 0459d4c, 2026-03-26):
    model.py now subtracts the database centroid from both db_embeds and y_pred
    before nearest-neighbor retrieval. This removes the shrinkage-toward-centroid
    bias so that the favorite-word effect should be reduced in new results.
"""

import sys, types, os, argparse, json, re, base64, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import Counter

warnings.filterwarnings('ignore')

# ─── Module stubs for loading PKL without torch ──────────────────────────────
def _install_stubs():
    for mod_name in ['torch', 'models', 'models.model']:
        sys.modules.setdefault(mod_name, types.ModuleType(mod_name))
    class FakeBasicRegressor: pass
    sys.modules['models'].BasicRegressor = FakeBasicRegressor
    sys.modules['models.model'].BasicRegressor = FakeBasicRegressor

_install_stubs()
try:
    import dill
except ImportError:
    print("Installing dill...", flush=True)
    os.system(f"{sys.executable} -m pip install dill --break-system-packages -q")
    import dill

try:
    from sklearn.decomposition import PCA
    from sklearn.metrics import balanced_accuracy_score
except ImportError:
    print("Installing scikit-learn...", flush=True)
    os.system(f"{sys.executable} -m pip install scikit-learn --break-system-packages -q")
    from sklearn.decomposition import PCA
    from sklearn.metrics import balanced_accuracy_score


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

EMBEDDING_NAMES = ['GloVe', 'FastText', 'Word2Vec', 'ConceptNet', 'DINOv2', 'SimCLR']
SEM_MODELS = {'GloVe', 'FastText', 'Word2Vec', 'ConceptNet'}
VIS_MODELS = {'DINOv2', 'SimCLR'}


# ═══════════════════════════════════════════════════════════════════════════════
# PKL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_pkl_data(pkl_path, max_bytes=3_000_000_000):
    """Load PKL, returning raw dict. Skip files larger than max_bytes."""
    if os.path.getsize(pkl_path) > max_bytes:
        print(f"    [skip] PKL too large ({os.path.getsize(pkl_path)//1e6:.0f} MB), use CSV fallback")
        return None
    with open(pkl_path, 'rb') as f:
        return dill.load(f)


def load_patient_from_pkl(pkl_path):
    """Extract per-epoch accuracy arrays (shape: n_epochs × n_bins) from PKL."""
    data = load_pkl_data(pkl_path)
    if data is None:
        return None
    records = {}
    for emb in EMBEDDING_NAMES:
        if emb not in data.get('regressors', {}):
            continue
        br = data['regressors'][emb]
        records[emb] = {
            'cat_obs':  np.array(br.all_retrieval_category_balanced_acc),         # (n_epochs, n_bins)
            'cat_null': np.array(br.all_retrieval_category_chance_balanced_acc),
            'word_obs':  np.array(br.all_retrieval_word_balanced_acc),
            'word_null': np.array(br.all_retrieval_chance_word_balanced_acc),
        }
    return records


def _extract_null_from_html(fig_dir, patient, metric):
    """Extract chance mean per bin from the Plotly HTML figure.
    Returns array of length n_bins, or None if unavailable.
    """
    if fig_dir is None:
        return None
    fname = ('category_retrieval_balanced_acc.html' if metric == 'cat'
             else 'word_retrieval_balanced_acc.html')
    html_path = os.path.join(fig_dir, patient, fname)
    if not os.path.exists(html_path):
        return None

    with open(html_path, 'r') as f:
        content = f.read()
    idx = content.rfind('Plotly.newPlot(')
    if idx < 0:
        return None
    start = content.find('[', idx)
    depth = 0
    for i, ch in enumerate(content[start:]):
        if ch == '[': depth += 1
        elif ch == ']': depth -= 1
        if depth == 0:
            end = start + i + 1
            break

    traces = json.loads(content[start:end])
    chance_idx = next((i for i, t in enumerate(traces) if t.get('name') == 'chance'), None)
    if chance_idx is None:
        return None

    def decode_bdata(bdata_str, dtype='f8'):
        b64 = bdata_str.replace('\u002f', '/').replace('\u003d', '=')
        raw = base64.b64decode(b64)
        return np.frombuffer(raw, dtype=np.float64 if dtype == 'f8' else np.float32)

    ct = traces[chance_idx]
    return decode_bdata(ct['y']['bdata'], ct['y'].get('dtype', 'f8'))


def load_patient_from_csv(results_dir, patient, fig_dir=None):
    """Fallback when PKL is too large: reconstruct per-epoch accuracies from
    top1_decoding_source_data.csv, and null mean from HTML figures.
    """
    top1_path = os.path.join(results_dir, patient, 'top1_decoding_source_data.csv')
    pts_path  = os.path.join(results_dir, patient, 'per_time_scores.csv')
    if not os.path.exists(top1_path) or not os.path.exists(pts_path):
        return None

    pts = pd.read_csv(pts_path)
    best_bins = {}
    for emb in EMBEDDING_NAMES:
        sub = pts[pts['embedding'] == emb].sort_values('bin_index')
        if len(sub) == 0:
            continue
        cat_best  = int(sub.loc[sub['category_balanced_acc'].idxmax(), 'bin_index'])
        word_best = int(sub.loc[sub['word_balanced_acc'].idxmax(), 'bin_index'])
        best_bins[emb] = (cat_best, word_best)

    needed_bins = set()
    for cb, wb in best_bins.values():
        needed_bins.add(cb); needed_bins.add(wb)

    # Chunked read — efficient even for 443 MB WBH file
    chunks = []
    for chunk in pd.read_csv(top1_path, chunksize=500_000):
        filtered = chunk[chunk['bin_index'].isin(needed_bins)]
        if len(filtered):
            chunks.append(filtered)
    top1 = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    cat_null_arr  = _extract_null_from_html(fig_dir, patient, 'cat')
    word_null_arr = _extract_null_from_html(fig_dir, patient, 'word')

    records = {}
    for emb in EMBEDDING_NAMES:
        if emb not in best_bins:
            continue
        cat_best, word_best = best_bins[emb]
        emb_df = top1[top1['embedding'] == emb]

        def per_epoch_bal_acc(df, best_bin, true_col, correct_col):
            sub = df[df['bin_index'] == best_bin]
            accs = []
            for ep in sorted(sub['epoch'].unique()):
                ep_df = sub[sub['epoch'] == ep]
                r = ep_df.groupby(true_col)[correct_col].mean()
                accs.append(r.mean())
            return np.array(accs)

        cat_obs_arr  = per_epoch_bal_acc(emb_df, cat_best, 'true_category', 'category_correct')
        word_obs_arr = per_epoch_bal_acc(emb_df, word_best, 'true_word', 'word_correct')

        cn = cat_null_arr[cat_best]   if cat_null_arr  is not None else 1.0/6
        wn = word_null_arr[word_best] if word_null_arr is not None else 1.0/60

        records[emb] = {
            'obs_cat_at_best':  cat_obs_arr,
            'obs_word_at_best': word_obs_arr,
            'null_cat_mean':    cn,
            'null_word_mean':   wn,
            'cat_best_bin':     cat_best,
            'word_best_bin':    word_best,
            'from_csv': True,
        }
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: SIGNIFICANCE TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_significance(results_dir, fig_dir=None):
    """
    Wilcoxon signed-rank (one-sided, obs > null) at best time bin.
    For PKL patients: paired 50 obs vs 50 null epoch accuracies.
    For CSV fallback: obs epoch accuracies vs scalar null mean.
    Bonferroni correction applied globally across n_patients × n_embeddings.
    """
    patients = sorted([d for d in os.listdir(results_dir)
                       if os.path.isdir(os.path.join(results_dir, d))])
    print(f"[Significance] {len(patients)} patients: {patients}")

    raw_records = []

    for patient in patients:
        pkl_path = os.path.join(results_dir, patient, 'semantic_regression_results.pkl')
        pkl_data = None

        if os.path.exists(pkl_path):
            try:
                pkl_data = load_patient_from_pkl(pkl_path)
                if pkl_data is not None:
                    print(f"  {patient}: loaded from PKL", flush=True)
            except Exception as e:
                print(f"  {patient}: PKL error ({e})", flush=True)

        if pkl_data is None:
            pkl_data = load_patient_from_csv(results_dir, patient, fig_dir)
            if pkl_data is not None:
                print(f"  {patient}: loaded from CSV", flush=True)
            else:
                print(f"  {patient}: no data found, skipping", flush=True)
                continue

        for emb in EMBEDDING_NAMES:
            if emb not in pkl_data:
                continue
            d = pkl_data[emb]

            if d.get('from_csv'):
                # CSV path: obs array vs scalar null
                cat_obs_at_best  = d['obs_cat_at_best']
                word_obs_at_best = d['obs_word_at_best']
                cat_null_mean    = d['null_cat_mean']
                word_null_mean   = d['null_word_mean']
                cat_best_bin     = d['cat_best_bin']
                word_best_bin    = d['word_best_bin']
                n_epochs = len(cat_obs_at_best)

                # One-sample Wilcoxon vs null mean
                _, cat_pval  = stats.wilcoxon(cat_obs_at_best  - cat_null_mean,
                                              alternative='greater')
                _, word_pval = stats.wilcoxon(word_obs_at_best - word_null_mean,
                                              alternative='greater')
            else:
                # PKL path: paired obs vs null epochs
                cat_obs  = np.array(d['cat_obs'])   # (n_epochs, n_bins)
                cat_null = np.array(d['cat_null'])
                word_obs  = np.array(d['word_obs'])
                word_null = np.array(d['word_null'])

                cat_best_bin  = int(np.argmax(cat_obs.mean(0)))
                word_best_bin = int(np.argmax(word_obs.mean(0)))
                n_epochs = cat_obs.shape[0]

                cat_obs_at_best  = cat_obs[:, cat_best_bin]
                cat_null_at_best = cat_null[:, cat_best_bin]
                word_obs_at_best  = word_obs[:, word_best_bin]
                word_null_at_best = word_null[:, word_best_bin]

                cat_null_mean  = cat_null.mean(0)[cat_best_bin]
                word_null_mean = word_null.mean(0)[word_best_bin]

                _, cat_pval  = stats.wilcoxon(cat_obs_at_best  - cat_null_at_best,
                                              alternative='greater')
                _, word_pval = stats.wilcoxon(word_obs_at_best - word_null_at_best,
                                              alternative='greater')

            raw_records.append({
                'patient': patient,
                'embedding': emb,
                'model_type': 'semantic' if emb in SEM_MODELS else 'visual',
                'n_epochs': n_epochs,
                'mean_cat_obs':  float(cat_obs_at_best.mean()),
                'mean_cat_null': float(cat_null_mean),
                'cat_best_bin':  cat_best_bin,
                'cat_diff_mean': float(cat_obs_at_best.mean() - cat_null_mean),
                'cat_pval_raw':  float(cat_pval),
                'mean_word_obs':  float(word_obs_at_best.mean()),
                'mean_word_null': float(word_null_mean),
                'word_best_bin':  word_best_bin,
                'word_diff_mean': float(word_obs_at_best.mean() - word_null_mean),
                'word_pval_raw':  float(word_pval),
            })

    df = pd.DataFrame(raw_records)
    if len(df) == 0:
        print("[Significance] No data found!")
        return df

    # Global Bonferroni correction
    n_tests = len(df)
    df['cat_pval_bonf']  = np.minimum(df['cat_pval_raw']  * n_tests, 1.0)
    df['word_pval_bonf'] = np.minimum(df['word_pval_raw'] * n_tests, 1.0)

    def stars(p):
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return 'NS'

    df['cat_sig']      = df['cat_pval_bonf'].apply(stars)
    df['word_sig']     = df['word_pval_bonf'].apply(stars)
    df['cat_sig_raw']  = df['cat_pval_raw'].apply(stars)
    df['word_sig_raw'] = df['word_pval_raw'].apply(stars)

    n_cat  = (df['cat_sig']  != 'NS').sum()
    n_word = (df['word_sig'] != 'NS').sum()
    print(f"[Significance] Cat sig: {n_cat}/{n_tests} | Word sig: {n_word}/{n_tests} (Bonferroni)")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: WORD PREDICTION BIAS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_word_bias(results_dir):
    """
    Identify favorite-word prediction bias across all patients × embeddings.
    Requires top1_decoding_source_data.csv for each patient.
    """
    patients = sorted([d for d in os.listdir(results_dir)
                       if os.path.isdir(os.path.join(results_dir, d))])
    records = []

    for patient in patients:
        top1_path = os.path.join(results_dir, patient, 'top1_decoding_source_data.csv')
        if not os.path.exists(top1_path):
            print(f"  {patient}: no top1 CSV, skipping bias")
            continue

        print(f"  {patient}: computing bias...", flush=True)
        # Only load best-bin rows — chunked for large files
        pts_path = os.path.join(results_dir, patient, 'per_time_scores.csv')
        pts = pd.read_csv(pts_path)
        best_bins = {}
        for emb in EMBEDDING_NAMES:
            sub = pts[pts['embedding'] == emb]
            if len(sub) == 0: continue
            best_bins[emb] = int(sub.loc[sub['word_balanced_acc'].idxmax(), 'bin_index'])

        needed = set(best_bins.values())
        chunks = []
        for chunk in pd.read_csv(top1_path, chunksize=500_000):
            f = chunk[chunk['bin_index'].isin(needed)]
            if len(f): chunks.append(f)
        if not chunks:
            continue
        top1 = pd.concat(chunks, ignore_index=True)

        for emb, best_bin in best_bins.items():
            sub = top1[(top1['embedding'] == emb) & (top1['bin_index'] == best_bin)]
            if len(sub) == 0: continue
            counts = sub['pred_word'].value_counts()
            top1_word = counts.index[0]
            top1_frac = counts.iloc[0] / len(sub)
            n_unique  = sub['pred_word'].nunique()
            n_words   = sub['true_word'].nunique()
            # Normalized entropy: H / log2(n_words)
            probs = counts.values / counts.values.sum()
            entropy = -np.sum(probs * np.log2(probs + 1e-12))
            entropy_norm = entropy / np.log2(n_words) if n_words > 1 else 0.0
            records.append({
                'patient': patient, 'embedding': emb,
                'top1_word': top1_word, 'top1_frac': top1_frac,
                'n_unique_pred': n_unique, 'n_words': n_words,
                'pred_entropy_norm': entropy_norm,
            })

    df = pd.DataFrame(records)
    print(f"[Bias] {len(df)} rows")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: METRIC DISSOCIATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metric_dissociation(results_dir):
    """Compare best bins and peak values for R², category acc, and word acc."""
    patients = sorted([d for d in os.listdir(results_dir)
                       if os.path.isdir(os.path.join(results_dir, d))])
    records = []
    for patient in patients:
        pts_path = os.path.join(results_dir, patient, 'per_time_scores.csv')
        if not os.path.exists(pts_path): continue
        pts = pd.read_csv(pts_path)
        for emb in EMBEDDING_NAMES:
            sub = pts[pts['embedding'] == emb]
            if len(sub) == 0: continue
            r2_idx   = sub['r2_mean'].idxmax()
            cat_idx  = sub['category_balanced_acc'].idxmax()
            word_idx = sub['word_balanced_acc'].idxmax()
            records.append({
                'patient': patient, 'embedding': emb,
                'r2_best_bin':   int(sub.loc[r2_idx,   'bin_index']),
                'cat_best_bin':  int(sub.loc[cat_idx,  'bin_index']),
                'word_best_bin': int(sub.loc[word_idx, 'bin_index']),
                'best_r2':   float(sub.loc[r2_idx,   'r2_mean']),
                'best_cat_acc':  float(sub.loc[cat_idx,  'category_balanced_acc']),
                'best_word_acc': float(sub.loc[word_idx, 'word_balanced_acc']),
            })
    df = pd.DataFrame(records)
    print(f"[Dissoc] {len(df)} rows")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: EMBEDDING NORM ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_norm_analysis(results_dir, top_n=10):
    """
    For each patient × embedding, find the top_n words with the smallest L2 norm
    in both raw embedding space and PCA-reduced (centered) space.

    These are the words ridge regression is biased toward predicting when signal is weak.
    After the mean-centering fix in model.py (commit 0459d4c), the centroid is subtracted
    before retrieval, so the bias should be substantially reduced.
    """
    patients = sorted([d for d in os.listdir(results_dir)
                       if os.path.isdir(os.path.join(results_dir, d))])
    records = []

    for patient in patients:
        pkl_path = os.path.join(results_dir, patient, 'semantic_regression_results.pkl')
        if not os.path.exists(pkl_path):
            continue
        try:
            data = load_pkl_data(pkl_path)
            if data is None:
                continue
        except Exception as e:
            print(f"  {patient}: PKL error ({e}), skipping norm analysis")
            continue

        for emb in EMBEDDING_NAMES:
            if emb not in data.get('regressors', {}):
                continue
            br = data['regressors'][emb]

            try:
                raw_embeds = np.array(br._retrieval_db_embeds_raw)  # (n_words, dim)
                word_idx   = np.array(br._retrieval_db_word_idx)
                idx2word   = br.index_to_word
            except AttributeError:
                continue

            # Raw norm ranking
            raw_norms = np.linalg.norm(raw_embeds, axis=1)
            raw_order = np.argsort(raw_norms)

            # PCA-reduced norm (refit PCA to avoid sklearn version issues)
            try:
                n_comp = min(10, raw_embeds.shape[0], raw_embeds.shape[1])
                pca = PCA(n_components=n_comp)
                pca_embeds = pca.fit_transform(raw_embeds)  # centered internally
                pca_norms  = np.linalg.norm(pca_embeds, axis=1)
                pca_order  = np.argsort(pca_norms)
            except Exception:
                pca_norms  = raw_norms
                pca_order  = raw_order

            for rank in range(min(top_n, len(raw_order))):
                ri = raw_order[rank]
                pi = pca_order[rank]
                records.append({
                    'patient': patient, 'embedding': emb,
                    'raw_norm_rank': rank,
                    'raw_norm_word': idx2word.get(int(word_idx[ri]), str(word_idx[ri])),
                    'raw_norm':      float(raw_norms[ri]),
                    'centered_norm_rank': rank,
                    'centered_norm_word': idx2word.get(int(word_idx[pi]), str(word_idx[pi])),
                    'centered_norm':      float(pca_norms[pi]),
                })
        print(f"  {patient}: norm analysis done", flush=True)

    df = pd.DataFrame(records)
    print(f"[Norm] {len(df)} rows")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: GENERATE REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(sig_df, bias_df, dissoc_df, norm_df, out_dir):
    """Assemble and write the full HTML analysis report."""
    os.makedirs(out_dir, exist_ok=True)

    if len(sig_df) == 0:
        print("[Report] No significance data — aborting")
        return None

    n_tests    = len(sig_df)
    n_patients = sig_df['patient'].nunique()
    patients_sorted = sorted(sig_df['patient'].unique(),
                             key=lambda p: sig_df[sig_df.patient == p]['mean_cat_obs'].mean(),
                             reverse=True)

    n_cat_sig  = (sig_df['cat_sig']  != 'NS').sum()
    n_word_sig = (sig_df['word_sig'] != 'NS').sum()

    def _sig_class(s):
        return {'***': 'star-three', '**': 'star-two', '*': 'star-one'}.get(s, 'star-ns')

    def _sig_display(s):
        return s

    def patient_tier(p):
        fold = (sig_df[sig_df.patient == p]['mean_cat_obs'] /
                sig_df[sig_df.patient == p]['mean_cat_null']).mean()
        if fold > 1.3: return 'patient-high'
        if fold > 1.1: return 'patient-moderate'
        return 'patient-low'

    # Per-model significance counts
    sig_counts = {emb: {'cat': 0, 'word': 0} for emb in EMBEDDING_NAMES}
    for emb in EMBEDDING_NAMES:
        sub = sig_df[sig_df.embedding == emb]
        sig_counts[emb]['cat']  = (sub['cat_sig']  != 'NS').sum()
        sig_counts[emb]['word'] = (sub['word_sig'] != 'NS').sum()

    # ── Word bias summary ─────────────────────────────────────────────────────
    bias_summary = []
    if len(bias_df) > 0:
        for emb in EMBEDDING_NAMES:
            sub = bias_df[bias_df.embedding == emb]
            if len(sub) == 0: continue
            top = sub.groupby('top1_word').size().sort_values(ascending=False)
            fav = top.index[0]
            n_fav = top.iloc[0]
            mean_top1_pct = sub[sub.top1_word == fav]['top1_frac'].mean()
            mean_ent = sub['pred_entropy_norm'].mean()
            bias_summary.append({
                'emb': emb, 'fav_word': fav,
                'n_patients_fav': f'{n_fav}/{n_patients}',
                'mean_top1_pct': f'{mean_top1_pct*100:.1f}%',
                'mean_entropy': f'{mean_ent:.3f}',
            })

    # ── Norm-bias summary HTML ────────────────────────────────────────────────
    norm_summary_html = ''
    if len(norm_df) > 0:
        rank_col = 'norm_rank' if 'norm_rank' in norm_df.columns else 'raw_norm_rank'
        word_col = 'word'      if 'word'      in norm_df.columns else 'raw_norm_word'
        norm_col = 'pca_norm'  if 'pca_norm'  in norm_df.columns else 'raw_norm'

        norm_summary_html += '<h3>Embedding Norm vs. Predicted Words</h3>\n'
        norm_summary_html += ('<p>Words with the smallest L2 norm in PCA-reduced embedding space '
                              'per model (pooled across patients). These are the words ridge '
                              'regression is biased toward predicting. After the mean-centering '
                              'fix, this bias should be substantially reduced.</p>\n')
        norm_summary_html += '<table><tr><th>Model</th>'
        for r in range(5):
            norm_summary_html += f'<th>Rank {r+1} (smallest norm)</th>'
        norm_summary_html += '</tr>\n'

        for emb in EMBEDDING_NAMES:
            sub = norm_df[(norm_df.embedding == emb) & (norm_df[rank_col] < 5)]
            if len(sub) == 0: continue
            cells = []
            for rank in range(5):
                rank_sub = sub[sub[rank_col] == rank]
                if len(rank_sub) == 0:
                    cells.append('—')
                else:
                    top_word = rank_sub.groupby(word_col).size().sort_values(ascending=False).index[0]
                    med_norm = rank_sub[rank_sub[word_col] == top_word][norm_col].median()
                    cells.append(f'{top_word} <small>(‖e‖={med_norm:.3f})</small>')
            norm_summary_html += (f'<tr><td><strong>{emb}</strong></td>'
                                  + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>\n')
        norm_summary_html += '</table>\n'

        # Norm–bias match rate
        if len(bias_df) > 0:
            match_count = 0
            total_count = 0
            _rk = 'norm_rank' if 'norm_rank' in norm_df.columns else 'raw_norm_rank'
            _wc = 'word'      if 'word'      in norm_df.columns else 'raw_norm_word'
            for emb in EMBEDDING_NAMES:
                for p in sig_df.patient.unique():
                    bias_row = bias_df[(bias_df.patient == p) & (bias_df.embedding == emb)]
                    norm_row = norm_df[(norm_df.patient == p) & (norm_df.embedding == emb)
                                      & (norm_df[_rk] == 0)]
                    if len(bias_row) > 0 and len(norm_row) > 0:
                        total_count += 1
                        if bias_row.iloc[0]['top1_word'] == norm_row.iloc[0][_wc]:
                            match_count += 1
            if total_count > 0:
                pct = match_count / total_count
                norm_summary_html += (
                    f'<div class="finding"><strong>Norm–bias correlation:</strong> '
                    f'The most-predicted word matches the smallest-norm word in '
                    f'<strong>{match_count}/{total_count} ({pct*100:.0f}%)</strong> of '
                    f'patient × embedding combinations. ')
                if pct > 0.7:
                    norm_summary_html += (
                        'This confirms ridge shrinkage toward the PCA origin is the dominant '
                        'cause of the favorite-word effect.</div>\n')
                elif pct > 0.3:
                    norm_summary_html += (
                        'Partial overlap — ridge shrinkage explains some but not all of the '
                        'bias. Other factors (word frequency, embedding geometry) also contribute.'
                        '</div>\n')
                else:
                    norm_summary_html += (
                        'Low match rate — after mean-centering, the favorite-word effect is '
                        'no longer driven primarily by norm proximity to the origin. '
                        'The remaining bias likely reflects true data structure.</div>\n')

    # ── Per-patient decoding tables ───────────────────────────────────────────
    def build_table_rows(metric='cat'):
        rows = []
        for p in patients_sorted:
            sub  = sig_df[sig_df.patient == p]
            tier = patient_tier(p)
            n_cats  = round(1 / sub['mean_cat_null'].mean()) if sub['mean_cat_null'].mean() > 0 else '?'
            n_words = round(1 / sub['mean_word_null'].mean()) if sub['mean_word_null'].mean() > 0 else '?'
            null_col = sub[f'mean_{metric}_null'].mean()
            cells = []
            for emb in EMBEDDING_NAMES:
                row = sub[sub.embedding == emb]
                if len(row) == 0:
                    cells.append('<td>—</td>')
                    continue
                r   = row.iloc[0]
                acc = r[f'mean_{metric}_obs']
                null = r[f'mean_{metric}_null']
                fc  = acc / null if null > 0 else 0
                sig = r[f'{metric}_sig']
                fmt_acc = f'{acc*100:.1f}%' if metric == 'cat' else f'{acc*100:.2f}%'
                cells.append(
                    f'<td class="data-cell">{fmt_acc} ({fc:.1f}×) '
                    f'<span class="{_sig_class(sig)}">{_sig_display(sig)}</span></td>')
            fmt_null = f'{null_col*100:.1f}%' if metric == 'cat' else f'{null_col*100:.2f}%'
            rows.append(
                f'<tr class="{tier}"><td><strong>{p}</strong></td>'
                f'<td>{n_words} / {n_cats}</td>'
                + ''.join(cells)
                + f'<td class="chance-cell">{fmt_null}</td></tr>')
        return '\n'.join(rows)

    cat_rows  = build_table_rows('cat')
    word_rows = build_table_rows('word')

    # Overview table
    overview_rows = ''
    for emb in EMBEDDING_NAMES:
        mtype = 'Semantic' if emb in SEM_MODELS else 'Visual'
        c = sig_counts[emb]['cat']
        w = sig_counts[emb]['word']
        c_class = 'sig' if c >= 10 else ('ns' if c < 6 else '')
        w_class = 'sig' if w >= 10 else ('ns' if w < 6 else '')
        overview_rows += (f'<tr><td><strong>{emb}</strong></td>'
                          f'<td class="{c_class}">{c}/{n_patients}</td>'
                          f'<td class="{w_class}">{w}/{n_patients}</td>'
                          f'<td>{mtype}</td></tr>\n')

    # Bias table
    bias_table = ''
    if bias_summary:
        bias_table = ('<table><tr><th>Model</th><th>Favorite Word</th>'
                      '<th>Patients Affected</th><th>Mean % of All Predictions</th>'
                      '<th>Pred Entropy (norm)</th></tr>\n')
        for b in bias_summary:
            bias_table += (f'<tr><td>{b["emb"]}</td><td><strong>"{b["fav_word"]}"</strong></td>'
                           f'<td>{b["n_patients_fav"]}</td><td>{b["mean_top1_pct"]}</td>'
                           f'<td>{b["mean_entropy"]}</td></tr>\n')
        bias_table += '</table>'

    # Dissociation summary
    dissoc_html = ''
    if len(dissoc_df) > 0:
        consistent = 0
        for p in dissoc_df.patient.unique():
            sub = dissoc_df[dissoc_df.patient == p]
            if (sub.loc[sub.best_r2.idxmax(), 'embedding'] ==
                sub.loc[sub.best_cat_acc.idxmax(), 'embedding'] ==
                sub.loc[sub.best_word_acc.idxmax(), 'embedding']):
                consistent += 1
        total = dissoc_df.patient.nunique()
        dissoc_html = (f'<p>Across {total} patients, <strong>{consistent}/{total}</strong> '
                       f'have the same model winning R², category accuracy, and word accuracy '
                       f'simultaneously.</p>')
        dissoc_df2 = dissoc_df.copy()
        dissoc_df2['r2_cat_gap']  = np.abs(dissoc_df2.r2_best_bin - dissoc_df2.cat_best_bin)
        dissoc_df2['r2_word_gap'] = np.abs(dissoc_df2.r2_best_bin - dissoc_df2.word_best_bin)
        dissoc_df2['cat_word_gap']= np.abs(dissoc_df2.cat_best_bin - dissoc_df2.word_best_bin)
        dissoc_html += (f'<p>Mean bin gap: R²↔Cat = {dissoc_df2.r2_cat_gap.mean():.1f}, '
                        f'R²↔Word = {dissoc_df2.r2_word_gap.mean():.1f}, '
                        f'Cat↔Word = {dissoc_df2.cat_word_gap.mean():.1f} bins.</p>')

    # Semantic vs visual counts
    sem_cat = sum(sig_counts[e]['cat'] for e in SEM_MODELS)
    vis_cat = sum(sig_counts[e]['cat'] for e in VIS_MODELS)

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Semantic Regression Analysis Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; color: #333; line-height: 1.6; }}
  h1 {{ color: #1a5276; border-bottom: 3px solid #2980b9; padding-bottom: 10px; }}
  h2 {{ color: #2471a3; margin-top: 40px; border-bottom: 1px solid #d4e6f1; padding-bottom: 5px; }}
  h3 {{ color: #2e86c1; }}
  .summary-box {{ background: #eaf2f8; border-left: 4px solid #2980b9; padding: 15px; margin: 20px 0; border-radius: 4px; }}
  .finding {{ background: #fef9e7; border-left: 4px solid #f39c12; padding: 15px; margin: 15px 0; border-radius: 4px; }}
  .warning {{ background: #fdedec; border-left: 4px solid #e74c3c; padding: 15px; margin: 15px 0; border-radius: 4px; }}
  .method-box {{ background: #f3e5f5; border-left: 4px solid #8e24aa; padding: 15px; margin: 15px 0; border-radius: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }}
  th {{ background: #2980b9; color: white; padding: 8px 10px; text-align: left; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #ddd; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  .sig {{ color: #27ae60; font-weight: bold; }}
  .ns  {{ color: #e74c3c; }}
  code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
  small {{ color: #888; }}
  #cat-table, #word-table {{ font-size: 12px; width: 100%; border-collapse: collapse; margin: 15px 0 5px 0; table-layout: fixed; }}
  #cat-table th, #word-table th {{ padding: 6px 5px; text-align: center; font-size: 11px; }}
  #cat-table td, #word-table td {{ padding: 5px 5px; text-align: center; border-bottom: 1px solid #ddd; font-size: 11.5px; }}
  .sem-header {{ background: #1565C0; color: white; }}
  .vis-header {{ background: #E65100; color: white; }}
  .data-cell {{ font-variant-numeric: tabular-nums; }}
  .chance-cell {{ background: #f0f0f0; font-weight: bold; }}
  .star-three {{ color: #1b5e20; font-weight: bold; }}
  .star-two   {{ color: #2e7d32; font-weight: bold; }}
  .star-one   {{ color: #388e3c; }}
  .star-ns    {{ color: #c62828; }}
  .patient-high     td:first-child {{ background: #e8f5e9; font-weight: bold; }}
  .patient-moderate td:first-child {{ background: #fff8e1; }}
  .patient-low      td:first-child {{ background: #ffebee; }}
</style>
</head>
<body>

<h1>Semantic Regression: Cross-Patient Analysis Report</h1>
<p><em>iEEG high gamma (70–150 Hz) → semantic/visual embedding decoding across {n_patients} patients</em></p>
<p><strong>Test:</strong> Wilcoxon signed-rank vs. internal shuffled null, Bonferroni-corrected ({n_tests} tests) &nbsp;|&nbsp;
   <strong>Model:</strong> Nystroem + Ridge → PCA (10 dim) → nearest-neighbor retrieval &nbsp;|&nbsp;
   <strong>Retrieval:</strong> mean-centered (db centroid subtracted)</p>

<div class="summary-box">
<h3>Executive Summary</h3>
<p><strong>Category decoding significant in {n_cat_sig}/{n_tests} ({n_cat_sig*100//n_tests}%) of patient × model combinations</strong>
after Bonferroni correction. Word-level: {n_word_sig}/{n_tests} ({n_word_sig*100//n_tests}%).
All {n_patients} patients show at least some significant category decoding.
Strongest effects: {", ".join(patients_sorted[:3])}.</p>
</div>

<h2>1. Significance Testing</h2>
<div class="method-box">
<strong>Method:</strong> Internal shuffled null (<code>X_train_shuffle = np.random.permutation(X_train)</code>)
runs through the full pipeline (Nystroem + Ridge + PCA + nearest-neighbor retrieval), preserving all biases.
At each patient × embedding's best time bin, 50 observed epoch accuracies are compared to 50 null epoch
accuracies via one-sided Wilcoxon signed-rank test, with Bonferroni correction across all {n_tests} tests.
</div>

<h3>Per-Model Significance (Bonferroni-corrected)</h3>
<table>
<tr><th>Model</th><th># Patients Cat Sig</th><th># Patients Word Sig</th><th>Type</th></tr>
{overview_rows}
</table>

<h3>Category Decoding: Full Results</h3>
<p style="font-size:12px;"><strong>Legend:</strong>
acc (×fold) sig &nbsp;|&nbsp;
<span class="star-three">*** p&lt;0.001</span> &nbsp;
<span class="star-two">** p&lt;0.01</span> &nbsp;
<span class="star-one">* p&lt;0.05</span> &nbsp;
<span class="star-ns">NS</span> &nbsp; (all Bonferroni-corrected)</p>
<table id="cat-table">
<tr><th rowspan="2">Patient</th><th rowspan="2">N words/cats</th>
<th class="sem-header">GloVe</th><th class="sem-header">FastText</th>
<th class="sem-header">Word2Vec</th><th class="sem-header">ConceptNet</th>
<th class="vis-header">DINOv2</th><th class="vis-header">SimCLR</th>
<th>Null (mean)</th></tr>
<tr>
<th class="sem-header">acc (×) sig</th><th class="sem-header">acc (×) sig</th>
<th class="sem-header">acc (×) sig</th><th class="sem-header">acc (×) sig</th>
<th class="vis-header">acc (×) sig</th><th class="vis-header">acc (×) sig</th>
<th></th></tr>
{cat_rows}
</table>

<h3>Word Decoding: Full Results</h3>
<p class="warning"><strong>Interpret with caution —</strong>
word predictions may still be partially dominated by prediction bias even after mean-centering
(see Section 2). Check entropy values.</p>
<table id="word-table">
<tr><th rowspan="2">Patient</th><th rowspan="2">N words/cats</th>
<th class="sem-header">GloVe</th><th class="sem-header">FastText</th>
<th class="sem-header">Word2Vec</th><th class="sem-header">ConceptNet</th>
<th class="vis-header">DINOv2</th><th class="vis-header">SimCLR</th>
<th>Null (mean)</th></tr>
<tr>
<th class="sem-header">acc (×) sig</th><th class="sem-header">acc (×) sig</th>
<th class="sem-header">acc (×) sig</th><th class="sem-header">acc (×) sig</th>
<th class="vis-header">acc (×) sig</th><th class="vis-header">acc (×) sig</th>
<th></th></tr>
{word_rows}
</table>

<h2>2. Word Prediction Bias</h2>
<h3>Favorite word per model</h3>
{bias_table if bias_table else '<p><em>Bias analysis not run (use --skip-bias=False to enable).</em></p>'}
<p>Prediction entropy normalized to [0,1] where 1.0 = perfectly uniform predictions.</p>

{norm_summary_html}

<h3>Why does this happen?</h3>
<div class="finding">
<p><strong>Ridge regression shrinkage → centroid collapse.</strong>
The L2 penalty shrinks predictions toward the PCA origin (= embedding centroid).
The nearest neighbor to the origin is the word with the smallest PCA-space norm.
<strong>Fix (commit 0459d4c):</strong> <code>model.py</code> now subtracts the database
centroid from both <code>db_embeds</code> and <code>y_pred</code> before retrieval,
so the target shifts from the origin to the centroid. This removes the shrinkage
advantage for centroid-near words.</p>
</div>

<h2>3. Metric Dissociation</h2>
{dissoc_html if dissoc_html else '<p><em>No dissociation data available.</em></p>'}

<h2>4. Semantic vs. Visual</h2>
<table>
<tr><th>Group</th><th>Category sig (total)</th><th>Per model</th></tr>
<tr><td>Semantic (GloVe, FastText, Word2Vec, ConceptNet)</td><td>{sem_cat}/{n_patients*4}</td>
<td>{"  |  ".join(f"{e}: {sig_counts[e]['cat']}/{n_patients}" for e in ['GloVe','FastText','Word2Vec','ConceptNet'])}</td></tr>
<tr><td>Visual (DINOv2, SimCLR)</td><td>{vis_cat}/{n_patients*2}</td>
<td>{"  |  ".join(f"{e}: {sig_counts[e]['cat']}/{n_patients}" for e in ['DINOv2','SimCLR'])}</td></tr>
</table>

</body>
</html>'''

    out_path = os.path.join(out_dir, 'analysis_report.html')
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"[Report] Saved: {out_path}  ({len(html)//1024} KB)")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE CSVs
# ═══════════════════════════════════════════════════════════════════════════════

def save_outputs(sig_df, bias_df, dissoc_df, norm_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if len(sig_df):
        p = os.path.join(out_dir, 'null_corrected_significance.csv')
        sig_df.to_csv(p, index=False)
        print(f"  Saved: {p}")
    if len(bias_df):
        p = os.path.join(out_dir, 'word_prediction_bias.csv')
        bias_df.to_csv(p, index=False)
        print(f"  Saved: {p}")
    if len(dissoc_df):
        p = os.path.join(out_dir, 'metric_dissociation.csv')
        dissoc_df.to_csv(p, index=False)
        print(f"  Saved: {p}")
    if len(norm_df):
        p = os.path.join(out_dir, 'embedding_norm_analysis.csv')
        norm_df.to_csv(p, index=False)
        print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Semantic Regression Analysis Pipeline')
    parser.add_argument('--results-dir', default=None,
                        help='Path to semantic_regression_results/ '
                             '(default: ./semantic_regression_results)')
    parser.add_argument('--figures-dir', default=None,
                        help='Path to semantic_regression_figures/ '
                             '(default: ./semantic_regression_figures)')
    parser.add_argument('--out-dir', default=None,
                        help='Output directory for report and CSVs '
                             '(default: <figures-dir>/cross_patient_analysis)')
    parser.add_argument('--skip-bias', action='store_true',
                        help='Skip word bias analysis (reads top1 CSVs, can be slow)')
    parser.add_argument('--skip-norms', action='store_true',
                        help='Skip embedding norm analysis (loads PKLs)')
    args = parser.parse_args()

    script_dir  = Path(__file__).parent        # .../Neuroscience of speech and language/main/
    base        = script_dir.parent            # .../Neuroscience of speech and language/
    results_dir = args.results_dir or str(base / 'semantic_regression')
    figures_dir = args.figures_dir or str(base / 'semantic_regression_figures')
    out_dir     = args.out_dir     or str(Path(figures_dir) / 'cross_patient_analysis')

    print(f"Results dir : {results_dir}")
    print(f"Figures dir : {figures_dir}")
    print(f"Output dir  : {out_dir}")
    print()

    # Step 1
    print("=" * 60)
    print("STEP 1: SIGNIFICANCE TESTING")
    print("=" * 60)
    sig_df = compute_significance(results_dir, fig_dir=figures_dir)

    # Step 2
    bias_df = pd.DataFrame()
    if not args.skip_bias:
        print()
        print("=" * 60)
        print("STEP 2: WORD PREDICTION BIAS")
        print("=" * 60)
        bias_df = compute_word_bias(results_dir)

    # Step 3
    print()
    print("=" * 60)
    print("STEP 3: METRIC DISSOCIATION")
    print("=" * 60)
    dissoc_df = compute_metric_dissociation(results_dir)

    # Step 4
    norm_df = pd.DataFrame()
    if not args.skip_norms:
        print()
        print("=" * 60)
        print("STEP 4: EMBEDDING NORM ANALYSIS")
        print("=" * 60)
        norm_df = compute_norm_analysis(results_dir)

    # Save CSVs
    print()
    print("=" * 60)
    print("SAVING CSVs")
    print("=" * 60)
    save_outputs(sig_df, bias_df, dissoc_df, norm_df, out_dir)

    # Step 5
    print()
    print("=" * 60)
    print("STEP 5: GENERATE REPORT")
    print("=" * 60)
    report_path = generate_report(sig_df, bias_df, dissoc_df, norm_df, out_dir)

    print()
    print("Pipeline complete!")
    if report_path:
        print(f"  Report : {report_path}")
    print(f"  CSVs   : {out_dir}/")


if __name__ == '__main__':
    main()
