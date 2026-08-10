# -*- coding: utf-8 -*-
"""
figures_for_paper/language_vs_visual — compute step.

Builds the source-data CSVs behind the paper figure
"Decoded picture-naming information reflects linguistic rather than visual structure".

Strict 2-vs-2 family contrast on the pinned picture-naming run
(``utils.config.PIC_RUN``), all 12 picture participants:

    language family = {GloVe, Word2Vec}   (distributional, text-only)
    vision   family = {DINOv3, MoCo}       (self-supervised, image-only; ViT + CNN)

All decoding is held-out (cross-validated test trials — that is what the run stored).

Inputs (fast; no big PKLs are loaded):
  * category & word obs/null means  → ../../figures/language_vs_visual/source_data/
        cache_null_means_100ep.csv   (obs & shuffled-null balanced accuracy per
        participant × embedding × bin; built from this same run's PKLs)
  * cosine-similarity decoding fidelity → each participant's per_time_scores.csv
        (column `cosine_mean`; cosine chance ≈ 0 by construction)
  * cue timings → each participant's cue_stats.json
  * bin geometry → the run's meta.json

Outputs → ./source_data/:
  panel_a_cosine_timecourse.csv     group mean±SEM cosine per model, over time
  panel_b_category_timecourse.csv   group mean±SEM category effect per family + language−vision LMM raster
  panel_c_preference_delta.csv      per-participant Δ(language−vision), category & word, + group Wilcoxon
  panel_d_peak_effects.csv          per-embedding peak category effect + bootstrap 95% CI
  panel_e_perbin_significance.csv   per-bin language−vision difference (cosine/word/category) + FDR
  panel_f_layer_sweep.csv           vision-model accuracy vs layer depth + language reference (if layer_sweep.csv exists)
  cue_timing.csv                    aggregated cue mean ± s.d.
  group_inference.csv               every headline number the Results text cites

Run (Speech conda env; cwd = main/):
  python figures_for_paper/language_vs_visual/compute_language_vs_visual_data.py
"""

import os
import re
import sys
import json
import glob
import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS_ROOT = os.path.dirname(HERE)                      # …/figures_for_paper
MAIN_DIR = os.path.dirname(FIGS_ROOT)                  # …/main
sys.path.insert(0, FIGS_ROOT)
from paper_common import display_id                    # noqa: E402  (display-ID mapping)
from utils.config import PIC_RUN, ALPHA, p_stars       # noqa: E402  (pinned run + cutoff)
from utils.config import ROI_ATLAS_DEFAULT as ROI_ATLAS  # noqa: E402  (atlas this figure reports)

RUN = PIC_RUN
RUN_DIR = os.path.join(MAIN_DIR, 'results', 'semantic_regression', RUN)
CACHE_CW = os.path.join(MAIN_DIR, 'figures', 'language_vs_visual', 'source_data',
                        'cache_null_means_100ep.csv')
# The vision layer sweep is produced by run_vision_layer_sweep.py into ONE file.
#
# This used to also glob ``results/layer_sweep_*/`` because that script overwrote rather
# than merged, so a participant added later had to be written to its own shard directory --
# ``results/layer_sweep_KAW/`` was exactly that. The script merges on (patient, roi_atlas)
# as of 2026-08-08, the whole cohort was re-swept into the single file on 2026-08-10, and
# the shard was retired, so the glob has nothing left to find. Keeping it would only invite
# a stale shard to reappear in the panel unnoticed.
LAYER_SWEEP_GLOBS = [os.path.join(MAIN_DIR, 'results', 'layer_sweep', 'layer_sweep.csv')]
SRC_DIR = os.path.join(HERE, 'source_data')

LANG = ['GloVe', 'Word2Vec']
VIS = ['DINOv3', 'MoCo']
EMB = LANG + VIS
FAMILY = {**{e: 'language' for e in LANG}, **{e: 'vision' for e in VIS}}

N_BINS = 50            # common analysis window across participants (0–4900 ms of stored bins)
FDR_Q = ALPHA          # BH-FDR q; the repo-wide cutoff (utils/config.py)


# ── Load ──────────────────────────────────────────────────────────────────────

def _meta():
    p = os.path.join(RUN_DIR, 'meta.json')
    m = json.load(open(p)) if os.path.exists(p) else {}
    return int(m.get('n_bins_history', 10)), float(m.get('bin_size_ms', 100))


def _load_long(n_hist, bin_ms):
    """One tidy frame: patient, embedding, bin_index, time_s, cat_ac, word_ac, cosine.
    cat_ac/word_ac are above-chance (obs − shuffled null); cosine is the raw decoding
    cosine (chance ≈ 0). Trimmed to the first N_BINS bins."""
    cw = pd.read_csv(CACHE_CW)
    cw = cw[cw.embedding.isin(EMB) & (cw.bin_index < N_BINS)].copy()
    cw['cat_ac'] = cw['cat_obs_mean'] - cw['cat_null_mean']
    cw['word_ac'] = cw['word_obs_mean'] - cw['word_null_mean']
    cw = cw[['patient', 'embedding', 'bin_index', 'cat_ac', 'word_ac']]

    cos_rows = []
    for d in sorted(glob.glob(os.path.join(RUN_DIR, '*', 'per_time_scores.csv'))):
        t = pd.read_csv(d, usecols=['patient', 'embedding', 'bin_index',
                                    'cosine_mean', 'r2_mean', 'chance_mean'])
        t = t[t.embedding.isin(EMB) & (t.bin_index < N_BINS)].copy()
        t['r2_ac'] = t['r2_mean'] - t['chance_mean']      # R² above its own shuffled chance
        cos_rows.append(t[['patient', 'embedding', 'bin_index', 'cosine_mean', 'r2_ac']])
    cos = pd.concat(cos_rows, ignore_index=True).rename(columns={'cosine_mean': 'cosine'})

    df = cw.merge(cos, on=['patient', 'embedding', 'bin_index'], how='inner')
    df['time_s'] = (df['bin_index'] - n_hist) * bin_ms / 1000.0
    return df


def _family_by_patient(df, value):
    """(patient, bin_index, time_s) → language mean, vision mean of `value`."""
    g = (df.assign(fam=df.embedding.map(FAMILY))
           .groupby(['patient', 'bin_index', 'time_s', 'fam'])[value].mean()
           .unstack('fam').reset_index())
    return g.rename(columns={'language': 'lang', 'vision': 'vis'})


# ── Per-bin language−vision test (linear mixed model, participant random intercept) ──

def _perbin_lmm(fam):
    """For each bin: language−vision difference tested with a linear mixed model
    (value ~ family + (1|participant)); balanced 2-level paired design. Returns a
    frame per bin: diff (mean lang−vis), p_one (one-sided, language>vision), and
    BH-FDR q / significant over post-onset bins (time_s ≥ 0)."""
    import statsmodels.formula.api as smf
    rows = []
    for (b, t), grp in fam.groupby(['bin_index', 'time_s']):
        g = grp.dropna(subset=['lang', 'vis'])
        pts = g['patient'].tolist()
        lang, vis = g['lang'].values, g['vis'].values
        diff = float(np.mean(lang - vis))
        p_one = np.nan
        if len(pts) >= 3:
            dd = pd.DataFrame({'val': np.concatenate([lang, vis]),
                               'fam': ['lang'] * len(pts) + ['vis'] * len(pts),
                               'pt': pts + pts})
            try:
                m = smf.mixedlm('val ~ C(fam, Treatment(reference="vis"))', dd,
                                groups=dd['pt']).fit(reml=False, method='lbfgs')
                key = [k for k in m.params.index if k.startswith('C(fam')][0]
                coef, p2 = float(m.params[key]), float(m.pvalues[key])
                p_one = p2 / 2 if coef > 0 else 1 - p2 / 2
            except Exception:
                pass
        if not np.isfinite(p_one):                      # fallback: paired t-test
            tt = sps.ttest_rel(lang, vis)
            p_one = tt.pvalue / 2 if diff > 0 else 1 - tt.pvalue / 2
        rows.append(dict(bin_index=int(b), time_s=float(t), diff=diff, p_one=float(p_one)))
    out = pd.DataFrame(rows).sort_values('bin_index').reset_index(drop=True)
    post = out['time_s'] >= 0
    out['q'] = np.nan
    if post.any():
        out.loc[post, 'q'] = multipletests(out.loc[post, 'p_one'], method='fdr_bh')[1]
    out['significant'] = (out['q'] < FDR_Q) & post
    return out


def _group_curve(fam, col):
    """Across-participant mean ± SEM of a family column, per bin."""
    g = fam.groupby(['bin_index', 'time_s'])[col].agg(['mean', 'sem', 'count']).reset_index()
    return g


# ── Panels ────────────────────────────────────────────────────────────────────

def supp_cosine_timecourse(df):
    """Supplement: group mean±SEM cosine decoding fidelity per model (4 lines)."""
    g = (df.groupby(['embedding', 'bin_index', 'time_s'])['cosine']
           .agg(['mean', 'sem', 'count']).reset_index())
    g['family'] = g.embedding.map(FAMILY)
    g = g.sort_values(['embedding', 'bin_index'])
    g.to_csv(os.path.join(SRC_DIR, 'supp_cosine_decoding_timecourse.csv'), index=False)
    return g


def panel_b_category(df):
    """Category effect (above-chance) per family over time + language−vision LMM raster."""
    fam = _family_by_patient(df, 'cat_ac')
    lang = _group_curve(fam, 'lang').assign(family='language')
    vis = _group_curve(fam, 'vis').assign(family='vision')
    curves = pd.concat([lang, vis], ignore_index=True)
    sig = _perbin_lmm(fam)
    out = curves.merge(sig[['bin_index', 'diff', 'p_one', 'q', 'significant']],
                       on='bin_index', how='left').sort_values(['family', 'bin_index'])
    out.to_csv(os.path.join(SRC_DIR, 'panel_b_category_timecourse.csv'), index=False)
    return out, sig


def panel_c_r2(df):
    """R² effect (R² − chance) per family over time + language−vision LMM raster (mirror of b)."""
    fam = _family_by_patient(df, 'r2_ac')
    lang = _group_curve(fam, 'lang').assign(family='language')
    vis = _group_curve(fam, 'vis').assign(family='vision')
    curves = pd.concat([lang, vis], ignore_index=True)
    sig = _perbin_lmm(fam)
    out = curves.merge(sig[['bin_index', 'diff', 'p_one', 'q', 'significant']],
                       on='bin_index', how='left').sort_values(['family', 'bin_index'])
    out.to_csv(os.path.join(SRC_DIR, 'panel_c_r2_timecourse.csv'), index=False)
    return out, sig


def panel_c_preference(df):
    """Per-participant post-stimulus Δ(language−vision) for category and word, + group Wilcoxon."""
    rows = []
    post = df['time_s'] >= 0
    for metric, col in [('category', 'cat_ac'), ('word', 'word_ac')]:
        fam = _family_by_patient(df[post], col)
        pm = fam.groupby('patient')[['lang', 'vis']].mean().reset_index()
        for _, r in pm.iterrows():
            rows.append(dict(display_id=display_id(r['patient']), patient=r['patient'],
                             metric=metric, language=r['lang'], vision=r['vis'],
                             delta=r['lang'] - r['vis'], prefers_language=bool(r['lang'] > r['vis'])))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(SRC_DIR, 'panel_e_preference_delta.csv'), index=False)

    grp = []
    for metric in ['category', 'word']:
        d = out[out.metric == metric]['delta'].values
        n_pref = int((d > 0).sum())
        try:
            w = sps.wilcoxon(d, alternative='greater')
            stat, p = float(w.statistic), float(w.pvalue)
        except ValueError:
            stat, p = np.nan, np.nan
        grp.append(dict(metric=metric, n_participants=len(d), n_prefer_language=n_pref,
                        mean_delta=float(np.mean(d)), wilcoxon_stat=stat, wilcoxon_p_greater=p))
    return out, pd.DataFrame(grp)


# ── Panel d — peak-bin between-model comparison (bars + per-participant dots + stars) ──

PAIRS = [('GloVe', 'DINOv3'), ('GloVe', 'MoCo'),
         ('Word2Vec', 'DINOv3'), ('Word2Vec', 'MoCo')]
PAIR_METRICS = [('r2', 'r2_ac'), ('category', 'cat_ac'), ('word', 'word_ac')]


def _semantic_peak_bin(df):
    """Group category-accuracy peak bin (t≥0): argmax across-participant mean cat_ac pooled
    over all four models — the single 'semantic peak' at which the model comparison is made."""
    post = df[df.time_s >= 0]
    g = post.groupby(['bin_index', 'time_s'])['cat_ac'].mean().reset_index()
    row = g.loc[g['cat_ac'].idxmax()]
    return int(row['bin_index']), float(row['time_s'])


def _stars(p):
    """Star ladder — thresholds come from utils.config.p_stars (one ladder, repo-wide)."""
    return p_stars(p)


def panel_d_peak_pairwise(df):
    """At the single semantic peak bin, per (language>vision) pair and metric (R²/category/word):
    per-participant diff = effect_language − effect_vision; group mean, SEM, one-sided Wilcoxon
    (language>vision), BH-FDR across the 12 pair×metric tests, and a star. Writes the per-participant
    long table (for the dots) and the per-cell stats (for the bars/stars)."""
    b_star, t_star = _semantic_peak_bin(df)
    at = df[df.bin_index == b_star]
    long_rows, stat_rows = [], []
    for mkey, mcol in PAIR_METRICS:
        wide = at.pivot_table(index='patient', columns='embedding', values=mcol)
        for A, B in PAIRS:
            sub = wide[[A, B]].dropna()
            diff = (sub[A] - sub[B])
            for pat, d in diff.items():
                long_rows.append(dict(display_id=display_id(pat), patient=pat, metric=mkey,
                                      pair=f'{A}>{B}', a_model=A, b_model=B, diff=float(d)))
            try:
                p = float(sps.wilcoxon(diff.values, alternative='greater').pvalue)
            except ValueError:
                p = np.nan
            stat_rows.append(dict(metric=mkey, pair=f'{A}>{B}', a_model=A, b_model=B,
                                  bin_index=b_star, peak_time_s=t_star,
                                  mean_diff=float(diff.mean()),
                                  sem=float(diff.std(ddof=1) / np.sqrt(len(diff))),
                                  n=len(diff), n_pos=int((diff > 0).sum()),
                                  wilcoxon_p=p))
    long = pd.DataFrame(long_rows)
    stats = pd.DataFrame(stat_rows)
    stats['q'] = multipletests(stats['wilcoxon_p'].fillna(1.0), method='fdr_bh')[1]
    stats['star'] = stats['q'].map(_stars)
    long.to_csv(os.path.join(SRC_DIR, 'panel_d_peak_pairwise.csv'), index=False)
    stats.to_csv(os.path.join(SRC_DIR, 'panel_d_peak_pairwise_stats.csv'), index=False)
    return stats


def panel_e_perbin(df):
    """Per-bin language−vision difference significance for R², cosine, word, category (supplement)."""
    frames = []
    for metric, col in [('r2', 'r2_ac'), ('cosine', 'cosine'), ('word', 'word_ac'), ('category', 'cat_ac')]:
        fam = _family_by_patient(df, col)
        sig = _perbin_lmm(fam)
        sig.insert(0, 'metric', metric)
        frames.append(sig)
    out = pd.concat(frames, ignore_index=True)
    # Family-level (language vs vision) per-bin significance — kept as a supplement / for the
    # headline sig-bin counts; the main between-model view is panel_d_pairwise.
    out.to_csv(os.path.join(SRC_DIR, 'supp_family_perbin_significance.csv'), index=False)
    return out


def cue_timing():
    """cue → mean ± s.d. across participants (from cue_stats.json)."""
    from collections import defaultdict
    acc = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(RUN_DIR, '*', 'cue_stats.json'))):
        for cue, v in json.load(open(p)).get('rel_cues', {}).items():
            m = v.get('mean')
            if m is not None and np.isfinite(m):
                acc[cue].append(float(m))
    rows = [dict(cue=c, mean_s=float(np.mean(vs)), std_s=float(np.std(vs)), n=len(vs))
            for c, vs in acc.items() if len(vs) >= 2 and np.std(vs) > 0]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(SRC_DIR, 'cue_timing.csv'), index=False)
    return out


def aggregate_layer_sweep(df):
    """Vision-model accuracy vs layer depth + a language pooled reference line.
    Reads main/results/layer_sweep/layer_sweep.csv (if the sweep has been run)."""
    paths = []
    for g in LAYER_SWEEP_GLOBS:
        paths.extend(sorted(glob.glob(g)))
    if not paths:
        print("  [panel f] no layer_sweep.csv (results/layer_sweep[_s*]/) — run visual_layer_sweep; skipping panel f.")
        return None
    ls = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    # Rows are stamped with the atlas that gated them (2026-08-08); anything older is
    # whole-brain 10-bin data. Select ONE atlas before deduplicating -- otherwise
    # drop_duplicates keeps whichever file happened to sort first and the panel silently
    # mixes two channel sets, or shows legacy rows for the patients not yet re-swept.
    if 'roi_atlas' not in ls.columns:
        ls['roi_atlas'] = 'legacy'
    have = sorted(ls['roi_atlas'].unique())
    if ROI_ATLAS not in have:
        print(f"  [panel f] no rows for atlas {ROI_ATLAS!r} (file has {have}); "
              f"re-run run_vision_layer_sweep.py --roi-atlas {ROI_ATLAS}; skipping panel f.")
        return None
    dropped = ls[ls['roi_atlas'] != ROI_ATLAS]
    if not dropped.empty:
        print(f"  [panel f] ignoring {dropped.patient.nunique()} participant(s) whose rows "
              f"are from atlas {sorted(dropped['roi_atlas'].unique())}")
    ls = ls[ls['roi_atlas'] == ROI_ATLAS].drop_duplicates(
        subset=['patient', 'layer_key', 'epoch'])
    print(f"  [panel f] {ls.patient.nunique()} participants from {len(paths)} sweep "
          f"file(s), atlas={ROI_ATLAS}")
    inter = ls[ls.layer_type == 'intermediate']
    agg = (inter.groupby(['model_family', 'layer_idx'])[['cat_bal_acc', 'word_bal_acc']]
                .agg(['mean', 'sem', 'count']))
    agg.columns = ['_'.join(c) for c in agg.columns]
    agg = agg.reset_index()
    # Language reference: per-participant post-onset peak *raw* balanced accuracy from
    # per_time_scores (the sweep reports raw acc, so compare like-for-like), averaged.
    lang_ref = []
    for d in sorted(glob.glob(os.path.join(RUN_DIR, '*', 'per_time_scores.csv'))):
        t = pd.read_csv(d, usecols=['patient', 'embedding', 'bin_index',
                                    'category_balanced_acc', 'word_balanced_acc'])
        t = t[t.embedding.isin(LANG) & (t.bin_index >= 10) & (t.bin_index < N_BINS)]
        lang_ref.append(t)
    lref = pd.concat(lang_ref, ignore_index=True)
    peak = lref.groupby(['patient', 'embedding'])[['category_balanced_acc', 'word_balanced_acc']].max()
    ref = dict(cat_ref_mean=float(peak['category_balanced_acc'].mean()),
               cat_ref_sem=float(peak['category_balanced_acc'].sem()),
               word_ref_mean=float(peak['word_balanced_acc'].mean()),
               word_ref_sem=float(peak['word_balanced_acc'].sem()))
    agg.attrs = {}
    agg.to_csv(os.path.join(SRC_DIR, 'panel_f_layer_sweep.csv'), index=False)
    pd.DataFrame([{'family': 'language(GloVe,Word2Vec) pooled reference', **ref}]).to_csv(
        os.path.join(SRC_DIR, 'panel_f_language_reference.csv'), index=False)
    print(f"  [panel f] wrote layer sweep aggregate ({agg.model_family.nunique()} model families)")
    return agg


def group_inference(peak_stats, pref_group, sig_by_metric):
    """One row per headline number the Results paragraph cites."""
    rows = []
    # peak-bin between-model comparison (semantic peak): per pair × metric, Δ + star
    t_star = float(peak_stats['peak_time_s'].iloc[0])
    rows.append(dict(quantity='semantic_peak_time_s', value=round(t_star, 2),
                     detail='group category-accuracy peak bin used for the model comparison'))
    for _, r in peak_stats.iterrows():
        rows.append(dict(quantity=f"peakdiff_{r['metric']}_{r['pair']}",
                         value=f"{r['star']} ({int(r['n_pos'])}/{int(r['n'])})",
                         detail=f"Δ(lang−vis)={r['mean_diff']:+.4f}±{r['sem']:.4f}, "
                                f"Wilcoxon greater p={r['wilcoxon_p']:.4g}, FDR q={r['q']:.4g}"))
    for _, r in pref_group.iterrows():
        rows.append(dict(quantity=f'prefers_language_{r.metric}',
                         value=f"{int(r.n_prefer_language)}/{int(r.n_participants)}",
                         detail=f"Wilcoxon greater p={r.wilcoxon_p_greater:.4g}, mean Δ={r.mean_delta:.4f}"))
    # family-level (language vs vision) per-bin significant-bin counts
    for metric in ['r2', 'category', 'word', 'cosine']:
        s = sig_by_metric[sig_by_metric.metric == metric]
        nsig = int(s['significant'].sum())
        tmin = s[s.significant]['time_s'].min() if nsig else np.nan
        tmax = s[s.significant]['time_s'].max() if nsig else np.nan
        rows.append(dict(quantity=f'family_lang_gt_vis_sig_bins_{metric}', value=nsig,
                         detail=(f"FDR q<{FDR_Q}; window [{tmin:.1f}, {tmax:.1f}]s"
                                 if nsig else "no significant bins")))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(SRC_DIR, 'group_inference.csv'), index=False)
    return out


def main():
    os.makedirs(SRC_DIR, exist_ok=True)
    n_hist, bin_ms = _meta()
    df = _load_long(n_hist, bin_ms)
    print(f"[compute] {df.patient.nunique()} participants × {sorted(df.embedding.unique())} "
          f"× {df.bin_index.nunique()} bins")

    supp_cosine_timecourse(df)                # supplement (decoding fidelity over time)
    panel_b_category(df)                      # b: category effect, language vs vision, + raster
    panel_c_r2(df)                            # c: R² effect, language vs vision, + raster
    peak_stats = panel_d_peak_pairwise(df)    # d: peak-bin pairwise model comparison (bars/dots/stars)
    _, pref_group = panel_c_preference(df)    # e: between-participant Δ(language−vision)
    fam_sig = panel_e_perbin(df)              # family per-bin sig counts (supplement + headline)
    cue_timing()
    aggregate_layer_sweep(df)
    gi = group_inference(peak_stats, pref_group, fam_sig)

    print("\n[compute] headline numbers (group_inference.csv):")
    for _, r in gi.iterrows():
        print(f"    {r.quantity:38s} {str(r.value):>10}   {r.detail}")
    print(f"\n[compute] source data → {SRC_DIR}")


if __name__ == '__main__':
    main()
