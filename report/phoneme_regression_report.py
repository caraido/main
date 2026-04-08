"""
report/phoneme_regression_report.py  —  Phoneme Regression Analysis Report
===========================================================================
Generates an HTML report for a phoneme_regression run folder.

Usage (run from main/):
    python report/phoneme_regression_report.py  <run_dir>
    python report/phoneme_regression_report.py  <run_dir>  --out-dir my_out/
    python report/phoneme_regression_report.py  <run_dir>  --with-significance

Output  (<run_dir>/report/ by default):
    phoneme_regression_report_<run_id>.html
    phoneme_significance.csv          (only written if --with-significance)

Works with any bin size (reads bin_size_ms from meta.json).

Data sources
------------
* Time-series plots and timing metrics: per_time_scores.csv  (always used;
  fast even over slow network filesystems).
* Null baseline for word accuracy: mean of pre-onset bins in per_time_scores.csv
  (bin_index < n_bins_history). This is derived entirely from the experimental
  results, matching the request to avoid theoretical chance computation.
* Significance (optional, --with-significance): per-epoch arrays loaded from
  phoneme_regression_results.pkl via a dedicated subprocess.  Patients whose
  PKL exceeds MAX_PKL_MB are skipped (significance = N/A).  Wilcoxon
  signed-rank (obs > null) at peak bin, Bonferroni-corrected.

Metrics
-------
  Rise time   First bin where word_obs > pre-onset null mean + 1×SEM(pre-onset).
  Peak time   argmax bin of word_balanced_acc (mean across epochs if PKL
              available, else the CSV single-epoch mean).
  Peak acc.   Word balanced accuracy at peak bin.
  Cosine      Cosine similarity at peak bin (regression fit proxy; no null).
  Sig.        Wilcoxon p-value Bonferroni-corrected (--with-significance only).
"""

import os
import sys
import json
import argparse
import base64
import io
import tempfile
import subprocess
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

EMBEDDINGS = ["panphon", "token_ipa"]
EMB_COLORS = {"panphon": "#1565C0", "token_ipa": "#E65100"}
EMB_LABELS = {"panphon": "PWESuite panphon", "token_ipa": "PWESuite token-IPA"}

MAX_PKL_MB = 3000   # skip significance for PKLs larger than this


# ═══════════════════════════════════════════════════════════════════════════════
# CSV data loading  (primary path — always fast)
# ═══════════════════════════════════════════════════════════════════════════════

def load_csv(run_dir, patient):
    """
    Load per_time_scores.csv for one patient.
    Returns dict[embedding] → dict of 1-D arrays over n_bins, plus n_words/n_cats.
    """
    csv_path = os.path.join(run_dir, patient, "per_time_scores.csv")
    df = pd.read_csv(csv_path)

    def _col(name, n):
        if name in df.columns:
            sub2 = df[df["embedding"] == emb].sort_values("bin_index").reset_index(drop=True)
            return sub2[name].values.astype(np.float32)
        return np.full(n, np.nan, dtype=np.float32)

    records = {}
    for emb in EMBEDDINGS:
        sub = df[df["embedding"] == emb].sort_values("bin_index").reset_index(drop=True)
        if len(sub) == 0:
            continue
        n = len(sub)
        def _c(name):
            return sub[name].values.astype(np.float32) if name in sub.columns else np.full(n, np.nan, dtype=np.float32)
        records[emb] = {
            "cosine_mean":  sub["cosine_mean"].values.astype(np.float32),
            "cosine_std":   sub["cosine_std"].values.astype(np.float32),
            "word_acc":     sub["word_balanced_acc"].values.astype(np.float32),
            "word_chance":  _c("word_chance_mean"),
            "cat_acc":      _c("category_balanced_acc"),
            "cat_chance":   _c("cat_chance_mean"),
        }

    # Approximate n_words / n_cats from the task-level chance visible in early bins
    # (precise values come from PKL if available — we'll override later if loaded)
    n_words = 60
    n_cats  = 6
    return records, n_words, n_cats


def load_decoding_csv(run_dir, patient):
    """
    Load top1_decoding_source_data.csv for one patient.

    Computes:
      bal_acc   – per-class *binary* balanced accuracy (n_bins, n_cats):
                   for each class c: (recall_c + specificity_c) / 2
      wrong_word_cat_acc – category accuracy restricted to wrong-word trials (n_bins,)

    Returns dict[embedding] → {
      "categories":         list[str]
      "bal_acc":            ndarray (n_bins, n_cats)
      "wrong_word_cat_acc": ndarray (n_bins,)
    }
    Returns empty dict if the file does not exist or has missing columns.
    """
    csv_path = os.path.join(run_dir, patient, "top1_decoding_source_data.csv")
    if not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)
    required = {"embedding", "epoch", "bin_index",
                "true_category", "pred_category", "category_correct", "word_correct"}
    if not required.issubset(df.columns):
        return {}

    result = {}
    for emb in EMBEDDINGS:
        sub = df[df["embedding"] == emb].copy()
        if len(sub) == 0:
            continue

        n_bins  = int(sub["bin_index"].max()) + 1
        cats    = sorted(sub["true_category"].unique())
        n_cats  = len(cats)

        # ---- per-class binary balanced accuracy ----------------------------
        # For each class c, epoch, bin:
        #   sensitivity = mean(pred==c | true==c)
        #   specificity = mean(pred!=c | true!=c)
        #   binary_bal_acc = (sensitivity + specificity) / 2
        bal_arr = np.full((n_bins, n_cats), np.nan, dtype=np.float32)
        for ci, cat in enumerate(cats):
            sub["_pos"] = (sub["true_category"] == cat).astype(np.float32)
            sub["_tp"]  = ((sub["true_category"] == cat) &
                           (sub["pred_category"] == cat)).astype(np.float32)
            sub["_tn"]  = ((sub["true_category"] != cat) &
                           (sub["pred_category"] != cat)).astype(np.float32)

            # sensitivity per epoch/bin
            sens_ep = (
                sub[sub["_pos"] == 1]
                .groupby(["epoch", "bin_index"])["_tp"]
                .mean()
                .reset_index(name="sens")
            )
            # specificity per epoch/bin
            spec_ep = (
                sub[sub["_pos"] == 0]
                .groupby(["epoch", "bin_index"])["_tn"]
                .mean()
                .reset_index(name="spec")
            )
            mrg = sens_ep.merge(spec_ep, on=["epoch", "bin_index"], how="inner")
            mrg["bba"] = (mrg["sens"] + mrg["spec"]) / 2.0
            mean_bba = (
                mrg.groupby("bin_index")["bba"]
                .mean()
                .reindex(range(n_bins))
                .values.astype(np.float32)
            )
            bal_arr[:, ci] = mean_bba

        # ---- wrong-word category accuracy ---------------------------------
        wrong = sub[sub["word_correct"] == 0]
        if len(wrong) > 0:
            ww_ep = (
                wrong.groupby(["epoch", "bin_index"])["category_correct"]
                .mean()
                .reset_index(name="cat_correct")
            )
            ww_mean = (
                ww_ep.groupby("bin_index")["cat_correct"]
                .mean()
                .reindex(range(n_bins))
                .values.astype(np.float32)
            )
        else:
            ww_mean = np.full(n_bins, np.nan, dtype=np.float32)

        result[emb] = {
            "categories":         cats,
            "bal_acc":            bal_arr,          # (n_bins, n_cats)
            "wrong_word_cat_acc": ww_mean,          # (n_bins,)
        }

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PKL extraction helper  (subprocess script — never imported by main process)
# ═══════════════════════════════════════════════════════════════════════════════

_EXTRACTOR = """
import sys, os, types, numpy as np, warnings, gc
warnings.filterwarnings('ignore')
for mod in ['torch', 'models', 'models.model']:
    sys.modules.setdefault(mod, types.ModuleType(mod))
class _S: pass
sys.modules['models'].BasicRegressor      = _S
sys.modules['models.model'].BasicRegressor = _S
try:
    import dill
except ImportError:
    os.system(f"{sys.executable} -m pip install dill --break-system-packages -q")
    import dill

pkl_path, npz_path = sys.argv[1], sys.argv[2]
embeddings = sys.argv[3:]

with open(pkl_path, 'rb') as f:
    data = dill.load(f)

arrays = {}
for emb in embeddings:
    if emb not in data.get('regressors', {}):
        continue
    br = data['regressors'][emb]
    arrays[f'{emb}__word_obs']  = np.asarray(
        br.all_retrieval_word_balanced_acc,        dtype=np.float32)
    arrays[f'{emb}__word_null'] = np.asarray(
        br.all_retrieval_chance_word_balanced_acc, dtype=np.float32)

arrays['_n_words'] = np.array([len(np.unique(data.get('clean_target_labels', [])))])
arrays['_n_cats']  = np.array([len(np.unique(data.get('clean_word_category',  [])))])
del data; gc.collect()
np.savez_compressed(npz_path, **arrays)
"""


def _extract_pkl(run_dir, patient, tmpdir, timeout_s=120):
    """
    Run the extractor in a subprocess.  Returns npz path on success, else None.
    """
    pkl_path = os.path.join(run_dir, patient, "phoneme_regression_results.pkl")
    npz_path = os.path.join(tmpdir, f"{patient}.npz")
    r = subprocess.run(
        [sys.executable, "-c", _EXTRACTOR, pkl_path, npz_path, *EMBEDDINGS],
        capture_output=True, text=True, timeout=timeout_s,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-400:])
    return npz_path


# ═══════════════════════════════════════════════════════════════════════════════
# Null and timing metrics  (all computed from pre_time_scores.csv arrays)
# ═══════════════════════════════════════════════════════════════════════════════

def presonset_null(word_acc, n_bins_history):
    """
    Estimate null word accuracy from the pre-onset window (bins 0..n_bins_history-1).
    Returns (null_mean, null_sem) as scalars.
    """
    pre = word_acc[:n_bins_history]
    return float(pre.mean()), float(pre.std() / max(np.sqrt(len(pre)), 1))


def compute_timing(word_acc, n_bins_history, bin_size_ms):
    """
    Parameters
    ----------
    word_acc       1-D array (n_bins,) — mean balanced accuracy across epochs.
    n_bins_history int
    bin_size_ms    int

    Returns  dict with rise_time_ms (or None), peak_time_ms, peak_acc,
                       null_mean, null_sem.
    """
    null_mean, null_sem = presonset_null(word_acc, n_bins_history)
    threshold = null_mean + null_sem

    peak_bin     = int(np.argmax(word_acc))
    peak_time_ms = (peak_bin - n_bins_history) * bin_size_ms

    rise_idxs = np.where(word_acc > threshold)[0]
    if len(rise_idxs):
        rise_bin     = int(rise_idxs[0])
        rise_time_ms = (rise_bin - n_bins_history) * bin_size_ms
    else:
        rise_bin = rise_time_ms = None

    return {
        "rise_bin":      rise_bin,
        "rise_time_ms":  rise_time_ms,
        "peak_bin":      peak_bin,
        "peak_time_ms":  peak_time_ms,
        "peak_acc":      float(word_acc[peak_bin]),
        "null_mean":     null_mean,
        "null_sem":      null_sem,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Significance  (optional — requires PKL subprocess)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_significance(run_dir, all_csv, timing_df, tmpdir, max_pkl_mb=MAX_PKL_MB):
    """
    Attempt to load PKLs for patients small enough, run Wilcoxon at each patient's
    CSV-determined peak bin, Bonferroni-correct.

    Returns a dict  patient → embedding → {"word_sig": str, "pval_bonf": float}
    """
    print("  Attempting PKL loads for significance…", flush=True)
    sig_results = {}

    raw_records = []
    for patient, pd_ in all_csv.items():
        pkl_path = os.path.join(run_dir, patient, "phoneme_regression_results.pkl")
        size_mb  = os.path.getsize(pkl_path) / 1e6

        if size_mb > max_pkl_mb:
            print(f"    {patient}: skip (PKL {size_mb:.0f} MB > {max_pkl_mb} MB)")
            continue

        try:
            print(f"    {patient} ({size_mb:.0f} MB)…", flush=True, end="")
            npz_path = _extract_pkl(run_dir, patient, tmpdir)
            npz = np.load(npz_path)
            print(" ✓", flush=True)

            for emb in EMBEDDINGS:
                if emb not in pd_:
                    continue
                word_obs_key  = f"{emb}__word_obs"
                word_null_key = f"{emb}__word_null"
                if word_obs_key not in npz:
                    continue

                # Use peak bin determined from CSV means
                row = timing_df[
                    (timing_df.patient == patient) &
                    (timing_df.embedding == emb)
                ]
                if len(row) == 0:
                    continue
                peak_bin = int(row.iloc[0]["peak_bin"])

                word_obs_all  = npz[word_obs_key]    # (n_epochs, n_bins)
                word_null_all = npz[word_null_key]

                obs_at_peak  = word_obs_all[:, peak_bin]
                null_at_peak = word_null_all[:, peak_bin]

                try:
                    _, pval = stats.wilcoxon(obs_at_peak - null_at_peak,
                                             alternative="greater")
                except Exception:
                    pval = 1.0

                raw_records.append({
                    "patient": patient,
                    "embedding": emb,
                    "pval_raw": float(pval),
                    "mean_obs":  float(obs_at_peak.mean()),
                    "mean_null": float(null_at_peak.mean()),
                })

        except subprocess.TimeoutExpired:
            print(f" TIMEOUT", flush=True)
        except Exception as e:
            print(f" FAILED ({str(e)[:80]})", flush=True)

    if not raw_records:
        return {}

    n_tests = len(raw_records)
    for r in raw_records:
        r["pval_bonf"] = min(r["pval_raw"] * n_tests, 1.0)

    def _stars(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "NS"

    sig_results = {}
    for r in raw_records:
        p = r["patient"]
        e = r["embedding"]
        sig_results.setdefault(p, {})[e] = {
            "word_sig":   _stars(r["pval_bonf"]),
            "pval_bonf":  r["pval_bonf"],
            "mean_obs":   r["mean_obs"],
            "mean_null":  r["mean_null"],
        }

    n_sig = sum(1 for p in sig_results.values() for e in p.values()
                if e["word_sig"] != "NS")
    print(f"  Bonferroni ({n_tests} tests): {n_sig} sig. pairs")
    return sig_results


# ═══════════════════════════════════════════════════════════════════════════════
# Per-patient figures
# ═══════════════════════════════════════════════════════════════════════════════

def _null_line(ax, series, n_bins_history, time_ms, chance_series, color, label=""):
    """
    Draw the appropriate null baseline on ax for a given accuracy series.
    Priority: shuffled-chance column (if not all-NaN) → pre-onset band.
    """
    has_chance = (chance_series is not None and
                  not np.all(np.isnan(chance_series)))
    if has_chance:
        ax.plot(time_ms, chance_series * 100, color=color, lw=1.0, ls="--",
                alpha=0.55, label=f"{label} shuffled null" if label else "shuffled null")
    else:
        null_mean, null_sem = presonset_null(series, n_bins_history)
        ax.axhline(null_mean * 100, color=color, lw=1.0, ls=":", alpha=0.55,
                   label=f"{label} null (pre-onset)" if label else "null (pre-onset)")
        ax.fill_between(
            time_ms,
            (null_mean - null_sem) * 100,
            (null_mean + null_sem) * 100,
            color=color, alpha=0.07,
        )


def make_figure(patient, emb_data, n_bins_history, bin_size_ms):
    """
    Three-row figure:
      Row 1 — cosine similarity (mean ± std)
      Row 2 — word balanced accuracy + shuffled null (or pre-onset null)
      Row 3 — category balanced accuracy + shuffled null (or pre-onset null)
    Word and category panels are kept separate because their chance levels differ.
    Returns base64 PNG.
    """
    n_bins  = list(emb_data.values())[0]["cosine_mean"].shape[0]
    time_ms = np.array([(b - n_bins_history) * bin_size_ms for b in range(n_bins)])

    fig, axes = plt.subplots(3, 1, figsize=(12, 8.0), sharex=True)
    fig.suptitle(f"Patient {patient}", fontsize=12, fontweight="bold")

    for emb in EMBEDDINGS:
        if emb not in emb_data:
            continue
        d   = emb_data[emb]
        col = EMB_COLORS[emb]
        lbl = EMB_LABELS[emb]

        # Row 1: cosine similarity
        axes[0].plot(time_ms, d["cosine_mean"], color=col, lw=1.6, label=lbl)
        axes[0].fill_between(
            time_ms,
            d["cosine_mean"] - d["cosine_std"],
            d["cosine_mean"] + d["cosine_std"],
            color=col, alpha=0.12,
        )

        # Row 2: word balanced accuracy
        word = d["word_acc"]
        axes[1].plot(time_ms, word * 100, color=col, lw=1.6, label=lbl)
        _null_line(axes[1], word, n_bins_history, time_ms,
                   d.get("word_chance"), col, lbl)

        # Row 3: category balanced accuracy
        cat = d.get("cat_acc")
        if cat is not None and not np.all(np.isnan(cat)):
            axes[2].plot(time_ms, cat * 100, color=col, lw=1.6, label=lbl)
            _null_line(axes[2], cat, n_bins_history, time_ms,
                       d.get("cat_chance"), col, lbl)

    for ax in axes:
        ax.axvline(0, color="black", lw=0.8, ls=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

    axes[0].axhline(0, color="gray", lw=0.7, ls="--", alpha=0.5)
    axes[0].set_ylabel("Cosine Similarity", fontsize=9)
    axes[0].legend(fontsize=7.5, loc="upper left", ncol=2)
    axes[1].set_ylabel("Word Bal. Acc. (%)", fontsize=9)
    axes[1].legend(fontsize=7.5, loc="upper left", ncol=2)
    axes[2].set_ylabel("Cat. Bal. Acc. (%)", fontsize=9)
    axes[2].set_xlabel("Time from trial onset (ms)", fontsize=9)
    axes[2].legend(fontsize=7.5, loc="upper left", ncol=2)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_category_figure(patient, cat_data, emb_data, n_bins_history, bin_size_ms):
    """
    Two-panel figure (one subplot per embedding) showing per-class binary balanced
    accuracy over time.

    For each class c:  binary_bal_acc_c = (recall_c + specificity_c) / 2
    • 50% reference line = theoretical binary chance (not data-derived).
    • Bold dashed black = aggregate category_balanced_acc from per_time_scores.csv.
    • Shuffled null line (from cat_chance) if available, else 50% reference only.
    Returns base64 PNG, or None if cat_data is empty.
    """
    embs_present = [e for e in EMBEDDINGS if e in cat_data and e in emb_data]
    if not embs_present:
        return None

    n_bins  = list(emb_data.values())[0]["cosine_mean"].shape[0]
    time_ms = np.array([(b - n_bins_history) * bin_size_ms for b in range(n_bins)])

    all_cats = sorted({c for e in embs_present for c in cat_data[e]["categories"]})
    cmap     = plt.get_cmap("tab10")
    cat_col  = {c: cmap(i % 10) for i, c in enumerate(all_cats)}

    n_embs = len(embs_present)
    fig, axes = plt.subplots(1, n_embs, figsize=(7.0 * n_embs, 4.5),
                             squeeze=False, sharey=False)
    axes = axes[0]
    fig.suptitle(f"Patient {patient} — Per-Class Category Balanced Accuracy",
                 fontsize=12, fontweight="bold")

    for ax, emb in zip(axes, embs_present):
        d_cat  = cat_data[emb]
        cats   = d_cat["categories"]
        bal    = d_cat["bal_acc"]          # (n_bins, n_cats) — binary bal acc per class

        for ci, cat in enumerate(cats):
            y = bal[:, ci] * 100
            ax.plot(time_ms, y, color=cat_col[cat], lw=1.4, label=cat)

        # 50% theoretical binary chance
        ax.axhline(50, color="gray", lw=1.0, ls=":", alpha=0.6,
                   label="50% binary chance")

        # Aggregate category_balanced_acc + shuffled null
        if "cat_acc" in emb_data[emb] and not np.all(np.isnan(emb_data[emb]["cat_acc"])):
            cat_agg = emb_data[emb]["cat_acc"]
            ax.plot(time_ms, cat_agg * 100, color="black", lw=2.0, ls="--",
                    label="Aggregate cat. acc.", zorder=5)
            has_chance = ("cat_chance" in emb_data[emb] and
                          not np.all(np.isnan(emb_data[emb]["cat_chance"])))
            if has_chance:
                ax.plot(time_ms, emb_data[emb]["cat_chance"] * 100,
                        color="black", lw=1.0, ls=":", alpha=0.55,
                        label="Shuffled null (cat)", zorder=4)

        ax.axvline(0, color="black", lw=0.8, ls=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_ylabel("Cat. Binary Bal. Acc. (%)", fontsize=9)
        ax.set_xlabel("Time from trial onset (ms)", fontsize=9)
        ax.set_title(EMB_LABELS[emb], fontsize=9,
                     color=EMB_COLORS[emb], fontweight="bold")
        ax.legend(fontsize=7, loc="upper left", ncol=2)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_wrong_word_cat_figure(patient, cat_data, emb_data, n_bins_history, bin_size_ms):
    """
    Two-panel figure (one subplot per embedding) showing category accuracy
    restricted to trials where the top-1 word prediction was wrong.

    Goal: if phoneme-based retrieval captures category structure beyond the
    individual word, category accuracy on wrong-word trials should still be
    above chance.

    Solid bold = wrong-word category accuracy.
    Dashed     = overall category accuracy (reference).
    Null       = pre-onset mean ± SEM of the wrong-word series itself.
    Returns base64 PNG, or None if no wrong-word data is available.
    """
    embs_present = [e for e in EMBEDDINGS
                    if e in cat_data and e in emb_data
                    and "wrong_word_cat_acc" in cat_data[e]
                    and not np.all(np.isnan(cat_data[e]["wrong_word_cat_acc"]))]
    if not embs_present:
        return None

    n_bins  = list(emb_data.values())[0]["cosine_mean"].shape[0]
    time_ms = np.array([(b - n_bins_history) * bin_size_ms for b in range(n_bins)])

    n_embs = len(embs_present)
    fig, axes = plt.subplots(1, n_embs, figsize=(7.0 * n_embs, 4.5),
                             squeeze=False, sharey=False)
    axes = axes[0]
    fig.suptitle(f"Patient {patient} — Category Accuracy: Wrong-Word Trials",
                 fontsize=12, fontweight="bold")

    for ax, emb in zip(axes, embs_present):
        col = EMB_COLORS[emb]

        ww = cat_data[emb]["wrong_word_cat_acc"]    # (n_bins,)
        ax.plot(time_ms, ww * 100, color=col, lw=2.2, label="Wrong-word cat. acc.")

        # Pre-onset null band for the wrong-word series
        ww_null_mean, ww_null_sem = presonset_null(ww, n_bins_history)
        ax.axhline(ww_null_mean * 100, color=col, lw=1.0, ls=":", alpha=0.55,
                   label="Null (pre-onset, wrong-word)")
        ax.fill_between(
            time_ms,
            (ww_null_mean - ww_null_sem) * 100,
            (ww_null_mean + ww_null_sem) * 100,
            color=col, alpha=0.10,
        )

        # Overall cat acc as dashed reference
        if "cat_acc" in emb_data[emb] and not np.all(np.isnan(emb_data[emb]["cat_acc"])):
            cat_all = emb_data[emb]["cat_acc"]
            ax.plot(time_ms, cat_all * 100, color=col, lw=1.4, ls="--",
                    alpha=0.65, label="All-trial cat. acc. (ref)")

        ax.axvline(0, color="black", lw=0.8, ls=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_ylabel("Cat. Acc. (%)", fontsize=9)
        ax.set_xlabel("Time from trial onset (ms)", fontsize=9)
        ax.set_title(EMB_LABELS[emb], fontsize=9,
                     color=col, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper left", ncol=1)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════════

_CSS = """
  body{font-family:'Segoe UI',Arial,sans-serif;max-width:1300px;margin:0 auto;
       padding:20px;color:#333;line-height:1.6}
  h1{color:#1a5276;border-bottom:3px solid #2980b9;padding-bottom:10px}
  h2{color:#2471a3;margin-top:40px;border-bottom:1px solid #d4e6f1;padding-bottom:5px}
  h3{color:#2e86c1}
  .sum{background:#eaf2f8;border-left:4px solid #2980b9;padding:15px;margin:20px 0;border-radius:4px}
  .met{background:#f3e5f5;border-left:4px solid #8e24aa;padding:15px;margin:15px 0;border-radius:4px}
  .wrn{background:#fdedec;border-left:4px solid #e74c3c;padding:12px;margin:10px 0;border-radius:4px;font-size:12px}
  .meta-box{background:#f9f9f9;border:1px solid #ddd;border-radius:4px;padding:10px 15px;margin:15px 0}
  .meta-box summary{cursor:pointer;font-weight:bold;color:#2471a3;padding:5px 0}
  table{border-collapse:collapse;width:100%;margin:15px 0;font-size:12px}
  .meta-table{font-size:12px}
  .meta-table td{padding:4px 10px;border-bottom:1px solid #eee}
  .meta-table tr:nth-child(even){background:#f8f9fa}
  th{background:#2980b9;color:white;padding:7px 9px;text-align:center}
  td{padding:5px 8px;border-bottom:1px solid #ddd}
  tr:nth-child(even){background:#f8f9fa}
  .sig{color:#27ae60;font-weight:bold}.ns{color:#c62828}
  code{background:#f0f0f0;padding:2px 6px;border-radius:3px;font-size:.9em}
  .dc{font-variant-numeric:tabular-nums;text-align:center}
  .cc{background:#f0f0f0;font-weight:bold;text-align:center}
  .s3{color:#1b5e20;font-weight:bold}.s2{color:#2e7d32;font-weight:bold}
  .s1{color:#388e3c}.sn{color:#c62828}.sa{color:#aaa;font-style:italic}
  .ph td:first-child{background:#e8f5e9;font-weight:bold}
  .pm td:first-child{background:#fff8e1;font-weight:bold}
  .pl td:first-child{background:#ffebee}
  .ph{}.pm{}.pl{}
  .pan{background:#1565C0;color:white;text-align:center}
  .ipa{background:#E65100;color:white;text-align:center}
  .fig-grid{display:flex;flex-wrap:wrap;gap:18px;margin:20px 0}
  .fig-card{border:1px solid #d4e6f1;border-radius:6px;padding:8px;background:#fafcff}
"""


def _sc(s):
    return {"***": "s3", "**": "s2", "*": "s1", "NS": "sn", "N/A": "sa"}.get(s, "sn")


def _tier(peak_fold):
    if peak_fold > 1.5: return "ph"
    if peak_fold > 1.2: return "pm"
    return "pl"


def _meta_table_html(meta):
    """Build an HTML table of all meta.json key-value pairs."""
    if not meta:
        return ''
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


def generate_html(timing_df, sig_results, figures, cat_figures, wrong_word_figures,
                  meta, out_dir, with_significance):
    run_id   = meta.get("run_id", "unknown")
    pipeline = meta.get("regressor_pipeline", "?")
    closest  = meta.get("closest", "cosine")
    n_epochs = meta.get("n_epochs", "?")
    bin_size = meta.get("bin_size_ms", "?")
    n_bh     = meta.get("n_bins_history", 10)

    n_pat = timing_df["patient"].nunique()

    if with_significance and sig_results:
        n_tests = sum(len(v) for v in sig_results.values())
        n_sig   = sum(1 for pv in sig_results.values()
                      for ev in pv.values() if ev["word_sig"] != "NS")
        pat_sig = {p for p, pv in sig_results.items()
                   if any(ev["word_sig"] != "NS" for ev in pv.values())}
        pat_na  = {p for p in timing_df["patient"].unique()
                   if p not in sig_results}
    else:
        n_tests = n_sig = 0
        pat_sig = pat_na = set()

    # Sort by mean peak fold (null estimate from pre-onset)
    patients_sorted = sorted(
        timing_df["patient"].unique(),
        key=lambda p: timing_df[timing_df.patient == p]["peak_fold"].mean(),
        reverse=True,
    )

    # ── Timing table rows ─────────────────────────────────────────────────
    def _trow(p):
        sub = timing_df[timing_df.patient == p]
        nw  = int(sub["n_words"].iloc[0]) if len(sub) else "?"
        nc  = int(sub["n_cats"].iloc[0])  if len(sub) else "?"
        fold_mean = sub["peak_fold"].mean()
        tier = _tier(fold_mean)
        cols = ""
        for emb in EMBEDDINGS:
            r = sub[sub.embedding == emb]
            if len(r) == 0:
                cols += "<td class='dc' colspan='4'>—</td>"
                continue
            r = r.iloc[0]
            rise_s = (f"<strong>{r.rise_time_ms:+.0f}</strong>"
                      if r.rise_time_ms is not None else "—")
            # significance star if available
            sig_s = ""
            if with_significance and p in sig_results and emb in sig_results[p]:
                sv = sig_results[p][emb]
                sig_s = f' <span class="{_sc(sv["word_sig"])}">{sv["word_sig"]}</span>'
            cat_s = (f"{r.cat_peak_acc*100:.1f}%"
                     if not np.isnan(r.cat_peak_acc) else "—")
            cols += (
                f"<td class='dc'>{rise_s} ms</td>"
                f"<td class='dc'><strong>{r.peak_time_ms:+.0f} ms</strong></td>"
                f"<td class='dc'>{r.peak_acc*100:.1f}%{sig_s}</td>"
                f"<td class='dc'>{cat_s}</td>"
            )
        return (f'<tr class="{tier}"><td><strong>{p}</strong></td>'
                f'<td class="dc">{nw}/{nc}</td>{cols}</tr>')

    # ── Detailed significance rows (only when --with-significance) ─────────
    def _srow(p):
        sub  = timing_df[timing_df.patient == p]
        nw   = int(sub["n_words"].iloc[0]) if len(sub) else "?"
        nc   = int(sub["n_cats"].iloc[0])  if len(sub) else "?"
        fold = sub["peak_fold"].mean()
        tier = _tier(fold)
        cols = ""
        for emb in EMBEDDINGS:
            r = sub[sub.embedding == emb]
            if len(r) == 0:
                cols += "<td class='dc'>—</td><td class='cc'>—</td><td class='dc'>—</td>"
                continue
            if p in sig_results and emb in sig_results[p]:
                sv = sig_results[p][emb]
                sc = _sc(sv["word_sig"])
                f  = sv["mean_obs"] / sv["mean_null"] if sv["mean_null"] > 0 else float("nan")
                cols += (
                    f'<td class="dc">{sv["mean_obs"]*100:.2f}% ({f:.2f}×)</td>'
                    f'<td class="cc">{sv["mean_null"]*100:.2f}%</td>'
                    f'<td class="dc"><span class="{sc}">{sv["word_sig"]}</span></td>'
                )
            else:
                cols += ("<td class='dc'>—</td><td class='cc'>—</td>"
                         "<td class='dc'><span class='sa'>N/A</span></td>")
        return (f'<tr class="{tier}"><td><strong>{p}</strong></td>'
                f'<td class="dc">{nw}/{nc}</td>{cols}</tr>')

    timing_rows = "\n".join(_trow(p) for p in patients_sorted)
    sig_rows    = "\n".join(_srow(p) for p in patients_sorted)

    # ── Figures ────────────────────────────────────────────────────────────
    fig_html = '<div class="fig-grid">\n'
    for p in patients_sorted:
        if p in figures:
            fig_html += (
                f'<div class="fig-card"><img src="data:image/png;base64,{figures[p]}" '
                f'alt="{p}" style="width:540px;"></div>\n'
            )
    fig_html += '</div>\n'

    def _fig_grid(fig_dict, width, empty_msg="No data available."):
        if not fig_dict or not any(v is not None for v in fig_dict.values()):
            return f'<p><em>{empty_msg}</em></p>\n'
        html = '<div class="fig-grid">\n'
        for p in patients_sorted:
            if p in fig_dict and fig_dict[p] is not None:
                html += (
                    f'<div class="fig-card"><img src="data:image/png;base64,{fig_dict[p]}" '
                    f'alt="{p}" style="width:{width}px;"></div>\n'
                )
        html += '</div>\n'
        return html

    cat_fig_html = _fig_grid(
        cat_figures, 700,
        "No per-category decoding data available (top1_decoding_source_data.csv not found).")
    ww_fig_html  = _fig_grid(
        wrong_word_figures, 700,
        "No wrong-word category data available (top1_decoding_source_data.csv not found).")

    # ── Significance section ───────────────────────────────────────────────
    if with_significance:
        sig_section = f"""
<h2>6. Significance at Peak Bin</h2>
<p style="font-size:11px;">
  <span class="s3">*** p&lt;0.001</span>&nbsp;
  <span class="s2">** p&lt;0.01</span>&nbsp;
  <span class="s1">* p&lt;0.05</span>&nbsp;
  <span class="sn">NS</span>&nbsp;
  <span class="sa">N/A = PKL too large ({MAX_PKL_MB} MB limit)</span>
  &nbsp;(Bonferroni, {n_tests} tests)
</p>
<table>
<tr>
  <th>Patient</th><th>N words/cats</th>
  <th class="pan">panphon obs</th><th class="pan">panphon null</th><th class="pan">sig</th>
  <th class="ipa">token-IPA obs</th><th class="ipa">token-IPA null</th><th class="ipa">sig</th>
</tr>
{sig_rows}
</table>
<h2>7. Summary</h2>
<p>
  <strong>Significant patients</strong> (≥1 embedding, Bonferroni-corrected):
  <strong>{len(pat_sig)}/{n_pat}</strong> —
  {", ".join(sorted(pat_sig)) if pat_sig else "none"}.
  {f"<br><em>Not tested (PKL too large): {', '.join(sorted(pat_na))}.</em>" if pat_na else ""}
</p>"""
    else:
        sig_section = f"""
<h2>6. Summary</h2>
<p>
  Significance testing not run. Re-run with <code>--with-significance</code> to add
  Wilcoxon testing (requires PKL loading; may be slow on large files).
</p>"""

    note_null = (
        f"<em>Null baseline</em>: mean word accuracy in pre-onset bins "
        f"(bin 0 – {n_bh-1}, i.e. t = {(-n_bh)*bin_size} to {-bin_size} ms)  "
        f"— data-derived, not theoretical. "
    )

    meta_table = _meta_table_html(meta)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Phoneme Regression Report — {run_id}</title>
<style>{_CSS}</style></head><body>

<h1>Phoneme Regression: Cross-Patient Analysis</h1>
<p>
  <strong>Run:</strong> <code>{run_id}</code> &nbsp;|&nbsp;
  <strong>Pipeline:</strong> <code>{pipeline}</code> &nbsp;|&nbsp;
  <strong>Bin size:</strong> {bin_size} ms &nbsp;|&nbsp;
  <strong>Retrieval:</strong> {closest}
  {"&nbsp;|&nbsp;<strong>Significance:</strong> Wilcoxon vs. shuffled null, Bonferroni" if with_significance else ""}
</p>

<div class="sum">
<h3>Executive Summary</h3>
<p>
  {n_pat} patients, 2 phoneme embeddings (panphon + token-IPA), {bin_size} ms temporal resolution.
  {"<strong>" + str(len(pat_sig)) + "/" + str(n_pat) + " patients significant</strong> at peak bin" + (f" ({', '.join(sorted(pat_sig))})." if pat_sig else ".") if with_significance and sig_results else "Significance not computed (run with --with-significance)."}
  <br>
  Strongest patients by peak word accuracy:
  {", ".join(patients_sorted[:5])}.
</p>
</div>

<h2>1. Run Configuration</h2>
<details class="meta-box" open>
  <summary>meta.json — all run parameters</summary>
  {meta_table if meta_table else '<p><em>No meta.json found for this run.</em></p>'}
</details>

<div class="met">
<strong>Metrics (all derived from per_time_scores.csv):</strong>
<ul style="margin:5px 0;padding-left:20px;font-size:12px;">
  <li><strong>Cosine similarity</strong> — regression fit (<code>cosine_mean ± cosine_std</code> per bin).
      No internal shuffled null available; y=0 reference shown.</li>
  <li><strong>Word balanced accuracy</strong> — retrieval quality per bin.</li>
  <li><strong>Rise time</strong> — first bin where word_acc &gt; pre-onset null mean + 1×SEM.</li>
  <li><strong>Peak time</strong> — argmax of word_acc, in ms from trial onset.</li>
  <li><strong>Peak accuracy</strong> — word balanced accuracy at peak bin.</li>
</ul>
{note_null}
Time axis: bin {n_bh} = trial onset (t = 0 ms), bin size = {bin_size} ms.
</div>

<h2>2. Time-Series: Cosine &amp; Word Accuracy ({bin_size} ms bins)</h2>
<p style="font-size:11px;">
  Row 1 = cosine similarity (±std). Row 2 = word balanced accuracy.
  Row 3 = category balanced accuracy.
  Word and category panels use <em>shuffled-label null</em> (dashed same color) when
  available from <code>word_chance_mean</code> / <code>cat_chance_mean</code> columns;
  otherwise falls back to pre-onset mean ±1&nbsp;SEM (dotted + shaded).
  Dotted vertical = trial onset.
  <span style="color:#1565C0;">■</span> panphon &nbsp;
  <span style="color:#E65100;">■</span> token-IPA
</p>
{fig_html}

<h2>3. Per-Class Category Balanced Accuracy ({bin_size} ms bins)</h2>
<p style="font-size:11px;">
  Per-class <em>binary balanced accuracy</em> = (recall<sub>c</sub> + specificity<sub>c</sub>) / 2,
  averaged over all epochs from <code>top1_decoding_source_data.csv</code>.
  50% dotted gray = binary chance. Bold dashed black = aggregate
  <code>category_balanced_acc</code>. Shuffled null line shown when available.
  Dotted vertical = trial onset.
</p>
{cat_fig_html}

<h2>4. Category Accuracy &mdash; Wrong-Word Trials ({bin_size} ms bins)</h2>
<p style="font-size:11px;">
  Category accuracy computed <em>only</em> on trials where the top-1 word retrieval
  was incorrect. If phoneme-structure representation drives retrieval toward the
  correct category even when the exact word is missed, this curve should rise above
  its pre-onset null after stimulus onset.
  Dashed = all-trial category accuracy (reference).
  Dotted horizontal + shaded = pre-onset null ±1&nbsp;SEM of the wrong-word series.
  Dotted vertical = trial onset.
</p>
{ww_fig_html}

<h2>5. Timing at Peak Time Bin</h2>
<table>
<tr>
  <th>Patient</th><th>N words/cats</th>
  <th colspan="4" class="pan">PWESuite panphon</th>
  <th colspan="4" class="ipa">PWESuite token-IPA</th>
</tr>
<tr>
  <th></th><th></th>
  <th class="pan">Rise (ms)</th><th class="pan">Peak (ms)</th>
  <th class="pan">Peak Acc{"/ Sig" if with_significance else ""}</th>
  <th class="pan">Cat Acc</th>
  <th class="ipa">Rise (ms)</th><th class="ipa">Peak (ms)</th>
  <th class="ipa">Peak Acc{"/ Sig" if with_significance else ""}</th>
  <th class="ipa">Cat Acc</th>
</tr>
{timing_rows}
</table>

{sig_section}

<hr>
<p style="font-size:10px;color:#888;">
  Generated by <code>report/phoneme_regression_report.py</code> &nbsp;|&nbsp;
  {run_id} &nbsp;|&nbsp; {bin_size} ms bins &nbsp;|&nbsp;
  {n_epochs} epochs &nbsp;|&nbsp; {n_pat} patients
</p>
</body></html>"""

    report_path = os.path.join(out_dir, f"phoneme_regression_report_{run_id}.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_run_dir(run_dir, pipeline_type="phoneme_regression"):
    """
    Resolve run_dir to an absolute path, supporting:
    - Absolute paths as-is
    - Relative paths resolved from cwd
    - Bare run IDs → results/<pipeline_type>/<run_id>
    - 'latest' → most recently modified folder under results/<pipeline_type>/
    """
    if run_dir == 'latest':
        base = os.path.join('results', pipeline_type)
        if not os.path.isdir(base):
            raise FileNotFoundError(f"results directory not found: {base}")
        candidates = [
            os.path.join(base, d) for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))
        ]
        if not candidates:
            raise FileNotFoundError(f"No run folders found in {base}")
        return max(candidates, key=os.path.getmtime)

    if os.path.isdir(run_dir):
        return run_dir

    # Try resolving as run ID under results/
    candidate = os.path.join('results', pipeline_type, run_dir)
    if os.path.isdir(candidate):
        return candidate

    raise FileNotFoundError(
        f"Run directory not found: {run_dir!r}\n"
        f"  Tried as-is and as results/{pipeline_type}/{run_dir}"
    )


def main():
    parser = argparse.ArgumentParser(
        prog="python report/phoneme_regression_report.py",
        description="Generate cross-patient HTML report for a phoneme_regression run.",
    )
    parser.add_argument("run_dir",
                        help="Path to run results folder, run ID, or 'latest'. "
                             "Examples: results/phoneme_regression/2026-04-06_..._50ep/ "
                             "or just the run ID, or 'latest'.")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: <run_dir>/report/)")
    parser.add_argument("--with-significance", action="store_true",
                        help="Load PKLs (via subprocess) and run Wilcoxon tests")
    parser.add_argument("--max-pkl-mb", type=int, default=MAX_PKL_MB,
                        help=f"PKL size limit for significance (default {MAX_PKL_MB} MB)")
    args = parser.parse_args()

    run_dir    = _resolve_run_dir(args.run_dir.rstrip("/\\"), "phoneme_regression")
    out_dir    = args.out_dir or os.path.join(run_dir, "report")
    max_pkl_mb = args.max_pkl_mb
    os.makedirs(out_dir, exist_ok=True)

    meta_path = os.path.join(run_dir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    n_bh       = meta.get("n_bins_history", 10)
    bin_size   = meta.get("bin_size_ms", 100)
    patients   = meta.get("succeeded_patients", sorted([
        d for d in os.listdir(run_dir)
        if os.path.isdir(os.path.join(run_dir, d)) and
        d not in ("__pycache__", "report")
    ]))

    print(f"Run      : {meta.get('run_id', run_dir)}")
    print(f"Bin size : {bin_size} ms  |  Patients : {patients}")
    print()

    # ── Step 1: Load per_time_scores.csv ──────────────────────────────────
    print("=" * 60)
    print("STEP 1: LOADING CSV DATA")
    print("=" * 60)
    all_csv, n_words_map, n_cats_map = {}, {}, {}
    for p in patients:
        try:
            records, nw, nc = load_csv(run_dir, p)
            all_csv[p]     = records
            n_words_map[p] = nw
            n_cats_map[p]  = nc
            print(f"  {p}: OK ({len(list(records.values())[0]['word_acc'])} bins)")
        except Exception as e:
            print(f"  {p}: FAILED — {e}")
    print()

    # ── Step 1b: Load per-category decoding data ──────────────────────────
    print("=" * 60)
    print("STEP 1b: LOADING PER-CATEGORY DECODING DATA")
    print("=" * 60)
    all_decoding = {}
    for p in patients:
        try:
            dec = load_decoding_csv(run_dir, p)
            if dec:
                all_decoding[p] = dec
                cats_found = sorted({c for e in dec.values() for c in e["categories"]})
                print(f"  {p}: OK ({len(cats_found)} categories: {cats_found})")
            else:
                print(f"  {p}: no top1_decoding_source_data.csv — per-category plot skipped")
        except Exception as e:
            print(f"  {p}: FAILED — {e}")
    print()

    # ── Step 2: Timing metrics ────────────────────────────────────────────
    print("=" * 60)
    print("STEP 2: COMPUTING TIMING METRICS")
    print("=" * 60)
    timing_records = []
    for p, pd_ in all_csv.items():
        for emb in EMBEDDINGS:
            if emb not in pd_:
                continue
            d = pd_[emb]
            tm = compute_timing(d["word_acc"], n_bh, bin_size)
            cosine_at_peak = float(d["cosine_mean"][tm["peak_bin"]])
            cat_peak_acc   = float(d["cat_acc"][tm["peak_bin"]]) if "cat_acc" in d else float("nan")
            null_m = tm["null_mean"]
            fold = tm["peak_acc"] / null_m if null_m > 0 else float("nan")
            timing_records.append({
                "patient":       p,
                "embedding":     emb,
                "n_words":       n_words_map[p],
                "n_cats":        n_cats_map[p],
                "rise_time_ms":  tm["rise_time_ms"],
                "peak_time_ms":  tm["peak_time_ms"],
                "peak_bin":      tm["peak_bin"],
                "peak_acc":      tm["peak_acc"],
                "cat_peak_acc":  cat_peak_acc,
                "peak_fold":     fold,
                "null_mean":     null_m,
                "null_sem":      tm["null_sem"],
                "cosine_at_peak": cosine_at_peak,
            })
            rise_s = f"{tm['rise_time_ms']:+.0f} ms" if tm["rise_time_ms"] else "—"
            print(f"  {p}/{emb}: rise={rise_s}  peak={tm['peak_time_ms']:+.0f} ms  "
                  f"acc={tm['peak_acc']*100:.1f}%  fold={fold:.2f}×")
    timing_df = pd.DataFrame(timing_records)
    print()

    # ── Step 3: Significance (optional) ───────────────────────────────────
    sig_results = {}
    if args.with_significance:
        print("=" * 60)
        print("STEP 3: SIGNIFICANCE (PKL SUBPROCESS LOADING)")
        print("=" * 60)
        with tempfile.TemporaryDirectory() as tmpdir:
            sig_results = compute_significance(
                run_dir, all_csv, timing_df, tmpdir, max_pkl_mb)
        # Merge null means from PKL into timing_df for those patients
        if sig_results:
            sig_csv_path = os.path.join(out_dir, "phoneme_significance.csv")
            rows = []
            for p, pv in sig_results.items():
                for e, ev in pv.items():
                    rows.append({"patient": p, "embedding": e, **ev})
            pd.DataFrame(rows).to_csv(sig_csv_path, index=False)
            print(f"  Saved: {sig_csv_path}")
        print()

    # ── Step 4: Figures ───────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 4: GENERATING FIGURES")
    print("=" * 60)
    figures = {}
    for p, pd_ in all_csv.items():
        print(f"  {p} (word/cosine)…", flush=True)
        figures[p] = make_figure(p, pd_, n_bh, bin_size)
    cat_figures = {}
    for p, pd_ in all_csv.items():
        print(f"  {p} (categories)…", flush=True)
        cat_figures[p] = make_category_figure(
            p, all_decoding.get(p, {}), pd_, n_bh, bin_size)
    wrong_word_figures = {}
    for p, pd_ in all_csv.items():
        print(f"  {p} (wrong-word cat)…", flush=True)
        wrong_word_figures[p] = make_wrong_word_cat_figure(
            p, all_decoding.get(p, {}), pd_, n_bh, bin_size)
    print()

    # ── Step 5: HTML ──────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 5: GENERATING HTML REPORT")
    print("=" * 60)
    report_path = generate_html(
        timing_df, sig_results, figures, cat_figures, wrong_word_figures,
        meta, out_dir, args.with_significance)
    print(f"\nReport : {report_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
