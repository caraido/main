# -*- coding: utf-8 -*-
"""Shared constants for the auditory_alignment pilot: cues, patients, metric registry."""

from collections import OrderedDict

ANALYSIS = "auditory_alignment"

# ── Cues ──────────────────────────────────────────────────────────────────────
# user's cue name -> semantic_regression.ALIGN_CUE value (== cue_style.json key).
# stim_on/off are derived (first prompt-word onset / last prompt-word offset); go_cue
# and voice_on are direct columns. voice_offset exists but is not part of this study.
CUES = OrderedDict([
    ("stim_on",  "aud_stim_onset"),
    ("stim_off", "aud_stim_offset"),
    ("go_cue",   "go_cue"),
    ("voice_on", "voice_onset"),
])

# Short labels for panels/tables (cue_style.json labels are the long form used on bands).
CUE_LABELS = OrderedDict([
    ("stim_on",  "Stim on"),
    ("stim_off", "Stim off"),
    ("go_cue",   "Go cue"),
    ("voice_on", "Voice on"),
])

# semantic_regression ALIGN_CUE value -> user cue key (inverse of CUES).
ALIGN_TO_CUEKEY = {v: k for k, v in CUES.items()}

# ── Patients ──────────────────────────────────────────────────────────────────
AUD_PATIENTS = ["AA", "AZ", "DR", "LH", "RB", "WBH"]

# Patients flagged in the report (not filtered): RB's go_cue precedes the auditory
# prompt (atypical cue geometry); AA/DR have very few auditory trials -> unstable
# word top-k. See CLAUDE.md.
ATYPICAL_PATIENT = "RB"
FEW_TRIAL_PATIENTS = ("AA", "DR")

# ── Metric registry ───────────────────────────────────────────────────────────
# Each entry: (key, label, obs_attr, null_attr_or_None, family).
# obs_attr / null_attr are BasicRegressor attributes, each shaped (n_epochs, n_bins).
# null_attr is None for cosine (no per-bin null stored -> descriptive only).
# Order = the user's requested metrics first (cosine, category, word top-1/3/5), then R2
# as a supplementary fit-significance metric (it carries a real null).
METRICS = [
    ("cosine",         "Cosine similarity", "all_cosine_sim",                             None,                                                "cosine"),
    ("category_indep", "Category accuracy", "all_retrieval_category_indep_balanced_acc",  "all_retrieval_category_indep_chance_balanced_acc",  "category"),
    ("word_top1",      "Word top-1",        "all_retrieval_top1",                         "all_retrieval_chance_top1",                         "word"),
    ("word_top3",      "Word top-3",        "all_retrieval_top3",                         "all_retrieval_chance_top3",                         "word"),
    ("word_top5",      "Word top-5",        "all_retrieval_top5",                         "all_retrieval_chance_top5",                         "word"),
    ("r2",             "R²",           "all_test_score",                             "all_chance",                                        "r2"),
]

METRIC_KEYS = [m[0] for m in METRICS]
METRIC_LABEL = {m[0]: m[1] for m in METRICS}
METRIC_HAS_NULL = {m[0]: (m[3] is not None) for m in METRICS}
# Metrics that carry a proper per-bin null (significance is testable for these).
TESTABLE_METRICS = [m[0] for m in METRICS if m[3] is not None]

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    epochs=50,
    embedding="GloVe",
    bin_size=100,       # ms
    n_bins_history=10,
    pctile=99,          # per-bin permutation threshold (~p<0.01 per patient)
    closest="cosine",
    model="kernel_pls",
)

# n=6 patients -> exact minimum one-sided Wilcoxon signed-rank p is 1/2**6.
N_PATIENTS_DEFAULT = len(AUD_PATIENTS)
WILCOXON_FLOOR = 1.0 / (2 ** N_PATIENTS_DEFAULT)   # 0.015625 for n=6
