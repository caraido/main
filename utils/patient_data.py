# -*- coding: utf-8 -*-
"""
utils.patient_data -- helpers for discovering patients and their data files.

Centralized constants and helpers previously duplicated (with docstring drift)
across phoneme_regression.py, semantic_regression.py, and
semantic_vanilla_retrieval.py.

Public API:
    INVALID_ANSWER_SET                 -- frozenset of sentinel/non-answer strings
    find_df_path(folder, patient, task) -- locate the per-patient df.pkl
    is_valid_answer(word)              -- True if word is a usable answer label
    extract_col(df, *candidates)       -- first matching column as float ndarray
    discover_patients(data_folder, task) -- sorted list of patients with data
"""

import os

import numpy as np


INVALID_ANSWER_SET = frozenset({
    '', 'nan', 'none', 'n/a', 'na', '?', 'x', 'pass', 'skip',
    'no response', 'nr', 'error',
})


def find_df_path(patient_folder, patient, task):
    """Locate the per-patient {patient}_{task}_df.pkl, falling back to a 'combined' variant."""
    std = os.path.join(patient_folder, f'{patient}_{task}_df.pkl')
    if os.path.exists(std):
        return std
    combined = os.path.join(patient_folder, f'{patient}_{task}_combined_df.pkl')
    if os.path.exists(combined):
        return combined
    return None


def is_valid_answer(word, invalid_set=INVALID_ANSWER_SET):
    """Return True if *word* is a usable answered label.

    Returns False for empty strings, sentinel values ('?', 'nan', 'pass', etc.),
    or strings with no alphabetic characters at all.
    """
    s = str(word).strip().lower()
    if s in invalid_set:
        return False
    if not any(c.isalpha() for c in s):
        return False
    return True


def extract_col(df, *candidates):
    """Return the first matching column from *df* as a float array.

    Tries each name in *candidates* in order.  Returns an all-NaN array of
    length ``len(df)`` if none of the candidates exist.
    """
    for col in candidates:
        if col in df.columns:
            return df[col].values.astype(float)
    return np.full(len(df), np.nan)


def discover_patients(data_folder, task):
    """Return sorted list of patient IDs that have a {patient}_{task}_df.pkl."""
    patients = []
    if not os.path.isdir(data_folder):
        return patients
    for name in sorted(os.listdir(data_folder)):
        folder = os.path.join(data_folder, name)
        if not os.path.isdir(folder):
            continue
        if find_df_path(folder, name, task) is not None:
            patients.append(name)
    return patients
