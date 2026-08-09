# -*- coding: utf-8 -*-
"""Which channels enter a model, decided in one place.

Extracted 2026-08-08 from three byte-identical copies of the same block --
``semantic_regression.py``, ``phoneme_regression.py`` and
``semantic_vanilla_retrieval.py`` -- which had been kept in sync by hand.  Editing a
channel-selection rule meant editing three files and hoping, which is exactly the failure
``utils/config.py`` exists to prevent for run ids.

TWO POLICIES, AND THEY ARE NOT MIXED
------------------------------------
``atlas="nmm"`` / ``"dk"`` is the **current** policy: artifact rejection, then per-trial bad
channels, then contacts excluded by name, then the temporal-parietal ROI whitelist.

``atlas=None`` is the **legacy** policy, reproducing runs made before 2026-08-08: artifact
rejection, per-trial bad channels, and the per-patient shank-prefix rule.  It exists so an
archived run can be reproduced, and for nothing else.  It is deliberately byte-identical to
the code it replaced -- including that rule's known asymmetry (see ``LEGACY_EXCLUDE_PREFIXES``).

The two do not compose.  The shank rule is not applied under a region gate, because the
region gate replaced it: a contact in supramarginal is in the analysis whichever shank it
sits on, and LH's excluded shanks carry 11 of the cohort's 12 ``superior parietal`` contacts
under NMM (14 of 14 under DK).  Keeping both would have removed the region from the paper.

THE COORDINATE SYSTEM
---------------------
Every index here is a position in the patient's FULL channel list, bad channels included.
That is the coordinate system the neural array is in.  ``bad_channels`` and ``remaining_idx``
partition it, and ``remaining_idx`` is what maps a model channel back to an electrode.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from utils.rois import ATLAS_COLUMN, IN_ANALYSIS, excluded_contacts, in_analysis

#: The retired per-patient shank rule, kept only for ``atlas=None``.
#:
#: Two things about it are facts, not intentions.  It was applied to the channel-name
#: strings, and RB's channels are integers at this stage, so ``str(cn).startswith('V')``
#: never matched and **RB's V shank was never actually excluded** -- while LH's rule did
#: fire.  Reproducing an archived run means reproducing that asymmetry, so it is preserved
#: here verbatim rather than corrected.
LEGACY_EXCLUDE_PREFIXES = {
    'LH': ('O', 'V', 'P', 'Q', 'R'),   # non-language shanks
    'RB': ('V',),                      # non-language shank
}


class ChannelSelection(NamedTuple):
    """The outcome of channel selection for one patient.

    ``bad_channels`` and ``remaining_idx`` index the patient's FULL channel list and
    partition it.  ``channel_names`` and ``channel_rois`` are parallel arrays over
    ``remaining_idx`` in the same order, so ``channel_rois[i]`` is the region of the channel
    that becomes model channel ``i``.
    """

    bad_channels: np.ndarray
    remaining_idx: np.ndarray
    channel_names: np.ndarray
    channel_rois: np.ndarray
    report: dict


def _roi_column(channels_df, atlas):
    """The ROI values for *atlas* as a str array, or ``None`` when unreadable."""
    if atlas in (None, 'none') or channels_df is None:
        return None
    column = ATLAS_COLUMN.get(atlas)
    if column is None:
        raise ValueError(f'unknown atlas {atlas!r}; expected one of {sorted(ATLAS_COLUMN)}')
    if column not in channels_df.columns:
        return None
    return channels_df[column].astype(str).values


def select_channels(patient, channels_df, n_channels, trial_bad_channels=None, *,
                    atlas=None) -> ChannelSelection:
    """Choose the channels for *patient*.

    ``channels_df``  the atlas pkl (``data/{PAT}/{PAT}_*channels.pkl``), or ``None``.
    ``n_channels``   the neural array's channel count; used only for the ``None`` fallback.
    ``trial_bad_channels``  ``trial_df['bad_channels'].values``, or ``None``.
    ``atlas``        ``"nmm"`` / ``"dk"`` for the current policy, ``None`` / ``"none"`` for
                     the legacy one.

    Call :func:`check_gate_is_applicable` first when *atlas* is set -- it fails early with a
    better message than this can give.
    """
    messages = []
    gated = atlas not in (None, 'none')

    if channels_df is not None:
        channel_names_all = channels_df['channel_name'].values.astype(str)
        bad_channels = (np.where(~channels_df['clean'].values.astype(bool))[0]
                        if 'clean' in channels_df.columns
                        else np.array([], dtype=int))
    else:
        channel_names_all = np.array([str(i) for i in range(int(n_channels))])
        bad_channels = np.array([], dtype=int)
    n_total = len(channel_names_all)
    n_not_clean = int(len(bad_channels))

    if trial_bad_channels is not None:
        for bc in trial_bad_channels:
            if bc is not None and len(bc) > 0:
                for ch in np.asarray(bc).ravel():
                    if (isinstance(ch, (int, float, np.integer, np.floating))
                            and not np.isnan(float(ch))):
                        bad_channels = np.union1d(bad_channels, [int(ch)])
    n_after_trials = int(len(bad_channels))

    rois_all = _roi_column(channels_df, atlas)
    n_by_name = 0
    n_out_of_scope = 0

    if gated:
        # ── Named exclusions ──────────────────────────────────────────────────
        # By contact NAME, because the reason is a fact about the recording rather than
        # about the region: EH's W16-W18 were recorded but never localised, and PV's RA/RB
        # shanks are right-hemisphere in a left-only cohort.  Applied under BOTH atlases
        # even though it is a verified no-op under NMM (643 -> 643) -- `nmm_roi` writes
        # right-hemisphere regions as `Right <roi>`, which the whitelist rejects, but
        # `dk_roi` carries no side prefix at all, so without this PV's right shanks would
        # enter a DK-gated run reading as ordinary left-hemisphere aMTG/aSTG/aFus
        # (verified: it removes exactly 9, 702 -> 693).  Applying it symmetrically is what
        # makes "the two arms differ only in the gate" true rather than nearly true.
        by_name = excluded_contacts(patient)
        if by_name:
            hit = np.array([i for i, cn in enumerate(channel_names_all)
                            if str(cn) in by_name], dtype=int)
            if len(hit):
                bad_channels = np.union1d(bad_channels, hit).astype(int)
                messages.append(f'{patient}: excluded {len(hit)} contact(s) by name')
        n_by_name = int(len(bad_channels)) - n_after_trials

        # ── The ROI gate ──────────────────────────────────────────────────────
        if rois_all is None:
            raise ValueError(
                f'{patient}: atlas {atlas!r} requested but its column is unreadable. '
                f'Call check_gate_is_applicable() first.')
        out = np.array([i for i, r in enumerate(rois_all) if not in_analysis(r)], dtype=int)
        if len(out):
            bad_channels = np.union1d(bad_channels, out).astype(int)
        n_out_of_scope = int(len(bad_channels)) - n_after_trials - n_by_name
    else:
        # ── Legacy shank rule ─────────────────────────────────────────────────
        remaining_ch_idx = np.delete(np.arange(n_total), bad_channels)
        channel_names = channel_names_all[remaining_ch_idx]
        prefixes = LEGACY_EXCLUDE_PREFIXES.get(patient)
        if prefixes:
            ex = np.array([i for i, cn in enumerate(channel_names)
                           if str(cn).startswith(prefixes)], dtype=int)
            if len(ex) > 0:
                bad_channels = np.union1d(bad_channels, remaining_ch_idx[ex]).astype(int)
                messages.append(
                    f'{patient}: removed {prefixes} shank(s) ({len(ex)} channels)')

    bad_channels = np.asarray(bad_channels, dtype=int)
    remaining_idx = np.delete(np.arange(n_total), bad_channels)
    channel_names = channel_names_all[remaining_idx]
    channel_rois = (rois_all[remaining_idx] if rois_all is not None
                    else np.array([''] * len(remaining_idx), dtype=object))

    if gated:
        messages.append(f'{patient}: ROI gate ({atlas}) kept {len(remaining_idx)} '
                        f'of {n_total} channels')

    per_roi: dict = {}
    if rois_all is not None:
        for r in channel_rois:
            per_roi[str(r)] = per_roi.get(str(r), 0) + 1

    report = {
        'atlas': atlas if gated else None,
        'n_total': n_total,
        'n_not_clean': n_not_clean,
        'n_bad_trial_channels': n_after_trials - n_not_clean,
        'n_excluded_by_name': n_by_name,
        'n_out_of_scope': n_out_of_scope,
        'n_kept': int(len(remaining_idx)),
        'per_roi': dict(sorted(per_roi.items())),
        'messages': messages,
    }
    return ChannelSelection(bad_channels, remaining_idx, channel_names, channel_rois, report)


def check_gate_is_applicable(patient, channels_df, n_channels, atlas):
    """Raise ``SystemExit`` if a region-gated run cannot honestly be run for *patient*.

    Call this BEFORE any training.  Each condition below used to be a silent degradation to
    a whole-brain, integer-named run -- and because the atlas goes into the run id, that run
    would still have been called ``roi-nmm``.  A run that says it is gated and is not is
    worse than no run at all, so these are fatal rather than warnings.
    """
    if atlas in (None, 'none'):
        return
    if channels_df is None:
        raise SystemExit(
            f'{patient}: --roi-atlas {atlas} was requested but no *_channels.pkl exists, '
            f'so the region filter cannot be applied. Fix the data, or pass '
            f'--roi-atlas none for a whole-brain run (which is named as such).')
    if len(channels_df) != int(n_channels):
        raise SystemExit(
            f'{patient}: --roi-atlas {atlas} was requested but the channels file has '
            f'{len(channels_df)} rows against {n_channels} recorded channels, so its rows '
            f'cannot be trusted to index the neural array. Refusing to run: positional '
            f'indexing here would give every channel the wrong region.')
    column = ATLAS_COLUMN[atlas]
    if column not in channels_df.columns:
        raise SystemExit(
            f'{patient}: --roi-atlas {atlas} was requested but {column!r} is not a column '
            f'of the channels file. The atlas columns were added 2026-08-07; a *.pkl.bak '
            f'sidecar predates them.')


def whitelist() -> tuple:
    """The in-analysis ROI names, in canonical order.  Re-exported for convenience."""
    return IN_ANALYSIS
