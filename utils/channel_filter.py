# -*- coding: utf-8 -*-
"""Which channels enter a model, decided in one place.

Extracted 2026-08-08 from three byte-identical copies of the same block --
``semantic_regression.py``, ``phoneme_regression.py`` and
``semantic_vanilla_retrieval.py`` -- which had been kept in sync by hand.  Editing a
channel-selection rule meant editing three files and hoping, which is exactly the failure
``utils/config.py`` exists to prevent for run ids.

NO HARD-CODED ELECTRODE NAMES
-----------------------------
``atlas="nmm"`` / ``"dk"``: artifact rejection, then per-trial bad channels, then contacts
excluded by name, then the ROI gate.  Which electrodes enter a model is decided by the
**atlas labels**, not by a list of shank letters.

The per-patient shank-prefix rule (``LEGACY_EXCLUDE_PREFIXES``: LH ``O/V/P/Q/R``, RB ``V``)
was **deleted 2026-08-11** on Alec's instruction -- the region gate replaced it, so nothing
needs hard-coding.  It had never composed with the gate anyway: a contact in supramarginal
is in the analysis whichever shank it sits on, and LH's excluded shanks carry 11 of the
cohort's 12 ``superior parietal`` contacts under NMM (14 of 14 under DK), so applying both
would have removed the region from the paper.

Consequence, stated plainly: ``atlas=None`` is now *whole-brain, ungated* and **no longer
reproduces a pre-2026-08-08 archived run** -- those applied the shank rule.  Every paper run
is gated, so no reported number depends on this; but an archived ungated run re-executed
today will keep LH's 45 excluded channels and RB's 8.

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
from utils.roi_scopes import DEFAULT as _DEFAULT_SCOPE, resolve as _resolve_scope


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
                    atlas=None, scope=None) -> ChannelSelection:
    """Choose the channels for *patient*.

    ``channels_df``  the atlas pkl (``data/{PAT}/{PAT}_*channels.pkl``), or ``None``.
    ``n_channels``   the neural array's channel count; used only for the ``None`` fallback.
    ``trial_bad_channels``  ``trial_df['bad_channels'].values``, or ``None``.
    ``atlas``        ``"nmm"`` / ``"dk"`` for the current policy, ``None`` / ``"none"`` for
                     the legacy one.  Selects the *column*.
    ``scope``        a name from :mod:`utils.roi_scopes` selecting the *region set*, or
                     ``None`` for :data:`utils.roi_scopes.DEFAULT` -- which is the 13-region
                     whitelist, i.e. exactly the behaviour that existed before scopes.
                     Ignored when ungated.  A *name*, never a region set: an unnamed set
                     could not be reproduced from a run's ``meta.json``.

    Call :func:`check_gate_is_applicable` first when *atlas* is set -- it fails early with a
    better message than this can give.
    """
    messages = []
    gated = atlas not in (None, 'none')
    scope_name = (scope or _DEFAULT_SCOPE) if gated else None
    scope_regions = _resolve_scope(scope) if gated else ()   # raises on an unknown name

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
        # Membership in the named scope, not rois.in_analysis(). The two are provably
        # identical at the default scope -- utils.roi_scopes guards at import that
        # IN_ANALYSIS has no duplicate names, which is the only way BY_NAME lookup and
        # set membership could ever disagree. `str(r)` mirrors in_analysis's own coercion.
        in_scope = frozenset(scope_regions)
        out = np.array([i for i, r in enumerate(rois_all) if str(r) not in in_scope],
                       dtype=int)
        if len(out):
            bad_channels = np.union1d(bad_channels, out).astype(int)
        n_out_of_scope = int(len(bad_channels)) - n_after_trials - n_by_name
    # No ungated branch: without a gate, channel selection is artifact rejection and
    # per-trial bad channels, both already applied above. The per-patient shank rule that
    # used to run here was deleted 2026-08-11 -- see the module docstring.

    bad_channels = np.asarray(bad_channels, dtype=int)
    remaining_idx = np.delete(np.arange(n_total), bad_channels)
    channel_names = channel_names_all[remaining_idx]
    channel_rois = (rois_all[remaining_idx] if rois_all is not None
                    else np.array([''] * len(remaining_idx), dtype=object))

    if gated:
        messages.append(f'{patient}: ROI gate ({atlas}/{scope_name}) kept '
                        f'{len(remaining_idx)} of {n_total} channels')

    per_roi: dict = {}
    if rois_all is not None:
        for r in channel_rois:
            per_roi[str(r)] = per_roi.get(str(r), 0) + 1

    report = {
        'atlas': atlas if gated else None,
        # Recorded per patient as well as per run, so a cell is self-describing and the
        # run-level claim can be cross-checked against what the gate actually did.
        'roi_scope': scope_name,
        'roi_scope_regions': list(scope_regions) if gated else None,
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
    """The **default** scope's ROI names, in canonical order.  Re-exported for convenience.

    Not "the whitelist" any more: since 2026-08-11 a run selects a named region set via
    ``select_channels(..., scope=)``.  :mod:`utils.roi_scopes` owns the rest.
    """
    return IN_ANALYSIS
