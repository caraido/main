# -*- coding: utf-8 -*-
"""Named region scopes — which slice of the ROI vocabulary an analysis gates on.

Boundary, and it matters
------------------------
:mod:`utils.rois` is the **vendored vocabulary**, copied verbatim from the sibling
``electrode_labeling`` repository and checked for drift by
``scripts/check_roi_vocabulary.py``. It answers "what regions exist, and how are they
classified". It is never edited here.

This module answers a different question that ``main`` owns alone: **which of those regions
a given run lets into the model.** The sibling has no stake in that, which is why the scope
registry lives here and not there. Everything below is *derived* from ``rois``'s public API
(``IN_ANALYSIS`` and ``EXCLUDED_BY_REASON``) — nothing is re-listed by hand, so a scope
cannot drift from the vocabulary the way a copied list would.

Scopes
------
``tpm``   the analysis whitelist: 18 regions — temporal-parietal plus medial/deep. **The
          default since 2026-08-23**, and ``resolve("tpm") is IN_ANALYSIS``.
``tpfm``  ``tpm`` plus the ``FRONTAL`` family (23 regions). Diagnostic.

``tp`` is RETIRED, and retired rather than redefined on purpose
---------------------------------------------------------------
Until 2026-08-23 the analysis was ``tp``: 13 temporal-parietal regions, with the five medial
and deep ones excluded under an ``EXCLUDED_BY_REASON["medial"]`` group. That group is now
empty — those regions are ``in_analysis`` in the vendored vocabulary — so ``tp`` can no
longer be *derived* at all, and re-listing its 13 names by hand here is exactly the drift
this module exists to prevent.

The tempting move was to keep the name and let ``tp`` mean the new 18. **That would silently
falsify provenance.** ``roi_scope`` is written into every run's ``meta.json`` and results pkl,
and every run produced before 2026-08-23 records ``"tp"`` meaning *thirteen* regions;
``cross_task_cotrain`` even reads a missing value as ``"tp"``. Redefining the token would
make all of those read as 18-region runs with nothing failing. So the token is retired and
:func:`resolve` raises on it with an explanation — an old config fails loudly instead of
quietly meaning something new. See :data:`RETIRED`.

``tpm`` and ``tpfm`` keep the exact region sets they have always had (18 and 23), so a run
recorded under either token still means what it said. Only the default moved.

Palette coverage
----------------
Both scopes are fully covered. ``utils.roi_palette`` was re-pinned on 2026-08-23 from 13
regions to all 23 of ``rois.PALETTE_SCOPE``, so a ``tpfm`` figure no longer renders its
frontal regions in the ``other`` grey. (It previously did, silently, and the cross-task ROI
report worked around it with report-only colours.) What keeps a figure from *naming* a region
it did not draw is now ``roi_palette.legend_entries(regions=...)``, which takes the regions
actually present.

Deliberately NOT in either: ``SENSORIMOTOR``, ``OCCIPITAL``, ``SUBCORTICAL`` and
``AUDITORY_BELT``. Two of those exclusions are load-bearing rather than arbitrary —
subcortical because DK has no subcortical parcels at all (a hippocampus scope is expressible
under NMM and simply cannot exist under DK, so the two atlases would stop being peers), and
auditory belt because Heschl's and planum temporale would let an auditory-naming decoder read
the acoustics of the spoken prompt rather than word meaning, which is the perceptual-vs-lexical
confound the sharpened claim rests on.

Resolved 2026-08-11: ``analysis/cross_task/cross_task_cotrain.load_patient`` now raises when a
picture and an auditory run disagree on *scope*, beside the check that already compared
*atlas*. It had been deferred on the grounds that the diagnostic runs were not cross-task
inputs; they became cross-task inputs the same day. Note the guard reads the run's
``meta.json`` rather than its results pkl — ``roi_scope`` reached the pkl only after the first
scope runs were produced, so a pkl-based check was silent on exactly the pairs it existed to
catch.
"""

# Python here is 3.9; `str | None` and `dict[str, ...]` in annotations only evaluate under
# this future import. utils/rois.py:62 and utils/channel_filter.py:31 do the same.
from __future__ import annotations

from utils.rois import EXCLUDED_BY_REASON, FRONTAL, IN_ANALYSIS

# ── Import-time guards ────────────────────────────────────────────────────────
# Both catch a re-vendor that would corrupt a scope *silently*, which is the failure mode
# this repository is organised against. They cost microseconds and run once.

# 1. EXCLUDED_BY_REASON is derived from the vendored table's `reason` field. If a re-vendor
#    renames or empties a reason group, the key vanishes and `tpfm` would quietly shrink
#    back toward `tp` — a run that says it gated on 23 regions while gating on 13.
# `MEDIAL` is deliberately NOT checked any more: the 2026-08-23 re-scope moved those five
#    regions into IN_ANALYSIS, which empties the group by design. This guard fired on that
#    change, which is what it is for.
if not EXCLUDED_BY_REASON.get(FRONTAL):
    raise ImportError(
        f"utils.roi_scopes: utils.rois.EXCLUDED_BY_REASON has no non-empty {FRONTAL!r} "
        f"group (present: {sorted(EXCLUDED_BY_REASON)}). The vendored vocabulary changed "
        "shape; re-derive the scopes rather than letting 'tpfm' silently shrink to 'tpm'."
    )

# 2. The gate below tests set membership, while the code it replaces called rois.in_analysis(),
#    which looks up BY_NAME. Those two agree for every possible input UNLESS two table rows
#    share a `name` -- BY_NAME would keep the last, IN_ANALYSIS could carry a duplicate or a
#    stale entry. rois._index() guards duplicate *labels*, not names, so nothing else checks
#    this. With this guard, the substitution is provably behaviour-preserving.
if len(set(IN_ANALYSIS)) != len(IN_ANALYSIS):
    raise ImportError(
        "utils.roi_scopes: utils.rois.IN_ANALYSIS contains duplicate region names "
        f"({len(IN_ANALYSIS)} entries, {len(set(IN_ANALYSIS))} distinct). Set-membership "
        "gating is only equivalent to rois.in_analysis() while names are unique."
    )

# ── The registry ──────────────────────────────────────────────────────────────

#: The default scope. Everything that does not ask for a scope gets this. Moved from ``tp``
#: to ``tpm`` on 2026-08-23 when the analysis was re-scoped; see the module docstring.
DEFAULT = "tpm"

#: scope name -> region names, in vocabulary order. Note ``SCOPES["tpm"] is IN_ANALYSIS``:
#: the default is the same object, not a copy, so it cannot drift from it.
SCOPES: dict[str, tuple[str, ...]] = {
    "tpm": IN_ANALYSIS,
    "tpfm": IN_ANALYSIS + EXCLUDED_BY_REASON[FRONTAL],
}

#: Retired scope name -> why, and what a run recorded under it actually meant. Kept rather
#: than deleted so :func:`resolve` can explain itself instead of reporting an unknown name,
#: and so the historical meaning of the token stays written down next to the live registry.
RETIRED: dict[str, str] = {
    "tp": ("13 temporal-parietal regions, the whitelist until 2026-08-23. Retired, not "
           "redefined: the five medial/deep regions joined the analysis that day, so 'tp' "
           "can no longer be derived from the vocabulary, and re-using the name for the new "
           "18 would make every run recorded before that date read as an 18-region run. "
           "Runs with roi_scope='tp' (or none, which predates the axis) mean the OLD 13 and "
           "must not be pooled with 'tpm' runs. Use 'tpm' for new work."),
}

#: Region counts pinned by name, so a future re-vendor cannot change what a recorded token
#: meant without failing here. This is the guard that the docstring's provenance argument
#: rests on: ``tpm`` has meant these 18 since it was introduced on 2026-08-11 and ``tpfm``
#: these 23, and a run's ``meta.json`` is only interpretable if that stays true.
_PINNED_SIZES = {"tpm": 18, "tpfm": 23}
for _name, _size in _PINNED_SIZES.items():
    if len(SCOPES[_name]) != _size:
        raise ImportError(
            f"utils.roi_scopes: scope {_name!r} resolves to {len(SCOPES[_name])} regions, "
            f"not the {_size} it has always meant. Every run's meta.json records this token; "
            "changing what it resolves to would silently reinterpret past runs. Add a NEW "
            "scope name instead, and retire this one via RETIRED."
        )

#: Prose for logs, ``--help`` and run manifests. Kept apart from the name so the short token
#: stays the single identifier: one string is the CLI value AND the run-id token, so they
#: cannot disagree.
DESCRIPTIONS: dict[str, str] = {
    "tpm": (f"{len(SCOPES['tpm'])} regions: temporal-parietal + medial/deep "
            "(the analysis whitelist)"),
    "tpfm": (f"{len(SCOPES['tpfm'])} regions: temporal-parietal + medial/deep + frontal "
             "(diagnostic)"),
}

#: Valid ``--roi-scope`` values, in registry order.
CHOICES: tuple[str, ...] = tuple(SCOPES)


def resolve(name: str | None) -> tuple[str, ...]:
    """Scope name -> its region names. ``None`` means :data:`DEFAULT`.

    Takes a *name*, never a region set: an ad-hoc unnamed set would be unreproducible from a
    run's ``meta.json``, which is the whole reason the run id carries the scope token.
    """
    if name is None:
        name = DEFAULT
    try:
        return SCOPES[name]
    except KeyError:
        pass
    if name in RETIRED:
        raise ValueError(
            f"ROI scope {name!r} is retired: {RETIRED[name]}"
        ) from None
    raise ValueError(
        f"unknown ROI scope {name!r}; choose one of {', '.join(CHOICES)}"
    ) from None
