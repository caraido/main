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
``tp``    the paper's whitelist: 13 temporal-parietal regions. The default, and byte-for-byte
          the behaviour that existed before this module — ``resolve("tp") is IN_ANALYSIS``.
``tpm``   ``tp`` plus the ``MEDIAL`` family (18 regions). Diagnostic.
``tpfm``  ``tp`` plus the ``FRONTAL`` and ``MEDIAL`` families (23 regions). Diagnostic.

Both non-default scopes were added 2026-08-11, to test whether the 2026-08 narrowing of the
gate is what cost ~5% of decoding performance. ``tpm`` exists because ``tpfm`` moved two
families at once and so could not say which one earned its gains: the three form a ladder —
``tp`` → ``tpm`` adds medial, ``tpm`` → ``tpfm`` adds frontal — which is what makes the two
families separable.

**A non-default scope has no palette coverage.** ``utils.roi_palette`` is vendored too and
cannot be extended from this repo, so a figure built from one renders every added region in a
single indistinguishable grey — the same grey as the ``other`` sentinel — and omits them from
legends without raising. The cross-task ROI report is the exception: it assigns report-only
colours to palette-less regions and says so in the page.

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

from utils.rois import EXCLUDED_BY_REASON, FRONTAL, IN_ANALYSIS, MEDIAL

# ── Import-time guards ────────────────────────────────────────────────────────
# Both catch a re-vendor that would corrupt a scope *silently*, which is the failure mode
# this repository is organised against. They cost microseconds and run once.

# 1. EXCLUDED_BY_REASON is derived from the vendored table's `reason` field. If a re-vendor
#    renames or empties a reason group, the key vanishes and `tpfm` would quietly shrink
#    back toward `tp` — a run that says it gated on 23 regions while gating on 13.
for _reason in (FRONTAL, MEDIAL):
    if not EXCLUDED_BY_REASON.get(_reason):
        raise ImportError(
            f"utils.roi_scopes: utils.rois.EXCLUDED_BY_REASON has no non-empty {_reason!r} "
            f"group (present: {sorted(EXCLUDED_BY_REASON)}). The vendored vocabulary changed "
            "shape; re-derive the scopes rather than letting 'tpfm' silently shrink to 'tp'."
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

#: The default scope. Everything that does not ask for a scope gets this, and it is the
#: whitelist every analysis used before this module existed.
DEFAULT = "tp"

#: scope name -> region names, in vocabulary order. Note ``SCOPES["tp"] is IN_ANALYSIS``:
#: the default is the same object, not a copy, so it cannot drift from it.
SCOPES: dict[str, tuple[str, ...]] = {
    "tp": IN_ANALYSIS,
    "tpm": IN_ANALYSIS + EXCLUDED_BY_REASON[MEDIAL],
    "tpfm": IN_ANALYSIS + EXCLUDED_BY_REASON[FRONTAL] + EXCLUDED_BY_REASON[MEDIAL],
}

#: Prose for logs, ``--help`` and run manifests. Kept apart from the name so the short token
#: stays the single identifier: one string is the CLI value AND the run-id token, so they
#: cannot disagree.
DESCRIPTIONS: dict[str, str] = {
    "tp": f"{len(SCOPES['tp'])} temporal-parietal regions (the paper's whitelist)",
    "tpm": (f"{len(SCOPES['tpm'])} regions: temporal-parietal + medial/deep "
            "(diagnostic; no palette coverage)"),
    "tpfm": (f"{len(SCOPES['tpfm'])} regions: temporal-parietal + frontal + medial/deep "
             "(diagnostic; no palette coverage)"),
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
        raise ValueError(
            f"unknown ROI scope {name!r}; choose one of {', '.join(CHOICES)}"
        ) from None
