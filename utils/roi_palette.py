"""ROI -> colour, for every figure that draws a brain region.

ONE palette serves BOTH atlases.  That is the whole point of it: a region drawn from
``nmm_roi`` and the same region drawn from ``dk_roi`` are the same colour by construction,
so an NMM panel and a DK panel can sit side by side and any colour difference the reader
sees is a difference in the *labels*, never in the palette.

--------------------------------------------------------------------------------
VENDORED VALUES -- do not hand-edit a hex here.

Source   : Speech/electrode_labeling -- hues in ``electrode_labeling/palette.py``
           (``FAMILY_HUES``), family membership in the tracked ``region_palette.json``.
Commit   : 974dc0d7218b58c9bff4f888afed1f5cc9737d49 (2026-08-08) plus **uncommitted
           working-tree changes** -- the 2026-08-23 re-pin has not been committed in the
           sibling.  Re-run the drift check after that commit lands.
Derived  : 2026-08-23 by running ``palette.region_colors()`` in that repo and pasting the
           result.  Regenerate there with ``python -m electrode_labeling.cli repin --write``.

Re-pinned 2026-08-23 from 13 regions to 23, for two reasons.  The analysis moved from `tp`
to `tpm`, adding five medial regions; and the pinned set was widened past the analysis to
`rois.PALETTE_SCOPE`, so a `tpfm` figure has real colours for its frontal regions instead of
drawing them in the reserved grey.  **All 13 original hexes are byte-identical** -- verified,
not assumed.  The one region that would have moved a colour was `precuneus`: putting it in
`dorsal parietal` beside `superior parietal` re-ran the shade ladder and pushed
`superior parietal` off ``#7f5539`` onto precuneus, so precuneus got its own family instead.
The palette's worst CIEDE2000 pair is unchanged at 7.84 (`pSTG`/`pSTS`) at 13, 18 and 23.

Only the *resolved* output is vendored, not the generator.  ``palette.py`` carries
``propose`` / ``diff`` / ``write_pinned`` machinery this repository will never run, plus a
``colorsys`` lightening step; re-deriving the hues here would be a second implementation of
that step, and the two would drift.  The resolved hex is the authoritative artifact.

Drift is caught by ``python scripts/check_roi_vocabulary.py --sibling <path>``.
--------------------------------------------------------------------------------

How the palette is built, so a new region can be placed sensibly rather than guessed:
each *family* owns one base hue, fixed in code and never changing with the cohort, and its
members are that hue blended towards white by a fixed ladder -- so the last member of a
family is the exact base hue and earlier members are progressively lighter.  Anterior /
ventral first.  A family holds at most two members wherever the anatomy allows it; that
constraint, not the choice of hues, is what sets the palette's worst colour pair.

``other`` is reserved grey.  No family may resolve to it.
"""
from __future__ import annotations

from utils.rois import IN_ANALYSIS

#: The sentinel for "not one of the named regions", and its reserved grey.
OTHER = "other"
OTHER_COLOR = "#9a9a9a"

#: family -> ROIs, anterior/ventral first.  Mirrors ``region_palette.json``.
FAMILIES: dict = {
    "middle temporal":   ("aMTG", "pMTG"),
    "superior temporal": ("aSTG", "pSTG", "pSTS"),
    "inferior temporal": ("aITG", "pITG"),
    "fusiform":          ("aFus", "pFus"),
    "medial temporal":   ("entorhinal", "parahippocampal"),
    "frontal":           ("orbitofrontal", "IFG"),
    "dorsal frontal":    ("middle frontal", "superior frontal"),
    "temporal pole":     ("temporal pole",),
    "insula":            ("insula",),
    "parietal":          ("supramarginal", "angular"),
    "dorsal parietal":   ("superior parietal",),
    "cingulate":         ("cingulate",),
    "operculum":         ("frontal operculum",),
    "medial parietal":   ("precuneus",),
}

#: ROI -> hex.  All 23 of ``rois.PALETTE_SCOPE`` plus ``other`` -- **wider than the 18-region
#: analysis**, because a `tpfm` figure needs colours for its five frontal regions.  What
#: stops a narrower figure *naming* a region it did not draw is :func:`legend_entries`,
#: which takes the regions actually present.
REGION_COLORS: dict = {
    "aMTG":              "#83b1e7",
    "pMTG":              "#2a78d6",
    "aSTG":              "#92d9bf",
    "pSTG":              "#56c49d",
    "pSTS":              "#1baf7a",
    "aITG":              "#968dcc",
    "pITG":              "#4a3aa7",
    "aFus":              "#c85391",
    "pFus":              "#b5176b",
    "entorhinal":        "#6bb76b",
    "parahippocampal":   "#008300",
    "orbitofrontal":     "#6bd6e6",
    "IFG":               "#00b8d4",
    "middle frontal":    "#e78282",
    "superior frontal":  "#d62728",
    "temporal pole":     "#eda100",
    "insula":            "#e87ba4",
    "supramarginal":     "#f3a789",
    "angular":           "#eb6834",
    "superior parietal": "#7f5539",
    "cingulate":         "#9b30b5",
    "frontal operculum": "#928024",
    "precuneus":         "#00746a",
    OTHER:               OTHER_COLOR,
}


def color_of(region) -> str:
    """Colour for *region*.  Anything outside the pinned set falls to the reserved grey.

    Deliberately total: a region that is not in the vocabulary gets grey rather than an
    invented colour, so an unexpected label shows up as grey in the figure instead of
    borrowing a neighbour's hue and reading as a real region.
    """
    return REGION_COLORS.get(str(region), OTHER_COLOR)


def legend_entries(regions=None):
    """``[(family, region, hex), ...]`` in canonical order, for a figure legend.

    Restricted to *regions* when given -- pass the regions actually present so the legend
    does not advertise coverage the run does not have.  ``other`` is appended last, and only
    if it is present.
    """
    present = None if regions is None else {str(r) for r in regions}
    out = []
    for family, members in FAMILIES.items():
        for region in members:
            if present is None or region in present:
                out.append((family, region, REGION_COLORS[region]))
    if present is not None and OTHER in present:
        out.append(("", OTHER, OTHER_COLOR))
    return out


def display(region) -> str:
    """Publication name for *region*.

    Near-identity on purpose: **the ROI name IS the display name.**  There is no separate
    abbreviation table to keep in sync, in this repository or in ``electrode_labeling``.
    """
    return str(region)


def ordered(regions):
    """*regions* in canonical (``IN_ANALYSIS``) order, with anything unrecognised last.

    Use this for every axis, legend and table so a region sits in the same place in every
    figure -- and, critically, in the same place in the NMM and DK panels even when one
    atlas has a region the other lacks (DK has ``pSTS``; NMM does not).
    """
    seen = [str(r) for r in regions]
    known = [r for r in IN_ANALYSIS if r in seen]
    rest = sorted(r for r in seen if r not in IN_ANALYSIS)
    return known + rest
