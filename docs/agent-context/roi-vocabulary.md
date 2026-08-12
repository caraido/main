# Reference — ROI vocabulary, inclusion, and colour

Loaded on demand — routed from `AGENTS.md`. Written 2026-08-08 when the repository adopted
the `electrode_labeling` conventions. Companion to `channel-and-roi-naming.md`, which owns
the channel-name → electrode plumbing; this file owns **which regions exist, which contacts
are in the analysis, and what colour each region is**.

Everything here has one source of truth in code. Import it; do not retype it.

| Fact | Lives in | Vendored from |
|---|---|---|
| The 13 regions, their order, both atlases' label maps | `utils/rois.py` | `electrode_labeling/electrode_labeling/roi.py` |
| Region → colour, families | `utils/roi_palette.py` | `electrode_labeling/palette.py` + `region_palette.json` |
| Contacts excluded by name | `utils/rois.py` (`EXCLUDED_CONTACTS`) | `electrode_labeling/config.py` |
| Which contacts a run kept | that run's `meta.json` (`channel_selection`) and its results pkl (`clean_channel_rois`) | — |

Drift between the copy and the original is checked by
`python scripts/check_roi_vocabulary.py --sibling <path-to-electrode_labeling>`.
Run it after pulling either repository. **To change the vocabulary, edit the sibling, re-run
its `writeback` stage so the pkl columns match, then re-vendor — never the other way round.**

## The one rule

**The whitelist is applied to a column; the column never encodes the scope.**

`nmm_roi` and `dk_roi` name *every* contact — `Hippocampus`, `precentral`, `planum polare`,
`Right MTG middle temporal gyrus`. Being named is not being included. Inclusion is
membership of the run's **named scope**, applied by `main`, and a label the vocabulary has
never seen is **out**, never in. That is deliberate: under a blacklist a new participant's
unfamiliar parcel would enter the analysis silently, and that is not hypothetical — SE
arrived with six supplementary-motor contacts on the day the whitelist went in.

### Scopes (added 2026-08-11)

The scope is the *region set*; the atlas is the *column*. Independent, and both are in the
run id (`_roi-<atlas>_scope-<scope>_h<bins>`). Registry: `utils/roi_scopes.py`, selected with
`--roi-scope`.

| scope | regions | use |
|---|---|---|
| `tp` | the 13 below — `IN_ANALYSIS` | **the default.** Every paper run, and every run before 2026-08-11 |
| `tpfm` | 23 — `tp` + the `FRONTAL` and `MEDIAL` families | **diagnostic only** |

`tpfm` is diagnostic because the palette cannot follow it: `utils/roi_palette.py` is vendored
and drift-checked, so its ten added regions all render as the reserved grey — the *same* grey
as the `other` sentinel — and `legend_entries()` omits them without raising. A figure built
from a `tpfm` run looks complete while describing under half its channels.

`utils/rois.py` is **never** edited to add a scope. `roi_scopes` derives every scope from that
module's public API (`IN_ANALYSIS`, `EXCLUDED_BY_REASON`), which is what keeps
`scripts/check_roi_vocabulary.py` passing and keeps the default byte-identical.

Two exclusions are load-bearing rather than arbitrary, and a wider scope must not quietly undo
them. **Subcortical**: DK has no subcortical parcels at all, so a hippocampal scope exists
under NMM and *cannot* exist under DK — the two atlases would stop being peers.
**Auditory belt**: Heschl's and planum temporale would let an auditory-naming decoder read the
acoustics of the spoken prompt rather than word meaning, which is the perceptual-vs-lexical
confound the sharpened claim rests on.

## The 13 regions

Canonical order (`utils.rois.IN_ANALYSIS`) — grouped by family, anterior/ventral first. This
is the order every axis, legend and table must use, via `utils.roi_palette.ordered()`.

| # | Region | Family | Colour | NMM | DK |
|---|---|---|---|---|---|
| 1 | `aMTG` | middle temporal | `#83b1e7` | ✓ | ✓ |
| 2 | `pMTG` | middle temporal | `#2a78d6` | ✓ | ✓ |
| 3 | `aSTG` | superior temporal | `#92d9bf` | ✓ | ✓ |
| 4 | `pSTG` | superior temporal | `#56c49d` | ✓ | ✓ |
| 5 | `pSTS` | superior temporal | `#1baf7a` | — | ✓ |
| 6 | `aITG` | inferior temporal | `#968dcc` | ✓ | ✓ |
| 7 | `pITG` | inferior temporal | `#4a3aa7` | ✓ | ✓ |
| 8 | `aFus` | fusiform | `#c85391` | ✓ | ✓ |
| 9 | `pFus` | fusiform | `#b5176b` | ✓ | ✓ |
| 10 | `temporal pole` | temporal pole | `#eda100` | ✓ | ✓ |
| 11 | `supramarginal` | parietal | `#f3a789` | ✓ | ✓ |
| 12 | `angular` | parietal | `#eb6834` | ✓ | ✓ |
| 13 | `superior parietal` | dorsal parietal | `#7f5539` | ✓ | ✓ |
| — | `other` (sentinel) | — | `#9a9a9a` | — | — |

**The region name IS the display name.** There is no abbreviation table, in this repository
or in `electrode_labeling`; `roi_palette.display()` is near-identity and exists only so
figure code has one call to make. Participant identifiers are the opposite case and are
governed separately — `figures_for_paper/participants.json`, `display_id()` only, never
initials in anything that ships (`data-conventions.md` §Participant identifiers).

**`pSTS` is DK-only.** NMM has no parcel for it — it divides the banks of the superior
temporal sulcus between STG and MTG. So `pSTS` is in the analysis on its own anatomical
merits while being expressible by one atlas only, and every NMM-vs-DK pairing involving it is
*not comparable* rather than a disagreement. Those are different facts and must never be
pooled (`roi.comparable()` / `roi.agrees()`).

### Colour convention

One palette, both atlases. Each **family** owns a base hue, fixed in code and independent of
the cohort; its members are that hue blended towards white on a fixed ladder, so the last
member is the exact base hue and earlier ones are lighter. At most two members per family
wherever the anatomy allows — that constraint, not the hue choice, is what sets the palette's
worst colour pair.

Two consequences that are the whole point:

- **A region is the same colour in the NMM panel and the DK panel by construction.** Any
  colour difference the reader sees is a difference in the labels, never in the palette.
  Never build a colour map from the regions *present in one panel* — that is what the
  retired `_region_colors` did, by alphabetical index, and it gave a region two different
  colours across the two arms.
- **A region outside the 13 gets the reserved grey, not an invented colour.** An unexpected
  label shows up as grey rather than borrowing a neighbour's hue and reading as real.

Regenerate in the sibling with `python -m electrode_labeling.cli repin --write`; the resolved
hex is then re-vendored here. Do not re-derive the lightening in `main`.

## Which contacts are in the analysis

Four filters, in this order, all in `utils/channel_filter.py`. The order matters because
every index is a position in the patient's **full** channel list.

1. **`clean`** — artifact rejection, from the atlas pkl. Per task.
2. **per-trial bad channels** — unioned in from `trial_df['bad_channels']`.
3. **excluded by name** — `EXCLUDED_CONTACTS`; a fact about the recording, not the region.
4. **the ROI gate** — `roi in IN_ANALYSIS`, on `nmm_roi` or `dk_roi`.

Counts, verified on disk 2026-08-08 over 15 patients / 1360 contacts:

| gate | NMM | DK |
|---|---|---|
| whitelist only | 643 | 702 |
| + excluded by name | 643 | **693** |
| + `clean` | **634** | **683** |

Per patient after all three (NMM / DK): AA 50/50 · AP 40/43 · AZ 37/45 · CP 57/57 ·
DR 42/48 · EH 55/62 · EM 43/40 · KAW 36/40 · LH 53/62 · MM 28/28 · PV 37/39 · RB 42/45 ·
SE 39/41 · VB 41/41 · WBH 34/42.
(CP retired 2026-08-12; its row is kept as the archived measurement.)

**Per-trial bad channels reduce these further at run time. Quote the run's own
`channel_selection` report from its `meta.json`, never this table.**

### It is a region filter, not an electrode-type filter

"Temporal-parietal" here means the contact sits in one of the 13 **cortical regions**. It
does **not** mean the electrode is subdural. Of the 643 NMM in-analysis contacts, 572 are
sEEG depth contacts and 71 are ECoG — a depth contact that lands in supramarginal is in the
analysis. There is no depth-vs-surface flag in `main`'s pkls at all; `is_depth` exists only
in `electrode_labeling/output/{nmm,dk}/contacts_labeled.csv`.

### The retired shank rule

`_PATIENT_EXCLUDE_PREFIXES` (LH: `O V P Q R`; RB: `V`) was deleted 2026-08-08. The ROI gate
replaced it. Two reasons, both measured:

- LH's excluded shanks carry **11 of the cohort's 12** `superior parietal` contacts under
  NMM and **14 of 14** under DK, plus 5–8 `angular`. Keeping both filters would have removed
  the region from the paper.
- It never applied evenly. RB's channels are integer-named at that stage, so
  `str(cn).startswith('V')` never matched and **RB's V shank was always kept** while LH's
  rule did fire. The asymmetry was invisible.

It survives, verbatim and including that asymmetry, as `channel_filter.LEGACY_EXCLUDE_PREFIXES`,
reachable only via `--roi-atlas none`, whose sole purpose is reproducing archived runs.

### Hemispheres — and the one trap

The cohort is **left-only**. `roi_of()` writes a left region unprefixed and a right one as
`Right <roi>`, which is deliberately absent from the vocabulary, so a right-hemisphere
contact cannot enter the analysis by accident.

**`dk_roi` has no hemisphere prefix at all.** It comes from the rh fsaverage annotation, so
PV's right-shank contacts read as ordinary `aMTG` / `aSTG` / `aFus`. Verified: without the
name-based exclusion, **9 of PV's right-hemisphere contacts enter a DK-gated analysis
silently** (702 → 693). `EXCLUDED_CONTACTS` is therefore applied under **both** atlases even
though it is a no-op under NMM — that is what makes "the two arms differ only in the gate"
true rather than nearly true.

| Patient | Contacts | Why |
|---|---|---|
| PV | `RA1–5, RA10–13, RB1–4, RB10–12` (16) | RA/RB shanks are right-hemisphere; this cohort is left-only |
| EH | `W16, W17, W18` (3) | recorded but never localised — LeGUI's W shank stops at W15 |

### `unassigned` and `white matter` are not "out of scope"

They mean *the atlas had nothing to say* — no grey-matter label within the search radius, or
the surgeons' own white-matter call. That is a different state from "this region is not
temporal-parietal", and a report must not present the first as the second.
`roi.exclusion_reason()` returns `""` for them, not a reason. They fall out of the analysis
because they are not in the whitelist, but they are not evidence about anatomy.

## The two atlases are peers

Neither is the reference the other is scored against.

|  | NMM (`nmm_roi`) | DK (`dk_roi`) |
|---|---|---|
| Kind | volumetric, native space | surface, fsaverage |
| a/p split plane | each participant's own parcel centroid | one cohort-wide plane |
| Subcortical parcels | yes | **none** — a contact NMM calls hippocampus gets the nearest cortex |
| Hemisphere prefix | yes (`Right …`) | **no** |
| Unique regions | `planum polare`, `planum temporale` | `pSTS`, `paracentral`, `lateral occipital` |
| Distinct values in the column | 44 | 31 |

They agree on the ROI for only **442 of the 718** contacts either whitelists (61.6%,
verified 2026-08-08). The disagreements concentrate on anterior/posterior boundaries and the
temporal-pole/aMTG cut — which is the axis the paper's anterior-vs-posterior temporal claims
run along. Say which atlas any number came from; never mix them within a panel.

**A run is gated by an atlas as well as grouped by one**, so an NMM run and a DK run are
different channel sets, not two labellings of one. `cross_task_cotrain.load_patient` raises
if the picture and auditory arms disagree; without that check a mixed pair would silently
intersect down to the 627 contacts they happen to share.

## Reading it back off a run

Prefer the run to the data directory. A run records what it actually did:

- `meta.json` → `roi_atlas`, `roi_scope`, `roi_whitelist` (**the resolved scope's** names in
  full — 13 under `tp`, 23 under `tpfm` — so the run stays self-describing if `utils/rois.py`
  or the scope registry later changes), `excluded_contacts`, `roi_vocabulary_source`, and
  per-patient `channel_selection` counts, which now carry `roi_scope` too so the run-level
  claim can be cross-checked against what the gate actually did.
- the results pkl → `clean_channel_rois`, parallel to `clean_channel_names`.

Re-deriving a region by globbing `data/` and replaying the exclusion logic is what produced
the bug this replaces: `sorted(glob(...))[0]` picked the **auditory** atlas file for all ten
dual-task patients, whose `clean` mask differs from the picture one for seven of them, so
LH's and WBH's `ch{N}` positions resolved to the wrong electrodes. Use `clean_channel_rois`;
the resolution chain in `cross_task_region_importance.py` is a fallback for older runs only.
