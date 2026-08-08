# Reference — channel names, ROI atlases, and the mapping between them

Loaded on demand — routed from `AGENTS.md` and from the **cross-task-roi** skill. Extracted
2026-07-26 from the former `Speech/CLAUDE.md` and merged with the channel-name-resolution
memory note; moved here from `.claude/references/` on 2026-07-27 so Codex can read it too.

## Data loading

- Project pkls use `dill` — the `Speech` conda env is required.
- `load_patient(patient, pic_run, aud_run)` returns `(pic, aud)` dicts with keys `X`, `y`,
  `words`, `cats`, `n_channels`, `n_hist`, `chan_names`.
- **Feature layout:** the column index for channel `c`, history bin `k` is
  `c + k * n_channels`. Helper: `_channel_columns(c, n_ch, n_hist)`.
- Default runs:
  - `PIC_RUN_DEFAULT = "2026-04-08_17-05-14_kernel_pls_cosine_50ep"`
  - `AUD_RUN_DEFAULT = "2026-05-07_22-26-06_auditory_naming_warp-linear_align-aud_stim_onset_kernel_pls_cosine_50ep"`
  - Treat these as defaults, not as the pinned paper runs — confirm against
    `docs/results_index.md`.

## Channel naming, per patient

Channel names in the importance CSVs depend on how `clean_channel_names` was stored:

| Patient | Name format | Resolution |
|---|---|---|
| AA | Electrode names directly (T4, O10, S2 …) | Already correct |
| AZ | `ch{N}` = sequential position in `clean_channel_names` | Load results pkl, index by N |
| LH | `ch{N}` | Load results pkl, index by N |
| WBH | `ch{N}` | Load results pkl, index by N |
| DR | Raw integer = position in the `channel_names` column of `DR_picture_naming_df.pkl` | Load df pkl, `row["channel_names"][N]` |
| RB | Raw integer = position in `RB_picture_naming_combined_df.pkl` | Load df pkl, `row["channel_names"][N]` |

**Why AZ/LH/WBH use `ch{N}`:** the picture-naming run stored electrode names but the
auditory-naming run stored a different format, so the name intersection was empty and
`load_patient` fell back to `[f"ch{i}" for i in range(n)]`.

**Why DR/RB use integers:** their `semantic_regression_results.pkl` stores numeric strings
("0", "1", …) as channel names. Originally because no `*_channels.pkl` existed for them; one
now does (2026-07-20), so those integer indices can be mapped to electrode names via the
atlas — but the results-pkl labels themselves are unchanged.

## The resolution chain

`_build_channel_map` (raw label → electrode) → `_elec_to_region` (electrode → `primary_roi`)
→ `_build_region_labels` (per model-channel region label; unmatched → `unknown`).

`_build_channel_map` resolves positional `ch{N}` labels against the channels the model
actually kept.

### LH shank bug — fixed 2026-06-12

`semantic_regression.py` physically deletes shank prefixes per `_PATIENT_EXCLUDE_PREFIXES`
(LH → O, V, P, Q, R; RB → V) from the neural data, but `_build_channel_map` was
reconstructing names from the *pre-exclusion* `clean` column, so `ch{N}` pointed at the wrong
electrode. The fix: the anatomical-name branch (AZ/LH/WBH) now replays the same prefix
exclusion. Historical consequence: LH's "V3" in the old Jacobian ranking was really W8.

### RB V-exclusion gap — still open

At the `semantic_regression.py` stage RB's channels are integer-named, so
`str(cn).startswith('V')` never fires and RB's V shank was never excluded from the SR
data/results. This remains a latent gap **for any patient whose SR labels are integers**.
Tracked in `.claude/open-questions.md`.

## ROI atlases — `data/{PAT}/*channels.pkl`

**All 15 patients now have an atlas** (AP/CP/DR/EM/RB gained one 2026-07-20; PV/SE arrived
with theirs 2026-08-06/07). Region knockout is therefore *possible* for every patient, though
it has only *run* for the cross-task six.

**Which file to use:** prefer `{PAT}_picture_naming_channels.pkl`. ROI info is task-invariant
(the picture and auditory files carry the same `primary_roi`, `nmm_roi` and `dk_roi` — checked
row-for-row across the 10 patients that have both files; where the two files differ at all it
is in `clean`, which is per-task), so pick picture-naming for consistency. **AA is the lone odd
name — `AA_channels.pkl`** — so a glob of `*_picture_naming_channels.pkl` silently drops AA.
Use `{PAT}_*channels.pkl` and prefer the picture-naming match when several exist.

Each pkl is a `pandas.DataFrame`, one row per channel, columns
`channel_name, rois, primary_roi, LE, SA, clean, nmm_roi, dk_roi` (`clean` = kept after
artifact rejection; drops 0–10 channels per patient). Sizes 50–115 channels, 8–13 distinct
`primary_roi` each. `primary_roi` is **fully populated in all 15 — no NaN, no `neither`**.
(The `neither` group that dominated channel-importance output is a *significance* label,
unrelated to ROI coverage.)

`rois` holds the channel's full hierarchy and is a superset of `primary_roi`: it adds coarse
parents — `temporal`, `ant temporal`, `post temporal`, `parietal` — that never appear as a
`primary_roi` (plus `PrCG`, 2 channels in AP only). Group on `primary_roi` for fine parcels,
on `rois` membership for lobe-level pooling.

### The 19 `primary_roi` labels (union over 15 patients, 1360 channels)

One file per patient (picture-naming, or `AA_channels.pkl` for AA). Recomputed 2026-08-07
after PV and SE arrived; label set is unchanged, counts are not.

| Label | Pats | Chans | | Label | Pats | Chans |
|---|---|---|---|---|---|---|
| `aMTG` | 15 | 189 | | `pITG` | 9 | 41 |
| `pMTG` | 15 | 120 | | `aPHG` | 7 | 27 |
| `pSTG` | 14 | 67 | | `pPHG` | 6 | 23 |
| `ant depth` | 13 | 282 | | `aSTG` | 5 | 32 |
| `frontal` | 13 | 254 | | `aFus` | 5 | 23 |
| `post depth` | 12 | 97 | | `occipital` | 2 | 21 |
| `IPL` | 10 | 56 | | `temporooccipital` | 2 | 13 |
| `aITG` | 10 | 48 | | `postcentral` | 2 (MM/VB) | 8 |
| `pFus` | 9 | 42 | | `SPL` | 1 (LH) | 12 |

Only `aMTG` and `pMTG` are present in all 15. `SPL` is LH-only, `postcentral` is MM/VB-only —
any cross-patient region comparison must handle ragged label sets.

**Label-normalization trap:** CP stores `temporo-occipital` (hyphenated, 5 channels) while
LH/RB store `temporooccipital`. Same region, two spellings — normalize before pooling
(`_ROI_NORMALIZE`) or CP's occipital-adjacent channels split into a phantom parcel. It is the
19th label in the union; folded in, `temporooccipital` would be 3 patients, not 2.

### `nmm_roi` and `dk_roi` — two extra parcellations, added 2026-08-07

Every `*channels.pkl` gained two columns on 2026-08-07. Each file's previous state is beside
it as `*.pkl.bak`; the six original columns are **value-identical to the `.bak`**
(`DataFrame.equals` on the shared columns) for all 26 files, and no file changed row count.
The update is purely additive — nothing that reads
`primary_roi` needs to change, and both existing readers
(`cross_task_region_importance._elec_to_region`, `cross_task_regression._resolve_names`)
select columns by name, so neither is affected.

Both columns are `object`/str, never NaN, and are task-invariant (identical in a patient's
picture and auditory files). The names look like `primary_roi` labels but the label *sets*
differ — `angular`, `supramarginal`, `superior parietal`, `temporal pole`, `white matter`,
`unassigned` appear in neither `primary_roi` nor `rois`, and `pSTS`, `parahippocampal`,
`entorhinal` are `dk_roi`-only.

| | distinct values (incl. `other`) | channels outside `other` | `other` | also present |
|---|---|---|---|---|
| `primary_roi` | 19 | 1360 / 1360 | — | non-cortical placeholders `ant depth`, `post depth`, `frontal`, `occipital` |
| `nmm_roi` | 15 | 649 / 1360 (48%) | 711 | `white matter` (3), `unassigned` (3) |
| `dk_roi` | 18 | 647 / 1360 (48%) | 713 | `white matter` (3), `unassigned` (3) |

**Three traps before using either.**

1. **`other` is half the data, and it is not only depth/frontal channels.** 711 of 1360
   channels are `nmm_roi == "other"`. Most are the depth/frontal/occipital ones that
   `primary_roi` also treats as non-cortical, but **89 channels whose `primary_roi` is a
   named cortical parcel (anything but `ant depth`, `post depth`, `frontal`, `occipital`)
   still land in `other`** — 15 in WBH, 14 in MM, 12 in PV, 10 in EH, 7 each in DR and VB.
   An ROI analysis keyed on `nmm_roi` therefore drops cortical channels that the
   `primary_roi` analysis keeps, and the loss is patient-dependent — i.e. it is not a
   constant offset you can ignore in a cross-patient comparison.
2. **The two atlases disagree, and not marginally.** They differ on 201/1360 channels (14.8%);
   restricted to the 647 channels *both* label, agreement is only 69%. The largest confusions
   are `nmm=aSTG → dk=pSTG` (28), `nmm=temporal pole → dk=aMTG` (23), `nmm=aMTG → dk=pMTG`
   (18), `nmm=pMTG → dk=pITG` (13), `nmm=pMTG → dk=pSTS` (11). Anterior/posterior boundaries
   and the temporal-pole/aMTG cut are where they part company — which is exactly the axis the
   anterior-vs-posterior temporal claims run along. Pick one atlas per analysis and say which;
   never mix them within a figure.
3. **Fusiform counts move.** `primary_roi` and the two new columns do not agree on which
   channels are aFus/pFus. Per patient (`primary_roi` / `nmm_roi` / `dk_roi`): AA 9/10/10,
   AP 0/4/5, AZ 0/3/2, CP 6/11/12, DR 6/6/7, EH 5/8/8, EM 6/7/8, KAW 0/4/3, LH 4/3/3, MM
   0/0/0, PV 13/7/5, RB 3/5/6, SE 0/3/3, VB 8/4/5, WBH 5/3/2. **The "KAW has no fusiform
   coverage" note in `AGENTS.md` is a `primary_roi` statement** — under `nmm_roi` KAW has 4
   and under `dk_roi` 3. Do not silently upgrade that caveat by switching atlas; if fusiform
   coverage is re-stated under a new atlas, say which atlas and re-run the analysis under it.

**Nothing in tracked code reads `nmm_roi` or `dk_roi`** as of 2026-08-07 (`grep` over
`*.py/*.md/*.json` returns `primary_roi` only). No result, figure, or source-data CSV depends
on them, and none needs regenerating because of this update.

**Provenance is undocumented.** The generating script is not in this repository, and the
column names are the only evidence of what the atlases are (`nmm` reads as
Neuromorphometrics, `dk` as Desikan-Killiany, and the `dk`-only `entorhinal` /
`parahippocampal` / `pSTS` labels are consistent with that). Treat the atlas identity as an
inference, not a fact, until Alec confirms it — and record the answer here.

## Historical: per-channel importance results (pre-fix run, 2026-06-08)

Retained for provenance only — the per-channel path is retired and these numbers predate the
significance fix. Retired CSVs are in `_archive/cross_task_channel_importance_results/`.

| Patient | Best pic channel | Δacc pic | Best aud channel | Δacc aud | Note |
|---|---|---|---|---|---|
| AA | T4 | +0.046 | T4 | +0.058 | #1 in both tasks |
| AZ | S3 | +0.034 | PC14 | +0.003 | S3/S4/S5 cluster (same shank) |
| DR | A2 | +0.004 | T2 | +0.024 | T2 = auditory-only candidate |
| LH | U9 | +0.031 | W8 | +0.002 | L2 had the highest Jacobian across patients |
| RB | V3 | +0.067 | V3 | +0.003 | Largest pic importance in the dataset |
| WBH | PC13 | +0.035 | O10 | +0.003 | O6–O9 cluster in ranks 2–5 |

Shank-level clustering (adjacent electrodes dominating together) appeared in AZ (S3–S5),
WBH (O6–O9), and RB (V2–V3), which argued for genuine cortical signal rather than noise.
Note the LH entries reflect the post-fix electrode names.
