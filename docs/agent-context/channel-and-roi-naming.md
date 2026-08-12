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

## The resolution chain — a fallback as of 2026-08-08

**New runs do not use it.** `semantic_regression.py` writes `clean_channel_rois` into the
results pkl, parallel to `clean_channel_names`, so a downstream ROI analysis reads the region
off the run instead of re-deriving it. Read the run.

The chain is kept for runs made before that:
`_build_channel_map` (raw label → electrode) → `_elec_to_region` (electrode → `nmm_roi` /
`dk_roi`, selected by `--atlas`) → `_build_region_labels` (per model-channel region;
unmatched → `unknown`). `_build_channel_map` resolves positional `ch{N}` labels against the
channels the model actually kept.

### Why re-deriving is the fallback and not the path — fixed 2026-08-08

Both functions selected their atlas file with `sorted(glob(f"{pat}_*channels*.pkl"))[0]`,
which picks the **auditory** file for all ten dual-task patients (`a` sorts before `p`). Its
`clean` mask differs from the picture one for seven of them (AA CP DR LH PV RB WBH), and
`_build_channel_map`'s `ch{N}` branch applies to AZ/LH/WBH — AZ's masks happen to be
identical, so **LH and WBH had their regions resolved against the wrong task's mask**. Every
LH and WBH region label in the cross-task ROI output predating 2026-08-08 is suspect. Both
functions now use the same explicit ladder `semantic_regression.py` uses to load the file in
the first place: task file → `{PAT}_channels.pkl` → picture file.

### LH shank bug — fixed 2026-06-12, then retired 2026-08-08

`semantic_regression.py` used to delete shank prefixes per `_PATIENT_EXCLUDE_PREFIXES`
(LH → O, V, P, Q, R; RB → V), and `_build_channel_map` had to replay the same exclusion or
`ch{N}` pointed at the wrong electrode. Historical consequence: LH's "V3" in the old Jacobian
ranking was really W8. The whole rule was retired 2026-08-08 in favour of the ROI whitelist —
see `roi-vocabulary.md` §The retired shank rule for the two measured reasons.

### RB V-exclusion gap — closed 2026-08-08

RB's channels are integer-named at the `semantic_regression.py` stage, so
`str(cn).startswith('V')` never fired and RB's V shank was never excluded, while LH's rule
did. The ROI gate keys on the atlas row rather than the channel-name string, so this class of
bug cannot recur. (The asymmetry is preserved verbatim under `--roi-atlas none`, which exists
only to reproduce archived runs.)

## ROI atlases — `data/{PAT}/*channels.pkl`

**All 15 enrolled patients have an atlas** (AP/CP/DR/EM/RB gained one 2026-07-20; PV/SE arrived
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
`primary_roi` (plus `PrCG`, 2 channels in AP only). **No code has ever read `rois`**, and
lobe-level pooling is not part of the analysis; group on `nmm_roi` / `dk_roi`.

### Historical: the 19 `primary_roi` labels (union over 15 patients, 1360 channels)

**Retired 2026-08-08 — kept to make old output readable, not as a grouping to use.** One
file per patient (picture-naming, or `AA_channels.pkl` for AA), recomputed 2026-08-07 after
PV and SE arrived. Note what is in here and not in the new vocabulary: `ant depth`,
`post depth`, `frontal` and `occipital` are non-cortical placeholders rather than
parcellation labels, and together they are 654 of the 1360 channels — which is most of the
difference between this scheme and the 634/683 the whitelist keeps.

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

**Label-normalization trap** (now archived data only — CP was retired 2026-08-12, so
`_ROI_NORMALIZE`'s hyphen case is dead for new runs but must stay for reading CP's existing
ones)**:** CP stores `temporo-occipital` (hyphenated, 5 channels) while
LH/RB store `temporooccipital`. Same region, two spellings — normalize before pooling
(`_ROI_NORMALIZE`) or CP's occipital-adjacent channels split into a phantom parcel. It is the
19th label in the union; folded in, `temporooccipital` would be 3 patients, not 2.

### `nmm_roi` and `dk_roi` — the two atlas columns

**Full reference: `docs/agent-context/roi-vocabulary.md`.** Only the plumbing facts are here.

Every `*channels.pkl` gained these two columns on 2026-08-07 and they were **rewritten on
2026-08-08** by `electrode_labeling` commit `974dc0d` ("Make NMM and DK peers via a shared
ROI vocabulary"). The 2026-08-08 form is the one on disk: 44 distinct `nmm_roi` values and 31
`dk_roi` values, **with no `other` bucket** — every contact carries its full anatomical name
(`Hippocampus`, `precentral`, `Right MTG middle temporal gyrus`, `unassigned`,
`white matter`). Scope is deliberately not encoded in the column; the whitelist is applied by
`main`. If a description of these columns mentions an `other` bucket, it predates 2026-08-08.

The `.pkl.bak` sidecars hold the pre-2026-08-07 six-column state; the original six columns are
value-identical to them (`DataFrame.equals`) and no row count changed. Both new columns are
`object`/str, never NaN, and **task-invariant** — identical in a patient's picture and
auditory files, which is what makes "prefer the picture-naming file" safe.

`primary_roi` is **retired** as of 2026-08-08. It is still in the pkls and still fully
populated, but nothing in tracked code reads it: the ROI analysis is keyed on `nmm_roi` /
`dk_roi` via `--atlas`, and the coarse anterior/posterior merge (`--merge-regions`,
`_merge_roi`, `_ROI_NORMALIZE`) is gone with it. The `temporo-occipital` spelling variant
that motivated `_ROI_NORMALIZE` exists in neither new column.

Two consequences that bit, kept because they explain existing artifacts:

- **Fusiform counts differ between the three columns**, so a claim about fusiform coverage is
  atlas-specific. Per patient (`primary_roi` / `nmm_roi` / `dk_roi`): AA 9/10/10, AP 0/4/5,
  AZ 0/3/2, CP 6/11/12, DR 6/6/7, EH 5/8/8, EM 6/7/8, KAW 0/4/3, LH 4/3/3, MM 0/0/0,
  PV 13/7/5, RB 3/5/6, SE 0/3/3, VB 8/4/5, WBH 5/3/2. **"KAW has no fusiform coverage" was a
  `primary_roi` statement** — KAW has 4 under NMM and 3 under DK. Do not upgrade that caveat
  by switching atlas without re-running the analysis under it.
- **The two atlases disagree on 38% of the contacts either whitelists** (they name the same
  region for 442 of 718). The disagreements concentrate on anterior/posterior boundaries and
  the temporal-pole/aMTG cut. Details in `roi-vocabulary.md`.
(CP retired 2026-08-12; its row is kept as the archived measurement.)

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
