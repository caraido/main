# Reference — cross-task region importance: mechanics, audit, report structure

Loaded on demand by the **cross-task-roi** skill, alongside which it now sits. Extracted
2026-07-26 from the former `Speech/CLAUDE.md`; moved here from `.claude/references/` on
2026-07-27. Exact CLI invocations live in `analysis/cross_task/README.md`; this file holds
the methods detail and the audit that constrains interpretation.

## The two pipeline scripts

### `cross_task_cotrain.py`

Co-trains a single kernel-PLS model on pooled picture + auditory trials. Evaluates 6
conditions: `within_pic`, `within_aud`, `cross_p2a`, `cross_a2p`, `pooled_pic`,
`pooled_aud`. The peak bin is found independently per task. Channels are the intersection of
both tasks' channel names. Report: `cross_task_cotrain_report.html`.

### `cross_task_region_importance.py`

Renamed from `cross_task_channel_importance.py` on 2026-07-20. **ROI/region attribution
only** — the per-channel path was retired because single-channel effects are weak under the
Nystroem-RBF dilution (almost every channel lands in `neither`). Retired channel outputs are
in `_archive/cross_task_channel_importance_results/`.

All measures report a per-region **total**, summed over the region's channels, keyed on
`primary_roi`. Selected with `--analysis {permutation,covariance,both}` (default `both`);
merged into one `region_importance_all.csv` plus per-patient
`region_importance_{PAT}_{tag}.csv` and a 3-panel PNG.

Runs for all 7 cross-task patients (AA AZ CP DR LH RB WBH — DR and RB gained an atlas
2026-07-20; CP joined the auditory cohort 2026-07-28).

## Function map

- `_build_region_labels(..., merge=)` — per model-channel region label; unmatched → `unknown`
- `_build_channel_map` — raw label → electrode
- `_elec_to_region` — electrode → `primary_roi`
- `_normalize_roi` / `_merge_roi` — naming variants and a/p merging
- `_grouped_permutation_importance` / `_grouped_permutation_importance_multi`
- `_grouped_null_importance` — returns a **pooled** `n_shuffles × n_groups` array
- `_significance_from_null` — tests against that whole pool
- `_feature_cov` / `_region_sum` — the model-free covariance path
- `analyze_patient_region_cov(patient, pic_run, aud_run, merge=False)` — standalone since
  2026-07-23: no PLS fit, no `balance`, no rng, ~3 s/patient
- `_add_standardized` — builds the per-task enrichment reference

## Whole-brain ceiling

The region CSV stores `wb_imp_pic` / `wb_imp_aud` (Δacc when ALL channels are shuffled — the
total accuracy the model attributes to the neural data) and each region's `frac_wb_pic` /
`frac_wb_aud` (its share of that ceiling). Essential for reading auditory: the pooled model
decodes auditory only slightly above chance on few trials, so the auditory ceiling is small
(~0.04–0.12) and a region can hold a large share while its absolute Δacc looks like noise.
Shares need not sum to 1 — coding is redundant and synergistic.

Examples: AA pFus ≈ 51 % of the whole-brain picture ceiling; LH post depth ≈ 87 % (picture) /
43 % (auditory).

`--wb-null-shuffles` defaults to 200. The whole-brain test is one group with no pooling, so
at the region default (20) its p quantized to 1/21 = 0.0476, where most patients sat.

## Merged-ROI mode

`--merge-regions` collapses anterior/posterior gyral pairs into a coarser ROI (aFus+pFus→Fus,
aMTG+pMTG→MTG, …; the single-letter a/p prefix is stripped) and normalizes naming variants
(temporo-occipital → temporooccipital via `_ROI_NORMALIZE`). `ant depth` and `post depth` are
kept separate. Writes a parallel `region_importance_merged_all.csv`.

It is a **recompute** on the coarser grouping, not a sum: the merged knockout shuffles all of
a region's a+p channels jointly, and Δacc is not additive across sub-regions. Same seed → the
whole-brain ceiling matches the fine run.

## Single-modality decoders

`--single-modality` (default off, ~2–2.5× cost) also trains a picture-only and an
auditory-only kernel-PLS per patient — same splits as the co-trained model, no cross-task
balancing — adding 6 `_solo` columns: `{perm_imp,cos_imp,jac_sens}_{pic,aud}_solo`, each the
solo decoder evaluated on its OWN task. No solo nulls or significance. Covariance is
model-free, so it has no solo form.

## The 2026-07-23 external audit — three things changed

**(a) Never read region totals; the report no longer plots them.** Within patient,
ρ(total, `n_channels`) = 0.99 (jac), 0.98 (VIP), 0.96 (cov) — only the two knockouts are
size-robust (0.19). The HTML report's raw-totals gallery was deleted; only per-channel and
enrichment galleries remain.

**(b) The pic = aud diagonal is NOT amodality evidence.** See the skill for the full
argument. This corrected the prior framing in `results_section.md` and `caption.md`, both
rewritten to say "shared, alignable subspace" and never "amodal code".

**Null-shuffle sizing — do not re-derive this.** `_grouped_null_importance` returns a
*pooled* `n_shuffles × n_groups` array and `_significance_from_null` tests against that whole
pool. The **region** test at 20 shuffles already had 20 × ~15 = 300 null values per bootstrap
(p-resolution ≈0.003) — never at a floor. The floor was the whole-brain test.

**(c) Retired: the retrieval-aligned Jacobian `jac_dir_*`.** A constant rescaling of
`jac_sens` (ratio CV 0.8–6.7 % within patient/task; ρ=0.99 totals, 0.95 per channel). Every
gradient row factors through the same rank-≤10 PLS map (`J_j = Aᵀv_j`) with the `v_j` sharing
a common kernel factor, so the projection onto the correct-answer direction is a per-trial
constant with no channel index. This is **not** the "σ₁ dominates" story — a synthetic fit
with `pr_A` ≈ 9.7/10 still collapses to CV 1.9 %. Survives only as the scalars
`jac_align_pic/aud` and `jac_pr_A`. CSVs written before 2026-07-23 still carry dead
`jac_dir_*` columns.

**Enrichment reference is now PER TASK** (`_add_standardized`), not joint over pic+aud. The
joint reference was defended as preserving the pic-vs-aud asymmetry; it imported a
trial-count scale offset instead — under it raw `cov` put 100 % of auditory ROIs above 1 and
94 % of picture ROIs below, with skew −0.04 (symmetric, so skew was not the cause).

## VIP removal (2026-07-23)

`--analysis vip`, all `vip*` columns, `pls_vip`, `_pls_component_ssy`, and
`--pls-components` / `--pls-bootstrap` / `--no-pls-scale` were deleted. VIP attributed a
*linear surrogate* the paper does not report (there is no well-defined input-space VIP under
the Nyström map), and as a region total it was an electrode-count proxy (ρ=0.98).

**The coupling that mattered:** covariance used to be computed inside
`analyze_patient_region_vip`, so `cov_*` came out of the VIP path. It is now standalone
`analyze_patient_region_cov`.

**Live inconsistency:** `figures_for_paper/cross_task/` still builds Fig. R3c-right and supp
S4 from the `vip` column of the shipped CSVs. Those CSVs retain the column, so nothing
breaks — but the paper and the report now disagree about whether VIP exists.

Historical note (per-channel, pre-retirement): plain-PLS VIP independently recovered the
permutation/Jacobian top *channels* (AA→T4, AZ→S3, WBH→PC13, RB→V2/V3, LH→L2), with 3/3
method agreement at 99–100 % consensus for most patients; **DR was the lone outlier**
(ρ(VIP,Jac) ≈ −0.60). The caveat that motivated retirement: top consensus channels were still
permutation-`neither` — the methods agreed on *ranking* but permutation could not *certify*
single-channel significance under Nystroem dilution.

## HTML report structure

`cross_task_region_importance_report.py` reads `region_importance_all.csv` and writes
`region_importance_report.html` to the same directory. The old per-channel
`cross_task_channel_importance_report.py` is **archived** with its inputs.

`--balance {none,downsample,upsample}` (default `none`) resolves the input dir to
`results/cross_task_cotrain/balance_<BALANCE>/`; `--in-dir` overrides. The setting appears in
the `<title>` and header — the two reports are otherwise visually identical.

**Use `--batch` to regenerate an arm** (added 2026-08-13). It writes the four standard pages
into the arm directory: `{all, significant} × {median, mean}`.

| file | cohort | cross-participant marker |
|---|---|---|
| `region_importance_report.html` | all | median |
| `region_importance_report_mean.html` | all | mean |
| `region_importance_report_significant.html` | significant | median |
| `region_importance_report_significant_mean.html` | significant | mean |

`median` + `all` keeps the bare historical filename. Cohort and aggregator appear in the
`<title>` and header, because four near-identical pages per arm is exactly how one gets
quoted for another.

- **`--aggregate {median,mean}`** switches every aggregation site at once (scatter rings,
  shared limits, heatmap, ROI ranking) via one module global, so a page cannot mix them.
  The aggregate is **unweighted** either way — a participant with 3 contacts in an ROI
  counts as much as one with 20, and marker size encodes participant count, not electrodes.
- **`--participants {all,significant}`** keeps only participants with ≥1 significant
  `category_indep` time bin in **both** tasks. 'Both' because 'either' and 'picture alone'
  each select the whole cohort. Read from
  `figures_for_paper/semantic_regression/source_data/source_data.csv`, which is a
  **different configuration** (`tp`/h5 picture, `tpfm`/h10 auditory) from any cross-task
  arm — the page carries that caveat in a box.
- **Sufficiency axes are framed from a shuffled-null chance band**, not from 0. The band is
  read from `figures_for_paper/semantic_regression/panels_cache_{picture,auditory}_GloVe.npz`
  (`{patient}__category_indep__null`, 100 shuffles x bins, <1 MB): average each kept
  participant's null, then take **mean +/- 1 SEM across participants** — the precision of
  the cohort's chance estimate, matched in scale to the markers, which are themselves
  cross-participant aggregates.
  **Exclude `utils.config.OLD_STIMULUS_SET_PATIENTS` ∩ cohort** — never hard-code the
  initials — since their category inventory differs (RB's measured auditory null 0.199
  against ~0.167). They still contribute accuracy to the markers; only the reference drops
  them, and the panel says so.
  Three forms were tried on tpm/h10 downsample (n=8); **do not retry the rejected two**:
  percentile across participants (0.1667-0.1676 / 0.1538-0.1672, CI-like, width tracks n);
  mean +/- pooled SD (0.138-0.196 / 0.093-0.235, too wide to inform — it measures how much a
  single shuffle moves, ~70x/~14x the between-participant spread); **mean +/- SEM
  (0.1668-0.1671 / 0.1621-0.1659), in use**. Never rebuild it from `1/n_categories` — a
  constant per participant whose "range" collapses to zero width once the odd-category
  participant is set aside; that was the original mistake.
  Anchor the axis on the chance **line**, not the band edge, or a wide band squashes the
  markers together. It is **not a test**: the nulls are the whole-brain decoder's (picture
  `tp`/h5, auditory `tpfm`/h10) while the markers are ROI-only decoders.
- **No per-region significance encoding.** A ring-weight Wilcoxon against chance was tried
  and removed (2026-08-13); markers are uniform.
- **Aggregated scatters draw only the aggregate markers.** Per-participant points and the
  participant legend are gone (the cloud buried the readout); the ROI-ranked strip keeps
  them because its x is a rank. Labels are de-collided in display space and flip to the
  outside of the panel, with a leader line when moved.
- **`suff_resid_*`, a size-detrended accuracy panel**, computed in the report: within
  participant, `acc ~ a + b·log2(n_channels)`, residual plotted. **Never divide an accuracy
  by channel count** — accuracy has a *chance* floor and saturates, so `acc/n` scores a
  1-channel region at chance/1 and inverts the ranking (measured ρ = −0.97 with channel
  count, against −0.11 for the residual and +0.27 raw). Knockout Δacc has a zero floor and
  is roughly additive, which is why per-electrode is right there and wrong here. The
  residual is a de-trending, not a test: only the matched-N null tests a region against
  random channels of its own size.

All of this degrades cleanly on an arm run without `--roi-sufficiency` (verified against
`scope-tpm_h5`): the sufficiency section is absent and the rest renders.

Restructured 2026-07-23 into **two parts** — Part 1 fine ROIs (15), Part 2 coarse/merged ROIs
(10) — each carrying the **same five sections** (`section_part`, one code path for both;
`slug` namespaces child ids `s-fine-*` / `s-coarse-*`):

1–2. **Region knockout · per electrode** — Δcat-acc and Δcosine as aggregated pic-vs-aud
   scatters (`MEASURES_KNOCKOUT_PC` → `section_measures` → `_aggregated_scatter`).
3. **Jacobian · cross-participant ROI ranking** — `_roi_ranked_strip` (`_STD_SPECS` →
   `section_ranked`): x = ROI ranked by descending cross-patient median, y = per-electrode
   enrichment (tasks collapsed), faded per-patient markers, ringed median sized by n, dashed
   1.0 reference, `n=`/`ch=` under every tick. No n_pat gating. The Jacobian is the only
   measure drawn this way — it is task-blind by construction.
4. **Neural–GloVe covariance · per electrode** — a pic-vs-aud scatter (`MEASURES_COV` →
   `section_cov`) on `cov_nc_*_std`. Model-free, so its task asymmetry is a data property.
5. **Co-trained vs single-modality** (`section_solo`) — 3 scatters + an ROI×decoder heatmap
   per measure. Fine ROIs only.

Then a global **Interpretation & caveats**.

Both plot types overlay a per-ROI **median across participants** (ringed, size ∝ #patients)
as the robust readout. Knockout scatters share one equal-scale range across Part 1 and Part 2
(`_shared_limits`, median-based) so the granularities are comparable. Part 2 only renders if
`region_importance_merged_all.csv` exists.

The report is collapsible: a nested TOC (`_toc_html`) + Expand/Collapse-all; parts and
sections wrapped via `_fold` (native `<details>`, ids in `_TOC`), each measure its own
`<details>`; only Part 1 and the overviews open by default. **`_fold` appends its TOC entry
on return**, so a part must file its own entry *before* building children or the parent lands
after them — `section_part` does this and `main` passes `add_toc=False`.

Deleted 2026-07-23 along with the raw-totals gallery: per-patient region scatters/tables,
cross-participant consensus, the group-enrichment table, and all VIP panels — and with them
`_patient_scatter`, `_region_table`, `_delta_cell`, `_consensus`, `_group_enrichment`, `_bh`,
`_full_limits`, `_merged_section`, `MEASURES`, `MEASURES_PC`, `MEASURES_STD`.

## Companion: `cross_task_prediction_mds.py`

The motivation for co-training. Semantic-organization map of the **two separate per-task
decoders** (no co-trainer). Per patient: word-stratified trial-level K-fold OOF predicted
GloVe (peak bin) from the picture-only and auditory-only kernel-PLS; stack both tasks →
cosine distance → metric MDS → 2D; scatter per trial by category, two subplots sharing the
space.

Companion statistic: cross-task **category-centroid alignment** (mean cosine of per-task,
mean-centered category centroids; category-shuffle p). **Mean-centering is essential** — raw
centroids are swamped by kernel-PLS's central shrinkage, which looks aligned but whose
shuffle-null is just as high. Trial-level (not word-grouped) CV is the default: word-grouping
forces auditory into a noisy pure-zero-shot regime. 5/7 significant (all but LH and CP —
CP's point estimate is negative, r = −0.14, p = 0.91, and it has the smallest shared
picture/auditory vocabulary in the cohort at 19 words); representative WBH r ≈ 0.43.
The `object/too` → `object/tool` truncation was fixed at source in `semantic_regression.py`
on 2026-07-28; runs produced after that no longer need the downstream `_CATEGORY_FIX`
normalisation, which is retained only for older runs.

## Paper figure

`figures_for_paper/cross_task/`: `compute_cross_task_data.py` (Speech env) →
`cross_task_panels.py` (CSV-only) → `caption.md` + `results_section.md`.

Canonical co-training run is **`balance=none`**. The paragraph's old 0.241/0.251 and 78 %/87 %
came from the upsample run. After CP joined the auditory cohort (2026-07-28, n=6 → 7) the
values are **0.275/0.235, 97 %/85 %** at n=7. Chance is **per participant and per task**
(1 / n_categories, 0.143–0.200), not a flat 0.167 — CP and RB ran an older auditory stimulus
set with a different category inventory. Compute reuses the run pinned as `NONE_BALANCE_RUN`
in `utils/config.py` (never retype it) for conditions/RSA plus the
region-importance CSV from `results/cross_task_cotrain/balance_none/` (path constant
`ROI_DIR`, moved there 2026-07-23; `roi()` → `panel_c_roi.csv` +
`panel_c_roi_coverage.csv`) and the latest MDS run. Maps `patient` → NUE `display_id`; the
group statistic is a paired `wilcoxon(zero_method="zsplit")` across 6 patients.

Main figure: **a** MDS (representative WBH) · **b** generalization within·cross·pooled ×
{cat-indep, word, cosine} × {pic, aud} · **c** ROI importance (representative **LH**,
`ROI_REPRESENTATIVE`: permutation Δacc + Jacobian, **2 panels** since VIP was removed,
`03_roi_importance`). Supplements: S1 MDS-all, S3 ROI-knockout Δacc all-6
(`S3_roi_importance_all`), S7 RSA. **S4 (region-total VIP) was deleted 2026-07-23** along with
`fig_roi_vip_all`.

Key claim: cross-task decoding is ≈ chance (a single-task decoder does NOT transfer) but
pooled retains 98 %/78 % — the co-trained model is what generalizes, and a few regions carry
the shared code.
