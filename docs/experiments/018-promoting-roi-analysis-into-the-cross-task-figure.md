---
id: 018
kind: decision
title: The cross-task figure is rebuilt around the ROI analyses, on the significant-7 cohort
status: answered
analysis: cross_task
opened: 2026-08-13
closed: 2026-08-13
runs: 2026-08-12_18-17-20_kernel_pls_balance-downsample_50boot
report: results/cross_task_cotrain/scope-tpm_h10/balance_downsample/region_importance_report_mean.html
answer: >
  Alec, 2026-08-13: figures_for_paper/cross_task ships three panels — generalization,
  normalized Jacobian ROI ranking, ROI-only accuracy — at scope-tpm_h10 /
  balance_downsample / NMM, aggregated by MEAN, on the 7 participants significant in both
  tasks. MDS, PCA, RSA and the single-participant ROI panel are retired; their source data
  is kept for the co-trained latent-space work. Tested contrasts are within-vs-cross and
  cross-vs-pooled. Region knockout was then dropped from the figure and the manuscript.
---

## The figure

| panel | measure | source column |
|---|---|---|
| a | co-training generalization, within / cross / pooled | `panel_b_generalization*.csv` |
| b | Jacobian sensitivity, cross-participant ROI ranking | `jac_sens_*_std` |
| c | ROI-only decoder, raw category accuracy | `suff_pooled_*` |

Storyline: a co-trained decoder works (**a**) → which regions does it lean on (**b**) → what
can a region do alone (**c**).

**Region knockout was dropped from the figure AND the manuscript** later the same day (Alec).
It was panel d. `04_region_knockout` still renders and keeps its `d`, but is **uncaptioned and
referenced by nothing** — internal. Results and Methods make no knockout claim, so the
whole-brain ceiling, `frac_wb_*` and "what breaks when a region is removed" are **withdrawn,
not merely un-illustrated**. Columns stay in `panel_c_roi.csv`; restoring the panel means
restoring the text too.

## The cohort change, and what it cost

Every panel is the **7** participants with ≥1 significant category-independent time bin in
**both** tasks (AA KAW LH PV RB SE WBH), derived by
`cross_task_region_importance_report.significant_participants()`, never typed.

**Effect sizes barely moved; resolution did.** Picture retention 0.818 → 0.810, auditory
0.919 → 0.922, but a two-sided paired Wilcoxon at n = 7 floors at 2/2⁷ = **0.0156**, so every
significant contrast lands there. A star means "as significant as n = 7 permits".

**The tested contrasts are `within-vs-cross` and `cross-vs-pooled`** — both against the
transfer baseline (Alec, revised same day; the first pass tested against the within-task
ceiling). **`within-vs-pooled` is not tested**, so the **retention ratio has no significance
test behind it** and must be reported descriptively. It read p = 0.0156 / 0.047 while it was
tested. Do not re-add the contrast to give a sentence a p-value; change the sentence. What
the figure shows instead is stronger: within **and** pooled each beat cross in **both** tasks
for category-independent accuracy, all at 0.0156.

**BH over the 12 tests changes nothing here** — all eight starred contrasts hold at q = 0.023,
the four auditory word/cosine ones are n.s. either way. Not true of the previous pair
(auditory within-vs-pooled failed at q = 0.070). Figure reports **uncorrected** P; both ship
in `panel_b_generalization_stats.csv`.

## What panel c can and cannot say

`suff_delta_*`, `suff_null_*` and `suff_p_*` are **NaN in all 74 rows** — the arm ran with
`--suff-null-draws 0`. Panel c is therefore raw ROI-only accuracy with **no size control**,
correlating with channel count at median ρ = **+0.42** within participant. Alec's call: ship
it, caveat in the caption; a `--suff-null-draws 50` pass is the fix and the obvious next compute.

The chance reference is measured, not assumed: the across-participant mean of the
label-shuffled null (picture 0.1671, auditory 0.1643), RB excluded via
`OLD_STIMULUS_SET_PATIENTS` ∩ cohort while still contributing to the markers. **The ±1 SEM
shading is in the HTML report but not on the paper panel** (Alec): at 0.1669–0.1672 it is
narrower than a marker and invited being read as a significance interval. Values stay in
`roi_chance_band.csv`; derivation and rejected forms in
[017](017-roi-report-four-page-convention.md).

**14/17 regions clear the picture reference, 17/17 the auditory one.** The three below picture
— aSTG, angular, pSTG — are perisylvian; only aSTG has more than two participants (6). A
**pattern, not a test**. pFus leads both surviving measures in both tasks.

## Display conventions (paper figure only)

Panel c (and the internal knockout panel): no title, no legend, no per-participant markers,
`alpha=0.80`, size ∝ contributing participants, dash-dot zero/chance lines, region labels
placed **radially around the cloud** with black leaders (`paper_common.place_labels`). Panel
b's axis is a bare "Normalized Jacobian sensitivity"; the definition lives in **Methods**, not
on the axis or in the caption. Three lessons from the label placer: a radius-only push cannot
separate regions collinear with the centroid; **character-count text widths under-measure bold
by ~18 %**, silently ending the de-collision loop while text still overlaps (it now measures
with the renderer); and the margin must be asymmetric — beside a panel reads as an annotation,
above the top spine as a broken figure. HTML report pages **unchanged** (Alec).

## Retired, and deliberately not deleted

The MDS panel, S1/S2 MDS+PCA (2D and 3D), S3 all-participant knockout, S7 RSA and the
single-participant ROI bar panel are gone from the package. `mds()` and `rsa()` remain
**defined but uncalled** in `compute_cross_task_data.py`, and their CSVs stay tracked **at the
previous N = 9 cohort**, verified byte-identical after the rebuild.

The successor question — can a 2-D scatter show both tasks landing in one space? — was
piloted in `tests/cotrain_latent_space/` and **did not produce a promotable panel**. Three
views off one co-trained out-of-fold fit: predicted-GloVe MDS best (mean cross-task centroid
alignment 0.356 at the word level, 1/7 significant), co-trained PLS latent space 0.175,
picture-defined discriminants projected to auditory 0.029. Single-trial clouds overlap almost
completely in all three and bootstrap ellipses put most per-category shifts below resolution
at this n. Two findings worth keeping: the latent space barely encodes task (max component
AUC 0.60–0.81), and trial-level alignment is optimistic against word-level because repeated
words tighten the permutation null.

## This reinstates manuscript material that entry 003 retired

Entry [003](003-cross-task-paragraph-overclaims.md) recorded Alec's 2026-08-11 decision to
**remove** the cross-task section rather than correct it, pending new auditory analyses. This
supersedes that: the section is rewritten in the draft as tracked changes. All four boundaries
003 set are satisfied — the upsampled numbers, VIP and the "amodal" wording are struck
(discharging the VIP removal owed by [009](009-tracked-doc-corrections-not-applied.md)), and
the claim is a shared, alignable subspace. **The `XX/YY/ZZ` "common loci" placeholder in the
Discussion was NOT filled**: [007](007-manuscript-fields-awaiting-content.md) flags it as
inviting an unsupported claim, and it still would — no region clears the BH-corrected group
test. Filling it needs the matched-N null.
