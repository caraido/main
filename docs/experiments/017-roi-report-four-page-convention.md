---
id: 017
kind: decision
title: The ROI importance report emits four pages per arm, and accuracy is de-trended not divided
status: answered
analysis: cross_task
opened: 2026-08-13
closed: 2026-08-13
runs: 2026-08-12_18-17-20_kernel_pls_balance-downsample_50boot
report: results/cross_task_cotrain/scope-tpm_h10/balance_downsample/region_importance_report.html
answer: >
  Alec, 2026-08-13: cross_task_region_importance_report.py --batch is the standard way to
  regenerate an arm. It writes {all, significant} x {median, mean}. Sufficiency axes are
  framed from chance rather than 0, and raw ROI-only accuracy gets a size-DETRENDED
  companion panel rather than a per-electrode divide.
---

## The four pages

| file | cohort | marker |
|---|---|---|
| `region_importance_report.html` | all | median |
| `region_importance_report_mean.html` | all | mean |
| `region_importance_report_significant.html` | significant | median |
| `region_importance_report_significant_mean.html` | significant | mean |

`median` + `all` keeps the bare historical filename so existing links survive. Cohort and
aggregator go in the `<title>` and header — four near-identical pages per arm is exactly how
one gets quoted for another. `--aggregate` switches every aggregation site through one module
global, so a page cannot mix median and mean markers. Verified to degrade cleanly on an arm
with no sufficiency columns (`scope-tpm_h5`): that section is absent, the rest renders.

## Why accuracy is de-trended, not divided

Alec asked whether ROI-only accuracy should be normalized by channel count, as the knockout
is. Measured on this arm, within participant, Spearman ρ against channel count:

| transform | ρ (want ≈ 0) |
|---|---|
| raw accuracy | +0.27 |
| **accuracy / n_channels** | **−0.97** |
| (accuracy − chance) / n_channels | −0.03 median, range −0.65…+0.41 |
| **residual on log2(n_channels)** | **−0.11** |

Dividing does not remove the size effect, it **inverts** it, for a reason specific to
accuracy: it has a **chance floor** and saturates, so an uninformative region still scores
~chance and `chance/n` explodes as n shrinks (AA: entorhinal on 1 channel at 0.160 scores
0.1595/channel against aMTG's 0.0139 on 14). Knockout Δacc has a **zero** floor and is
roughly additive, which is why per-electrode is right there and wrong here.

`suff_resid_*` is fitted **within participant** (implant coverage is a participant property)
as `acc ~ a + b·log2(n_channels)`, residual plotted — a **de-trending, not a test**.

## Chance: the shuffled-null band

The shared axis limit forced 0 into range — right for a Δ, wrong for an accuracy living at
0.16–0.25, which put every region in one corner of a mostly-empty axis. The raw-accuracy
panel is anchored on the chance band instead (the Δ panel keeps its zero anchor), with a
**separate** reference per axis, since one line would misstate one of them.

**The band is measured, not assumed** (corrected 2026-08-13). It was first drawn as the
range of `1/n_categories` — a deterministic constant per participant, 1/6 for everyone on
the current stimulus set — so setting RB aside left identical values and the range collapsed
to zero width. **A constant has no distribution.** It now comes from the shuffled nulls in
`figures_for_paper/semantic_regression/panels_cache_{picture,auditory}_GloVe.npz`
(`{patient}__category_indep__null`, 100 shuffles × bins, <1 MB, no pkl reads):
each kept participant's null is averaged and the band is the **mean ± 1 SEM across
participants** — the precision of the cohort's chance estimate, which is the quantity on the
same footing as the markers (each of which is itself a cross-participant aggregate).

**The old-stimulus-set participants are excluded**, via
`utils.config.OLD_STIMULUS_SET_PATIENTS` ∩ cohort (never hard-coded — that membership changed
once already when CP was retired): RB's inventory differs (7 picture / 5 auditory against
6/6), putting its measured auditory null at 0.199 against ~0.167 for everyone else. It still
contributes accuracy to the markers; only the reference drops it, and the panel says so.

The picture strip draws as a line. The centre was empirical before this change too — it
merely looks theoretical because picture chance lands on 0.1670 against 1/6 = 0.1667.

**Three forms were tried and two rejected** (tpm/h10 downsample, n=8), kept so they are not
retried:

| form | picture | auditory | verdict |
|---|---|---|---|
| percentile across participants | 0.1667–0.1676 | 0.1538–0.1672 | CI-like, width tracks n |
| mean ± pooled SD | 0.138–0.196 | 0.093–0.235 | too wide to inform |
| **mean ± SEM across participants** | **0.1668–0.1671** | **0.1621–0.1659** | **in use** |

The pooled SD measures how much a *single shuffle* moves (0.029 / 0.071), ~70× and ~14× the
between-participant spread, so it swallowed the panel — only pFus cleared it in picture,
nothing in auditory. The SEM does narrow with n, the property that ruled out the percentile
form; it is used because it is matched in scale to the markers, not because it escapes that.
The axis anchors on the chance **line**, not the band edge — under the SD band, anchoring on
the edge stretched the axis to 0.09 and squashed the markers together.

**No per-region significance is drawn.** An earlier version encoded a per-region Wilcoxon
against chance in the ring weight; removed 2026-08-13 with its legend and helpers.

## Panel display (all aggregated scatters: knockout, covariance, sufficiency)

Per-participant markers are **not drawn** — with 17 regions × 9 participants the faded cloud
buried the aggregate markers. The values stay in `region_importance_<atlas>_all.csv`, and the
ROI-ranked strip still shows individuals because its x is a rank. The participant legend went
with them. Labels carry no participant count, are de-collided in display space, and flip to
the outside of the panel with a leader line when moved.

## The significant-participant filter

Rule: ≥1 significant `category_indep` time bin in **both** tasks. 'Both' because it is the
only rule that filters at this cohort — 'either task' and 'picture alone' each select all 9.
Drops AZ and DR (9 → 7).

**Source: `figures_for_paper/semantic_regression/source_data/source_data.csv`** (Alec's
choice). Its picture arm is `tp`/h5 and its auditory arm `tpfm`/h10 — **a different
configuration from any cross-task arm**, so the filter means "participants whose semantic
decoding was significant in the shipped time-course figure", not "…in this report's runs".
The page states this in a box. The self-consistent alternative would read per-bin
significance from the arm's own runs, but the shuffled null lives in the 92 MB per-patient
pkls and a CSV-only report script should not be loading those; if it is ever wanted, cache
the derived list to a small CSV rather than loading pkls at render time.
