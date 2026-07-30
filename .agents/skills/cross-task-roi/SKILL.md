---
name: cross-task-roi
description: Running and interpreting cross_task_region_importance.py and its HTML report — the four per-task measures, the normalization ladder, and the interpretation traps established by the 2026-07-23 external audit. Use before running the ROI pipeline or quoting any ROI number, enrichment value, or pic-vs-aud comparison.
---

# Cross-task ROI importance

## Purpose

Attribute the co-trained picture+auditory kernel-PLS decoder's performance to brain regions,
and — equally important — avoid the four ways this analysis has already produced a
misleading answer.

## Trigger conditions

- Running `analysis.cross_task.cross_task_region_importance` or its `_report`.
- Quoting any ROI number, enrichment value, or picture-vs-auditory comparison.
- Editing the cross-task paper figure (panel c, supplement S3).
- Being asked whether a region is "amodal" or task-specific.

## Required inputs

- `Speech` conda env, cwd = `main/`.
- A picture run and an auditory run id (defaults in the module).
- `--balance {none,downsample,upsample}` — **canonical is `none`**; output lands in
  `results/cross_task_cotrain/balance_<BALANCE>/`.
- Exact invocations live in `analysis/cross_task/README.md` §2. This skill does not
  duplicate them.

## Procedure

1. Choose the measure set with `--analysis {permutation,covariance,both}` (default `both`).
   `permutation` computes measures 1–3; `covariance` adds 4 and is very cheap (~3 s/patient,
   no PLS fit).
2. Add `--merge-regions` for the coarser a/p-merged ROIs. This is a **recompute**, not a
   sum — the merged knockout shuffles all of a region's anterior+posterior channels jointly,
   and Δacc is not additive across sub-regions.
3. Add `--single-modality` (~2–2.5× cost) whenever any task-specificity claim is in scope.
4. Render the report, then read it as described under *Decision points*.

**When the cohort or a pinned run changes, this is not a menu — run the whole refresh
sequence in `analysis/cross_task/README.md` §"Full refresh".** Five artifacts go stale
together (fine + merged CSVs × `balance_none` + `balance_downsample`, then both reports)
and **nothing errors if you skip four of them**: the report just re-renders older CSVs and
looks entirely normal. This bit on 2026-07-28 — CP was added, only the `balance_none` fine
pass was re-run, and the merged CSVs, the downsample arm and both HTML reports kept
describing the previous 6-participant, pre-channel-fix data. Two ordering constraints:
`--merge-regions` is a **separate pass** (`--analysis both` means permutation+covariance,
not both merge levels, and it writes a different file stem), and the reports must run
**last** because each reads the fine CSV plus the optional merged one.

**Carry `--single-modality` and `--roi-sufficiency` through every pass of that sequence**,
not just when the corresponding claim is already in scope. Omitting either silently drops
its columns and the report section that consumes them; the run still succeeds and the CSV
just shrinks — 54 with both, 38 without sufficiency, 32 without either. On 2026-07-28
`--single-modality` was dropped and both reports re-rendered without their single-modality
section, noticed only because the HTML got smaller. **Check the column count before trusting
a report.**

`--roi-sufficiency` (added 2026-07-29) is the one measure here that is *not* a knockout.
Everything else asks what breaks when a region is removed (**necessity**); it trains the
co-trained decoder on **only that region's channels** and asks what the region can do alone
(**sufficiency**). The pairing is the point: low knockout + high sufficiency = a redundant
region, which knockout alone reports as unimportant. Three things to know before quoting it:
γ is pinned to the whole-brain value across regions (sklearn's `1/n_features` default would
make kernel width a 97× function of region size); the size control is `suff_delta_*` against
same-size random channel sets, **not** a per-electrode divide (dividing an accuracy by
electrode count ranks the smallest regions highest); and its p floors at `1/(K+1)` from
`--suff-null-draws`.

### The four per-task measures

Each is a `_pic`/`_aud` column pair, ordered from the end task toward the decoder's own
objective:

1. **Δcat-acc knockout** (`perm_imp_*`) — the whole region shuffled jointly. **The only
   measure carrying a significance `group`.**
2. **Δcosine knockout** (`cos_imp_*`) — same knockout, Δ`cosine_mean`. Closest knockout to
   the PLS objective; computed free alongside measure 1.
3. **Jacobian sensitivity** (`jac_sens_*`) — analytic ‖∂ŷ/∂x‖ summed over the region.
4. **Neural–GloVe covariance** (`cov_*`, and null-corrected `cov_nc_*`) — region-total
   ‖zscore(X)ᵀ(Y−Ȳ)/(n−1)‖. The only **model-free** measure: no fit, no split, so it reads
   each task's own trials directly. **Prefer `cov_nc` cross-participant** — the raw form
   carries a 1/√n_trials sampling floor that makes covariance cluster by patient.

Measures 3–4 are magnitudes with no significance; the `group` label stays Δcat-acc-based.

## Decision points — the interpretation traps

**Never quote a region total.** Within patient, ρ(total, `n_channels`) = 0.99 (Jacobian),
0.96 (covariance). Only the two knockouts are size-robust (0.19). Totals are an implant
readout, not a brain one — which is why the report's raw-totals gallery was deleted.

**The pic = aud diagonal is not amodality evidence.** For the **Jacobian** it is
*structural*: one co-trained model scores both tasks through one shared map, so
ρ(pic, aud) = +0.99 **even per electrode**. That is why the Jacobian is drawn as a
cross-participant ROI *ranking* rather than a scatter. **Covariance keeps its scatter**
because it is model-free, so an asymmetry there is a property of the data — but its raw
region-total diagonal (+0.96) was purely the size artifact above, falling to **−0.09** per
electrode, so that panel is normalized.

**Task specificity comes only from the knockouts and the `_solo` columns.** Per electrode,
solo-pic vs solo-aud is ρ = +0.08 (Δcat-acc), +0.02 (Δcosine), +0.43 (Jacobian), against
+0.99 for the co-trained Jacobian. `--single-modality` is the **only** two-independent-
decoders control in the CSV, so any task-specificity claim must be sourced there. Second
result from it: ρ(co-trained, solo) is 0.94–0.99 for picture but 0.53–0.78 for auditory —
co-training **preserves** picture ROI reliance and **reorganizes** auditory.

**Enrichment is per task**, not joint. A joint reference imported a trial-count scale
offset — under it raw `cov` put 100 % of auditory ROIs above 1 and 94 % of picture ROIs
below. The cost of the per-task reference: diagonal distance now means *relative ROI rank
between tasks*, not absolute magnitude difference. Caveats that cannot be normalized away:
"1" is a channel-weighted mean of a right-skewed quantity, so the median ROI sits below it;
enrichment still correlates ≈ **−0.33** with ROI channel count because size and identity are
collinear by implant design; and the reference is the patient's own sampled electrodes, so
1.0 is implant-relative.

**No ROI survives a BH-corrected group test.** MTG is the only p<0.05 (0.031 → q=0.281 over
9 ROIs), and MTG is the largest or second-largest ROI in all seven patients — exactly what the
size artifact predicts. At ROI level ρ(median enrichment, mean channel count) = −0.71 fine /
−0.75 coarse, so a cross-participant ROI ranking is substantially a size ranking. Annotate
every ranked plot with `n=` participants and `ch=` mean channel count.

**Run AA separately, at `--zero-shot-frac 0.3`.** AA has 52 unique words across 53 auditory
trials (1 repeat), so a seen-word split (`--zero-shot-frac 0`) yields ~1 auditory test trial
and every bootstrap is skipped (all-NaN). AA's auditory decoding is inherently zero-shot.
AZ/DR/LH/RB/WBH are fine at 0.

**Do not raise `--region-null-shuffles` for resolution.** The region null is *pooled* over
~15 groups, so 20 shuffles already gives ~300 null values per bootstrap (p-resolution
≈0.003). The floor was the **whole-brain** test — 1 group × 20 = 20 values → 1/21 = 0.0476,
exactly the observed `wb_p_pic` in 3/6 patients — and `--wb-null-shuffles` (default 200)
fixes that for free. Raising the region count scales the dominant cost ~linearly.

**Reading auditory shares:** the pooled model decodes auditory only slightly above chance on
few trials, so the auditory whole-brain ceiling is small (~0.04–0.12). A region can hold a
large *share* (`frac_wb_aud`) while its absolute Δacc looks like noise. Shares need not sum
to 1 — coding is redundant and synergistic.

**Auditory-only decoders are underpowered** where auditory trials are few (AA 53 trials / 52
unique, DR 51/45 → ~1 trial per word). Flag this; do not filter it out.

**VIP no longer exists** (removed 2026-07-23). It attributed a linear surrogate the paper
does not report, and as a region total it was an electrode-count proxy (ρ=0.98). Note the
live inconsistency: `figures_for_paper/cross_task/` still reads a `vip` column from shipped
CSVs that retain it. Say so rather than regenerating VIP. Also gone: the retrieval-aligned
Jacobian `jac_dir_*`, a constant rescaling of `jac_sens` (ρ=0.99), surviving only as the
scalars `jac_align_pic/aud` and `jac_pr_A`. CSVs written before 2026-07-23 carry dead
`jac_dir_*` columns.

## Validation

1. `python -m py_compile analysis/cross_task/cross_task_region_importance.py` and `_report.py`.
2. Render the report and confirm both parts appear (Part 2 only renders if
   `region_importance_merged_all.csv` exists).
3. Confirm the whole-brain ceiling matches the fine run at the same seed.
4. Before quoting any number: confirm it is per-electrode or enrichment, **not** a total.

## Failure handling

- All units landing in group `neither` was a real bug, fixed 2026-06-08: mean-over-bootstraps
  observed Δacc was being compared against single-bootstrap nulls, inflating the null by
  ~√N ≈ 4.5×. `_significance_from_null` now computes per-bootstrap p-values and averages
  them. If this symptom returns, look there first.
- Set `PYTHONIOENCODING=utf-8 PYTHONUTF8=1` — the report generator crashes on cp1252.

## Outputs

`region_importance_all.csv` (+ per-patient CSVs, a 3-panel PNG, and
`region_importance_merged_all.csv` under `--merge-regions`), and
`region_importance_report.html` — all under
`results/cross_task_cotrain/balance_<BALANCE>/`.

## References

- `analysis/cross_task/README.md` §2 — exact invocations and flags
- `references/cross_task_region_importance.md` — the 2026-07-23 audit in full,
  report structure, engine function names
- `docs/agent-context/channel-and-roi-naming.md` — channel → electrode → `primary_roi`
- `figures_for_paper/cross_task/` — `compute_cross_task_data.py` → `cross_task_panels.py`
