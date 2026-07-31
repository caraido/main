# Cross-task analyses

Within-patient comparison of **picture naming** vs **auditory naming** for the
ECoG → GloVe semantic decoder. All scripts operate on the eight shared patients
(AA, AZ, CP, DR, KAW, LH, RB, WBH — CP added 2026-07-28, KAW added 2026-07-30),
align each task at its own loose-category peak bin, and use the channel
**intersection** of the two tasks.

Note that CP and RB ran an older auditory stimulus set than the other six, with
longer prompts and a different category inventory (`abstract`/`action`, no
`vehicle`). Chance for `cat_indep_bal_acc` is therefore per participant and per
task, not a flat 1/6 — see `docs/agent-context/data-conventions.md`.

Both arms are 100 epochs as of 2026-07-30: the picture side moved
`PIC_RUN_50EP` → `PIC_RUN`, ending the epoch asymmetry that left the two arms'
permutation nulls unequally resolved (p floors at ~1/(n_epochs+1)).

Run everything as a module from `main/`, using the
`Speech` conda env (`C:\Users\Owner\miniconda3\envs\Speech\python.exe`) so the
`dill`-pickled project data loads:

```bash
python -m analysis.cross_task.<script>
```

## Files

| File | Role | Output dir |
|---|---|---|
| `cross_task_cotrain.py` | **Co-training**: one kernel-PLS on pooled pic+aud trials. Answers (1) is the representation shared? (2) which electrodes are amodal? (3) can one decoder serve both tasks? | `results/cross_task_cotrain/<run>/` |
| ~~`cross_task_cotrain_report.py`~~ | **ARCHIVED** → `_archive/cross_task_reports/`. Superseded by `figures_for_paper/cross_task/cross_task_panels.py`. | — |
| `cross_task_transfer.py` | **Transfer learning**: 3-arm framework (`transfer` / `no_transfer` / `cca` / `pca_cca`) mapping one task's HGA onto the other, both directions. | `results/cross_task_transfer/` |
| `cross_task_transfer_report.py` | HTML report from transfer CSVs. | same dir → `cross_task_transfer_report.html` |
| `cross_task_regression.py` | **Subspace geometry**: compares the two tasks' PLS subspaces at peak bin (principal angles, alignment index, CCA, 2D co-projection) + cross-task decoding. | `results/cross_task_regression/` |
| `cross_task_region_importance.py` | **ROI/region importance** for the pooled model (per-channel path retired): permutation region-knockout Δacc/Δcosine + Jacobian (`--analysis permutation`) and model-free neural–GloVe covariance (`--analysis covariance`), merged into one `region_importance_all.csv` (`--analysis both`, default). CSV stores region totals — **read them per electrode** (see below). `--merge-regions` recomputes on coarser anterior/posterior-merged ROIs → `region_importance_merged_all.csv`. | `results/cross_task_cotrain/` |
| `cross_task_region_importance_report.py` | **HTML report** from `region_importance_all.csv` (+ `..._merged_all.csv`). **Two parts — fine ROIs and coarse ROIs — with the same five sections each:** (1) Δcat-acc knockout, (2) Δcosine knockout, (4) neural–GloVe covariance — all per electrode, as pic-vs-aud scatters; (3) Jacobian per electrode as a *cross-participant ROI ranking* (it is the one task-blind measure); (5) co-trained vs single-modality decoders. Everything foldable, with a nested TOC. `--balance` picks which run to report. → `region_importance_report.html` | `results/cross_task_cotrain/balance_<BALANCE>/` |
| ~~`cross_task_channel_importance_report.py`~~ | **ARCHIVED** → `_archive/cross_task_reports/`. The per-channel predecessor; its inputs are archived at `_archive/cross_task_channel_importance_results/`. | — |

## Typical workflow

### 1. Co-training (shared representation / amodal electrodes / one decoder)

```bash
# all patients, default kernel_pls model
python -m analysis.cross_task.cross_task_cotrain
# one patient, multiple models
python -m analysis.cross_task.cross_task_cotrain --patient AA --models kernel_pls ridge
# then build the paper figures (the old HTML report is archived)
python figures_for_paper/cross_task/compute_cross_task_data.py
python figures_for_paper/cross_task/cross_task_panels.py
```

Evaluates 6 conditions per bootstrap: `within_pic`, `within_aud`, `cross_p2a`,
`cross_a2p`, `pooled_pic`, `pooled_aud`. Useful flags: `--n-bootstrap`,
`--balance {none,downsample,upsample}`, `--n-perm`, `--no-figs`.

**Each run is saved separately.** The pipeline writes every invocation into its
own timestamped subfolder, e.g.
`results/cross_task_cotrain/2026-06-30_14-22-01_kernel_pls_balance-none_50boot/`,
so previous runs are never overwritten. The folder name encodes the key
parameters (timestamp, model(s), balance, bootstraps, and patient/perm if set),
and a `run_metadata.json` inside it records the full parameter set. The report
auto-selects the latest run; pass `--in-dir <run-folder>` to report on an older
one. (`--out-dir` overrides the *parent* directory the run folders are grouped
under.)

### 2. ROI region importance (which brain regions drive each task)

Reuses the co-training output dir, so run after / alongside step 1. The per-channel
analysis was retired (single-channel effects are weak under the Nystroem-RBF
dilution — almost every channel lands in `neither`); all three methods now report a
per-**region** total, keyed on `primary_roi`.

```bash
# permutation region-knockout Δacc/Δcosine + Jacobian (kernel PLS, with significance)
python -m analysis.cross_task.cross_task_region_importance --analysis permutation
# model-free neural-GloVe cross-covariance (no fit, cheap; was --analysis vip before 2026-07-23)
python -m analysis.cross_task.cross_task_region_importance --analysis covariance
# or both (default), merged into one region_importance_all.csv
python -m analysis.cross_task.cross_task_region_importance --analysis both
# coarser ROIs: merge anterior/posterior pairs (aFus+pFus->Fus, ...) -> region_importance_merged_all.csv
python -m analysis.cross_task.cross_task_region_importance --analysis both --merge-regions
# also fit picture-only & auditory-only decoders (+6 _solo cols) for the co-trained-vs-single-modality
# comparison section (~2-2.5x cost; auditory-only underpowered for AA/DR)
python -m analysis.cross_task.cross_task_region_importance --analysis both --single-modality
# HTML report: two parts (fine ROIs / coarse ROIs), five sections each.
# Part 2 only appears if region_importance_merged_all.csv exists.
python -m analysis.cross_task.cross_task_region_importance_report --balance none
python -m analysis.cross_task.cross_task_region_importance_report --balance downsample
# publication figures (read balance_none/)
python figures_for_paper/cross_task/compute_cross_task_data.py
python figures_for_paper/cross_task/cross_task_panels.py
```

#### Full refresh — run this whole block, in order, as one unit

The commands above are a **menu**; this is the **sequence**. Whenever the cohort changes
or `AUD_RUN` / `PIC_RUN*` / `NONE_BALANCE_RUN` is repointed, every artifact below goes
stale together, and three of the five are easy to forget because nothing errors when they
are skipped — the report simply renders older CSVs and looks fine. This bit on 2026-07-28:
CP was added, only the `balance_none` fine pass was re-run, and the merged CSVs, the
`balance_downsample` arm and both HTML reports silently kept describing the previous
6-participant, pre-channel-fix data.

**Keep `--single-modality` on every pass.** It adds the six `_solo` columns
(`perm_imp`/`cos_imp`/`jac_sens` × pic/aud), which are what populate the report's
co-trained-vs-single-modality section — the only place two independently trained decoders
are compared, and therefore the only support for a task-specificity claim. Dropping it
costs nothing visible: the run succeeds, the CSV goes 38 → 32 columns, and the report
renders with that section simply absent. That happened on 2026-07-28.

**Keep `--roi-sufficiency` on every pass too** (added 2026-07-29). It adds the 16 `suff_*`
columns and the report's *sufficiency* section. Every other measure on the page asks what
**breaks when a region is removed** (necessity); this asks what a region **can do alone**, by
training the decoder on that region's channels only. A region redundant with another scores
~0 on knockout while decoding well by itself, so knockout alone cannot see it. Same silent
failure mode as above: drop the flag and the run still succeeds, the CSV goes 54 → 38, and
the section simply is not there.

```bash
FLAGS="--analysis both --single-modality --roi-sufficiency"
# 1. fine ROIs, per balance setting
python -m analysis.cross_task.cross_task_region_importance $FLAGS
python -m analysis.cross_task.cross_task_region_importance $FLAGS --balance downsample
# 2. coarse (merged) ROIs — a SEPARATE pass; --analysis both means permutation+covariance,
#    NOT both merge levels, and it writes a different file stem
python -m analysis.cross_task.cross_task_region_importance $FLAGS --merge-regions
python -m analysis.cross_task.cross_task_region_importance $FLAGS --merge-regions --balance downsample
# 3. reports LAST — each reads region_importance_all.csv (required) AND
#    region_importance_merged_all.csv (optional Part 2), so both passes must exist first
python -m analysis.cross_task.cross_task_region_importance_report --balance none
python -m analysis.cross_task.cross_task_region_importance_report --balance downsample
```

Expect ~2–2.5× the base runtime per pass with `--single-modality`, plus ~45 min per pass for
`--roi-sufficiency` at the default `--suff-null-draws 50` (measured on AZ and extrapolated:
the matched-N null is ~90 % of that cost and scales linearly in K). Roughly 4–5 h for the
whole block before sufficiency, ~3 h more with it.

**Verify `region_importance_all.csv` has 54 columns** — 38 without `--roi-sufficiency`, 32
without `--single-modality` either. A column count is checkable in a second; noticing that an
HTML report got smaller is not, which is how both flags came to be dropped before.

`balance_downsample/` feeds no paper figure (`compute_cross_task_data.py` reads
`balance_none/` only) — it is the resampling control. Regenerate it anyway: an
inconsistent control is worse than no control, and whether this analysis enters the paper
is still undecided.

**Full regeneration of both resampling settings** (what the shipped CSVs come from). Four
invocations — 2 balance settings × {fine, merged} — run **sequentially**: each loads
100 MB–2.6 GB pkls one patient at a time, so parallelising risks OOM.

```bash
for BAL in none downsample; do
  for MERGE in "" --merge-regions; do
    python -m analysis.cross_task.cross_task_region_importance \
      --balance $BAL --analysis both --single-modality $MERGE \
      --region-null-shuffles 200 --wb-null-shuffles 200
  done
done
```

`--single-modality` is required for the report's section 5. Leave `--zero-shot-frac` at its 0.3
default: AA has 52 unique words / 53 auditory trials, so a seen-word split (`0`) leaves it ~1
auditory test trial and every bootstrap NaNs out. `--aud-run` defaults to the aligned linear-warp
auditory run, which is the reference for all shipped results. **This is a long job** — see the
null-shuffle sizing note below before choosing 200.

**Four per-task measures** are written per region (each a `_pic`/`_aud` pair), running from the
end task toward the decoder's covariance objective: **(1)** Δcat-acc knockout
(`perm_imp_*`, the only one with a significance `group`), **(2)** Δcosine-to-GloVe knockout
(`cos_imp_*`), **(3)** Jacobian sensitivity (`jac_sens_*`), **(4)** neural–GloVe covariance
(`cov_pic/aud`, plus null-corrected `cov_nc_*`). Motivation: Δcat-acc is downstream of what
kernel-PLS optimizes, so the more model-intrinsic measures better show which ROI the decoder
leans on.

**Read these per electrode, not as totals** (external audit, 2026-07-23). Within participant,
ρ(region total, `n_channels`) = 0.99 (Jacobian), 0.96 (covariance) — only the two knockouts are
size-robust (0.19). The report no longer plots totals at all.

**Do not read the pic = aud diagonal as amodality.** For the **Jacobian** the diagonal is
structural — one co-trained model scores both tasks through one shared map, so ρ(pic, aud) =
+0.99 even per electrode, whatever the anatomy is. There is no interpretable off-diagonal, so
the report draws it as a cross-participant ROI *ranking* instead of a scatter. **Covariance**
keeps its pic-vs-aud scatter: it involves no model at all (computed separately on each task's
own trials), so an asymmetry there is a property of the data — but note its raw region-total
diagonal (ρ = +0.96) was the size artifact, falling to −0.09 per electrode, which is why the
panel is normalized. Task specificity lives in the two knockouts (per electrode +0.07 / −0.01)
and in the `_solo` single-modality columns (`--single-modality`), the only place two
independently trained decoders are compared.

**Output is keyed on `--balance`** (2026-07-23): `results/cross_task_cotrain/balance_none/`,
`balance_downsample/`, `balance_upsample/`. Previously `none` sat loose at the analysis root
while `downsample` had a folder. `--out` still overrides. The report takes a matching
`--balance` (default `none`) and puts the setting in its `<title>` and header, since the two
reports otherwise look identical. **Do not write region files to the analysis root** — the
per-patient subdirs there (`results/cross_task_cotrain/{PAT}/`) are shared with
`cross_task_cotrain.py`, which writes `cotrain_{PAT}_*` into the same folders.

**Null-shuffle sizing — read before raising `--region-null-shuffles`.**
`_grouped_null_importance` returns a *pooled* array of `n_shuffles × n_groups` values and
`_significance_from_null` tests each unit against that whole pool. So for the **region** test
20 shuffles already gives 20 × ~15 ROIs = 300 null values per bootstrap (p-resolution ≈0.003) —
it was never at a resolution floor. The floor was the **whole-brain** test: one group, so
20 × 1 = 20 values → 1/21 = 0.0476, which is exactly the `wb_p_pic` seen in 3/6 patients. That
is what `--wb-null-shuffles` (default 200) fixes, at almost no cost since it is a single group.
Raising `--region-null-shuffles` multiplies the dominant cost roughly linearly (fine grouping:
20 → 200 takes a bootstrap-task from ~500 to ~3200 model predictions) for resolution that is not
what limits significance — BH q ≈ 0.28 is effect-size limited.

**Retired 2026-07-23: plain-PLS VIP** (`--analysis vip`, `vip*` columns, `pls_vip`,
`_pls_component_ssy`, and the `--pls-components` / `--pls-bootstrap` / `--no-pls-scale` flags).
It attributed a *linear surrogate* the paper does not report — there is no well-defined
input-space VIP under the Nyström map, which destroys the input↔feature correspondence — and as
a region total it was an electrode-count proxy (ρ = 0.98 with `n_channels`). Its one legitimate
use, "is the region ranking a Nyström artifact?", is a linear-decoder control rather than an
importance measure.

**Note the coupling:** covariance used to be computed *inside* `analyze_patient_region_vip`, so
`cov_*` / `cov_nc_*` came out of the `--analysis vip` path. It is now standalone in
`analyze_patient_region_cov(patient, pic_run, aud_run, merge=False)`, which needs no PLS fit,
no balancing and no rng — `_feature_cov` reads each task's own X/y directly, so all of that
machinery belonged to VIP alone. Consequence: `--analysis covariance` is now very cheap.

**Retired 2026-07-23: the retrieval-aligned Jacobian** (`jac_dir_*`). It was a constant
rescaling of `jac_sens` — ratio CV 0.8–6.7 % within patient/task, ρ = 0.99 as region totals and
0.95 per channel — so it carried no independent regional information. Structurally, every
gradient row factors through the same rank-≤10 PLS map (`J_j = Aᵀv_j`) and the `v_j` share a
common kernel factor, so the projection onto the correct-answer direction reduces to a per-trial
constant with no channel index. No reprojection fixes it: a margin gradient collapses
identically. **This is not the "leading singular value dominates" story** — a synthetic fit with
a nearly flat spectrum (`pr_A` ≈ 9.7 of 10) still collapses to CV 1.9 %. Only the scalar
diagnostics survive: `jac_align_pic/aud` (the ratio the region column reduced to) and
`jac_pr_A` (participation ratio of the PLS map's spectrum, recorded as a diagnostic — not as the
cause). Shipped CSVs written before this date still carry the dead `jac_dir_*` columns.

Regions are grouped `both` / `picture_only` / `auditory_only` / `neither` from the
region-knockout permutation null. Every channel sharing a `primary_roi` (from
`main/data/{PAT}/{PAT}_*channels.pkl`) is shuffled *together*, so Δacc measures the
drop when an entire region is removed — the right granularity when information is
encoded redundantly at the population level. Each region is read against the
**whole-brain ceiling** (`wb_imp_pic`/`wb_imp_aud` = Δacc when *all* channels are
shuffled = total accuracy the model attributes to the neural data); `frac_wb_*` is
each region's share of it. This is essential for **auditory**, whose ceiling is
small (few trials, weak-above-chance pooled decoding) — a region can hold a large
*share* while its absolute Δacc looks like noise. **Runs for all 6 patients** — DR/RB
gained an ROI atlas 2026-07-20. The region significance test uses
`--region-null-shuffles` (default 20, separate rng) because the region null is
pooled over ~10 regions. Keep **AA on `--zero-shot-frac 0.3`** (its auditory task is
inherently zero-shot) and the rest on `--zero-shot-frac 0`; run AA separately.
See `CLAUDE.md` for the channel-name → electrode → `primary_roi` resolution scheme
per patient (used internally by `_build_region_labels`).

**Auditory split caveat:** the auditory task has few trials; **AA has essentially
no repeated words** (52 words / 53 trials), so `--zero-shot-frac 0` leaves it ~1
auditory test trial and every bootstrap is skipped. Keep AA on the default
`--zero-shot-frac 0.3` (its auditory decoding is inherently zero-shot); AZ/LH/WBH
have 30–69 repeated words so `--zero-shot-frac 0` gives them larger, more stable
seen-word test sets and more auditory power.

### 3. Transfer learning (can one task's decoder be adapted to the other)

```bash
python -m analysis.cross_task.cross_task_transfer
python -m analysis.cross_task.cross_task_transfer_report
```

Runs both directions (`pic_to_aud`, `aud_to_pic`) for all 4 arms and reports
gain over the `no_transfer` within-task baseline.

### 4. Subspace geometry (how the two PLS subspaces relate)

```bash
python -m analysis.cross_task.cross_task_regression                 # all
python -m analysis.cross_task.cross_task_regression --patient AA --no-figs
```

## Notes

- Default runs (override with `--pic-run` / `--aud-run`):
  - picture: `2026-04-08_17-05-14_kernel_pls_cosine_50ep`
  - auditory: `2026-05-07_22-26-06_auditory_naming_warp-linear_align-aud_stim_onset_kernel_pls_cosine_50ep`

**Reproducing the auditory run (fixed 2026-07-23).** That run's `meta.json` records

```
--task auditory_naming --warp linear --align aud_stim_onset
```

at `git_commit 1aca186`, `git_dirty: true`. `--warp` was later generalized from
`{none, linear}` to `{none, stim, voice}` plus `--warp-scope`, so **that command line
stopped parsing** and the run could not be reproduced from its own provenance record.
`semantic_regression.py` now accepts `linear` as a deprecated alias and rewrites it to

```
--warp stim --warp-scope patient
```

which is *exact*, not approximate: old `linear` warped `[aud_stim_onset,
aud_stim_offset]` to **that patient's own** median stimulus duration, and `stim` +
scope `patient` is the same operation. The alias prints a deprecation line and should
not be used in new runs. Note `git_dirty: true` — the working tree at that commit had
uncommitted changes, so exact bit-level reproduction is still not guaranteed; pin
`1aca186` in Methods, or re-run the auditory features under current flag semantics
before freezing numbers.
- Model: `Pipeline([Nystroem(rbf, n_components=100), PLSRegression(n_components=10)])` → GloVe; scored by 1-NN cosine retrieval (`word_bal_acc`, `cat_indep_bal_acc`).
- Each analysis script writes per-bootstrap CSVs + static PNGs; the matching
  `*_report.py` reads those and emits a self-contained HTML report.
