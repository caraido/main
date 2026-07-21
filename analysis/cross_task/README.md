# Cross-task analyses

Within-patient comparison of **picture naming** vs **auditory naming** for the
ECoG → GloVe semantic decoder. All scripts operate on the six shared patients
(AA, AZ, DR, LH, RB, WBH), align each task at its own loose-category peak bin,
and use the channel **intersection** of the two tasks.

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
| `cross_task_region_importance.py` | **ROI/region importance** for the pooled model (per-channel path retired): permutation region-knockout Δacc + Jacobian (`--analysis permutation`) and region-total plain-PLS VIP (`--analysis vip`), merged into one `region_importance_all.csv` (`--analysis both`, default). Region score = total over the region's channels. `--merge-regions` recomputes on coarser anterior/posterior-merged ROIs → `region_importance_merged_all.csv`. | `results/cross_task_cotrain/` |
| `cross_task_region_importance_report.py` | **HTML report** from `region_importance_all.csv`: cross-participant overview + aggregated region scatter (Δpic vs Δaud, colour=region across subjects, marker=participant), consensus ranking, per-patient region scatters + tables. → `region_importance_report.html` | `results/cross_task_cotrain/` |
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
# permutation region-knockout Δacc + Jacobian (kernel PLS, with significance)
python -m analysis.cross_task.cross_task_region_importance --analysis permutation
# region-total plain-PLS VIP (linear, no significance test)
python -m analysis.cross_task.cross_task_region_importance --analysis vip
# or both (default), merged into one region_importance_all.csv
python -m analysis.cross_task.cross_task_region_importance --analysis both
# coarser ROIs: merge anterior/posterior pairs (aFus+pFus->Fus, ...) -> region_importance_merged_all.csv
python -m analysis.cross_task.cross_task_region_importance --analysis both --merge-regions
# analysis-wise HTML report (region scatters + tables, per-patient + aggregated;
# also shows a Merged ROIs section if region_importance_merged_all.csv exists)
python -m analysis.cross_task.cross_task_region_importance_report
# publication figures
python figures_for_paper/cross_task/cross_task_panels.py
```

**Six per-task measures** are written per region (each a `_pic`/`_aud` pair, shown as a
pic-vs-aud scatter in the report's "Task-importance measures" gallery), running from the
end task toward the decoder's covariance objective: **(1)** Δcat-acc knockout
(`perm_imp_*`, the only one with a significance `group`), **(2)** Δcosine-to-GloVe knockout
(`cos_imp_*`), **(3)** Jacobian sensitivity (`jac_sens_*`), **(4)** retrieval-aligned
Jacobian (`jac_dir_*`), **(5)** per-task VIP from separate pic/aud fits (`vip_pic/aud`),
**(6)** neural–GloVe covariance (`cov_pic/aud`). Motivation: Δcat-acc is downstream of what
kernel-PLS optimizes, so the more model-intrinsic measures better show which ROI the decoder
leans on.

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
- Model: `Pipeline([Nystroem(rbf, n_components=100), PLSRegression(n_components=10)])` → GloVe; scored by 1-NN cosine retrieval (`word_bal_acc`, `cat_indep_bal_acc`).
- Each analysis script writes per-bootstrap CSVs + static PNGs; the matching
  `*_report.py` reads those and emits a self-contained HTML report.
