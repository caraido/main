# Cross-task analyses

Within-patient comparison of **picture naming** vs **auditory naming** for the
ECoG → GloVe semantic decoder. All scripts operate on the six shared patients
(AA, AZ, DR, LH, RB, WBH), align each task at its own loose-category peak bin,
and use the channel **intersection** of the two tasks.

Run everything as a module from the project root (`d:\...\Speech`), using the
`Speech` conda env (`C:\Users\Owner\miniconda3\envs\Speech\python.exe`) so the
`dill`-pickled project data loads:

```bash
python -m main.analysis.cross_task.<script>
```

## Files

| File | Role | Output dir |
|---|---|---|
| `cross_task_cotrain.py` | **Co-training**: one kernel-PLS on pooled pic+aud trials. Answers (1) is the representation shared? (2) which electrodes are amodal? (3) can one decoder serve both tasks? | `results/cross_task_cotrain/<run>/` |
| `cross_task_cotrain_report.py` | HTML report from co-training CSVs (auto-selects the latest run). | inside the run folder → `cross_task_cotrain_report.html` |
| `cross_task_transfer.py` | **Transfer learning**: 3-arm framework (`transfer` / `no_transfer` / `cca` / `pca_cca`) mapping one task's HGA onto the other, both directions. | `results/cross_task_transfer/` |
| `cross_task_transfer_report.py` | HTML report from transfer CSVs. | same dir → `cross_task_transfer_report.html` |
| `cross_task_regression.py` | **Subspace geometry**: compares the two tasks' PLS subspaces at peak bin (principal angles, alignment index, CCA, 2D co-projection) + cross-task decoding. | `main/test/results/semantic_regression/cross_task_regression/` |
| `cross_task_channel_importance.py` | **Per-channel + per-region importance** for the pooled model: permutation Δacc + Jacobian sensitivity (`--analysis permutation`), and plain-PLS VIP (`--analysis pls`). The permutation pass also knocks out whole brain regions (`region_importance_*.csv`). | `results/cross_task_cotrain/` |
| `cross_task_channel_importance_report.py` | HTML report synthesizing all three importance methods + cross-patient consensus ranking. | `results/cross_task_cotrain/channel_importance_report.html` |

## Typical workflow

### 1. Co-training (shared representation / amodal electrodes / one decoder)

```bash
# all patients, default kernel_pls model
python -m main.analysis.cross_task.cross_task_cotrain
# one patient, multiple models
python -m main.analysis.cross_task.cross_task_cotrain --patient AA --models kernel_pls ridge
# then build the report
python -m main.analysis.cross_task.cross_task_cotrain_report
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

### 2. Channel importance (which electrodes drive each task)

Reuses the co-training output dir, so run after / alongside step 1:

```bash
# permutation Δacc + Jacobian (kernel PLS, with significance)
python -m main.analysis.cross_task.cross_task_channel_importance --analysis permutation
# plain-PLS VIP (linear, no significance test)
python -m main.analysis.cross_task.cross_task_channel_importance --analysis pls
# or both
python -m main.analysis.cross_task.cross_task_channel_importance --analysis both
# synthesis report
python -m main.analysis.cross_task.cross_task_channel_importance_report
```

Channels are grouped `both` / `picture_only` / `auditory_only` / `neither` from
the permutation null. VIP fills the gap where kernel permutation is underpowered
(Nystroem dilution → few significant channels). See `CLAUDE.md` for channel-name
conventions per patient (AA = electrode names; AZ/LH/WBH = `ch{N}` positional;
DR/RB = integer index).

**Brain-region permutation** runs automatically inside `--analysis permutation`:
every channel sharing a `primary_roi` (from `main/data/{PAT}/{PAT}_*channels.pkl`)
is shuffled *together*, so Δacc measures the drop when an entire region is removed
— the right granularity when information is encoded redundantly at the population
level, so no single channel is indispensable. Writes `region_importance_all.csv`
(+ per-patient `region_importance_{PAT}_{metric}.csv`/PNG) and the report gains a
**Brain-region permutation importance** section. Each region is read against the
**whole-brain ceiling** (`wb_imp_pic`/`wb_imp_aud` = Δacc when *all* channels are
shuffled = total accuracy the model attributes to the neural data); `frac_wb_*` is
each region's share of it. This is essential for **auditory**, whose ceiling is
small (few trials, weak-above-chance pooled decoding) — a region can hold a large
*share* while its absolute Δacc looks like noise. Runs for the four patients with a
region file (AA/AZ/LH/WBH); DR/RB have none and stay channel-only. The region
significance test uses its own `--region-null-shuffles` (default 20, independent of
`--null-shuffles`, separate rng — never changes the channel results) because the
region null is pooled over ~10 regions vs. ~90 channels. Disable with `--no-regions`.

**Auditory split caveat:** the auditory task has few trials; **AA has essentially
no repeated words** (52 words / 53 trials), so `--zero-shot-frac 0` leaves it ~1
auditory test trial and every bootstrap is skipped. Keep AA on the default
`--zero-shot-frac 0.3` (its auditory decoding is inherently zero-shot); AZ/LH/WBH
have 30–69 repeated words so `--zero-shot-frac 0` gives them larger, more stable
seen-word test sets and more auditory power.

### 3. Transfer learning (can one task's decoder be adapted to the other)

```bash
python -m main.analysis.cross_task.cross_task_transfer
python -m main.analysis.cross_task.cross_task_transfer_report
```

Runs both directions (`pic_to_aud`, `aud_to_pic`) for all 4 arms and reports
gain over the `no_transfer` within-task baseline.

### 4. Subspace geometry (how the two PLS subspaces relate)

```bash
python -m main.analysis.cross_task.cross_task_regression                 # all
python -m main.analysis.cross_task.cross_task_regression --patient AA --no-figs
```

## Notes

- Default runs (override with `--pic-run` / `--aud-run`):
  - picture: `2026-04-08_17-05-14_kernel_pls_cosine_50ep`
  - auditory: `2026-05-07_22-26-06_auditory_naming_warp-linear_align-aud_stim_onset_kernel_pls_cosine_50ep`
- Model: `Pipeline([Nystroem(rbf, n_components=100), PLSRegression(n_components=10)])` → GloVe; scored by 1-NN cosine retrieval (`word_bal_acc`, `cat_indep_bal_acc`).
- Each analysis script writes per-bootstrap CSVs + static PNGs; the matching
  `*_report.py` reads those and emits a self-contained HTML report.
