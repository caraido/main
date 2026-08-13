# Caption for `00_cross_task_combined.pdf` (panels also shipped as `01_`–`03_`)

**Figure | Brain regions supporting a decoder co-trained on picture and auditory naming.**
Held-out semantic decoding in participants who performed both tasks and reached significance
in both (N = 7). High-gamma at each task's peak bin was mapped onto GloVe
embeddings by kernel partial-least-squares regression and nearest-neighbour retrieval; the
pooled decoder saw both tasks' trials, the majority
downsampled. **a** Within-task (grey), other-task (cross, red) and pooled co-trained (blue)
decoders on held-out picture (top) and auditory (bottom) trials, for category-independent and
word balanced accuracy and cosine similarity. Bar: mean ± s.e.m.; dot:
one participant; dashed line and band: mean and range of per-participant chance
(1 / n_categories). Brackets: two-sided paired Wilcoxon, uncorrected (\* p < 0.05; n.s. not
significant); both contrasts are against the cross arm; within versus pooled was not
tested. At n = 7 the smallest attainable P is 0.0156, the value of every starred contrast.
**b** Region (`nmm_roi`) normalized analytic Jacobian sensitivity of the co-trained model
(Methods). Faded dot: one participant; filled circle: mean across participants; dashed line:
that participant's average electrode; regions ordered by the mean. **c** Held-out
category-independent balanced accuracy of a decoder trained on that region's channels alone.
Circle: one region, mean across participants, area ∝ contributing participants (four have one
or two); dash-dot lines: mean across participants of the label-shuffled chance accuracy, one
per task, from the whole-brain decoder's null, excluding NUE031 (earlier stimulus set), which
still contributes to the circles; dotted line: task equality. Channel set: the
18-region `tpm` scope (Methods), 17 regions occupied, 1000 ms of history — not
temporal-parietal cortex alone. **c** has no matched-N null: ROI-only accuracy correlates
with electrode count at ρ = +0.42 within participant. Tasks are averaged in **b**: one shared
map scores both, so their rankings agree by construction.

## Notes — not part of the caption

**Cohort changed on 2026-08-13 and every number moved.** The figure was N = 9 (all
participants with both tasks); it is now the 7 with at least one significant
category-independent time bin in **both** tasks, dropping NUE044 and NUE045. Superseded
values that must not be quoted from any earlier draft: N = 9; retention 82 % / 92 %; picture
within-vs-pooled p = 0.0039 and auditory p = 0.012; whole-brain picture ceiling significant
in 3/9; MDS alignment significant in 4/9; ROI representative NUE041; top-region ceiling
share 51 %.

**The tested contrasts changed the same day.** They are now **within-vs-cross** and
**cross-vs-pooled** — both against the transfer baseline. **within-vs-pooled is no longer
tested**, so the retention ratio (81 % picture, 92 % auditory) is now a **descriptive
number with no significance test behind it**. It had one until this change (p = 0.0156
picture, 0.047 auditory); do not quote those. Any sentence that called the co-training cost
"statistically detectable" has to be reworded, not re-sourced.

**The cohort filter costs resolution, not effect.** Picture retention moves 0.818 → 0.810
and auditory 0.919 → 0.922, but the two-sided paired Wilcoxon floors at 2/2⁷ = 0.0156, so
every significant contrast in this figure sits on that one value. Read a starred contrast as
"as significant as n = 7 permits", not as an effect size.

**Significance is uncorrected, deliberately.** `panel_b_generalization_stats.csv` also ships
`q_bh` over its 12 tests. Under BH all eight starred contrasts survive (q = 0.023) and the
four auditory word/cosine contrasts remain n.s. either way, so at these contrasts correction
changes nothing — unlike the previous pair, where auditory within-vs-pooled failed BH.

**Panel c has no size control.** `suff_delta_*`, `suff_null_*` and `suff_p_*` are NaN in all
74 rows — the arm ran with `--suff-null-draws 0`. The cross-region ordering in **c** is
therefore partly an implant-coverage ordering. A `--suff-null-draws 50` pass would give each
region a p-value against random channel sets of its own size.

**Region knockout left the figure and the manuscript on 2026-08-13.** It was panel **d**.
The Results and Methods no longer make a knockout claim of any kind, so nothing in the
manuscript should cite one. `04_region_knockout.png/.pdf` is still rendered by
`cross_task_panels.py` and still carries its `d`, but it is an **internal working figure**:
deliberately uncaptioned, referenced by nothing. Its columns (`perm_imp_*`, `frac_wb_*`,
`wb_*`) are still shipped in `panel_c_roi.csv`, and `group_inference.csv` still carries the
`roi_top3_ko_*` and `wb_ceiling_*` rows, so the analysis is one command from coming back —
but restoring the panel means restoring the text with it.

Knockout numbers, retained here only so they are not lost: pFus led picture knockout at
0.0133 per electrode, ~4× the next region, and was not driven by one participant (0.0229,
0.0192, 0.0111, −0.0001 across its four; median 0.0152 above the mean). The whole-brain
ceiling was significant in 3/7 participants for picture (p = 0.005–0.199) and 0/7 for
auditory (p = 0.080–0.318), mean Δcat-indep 0.086 picture / 0.081 auditory, and region
knockout cleared BH-FDR for 0.14 regions per participant on average.

**Five of the 17 regions are outside the vendored 13-colour ROI palette** (insula, cingulate,
entorhinal, parahippocampal, precuneus) and carry report-only colours, so their colours do
not match the `electrode_labeling` brain figures; the other 12 do. Every region is labelled
in place, so colour is not load-bearing.

**Inputs** (pinned in `utils/config.py`):
- co-training — `CROSS_TASK_FIGURE_COTRAIN_RUN` = `2026-08-12_18-17-20_kernel_pls_balance-downsample_50boot`
- ROI importance — `CROSS_TASK_FIGURE_ROI_DIR` = `scope-tpm_h10/balance_downsample`, atlas `nmm`
- chance band — `figures_for_paper/semantic_regression/panels_cache_{picture,auditory}_GloVe.npz`
- upstream pair — picture
  `2026-08-11_23-42-55_picture_naming_roi-nmm_scope-tpm_h10_kernel_pls_cosine_100ep`, auditory
  `2026-08-12_09-14-11_auditory_naming_warp-stim-group_align-aud_stim_onset_roi-nmm_scope-tpm_h10_kernel_pls_cosine_100ep`

**The cohort filter reads a different configuration from this figure.** Significance comes
from `figures_for_paper/semantic_regression/source_data/source_data.csv`, whose picture arm
is `tp`/h5 and auditory arm `tpfm`/h10. It means "participants whose semantic decoding was
significant in the shipped time-course figure", not "…in this figure's runs".

**Retired 2026-08-13 — no longer shipped.** The semantic-organization MDS panel and its
S1/S2 (2D and 3D) MDS and PCA supplements, the S3 all-participant knockout supplement, the S7
cross-task RSA supplement, and the single-participant ROI bar panel. Their source data
(`panel_a_mds_points.csv`, `panel_a_mds_alignment.csv`, `panel_s7_rsa.csv`,
`category_style.csv`) is **still tracked at the previous N = 9 cohort** and is deliberately
not regenerated: it is the input to the pending co-trained latent-space visualisation
(`docs/experiments/018`). It is not a source for anything drawn in this figure.

_Display IDs (NUE###) map to internal initials in `participants.json`. Source data:
`figures_for_paper/cross_task/source_data/`; regenerate with `compute_cross_task_data.py`
then `cross_task_panels.py`._
