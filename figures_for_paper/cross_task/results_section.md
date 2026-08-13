_Figure package: `main/figures_for_paper/cross_task/`. Every number below is in
`source_data/group_inference.csv` unless marked otherwise; reproduce with
`compute_cross_task_data.py` (Speech env) then `cross_task_panels.py` (any env). Underlying
analyses: the **`balance=downsample`** co-training run and the ROI region-importance pass
(`analysis/cross_task/cross_task_region_importance.py`) at `scope-tpm_h10`, both under
`main/results/cross_task_cotrain/` and both pinned as `CROSS_TASK_FIGURE_*` in
`utils/config.py`._

> **⚠ Note for co-authors — the figure was rebuilt on 2026-08-13 and the cohort changed.**
> It now reports the **7** participants with at least one significant category-independent
> time bin in **both** tasks (NUE031, NUE036, NUE038, NUE041, NUE047, NUE050, NUE051),
> dropping NUE044 and NUE045 from the 9 who performed both tasks. The figure is also
> restructured: the semantic-organization MDS panel, its MDS/PCA supplements, the
> all-participant knockout supplement and the cross-task RSA supplement are retired, and the
> ROI analyses previously living only in the HTML report are promoted into it.
>
> **Region knockout was then dropped too.** The figure is **three** panels — generalization,
> normalized Jacobian sensitivity, ROI-alone accuracy — and the Results and Methods no longer
> make a knockout claim of any kind. Anything in an earlier draft citing region-knockout
> Δaccuracy, the whole-brain knockout ceiling, `frac_wb_*`, or "what breaks when a region is
> removed" is **withdrawn, not merely un-illustrated**. The panel still renders as an
> internal file (`04_region_knockout`, uncaptioned) and the columns are still in
> `panel_c_roi.csv`, so it can come back — but only together with the text.
>
> **The tested contrasts changed too, and this is the consequential one.** The figure now
> tests **within-vs-cross** and **cross-vs-pooled** — both against the transfer baseline —
> and **no longer tests within-vs-pooled**. So **the retention shortfall has no significance
> test behind it**: 81 % picture and 92 % auditory are descriptive ratios, and the sentence
> that used to call the co-training cost "statistically detectable" is withdrawn rather than
> re-sourced. It read p = 0.0156 / 0.047 while it was tested; do not quote those.
>
> What the figure now shows instead is cleaner and stronger: **both the within-task and the
> co-trained decoder beat naive transfer, in both tasks**, for category-independent accuracy
> (all four p = 0.0156). The claim is "a co-trained decoder works where transfer does not",
> not "co-training costs you something measurable".
>
> **The cohort filter cost resolution, not effect.** Picture retention moves 0.818 → 0.810
> and auditory 0.919 → 0.922. But a two-sided paired Wilcoxon at n = 7 cannot return a P
> below 2/2⁷ = **0.0156**, and every significant contrast in this figure sits on that single
> value. A starred contrast should be read as "as significant as n = 7 permits", never as an
> effect size.
>
> **BH correction changes nothing at these contrasts.** Across the 12 tests in
> `panel_b_generalization_stats.csv`, all eight starred contrasts hold (q = 0.023) and the
> four auditory word/cosine contrasts are n.s. either way. The figure and the text below
> report uncorrected P values, as previous drafts did, and both are shipped in the source
> data. (Under the previous contrast pair this was not true — auditory within-vs-pooled
> failed BH at q = 0.070 — which is worth remembering if that contrast ever comes back.)
>
> **Superseded values — do not quote from any earlier draft:** N = 9; retention 82 % / 92 %,
> 97 % / 86 %, or 99 % / 94 %; within-vs-pooled p = 0.0039 / 0.012, 0.055 / 0.016, or
> 0.098 / 0.098; picture whole-brain ceiling significant in 8/9 or 3/9; centroid alignment
> significant in 4/9 or 5/9; ROI representative NUE041 or NUE044; top-region ceiling share
> 51 % or 59 %.
>
> **Marker [AL7.1]:** the pooled training set is downsampled to equalise the two tasks;
> within-task baselines are not resampled. **[AL8.1]** unchanged — electrode-level
> importance remains replaced by ROI/region importance.
>
> **Earlier changes, retained for the record.** (2026-07-28) NUE030 joined the auditory
> cohort; a category label had been silently truncated to 10 characters, and picture/auditory
> channels had been paired *by position* rather than by electrode for three participants —
> both fixed. (2026-07-30) NUE047 joined and the picture arm moved to the 100-epoch run.
> (2026-08-12) **NUE030 was retired** (`docs/experiments/015-retiring-cp.md`).
> (2026-08-13) The figure moved from `tp`/h5 + no resampling to `tpm`/h10 + downsampling;
> the retention drop from 99 % to 82 % was isolated to the resampling change (scope and
> history at fixed balance moved picture retention by +0.002; balance at fixed scope and
> history by −0.171).

## A single decoder co-trained on both tasks decodes semantic category from either

**One co-trained decoder serves both tasks, at a measurable cost.** We pooled the trials of
both tasks, downsampling the majority task so the two contribute equally, and trained a
single kernel-PLS semantic regressor (Methods). The co-trained decoder performed above
chance on held-out trials of both tasks (category-independent balanced accuracy: picture
0.248 ± 0.015, auditory 0.252 ± 0.015; n = 7). Chance is per participant rather than a single
value — 1 / n_categories, mean 0.163 for picture and 0.172 for auditory — because NUE031 ran
an earlier auditory stimulus set with a different category inventory (Methods). Crucially, a
decoder trained on one task alone did **not** transfer to the other: cross-task decoding sat
near chance (0.193 picture, 0.198 auditory), below the within-task decoder in both tasks
(paired Wilcoxon p = 0.0156, the smallest value attainable at n = 7) **and** below the
co-trained decoder in both tasks (p = 0.0156). For picture naming the same pattern held for
word-level balanced accuracy and cosine similarity (all p = 0.0156); **for auditory naming
neither of those metrics separated any arm** (p = 0.11–0.22), so the auditory result rests on
the category-level metric alone. A single co-trained model — not reuse of a task-specific
one — is therefore what decodes semantic category from either modality (Fig. R3a). The
co-trained decoder recovered **81 % of the picture-naming ceiling and 92 % of the auditory
ceiling** (Table R1); we report these as descriptive ratios, because the within-task versus
pooled contrast is not among the comparisons tested here.

**The co-trained decoder leans hardest on ventral occipitotemporal cortex.** Single-electrode
attribution is uninformative here — under the Nystroem-RBF map information is spread
redundantly across electrodes, so dropping any one channel barely moves accuracy and no
electrode reaches BH-FDR significance. We therefore interrogated the model at the level of
brain regions (`nmm_roi`, 17 occupied regions of the 18-region `tpm` scope). Normalized
Jacobian sensitivity — per electrode, against each participant's own whole-brain
per-electrode average (Methods) — ranks **posterior fusiform highest (1.22× that participant's average
electrode), then posterior ITG (1.18×) and anterior fusiform (1.15×)**, and anterior MTG
(0.87×) and temporal pole (0.73×) lowest (Fig. R3b). This ranking is deliberately not split
by task: one co-trained model scores both tasks through a single shared map, so it ranks
regions near-identically for the two (ρ = +0.99 per electrode) whatever the anatomy is, and
that agreement is a property of the model rather than evidence of amodal coding.

**Region-alone decoding separates task-general from auditory-preferring regions.** We then
asked what each region can decode *by itself*, training the co-trained decoder on that
region's channels only (Fig. R3c). **Posterior fusiform is the strongest region in both
tasks** (0.237 picture, 0.216 auditory), making it this cohort's clearest task-general
candidate, and it is also the region the Jacobian ranks first. Against the shuffled-null
chance level, **14 of 17 regions sit above the picture reference and all 17 above the
auditory one**; the three below the picture reference — anterior STG (0.163 picture vs 0.205
auditory), angular gyrus (0.162 vs 0.216) and posterior STG (0.167 vs 0.174) — are exactly
the perisylvian regions, each decoding auditory naming but not picture naming. We report this
as a **pattern, not a test**: the reference marks where chance sits and carries no per-region
P value (see the caveats below), and angular and posterior STG are contributed by two
participants each. Anterior STG, at six participants, is the one auditory-preferring region
here with a cohort behind it.

Three features of this analysis should not be over-read. First, **the ROI-alone panel has no
size control**: this pass ran without the matched-N null, and ROI-only accuracy rises with
electrode count (within participant, median Spearman ρ = +0.42 with channel count), so the
cross-region ordering in Fig. R3c is partly an implant-coverage ordering. A
`--suff-null-draws 50` pass would supply each region with a P value against random channel
sets of its own size and is the natural next step. Second, **ROI size and ROI identity are
collinear** by implant design — depth shanks and MTG strips carry ~11 contacts while ventral
gyral ROIs carry 2–4 — so per-electrode normalization removes most but not all of the size
dependence. Third, **regions are unequally sampled**: superior parietal and precuneus come
from one participant each and angular and posterior STG from two, and although they are drawn
(scaled by contributing participants) rather than dropped, they are single- or
two-participant observations. These regions are candidate targets for a future implant, and
candidate sites of the **shared, alignable subspace**, rather than demonstrated sites of an
amodal code.

> **Claims from the previous draft that this figure no longer supports, and must be
> recomputed before use.** (1) The cross-task category-centroid alignment of the two separate
> per-task decoders (quoted as significant in 4 of 9, group mean r = 0.258): the MDS panel is
> retired and its source data is at the old N = 9 cohort. (2) The cross-task RSA between the
> per-word neural geometries. (3) The per-electrode agreement between two *independently*
> trained single-task decoders (ρ = 0.02–0.43), which needs the `_solo` columns from a
> `--single-modality` pass this arm did not run. (4) The top region's share of the
> whole-brain ceiling (51 %), which is not read by this figure.

### Methods note
For each participant, picture- and auditory-naming trials (high-gamma activity at each task's
own loose-category peak bin, on the shared channel set) were pooled into one training set
with the **majority task downsampled** so the two tasks contribute equally, and a single
kernel-PLS regressor (Nystroem-RBF, 100 landmarks → PLS, 10 components) was fit to GloVe.
Within-task baselines are not resampled. Held-out evaluation used a per-word train/test split
with a 30 % zero-shot word holdout, 50 bootstraps; within-task, cross-task and pooled
conditions share the same test sets so comparisons are paired. Group statistics are two-sided
paired Wilcoxon signed-rank across the seven participants, **uncorrected**, and only two
contrasts are tested per metric and task — **within-vs-cross** and **cross-vs-pooled**, both
against the transfer baseline. **Within-vs-pooled is not tested**, so the retention ratios in
Table R1 are descriptive. BH-corrected values over all twelve tests are shipped alongside in
the source data.

**Cohort.** The reported cohort is the 7 participants with at least one significant
category-independent time bin in **both** tasks. Significance is read from the shipped source
data of the semantic-decoding time-course figure, whose picture arm uses the `tp` scope with
500 ms of history and whose auditory arm uses `tpfm` with 1000 ms — **a different
configuration from the runs analysed here** — so the criterion means "participants whose
semantic decoding was significant in that figure", not "in these runs". The alternative rules
do not filter: at this cohort, "significant in either task" and "significant in picture alone"
both select all nine.

**Channel set.** These analyses use the 18-region `tpm` scope — the temporal-parietal
whitelist plus insula, cingulate, entorhinal, parahippocampal and precuneus, of which 17 are
occupied in this cohort — with 1000 ms (10 bins) of history. "Temporal-parietal cortex"
therefore does not describe this figure's channel set, and its region counts are not
comparable with analyses run on the 13-region `tp` scope. Regions are grouped by `nmm_roi`
(Neuromorphometrics, volumetric, native space); the Desikan-Killiany arm was computed and is
archived but is not shown, and the two are different channel sets rather than two labellings
of one.

**Region measures.** Two are reported. **Normalized Jacobian sensitivity** (Fig. R3b) is the
analytic sensitivity of the predicted embedding to the neural input, ‖∂ŷ/∂x‖, summed over a
region's channels and then normalized twice: divided by the region's electrode count, and
divided again by that participant's whole-brain per-electrode average **for the same task**,
before averaging the two tasks. Both divisions are load-bearing. As a region total the
Jacobian tracks how many contacts happened to land in an ROI, so as a cross-participant
quantity it would measure the implant rather than the brain; and the per-participant
magnitude scale (γ, ‖A‖, high-gamma amplitude) differs enough between people that unnormalized
values cannot be pooled. On the resulting scale **1.0 is that participant's own average
electrode**, which is the dashed reference in Fig. R3b; above 1 means the region is more
informative per electrode than that participant's average. Note that the reference is the
participant's *implant*, so 1.0 is implant-relative rather than brain-relative.

**ROI-alone accuracy** (Fig. R3c) is the held-out category-independent balanced accuracy of
the same co-trained decoder trained on one region's channels only. It is **not** divided by
electrode count: an accuracy has a chance floor and saturates, so a per-electrode accuracy
would rank the smallest regions highest. The RBF kernel width is pinned to the whole-brain
value for every region, so bandwidth is not a function of region size. Regions inherit the
whole-brain per-task peak bin, so a region peaking elsewhere is evaluated off-peak and
understated.

**Chance reference.** The dash-dot lines in Fig. R3c are the label-shuffled
category-independent accuracy of the whole-brain decoder, averaged within participant and
then across participants (picture 0.1671, auditory 0.1643). The ± 1 s.e.m. band around them
was drawn until 2026-08-13 and removed: it is narrower than a marker on the picture axis
(0.1669–0.1672) and invited being read as a significance interval, which it is not. The
values remain in `source_data/roi_chance_band.csv`.
NUE031 is excluded from the reference, because its earlier stimulus set
has a different category inventory and its measured auditory null is 0.199 against ~0.167 for
everyone else; it still contributes to the region means. The nulls come from whole-brain
decoders while the plotted markers are ROI-only decoders with far fewer channels, so a small
region's own null would be wider and the band is anti-conservative for small ROIs. **It marks
where chance sits; it is not a test.**

### Table R1 — retention of the within-task ceiling (`table_r1_retention.csv`)
Per-participant within-task vs pooled category-independent balanced accuracy for each task,
and their ratio (pooled ÷ within). Group means: picture **81 %**, auditory **92 %** (n = 7).
Per-participant picture retention ranges 0.75 (NUE051) to 0.90 (NUE036); auditory ranges 0.83
(NUE050) to 1.05 (NUE041, the only participant whose pooled decoder exceeds its within-task
auditory ceiling).
