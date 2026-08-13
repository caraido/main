# Figure caption — language versus visual structure in decoded picture naming

The paragraph below is the caption as it should appear in the manuscript — copy it whole.
Everything under "Notes" is provenance for this repository and is not part of the caption.
Hand-written; keep it in the Nature legend style recorded in `figures_for_paper/README.md` §4.

**Figure | Decoding of picture-naming high-gamma activity onto language and vision embedding
families.** Held-out decoding accuracy for picture naming (N = 14). High-gamma activity was
regressed by kernel-PLS onto two embedding families blind to one another by construction: a
language family (GloVe, Word2Vec; trained only on lexical co-occurrence) and a vision family
(DINOv3, MoCo; trained only on images, self-supervised).
**a** Procrustes similarity between the four embedding spaces on the shared stimulus set
(1 − Procrustes disparity of the top-10-PC concept geometries; higher = more similar); black
lines separate the language and vision blocks. **b** Variance explained (R² −
shuffled chance) for each family; line = mean across participants, shaded band = ± s.e.m.
Black horizontal bar below zero: time bins where the language family exceeds the vision family
(per-bin linear mixed model of the language−vision difference, participant random
intercept, Benjamini–Hochberg FDR q < 0.05 across bins; pre-onset bins are not tested and
never marked). Dotted vertical line: picture onset (0 s). Vertical coloured lines and shaded areas: cue times (go cue, voice onset, voice
offset), mean ± 1 s.d. across participants. **c** Pairwise difference Δ = language − vision at the
semantic peak bin (group category-accuracy peak, ~1.1 s), for R² (top), category (middle) and
word (bottom) decoding; bar = group mean, error bar = s.e.m., grey dots = participants,
jittered horizontally; stars = one-sided Wilcoxon signed-rank (language >
vision), BH-FDR-corrected over the 12 pair × metric tests (\*\*\* q<0.001, \*\* q<0.01,
\* q<0.05, n.s. otherwise). **d** Δ = language − vision in post-stimulus mean accuracy per
participant, ranked, for category (left) and word (right); blue = favours language, red =
favours vision. **e** Category (left) and word (right) accuracy of DINOv3 and MoCo by layer
depth (1-indexed), mean ± s.e.m.; dashed line and band = pooled peak accuracy ± s.e.m. of the
language decoder. Channels: the 13-region temporal-parietal whitelist on `nmm_roi` (633 of
1,360 contacts); feature window 500 ms. Participants are identified by display ID. N = 14,
except the last four bins (4.1–4.4 s) of **b**, where N = 13 — one participant's recording
ends at 4.0 s.

## Notes — not part of the caption

- Figure: `00_combined.{png,pdf}`, rendered by `language_vs_visual_panels.py` from
  `source_data/*.csv` only (no PKLs). Per-panel standalones: `01_procrustes_matrix` (**a**),
  `02_r2_timecourse` (**b**), `04_peak_model_comparison` (**c**), `05_preference_delta`
  (**d**), `06_layer_sweep` (**e**).
- Plotted values: `panel_a_procrustes_matrix.csv`, `panel_c_r2_timecourse.csv`,
  `panel_d_peak_pairwise.csv` + `panel_d_peak_pairwise_stats.csv`,
  `panel_e_preference_delta.csv`, `panel_f_layer_sweep.csv` +
  `panel_f_language_reference.csv`. Per-bin participant counts are the `count` column of
  `panel_c_r2_timecourse.csv`.
- **The category-effect timecourse left this figure on 2026-08-11** and the panels were
  relettered (old d/e/f → new c/d/e). It still renders as `03_category_timecourse.{png,pdf}`
  from `panel_b_category_timecourse.csv`, deliberately without a panel letter; give it its own
  caption before using it anywhere.
- Panel *functions* in the renderer keep their historical names — `panel_d` draws **c**,
  `panel_e` draws **d**, `panel_f` draws **e**. A function name is not a claim about a letter.
- Numbers are not comparable with the pre-2026-08 whole-brain, 1000 ms version of this figure.
- `S1_preference_delta_per_participant.{png,pdf}` is a supplement and has no caption yet.
- `results_section.md` beside this file was rewritten 2026-08-12 against the current
  source data and is N = 14. Two claims WEAKENED in that rewrite and are called out
  there: the per-bin category contrast is significant across 4 bins (was 20 at N = 12,
  whole-brain/1000 ms), and GloVe > MoCo in category accuracy is no longer significant
  (q = 0.19). The R² contrast is unaffected.
