---
name: paper-figure
description: Conventions and design decisions for anything under main/figures_for_paper/ — participant identity, file layout, plotting style, caption voice, and the display choices Alec has already ruled on (box+points not spaghetti, MDS not t-SNE, median not mean). Use when creating or editing a paper figure, writing a caption, or choosing how to display a per-participant comparison.
---

# Paper figures

## Purpose

Keep every manuscript figure consistent, reproducible from its own `source_data/`, and
faithful to what the data support. The mechanical rules are enforced by
`figures_for_paper/paper_common.py` and the JSON style files; this skill carries the
*decisions* — the things that have been got wrong before and corrected.

## Trigger conditions

- Creating or editing anything under `main/figures_for_paper/`.
- Writing or revising a `caption.md`.
- Choosing how to display a per-participant or cross-participant comparison.
- Building a long HTML analysis report.

## Required inputs

- The analysis name (becomes the subfolder under `figures_for_paper/`).
- Its source data, already written to `{analysis}/source_data/*.csv`.
- The `Speech` conda env for compute steps; plotting-only scripts run in any env with
  numpy/pandas/matplotlib/scipy/sklearn.

## Procedure

**Read `figures_for_paper/README.md` first** — it is the authoritative, git-tracked
statement of rules 1–5. This skill does not restate it; it adds the decisions below.

### Identity and colour

- `display_id` (`NUE###`) **only**, never internal initials, in figures *and* in shipped
  source-data CSVs. A `patient` initials column may follow `display_id` for internal
  traceability, but `display_id` is canonical and goes first.
- Import from `paper_common`; never hard-code a palette or an id map. Sources of truth:
  `participants.json` (participant → display_id, color), `cue_style.json` (cue → color,
  label), `embedding_style.json` (model/family → color, label, optional `group_color`).
- **Colour carries meaning; do not reuse a meaning-bearing colour.** Blue denotes the
  language family across the time-course and preference panels, which is why the
  model-grouped bars in the peak-comparison panel use `group_color` (purple/green) instead.

### Layout and outputs

- Save **both** `.png` and `.pdf`, every time. 200 dpi single panel, 300 dpi combined.
- Figures → `{analysis}/`, numbered by panel order (`00_*` combined/legend, `01_*`, …).
  **Every plotted CSV** → `{analysis}/source_data/`. Never scatter CSVs elsewhere,
  especially not in `notebooks/`.
- Define once near the top: `FIG_DIR = .../figures_for_paper/{analysis}` and
  `SRC_DIR = FIG_DIR/'source_data'`.
- Panel letters: bold lowercase, top-left, flowing **left→right then top→bottom**. A
  bottom-row panel labelled `d` has been called out as unacceptable.

### Display decisions (already ruled on — do not re-litigate)

- **Distribution panels = boxplot (IQR + median) + jittered per-participant coloured points
  + a black mean line.** Prefer this over per-participant "spaghetti" lines *even for paired
  data*, accepting the loss of the within-participant visual. Use one shared `_box_points`
  helper across panels.
- **Show significance explicitly**: stars (`***`/`**`/`*`, and `n.s.`) versus chance per
  group, plus a bracket with stars between the two compared groups. Not an implicit
  solid/dashed encoding.
- **2D word-neighbourhood maps use metric MDS on cosine distance, never t-SNE** (t-SNE
  distorts global distance). Plot the predicted **word at its own GloVe vector**, never the
  raw predicted embedding — PLS shrinks predictions toward the centroid, so raw predictions
  clump centrally and the true nearest word looks far away. Every plotted point should be a
  real word laid out by mutual cosine geometry.
- **Quality-gate qualitative showcases**: require the true word to be retrieved at low rank
  *and* to clear a similarity floor, so a "best case" is not a
  high-similarity-but-failed-retrieval artifact. Dedup labels, force distinct bold words,
  and use marker dots with offset labels when near-synonyms overlap (spring/fall).
- **Aggregate across participants by MEDIAN, not mean** — one participant's noisy item
  otherwise dominates. Draw the median as a prominent ring **sized by number of
  participants** (a reliability cue), with individual points faded behind. **Do not drop
  small or low-n items**; make unreliability visible instead.
- For cross-participant magnitude measures, **normalize by both size and per-participant
  scale** (e.g. ÷ n_channels *and* ÷ the participant's whole-brain per-channel average).
  One alone leaves clustering; verify de-clustering with patient-η² (want ≪ 1). Keep each
  step of the ladder as its own gallery rather than replacing the previous one.
- Sparse integer axis ticks (layers 1/4/7/10/13, not every integer). Fold a per-subplot
  title into its axis label. Keep significance rasters below and clear of the data.
- **Long HTML analysis reports must be collapsible**: a table of contents plus
  Expand/Collapse-all, native `<details>` per section and sub-block, most collapsed by
  default.

### Captions

**Read `figures_for_paper/README.md` §4 before writing or revising one** — it carries the
Nature legend convention in full (title sentence, panel-letter form, where N goes, the
350-word cap, what belongs in the notes instead), with a worked example and the Nature
Portfolio sources. Do not write a caption from memory of this paragraph; the convention
changed on 2026-08-11 and the old *N at the end* rule is retired.

The parts that decide whether a caption is acceptable at all:

- **One shipped figure, one caption**, and the caption names the file it captions.
- **Describe, never interpret.** No result, trend or comparison. Exact n, the test and its
  P values, yes.
- **Repository provenance is not caption text** — file stems, run ids, input paths and
  effect sizes go below a `## Notes — not part of the caption` heading, so the caption
  itself stays one pasteable paragraph.
- Participants by display ID only.
- Caveats a reader cannot see in the panels (channel set, integration window, a
  heterogeneous cohort, axis scales not comparable with a sibling figure) belong *in* the
  caption. Dropping one to keep the caption short is the failure this skill exists to stop.

## Decision points

| Situation | Choice |
|---|---|
| Paired per-participant data | Still box+points; the paired visual is an accepted loss |
| Embedding neighbourhood illustration | MDS on cosine, real words at their own vectors |
| An ROI/item with very few channels or participants | Keep it, annotate `n=`; do not filter |
| A vision model beats a language model on some metric | Show it honestly; do not hide it |
| Significance unreachable at this n | Say so (one-sided Wilcoxon floors at 1/2^n: 0.0039 for the n=8 auditory cohort, 1.2e-4 for n=13) |

## Validation

1. Re-run the analysis's `*_panels.py`.
2. `git diff --stat` the tracked `source_data/*.csv`. **No diff is the pass condition**;
   an unexplained diff is a regression. Rendered PDFs always differ — ignore them.
3. Confirm both `.png` and `.pdf` exist for every panel.
4. Confirm no CSV was written outside `{analysis}/source_data/`.

## Failure handling

- **`fig.supxlabel` / `fig.supylabel` do not exist** — the Speech env's matplotlib is old.
  Use `fig.text`.
- Scripts that print a `▸` glyph crash on cp1252: set
  `PYTHONIOENCODING=utf-8 PYTHONUTF8=1`.
- If a cache under `source_data/` looks stale, remember caches are git-tracked *on purpose*
  — a `*cache*` gitignore rule once let them drift out of sync with committed figures
  silently. Never reintroduce such a rule.

## Outputs

`{analysis}/{00..NN}_*.png` + `.pdf`, `{analysis}/caption.md`, and every plotted table in
`{analysis}/source_data/*.csv` with `display_id` first.

## References

- `figures_for_paper/README.md` — authoritative rules 1–5; **§4 is the caption convention**
- `figures_for_paper/paper_common.py` — `display_id`, `assign_colors`, `participant_color`,
  `load_cue_style`, `load_embedding_style`, `embedding_colors`, `apply_paper_style`
- `figures_for_paper/{participants,cue_style,embedding_style}.json` — sources of truth
- `figures_for_paper/semantic_regression/semantic_regression_panels.py` — reference
  implementation, including `perbin_significance` and the caption generator
- `figures_for_paper/semantic_regression/caption.md` (generated) and
  `S5_within_category_null_caption.md` (hand-written) — the endorsed voice. Captions in the
  other analysis folders predate the 2026-08-11 convention; do not copy their voice
