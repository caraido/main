# `figures_for_paper/` — conventions for publication figures

**Read this before generating or editing anything in this folder.** Every figure that
goes into the manuscript lives here, one subfolder per analysis
(`semantic_regression/`, `pls_components/`, …). These rules are enforced by
[`paper_common.py`](paper_common.py) and [`participants.json`](participants.json); follow
them so every figure is consistent and publication-ready.

## 1. Participant identity & colours — display IDs only

- **Figures and published source-data tables must identify participants by
  `display_id` (`NUE###`), never by internal initials** (AA, VB, …). Initials are the
  keys used inside data pkls / result dirs; they must not appear in anything that ships.
- The mapping is [`participants.json`](participants.json) — the **single source of
  truth**. When a new participant joins the paper, add one row there (with `display_id`
  and a distinct `color`) and nothing else changes.
- **A participant's plotting colour is fixed in `participants.json`** (`color` field) and
  reused in every figure/panel/legend — never hard-code a palette in a figure script.
- Never hard-code any of the mapping in a figure script. Import it:

  ```python
  import sys, os
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # figures_for_paper/
  from paper_common import display_id, assign_colors, load_cue_style, apply_paper_style
  label  = display_id("AA")            # -> "NUE041"; unknown initials pass through unchanged
  colors = assign_colors(patients)     # fixed colour per participant, in list order
  cues   = load_cue_style()            # cue -> {'color', 'label'} (drawing order preserved)
  ```

  From a notebook, insert the absolute path to `figures_for_paper/` on `sys.path` first
  (e.g. `sys.path.insert(0, str(FIG_DIR.parent))`).

## 2. File layout & outputs (per analysis subfolder)

- **Figures** (`.png` **and** `.pdf`) → the analysis subfolder itself, numbered by panel
  order: `00_*` = combined/legend, `01_*`, `02_*`, … Save **both** formats every time.
- **Source data** — the arrays/tables *directly plotted* on each figure, plus any reusable
  computation cache — → `{analysis}/source_data/*.csv`. Never scatter CSVs elsewhere.
- **`source_data/` is tracked in git, including the `cache_*` files.** A blanket
  `*cache*` rule in `.gitignore` used to untrack 22 of them (18 `cache_*.csv` here plus
  `semantic_regression/panels_cache_*.npz`). Because those caches *determine* what gets
  rendered but were invisible to git, they drifted out of sync with the committed figures
  with nothing showing up in `git status` — which is exactly what happened to the auditory
  arm of `semantic_regression`. Do not reintroduce such a rule; `__pycache__/` and
  `.vector_cache` are already ignored explicitly.
- Define once near the top: `FIG_DIR = .../figures_for_paper/{analysis}` and
  `SRC_DIR = FIG_DIR/'source_data'`; write PNG/PDF to `FIG_DIR`, all CSVs to `SRC_DIR`.
- Every source-data CSV that is per-participant carries a **`display_id`** column as the
  participant identifier (put it first). A `patient` initials column may be kept *after*
  it for internal traceability during development, but `display_id` is canonical.

## 3. Plotting style (Nature-style house rules)

- **Vector text stays editable:** `pdf.fonttype = 42`, `ps.fonttype = 42`,
  `svg.fonttype = 'none'`. `apply_paper_style()` sets these plus the house rcParams
  (small fonts, no top/right spines, frameless legends). Call it once at the top.
  The values themselves live in [`utils/config.py`](../utils/config.py) (`FONT_SIZE`,
  `AXES_TITLE_SIZE`, `TICK_SIZE`, `LEGEND_SIZE`, `DPI_PANEL`, `DPI_COMBINED`) and are
  re-exported by `paper_common`, so `from paper_common import DPI_PANEL` is enough —
  never retype a size or a dpi in a figure script.
- **Show the data:** plot individual-participant traces; use SEM (not SD) for
  across-participant summary bands, or plot all points with no band. State N in the caption.
- **Panel letters:** bold lowercase `a`, `b`, `c`, … at the top-left of each panel
  (`ax.set_title(letter, loc='left', fontweight='bold')`, or an axes-fraction annotation
  offset above-left of the corner so it clears long y-labels).
- **PNG dpi:** 200 for single panels, 300 for the combined figure.
- A given participant keeps the **same colour** in every panel and legend — pulled from
  `participants.json` via `assign_colors`, not a per-figure palette. Cue marker
  colours/labels come from [`cue_style.json`](cue_style.json) via `load_cue_style`.
- **Embedding-model / family colours** (for figures that contrast decoding targets, e.g.
  `language_vs_visual`) are fixed in [`embedding_style.json`](embedding_style.json) and read via
  `load_embedding_style()` / `embedding_colors()` — never hard-code them in a figure script.

## 4. Figure captions

- Each analysis writes a `caption.md` (or `00_figure_caption.md`) beside its figures.
- **One shipped figure, one caption.** An analysis folder that ships more than one figure
  writes one caption file per figure and each names the file it captions — e.g.
  `semantic_regression/` ships `caption.md` (main, picture naming), `caption_auditory.md`
  (supplementary) and `S5_within_category_null_caption.md`, and its `11_combined_both_tasks`
  is uncaptioned because it is an internal view, not a paper figure. A figure that ships
  without its own caption is how a panel ends up described by a caption written for a
  different figure.
- Captions refer to participants only by display ID.

### Style: Nature journal legend

Nature Portfolio's own instruction to authors: a legend **"should begin with a brief title
sentence for the whole figure and continue with a short description of what is shown in each
panel"**, authors should **"minimise methodological details as much as possible"** and use
**"verbal cues … not visual cues or symbols"**, each legend stays **under 350 words**, and it
must give centre values, error bars and how they were calculated, the **exact n**, the
statistical test used and its **P values**. Sources:
[Scientific Reports submission guidelines](https://www.nature.com/srep/author-instructions/submission-guidelines),
[Nature Neuroscience AIP and formatting](https://www.nature.com/neuro/submission-guidelines/aip-and-formatting).

Write a caption in this order:

1. **Bold title sentence** — `**Figure | <noun phrase>.**` A phrase, not a claim: no verb of
   result, no comparison. Leave the *number* out (`Figure |`, `Supplementary Figure |`)
   unless it is already assigned (`Supplementary Figure S5 |`); numbering happens when the
   manuscript is assembled, and a guessed number outlives the guess.
2. **What is plotted, with N in that same opening sentence** — "Held-out decoding accuracy as
   a function of time for picture naming (N = 15)." (This replaces the older *N at the end*
   rule, 2026-08-11.)
3. **Panel descriptions in order.** Bold letter, **no period after the letter**, description
   capitalised: `**a** Independent balanced category accuracy.` Consecutive panels of one
   family collapse into a range: `**b**–**d** Top-1, top-3 and top-5 word-retrieval accuracy,
   respectively.`
4. **Every visual element defined**, colon-form and terse, naming the mark in words:
   `Dashed horizontal line: mean shuffled chance across participants.`
5. **The test, its threshold, its P values** — `(one-sided permutation test against that
   bin's shuffled null, p < 0.05; pre-onset bins are not tested)`.
6. **Caveats a reader cannot see in the panels** — channel set, integration window, a
   heterogeneous cohort, axis scales that are not comparable with a sibling figure.

Two rules that decide what does *not* go in:

- **Describe, never interpret.** No result, trend or comparison ("higher than", "sustained",
  "above chance from 0.8 s") belongs in a caption. Exact n, the test and its P values do.
- **Repository provenance is not caption text.** File stems, run ids, input paths, which
  module generated the figure, effect sizes quoted in the Results — all go *below* the
  caption under a `## Notes — not part of the caption` heading. The caption itself stays one
  paragraph that can be pasted into the manuscript unedited.

Worked examples: [`semantic_regression/caption.md`](semantic_regression/caption.md)
(generated by `semantic_regression_panels.py`) and
[`semantic_regression/S5_within_category_null_caption.md`](semantic_regression/S5_within_category_null_caption.md)
(hand-written). Opening of the former:

> **Figure | Cross-patient semantic-decoding using a regression-retrieval based decoder.**
> Held-out decoding accuracy as a function of time for picture naming (N = 15). High-gamma
> activity was mapped onto GloVe word embeddings by kernel partial-least-squares regression
> and scored by nearest-neighbour retrieval. **a** Independent balanced category accuracy.
> **b**–**d** Top-1, top-3 and top-5 word-retrieval accuracy, respectively. …

**Only `semantic_regression/` has been migrated to this style** (2026-08-11). The captions
under `cross_task/`, `extendability/`, `extendability_co_trained/`, `language_vs_visual/` and
`pls_components/` still follow the older rule and have not been re-read against this section.

## 5. Statistics — significance rasters

- The significance *test* is documented at the point of use (e.g. `perbin_significance` in
  `semantic_regression/semantic_regression_panels.py`): a per-bin one-sided permutation
  test, observed mean vs. the `pctile`-th percentile of the shuffled null.
- The *cutoff* is not. It comes from `utils.config.ALPHA` (**0.05**, repo-wide), with
  `PCTILE = 100*(1-ALPHA) = 95.0` derived from it. Do not type a cutoff into a figure
  script, and do not derive one from a different alpha than the rest of the repo. Any
  in-plot "p<…" annotation must be computed from the threshold actually used, never
  hard-coded — that is what keeps a caption from outliving its own figure.
- Star ladders come from `utils.config.p_stars`, not a per-script `'***' if p < …` chain.
- **No significance is claimed before trial onset** (t < 0): pre-onset bins are masked out
  of both the raster and the source-data `significant` column.
- Significance rasters are **ordered by peak accuracy** (highest at the top).

## Files in this folder

- [`participants.json`](participants.json) — initials → `display_id`, `color`, tasks. Source of truth.
- [`cue_style.json`](cue_style.json) — task cue → `color`, `label` (drawing order). Source of truth.
- [`embedding_style.json`](embedding_style.json) — embedding families (language/vision) & models
  (GloVe/Word2Vec/DINOv3/MoCo) → `color`, `label`, optional `group_color`. Source of truth.
- [`paper_common.py`](paper_common.py) — `display_id()`, `assign_colors()`, `participant_color()`,
  `load_cue_style()`, `apply_paper_style()`, `PARTICIPANTS`, `DEFAULT_PALETTE`. Also
  re-exports the repo-wide names from `utils/config.py` (`ALPHA`, `PCTILE`, `p_stars`,
  `DPI_PANEL`, `DPI_COMBINED`, type sizes) and puts `main/` on `sys.path`, so importing it
  is enough to reach `utils.*`.
- [`../utils/config.py`](../utils/config.py) — pinned run ids (`PIC_RUN`, `AUD_RUN`),
  `ALPHA`/`PCTILE`, permutation counts, figure style. Source of truth for anything shared
  with `analysis/`; the three JSONs above remain the source of truth for figure identity.
- `{analysis}/` — one subfolder per figure: scripts/notebook, `*.png`/`*.pdf`, `caption.md`,
  `source_data/`.
