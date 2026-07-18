# `figures_for_paper/` — conventions for publication figures

**Read this before generating or editing anything in this folder.** Every figure that
goes into the manuscript lives here, one subfolder per analysis
(`semantic_regression/`, `pls_components/`, …). These rules are enforced by
[`paper_common.py`](paper_common.py) and [`participants.json`](participants.json); follow
them so every figure is consistent and publication-ready.

## 1. Participant identity & colours — display IDs only

- **Figures and published source-data tables must identify participants by
  `display_id` (`NUEx###`), never by internal initials** (AA, VB, …). Initials are the
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
  label  = display_id("AA")            # -> "NUEx041"; unknown initials pass through unchanged
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
- Style: concise, Nature-like. Terse fragments, describe-don't-interpret, **N at the end**.
  Bold panel letters (**a**, **b**, …) introduce each panel. Name the model, the target,
  the statistical test and its threshold, and what every visual element encodes.
- Captions refer to participants only by display ID.

## 5. Statistics — significance rasters

- The significance test is documented at the point of use (e.g. `perbin_significance` in
  `semantic_regression/semantic_regression_panels.py`): a per-bin one-sided permutation
  test (observed mean vs. the `pctile`-th percentile of the shuffled null; default
  `pctile=99` ≈ p<0.01). Any in-plot "p<…" annotation must be derived from the actual
  threshold, not hard-coded.
- **No significance is claimed before trial onset** (t < 0): pre-onset bins are masked out
  of both the raster and the source-data `significant` column.
- Significance rasters are **ordered by peak accuracy** (highest at the top).

## Files in this folder

- [`participants.json`](participants.json) — initials → `display_id`, `color`, tasks. Source of truth.
- [`cue_style.json`](cue_style.json) — task cue → `color`, `label` (drawing order). Source of truth.
- [`embedding_style.json`](embedding_style.json) — embedding families (language/vision) & models
  (GloVe/Word2Vec/DINOv3/MoCo) → `color`, `label`, optional `group_color`. Source of truth.
- [`paper_common.py`](paper_common.py) — `display_id()`, `assign_colors()`, `participant_color()`,
  `load_cue_style()`, `apply_paper_style()`, `PARTICIPANTS`, `DEFAULT_PALETTE`.
- `{analysis}/` — one subfolder per figure: scripts/notebook, `*.png`/`*.pdf`, `caption.md`,
  `source_data/`.
