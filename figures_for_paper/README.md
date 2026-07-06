# `figures_for_paper/` — conventions for publication figures

**Read this before generating or editing anything in this folder.** Every figure that
goes into the manuscript lives here, one subfolder per analysis
(`semantic_regression/`, `pls_components/`, …). These rules are enforced by
[`paper_common.py`](paper_common.py) and [`participants.json`](participants.json); follow
them so every figure is consistent and publication-ready.

## 1. Participant identity — display IDs only

- **Figures and published source-data tables must identify participants by
  `display_id` (`NUEx###`), never by internal initials** (AA, VB, …). Initials are the
  keys used inside data pkls / result dirs; they must not appear in anything that ships.
- The mapping is [`participants.json`](participants.json) — the **single source of
  truth**. When a new participant joins the paper, add one row there and nothing else
  changes.
- Never hard-code the mapping in a figure script. Import it:

  ```python
  import sys, os
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # figures_for_paper/
  from paper_common import display_id, apply_paper_style
  label = display_id("AA")        # -> "NUEx041"; unknown initials pass through unchanged
  ```

  From a notebook, insert the absolute path to `figures_for_paper/` on `sys.path` first
  (e.g. `sys.path.insert(0, str(FIG_DIR.parent))`).

## 2. File layout & outputs (per analysis subfolder)

- **Figures** (`.png` **and** `.pdf`) → the analysis subfolder itself, numbered by panel
  order: `00_*` = combined/legend, `01_*`, `02_*`, … Save **both** formats every time.
- **Source data** — the arrays/tables *directly plotted* on each figure, plus any reusable
  computation cache — → `{analysis}/source_data/*.csv`. Never scatter CSVs elsewhere.
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
- Keep a consistent, colour-blind-aware palette across panels; a given participant keeps
  the **same colour** in every panel and in the legend.

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

- [`participants.json`](participants.json) — initials → `display_id` (+ tasks). Source of truth.
- [`paper_common.py`](paper_common.py) — `display_id()`, `apply_paper_style()`, `PARTICIPANTS`.
- `{analysis}/` — one subfolder per figure: scripts/notebook, `*.png`/`*.pdf`, `caption.md`,
  `source_data/`.
