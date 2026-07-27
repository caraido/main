# Environments

Stable environment facts that change what code does. Routed from `AGENTS.md`.

**No absolute paths here.** Machine-specific locations (interpreter paths, drive letters,
SSH aliases) belong in an untracked local file — `CLAUDE.local.md` for Claude Code — so
this document stays valid on every machine.

## Interpreter

- Conda environment **`Speech`**. Invoke portably as `conda run -n Speech python …`, or
  resolve the path once with `conda env list`.
- **Project pickles require `dill`.** A standalone Python installation cannot load them,
  which is the usual cause of an unpickling error that looks like data corruption.
- **Run from `main/`**, as `python -m analysis.<topic>.<name>`. Data paths inside the
  analysis modules are relative to `main/`.
  - Exception: `analysis/model_diagnostics/pls_components_sweep.py` has off-by-one
    `os.chdir` logic for its own location, so `python -m …` cannot find `data/`. Run with
    cwd = `main/` and call `run_learning_curve` directly.

## Console encoding

Set `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` for anything that logs box-drawing
characters or a `▸` glyph. Without them the process crashes on cp1252. Known offenders:
`analysis/embedding_sweeps/visual_layer_sweep.py`, the cross-task report generator, and
`semantic_regression.py`'s progress logging.

A driver that calls `semantic_regression.py` in-process must additionally reconfigure
`stdout` with `errors='replace'`.

## matplotlib

The `Speech` environment's matplotlib is old: **`fig.supxlabel` and `fig.supylabel` do not
exist.** Use `fig.text`. Plotting-only scripts (the `*_panels.py` layer) deliberately avoid
the heavy dependencies and run in any environment with numpy, pandas, matplotlib, scipy,
and scikit-learn.

## Memory

Participant result pkls are large — roughly two fit in RAM at once. The vision layer sweep
OOMs at 12-way and 4-way parallelism; run **2 shards × 6 participants** and pin
`OMP_NUM_THREADS`.

## OneDrive Files-On-Demand

The repository lives inside a OneDrive-synced tree, and everything under `results/` and
`figures/` is a cloud placeholder rather than a local file.

| Operation | Effect |
|---|---|
| `mv` within the same volume | Instant metadata rename. Safe. |
| `cp` | **Hydrates** the source tree — can pull down hundreds of GB. |
| Checksum / hash over the tree | **Hydrates.** |
| Move outside the OneDrive root | **Hydrates.** |

Prefer same-volume renames. Before any bulk operation on `results/` or `figures/`, use the
`results-hygiene` skill.

A recursive `find` from the repository root is extremely slow on this path. Prefer indexed
search, and exclude `results/`, `data/`, `embeddings/`, `figures/`, `.vector_cache`, and
`__pycache__` from any traversal.

Two further consequences seen on this workstation: a deleted directory can briefly refuse
removal while OneDrive holds a handle (retry), and reparse points inside the synced tree
behave inconsistently — which is why `.claude/skills/` is a **copy**, produced by
`scripts/sync_agent_skills.py`, rather than a symlink or junction.

## Shells

PowerShell and a POSIX shell are both in use. PowerShell 5.1 has no `&&`, no ternary, and
no null-coalescing operator — chain with `;` plus `if ($?)`.

`ripgrep` honours `.gitignore`, so a repository-wide search will **not** look inside
`.claude/`. Search that directory by explicit path.
