---
name: results-hygiene
description: Mandatory procedure before deleting, moving, renaming, or pruning anything under results/, figures/, or data/, and before editing .gitignore. Checks the PINNED-run ledger, respects OneDrive Files-On-Demand, and stops for explicit approval. Use whenever a request would remove or relocate generated data.
---

# Results hygiene

## Purpose

`results/` is 169 GB of expensive-to-regenerate output, and the runs that matter do not
look like the runs that matter. This procedure exists so that no deletion is ever made
from a guess about a directory name or a date.

## Trigger conditions

Before **any** of:

- Deleting, moving, renaming, or pruning under `results/`, `figures/`, or `data/`.
- Editing `.gitignore`.
- "Cleaning up", "freeing space", "removing old runs", or archiving a suite.
- Renaming the `results` tree itself.

## Required inputs

- The exact paths proposed for removal.
- The `Speech` conda env, run with cwd = `main/`.

## Procedure

1. **Regenerate the ledger** — do not trust a stale copy:
   ```
   python -m utils.audit_runs --write
   ```
   This rewrites `docs/results_index.md`, classifying every run as `PINNED`,
   `incomplete`, or `unreferenced`, and citing the exact `file:line` that pins it.

2. **Look up every proposed path** in `docs/results_index.md`. Reject immediately if it is
   marked `PINNED` or appears under "referenced in code but not present on disk".

3. **Never prune by date.** `2026-04-08_17-05-14` and `2026-06-02_17-25-11` read as stale
   April runs and are hard-coded paper-figure defaults worth ~31 GB. Age is not evidence.

4. **Grep for importers and hard-coded run ids** before removing anything that code might
   name. Failure only surfaces when a figure is *regenerated*, long after the deletion.

5. **Prefer a same-volume `mv` to a `cp` + delete.** Everything under `results/` and
   `figures/` is a OneDrive Files-On-Demand placeholder: `mv` within the volume is an
   instant metadata rename, while `cp`, a checksum, or a move outside the OneDrive root
   forces hydration and downloads the whole tree.

6. **Report and stop.** Present the exact path list, each one's ledger status, and the total
   bytes. **Wait for explicit approval before removing anything.** Regenerating any of this
   is expensive, so each deletion wants its own yes.

## Decision points

| Situation | Action |
|---|---|
| Run marked `PINNED` | Refuse. Report which `file:line` pins it |
| Run marked `unreferenced` | Eligible — still report and wait for approval |
| Run marked `incomplete` | Usually an aborted attempt retaining only `meta.json` (command line + git commit). Tiny; kept for provenance. Delete only for tidiness |
| Directory marked `derived` | **Not a run.** It is a run's own output (`figures/`, `source_data/`, `report/`). Deleting it deletes that run's report and plotted source data. Treat as part of its parent |
| Directory marked `per-patient` | **Not a run.** The pre-run-directory `<root>/<PATIENT>/` layout. For `label_generation/` it is atlas provenance, not a regenerable output |
| Path under `figures/` marked `orphan` | The `results/` twin is already gone. The safest thing in the ledger to delete — still report and wait |
| Path under `figures/` marked `twin` / `twin:PINNED` | Inherits its twin's fate. Never delete the figure tree of a PINNED run |
| Path under `figures/` not in the ledger at all | Extra care — `figures/` is gitignored and mixes genuine junk with directories paper pipelines read. See the untracked-inputs table in `docs/repo_layout.md` |
| Asked to rename `results/` | Refuse. `.gitignore` excludes `*results`; renaming would stage 169 GB |
| Asked to add a `.gitignore` pattern | Only an explicit, narrow path. Never a blanket glob |

## Validation

After any approved operation:

1. `python -m utils.audit_runs --write` again.
2. `git -C . diff docs/results_index.md` — confirm no new `incomplete` entry and no new
   "referenced in code but not present" entry appeared.
3. Report the actual reclaimed size, measured, not estimated.

## Failure handling

- If `audit_runs` cannot run, **stop**. Do not proceed on judgment alone — the ledger is
  the only authority here.
- If a path is ambiguous between two spellings, establish which is authoritative *before*
  touching either. The worked example, resolved 2026-08-10 and recorded in
  `docs/repo_layout.md`: `results/.../original_KRR_l2_50ep` and
  `figures/.../original_KSS_l2_50ep` are one run under two spellings — same 12
  participants — and `KRR` is authoritative because it is what the only consumer names.
  `KSS` occurs in no code at all. The method generalises: compare the participant sets,
  then grep for which spelling something actually reads.
- **An `unreferenced` marking is evidence, not proof, and it has been wrong.** Until
  2026-08-10 `audit_runs` could only pin timestamp-shaped names, so every directory
  without a date in its name read `unreferenced` however much code named it —
  `original_KRR_l2_50ep` (13.5 GB, read by a notebook) among them. That class of pin now
  exists, but the lesson stands: grep for the literal directory name yourself before
  deleting, especially when the name has no timestamp.

## Outputs

A regenerated `docs/results_index.md`, a written record of what was removed and why, and
the measured space reclaimed.

## References

- `utils/audit_runs.py` — the executable specification; stdlib-only
- `docs/results_index.md` — the generated ledger. Do not hand-edit
- `docs/repo_layout.md` §Results, §Two operational constraints, §Proposed pruning
- `utils/paths.py` — `results_dir()`, the only sanctioned way to compose a results path
- `.gitignore` — carries a comment explaining the `*cache*` incident; do not undo it
