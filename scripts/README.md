# `scripts/` — repository maintenance tooling

**Charter.** `scripts/` holds programs that check or regenerate *repository configuration*.
Nothing here reads participant data as an analysis, imports the analysis packages to compute a
result, or writes under `results/`. Read-only by default; anything that writes says so in the
table below and says what.

These are run by a human or an agent. **There is no CI, no `Makefile`, no pre-commit config and
no hook that invokes them** — `docs/agent-context/validation.md` is what binds them into a
workflow, and nothing fails if they are skipped. Treat that as a reason to run them, not as
permission not to.

## The boundary against the neighbouring trees

| Tree | Operates on | Invoked how |
|---|---|---|
| `scripts/` | the repository — its config, its vendored copies, its tracked/ignored split | `python scripts/<name>.py`, by hand, before committing |
| `analysis/` | the data. Writes `results/` | `python -m analysis.<topic>.<name>` |
| `tests/` | the data, at pilot stage. Writes `results/` | `python -m tests.<pilot>.run` |
| `utils/` | nothing on its own — it is imported, never run | `from utils.x import y` |

**One documented exception:** [`utils/audit_runs.py`](../utils/audit_runs.py) is a maintenance
program by charter and a library by import, and it lives in `utils/` anyway.
`python -m utils.audit_runs` is named in `AGENTS.md`, `docs/agent-context/validation.md`,
`docs/repo_layout.md` and the `results-hygiene` skill; moving it would be a documentation blast
radius for zero gain. **Do not add a `scripts/audit_runs.py` shim** — two entry points is how
drift starts.

## The scripts

| Script | Writes? | Exit code | What it is for |
|---|---|---|---|
| [`validate_agent_config.py`](validate_agent_config.py) | never | nonzero on any FAIL; WARN never changes it | The gate. Validates the two-tier agent-context layout: `AGENTS.md` non-empty; `CLAUDE.md` imports it via `@AGENTS.md`; every canonical `SKILL.md` carries portable frontmatter only; the `.claude/skills/` mirror is in sync; every markdown link **and every backticked inline repo path** resolves; no machine-specific path or credential in a tracked agent file; the `MUST_TRACK`/`MUST_IGNORE` split holds **per file**, not just per directory; and the memory-promotion policy is stated. |
| [`sync_agent_skills.py`](sync_agent_skills.py) | **yes — rewrites `.claude/skills/`** | `--check` exits 1 on drift; a plain run exits 0 | Mirrors `.agents/skills/` (canonical, tracked, read natively by Codex) into `.claude/skills/` (gitignored, Claude-only). Merges each skill's `platform/claude.frontmatter.yaml` into the mirrored `SKILL.md`, stamps a GENERATED banner, and deletes orphans. A copy rather than a symlink because `core.symlinks` is false here and reparse points misbehave inside the synced tree. |
| [`check_roi_vocabulary.py`](check_roi_vocabulary.py) | never | nonzero on any FAIL; a **missing sibling repo is a WARN** | Audits the vendored ROI vocabulary — [`utils/rois.py`](../utils/rois.py) and [`utils/roi_palette.py`](../utils/roi_palette.py) — against the sibling `electrode_labeling` repository that generated the `nmm_roi`/`dk_roi` columns. Also verifies that every ROI value actually present in `data/*/*channels.pkl` is a name the vocabulary recognises. If the copy drifts, channels are filtered and coloured by a vocabulary the data was not labelled with and **nothing fails — the counts are simply wrong.** |

### Invocation

```bash
# from main/
python scripts/validate_agent_config.py          # add -v to list what passed
python scripts/sync_agent_skills.py              # or --check to report drift only
python scripts/check_roi_vocabulary.py --sibling ../electrode_labeling
```

The sibling path is a parameter and never a literal: `validate_agent_config.py` fails any tracked
file containing a machine-specific path, so a hard-coded absolute path here would trip its own
gate. `$ELECTRODE_LABELING` and `../electrode_labeling` are the fallbacks.

### Dependencies

`validate_agent_config.py` and `sync_agent_skills.py` are **stdlib only, deliberately** — they
must run in a bare checkout before any environment exists. `check_roi_vocabulary.py` is stdlib at
import time and imports `dill` lazily for the on-disk column check only, degrading to a WARN when
it is unavailable. Keep it that way: there is no tracked dependency manifest in this repository,
so every third-party import here would be an undeclared dependency.

## Adding a script

1. It belongs here only if it operates on the repository. If it touches participant data to
   produce a result, it is a pilot (`tests/`) or a promoted analysis (`analysis/`).
2. Prefer stdlib. If you cannot, import it lazily and degrade to a WARN.
3. Read-only unless there is a reason; say what it writes in its docstring *and* in the table
   above.
4. **Add its row to this file in the same commit.** `validate_agent_config.py` walks every file
   under `scripts/` for the tracked/ignored split, and an undocumented script is one nobody will
   run.
