# `docs/agent-context/` — durable shared knowledge for coding agents

Reviewed, version-controlled context that both Claude Code and Codex may load on demand.
Everything here is repository knowledge, not session knowledge: it is expected to still be
true next month, on another machine, and to a collaborator reading the repo cold.

## Authority order

When two sources disagree, the higher one wins:

1. **`AGENTS.md`** — always loaded. Repository-wide rules, boundaries, routing.
2. **`docs/agent-context/`** (this directory) — loaded on demand via the routing table in
   `AGENTS.md`. Detail too long for the always-loaded tier.
3. **`.agents/skills/<name>/`** — procedures for a specific pipeline, loaded when the
   skill's `description` matches the task.
4. **Tracked source documentation** — `README.md`, `docs/repo_layout.md`,
   `docs/results_index.md`, `analysis/README.md`, `figures_for_paper/README.md`. These are
   authoritative for anything mechanical; the tiers above should point at them rather than
   restate them.
5. **Generated memory** — Claude Code auto memory, Codex local memories.
   **Non-authoritative.** See below.

## What belongs here

- Stable conventions an agent cannot infer from the code (identifier formats, units,
  indexing, missing-data handling).
- Verified, recurring failure modes and the rule that prevents them.
- Standards that constrain scientific claims.
- The validation hierarchy and what to report when a check cannot run.

## What does not belong here

| Instead of here | Put it in |
|---|---|
| Task state, undecided questions, "waiting on Alec" | `.claude/open-questions.md` (untracked) |
| Absolute paths, drive letters, shell quirks, machine names | `CLAUDE.local.md` (untracked) |
| A procedure for one pipeline | `.agents/skills/<name>/` |
| Anything a docstring, `README.md`, or `git log` already says | nowhere — point at it |
| Claude permissions / hooks | `.claude/settings.json` |
| Codex sandbox / model / MCP settings | `~/.codex/config.toml` |
| Credentials, participant-identifying information | nowhere in this repository |

Participant **initials** (AA, RB, …) are internal keys and appear in code and in this
directory. They must never appear in a figure or a shipped source-data table — see
`data-conventions.md`.

## Memory promotion policy

Claude Code auto memory and Codex local memories are **generated, machine-local, and
non-authoritative**. They record what one session happened to discover; they are not
reviewed, they do not travel through git, and they may be stale.

A remembered item may be promoted into shared repository context only when it is:

- repeatedly useful,
- stable,
- verified,
- relevant across sessions, tools, or machines, and
- appropriate to share with repository collaborators.

Promote it into exactly one of `AGENTS.md`, this directory, a shared skill, or — better
than any prose — a test, schema, hook, or validation script. Then **delete the memory
file**, so there is one source rather than two that can drift.

Do **not** promote: temporary task state, unverified hypotheses, secrets, credentials,
personal or participant-identifying information, completed-task history, transient paths,
or machine-specific implementation detail.

**No mandatory rule may live only in generated memory.** If an agent would be wrong without
knowing something, that something belongs in `AGENTS.md` or here.

### The two memory stores are never synchronized

`~/.claude/projects/<project-key>/memory/` and `~/.codex/memories_1.sqlite` have different
formats, different lifecycles, and different scope. Do not symlink, copy, or mirror one
into the other. They converge only by promotion into this repository.

Claude Code keys its memory directory by working directory, so opening the repo at a
different path yields a different, empty memory store. That is another reason not to let
anything important live there.

## Files

| File | Holds |
|---|---|
| `scientific-integrity.md` | Claim, statistics, exclusion, and provenance rules |
| `validation.md` | What "done" means, and what to report when a check cannot run |
| `data-conventions.md` | Identifiers, indexing, units, file naming, raw vs derived |
| `channel-and-roi-naming.md` | Channel → electrode → `primary_roi` plumbing, per patient |
| `environments.md` | Interpreter, encoding, matplotlib, OneDrive Files-On-Demand |
