@AGENTS.md

# Claude Code–specific notes

`AGENTS.md` above is the authoritative, shared instruction file — it is read by Codex too.
Do not duplicate any of it here; add a rule there, not in this file.

- **Shared skills are canonical in `.agents/skills/`.** `.claude/skills/` is **generated**
  by `python scripts/sync_agent_skills.py`. Never edit `.claude/skills/` directly — the
  change is lost on the next sync, and `scripts/validate_agent_config.py` fails on drift.
  Claude-only frontmatter (e.g. `user-invocable`) lives in a skill's
  `platform/claude.frontmatter.yaml` and is merged in at sync time.
- **Claude Code auto memory is generated, machine-local, and non-authoritative.** Do not
  move a mandatory repository rule into it. Promotion rules:
  `docs/agent-context/README.md`.
- Machine-specific paths (conda executable, drive letters) belong in `CLAUDE.local.md`,
  which is untracked. Never put one in `AGENTS.md` or `docs/agent-context/`.
- Permissions and additional directories: `.claude/settings.json` (untracked).
- Task state awaiting Alec's decision: `.claude/open-questions.md` (untracked). Keep it out
  of `AGENTS.md`, which carries only stable rules.
