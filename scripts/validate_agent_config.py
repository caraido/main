#!/usr/bin/env python
"""Validate the shared Claude Code + Codex context configuration. Read-only.

Checks that the two-tier layout established 2026-07-27 still holds: one canonical source
per fact, both agents able to discover it, and nothing machine-specific or secret in the
tracked tier.

Usage
-----
    python scripts/validate_agent_config.py          # exit 0 if healthy
    python scripts/validate_agent_config.py -v       # also list what passed

Exits nonzero on any FAIL. WARN never changes the exit code. Writes nothing, ever.
Stdlib only, so it runs in a bare checkout.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILLS = REPO / ".agents" / "skills"
MIRROR = REPO / ".claude" / "skills"
CONTEXT = REPO / "docs" / "agent-context"

PORTABLE_KEYS = {"name", "description"}

# Paths that must be reachable by git, and paths that must not be.
MUST_TRACK = ["AGENTS.md", "CLAUDE.md", ".agents/skills", "docs/agent-context", "scripts"]
MUST_IGNORE = ["CLAUDE.local.md", ".claude/settings.json", ".claude/skills"]

# Tracked agent files are scanned for these.
MACHINE_PATH = re.compile(r"[A-Za-z]:[\\/]{1,2}Users[\\/]|/Users/|/home/[a-z]")
SECRETISH = re.compile(
    r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|password|bearer\s+[A-Za-z0-9._-]{16,})\b"
    r"|\bsk-[A-Za-z0-9]{20,}\b"
)
# Markdown links to relative paths: [text](path). Skips URLs and anchors.
MD_LINK = re.compile(r"\[[^\]]*\]\(([^)#][^)]*)\)")


class Report:
    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
        self.failures = 0
        self.warnings = 0

    def ok(self, msg: str) -> None:
        if self.verbose:
            print(f"  ok   {msg}")

    def fail(self, msg: str) -> None:
        self.failures += 1
        print(f"  FAIL {msg}")

    def warn(self, msg: str) -> None:
        self.warnings += 1
        print(f"  warn {msg}")

    def section(self, title: str) -> None:
        print(title)


def parse_frontmatter(text: str) -> dict[str, str] | None:
    """Minimal top-level YAML mapping parse. None if there is no frontmatter block."""
    lines = text.lstrip("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    keys: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return keys
        if not line.strip() or line.startswith((" ", "\t", "#")):
            continue  # nested value or comment; only top-level keys matter here
        key, sep, value = line.partition(":")
        if sep:
            keys[key.strip()] = value.strip().strip("'\"")
    return None  # unterminated block


def git_ignored(path: str) -> bool | None:
    """True/False, or None when git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO), "check-ignore", "-q", path],
            capture_output=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    return None


def check_roots(r: Report) -> None:
    r.section("Shared root")
    agents = REPO / "AGENTS.md"
    claude = REPO / "CLAUDE.md"

    if not agents.is_file() or not agents.read_text(encoding="utf-8").strip():
        r.fail("AGENTS.md is missing or empty")
        return
    r.ok("AGENTS.md exists and is non-empty")

    if not claude.is_file():
        r.fail("CLAUDE.md is missing — Claude Code would not load AGENTS.md")
        return
    text = claude.read_text(encoding="utf-8")
    if "@AGENTS.md" in text:
        r.ok("CLAUDE.md imports AGENTS.md via @AGENTS.md")
    elif "AGENTS.md" in text:
        r.warn("CLAUDE.md mentions AGENTS.md but does not import it with @AGENTS.md")
    else:
        r.fail("CLAUDE.md does not reference AGENTS.md at all")

    body = text.split("\n", 1)[1] if "\n" in text else ""
    if len(body.splitlines()) > 60:
        r.warn("CLAUDE.md is long for an adapter — shared content belongs in AGENTS.md")


def check_skills(r: Report) -> list[Path]:
    r.section("Canonical skills (.agents/skills/)")
    if not SKILLS.is_dir():
        r.fail(".agents/skills/ does not exist")
        return []

    dirs = sorted(d for d in SKILLS.iterdir() if d.is_dir())
    if not dirs:
        r.fail(".agents/skills/ contains no skills")
        return []

    seen: dict[str, Path] = {}
    for skill in dirs:
        md = skill / "SKILL.md"
        if not md.is_file():
            r.fail(f"{skill.name}: no SKILL.md")
            continue
        front = parse_frontmatter(md.read_text(encoding="utf-8"))
        if front is None:
            r.fail(f"{skill.name}: SKILL.md has no parseable YAML frontmatter")
            continue

        missing = PORTABLE_KEYS - front.keys()
        if missing:
            r.fail(f"{skill.name}: frontmatter missing {', '.join(sorted(missing))}")
        if front.get("name") and front["name"] != skill.name:
            r.fail(f"{skill.name}: frontmatter name is '{front['name']}'")
        if front.get("name") in seen:
            r.fail(f"{skill.name}: duplicate skill name '{front['name']}'")
        elif front.get("name"):
            seen[front["name"]] = skill

        extra = set(front) - PORTABLE_KEYS
        if extra:
            r.fail(
                f"{skill.name}: non-portable frontmatter key(s) {', '.join(sorted(extra))}"
                " — move them to platform/claude.frontmatter.yaml"
            )
        else:
            r.ok(f"{skill.name}: portable frontmatter")
    return dirs


def check_mirror(r: Report) -> None:
    r.section("Claude mirror (.claude/skills/)")
    script = REPO / "scripts" / "sync_agent_skills.py"
    if not script.is_file():
        r.fail("scripts/sync_agent_skills.py is missing")
        return
    if not MIRROR.is_dir():
        r.fail(".claude/skills/ does not exist — run python scripts/sync_agent_skills.py")
        return
    result = subprocess.run(
        [sys.executable, str(script), "--check"], capture_output=True, text=True
    )
    if result.returncode == 0:
        r.ok("mirror matches canonical source")
    else:
        for line in (result.stdout + result.stderr).strip().splitlines():
            r.fail(line.strip())


def check_links(r: Report) -> None:
    r.section("Cross-references")
    targets = [REPO / "AGENTS.md", REPO / "CLAUDE.md"]
    targets += sorted(CONTEXT.glob("*.md")) if CONTEXT.is_dir() else []
    targets += sorted(SKILLS.rglob("SKILL.md")) if SKILLS.is_dir() else []

    if not CONTEXT.is_dir():
        r.fail("docs/agent-context/ does not exist")

    broken = 0
    for path in targets:
        if not path.is_file():
            continue
        base = path.parent
        for match in MD_LINK.finditer(path.read_text(encoding="utf-8")):
            target = match.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            if not (base / target).exists() and not (REPO / target).exists():
                r.fail(f"{path.relative_to(REPO)}: broken link -> {target}")
                broken += 1
    if not broken:
        r.ok(f"all relative markdown links resolve ({len(targets)} files scanned)")


def check_hygiene(r: Report) -> None:
    r.section("Tracked-tier hygiene")
    files = [REPO / "AGENTS.md", REPO / "CLAUDE.md"]
    files += sorted(CONTEXT.rglob("*.md")) if CONTEXT.is_dir() else []
    files += sorted(SKILLS.rglob("*.md")) if SKILLS.is_dir() else []

    clean = True
    for path in files:
        if not path.is_file():
            continue
        rel = path.relative_to(REPO)
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if MACHINE_PATH.search(line):
                r.fail(f"{rel}:{n}: machine-specific path — move it to CLAUDE.local.md")
                clean = False
            if SECRETISH.search(line):
                r.fail(f"{rel}:{n}: looks like a credential")
                clean = False
    if clean:
        r.ok(f"no machine paths or credentials in {len(files)} tracked agent files")


def check_split(r: Report) -> None:
    r.section("Tracked / ignored split")
    if git_ignored("AGENTS.md") is None:
        r.warn("git unavailable — skipped the tracked/ignored split check")
        return
    for path in MUST_TRACK:
        if not (REPO / path).exists():
            r.fail(f"{path} does not exist")
        elif git_ignored(path):
            r.fail(f"{path} is gitignored but must be tracked")
        else:
            r.ok(f"{path} is trackable")
    for path in MUST_IGNORE:
        if not (REPO / path).exists():
            continue
        if git_ignored(path):
            r.ok(f"{path} is ignored")
        else:
            r.fail(f"{path} is tracked but must stay machine-local")


def check_memory_authority(r: Report) -> None:
    r.section("Memory authority")
    readme = CONTEXT / "README.md"
    if not readme.is_file():
        r.fail("docs/agent-context/README.md is missing — no memory-promotion policy")
        return
    text = readme.read_text(encoding="utf-8").lower()
    if "non-authoritative" in text and "promot" in text:
        r.ok("memory-promotion policy is stated in docs/agent-context/README.md")
    else:
        r.fail("docs/agent-context/README.md does not state the memory-promotion policy")

    agents = REPO / "AGENTS.md"
    if agents.is_file() and "docs/agent-context/scientific-integrity.md" in agents.read_text(
        encoding="utf-8"
    ):
        r.ok("AGENTS.md routes to the scientific-integrity rules")
    else:
        r.fail("AGENTS.md does not route to docs/agent-context/scientific-integrity.md")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="store_true", help="list passing checks")
    args = parser.parse_args()

    print(f"validate-agent-config  {REPO}\n")
    r = Report(args.verbose)
    check_roots(r)
    check_skills(r)
    check_mirror(r)
    check_links(r)
    check_hygiene(r)
    check_split(r)
    check_memory_authority(r)

    print()
    if r.failures:
        print(f"FAILED  {r.failures} error(s), {r.warnings} warning(s)")
        return 1
    print(f"PASSED  0 errors, {r.warnings} warning(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
