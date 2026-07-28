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

# Backtick-quoted repo-relative paths: `docs/agent-context/validation.md`. The routing
# tables write paths this way rather than as markdown links, so MD_LINK alone never saw
# them -- which is how AGENTS.md routed to a data-conventions.md that did not exist, in
# four separate files, without failing this script.
INLINE_PATH = re.compile(r"`([^`\s]+/[^`\s]+)`")
INLINE_SUFFIXES = (".md", ".py", ".json")


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


_IGNORE_CACHE: dict[str, bool | None] = {}


def git_ignored_cached(path: str) -> bool | None:
    """``git_ignored`` memoised -- the link check asks about the same paths repeatedly."""
    if path not in _IGNORE_CACHE:
        _IGNORE_CACHE[path] = git_ignored(path)
    return _IGNORE_CACHE[path]


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
        text = path.read_text(encoding="utf-8")
        for match in MD_LINK.finditer(text):
            target = match.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            if not (base / target).exists() and not (REPO / target).exists():
                r.fail(f"{path.relative_to(REPO)}: broken link -> {target}")
                broken += 1
        broken += _check_inline_paths(r, path, base, text)
    if not broken:
        r.ok(f"all relative links and inline paths resolve ({len(targets)} files scanned)")


def _check_inline_paths(r: Report, path: Path, base: Path, text: str) -> int:
    """Fail on a backticked repo-relative path that does not exist. Returns the count.

    Deliberately narrow, because these files are prose and a false positive here is worse
    than a miss. A token counts as a path only if it ends in a resolvable suffix, sits
    outside a fenced code block, carries no ``{}`` template or brace expansion, and is
    rooted at a real top-level entry of the repository. That last condition is what makes
    the check precise: it validates routes like ``docs/agent-context/validation.md`` while
    ignoring prose shorthand (``helpers/_cross_patient_helpers.py``) and references to
    things outside the repo (``Speech/CLAUDE.md``), neither of which is a route.

    Gitignored targets are skipped so that intentionally machine-local references --
    ``.claude/open-questions.md``, ``.claude/settings.json`` -- do not fail a clean
    checkout that never had them.
    """
    roots = {entry.name for entry in REPO.iterdir()}
    broken = 0
    fenced = False
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        for match in INLINE_PATH.finditer(line):
            target = match.group(1)
            if not target.endswith(INLINE_SUFFIXES):
                continue
            if "{" in target or "}" in target:
                continue          # {analysis}/… template or a {a,b}.json brace expansion
            if target.split("/", 1)[0] not in roots:
                continue          # prose shorthand, or a path outside this repository
            if (base / target).exists() or (REPO / target).exists():
                continue
            if git_ignored_cached(target):
                continue          # untracked by design; absent in a clean checkout
            r.fail(f"{path.relative_to(REPO)}:{lineno}: inline path does not exist -> {target}")
            broken += 1
    return broken


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
        target = REPO / path
        if not target.exists():
            r.fail(f"{path} does not exist")
            continue
        if git_ignored_cached(path):
            r.fail(f"{path} is gitignored but must be tracked")
            continue
        r.ok(f"{path} is trackable")
        # A tracked directory is not enough: .gitignore matches on basename at any
        # depth, so an individual file inside it can still be excluded while the
        # directory itself is fine. That is not hypothetical -- the `data*` rule
        # silently untracked docs/agent-context/data-conventions.md, which existed on
        # disk and resolved every route pointing at it while being invisible to git.
        if target.is_dir():
            for child in sorted(target.rglob("*")):
                if child.is_dir() or "__pycache__" in child.parts:
                    continue
                rel = child.relative_to(REPO).as_posix()
                if git_ignored_cached(rel):
                    r.fail(
                        f"{rel} is gitignored but sits inside must-track {path}"
                        " -- add a `!` negation in .gitignore"
                    )
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
