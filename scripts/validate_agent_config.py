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


# ── Output-destination policy ────────────────────────────────────────────────────
# Trees whose .py files must reach disk through utils.paths. notebooks/ is excluded
# here and gets its own rule; _archive/ is unmaintained by definition.
OUTPUT_SCAN_DIRS = ("utils", "analysis", "figures_for_paper", "report", "tests", "scripts")

#: Narrow on purpose. Only two shapes are actually *bugs* rather than style:
#:   1. a RELATIVE output root -- resolves against the working directory, which is what
#:      put one analysis suite's output outside the repository entirely;
#:   2. main/tests/results, a root the 2026-07 reorganisation deleted, so any code still
#:      naming it cannot resolve.
#: An absolute hand-composed `MAIN_DIR / "results"` is NOT flagged: most occurrences are
#: read paths, they resolve correctly, and flagging ~15 of them would bury the two shapes
#: that break. Prefer a check that is believed over one that is comprehensive.
_BAD_PATH_PATTERNS = [
    (re.compile(r"""os\.path\.join\(\s*['"](results|figures|logs)['"]"""),
     "relative output root -- resolves against the working directory"),
    (re.compile(r"""\b(base_dir|out_dir|outdir|root)\s*=\s*['"](results|figures|logs)['"]"""),
     "relative directory as a default argument"),
    (re.compile(r"""['"]tests['"]\s*,\s*['"]results['"]|\btests/results\b|\btest_results\b"""),
     "names main/tests/results, a root the 2026-07 reorganisation deleted"),
]

#: Modules that legitimately name a root because they DEFINE one, plus this file, whose
#: own pattern strings would otherwise match themselves.
_OUTPUT_EXEMPT = {
    "utils/paths.py",                    # the accessor module itself
    "utils/audit_runs.py",               # stdlib-only ledger; owns RESULTS_ROOT by design
    "scripts/validate_agent_config.py",  # contains the patterns above as literals
}

#: Violations that predate the policy, by "relpath:line". May only SHRINK. Seeding it is
#: what lets the check exist at all: a gate that goes red the day it lands is a gate
#: someone disables.
KNOWN_LEGACY: set = set()


def _iter_output_scan_files():
    for rel in OUTPUT_SCAN_DIRS:
        root = REPO / rel
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts or "_archive" in path.parts:
                continue
            yield path
    for path in sorted(REPO.glob("*.py")):
        yield path


def _code_lines(text: str):
    """Yield (lineno, line) for lines that are actually code.

    Skips ``#`` comments and triple-quoted blocks. Prose *about* a failure is not the
    failure, and this repository documents its traps at length inside docstrings -- a
    checker that cannot tell the two apart reports its own documentation as a violation.
    """
    in_block, delim = False, ""
    for n, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if in_block:
            if delim in line:
                in_block = False
            continue
        if stripped.startswith("#"):
            continue
        opened = None
        for d in ('"""', "'''"):
            if d in line:
                opened = d
                break
        if opened and line.count(opened) % 2 == 1:
            in_block, delim = True, opened
            # the text before the opening delimiter is still code
            line = line.split(opened, 1)[0]
        yield n, line


def check_output_paths(r: Report) -> None:
    """Fail on a relative or deleted output root.

    Three competing roots plus a relative fallback that escaped the repository is how one
    analysis suite ended up split across two directories, half of it outside the project.
    utils/paths.py exists to prevent that; this makes the rule enforceable rather than
    remembered.
    """
    r.section("Output destinations")
    allowlisted, new = 0, 0
    for path in _iter_output_scan_files():
        rel = path.relative_to(REPO).as_posix()
        if rel in _OUTPUT_EXEMPT:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for n, line in _code_lines(text):
            for pattern, why in _BAD_PATH_PATTERNS:
                if pattern.search(line):
                    key = f"{rel}:{n}"
                    if key in KNOWN_LEGACY:
                        allowlisted += 1
                    else:
                        new += 1
                        r.fail(f"{key}: {why}")
                    break
    if not new:
        r.ok(f"no relative or deleted output roots in code "
             f"({allowlisted} allowlisted, {len(_OUTPUT_EXEMPT)} module(s) exempt)")


def check_experiments(r: Report) -> None:
    """The tracked experiment record stays parseable, honest and short."""
    r.section("Experiment record")
    journal = REPO / "docs" / "experiments"
    if not journal.is_dir():
        r.warn("docs/experiments/ does not exist")
        return
    results_root = REPO / "results"
    entries = [p for p in sorted(journal.glob("*.md")) if p.name != "README.md"]
    if not entries:
        r.ok("no entries yet")
        return
    run_id_re = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}[A-Za-z0-9_.-]*")
    seen_ids: dict[str, Path] = {}
    for path in entries:
        rel = path.relative_to(REPO).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        front = parse_frontmatter(text)
        if front is None:
            r.fail(f"{rel}: no parseable YAML frontmatter")
            continue
        missing = {"id", "kind", "title", "status", "analysis"} - front.keys()
        if missing:
            r.fail(f"{rel}: frontmatter missing {', '.join(sorted(missing))}")
        if front.get("kind") not in (None, "experiment", "decision", "manuscript"):
            r.fail(f"{rel}: kind '{front['kind']}' is not experiment|decision|manuscript")
        status = front.get("status")
        if status not in (None, "open", "answered", "abandoned", "superseded"):
            r.fail(f"{rel}: status '{status}' is not open|answered|abandoned|superseded")
        if status == "answered" and not front.get("answer"):
            r.fail(f"{rel}: status is answered but `answer:` is empty")
        if front.get("id") in seen_ids:
            r.fail(f"{rel}: id {front['id']} already used by "
                   f"{seen_ids[front['id']].name}")
        elif front.get("id"):
            seen_ids[front["id"]] = path

        # The line cap is the mechanical form of "this is not a log". An entry that
        # needs more than this is more than one question.
        n_lines = len(text.splitlines())
        if n_lines > 120:
            r.fail(f"{rel}: {n_lines} lines; an entry over 120 is more than one question")
        elif n_lines > 80:
            r.warn(f"{rel}: {n_lines} lines; entries are meant to stay under 80")

        # Every run id cited must exist, unless the entry says the work was abandoned.
        # PREFIX match, not equality: entries routinely cite an id elided for width
        # ("2026-08-09_09-04-16…"), and the regex stops at the ellipsis. Comparing that
        # fragment for equality reported four runs as missing that are on disk and PINNED.
        if status != "abandoned" and results_root.is_dir():
            on_disk = [d.name for a in results_root.iterdir() if a.is_dir()
                       for d in a.iterdir() if d.is_dir()]
            for run_id in sorted(set(run_id_re.findall(text))):
                if not any(name.startswith(run_id) for name in on_disk):
                    r.warn(f"{rel}: cites {run_id}, which is not under results/ "
                           f"(deleted, or not on this machine)")
    r.ok(f"{len(entries)} entry(ies) parse and satisfy the schema")


def check_scripts_documented(r: Report) -> None:
    """Every program in scripts/ appears in scripts/README.md."""
    r.section("scripts/ documentation")
    readme = REPO / "scripts" / "README.md"
    if not readme.is_file():
        r.fail("scripts/README.md is missing -- scripts/ has no charter")
        return
    text = readme.read_text(encoding="utf-8", errors="replace")
    undocumented = [p.name for p in sorted((REPO / "scripts").glob("*.py"))
                    if p.name not in text]
    if undocumented:
        for name in undocumented:
            r.fail(f"scripts/{name} is not documented in scripts/README.md")
    else:
        r.ok("every script is documented")


#: Modules that exist to be imported. A shared helper with no callers is not shared.
SHARED_MODULES = ["utils/cli.py", "utils/run_context.py", "utils/roi_scopes.py",
                  "utils/paths.py", "figures_for_paper/paper_common.py"]

#: Shared modules known to have no callers. May only SHRINK -- each entry is a standing
#: adopt-or-delete decision, not a permanent exemption.
KNOWN_UNADOPTED = {
    # Built in the 2026-05 cleanup as a factory for a 31-script migration that never
    # happened, and documented as live at README.md:57,336. Adopt it in the new report
    # CLIs or delete it; leaving it is the third option that has been taken for a year.
    "utils/cli.py",
}


def check_shared_modules_adopted(r: Report) -> None:
    """A shared module with zero importers outside its own package is dead on arrival.

    utils/cli.py is why this exists: 7.5 KB of shared argparse builders, written to be
    adopted, never imported once, and still described as live in README.md. The failure
    mode is silent, so it needs a check rather than a habit.
    """
    r.section("Shared-module adoption")
    haystacks = []
    for rel in (*OUTPUT_SCAN_DIRS, "notebooks"):
        root = REPO / rel
        if root.is_dir():
            haystacks += [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]
            haystacks += list(root.rglob("*.ipynb"))
    haystacks += list(REPO.glob("*.py"))

    for rel in SHARED_MODULES:
        path = REPO / rel
        if not path.is_file():
            r.warn(f"{rel} is listed as shared but does not exist")
            continue
        module = path.stem
        # Match the module name only inside an actual import statement. An earlier
        # version searched for the bare word anywhere, with a lookbehind that excluded a
        # preceding dot -- which both counted prose ("cli") as an importer and missed
        # every real `from utils.run_context import ...`. It reported 1 importer for a
        # module with none and 1 for a module with four.
        pattern = re.compile(
            rf"^\s*(?:from\s+[\w.]*\b{re.escape(module)}\b|"
            rf"from\s+[\w.]+\s+import\s+[^\n]*\b{re.escape(module)}\b|"
            rf"import\s+[\w.]*\b{re.escape(module)}\b)",
            re.MULTILINE)
        importers = 0
        for h in haystacks:
            if h == path:
                continue
            try:
                text = h.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if pattern.search(text):
                importers += 1
        if importers == 0:
            if rel in KNOWN_UNADOPTED:
                r.warn(f"{rel}: no importers (known; adopt-or-delete decision pending)")
            else:
                r.fail(f"{rel}: no importers -- a shared module nobody calls is not shared")
        else:
            r.ok(f"{rel}: {importers} importer(s)")


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
    check_output_paths(r)
    check_experiments(r)
    check_scripts_documented(r)
    check_shared_modules_adopted(r)

    print()
    if r.failures:
        print(f"FAILED  {r.failures} error(s), {r.warnings} warning(s)")
        return 1
    print(f"PASSED  0 errors, {r.warnings} warning(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
