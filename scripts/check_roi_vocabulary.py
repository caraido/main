#!/usr/bin/env python
"""Check the vendored ROI vocabulary against the electrode_labeling repo. Read-only.

``utils/rois.py`` and ``utils/roi_palette.py`` are copies of files that live in the
sibling ``electrode_labeling`` repository -- which is also what *generated* the
``nmm_roi`` and ``dk_roi`` columns in ``data/{PAT}/{PAT}_*channels.pkl``.  If the copy
drifts from the original, this repository will filter and colour channels by a vocabulary
the data was not labelled with, and nothing will fail: the counts will simply be wrong.

Usage
-----
    python scripts/check_roi_vocabulary.py --sibling ../electrode_labeling
    python scripts/check_roi_vocabulary.py                  # uses $ELECTRODE_LABELING
    python scripts/check_roi_vocabulary.py -v               # list what passed

Exits nonzero on any FAIL.  A missing sibling is a WARN, not a FAIL: collaborators and CI
will not have that repository checked out, and this must not block them.  Writes nothing.

Why a checker and not a copier: ``scripts/sync_agent_skills.py`` copies *within* this
repository, where the source is tracked and always present.  This crosses a repository
boundary that only exists on a machine with both checkouts, so the vendored file has to
stand alone and be *audited* rather than regenerated on demand.

The sibling path is a parameter, never a literal, because
``scripts/validate_agent_config.py`` fails any tracked file containing a machine-specific
path -- a hard-coded ``d:\\...`` here would trip that check.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


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


def resolve_sibling(explicit: str | None) -> Path | None:
    """The electrode_labeling checkout, or None if it cannot be located.

    An explicit --sibling that does not resolve raises rather than falling through to the
    default: silently checking a different repository than the one named is how a green
    result comes to mean nothing.
    """
    if explicit:
        path = Path(explicit).expanduser()
        if not (path / "electrode_labeling" / "roi.py").is_file():
            raise SystemExit(
                f"--sibling {explicit!r} does not look like an electrode_labeling "
                f"checkout (no electrode_labeling/roi.py under it). Refusing to fall "
                f"back to a different repository.")
        return path
    for candidate in (os.environ.get("ELECTRODE_LABELING"),
                      str(REPO.parent / "electrode_labeling")):
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if (path / "electrode_labeling" / "roi.py").is_file():
            return path
    return None


def load_module(path: Path, name: str):
    """Import a single .py file without importing its package.

    ``electrode_labeling/roi.py`` imports nothing but ``typing``, so it loads standalone --
    which matters, because importing the package proper would pull in nibabel, scipy and
    the imaging share this repository deliberately does not depend on.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_from_package(sibling: Path, name: str):
    """Import ``electrode_labeling.<name>`` with *sibling* temporarily on the path.

    Needed for the modules that use relative imports.  Returns None rather than raising:
    a module needing nibabel or the imaging share is a reason to skip a check, not to fail
    the run on a machine that does not have them.
    """
    sys.path.insert(0, str(sibling))
    try:
        pkg = __import__(f"electrode_labeling.{name}", fromlist=[name])
        return pkg
    except Exception:                              # noqa: BLE001
        return None
    finally:
        sys.path.remove(str(sibling))


def check_vocabulary(r: Report, sibling: Path) -> None:
    r.section("ROI vocabulary")
    from utils import rois as ours

    theirs = load_module(sibling / "electrode_labeling" / "roi.py", "_el_roi")

    if tuple(ours.IN_ANALYSIS) != tuple(theirs.IN_ANALYSIS):
        r.fail(f"IN_ANALYSIS differs\n"
               f"        ours   ({len(ours.IN_ANALYSIS)}): {list(ours.IN_ANALYSIS)}\n"
               f"        theirs ({len(theirs.IN_ANALYSIS)}): {list(theirs.IN_ANALYSIS)}")
    else:
        r.ok(f"IN_ANALYSIS identical ({len(ours.IN_ANALYSIS)} regions, same order)")

    for attr in ("NMM_TO_ROI", "DK_TO_ROI"):
        a, b = getattr(ours, attr), getattr(theirs, attr)
        if a != b:
            only_ours = {k: v for k, v in a.items() if b.get(k) != v}
            only_theirs = {k: v for k, v in b.items() if a.get(k) != v}
            r.fail(f"{attr} differs; ours-only={only_ours} theirs-only={only_theirs}")
        else:
            r.ok(f"{attr} identical ({len(a)} labels)")

    # The full table, not just the whitelist: an ROI moving between `reason` groups changes
    # what a figure says about WHY a region was set aside, which is a methods claim.
    ours_table = {x.name: (x.family, x.order, x.reason) for x in ours._TABLE}
    theirs_table = {x.name: (x.family, x.order, x.reason) for x in theirs._TABLE}
    if ours_table != theirs_table:
        diff = {k for k in set(ours_table) | set(theirs_table)
                if ours_table.get(k) != theirs_table.get(k)}
        r.fail(f"ROI table differs for: {sorted(diff)}")
    else:
        r.ok(f"full ROI table identical ({len(ours_table)} regions incl. excluded)")


def check_excluded_contacts(r: Report, sibling: Path) -> None:
    r.section("Excluded contacts")
    from utils import rois as ours

    # Unlike roi.py, config.py uses a relative import, so it has to be loaded as part of
    # its package rather than standalone.
    theirs = import_from_package(sibling, "config")
    if theirs is None:
        r.warn("could not import the sibling's config.py; EXCLUDED_CONTACTS unverified")
        return
    a = {k: tuple(v) for k, v in ours.EXCLUDED_CONTACTS.items()}
    b = {k: tuple(v) for k, v in theirs.EXCLUDED_CONTACTS.items()}
    if a != b:
        r.fail(f"EXCLUDED_CONTACTS differs; ours={a} theirs={b}")
    else:
        r.ok(f"EXCLUDED_CONTACTS identical ({sum(len(v) for v in a.values())} contacts, "
             f"{len(a)} patients)")


def check_palette(r: Report, sibling: Path) -> None:
    r.section("Palette")
    from utils import roi_palette as ours

    sys.path.insert(0, str(sibling))
    try:
        from electrode_labeling import palette as theirs  # noqa: WPS433
        resolved = dict(theirs.region_colors())
    except Exception as exc:                       # noqa: BLE001
        r.warn(f"could not resolve the sibling palette ({type(exc).__name__}: {exc}); "
               f"hues unverified")
        return
    finally:
        sys.path.remove(str(sibling))

    if dict(ours.REGION_COLORS) != resolved:
        diff = {k for k in set(ours.REGION_COLORS) | set(resolved)
                if ours.REGION_COLORS.get(k) != resolved.get(k)}
        r.fail("REGION_COLORS differs for: "
               + ", ".join(f"{k} ours={ours.REGION_COLORS.get(k)} "
                           f"theirs={resolved.get(k)}" for k in sorted(diff)))
    else:
        r.ok(f"REGION_COLORS identical ({len(resolved)} entries)")

    flat = [region for members in ours.FAMILIES.values() for region in members]
    missing = [x for x in ours.IN_ANALYSIS if x not in flat] if hasattr(ours, "IN_ANALYSIS") \
        else [x for x in __import__("utils.rois", fromlist=["IN_ANALYSIS"]).IN_ANALYSIS
              if x not in flat]
    if missing:
        r.fail(f"in-analysis regions with no palette family: {missing}")
    else:
        r.ok("every in-analysis region has a palette family")


def check_columns_match_vocabulary(r: Report) -> None:
    """The data must actually speak the vocabulary we filter it with.

    This is the check that catches the failure the other three cannot: the vendored copy
    and the sibling can agree perfectly while the pkls on disk were written by an OLDER
    version of the sibling.  Every ROI value in the columns must be a name the vocabulary
    recognises -- as an in-analysis region, an excluded region, or the two sentinels.
    """
    r.section("Atlas columns on disk")
    try:
        import dill
    except ImportError:
        r.warn("dill not importable; skipping (needs the Speech env)")
        return
    from utils.rois import ATLAS_COLUMN, BY_NAME, UNASSIGNED, RIGHT_PREFIX

    data = REPO / "data"
    files = sorted(data.glob("*/*channels.pkl"))
    if not files:
        r.warn(f"no atlas pkls under {data}; skipping")
        return

    known = set(BY_NAME) | {UNASSIGNED, "white matter", ""}
    unknown: dict = {}
    n_rows = 0
    for f in files:
        with open(f, "rb") as fh:
            df = dill.load(fh)
        n_rows += len(df)
        for atlas, column in ATLAS_COLUMN.items():
            if column not in df.columns:
                r.fail(f"{f.name}: no {column!r} column")
                continue
            for value in df[column].astype(str).unique():
                bare = value[len(RIGHT_PREFIX):] if value.startswith(RIGHT_PREFIX) else value
                if bare not in known:
                    unknown.setdefault(f"{atlas}:{value}", set()).add(f.parent.name)
    if unknown:
        # A label the vocabulary has never seen is not necessarily an error -- a new
        # participant can carry a parcel nobody has hit before -- but it is silently
        # dropped by the whitelist, so it must be visible.
        for label, pats in sorted(unknown.items()):
            r.warn(f"label not in the ROI table: {label} ({', '.join(sorted(pats))})")
    else:
        r.ok(f"every value in both columns is a known region "
             f"({len(files)} files, {n_rows} rows)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sibling", default=None,
                       help="path to the electrode_labeling checkout "
                            "(default: $ELECTRODE_LABELING, then ../electrode_labeling)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="list passing checks")
    args = parser.parse_args()

    print(f"check-roi-vocabulary  {REPO}\n")
    r = Report(args.verbose)

    sibling = resolve_sibling(args.sibling)
    if sibling is None:
        r.section("Sibling repository")
        r.warn("electrode_labeling not found -- vocabulary drift NOT checked. "
               "Pass --sibling or set $ELECTRODE_LABELING.")
    else:
        print(f"sibling: {sibling}\n")
        check_vocabulary(r, sibling)
        check_excluded_contacts(r, sibling)
        check_palette(r, sibling)

    check_columns_match_vocabulary(r)

    print()
    if r.failures:
        print(f"FAILED  {r.failures} error(s), {r.warnings} warning(s)")
        return 1
    print(f"PASSED  0 errors, {r.warnings} warning(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
