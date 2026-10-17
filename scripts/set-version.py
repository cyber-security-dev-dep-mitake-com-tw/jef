#!/usr/bin/env python3
"""Set one version across every package in the monorepo.

Versions are locked in lockstep: `jef`, `jef-server`, `jef-train` and both npm
packages all carry the same number, and inter-package dependencies pin to it
exactly.

That pinning is the point. Without it `jef-server` would accept any `jef` on
PyPI, including a future incompatible one -- a user could end up with
`jef-server==0.2.0` against `jef==0.9.0` and get an import error at startup
rather than a resolver error at install time.

    python scripts/set-version.py 0.2.0
    python scripts/set-version.py --check 0.2.0        # verify without writing
    python scripts/set-version.py --canonical v0.2.0-rc.4   # -> 0.2.0rc4
    python scripts/set-version.py --next      # next unused version, from the registries

Any reasonable spelling of a version is accepted on input (`v0.2.0rc.4`,
`0.2.0-rc.4`, `0.2.0-alpha.1`) and normalised to PEP 440 before anything is
compared or written. The release workflow uses `--canonical` so a tag typed
the semver way does not fail the release at the first step.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Python distributions, in dependency order.
PY_PACKAGES = [
    "jef",
    "jef-server",
    "jef-train",
]

#: Distribution names that siblings may depend on, mapped from directory name.
DIST_NAMES = {
    "jef": "jef",
    "jef-server": "jef-server",
    "jef-train": "jef-train",
}

NPM_PACKAGES = [
    Path("packages/jef-sdk-ts/package.json"),
    Path("connectors/n8n/package.json"),
]

#: Source files that carry the version as a constant, with the package.json
#: they must agree with. `@jef-ai/sdk` exported `VERSION = '0.1.0'` for two
#: minor versions after package.json said otherwise; nothing checked it. Now
#: this script writes it and `--check` fails when it drifts. Python packages
#: need no entry: their `__version__` reads the installed distribution.
TS_VERSION_FILES: dict[Path, Path] = {
    Path("packages/jef-sdk-ts/src/version.ts"): Path("packages/jef-sdk-ts/package.json"),
}

_TS_VERSION_RE = re.compile(r"(?m)^export const VERSION = '([^']*)';$")


def _ts_version_source(version: str) -> str:
    return (
        "// Written by scripts/set-version.py from package.json. Do not edit by hand:\n"
        "// `set-version.py --check` fails CI when this and package.json disagree.\n"
        f"export const VERSION = '{version}';\n"
    )


#: PEP 440 subset: what a release tag may look like.
VERSION_RE = re.compile(r"^(\d+\.\d+\.\d+)(?:(a|b|rc)(\d+))?$")

#: Every spelling of a prerelease a human has typed into a tag, mapped to the
#: canonical PEP 440 marker. `c`, `pre` and `preview` are PEP 440's own
#: synonyms for `rc`; `alpha` and `beta` are what npm and semver call `a`/`b`.
_PRERELEASE_LABELS = {
    "a": "a",
    "alpha": "a",
    "b": "b",
    "beta": "b",
    "rc": "rc",
    "c": "rc",
    "pre": "rc",
    "preview": "rc",
}

#: Loose form: optional `v`, the base, then an optional prerelease with any of
#: the separators people reach for -- `0.2.0rc4`, `0.2.0rc.4`, `0.2.0-rc.4`,
#: `0.2.0-rc4`, `v0.2.0_rc4`, `0.2.0-alpha.1` are all one version.
_LOOSE_RE = re.compile(
    r"^[vV]?(?P<base>\d+\.\d+\.\d+)"
    r"(?:[-._]?(?P<label>alpha|beta|preview|pre|rc|a|b|c)[-._]?(?P<number>\d+))?$",
    re.IGNORECASE,
)


def canonical(raw: str) -> str:
    """Normalise a tag or user-typed version to the PEP 440 spelling.

    The release workflow runs on `v*` tags, and `v0.2.0rc.4` is a perfectly
    natural thing to type -- it is how semver spells a prerelease and how the
    npm packages carry it. It is not PEP 440, so a strict check against the
    pyproject files failed the whole release at the first step (Release #10),
    even though the files were at `0.2.0rc4` and every package would have
    published correctly. Canonicalising first means the tag only has to name
    the right version, not spell it the Python way.

    Raises ``ValueError`` for anything that is not a release version at all.
    Post-releases, dev releases and epochs are rejected on purpose: this repo
    does not publish them, and silently accepting one would get a version onto
    PyPI that nothing else in the pipeline expects.
    """
    text = raw.strip()
    match = _LOOSE_RE.match(text)
    if not match:
        raise ValueError(f"'{text}' is not a release version (expected 1.2.3 or 1.2.3rc1)")
    base, label, number = match.group("base", "label", "number")
    if label is None:
        return base
    return f"{base}{_PRERELEASE_LABELS[label.lower()]}{int(number)}"


#: PEP 440 prerelease markers to the words npm expects.
_NPM_LABELS = {"a": "alpha", "b": "beta", "rc": "rc"}


def npm_version(version: str) -> str:
    """Translate a PEP 440 version into npm's strict semver.

    npm rejects `0.2.0rc1` under strict semver, but its *loose* parser reads it
    as `0.2.0-rc1` -- and the publish path normalises. So a prerelease would
    land on the registry under a version that matches neither the git tag nor
    the pyproject files, and nothing here would notice: `--check` reads the
    files, which still hold the string we wrote.

    The dot in `-rc.1` is not cosmetic. Semver compares dot-separated numeric
    identifiers numerically, so `rc.9 < rc.10`; without the dot they are one
    alphanumeric identifier compared as a string, and `rc10 < rc9`.
    """
    match = VERSION_RE.match(version)
    if not match:
        raise ValueError(f"not a release version: {version}")
    base, kind, number = match.groups()
    return base if kind is None else f"{base}-{_NPM_LABELS[kind]}.{number}"


#: Prerelease markers in release order; a final release sorts after all of them.
_PRERELEASE_ORDER = {"a": 0, "b": 1, "rc": 2}


def version_key(version: str) -> tuple[int, int, int, int, int]:
    """Sort key for a canonical version: `0.2.0a1 < 0.2.0rc4 < 0.2.0 < 0.2.1rc1`."""
    match = VERSION_RE.match(version)
    if not match:
        raise ValueError(f"not a release version: {version}")
    base, kind, number = match.groups()
    major, minor, patch = (int(part) for part in base.split("."))
    if kind is None:
        return (major, minor, patch, len(_PRERELEASE_ORDER), 0)
    return (major, minor, patch, _PRERELEASE_ORDER[kind], int(number))


def bump(version: str) -> str:
    """The version after this one, staying on the same track.

    A prerelease advances its own number (`0.2.0rc4 -> 0.2.0rc5`), a final
    release its patch (`0.2.0 -> 0.2.1`). Leaving the rc track -- cutting
    `0.2.0` from `0.2.0rc5`, or starting `0.3.0` -- is a decision, not an
    increment, so it is made by setting the version in the files or by
    tagging, never guessed here.
    """
    match = VERSION_RE.match(version)
    if not match:
        raise ValueError(f"not a release version: {version}")
    base, kind, number = match.groups()
    if kind is not None:
        return f"{base}{kind}{int(number) + 1}"
    major, minor, patch = base.split(".")
    return f"{major}.{minor}.{int(patch) + 1}"


def next_version(current: str, published: Iterable[str]) -> str:
    """The version a release without an explicit one should use.

    The files win when they are already ahead of everything published: that
    is someone having run `set-version.py 0.3.0rc1` on purpose. Otherwise the
    newest version that went out anywhere is bumped, so a release never asks
    a registry for a number it already holds -- npm and PyPI both refuse that
    for ever, and a manual run that defaulted to the files' version tried to
    republish 0.2.0-rc.4 on 2026-10-16.

    `published` may hold any spelling (`0.2.0-rc.4`, `v0.2.0rc4`); anything
    that is not a release version, like npm's `0.0.0-stage`, is ignored.
    """
    seen = set()
    for raw in published:
        try:
            seen.add(canonical(raw))
        except ValueError:
            continue
    if not seen:
        return current
    latest = max(seen, key=version_key)
    if version_key(current) > version_key(latest):
        return current
    return bump(latest)


#: Package indexes the Python distributions may be on. Both, always: an rc goes
#: to TestPyPI and a final to PyPI, and a number used on either is used.
PY_INDEXES = ["https://pypi.org/pypi", "https://test.pypi.org/pypi"]

NPM_REGISTRY = "https://registry.npmjs.org"


def _fetch_json(url: str) -> dict | None:
    """GET a registry document; None when the package does not exist there.

    Any other failure raises. Guessing a version because a registry was
    unreachable is how a number gets reused, so this fails closed.
    """
    # S310: every URL is built from the https constants above, never from input.
    request = urllib.request.Request(url, headers={"Accept": "application/json"})  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def published_versions() -> list[str]:
    """Every version any of this repo's packages has on any channel.

    The npm registry, PyPI and TestPyPI, plus the `v*` tags on origin -- a tag
    can exist for a release whose publishing failed, and `gh release create`
    would attach to the old tag rather than mint a new one.
    """
    found: list[str] = []
    for rel in NPM_PACKAGES:
        name = json.loads((ROOT / rel).read_text(encoding="utf-8"))["name"]
        doc = _fetch_json(f"{NPM_REGISTRY}/{urllib.parse.quote(name, safe='@')}")
        found.extend((doc or {}).get("versions", {}))
    for index in PY_INDEXES:
        for name in DIST_NAMES.values():
            doc = _fetch_json(f"{index}/{name}/json")
            found.extend((doc or {}).get("releases", {}))
    # git from PATH, as every workflow step runs it; the arguments are constants.
    tags = subprocess.run(
        ["git", "ls-remote", "--tags", "--refs", "origin", "v*"],  # noqa: S607
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    found.extend(line.rsplit("refs/tags/", 1)[-1] for line in tags.splitlines() if line)
    return found


_SIBLINGS = set(DIST_NAMES.values())


def _in_lockstep(current: dict[str, str]) -> bool:
    """One Python version, and npm carrying its semver translation.

    The two families are compared through `npm_version` rather than for string
    equality, because a prerelease is spelled differently on each side.
    """
    python = {v for label, v in current.items() if label.startswith("py:")}
    npm = {v for label, v in current.items() if label.startswith(("npm:", "ts:"))}
    if len(python) != 1:
        return False
    try:
        return npm == {npm_version(next(iter(python)))}
    except ValueError:
        return False


def _pyproject(package: str) -> Path:
    return ROOT / "packages" / package / "pyproject.toml"


def read_versions() -> dict[str, str]:
    """Current version of every package, keyed by a human-readable label."""
    found: dict[str, str] = {}
    for package in PY_PACKAGES:
        path = _pyproject(package)
        if not path.exists():
            continue
        match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', path.read_text(encoding="utf-8"))
        found[f"py:{package}"] = match.group(1) if match else "?"
    for path in NPM_PACKAGES:
        full = ROOT / path
        if not full.exists():
            continue
        data = json.loads(full.read_text(encoding="utf-8"))
        found[f"npm:{data.get('name', path.parent.name)}"] = data.get("version", "?")
    for path in TS_VERSION_FILES:
        full = ROOT / path
        if not full.exists():
            continue
        match = _TS_VERSION_RE.search(full.read_text(encoding="utf-8"))
        found[f"ts:{path.as_posix()}"] = match.group(1) if match else "?"
    return found


#: Where a sibling name may legitimately appear as a requirement. Scoped on
#: purpose: a naive search-and-replace also rewrites `name = "jef"` in the
#: [project] table, producing the requirement string `jef==0.1.0` as the
#: distribution's own name -- which fails to build with an obscure message.
_DEPENDENCY_BLOCK = re.compile(
    r"(?ms)^(?:dependencies\s*=\s*\[.*?\]|\[project\.optional-dependencies\].*?(?=^\[|\Z))"
)


def _rewrite_dependencies(text: str, version: str) -> str:
    """Pin every sibling dependency to an exact version.

    Rewrites `"jef-server"`, `"jef[torch]"` and an already-pinned
    `"jef-train==0.1.0"` alike, so the script is idempotent and safe to re-run.
    """

    def pin(match: re.Match[str]) -> str:
        name = match.group("name")
        if name not in _SIBLINGS:
            return match.group(0)
        # Extras must survive the rewrite: "jef[torch]" has to become
        # "jef[torch]==0.2.0", not "jef==0.2.0", or the torch extra silently
        # stops being installed.
        extras = match.group("extras") or ""
        return f'"{name}{extras}=={version}"'

    requirement = re.compile(r'"(?P<name>jef(?:-[a-z]+)?)(?P<extras>\[[a-z,]+\])?(?:==[^"\]]+)?"')
    return _DEPENDENCY_BLOCK.sub(lambda block: requirement.sub(pin, block.group(0)), text)


def apply(version: str, *, check: bool) -> list[str]:
    """Write (or verify) the version everywhere. Returns problems found."""
    problems: list[str] = []

    for package in PY_PACKAGES:
        path = _pyproject(package)
        if not path.exists():
            problems.append(f"missing {path.relative_to(ROOT)}")
            continue
        original = path.read_text(encoding="utf-8")
        updated = re.sub(r'(?m)^(version\s*=\s*)"[^"]+"', rf'\1"{version}"', original, count=1)
        updated = _rewrite_dependencies(updated, version)
        if check:
            if updated != original:
                problems.append(f"{path.relative_to(ROOT)} is not at {version}")
        elif updated != original:
            path.write_text(updated, encoding="utf-8")

    for rel in NPM_PACKAGES:
        path = ROOT / rel
        if not path.exists():
            problems.append(f"missing {rel}")
            continue
        original = path.read_text(encoding="utf-8")
        data = json.loads(original)
        wanted = npm_version(version)
        if check:
            if data.get("version") != wanted:
                problems.append(f"{rel} is at {data.get('version')}, not {wanted}")
            continue
        data["version"] = wanted
        # Keep npm's own formatting: two-space indent and a trailing newline.
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for rel in TS_VERSION_FILES:
        path = ROOT / rel
        wanted = npm_version(version)
        if not path.exists():
            problems.append(f"missing {rel}")
            continue
        match = _TS_VERSION_RE.search(path.read_text(encoding="utf-8"))
        current = match.group(1) if match else None
        if check:
            if current != wanted:
                problems.append(f"{rel} exports VERSION {current!r}, not {wanted!r}")
            continue
        if current != wanted:
            path.write_text(_ts_version_source(wanted), encoding="utf-8")

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Set the monorepo version")
    parser.add_argument("version", nargs="?", help="e.g. 0.2.0 or 0.2.0rc1")
    parser.add_argument("--check", action="store_true", help="verify without writing")
    parser.add_argument("--show", action="store_true", help="print current versions")
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="print the PEP 440 spelling of VERSION (accepts v0.2.0-rc.4 etc.) and exit",
    )
    parser.add_argument(
        "--next",
        action="store_true",
        help="print the next unused version: the files' own if they are ahead of "
        "everything published, else the newest published version bumped",
    )
    args = parser.parse_args(argv)

    if args.next:
        current = read_versions()
        if not _in_lockstep(current):
            print("versions disagree in a lockstep repo; run --show", file=sys.stderr)
            return 1
        print(next_version(canonical(current["py:jef"]), published_versions()))
        return 0

    if args.show or not args.version:
        current = read_versions()
        for label, value in current.items():
            print(f"{label:28s} {value}")
        if not _in_lockstep(current):
            print("\nWARNING: versions disagree in a lockstep repo", file=sys.stderr)
            return 1
        return 0

    try:
        version = canonical(args.version)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.canonical:
        print(version)
        return 0

    problems = apply(version, check=args.check)
    if problems:
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            f"\n{len(problems)} package(s) not at {version}."
            + ("" if args.check else " Writing failed."),
            file=sys.stderr,
        )
        return 1

    npm = npm_version(version)
    suffix = f" (npm: {npm})" if npm != version else ""
    print(
        f"{'verified' if args.check else 'set'} {version} across {len(PY_PACKAGES)} python "
        f"+ {len(NPM_PACKAGES)} npm package(s) + {len(TS_VERSION_FILES)} source constant(s){suffix}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
