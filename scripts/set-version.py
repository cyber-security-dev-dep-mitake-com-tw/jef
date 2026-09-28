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
    python scripts/set-version.py --check 0.2.0   # verify without writing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
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

#: PEP 440 subset: what a release tag may look like.
VERSION_RE = re.compile(r"^(\d+\.\d+\.\d+)(?:(a|b|rc)(\d+))?$")

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


_SIBLINGS = set(DIST_NAMES.values())


def _in_lockstep(current: dict[str, str]) -> bool:
    """One Python version, and npm carrying its semver translation.

    The two families are compared through `npm_version` rather than for string
    equality, because a prerelease is spelled differently on each side.
    """
    python = {v for label, v in current.items() if label.startswith("py:")}
    npm = {v for label, v in current.items() if label.startswith("npm:")}
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

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Set the monorepo version")
    parser.add_argument("version", nargs="?", help="e.g. 0.2.0 or 0.2.0rc1")
    parser.add_argument("--check", action="store_true", help="verify without writing")
    parser.add_argument("--show", action="store_true", help="print current versions")
    args = parser.parse_args(argv)

    if args.show or not args.version:
        current = read_versions()
        for label, value in current.items():
            print(f"{label:28s} {value}")
        if not _in_lockstep(current):
            print("\nWARNING: versions disagree in a lockstep repo", file=sys.stderr)
            return 1
        return 0

    version = args.version.removeprefix("v")
    if not VERSION_RE.match(version):
        print(f"'{version}' is not a release version (expected 1.2.3 or 1.2.3rc1)", file=sys.stderr)
        return 2

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
        f"+ {len(NPM_PACKAGES)} npm package(s){suffix}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
