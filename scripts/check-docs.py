#!/usr/bin/env python3
"""Docs drift: fail when the documentation stops describing the code.

    python scripts/check-docs.py

Checks, each of which has drifted in this repository before:

1. **Links.** Every relative link in every tracked Markdown file resolves to a
   file, and every ``#anchor`` into a Markdown file names a heading in it.
2. **Server settings.** Every ``JEF_*`` variable that
   ``jef_server/settings.py`` reads is documented in
   ``packages/jef-server/README.md``.
3. **Install lines.** While the packages are at a prerelease, an
   ``npm install @jef-ai/sdk`` or ``npm install n8n-nodes-jef`` without a tag
   or version installs whatever ``latest`` points at, which is not what a
   reader expects; it must say ``@next`` or pin a version.
4. **Capability matrix.** Every ``shipped`` row of ``docs/CAPABILITIES.md``
   cites at least one repository path in backticks, and every cited path
   exists.

Stdlib only. Exit 1 with one line per problem.
"""

from __future__ import annotations

import re
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"^(```|~~~)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def tracked_markdown() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.md"],  # noqa: S607
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    ).stdout.split()
    return [ROOT / p for p in out if "node_modules" not in p]


def _strip_code(text: str) -> list[str]:
    lines, fenced = [], False
    for line in text.splitlines():
        if FENCE.match(line.strip()):
            fenced = not fenced
            lines.append("")
            continue
        lines.append("" if fenced else re.sub(r"`[^`]*`", "", line))
    return lines


def github_slug(heading: str) -> str:
    """GitHub's anchor for a heading: lowercase, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"<[^>]+>", "", heading)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.replace("`", "").strip().lower()
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in " -":
            out.append("-" if ch == " " else ch)
        elif ch == "_" or cat[0] in {"L", "N", "M"}:
            out.append(ch)
    return "".join(out)


def anchors(path: Path) -> set[str]:
    seen: dict[str, int] = {}
    result: set[str] = set()
    fenced = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if FENCE.match(line.strip()):
            fenced = not fenced
            continue
        m = None if fenced else HEADING.match(line)
        if not m:
            continue
        slug = github_slug(m.group(2))
        n = seen.get(slug, 0)
        result.add(slug if n == 0 else f"{slug}-{n}")
        seen[slug] = n + 1
    return result


def check_links(files: list[Path]) -> list[str]:
    problems: list[str] = []
    cache: dict[Path, set[str]] = {}
    for md in files:
        for lineno, line in enumerate(_strip_code(md.read_text(encoding="utf-8")), 1):
            for target in LINK.findall(line):
                if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("//"):
                    continue  # http:, https:, mailto:, ...
                path_part, _, anchor = target.partition("#")
                dest = md if not path_part else (md.parent / path_part).resolve()
                where = f"{md.relative_to(ROOT)}:{lineno}"
                if not dest.exists():
                    problems.append(f"{where}: link to {target} does not resolve")
                    continue
                if anchor and dest.suffix == ".md":
                    found = cache.setdefault(dest, anchors(dest))
                    if anchor.lower() not in found:
                        problems.append(
                            f"{where}: {target} names no heading in {dest.relative_to(ROOT)}"
                        )
    return problems


def check_server_settings() -> list[str]:
    settings = (ROOT / "packages/jef-server/src/jef_server/settings.py").read_text(encoding="utf-8")
    readme = (ROOT / "packages/jef-server/README.md").read_text(encoding="utf-8")
    names = sorted(set(re.findall(r'"(JEF_[A-Z_]+)"', settings)))
    return [
        f"packages/jef-server/README.md does not document {n} (read in settings.py)"
        for n in names
        if n not in readme
    ]


def _version() -> str:
    text = (ROOT / "packages/jef/pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    return m.group(1) if m else ""


def check_install_lines(files: list[Path]) -> list[str]:
    if not re.search(r"(a|b|rc)\d+$", _version()):
        return []
    problems: list[str] = []
    bare = re.compile(r"npm (?:install|i)\s+(?:-[^\s]+\s+)*(@jef-ai/sdk|n8n-nodes-jef)(?![@\w-])")
    for md in files:
        for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            for pkg in bare.findall(line):
                problems.append(
                    f"{md.relative_to(ROOT)}:{lineno}: `npm install {pkg}` without @next or a version "
                    f"while the packages are at {_version()}"
                )
    return problems


def check_capabilities() -> list[str]:
    path = ROOT / "docs/CAPABILITIES.md"
    if not path.is_file():
        return ["docs/CAPABILITIES.md is missing"]
    problems: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("|") or "---" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[1].lower() not in {"shipped", "**shipped**"}:
            continue
        cited = re.findall(r"`([^`]+)`", cells[2] if len(cells) > 2 else "")
        paths = [
            c
            for c in cited
            if "/" in c or c.endswith((".py", ".md", ".yaml", ".yml", ".json", ".mjs", ".ts"))
        ]
        if not paths:
            problems.append(
                f"docs/CAPABILITIES.md:{lineno}: shipped row '{cells[0]}' cites no path"
            )
        for p in paths:
            if not (ROOT / p.rstrip("/")).exists():
                problems.append(
                    f"docs/CAPABILITIES.md:{lineno}: '{cells[0]}' cites {p}, which does not exist"
                )
    return problems


def main() -> int:
    files = tracked_markdown()
    problems = (
        check_links(files)
        + check_server_settings()
        + check_install_lines(files)
        + check_capabilities()
    )
    for p in problems:
        print(p)
    if problems:
        print(f"\n{len(problems)} docs problem(s)", file=sys.stderr)
        return 1
    print(f"docs: {len(files)} Markdown files checked, no drift")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
