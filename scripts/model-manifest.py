#!/usr/bin/env python3
"""Write or check a model directory's manifest.json.

    python scripts/model-manifest.py write models/jef-v0 --version 0.2.0rc5
    python scripts/model-manifest.py check models/jef-v0      # exit 1 on any mismatch

Stdlib only: the release workflow runs `check` before installing anything, so
weights whose bytes disagree with the manifest are refused before a single
upload (REL-003). The logic lives in jef_core/model_manifest.py, loaded here by
path so jef_core's numpy-importing package init never runs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "packages" / "jef" / "src" / "jef_core" / "model_manifest.py"
    spec = importlib.util.spec_from_file_location("jef_model_manifest", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["jef_model_manifest"] = module
    spec.loader.exec_module(module)
    return module


mm = _load()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("write", help="describe the directory as it is now")
    w.add_argument("model_dir", type=Path)
    w.add_argument("--version", required=True)
    w.add_argument("--model-id", default=None, help="defaults to the directory name")
    c = sub.add_parser("check", help="exit 1 when the directory disagrees with its manifest")
    c.add_argument("model_dir", type=Path)
    c.add_argument("--require", action="store_true", help="a missing manifest is an error too")
    args = parser.parse_args(argv)

    if args.command == "write":
        manifest = mm.build_manifest(
            args.model_dir,
            model_id=args.model_id or args.model_dir.name,
            version=args.version,
            git_commit=_git_commit(),
        )
        target = args.model_dir / mm.MANIFEST_NAME
        target.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {target}: {', '.join(manifest['artifacts'])}")
        return 0

    result = mm.check_model_dir(args.model_dir)
    if result.status == "absent":
        print(f"{args.model_dir} has no {mm.MANIFEST_NAME}", file=sys.stderr)
        return 1 if args.require else 0
    if not result.ok:
        for problem in result.problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            f"\n{args.model_dir} does not match its manifest. If the weights changed on purpose, "
            f"regenerate it: python scripts/model-manifest.py write {args.model_dir} --version <v>",
            file=sys.stderr,
        )
        return 1
    print(f"{args.model_dir} matches its manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
