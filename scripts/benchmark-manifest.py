#!/usr/bin/env python3
"""Pin nightly benchmark results to what produced them, and check that pin later.

    python scripts/benchmark-manifest.py write --out benchmark-manifest.json \\
        --source-sha "$GITHUB_SHA" --run-id "$GITHUB_RUN_ID" docs/benchmarks.json docs/badges
    python scripts/benchmark-manifest.py verify benchmark-manifest.json --root <dir>

The nightly benchmark no longer commits to ``main`` (plan v3, NEXT-V3-01): it
uploads its results as an artifact with this manifest beside them. The
publisher workflow downloads the artifact, runs ``verify``, and only then opens
a pull request, so a reviewed number can always be traced to the commit, the
model and the exact files the run produced.

``write`` records the source commit and run, the model head's SHA-256 and
calibration schema (read from ``models/jef-v0`` unless ``--model-dir`` says
otherwise), every result file's SHA-256, one digest over all of them, and the
per-task case counts from ``benchmarks.json``. ``verify`` recomputes the file
digests and exits non-zero on any difference, missing file or extra file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _files(paths: list[Path], base: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in paths:
        full = p if p.is_absolute() else base / p
        if full.is_dir():
            for f in sorted(full.rglob("*")):
                if f.is_file():
                    out[f.relative_to(base).as_posix()] = _sha256(f)
        elif full.is_file():
            out[full.relative_to(base).as_posix()] = _sha256(full)
    return dict(sorted(out.items()))


def _digest(files: dict[str, str]) -> str:
    raw = "\n".join(f"{k} {v}" for k, v in sorted(files.items()))
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()


def write(args: argparse.Namespace) -> int:
    base = args.root.resolve()
    files = _files(args.paths, base)
    if not any(k.endswith("benchmarks.json") for k in files):
        print("no benchmarks.json among the inputs; refusing to pin an empty run", file=sys.stderr)
        return 1
    model_dir = (base / args.model_dir) if not args.model_dir.is_absolute() else args.model_dir
    model_manifest = model_dir / "manifest.json"
    cal = (
        json.loads(model_manifest.read_text()).get("calibration", {})
        if model_manifest.is_file()
        else {}
    )
    bench_path = next(base / k for k in files if k.endswith("benchmarks.json"))
    bench = json.loads(bench_path.read_text())
    tasks = {k: {"n": v.get("n")} for k, v in (bench.get("tasks") or {}).items()}
    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "kind": "nightly-third-party-benchmark",
        "source_sha": args.source_sha,
        "workflow_run_id": args.run_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": {
            "dir": args.model_dir.as_posix(),
            "head_sha256": _sha256(model_dir / "head.npz")
            if (model_dir / "head.npz").is_file()
            else None,
            "calibration_schema": cal.get("schema"),
            "calibration_legacy": cal.get("legacy"),
        },
        "backbone": bench.get("backbone"),
        "datasets": sorted({k.split("/")[0] for k in tasks}),
        "tasks": tasks,
        "files": files,
        "result_digest": _digest(files),
        "note": (
            "Results from a scheduled run. They reach main only through a reviewed pull request "
            "opened by benchmark-publish.yml after `verify` passes."
        ),
    }
    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}: {len(files)} files, {manifest['result_digest']}")
    return 0


def verify(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text())
    base = args.root.resolve()
    recorded: dict[str, str] = manifest.get("files", {})
    actual = _files([Path(k) for k in recorded], base)
    problems = []
    for name, digest in recorded.items():
        if name not in actual:
            problems.append(f"{name}: missing")
        elif actual[name] != digest:
            problems.append(f"{name}: {actual[name]} != recorded {digest}")
    if _digest(recorded) != manifest.get("result_digest"):
        problems.append("result_digest does not match the recorded file digests")
    if args.source_sha and manifest.get("source_sha") != args.source_sha:
        problems.append(f"source_sha {manifest.get('source_sha')} != expected {args.source_sha}")
    if problems:
        print(
            "benchmark manifest verification FAILED:\n  " + "\n  ".join(problems), file=sys.stderr
        )
        return 1
    print(
        f"verified {len(recorded)} files against {args.manifest} ({manifest.get('result_digest')})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    w = sub.add_parser("write")
    w.add_argument("paths", nargs="+", type=Path)
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--source-sha", required=True)
    w.add_argument("--run-id", required=True)
    w.add_argument("--model-dir", type=Path, default=Path("models/jef-v0"))
    w.add_argument("--root", type=Path, default=ROOT)
    v = sub.add_parser("verify")
    v.add_argument("manifest", type=Path)
    v.add_argument("--root", type=Path, default=ROOT)
    v.add_argument("--source-sha", help="also require this source commit")
    args = p.parse_args(argv)
    return write(args) if args.command == "write" else verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
