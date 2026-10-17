"""``jef-bench``: list sources, fetch the public ones, check the cache.

    jef-bench sources                       # every source, access, priority
    jef-bench fetch --source ait-ads --pin  # download, hash, record the digest
    jef-bench fetch --all-public            # every fetchable source
    jef-bench check                         # offline: cache vs pins
    jef-bench pin --source lanl-cyber1 --local   # digests for hand-placed files
    jef-bench build --source ait-ads        # cases.jsonl + gold.jsonl from a fetched source

Exit codes: 0 all good, 1 a MISMATCH or a failed download, 2 a restricted
source was asked for by name or the arguments made no sense.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .adapters import ADAPTERS
from .manifest import SocbenchRepo, Source
from .pipeline import (
    DATA_DIR,
    AccessRestricted,
    ArtifactStatus,
    check_source,
    fetch_source,
    pin_local,
)

__all__ = ["main"]


def _print(source_id: str, statuses: Sequence[ArtifactStatus]) -> bool:
    bad = False
    for s in statuses:
        print(f"{source_id:26s} {s.name:40s} {s.state:9s} {s.detail}")
        bad = bad or s.failed
    return bad


def _select(repo: SocbenchRepo, ids: Sequence[str] | None, *, all_public: bool) -> list[Source]:
    sources = repo.load_sources()
    if all_public:
        return [s for s in sources if s.fetchable and not ADAPTERS[s.id].reference_only]
    if not ids:
        return sources
    by_id = {s.id: s for s in sources}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"unknown source id(s): {', '.join(missing)}")
    return [by_id[i] for i in ids]


def cmd_sources(repo: SocbenchRepo) -> int:
    for s in repo.load_sources():
        adapter = ADAPTERS[s.id]
        kind = "reference" if adapter.reference_only else ("fetch" if s.fetchable else "HUMAN")
        print(f"{s.id:26s} {s.access:30s} {kind:9s} priority={s.priority!s:10s} {s.name}")
    return 0


def cmd_fetch(repo: SocbenchRepo, args: argparse.Namespace) -> int:
    rc = 0
    for source in _select(repo, args.source, all_public=args.all_public):
        try:
            statuses, _ = fetch_source(
                repo,
                source,
                base=args.data_dir,
                refresh=args.refresh,
                pin=args.pin,
                inspect=args.inspect,
            )
        except AccessRestricted as exc:
            print(f"{source.id:26s} {'-':40s} {'HUMAN':9s} {exc.instructions}")
            if args.source:  # asked for by name: that is an error worth a non-zero exit
                rc = max(rc, 2)
            continue
        except OSError as exc:
            print(f"{source.id:26s} {'-':40s} {'FAILED':9s} {exc}")
            rc = max(rc, 1)
            continue
        if _print(source.id, statuses):
            rc = max(rc, 1)
        if source.access == "sample-public-full-restricted":
            print(f"{'':26s} note: {ADAPTERS[source.id].instructions}")
    return rc


def cmd_check(repo: SocbenchRepo, args: argparse.Namespace) -> int:
    rc = 0
    for source in _select(repo, args.source, all_public=False):
        if _print(source.id, check_source(repo, source, base=args.data_dir)):
            rc = 1
    return rc


def cmd_pin(repo: SocbenchRepo, args: argparse.Namespace) -> int:
    if not args.local:
        print(
            "pin without --local is `fetch --pin`; --local records hand-placed files",
            file=sys.stderr,
        )
        return 2
    rc = 0
    for source in _select(repo, args.source, all_public=False):
        statuses, _ = pin_local(repo, source, base=args.data_dir)
        if _print(source.id, statuses):
            rc = 1
    return rc


def cmd_build(repo: SocbenchRepo, args: argparse.Namespace) -> int:
    from .build import build_source

    options: dict[str, object] = {}
    if args.source == "ait-ads" and args.bucket_seconds:
        options["bucket_seconds"] = args.bucket_seconds
    if args.source == "ctu-13":
        if args.background_rate is not None:
            options["background_rate"] = args.background_rate
        if args.max_cases_per_class is not None:
            options["max_cases_per_class"] = args.max_cases_per_class
    try:
        summary = build_source(
            repo,
            args.source,
            base=args.data_dir,
            input_path=args.input,
            labels_path=args.labels,
            out_dir=args.out,
            options=options,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    print(
        f"{args.source}: {summary['written']['cases']} cases, tier {summary['trust_tier']}, "
        f"verdicts {summary['by_verdict']} -> {summary['out_dir']}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jef-bench", description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--root", type=Path, default=None, help="benchmark/socbench directory (default: find it)"
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"where downloads live (default {DATA_DIR}, env JEF_SOCBENCH_DATA)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("sources", help="list every source with its access status")

    f = sub.add_parser("fetch", help="download public artifacts and hash them")
    g = f.add_mutually_exclusive_group(required=True)
    g.add_argument("--source", action="append", help="source id (repeatable)")
    g.add_argument("--all-public", action="store_true", help="every fetchable source")
    f.add_argument("--refresh", action="store_true", help="re-download even if cached")
    f.add_argument("--pin", action="store_true", help="record sha256 for unpinned artifacts")
    f.add_argument("--inspect", action="store_true", help="also report archive/CSV shape")

    c = sub.add_parser("check", help="offline: compare the cache against the pins")
    c.add_argument("--source", action="append", help="source id (default: all)")

    n = sub.add_parser("pin", help="record digests for files placed by hand")
    n.add_argument("--source", action="append", required=True)
    n.add_argument("--local", action="store_true", help="hash files already in the data dir")

    b = sub.add_parser("build", help="derive validated cases and weak gold from a fetched source")
    b.add_argument("--source", required=True, choices=["ait-ads", "ctu-13"])
    b.add_argument("--input", type=Path, help="archive (default: the pinned file in the data dir)")
    b.add_argument("--labels", type=Path, help="AIT-ADS labels.csv (default: beside the archive)")
    b.add_argument("--out", type=Path, help="output dir (default: <data-dir>/derived/<source>)")
    b.add_argument("--bucket-seconds", type=int, help="AIT-ADS: case window (600)")
    b.add_argument("--background-rate", type=float, help="CTU-13: Background sampling (0.001)")
    b.add_argument("--max-cases-per-class", type=int, help="CTU-13: cap per verdict (2000)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        repo = SocbenchRepo(args.root)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.command == "sources":
        return cmd_sources(repo)
    if args.command == "fetch":
        return cmd_fetch(repo, args)
    if args.command == "check":
        return cmd_check(repo, args)
    if args.command == "pin":
        return cmd_pin(repo, args)
    if args.command == "build":
        return cmd_build(repo, args)
    return 2  # pragma: no cover -- argparse rejects unknown commands first


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
