"""``jef-bench``: list sources, fetch the public ones, check the cache.

    jef-bench sources                       # every source, access, priority
    jef-bench fetch --source ait-ads --pin  # download, hash, record the digest
    jef-bench fetch --all-public            # every fetchable source
    jef-bench check                         # offline: cache vs pins
    jef-bench pin --source lanl-cyber1 --local   # digests for hand-placed files
    jef-bench build --source ait-ads        # cases.jsonl + gold.jsonl from a fetched source
    jef-bench annotate export --cases c.jsonl --annotator analyst-anon-01 --out tasks.jsonl
    jef-bench annotate agree --cases c.jsonl --labels a.jsonl --labels b.jsonl --out report.json
    jef-bench annotate adjudicate --cases c.jsonl --labels a.jsonl --labels b.jsonl \
        --decisions d.jsonl --out gold.jsonl
    jef-bench lock --cases c.jsonl --gold gold.jsonl --out lock.json [--pin family=test]
    jef-bench lock --verify lock.json --cases c.jsonl --gold gold.jsonl
    jef-bench evaluate --lock lock.json --cases c.jsonl --gold gold.jsonl \
        --scorer rules --scorer logreg [--scorer jef] [--scorer http=http://127.0.0.1:4141] --out report.json

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


def _cases_by_id(path: Path) -> dict[str, dict[str, object]]:
    from .records import read_jsonl

    return {str(c["case_id"]): c for c in read_jsonl(path)}


def cmd_annotate(repo: SocbenchRepo, args: argparse.Namespace) -> int:
    import json

    from .annotate import AnnotationError, adjudicate, agreement, export_tasks, parse_annotations
    from .records import read_jsonl, validate_records, write_jsonl

    cases = _cases_by_id(args.cases)
    try:
        if args.annotate_command == "export":
            n = write_jsonl(args.out, export_tasks(cases.values(), args.annotator))
            print(f"wrote {n} blind task(s) for {args.annotator} -> {args.out}")
            return 0
        annotations = [
            a for path in args.labels for a in parse_annotations(read_jsonl(path), cases)
        ]
        if args.annotate_command == "agree":
            report = agreement(annotations, cases)
            args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", "utf-8")
            if args.queue:
                write_jsonl(args.queue, report["queue"])
            for task, r in report["by_task"].items():
                print(
                    f"{task}: {r['pairs']} pairs, agreement {r['percent_agreement']}, kappa {r['kappa']}"
                )
            print(f"queue: {len(report['queue'])} case(s) -> {args.queue or args.out}")
            return 0
        decisions = read_jsonl(args.decisions) if args.decisions else []
        golds = adjudicate(annotations, cases, decisions)
    except AnnotationError as exc:
        print(exc, file=sys.stderr)
        return 2
    validate_records(repo, "gold", golds)
    n = write_jsonl(args.out, golds)
    disputed = sum(g["adjudication_status"] == "disputed" for g in golds)
    print(f"wrote {n} gold record(s) -> {args.out} ({disputed} still disputed)")
    return 0


def cmd_lock(repo: SocbenchRepo, args: argparse.Namespace) -> int:
    import json

    from .records import read_jsonl, validate_records
    from .splits import LockError, make_lock, verify_lock

    cases, golds = read_jsonl(args.cases), read_jsonl(args.gold)
    if args.verify:
        problems = verify_lock(json.loads(args.verify.read_text("utf-8")), cases, golds)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print("verified" if not problems else "MISMATCH")
        return 1 if problems else 0
    validate_records(repo, "case", cases)
    validate_records(repo, "gold", golds)
    pinned = dict(item.split("=", 1) for item in args.pin or [])
    try:
        lock = make_lock(cases, golds, seed=args.seed, pinned=pinned)
    except LockError as exc:
        print(exc, file=sys.stderr)
        return 1
    args.out.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(
        f"locked {len(cases)} case(s): "
        + ", ".join(f"{k}={len(v)}" for k, v in lock["splits"].items())
    )
    return 0


def cmd_evaluate(repo: SocbenchRepo, args: argparse.Namespace) -> int:
    import json

    from .baselines import HttpScorer, JefScorer, LogisticScorer, RuleScorer, Scorer
    from .records import read_jsonl
    from .report import evaluate

    cases, golds = read_jsonl(args.cases), read_jsonl(args.gold)
    lock = json.loads(args.lock.read_text("utf-8"))
    gold_by = {g["case_id"]: g for g in golds}
    train_ids = set(lock["splits"].get("train", []))
    train = [c for c in cases if c["case_id"] in train_ids]
    scorers: list[Scorer] = []
    for spec in args.scorer:
        if spec == "rules":
            scorers.append(RuleScorer())
        elif spec == "logreg":
            scorers.append(LogisticScorer().fit(train, gold_by))
        elif spec == "jef":
            from jef_core import BilinearHead, Calibrator, Engine

            engine = Engine(
                args.backbone,
                BilinearHead.load(args.head) if args.head else None,
                Calibrator.load(args.calibration) if args.calibration else None,
            )
            scorers.append(JefScorer(engine))
        elif spec.startswith("http="):
            url = spec.removeprefix("http=")
            http = HttpScorer(url, name=f"http:{url}")
            if http.egress == "external" and http.host not in set(args.allow_egress or []):
                print(
                    f"refusing {url}: cases would leave this machine for {http.host!r}. "
                    "If that egress is approved (data licence, masking, processor), pass "
                    f"--allow-egress {http.host}",
                    file=sys.stderr,
                )
                return 2
            scorers.append(http)
        else:
            print(f"unknown scorer {spec!r}; use rules, logreg, jef or http=URL", file=sys.stderr)
            return 2
    try:
        report = evaluate(cases, golds, lock, scorers, split=args.split)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    report["manifest"]["egress_allowed"] = sorted(set(args.allow_egress or []))
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", "utf-8")
    for row in report["comparison"]:
        print(
            f"{row['scorer']:10s} {row['task']:26s} macro-F1 {row['macro_f1']}  "
            f"PR-AUC {row['pr_auc']}  ECE {row['ece']}  mistaken closures {row['mistaken_closures']}"
        )
    return 0


def cmd_sample(repo: SocbenchRepo, args: argparse.Namespace) -> int:
    import json

    from .records import read_jsonl, write_jsonl
    from .sample import SampleError, sample_cases

    cases = read_jsonl(args.cases)
    weak = read_jsonl(args.weak_gold) if args.weak_gold else []
    exclude: set[str] = set()
    for path in args.exclude or []:
        exclude |= {str(c["case_id"]) for c in read_jsonl(path)}
    try:
        chosen, manifest = sample_cases(
            cases,
            weak,
            task=args.task,
            per_family=args.per_family,
            min_per_stratum=args.min_per_stratum,
            seed=args.seed,
            families=args.family or None,
            exclude=exclude,
        )
    except SampleError as exc:
        print(exc, file=sys.stderr)
        return 1
    write_jsonl(args.out, chosen)
    manifest_path = args.manifest or args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(
        f"sampled {manifest['sampled']} cases from {manifest['families']} families "
        f"{manifest['sampled_by_weak_label']} -> {args.out} (manifest {manifest_path})"
    )
    for w in manifest["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    return 0


def cmd_export_corpus(repo: SocbenchRepo, args: argparse.Namespace) -> int:
    import json

    from .corpus_export import ExportError, export_corpus
    from .records import read_jsonl
    from .splits import LockError

    cases, golds = read_jsonl(args.cases), read_jsonl(args.gold)
    lock = json.loads(args.lock.read_text("utf-8"))
    try:
        manifest = export_corpus(
            cases,
            golds,
            lock,
            args.out,
            min_tier=args.min_tier,
            question_lang=args.question_lang,
        )
    except (LockError, ExportError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"wrote {args.out}: " + ", ".join(f"{k}={v}" for k, v in manifest["splits"].items()))
    if manifest["excluded_by_tier"]:
        print(
            "excluded below tier "
            f"{args.min_tier}: {manifest['excluded_by_tier']} (weak labels are reported, not trained on)"
        )
    print(f"next: python -m jef_train.fit --corpus {args.out} --out models/jef-soc-candidate")
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
    b.add_argument(
        "--max-cases-per-class", type=int, help="CTU-13: cap per verdict per scenario (300)"
    )

    a = sub.add_parser("annotate", help="blind two-reviewer annotation and adjudication")
    asub = a.add_subparsers(dest="annotate_command", required=True)
    ae = asub.add_parser("export", help="blind tasks for one reviewer")
    ae.add_argument("--cases", type=Path, required=True)
    ae.add_argument("--annotator", required=True, help="anonymised handle, e.g. analyst-anon-01")
    ae.add_argument("--out", type=Path, required=True)
    for name, help_ in (
        ("agree", "agreement, kappa and the disagreement queue"),
        ("adjudicate", "gold from reviews and decisions"),
    ):
        ap = asub.add_parser(name, help=help_)
        ap.add_argument("--cases", type=Path, required=True)
        ap.add_argument(
            "--labels", type=Path, action="append", required=True, help="one file per reviewer"
        )
        ap.add_argument("--out", type=Path, required=True)
        if name == "agree":
            ap.add_argument("--queue", type=Path, help="write the disagreement queue here")
        else:
            ap.add_argument("--decisions", type=Path, help="adjudicator decisions for the queue")

    lk = sub.add_parser("lock", help="family-grouped splits and the gold lock")
    lk.add_argument("--cases", type=Path, required=True)
    lk.add_argument("--gold", type=Path, required=True)
    lk.add_argument("--out", type=Path, help="lock.json to write")
    lk.add_argument("--verify", type=Path, help="check these files against an existing lock")
    lk.add_argument("--seed", default="socbench")
    lk.add_argument(
        "--pin", action="append", metavar="FAMILY=SPLIT", help="force a family into a split"
    )

    sm = sub.add_parser(
        "sample", help="family-aware stratified sample of cases for blind annotation"
    )
    sm.add_argument("--cases", type=Path, required=True, help="cases.jsonl from build")
    sm.add_argument(
        "--weak-gold", type=Path, help="gold.jsonl from build: used to stratify, never written out"
    )
    sm.add_argument("--task", help="restrict to one task_id, e.g. soc.ndr.technique.v1")
    sm.add_argument("--per-family", type=int, default=60)
    sm.add_argument("--min-per-stratum", type=int, default=5)
    sm.add_argument("--seed", default="socbench-sample")
    sm.add_argument("--family", action="append", help="only these family_ids (repeatable)")
    sm.add_argument(
        "--exclude", action="append", type=Path, help="cases.jsonl already sampled (repeatable)"
    )
    sm.add_argument("--out", type=Path, required=True, help="sampled-cases.jsonl to write")
    sm.add_argument("--manifest", type=Path, help="default: <out>.manifest.json")

    ex = sub.add_parser(
        "export-corpus", help="locked gold -> jef_train.fit corpus, split as the lock says"
    )
    ex.add_argument("--lock", type=Path, required=True)
    ex.add_argument("--cases", type=Path, required=True)
    ex.add_argument("--gold", type=Path, required=True)
    ex.add_argument("--out", type=Path, required=True, help="corpus directory to write")
    ex.add_argument(
        "--min-tier",
        default="A",
        choices=list("ABCDE"),
        help="weakest label trust tier to train on (default A: two independent reviewers)",
    )
    ex.add_argument("--question-lang", default="en", choices=["en", "zh-TW"])

    ev = sub.add_parser("evaluate", help="score a locked split with baselines and JEF")
    ev.add_argument("--lock", type=Path, required=True)
    ev.add_argument("--cases", type=Path, required=True)
    ev.add_argument("--gold", type=Path, required=True)
    ev.add_argument(
        "--scorer", action="append", required=True, help="rules | logreg | jef | http=URL"
    )
    ev.add_argument("--split", default="test")
    ev.add_argument(
        "--allow-egress",
        action="append",
        metavar="HOST",
        help="permit an http= scorer whose host is not loopback; repeatable, recorded in the manifest",
    )
    ev.add_argument("--backbone", default="hashing", help="jef scorer: backbone id")
    ev.add_argument("--head", help="jef scorer: head.npz")
    ev.add_argument("--calibration", help="jef scorer: calibration.json")
    ev.add_argument("--out", type=Path, required=True)
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
    if args.command == "annotate":
        return cmd_annotate(repo, args)
    if args.command == "sample":
        return cmd_sample(repo, args)
    if args.command == "export-corpus":
        return cmd_export_corpus(repo, args)
    if args.command == "evaluate":
        return cmd_evaluate(repo, args)
    if args.command == "lock":
        if not args.verify and not args.out:
            print("lock needs --out (or --verify)", file=sys.stderr)
            return 2
        return cmd_lock(repo, args)
    return 2  # pragma: no cover -- argparse rejects unknown commands first


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
