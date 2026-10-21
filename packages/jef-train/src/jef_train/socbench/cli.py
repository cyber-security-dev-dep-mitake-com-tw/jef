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
from typing import Any

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
    if args.source == "iot-23":
        options["unit"] = "host-fusion"
        options["window_s"] = args.window_seconds
        if args.max_cases_per_class is not None:
            options["max_cases_per_class"] = args.max_cases_per_class
    if args.source == "ctu-13":
        if args.background_rate is not None:
            options["background_rate"] = args.background_rate
        if args.max_cases_per_class is not None:
            options["max_cases_per_class"] = args.max_cases_per_class
        options["flow_labels"] = args.flow_labels
        if args.unit in {"host-window", "host-fusion"}:
            options["unit"] = args.unit
            options["window_s"] = args.window_seconds
            if args.background_host_rate is not None:
                options["background_host_rate"] = args.background_host_rate
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

    if args.annotate_command in {"new-batch", "check-batch"}:
        return _batch_command(args)
    cases = _cases_by_id(args.cases)
    try:
        if args.annotate_command == "export":
            n = write_jsonl(args.out, export_tasks(cases.values(), args.annotator))
            print(f"wrote {n} blind task(s) for {args.annotator} -> {args.out}")
            return 0
        annotations = [
            a for path in args.labels for a in parse_annotations(read_jsonl(path), cases)
        ]
        if args.annotate_command == "pilot-report":
            from .pilot import pilot_report

            pins: dict[str, set[str]] = {}
            for source in {str(c.get("source_dataset")) for c in cases.values()}:
                try:
                    # Pins store bare hex; cases store "sha256:<hex>". Compare one form.
                    pins[source] = {
                        a.sha256 if a.sha256.startswith("sha256:") else f"sha256:{a.sha256}"
                        for a in repo.load_pins(source).artifacts
                        if a.sha256
                    }
                except (FileNotFoundError, ValueError, KeyError):
                    continue
            try:
                report = pilot_report(
                    annotations,
                    cases,
                    weak_golds=read_jsonl(args.weak_gold) if args.weak_gold else [],
                    pinned_hashes=pins,
                    hourly_rate=args.hourly_rate,
                )
            except ValueError as exc:
                print(exc, file=sys.stderr)
                return 2
            args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", "utf-8")
            t, c = report["time"], report["completeness"]
            print(
                f"Tier B pilot by {report['reviewer']}: {report['cases']} cases, complete "
                f"{c['complete_share']}, undetermined {report['undetermined']['share']}, "
                f"median {t['median_seconds']}s, p90 {t['p90_seconds']}s, traceable "
                f"{report['traceability']['traceable_share']} -> {args.out}"
            )
            return 0
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


def _batch_command(args: argparse.Namespace) -> int:
    import json

    from .batch import BatchError, check_batch, git_authors, new_batch
    from .records import read_jsonl

    if args.annotate_command == "new-batch":
        cases = read_jsonl(args.cases)
        meta = new_batch(
            cases,
            batch_id=args.batch_id,
            purpose=args.purpose,
            reviewers=args.reviewer,
            recipe=args.recipe,
            adjudicator=args.adjudicator,
            note=args.note,
        )
        out = args.batches_dir / args.batch_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "batch.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", "utf-8"
        )
        try:
            check_batch(out)  # refuses a malformed batch before anyone labels it
        except BatchError as exc:
            print(exc, file=sys.stderr)
            return 2
        print(f"{out}/batch.json: {len(meta['cases'])} case(s), {meta['cases_digest']}")
        return 0

    batch_cases = read_jsonl(args.cases) if args.cases else None
    reports, failed = [], 0
    for batch_dir in args.batch:
        try:
            report = check_batch(
                batch_dir, cases=batch_cases, authors=git_authors(batch_dir) if args.git else None
            )
        except BatchError as exc:
            print(f"FAIL  {exc}", file=sys.stderr)
            failed += 1
            continue
        reports.append(report)
        failed += bool(report["problems"])
        status = "FAIL" if report["problems"] else "ok  "
        done = ", ".join(
            f"{r} {c['labelled']}/{c['of']}" for r, c in report["completeness"].items()
        )
        print(
            f"{status}  {report['batch_id']} ({report['purpose']}): {done}; kappa {report['kappa']}; "
            f"open disagreements {report['open_disagreements']}; tier if merged {report['tier_if_merged']}"
        )
        for p in report["problems"]:
            print(f"      - {p}")
        for n in report["notes"]:
            print(f"      note: {n}")
    if args.out:
        args.out.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return 1 if failed else 0


def cmd_dedupe(repo: SocbenchRepo, args: argparse.Namespace) -> int:
    import json

    from .records import read_jsonl, write_jsonl
    from .splits import dedupe_states

    cases, golds, report = dedupe_states(read_jsonl(args.cases), read_jsonl(args.gold))
    write_jsonl(args.out_cases, cases)
    write_jsonl(args.out_gold, golds)
    report_path = args.report or args.out_cases.with_suffix(".dedupe.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    print(
        f"kept {report['kept_cases']} of {report['input_cases']} cases; dropped "
        f"{len(report['dropped_same_label'])} same-label and "
        f"{len(report['dropped_conflicting_labels'])} conflicting-label duplicates -> {report_path}"
    )
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

    from .baselines import JefScorer, LogisticScorer, RuleScorer, Scorer
    from .records import read_jsonl
    from .remote import HttpScorer, ScorerConfigError, load_scorer_config
    from .report import evaluate

    cases, golds = read_jsonl(args.cases), read_jsonl(args.gold)
    lock = json.loads(args.lock.read_text("utf-8"))
    gold_by = {g["case_id"]: g for g in golds}
    train_ids = set(lock["splits"].get("train", []))
    train = [c for c in cases if c["case_id"] in train_ids]
    scorers: list[Scorer] = []
    if not args.scorer and not args.scorer_config:
        print("name at least one --scorer or --scorer-config", file=sys.stderr)
        return 2
    for spec in args.scorer or []:
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
            scorers.append(HttpScorer(url, name=f"http:{url}"))
        else:
            print(f"unknown scorer {spec!r}; use rules, logreg, jef or http=URL", file=sys.stderr)
            return 2
    for config in args.scorer_config or []:
        try:
            scorers.extend(load_scorer_config(config))
        except (ScorerConfigError, OSError, ValueError) as exc:
            print(exc, file=sys.stderr)
            return 2
    names = [s.name for s in scorers]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        print(f"scorer names must be unique; repeated: {dupes}", file=sys.stderr)
        return 2
    allowed = set(args.allow_egress or [])
    for s in scorers:
        if getattr(s, "egress", "none") == "external" and getattr(s, "host", "") not in allowed:
            print(
                f"refusing scorer {s.name!r}: cases would leave this machine for "
                f"{getattr(s, 'host', '')!r}. If that egress is approved (data licence, "
                f"masking, processor), pass --allow-egress {getattr(s, 'host', '')}",
                file=sys.stderr,
            )
            return 2
    try:
        report = evaluate(cases, golds, lock, scorers, split=args.split)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    report["manifest"]["egress_allowed"] = sorted(set(args.allow_egress or []))
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(f"[{report['evidence_class']}] {report['evidence_note']}")
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
    b.add_argument("--source", required=True, choices=["ait-ads", "ctu-13", "iot-23"])
    b.add_argument("--input", type=Path, help="archive (default: the pinned file in the data dir)")
    b.add_argument("--labels", type=Path, help="AIT-ADS labels.csv (default: beside the archive)")
    b.add_argument("--out", type=Path, help="output dir (default: <data-dir>/derived/<source>)")
    b.add_argument("--bucket-seconds", type=int, help="AIT-ADS: case window (600)")
    b.add_argument("--background-rate", type=float, help="CTU-13: Background sampling (0.001)")
    b.add_argument(
        "--max-cases-per-class", type=int, help="CTU-13: cap per verdict per scenario (300)"
    )
    b.add_argument(
        "--unit",
        choices=["conversation", "host-window", "host-fusion"],
        default="conversation",
        help="CTU-13: one conversation per case, or one source host per time window "
        "(task soc.ndr.host.v1, host-level labels); host-fusion adds behaviour gold, conversation "
        "evidence and a per-minute series",
    )
    b.add_argument(
        "--window-seconds", type=int, default=300, help="CTU-13 host-window: window (300)"
    )
    b.add_argument(
        "--flow-labels",
        choices=["behaviour", "host"],
        default="behaviour",
        help="CTU-13 conversations: label a bot's flows by behaviour (default; DNS and plain "
        "sessions become unknown) or by host (every bot flow is botnet; the G1/NDR v1-v2 mapping)",
    )
    b.add_argument(
        "--background-host-rate",
        type=float,
        help="CTU-13 host-window: Background hosts kept (0.002)",
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
        ("pilot-report", "one reviewer's pilot: completeness, time, traceability (Tier B)"),
    ):
        ap = asub.add_parser(name, help=help_)
        ap.add_argument("--cases", type=Path, required=True)
        ap.add_argument(
            "--labels", type=Path, action="append", required=True, help="one file per reviewer"
        )
        ap.add_argument("--out", type=Path, required=True)
        if name == "agree":
            ap.add_argument("--queue", type=Path, help="write the disagreement queue here")
        elif name == "adjudicate":
            ap.add_argument("--decisions", type=Path, help="adjudicator decisions for the queue")
        else:
            ap.add_argument(
                "--weak-gold", type=Path, help="source labels, for a labelled source-label check"
            )
            ap.add_argument("--hourly-rate", type=float, help="reviewer cost per hour")

    nb = asub.add_parser("new-batch", help="batch.json for sampled cases (ids and digest, no text)")
    nb.add_argument("--cases", type=Path, required=True)
    nb.add_argument("--batch-id", required=True)
    nb.add_argument("--purpose", choices=["pilot", "gold"], required=True)
    nb.add_argument(
        "--reviewer", action="append", required=True, help="analyst-anon-NN, repeatable"
    )
    nb.add_argument("--adjudicator", help="analyst-anon-NN (gold batches)")
    nb.add_argument("--recipe", required=True, help="commands that regenerate the cases")
    nb.add_argument("--note", default="")
    nb.add_argument(
        "--batches-dir", type=Path, default=Path("benchmark/socbench/annotation/batches")
    )
    cb = asub.add_parser("check-batch", help="what CI checks on a batch pull request")
    cb.add_argument("--batch", type=Path, required=True, action="append")
    cb.add_argument("--cases", type=Path, help="the case file, for full validation")
    cb.add_argument("--git", action="store_true", help="check reviewer independence from git")
    cb.add_argument("--out", type=Path, help="write the report as JSON")

    au = sub.add_parser(
        "audit", help="leakage audit: host reuse, near-duplicates, how window labels were decided"
    )
    au.add_argument("--cases", type=Path, required=True)
    au.add_argument("--gold", type=Path, required=True)
    au.add_argument("--out", type=Path, required=True)

    dd = sub.add_parser(
        "dedupe", help="drop byte-identical states before locking (same label: keep one)"
    )
    dd.add_argument("--cases", type=Path, required=True)
    dd.add_argument("--gold", type=Path, required=True)
    dd.add_argument("--out-cases", type=Path, required=True)
    dd.add_argument("--out-gold", type=Path, required=True)
    dd.add_argument("--report", type=Path, help="default: <out-cases>.dedupe.json")

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
    sm.add_argument(
        "--per-family", type=int, default=60, help="hard cap on cases drawn from each family"
    )
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
        "--scorer", action="append", help="rules | logreg | jef | http=URL (repeatable)"
    )
    ev.add_argument(
        "--scorer-config",
        action="append",
        type=Path,
        help="JSON file of named, pinned remote scorers (systemone or gateway); see "
        "benchmark/socbench/scorers.example.json",
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

    tr = sub.add_parser("triage", help="incident-triage scene: five questions, or the scene")
    trsub = tr.add_subparsers(dest="triage_command", required=True)
    te = trsub.add_parser(
        "evaluate",
        help="score the five soar_zh questions (questions) or what the scene proposed (scene)",
    )
    te.add_argument("--mode", choices=["questions", "scene"], required=True)
    te.add_argument("--pack", type=Path, required=True, help="a benchmark/incident-triage pack")
    te.add_argument(
        "--scene",
        type=Path,
        help="scene YAML (default: scenes/incident-triage.zh-tw.yaml in this checkout)",
    )
    te.add_argument(
        "--scorer",
        choices=["jef", "replay"],
        default="jef",
        help="jef: run the engine; replay: the pack's recorded answers (contract smoke)",
    )
    te.add_argument("--backbone", default="hashing")
    te.add_argument("--head")
    te.add_argument("--calibration")
    te.add_argument(
        "--include-pending",
        action="store_true",
        help="also score labels awaiting confirmation; the report says DIAGNOSTIC",
    )
    te.add_argument("--source-sha", help="fail unless HEAD starts with this")
    te.add_argument("--model-sha", help="fail unless --head has this sha256")
    te.add_argument("--scene-sha", help="fail unless the scene file has this sha256")
    te.add_argument("--out", type=Path, required=True)

    ts = trsub.add_parser("synth", help="render the synthetic families into a pack")
    ts.add_argument("--families", type=Path, required=True)
    ts.add_argument("--out", type=Path, required=True)

    tb = trsub.add_parser("baselines", help="B0-B3 on one locked split, and the routing decision")
    tb.add_argument("--pack", type=Path, required=True)
    tb.add_argument("--lock", type=Path, required=True)
    tb.add_argument("--split", default="test")
    tb.add_argument("--scene", type=Path)
    tb.add_argument(
        "--backbone", help="B3: run the jef engine with this backbone (omit to skip B3)"
    )
    tb.add_argument("--head")
    tb.add_argument("--calibration")
    tb.add_argument("--include-pending", action="store_true")
    tb.add_argument("--out", type=Path, required=True)

    ce = trsub.add_parser(
        "candidate-export", help="a locked pack -> a jef_train.fit corpus for the five tasks"
    )
    ce.add_argument("--pack", type=Path, required=True)
    ce.add_argument("--lock", type=Path, required=True)
    ce.add_argument("--scene", type=Path)
    ce.add_argument("--min-tier", default="B", choices=list("AB"))
    ce.add_argument(
        "--dry-run",
        action="store_true",
        help="export synthetic (Tier E) labels to exercise the pipeline; never deployable",
    )
    ce.add_argument(
        "--with-vendor-signals",
        action="store_true",
        help="also train on cases whose state shows the detector's own output (ablation arm only)",
    )
    ce.add_argument("--out", type=Path, required=True)

    cc = trsub.add_parser(
        "candidate-check", help="is a fitted candidate more than insufficient? (ceiling: research)"
    )
    cc.add_argument("--candidate", type=Path, required=True)
    cc.add_argument("--corpus", type=Path, required=True)
    cc.add_argument("--scene", type=Path)
    cc.add_argument("--backbone", required=True)
    cc.add_argument("--out", type=Path, required=True)

    ic = trsub.add_parser(
        "ingest-cloudflare",
        help="read-only: pull your own zone's security events into a private directory",
    )
    ic.add_argument("--token-env", required=True, help="env var holding an Analytics-read token")
    ic.add_argument("--zone-id-env", required=True, help="env var holding the zone id")
    ic.add_argument(
        "--zone-alias", required=True, help="name recorded instead of the zone id (e.g. site-a)"
    )
    ic.add_argument("--start", required=True, help="UTC, e.g. 2026-10-19T00:00:00Z")
    ic.add_argument("--end", required=True)
    ic.add_argument("--window-minutes", type=int, default=60)
    ic.add_argument("--limit", type=int, default=1000)
    ic.add_argument(
        "--out-private",
        type=Path,
        required=True,
        help="a directory outside any git checkout; files are written owner-only",
    )
    ic.add_argument(
        "--mock-events",
        type=Path,
        help="answer from a synthetic fixture instead of Cloudflare (contract tests and CI)",
    )

    nc = trsub.add_parser(
        "normalize-cloudflare",
        help="pseudonymise raw events and fold them into requests (needs JEF_PSEUDONYM_KEY)",
    )
    nc.add_argument("--input", type=Path, required=True, help="the ingest-cloudflare directory")
    nc.add_argument("--output", type=Path, required=True, help="private directory for requests")
    nc.add_argument("--key-env", default="JEF_PSEUDONYM_KEY")

    bc = trsub.add_parser(
        "build-cloudflare-cases",
        help="normalised requests -> a private, unlabelled incident-triage pack",
    )
    bc.add_argument("--input", type=Path, required=True, help="the normalize-cloudflare directory")
    bc.add_argument("--out", type=Path, required=True, help="private pack directory")
    bc.add_argument("--zone-alias", required=True)
    bc.add_argument("--context", type=Path, help="approval evidence (JSONL)")
    bc.add_argument("--gap-minutes", type=int, default=15)
    bc.add_argument(
        "--with-vendor-signals",
        action="store_true",
        help="also write a marked variant of each case that shows the WAF's own outputs",
    )
    bc.add_argument("--key-env", default="JEF_PSEUDONYM_KEY")

    cr = trsub.add_parser(
        "cloudflare-report", help="counts safe to publish from a private Cloudflare pack"
    )
    cr.add_argument("--pack", type=Path, required=True)
    cr.add_argument("--out", type=Path, required=True)

    tl = trsub.add_parser("lock", help="family-grouped splits; refuses leaks and near copies")
    tl.add_argument("--pack", type=Path, required=True)
    tl.add_argument("--out", type=Path, help="write the lock here")
    tl.add_argument("--verify", type=Path, help="check the pack against an existing lock")
    tl.add_argument("--seed", default="incident-triage")
    tl.add_argument(
        "--holdout-after",
        help="temporal test split: families observed at or after this UTC date/minute are test",
    )
    return p


def cmd_triage(args: argparse.Namespace) -> int:
    if args.triage_command == "synth":
        from .triage_synth import build_families, write_pack

        cases, exps = build_families(args.families)
        write_pack(args.out, cases, exps)
        print(
            f"{len(cases)} cases in {len({c['case_family'] for c in cases})} families -> {args.out}"
        )
        return 0
    if args.triage_command == "ingest-cloudflare":
        return cmd_triage_ingest_cloudflare(args)
    if args.triage_command in (
        "normalize-cloudflare",
        "build-cloudflare-cases",
        "cloudflare-report",
    ):
        return cmd_triage_cloudflare_pack(args)
    if args.triage_command == "lock":
        import json

        from .triage import PackError, load_pack
        from .triage_synth import LockError, triage_lock, verify_triage_lock

        try:
            pack = load_pack(args.pack)
        except PackError as exc:
            print(exc, file=sys.stderr)
            return 2
        exps = [pack.expectations[c["case_id"]] for c in pack.cases]
        if args.verify:
            problems = verify_triage_lock(
                json.loads(args.verify.read_text("utf-8")), pack.cases, exps
            )
            for problem in problems:
                print(problem, file=sys.stderr)
            return 1 if problems else 0
        if not args.out:
            print("lock needs --out (or --verify)", file=sys.stderr)
            return 2
        try:
            lock = triage_lock(pack.cases, exps, seed=args.seed, holdout_after=args.holdout_after)
        except LockError as exc:
            print(exc, file=sys.stderr)
            return 1
        args.out.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", "utf-8")
        print(
            f"locked {len(pack.cases)} cases: "
            f"{ {k: len(v) for k, v in lock['splits'].items()} } -> {args.out}"
        )
        return 0
    if args.triage_command == "baselines":
        return cmd_triage_baselines(args)
    if args.triage_command in ("candidate-export", "candidate-check"):
        return cmd_triage_candidate(args)
    return cmd_triage_evaluate(args)


def cmd_triage_ingest_cloudflare(args: argparse.Namespace) -> int:
    import json
    import os
    from datetime import datetime, timedelta

    from .cloudflare_mock import FakeZone, load_fixture_events
    from .cloudflare_source import CloudflareError, GraphQLClient, ingest

    token = os.environ.get(args.token_env, "")
    zone_id = os.environ.get(args.zone_id_env, "")
    for name, value in ((args.token_env, token), (args.zone_id_env, zone_id)):
        if not value:
            print(f"environment variable {name} is not set", file=sys.stderr)
            return 2
    transport = (
        FakeZone(token, zone_id, load_fixture_events(args.mock_events))
        if args.mock_events
        else None
    )
    try:
        start = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
        end = datetime.fromisoformat(args.end.replace("Z", "+00:00"))
        client = GraphQLClient(token, zone_id, transport=transport)
        result = ingest(
            client,
            zone_alias=args.zone_alias,
            start=start,
            end=end,
            out_dir=args.out_private,
            step=timedelta(minutes=args.window_minutes),
            limit=args.limit,
        )
    except (CloudflareError, ValueError) as exc:
        message = str(exc).replace(token, "***").replace(zone_id, "***")
        print(f"ingest-cloudflare: {message}", file=sys.stderr)
        return 1
    print(json.dumps({**result.to_json(), "mock": bool(args.mock_events)}, indent=2))
    return 1 if result.incomplete else 0


def cmd_triage_cloudflare_pack(args: argparse.Namespace) -> int:
    import json

    from .cloudflare_cases import build_cases, public_report, write_private_pack, write_requests
    from .cloudflare_normalize import PseudonymError, Pseudonymizer, Request, normalize_events
    from .cloudflare_normalize import read_raw as read_cloudflare_raw
    from .cloudflare_source import CloudflareError
    from .triage import PackError, load_pack, privacy_findings

    try:
        if args.triage_command == "cloudflare-report":
            pack = load_pack(args.pack)
            weak_path = args.pack / "weak.jsonl"
            weak = [json.loads(x) for x in weak_path.read_text("utf-8").splitlines() if x.strip()]
            exps = [pack.expectations[c["case_id"]] for c in pack.cases]
            text = json.dumps(public_report(pack.cases, weak, exps), indent=2, sort_keys=True)
            findings = privacy_findings(text)
            if findings:
                print("refusing to write the report:", *findings, sep="\n  ", file=sys.stderr)
                return 1
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n", "utf-8")
            print(f"report -> {args.out}")
            return 0
        pseud = Pseudonymizer.from_env(args.key_env)
        if args.triage_command == "normalize-cloudflare":
            requests = normalize_events(read_cloudflare_raw(args.input), pseud)
            path = write_requests(args.output, requests)
            print(f"{len(requests)} requests -> {path}")
            return 0
        rows = (args.input / "requests.jsonl").read_text("utf-8").splitlines()
        requests = [Request(**json.loads(x)) for x in rows if x.strip()]
        context = []
        if args.context:
            context = [
                json.loads(x) for x in args.context.read_text("utf-8").splitlines() if x.strip()
            ]
        cases, exps, weak = build_cases(
            requests,
            zone_alias=args.zone_alias,
            pseud=pseud,
            context=context,
            gap_minutes=args.gap_minutes,
            with_vendor_signals=args.with_vendor_signals,
        )
        out = write_private_pack(args.out, cases, exps, weak)
        load_pack(out)  # the pack must pass the same loader as every other pack
        print(
            f"{len(cases)} cases ({len({c['case_family'] for c in cases})} families), "
            f"all unlabelled -> {out}"
        )
        return 0
    except (CloudflareError, PseudonymError, PackError, ValueError, KeyError) as exc:
        print(f"{args.triage_command}: {exc}", file=sys.stderr)
        return 1


def cmd_triage_candidate(args: argparse.Namespace) -> int:
    import json

    from jef_scene import load_scene

    from .triage import PackError, find_root, load_pack
    from .triage_candidate import CandidateError, check_candidate, export_triage_corpus

    repo = find_root(Path.cwd()).parents[1]
    scene = load_scene(args.scene or repo / "scenes" / "incident-triage.zh-tw.yaml")
    try:
        if args.triage_command == "candidate-export":
            pack = load_pack(args.pack)
            lock = json.loads(args.lock.read_text("utf-8"))
            m = export_triage_corpus(
                pack,
                lock,
                scene,
                args.out,
                min_tier=args.min_tier,
                dry_run=args.dry_run,
                with_vendor_signals=args.with_vendor_signals,
            )
            print(f"exported {m['splits']} ({m['note']}) -> {args.out}")
            return 0
        report = check_candidate(
            args.candidate, args.corpus, scene, backbone=args.backbone, repo_root=repo
        )
    except (CandidateError, PackError) as exc:
        print(exc, file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8")
    short = {k: v["status"] for k, v in report["buckets"].items()}
    print(f"candidate status: {report['status']} (ceiling research); buckets {short} -> {args.out}")
    return 0


def cmd_triage_baselines(args: argparse.Namespace) -> int:
    import json

    from jef_core import BilinearHead, Calibrator, Engine
    from jef_scene import load_scene

    from .triage import PackError, find_root, load_pack
    from .triage_baselines import render_baselines, run_baselines
    from .triage_eval import EvalError, Provenance, report_class
    from .triage_synth import verify_triage_lock

    try:
        pack = load_pack(args.pack)
    except PackError as exc:
        print(exc, file=sys.stderr)
        return 2
    lock = json.loads(args.lock.read_text("utf-8"))
    problems = verify_triage_lock(
        lock, pack.cases, [pack.expectations[c["case_id"]] for c in pack.cases]
    )
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 2
    scene_path = (
        args.scene or find_root(args.pack).parents[1] / "scenes" / "incident-triage.zh-tw.yaml"
    )
    scene = load_scene(scene_path)
    jef = None
    if args.backbone:
        jef = Engine(
            args.backbone,
            BilinearHead.load(args.head) if args.head else None,
            Calibrator.load(args.calibration) if args.calibration else None,
        )
    try:
        report = run_baselines(
            pack, lock, scene, split=args.split, jef=jef, include_pending=args.include_pending
        )
    except EvalError as exc:
        print(exc, file=sys.stderr)
        return 2
    smoke = jef is not None and jef.backbone.name.startswith("hashing")
    cls, note = report_class(
        pack.expectations, contract_smoke=smoke, include_pending=args.include_pending
    )
    report["evidence_class"], report["evidence_note"] = cls, note
    if jef is not None:
        report["provenance"] = Provenance(
            scene_path, scene, jef, args.head, args.calibration, "jef", pack
        ).to_dict()
    report["lock"] = {"cases_digest": lock["cases_digest"], "seed": lock["seed"]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        "utf-8",
    )
    (args.out / "summary.md").write_text(render_baselines(report), "utf-8")
    print(f"baselines on {args.split}: {report['routing_decision'].get('routing')} -> {args.out}")
    return 0


def cmd_triage_evaluate(args: argparse.Namespace) -> int:
    """jef-bench triage evaluate: the two modes never share a report."""
    from jef_core import BilinearHead, Calibrator, Engine
    from jef_scene import load_scene

    from .triage import PackError, ReplayEngine, find_root, load_pack
    from .triage_eval import (
        EvalError,
        Provenance,
        report_class,
        run_questions,
        run_scene,
        scene_failures,
        score_questions,
        score_scene,
        write_report,
    )

    try:
        pack = load_pack(args.pack)
    except (PackError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        return 2
    scene_path = (
        args.scene or find_root(args.pack).parents[1] / "scenes" / "incident-triage.zh-tw.yaml"
    )
    scene = load_scene(scene_path)
    calibrator = Calibrator.load(args.calibration) if args.calibration else None
    replay_sources: list[str] | None = None
    excluded_cases: dict[str, list[str]] = {}
    if args.scorer == "replay":
        # The replayed answers stand in for the model; the engine only carries
        # the calibration. Cases without a record are counted, not dropped.
        engine = Engine(calibrator=calibrator)
        cases = [c for c in pack.cases if c["case_id"] in pack.replay]
        missing = sorted(c["case_id"] for c in pack.cases if c["case_id"] not in pack.replay)
        if missing:
            excluded_cases["no_replay_record"] = missing
        replay_sources = sorted({str(r.get("source")) for r in pack.replay.values()})

        def engine_for(case: Any) -> Engine:
            return ReplayEngine(pack.replay[case["case_id"]], calibrator)
    else:
        engine = Engine(
            args.backbone, BilinearHead.load(args.head) if args.head else None, calibrator
        )
        cases = pack.cases

        def engine_for(case: Any) -> Engine:
            return engine

    prov = Provenance(
        scene_path,
        scene,
        engine,
        args.head,
        args.calibration,
        args.scorer,
        pack,
        replay_sources=replay_sources,
    )
    try:
        prov.check(source_sha=args.source_sha, model_sha=args.model_sha, scene_sha=args.scene_sha)
    except EvalError as exc:
        print(exc, file=sys.stderr)
        return 2

    smoke = args.scorer == "replay" or engine.backbone.name.startswith("hashing")
    cls, note = report_class(
        pack.expectations, contract_smoke=smoke, include_pending=args.include_pending
    )
    failures: list[str] = []
    if args.mode == "questions":
        rows = run_questions(engine_for, cases, scene)
        metrics = score_questions(
            rows, pack.expectations, scene=scene, include_pending=args.include_pending
        )
    else:
        rows = run_scene(engine_for, cases, scene)
        metrics = score_scene(rows, pack.expectations, scene=scene)
        failures = scene_failures(metrics)
    metrics["excluded_cases"] = excluded_cases
    report = {
        "mode": args.mode,
        "evidence_class": cls,
        "evidence_note": note,
        "provenance": prov.to_dict(),
        "metrics": metrics,
        "rows": rows,
        "failures": failures,
    }
    write_report(args.out, report)
    print(f"{args.mode}: {len(rows)} cases, {cls}, failures {len(failures)} -> {args.out}")
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "triage":
        # Needs the incident-triage pack, not the socbench sources.
        return cmd_triage(args)
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
    if args.command == "dedupe":
        return cmd_dedupe(repo, args)
    if args.command == "audit":
        import json

        from .leakage import audit
        from .records import read_jsonl

        report = audit(read_jsonl(args.cases), read_jsonl(args.gold))
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", "utf-8")
        hr, nd = report["host_reuse"], report["near_duplicates"]
        print(
            f"hosts in more than one scenario, per class: "
            f"{ {k: v['in_more_than_one_scenario'] for k, v in hr['per_class'].items()} }; "
            f"median share of held-out cases whose host is in training "
            f"{hr['median_share_of_test_cases_whose_host_is_in_training']}; near-duplicates "
            f"same label {nd['share_same_label_elsewhere']}, other label only "
            f"{nd['share_other_label_only']} -> {args.out}"
        )
        return 0
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
