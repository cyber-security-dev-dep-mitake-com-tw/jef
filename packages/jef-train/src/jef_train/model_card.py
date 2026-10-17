"""Generate a model card from a model directory's own artifacts (plan v2, §C.4).

    python -m jef_train.model_card models/jef-v0
    python -m jef_train.model_card models/jef-soc-candidate-v1 \\
        --corpus-manifest data/corpus-soc-a-v1/manifest.json \\
        --eval-report evidence/benchmark/jef-v1.json
    python -m jef_train.model_card models/jef-v0 --check   # CI: is the card current?

Every statement in the card is read from ``head.npz``, ``calibration.json``,
``report.json`` and ``manifest.json`` (and, when given, the corpus manifest
and a ``jef-bench evaluate`` report). Nothing is typed in by hand, so a card
cannot claim a calibration the artifact does not have. The output is
deterministic (no timestamps), which is what lets ``--check`` fail CI when a
model changes and its card does not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from jef_core.calibration import Calibrator, split_bucket_key

__all__ = ["build_card", "main"]

ALPHA = 0.10
ARTIFACTS = ("head.npz", "calibration.json", "report.json")


def _sha256(path: Path) -> str | None:
    """``sha256:<hex>``, the form the model manifest records."""
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _identity(model_dir: Path, manifest: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    lines = ["| Artifact | SHA-256 (actual) | Manifest agrees |", "|---|---|---|"]
    problems: list[str] = []
    listed = (manifest or {}).get("artifacts", {})
    for name in ARTIFACTS:
        actual = _sha256(model_dir / name)
        expected = (listed.get(name) or {}).get("sha256")
        if actual is None:
            agrees = "missing"
            problems.append(f"{name} is missing")
        elif expected is None:
            agrees = "not listed" if manifest else "no manifest"
        else:
            agrees = "yes" if expected == actual else "**NO**"
            if expected != actual:
                problems.append(f"{name} does not match its manifest")
        lines.append(f"| `{name}` | `{actual or '—'}` | {agrees} |")
    return lines, problems


def _calibration(model_dir: Path) -> tuple[list[str], dict[str, Any]]:
    path = model_dir / "calibration.json"
    if not path.is_file():
        return ["No `calibration.json`: every answer is uncalibrated."], {
            "tasks": [],
            "legacy": None,
        }
    cal = Calibrator.load(path)
    rows = [
        "| Bucket | Task | Samples | Status at α=0.10 | P(correct) map |",
        "|---|---|---|---|---|",
    ]
    tasks: dict[str, list[str]] = {}
    for key in sorted(cal.buckets):
        task, kind, n = split_bucket_key(key)
        b = cal.buckets[key]
        status = cal.status(kind, n, ALPHA, task_id=task)
        tasks.setdefault(task or "(none: shape-scoped)", []).append(status)
        rows.append(
            f"| `{key}` | {task or '—'} | {b.n_samples or 'n/a'} | {status} | "
            f"{'yes' if b.correctness_map else 'no'} |"
        )
    head = [
        f"- Schema **{cal.schema}**{' (legacy: predates task-scoped calibration)' if cal.legacy else ''}.",
        f"- Backbone declared: `{cal.backbone or 'n/a'}`; head declared: `{cal.head or 'n/a'}`.",
        f"- Declared head SHA-256: `{cal.artifact.model_sha256 or 'not declared'}`.",
        f"- Dataset digest: `{cal.artifact.dataset_digest or 'not declared'}`; "
        f"split policy: {cal.artifact.split_policy or 'n/a'}; "
        f"calibration samples: {cal.artifact.sample_count or 'n/a'}.",
        "",
    ]
    validated = sorted(
        t
        for t, ss in tasks.items()
        if all(s == "validated" for s in ss) and "|" not in t and not t.startswith("(")
    )
    return head + rows, {"tasks": validated, "legacy": cal.legacy, "all": tasks}


def _report_metrics(report: dict[str, Any]) -> list[str]:
    cal = report.get("calibrated") or {}
    unc = report.get("uncalibrated") or {}
    lines = [
        f"- Splits: train {report.get('splits', {}).get('train', 'n/a')}, calibration "
        f"{report.get('splits', {}).get('calibration', 'n/a')}, test "
        f"{report.get('splits', {}).get('test', 'n/a')}; split unit: "
        f"{report.get('split_unit', 'n/a')}.",
        f"- Independent groups: {report.get('groups', 'n/a')}.",
        f"- Excluded sources: {', '.join(report.get('excluded_sources') or []) or 'none'}.",
        "",
        "| Metric (test split of the training corpus) | Uncalibrated | Calibrated |",
        "|---|---|---|",
    ]
    for key, label in (
        ("overall_accuracy", "Accuracy"),
        ("overall_macro_f1", "Macro-F1"),
        ("overall_ece", "ECE (top probability)"),
        ("overall_brier", "Brier"),
        ("overall_ece_p_correct", "ECE of P(correct)"),
    ):
        lines.append(f"| {label} | {_fmt(unc.get(key))} | {_fmt(cal.get(key))} |")
    lines.append(
        f"| Majority-class baseline accuracy | {_fmt(report.get('majority_baseline_accuracy'))} | |"
    )
    by_source = cal.get("by_source") or {}
    if by_source:
        lines += ["", "| Source | n | Accuracy | Macro-F1 |", "|---|---|---|---|"]
        for src, d in sorted(by_source.items()):
            lines.append(
                f"| `{src}` | {d.get('n', 'n/a')} | {_fmt(d.get('accuracy'))} | {_fmt(d.get('macro_f1'))} |"
            )
    return lines


def _eval_section(ev: dict[str, Any] | None) -> list[str]:
    if not ev:
        return [
            "**Not evaluated.** No `jef-bench evaluate` report on a locked SOCBench split was "
            "supplied. Until one exists, nothing in this card says how the model performs on "
            "SOC alerts, and no detection-quality claim should be made from it.",
        ]
    lock = ev.get("lock", {})
    lines = [
        f"Locked split `{lock.get('split')}`, n={lock.get('n')}, cases "
        f"`{lock.get('cases_digest', '')[:19]}…`, gold `{lock.get('gold_digest', '')[:19]}…`.",
        "",
        "| Scorer | Task | Macro-F1 | PR-AUC | ECE | Mistaken closures | Failure rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in ev.get("comparison", []):
        lines.append(
            f"| {row.get('scorer')} | {row.get('task')} | {_fmt(row.get('macro_f1'))} | "
            f"{_fmt(row.get('pr_auc'))} | {_fmt(row.get('ece'))} | "
            f"{_fmt(row.get('mistaken_closures'))} | {_fmt(row.get('failure_rate'))} |"
        )
    return lines


def _corpus_section(cm: dict[str, Any] | None) -> list[str]:
    if not cm:
        return [
            "No corpus manifest supplied: training-data provenance is limited to `report.json` above."
        ]
    lock = cm.get("lock", {})
    return [
        f"- Built by `{cm.get('builder')}` from lock `{str(lock.get('cases_digest'))[:19]}…` "
        f"(seed {lock.get('seed')}, ratios {lock.get('ratios')}).",
        f"- Minimum trust tier trained on: **{cm.get('min_trust_tier')}**; excluded below it: "
        f"{cm.get('excluded_by_tier') or 'none'}.",
        f"- Samples per split: {cm.get('splits')}; trust tiers by split: {cm.get('trust_tiers_by_split')}.",
        f"- Label origins: {cm.get('label_origins')}.",
    ]


def build_card(
    model_dir: Path,
    *,
    corpus_manifest: dict[str, Any] | None = None,
    eval_report: dict[str, Any] | None = None,
) -> str:
    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    report_path = model_dir / "report.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    ident, problems = _identity(model_dir, manifest)
    cal_lines, cal_facts = _calibration(model_dir)
    name = (manifest or {}).get("model_id") or model_dir.name

    automation = (
        "possible for: " + ", ".join(f"`{t}`" for t in cal_facts["tasks"])
        if cal_facts["tasks"]
        else "**not possible**: no task-scoped calibration bucket is validated"
    )
    limitations = []
    if cal_facts.get("legacy"):
        limitations.append(
            "Calibration is schema 1. `/v1/readiness` reports `calibration:legacy` and every "
            "required task as `legacy_calibration`; automation stays blocked."
        )
    if not eval_report:
        limitations.append("No blind SOC evaluation on Tier A gold (see *Blind SOC evaluation*).")
    if report:
        limitations.append(
            "The metrics under *Held-out metrics* come from the training corpus's own test "
            "split. They measure the tasks in that corpus (CVE and ATT&CK questions), not SOC "
            "alert triage."
        )
    limitations += problems

    out = [
        f"# Model card: {name}",
        "",
        "> Generated by `python -m jef_train.model_card` from the artifacts in this directory. "
        "Do not edit by hand; regenerate, and CI checks that it is current.",
        "",
        "## Identity",
        "",
        f"- Model id `{name}`, version `{(manifest or {}).get('version', 'n/a')}`, built from "
        f"commit `{(manifest or {}).get('git_commit', 'n/a')}`.",
        f"- Backbone `{report.get('backbone') or (manifest or {}).get('backbone', 'n/a')}` (frozen); "
        f"head `{report.get('head', 'n/a')}`, rank {report.get('rank', 'n/a')}.",
        "",
        *ident,
        "",
        "## Calibration",
        "",
        *cal_lines,
        "",
        f"Automation readiness from this artifact alone: {automation}. Readiness also needs a "
        "real backbone, a verified manifest, authentication, a persistent decision store and "
        "the deployment's `JEF_REQUIRED_TASK_IDS`; see `/v1/readiness`.",
        "",
        "## Training data",
        "",
        *_corpus_section(corpus_manifest),
        "",
        "## Held-out metrics (self-evaluated)",
        "",
        *(_report_metrics(report) if report else ["No `report.json`."]),
        "",
        "## Blind SOC evaluation",
        "",
        *_eval_section(eval_report),
        "",
        "## Intended use",
        "",
        "- **L0 research** and **L1 read-only decision support**: ranking, routing suggestions and "
        "analyst review, behind authentication.",
        "- **Not** for unsupervised T2–T4 actions (isolate, disable, block, close or suppress). "
        "Those always need a signed grant from a person, whatever the model's confidence.",
        "- `confidence` measures how concentrated a distribution is, not how often it is right; "
        "gate on `p_correct` only where the bucket has a validated correctness map.",
        "",
        "## Known limitations",
        "",
        *[f"- {x}" for x in limitations],
        "",
        "## Sign-off",
        "",
        "| Role | Name | Date | Decision |",
        "|---|---|---|---|",
        "| SOC lead | | | |",
        "| Security reviewer | | | |",
        "| Sponsor | | | |",
        "",
        "Unsigned rows mean this model has not been accepted for any use beyond L0.",
        "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model_dir", type=Path)
    p.add_argument("--corpus-manifest", type=Path)
    p.add_argument("--eval-report", type=Path)
    p.add_argument("--out", type=Path, help="default: <model_dir>/model-card.md")
    p.add_argument("--check", action="store_true", help="exit 1 if the card on disk is stale")
    args = p.parse_args(argv)
    card = build_card(
        args.model_dir,
        corpus_manifest=json.loads(args.corpus_manifest.read_text())
        if args.corpus_manifest
        else None,
        eval_report=json.loads(args.eval_report.read_text()) if args.eval_report else None,
    )
    out = args.out or args.model_dir / "model-card.md"
    if args.check:
        current = out.read_text() if out.is_file() else ""
        if current != card:
            print(
                f"{out} is stale or missing; regenerate with: "
                f"python -m jef_train.model_card {args.model_dir}",
                file=sys.stderr,
            )
            return 1
        print(f"{out} is current")
        return 0
    out.write_text(card)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
