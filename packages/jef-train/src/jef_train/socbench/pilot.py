"""The single-reviewer pilot report (plan v3, §2.2–2.3, NEXT-V3-06).

When one security engineer labels the first 50–65 cases alone, the result is
Tier B (``single-human-reviewed``) and must be reported as exactly that. This
report measures what a single reviewer *can* show, and nothing it cannot:

* **Completeness**: verdict present and allowed, evidence cited (unless the
  verdict is undetermined), a note written, time recorded. Field by field.
* **Undetermined rate**: the real share of ``unknown`` / ``insufficient_evidence``.
* **Time**: median and p90 seconds per case, from ``seconds_spent``.
* **Traceability**: the share of cases whose ``source_hash`` is one of the
  pinned artifacts in ``manifests/<source>.json``, so the evidence can be re-read.
* **Agreement with the source label**, when the weak gold is given: a check on
  the reviewer and the source together, labelled as such. It is **not**
  inter-annotator agreement and no kappa is computed.
* **Reviewer cost**, when an hourly rate is given, kept apart from inference cost.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .annotate import TASK_LABELS, Annotation
from .report import UNDETERMINED

__all__ = ["pilot_report"]


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def pilot_report(
    annotations: Iterable[Annotation],
    cases: Mapping[str, Mapping[str, Any]],
    *,
    weak_golds: Iterable[Mapping[str, Any]] = (),
    pinned_hashes: Mapping[str, set[str]] | None = None,
    hourly_rate: float | None = None,
) -> dict[str, Any]:
    anns = list(annotations)
    reviewers = sorted({a.annotator for a in anns})
    if len(reviewers) != 1:
        raise ValueError(
            f"a pilot report is for one reviewer; got {reviewers}. Two reviewers use `annotate agree`."
        )
    weak = {g["case_id"]: g["gold"]["verdict"] for g in weak_golds}
    pins = pinned_hashes or {}

    fields: Counter[str] = Counter()
    complete = 0
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    seconds: list[float] = []
    untraceable: Counter[str] = Counter()
    agree, compared = 0, 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for a in anns:
        case = cases[a.case_id]
        task = case["task_id"]
        undetermined = a.verdict in UNDETERMINED
        has_verdict = a.verdict in TASK_LABELS[task]
        has_evidence = bool(a.evidence_ids) or undetermined
        has_note = bool(a.note.strip())
        timed = a.seconds_spent is not None
        fields["verdict"] += has_verdict
        fields["evidence"] += has_evidence
        fields["note"] += has_note
        fields["timed"] += timed
        complete += has_verdict and has_evidence and has_note and timed
        by_task[task][a.verdict] += 1
        if timed:
            seconds.append(float(a.seconds_spent or 0))
        source = str(case.get("source_dataset"))
        if source not in pins:
            untraceable[f"{source}: no pinned source manifest"] += 1
        elif case.get("source_hash") not in pins[source]:
            untraceable[f"{source}: source_hash is not a pinned artifact"] += 1
        if a.case_id in weak:
            compared += 1
            agree += a.verdict == weak[a.case_id]
            confusion[weak[a.case_id]][a.verdict] += 1

    n = len(anns)
    undetermined_n = sum(a.verdict in UNDETERMINED for a in anns)
    median = _quantile(seconds, 0.5)
    mean = (sum(seconds) / len(seconds)) if seconds else None
    report: dict[str, Any] = {
        "kind": "single-reviewer pilot",
        "trust_tier": "B",
        "label_origin": "single-human-reviewed",
        "reviewer": reviewers[0],
        "cases": n,
        "completeness": {
            "complete_share": round(complete / n, 4) if n else None,
            "by_field": {k: round(v / n, 4) for k, v in sorted(fields.items())} if n else {},
            "rule": "verdict allowed; evidence ids unless undetermined; a note; seconds_spent",
        },
        "undetermined": {
            "count": undetermined_n,
            "share": round(undetermined_n / n, 4) if n else None,
        },
        "verdicts_by_task": {t: dict(c) for t, c in sorted(by_task.items())},
        "time": {
            "timed": len(seconds),
            "median_seconds": median,
            "p90_seconds": _quantile(seconds, 0.9),
            "hours_per_100_cases_single_review": round(100 * mean / 3600, 2) if mean else None,
        },
        "traceability": {
            "traceable_share": round((n - sum(untraceable.values())) / n, 4) if n else None,
            "untraceable": dict(untraceable),
        },
        "source_label_check": (
            {
                "compared": compared,
                "matches_source_label": agree,
                "share": round(agree / compared, 4) if compared else None,
                "confusion_source_to_reviewer": {k: dict(v) for k, v in sorted(confusion.items())},
                "caveat": "one reviewer against the source's own label; NOT inter-annotator "
                "agreement, and no kappa is reported",
            }
            if weak
            else None
        ),
        "reviewer_cost": (
            {
                "hourly_rate": hourly_rate,
                "per_case": round(mean / 3600 * hourly_rate, 4) if mean else None,
                "per_100_cases": round(100 * mean / 3600 * hourly_rate, 2) if mean else None,
                "note": "reviewer time only; inference and hosting cost are reported separately",
            }
            if hourly_rate is not None
            else None
        ),
        "may_be_described_as": (
            f"a {n}-case single-reviewer research pilot with measured time, by one security "
            "engineer; not a double-reviewed gold set and not an agreement measurement"
        ),
    }
    return report
