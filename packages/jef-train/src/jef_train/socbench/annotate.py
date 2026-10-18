"""Two-reviewer annotation: blind export, agreement, adjudication into gold.

The workflow the review fixes (section 10.3):

    case candidates -> analyst A blind label -> analyst B blind label
    -> disagreement queue -> adjudicator -> gold (tier A or B)

What "blind" means here, and is enforced:

* an exported task carries the case's state and evidence and the labels the
  task allows. It never carries the source's weak gold, another reviewer's
  label, or a model's answer;
* each reviewer gets the tasks in their own deterministic order, so two
  reviewers working side by side do not see the same case at the same time;
* an adjudicator must be a third person: a decision signed by one of the two
  reviewers is refused.

``label_origin`` follows from what happened, never from what someone typed:
two reviewers -> ``double-human-reviewed`` (agreed or adjudicated), one ->
``single-human-reviewed``. A case the two disagree on with no decision yet is
``disputed`` and gets ``unknown``. The schema refuses double-human-reviewed
with fewer than two annotators (DATA-005), and this module cannot produce it.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "NEGATIVE_LABELS",
    "TASK_LABELS",
    "AnnotationError",
    "adjudicate",
    "agreement",
    "cohen_kappa",
    "export_tasks",
    "parse_annotations",
]

#: Allowed verdicts per task (roadmap 7.1, review 11.1). `unknown` and
#: `insufficient_evidence` are answers, not gaps.
TASK_LABELS: Mapping[str, tuple[str, ...]] = {
    "soc.alert.verdict.v1": (
        "confirmed_malicious",
        "suspicious",
        "confirmed_benign",
        "insufficient_evidence",
        "unknown",
    ),
    "soc.ndr.technique.v1": (
        "c2",
        "botnet",
        "dns_tunnel",
        "normal_automation",
        "benign",
        "unknown",
    ),
    # One source host over a time window; labels are host-level (ctu13_hosts):
    # botnet = the host is infected, c2 = a C&C-tagged flow in the window.
    "soc.ndr.host.v1": ("c2", "botnet", "benign", "unknown"),
    "soc.edr.execution.v1": ("malicious", "benign", "unknown"),
    "soc.identity.abuse.v1": ("compromised", "suspicious", "legitimate", "unknown"),
    "soc.phishing.verdict.v1": (
        "phishing",
        "suspicious",
        "spam_other",
        "legitimate",
        "insufficient_evidence",
        "unknown",
    ),
    "soc.response.authority.v1": ("allowed", "review_required", "prohibited"),
}

#: Verdicts that mean "this is not an attack", for hard-negative accounting.
NEGATIVE_LABELS = frozenset({"confirmed_benign", "benign", "legitimate", "normal_automation"})

_HANDLE = re.compile(r"^analyst-anon-[0-9]{2}$")


class AnnotationError(ValueError):
    pass


def _order_key(annotator: str, case_id: str) -> str:
    return hashlib.sha256(f"{annotator}\x00{case_id}".encode()).hexdigest()


def export_tasks(cases: Iterable[Mapping[str, Any]], annotator: str) -> list[dict[str, Any]]:
    """Blind tasks for one reviewer, in that reviewer's own order."""
    if not _HANDLE.match(annotator):
        raise AnnotationError(f"annotator handle must look like analyst-anon-07, not {annotator!r}")
    tasks = []
    for case in cases:
        task = case["task_id"]
        if task not in TASK_LABELS:
            raise AnnotationError(f"{case['case_id']}: no label set for task {task!r}")
        tasks.append(
            {
                "case_id": case["case_id"],
                "task_id": task,
                "annotator": annotator,
                "allowed_labels": list(TASK_LABELS[task]),
                "language": case.get("language"),
                "state": case["state"],
                "evidence": case["evidence"],
                # Filled in by the reviewer:
                "verdict": None,
                "confidence": None,
                "evidence_ids": [],
                "prohibited_actions": [],
                "required_followup": [],
                "note": "",
                # Optional, but the pilot needs it: wall-clock seconds spent on
                # this case, so the full gold set can be scheduled from measured
                # throughput rather than a guess.
                "seconds_spent": None,
            }
        )
    return sorted(tasks, key=lambda t: _order_key(annotator, t["case_id"]))


@dataclass(frozen=True)
class Annotation:
    case_id: str
    annotator: str
    verdict: str
    confidence: str
    evidence_ids: tuple[str, ...] = ()
    prohibited_actions: tuple[str, ...] = ()
    required_followup: tuple[str, ...] = ()
    note: str = ""
    seconds_spent: float | None = None


def parse_annotations(
    rows: Iterable[Mapping[str, Any]], cases: Mapping[str, Mapping[str, Any]]
) -> list[Annotation]:
    """Validate filled-in tasks against the cases they claim to label."""
    out: list[Annotation] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        cid, who = row.get("case_id"), row.get("annotator")
        if not isinstance(cid, str) or cid not in cases:
            raise AnnotationError(f"annotation for unknown case {cid!r}")
        if not isinstance(who, str) or not _HANDLE.match(who):
            raise AnnotationError(f"{cid}: annotator {who!r} is not an anonymised handle")
        if (cid, who) in seen:
            raise AnnotationError(f"{cid}: {who} labelled it twice")
        seen.add((cid, who))
        case = cases[cid]
        allowed = TASK_LABELS[case["task_id"]]
        verdict = row.get("verdict")
        if verdict not in allowed:
            raise AnnotationError(f"{cid}: verdict {verdict!r} is not one of {', '.join(allowed)}")
        confidence = row.get("confidence") or "medium"
        if confidence not in {"high", "medium", "low"}:
            raise AnnotationError(f"{cid}: confidence must be high, medium or low")
        known_ids = {e["id"] for e in case["evidence"]}
        ids = tuple(row.get("evidence_ids") or ())
        unknown_ids = sorted(set(ids) - known_ids)
        if unknown_ids:
            raise AnnotationError(f"{cid}: evidence ids {unknown_ids} are not in the case")
        spent = row.get("seconds_spent")
        if spent is not None and (
            isinstance(spent, bool) or not isinstance(spent, (int, float)) or spent < 0
        ):
            raise AnnotationError(f"{cid}: seconds_spent must be a non-negative number or null")
        out.append(
            Annotation(
                case_id=cid,
                annotator=who,
                verdict=verdict,
                confidence=confidence,
                evidence_ids=ids,
                prohibited_actions=tuple(row.get("prohibited_actions") or ()),
                required_followup=tuple(row.get("required_followup") or ()),
                note=str(row.get("note") or ""),
                seconds_spent=float(spent) if spent is not None else None,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float | None:
    """Cohen's kappa for two raters over the same items; ``None`` when undefined."""
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe == 1.0:
        return None  # both raters used one label throughout; kappa is undefined
    return (po - pe) / (1 - pe)


def agreement(
    annotations: Iterable[Annotation], cases: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Per-task percent agreement, Cohen's kappa overall and per class, and the queue."""
    by_case: dict[str, list[Annotation]] = defaultdict(list)
    for ann in annotations:
        by_case[ann.case_id].append(ann)
    reviewers = sorted({a.annotator for anns in by_case.values() for a in anns})

    per_task: dict[str, dict[str, Any]] = {}
    queue: list[dict[str, Any]] = []
    pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for cid, anns in sorted(by_case.items()):
        task = cases[cid]["task_id"]
        if len(anns) < 2:
            queue.append(
                {
                    "case_id": cid,
                    "reason": "single_review",
                    "labels": {a.annotator: a.verdict for a in anns},
                }
            )
            continue
        first, second = sorted(anns, key=lambda a: a.annotator)[:2]
        pairs[task].append((first.verdict, second.verdict))
        if first.verdict != second.verdict:
            queue.append(
                {
                    "case_id": cid,
                    "reason": "disagreement",
                    "labels": {first.annotator: first.verdict, second.annotator: second.verdict},
                }
            )

    for task, items in sorted(pairs.items()):
        a, b = [x for x, _ in items], [y for _, y in items]
        labels = sorted(set(a) | set(b))
        per_class = {}
        for label in labels:
            ka = cohen_kappa([x == label for x in a], [y == label for y in b])
            per_class[label] = {
                "kappa": None if ka is None else round(ka, 4),
                "count_a": a.count(label),
                "count_b": b.count(label),
            }
        kappa = cohen_kappa(a, b)
        per_task[task] = {
            "pairs": len(items),
            "percent_agreement": round(sum(x == y for x, y in items) / len(items), 4),
            "kappa": None if kappa is None else round(kappa, 4),
            "per_class": per_class,
            "caveat": "kappa is unstable under class imbalance; read it with the per-class counts",
        }
    timing: dict[str, dict[str, Any]] = {}
    spent_by_task: dict[str, list[float]] = defaultdict(list)
    for cid, anns in by_case.items():
        for ann in anns:
            if ann.seconds_spent is not None:
                spent_by_task[cases[cid]["task_id"]].append(ann.seconds_spent)
    for task, values in sorted(spent_by_task.items()):
        ordered = sorted(values)
        timing[task] = {
            "reviews_timed": len(ordered),
            "median_seconds": ordered[len(ordered) // 2],
            "p90_seconds": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
            "reviewer_hours_per_100_cases_double_reviewed": round(
                2 * 100 * (sum(ordered) / len(ordered)) / 3600, 2
            ),
        }
    return {
        "reviewers": reviewers,
        "cases_reviewed": len(by_case),
        "by_task": per_task,
        # Measured throughput, for scheduling the rest of the gold set. Excludes
        # adjudication time, which the third reviewer should log separately.
        "timing": timing,
        "queue": queue,
        "adjudication_rate": round(
            sum(q["reason"] == "disagreement" for q in queue)
            / max(1, sum(len(v) for v in pairs.values())),
            4,
        ),
    }


# --------------------------------------------------------------------------- #
# Adjudication -> gold
# --------------------------------------------------------------------------- #


def _confidence(anns: Sequence[Annotation]) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    return min((a.confidence for a in anns), key=lambda c: order[c])


def adjudicate(
    annotations: Iterable[Annotation],
    cases: Mapping[str, Mapping[str, Any]],
    decisions: Iterable[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Gold records from reviews and, for disagreements, a third person's decisions."""
    by_case: dict[str, list[Annotation]] = defaultdict(list)
    for ann in annotations:
        by_case[ann.case_id].append(ann)
    decided: dict[str, Mapping[str, Any]] = {}
    for d in decisions:
        decided[str(d.get("case_id"))] = d

    out: list[dict[str, Any]] = []
    for cid, anns in sorted(by_case.items()):
        task = cases[cid]["task_id"]
        anns = sorted(anns, key=lambda a: a.annotator)[:2]
        handles = [a.annotator for a in anns]
        prohibited = sorted({p for a in anns for p in a.prohibited_actions})
        followup = sorted({f for a in anns for f in a.required_followup})
        record: dict[str, Any] = {"case_id": cid, "task_id": task, "annotators": handles}

        if len(anns) == 1:
            a = anns[0]
            record.update(
                label_origin="single-human-reviewed",
                adjudication_status="not-applicable",
                gold={
                    "verdict": a.verdict,
                    "confidence_of_gold": _confidence(anns),
                    "correct_evidence_ids": list(a.evidence_ids),
                    "prohibited_actions": prohibited,
                    "required_followup": followup,
                },
            )
        elif anns[0].verdict == anns[1].verdict:
            record.update(
                label_origin="double-human-reviewed",
                adjudication_status="agreed",
                gold={
                    "verdict": anns[0].verdict,
                    "confidence_of_gold": _confidence(anns),
                    "correct_evidence_ids": sorted(
                        set(anns[0].evidence_ids) & set(anns[1].evidence_ids)
                    ),
                    "prohibited_actions": prohibited,
                    "required_followup": followup,
                },
            )
        elif cid in decided:
            d = decided[cid]
            judge = d.get("adjudicator")
            if not isinstance(judge, str) or not _HANDLE.match(judge):
                raise AnnotationError(f"{cid}: adjudicator {judge!r} is not an anonymised handle")
            if judge in handles:
                raise AnnotationError(f"{cid}: {judge} reviewed this case and cannot adjudicate it")
            verdict = d.get("verdict")
            if verdict not in TASK_LABELS[task]:
                raise AnnotationError(
                    f"{cid}: adjudicated verdict {verdict!r} is not allowed for {task}"
                )
            note = str(d.get("note") or "").strip()
            if not note:
                raise AnnotationError(f"{cid}: an adjudication needs a written reason")
            record.update(
                label_origin="double-human-reviewed",
                adjudication_status="adjudicated",
                adjudication_note=f"{judge}: {note}",
                gold={
                    "verdict": verdict,
                    "confidence_of_gold": "medium" if _confidence(anns) != "low" else "low",
                    "allowed_alternatives": sorted({a.verdict for a in anns} - {verdict}),
                    "correct_evidence_ids": sorted(set(d.get("evidence_ids") or ())),
                    "prohibited_actions": prohibited,
                    "required_followup": followup,
                },
            )
        else:
            record.update(
                label_origin="double-human-reviewed",
                adjudication_status="disputed",
                adjudication_note="reviewers disagree and no adjudication has been recorded",
                gold={
                    "verdict": "unknown" if "unknown" in TASK_LABELS[task] else anns[0].verdict,
                    "confidence_of_gold": "low",
                    "allowed_alternatives": sorted({a.verdict for a in anns}),
                    "prohibited_actions": prohibited,
                },
            )
        out.append(record)
    return out
