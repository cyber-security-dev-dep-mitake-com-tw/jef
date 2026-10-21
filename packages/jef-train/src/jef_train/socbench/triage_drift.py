"""Weekly drift of the five incident-triage answers, as publishable aggregates.

Compares a current questions-mode report with a reference one -- typically
last week's shadow run on private telemetry against the run the model was
checked on. Per task it reports:

* the population stability index (PSI) of the answers: over the predicted
  option for choice and score questions, over ten fixed bins of P(true) for
  noul questions. 0.10 and 0.25 are the conventional "watch" and "shifted"
  lines, used here as descriptive flags, not pre-registered gates;
* the share of questions not answered (missing answers, contract failures);
* calibration status counts;
* when the current pack has confirmed labels and a calibration file is
  given, ``Calibrator.coverage_check`` on them -- otherwise ``NOT_MEASURED``,
  never an assumed pass.

Nothing here changes behaviour: JEF has no automation enabled to switch off.
The ``recommendation`` field says what a person should do (keep everything
human-reviewed; do not promote a candidate) when coverage is violated or a
task has shifted. Output holds counts and rates only, and is refused if the
privacy lint finds anything in it.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .triage import TASK_OF_QUESTION, Pack, privacy_findings

__all__ = ["PSI_SHIFTED", "PSI_WATCH", "DriftError", "drift_report", "psi"]

PSI_WATCH = 0.10
PSI_SHIFTED = 0.25
_EPS = 1e-4
_NOUL_BINS = [i / 10 for i in range(11)]


class DriftError(ValueError):
    """Reports that cannot be compared for drift."""


def psi(reference: Mapping[str, float], current: Mapping[str, float]) -> float:
    """Population stability index between two distributions over the same bins."""
    keys = sorted(set(reference) | set(current))
    total = 0.0
    for k in keys:
        r = max(float(reference.get(k, 0.0)), _EPS)
        c = max(float(current.get(k, 0.0)), _EPS)
        total += (c - r) * math.log(c / r)
    return round(total, 4)


def _shares(values: Sequence[str]) -> dict[str, float]:
    n = len(values)
    return {k: v / n for k, v in Counter(values).items()} if n else {}


def _bin(p: float) -> str:
    i = min(int(p * 10), 9)
    return f"{_NOUL_BINS[i]:.1f}-{_NOUL_BINS[i + 1]:.1f}"


def _answers(report: Mapping[str, Any], qid: str) -> tuple[list[str], int, Counter[str]]:
    vals: list[str] = []
    unanswered = 0
    status: Counter[str] = Counter()
    for row in report["rows"]:
        q = row["questions"].get(qid, {})
        if q.get("status") != "answered":
            unanswered += 1
            continue
        status[str(q.get("calibration_status"))] += 1
        if q.get("kind") == "noul":
            vals.append(_bin(float(q["p_true"])))
        else:
            vals.append(str(q["prediction"]))
    return vals, unanswered, status


def _load(path: Path) -> dict[str, Any]:
    report = json.loads(Path(path).read_text("utf-8"))
    if report.get("mode") != "questions":
        raise DriftError(f"{path}: drift needs a questions-mode report")
    return report


def _coverage(
    report: Mapping[str, Any], pack: Pack, calibration: Path, alpha: float, qid: str
) -> dict[str, Any]:
    from jef_core.calibration import Calibrator

    cal = Calibrator.load(str(calibration))
    task = TASK_OF_QUESTION[qid]
    probs, labels = [], []
    kind = n = task_id = None
    for row in report["rows"]:
        e = pack.expectations.get(row["case_id"])
        q = row["questions"].get(qid, {})
        if not e or e["label_status"] != "confirmed" or q.get("status") != "answered":
            continue
        gold = e["expected"][task]
        if gold == "unknown":
            continue
        keys = list(q["distribution"])
        primary = gold[0] if isinstance(gold, list) else ("true" if gold else "false")
        if primary not in keys:
            continue
        probs.append(np.asarray([q["distribution"][k] for k in keys], dtype=np.float64))
        labels.append(keys.index(primary))
        kind, n, task_id = q["kind"], len(keys), q.get("task_id")
    if not labels or kind is None or n is None:
        return {"status": "NOT_MEASURED", "reason": "no confirmed labels answered"}
    return dict(cal.coverage_check(probs, labels, kind, n, alpha, task_id))


def drift_report(
    reference: Path,
    current: Path,
    *,
    pack: Pack | None = None,
    calibration: Path | None = None,
    alpha: float = 0.10,
) -> dict[str, Any]:
    ref, cur = _load(reference), _load(current)
    if ref["provenance"]["scene"]["sha256"] != cur["provenance"]["scene"]["sha256"]:
        raise DriftError("the two reports asked different scenes, so different questions")
    tasks: dict[str, Any] = {}
    shifted, violated = [], []
    for qid, task in TASK_OF_QUESTION.items():
        r_vals, r_un, _ = _answers(ref, qid)
        c_vals, c_un, c_status = _answers(cur, qid)
        value = psi(_shares(r_vals), _shares(c_vals)) if r_vals and c_vals else None
        flag = (
            None
            if value is None
            else "shifted"
            if value >= PSI_SHIFTED
            else "watch"
            if value >= PSI_WATCH
            else "stable"
        )
        entry: dict[str, Any] = {
            "psi": value,
            "flag": flag,
            "reference_answered": len(r_vals),
            "current_answered": len(c_vals),
            "reference_unanswered_rate": round(r_un / max(1, r_un + len(r_vals)), 4),
            "current_unanswered_rate": round(c_un / max(1, c_un + len(c_vals)), 4),
            "current_calibration_status": dict(c_status),
            "current_distribution": {k: round(v, 4) for k, v in sorted(_shares(c_vals).items())},
            "coverage": (
                _coverage(cur, pack, calibration, alpha, qid)
                if pack is not None and calibration is not None
                else {"status": "NOT_MEASURED", "reason": "no labels or no calibration given"}
            ),
        }
        if flag == "shifted":
            shifted.append(task)
        if entry["coverage"].get("status") == "violated":
            violated.append(task)
        tasks[task] = entry
    report = {
        "evidence_class": "DRIFT AGGREGATES — DESCRIPTIVE",
        "psi_lines": {
            "watch": PSI_WATCH,
            "shifted": PSI_SHIFTED,
            "note": "conventional, not pre-registered",
        },
        "reference_cases": len(ref["rows"]),
        "current_cases": len(cur["rows"]),
        "tasks": tasks,
        "recommendation": (
            "keep every decision human-reviewed and do not promote a candidate: "
            + ", ".join(
                [f"coverage violated for {', '.join(violated)}"] * bool(violated)
                + [f"answers shifted for {', '.join(shifted)}"] * bool(shifted)
            )
            if shifted or violated
            else "no shift flagged; nothing changes (no automation is enabled)"
        ),
    }
    findings = privacy_findings(json.dumps(report, ensure_ascii=False))
    if findings:
        raise DriftError("refusing to emit a drift report with " + "; ".join(findings))
    return report
