"""Blind head-to-head on the five incident-triage tasks: paired, clustered, pre-registered.

Input is one ``jef-bench triage evaluate --mode questions`` report per system
(JEF candidates, baselines, remote systems), all on the same pack, plus the
pack's lock. The comparison is only made where it is fair:

* the same pack bytes, the same scene, and the locked split (``test`` by
  default) -- a report from different inputs is refused;
* *paired*: a task is compared on the cases every system answered for it;
  what a system failed to answer is listed per system, never dropped quietly;
* *clustered*: confidence intervals resample whole families
  (``metrics.cluster_bootstrap``), so twenty variants of one scenario count
  as one;
* the criteria are the ones fixed in
  ``docs/validation/plans/2026-10-21-soc-head-to-head-preregistration.md``
  before any number existed. ``verdicts`` applies them; nothing else picks a
  winner.

Two leaderboards are kept apart: the bare decision quality of each system
(this module), and what a scene did end to end (``--mode scene`` reports),
so JEF's conservative policy is never counted as model accuracy.

Unless every scored label is double-human-reviewed and confirmed, the lock
verifies, and each compared task has at least ``MIN_FAMILIES`` families, the
report is stamped ``NOT A CLAIM`` and ``verdicts`` says why.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .metrics import cluster_bootstrap
from .triage import TASK_OF_QUESTION, Pack
from .triage_eval import _macro_f1
from .triage_synth import verify_triage_lock

__all__ = [
    "MIN_FAMILIES",
    "NON_INFERIORITY_MARGIN",
    "PRIMARY",
    "CompareError",
    "compare",
]

#: Families with gold a task needs before any verdict on it is more than descriptive.
MIN_FAMILIES = 30
#: Non-inferiority margin on each task's primary metric (absolute, same scale).
NON_INFERIORITY_MARGIN = 0.05
#: Review budgets for the priority ranking (share of cases an analyst reads first).
BUDGETS = (0.10, 0.20)
SEVERITY = ["資訊", "低", "中", "高", "危急"]

#: The pre-registered primary metric per task, and whether higher is better.
PRIMARY: dict[str, tuple[str, bool]] = {
    "false_positive": ("malicious_recall", True),  # 1 - P(an attack is called benign)
    "evidence_sufficiency": ("insufficient_recall", True),
    "owner": ("macro_f1", True),
    "severity": ("mae", False),
    "urgency": ("urgent_recall", True),
}


class CompareError(ValueError):
    """Reports that cannot be compared fairly."""


def _recall(golds: Sequence[Any], preds: Sequence[Any], cls: bool) -> float | None:
    hits = [p is cls for g, p in zip(golds, preds, strict=True) if g is cls]
    return None if not hits else sum(hits) / len(hits)


def _stat(
    task: str, golds: Sequence[Any], preds: Sequence[Any]
) -> Callable[[NDArray[np.int64]], float | None]:
    def stat(idx: NDArray[np.int64]) -> float | None:
        return _primary_value(task, [golds[j] for j in idx], [preds[j] for j in idx])

    return stat


def _diff(
    task: str, golds: Sequence[Any], a: Sequence[Any], b: Sequence[Any], higher: bool
) -> Callable[[NDArray[np.int64]], float | None]:
    """Paired difference on resampled families; positive means ``a`` is better than ``b``."""

    def stat(idx: NDArray[np.int64]) -> float | None:
        g = [golds[j] for j in idx]
        va = _primary_value(task, g, [a[j] for j in idx])
        vb = _primary_value(task, g, [b[j] for j in idx])
        if va is None or vb is None:
            return None
        return (va - vb) if higher else (vb - va)

    return stat


def _primary_value(task: str, golds: Sequence[Any], preds: Sequence[Any]) -> float | None:
    if not golds:
        return None
    if task in ("false_positive", "evidence_sufficiency"):
        # false_positive: gold False is a real event; recall = not called benign.
        # evidence_sufficiency: gold False is too little evidence; recall of that.
        return _recall(golds, preds, False)
    if task == "urgency":
        return _recall(golds, preds, True)
    if task == "owner":
        # As the harness scores it: a prediction among the acceptable owners
        # counts as that owner, otherwise the primary owner is the gold.
        resolved = [p if p in g else g[0] for g, p in zip(golds, preds, strict=True)]
        labels = sorted(set(resolved) | set(preds))
        return _macro_f1(resolved, list(preds), labels)
    if task == "severity":
        errs = [
            min(abs(SEVERITY.index(p) - SEVERITY.index(x)) for x in g)
            for g, p in zip(golds, preds, strict=True)
        ]
        return statistics.mean(errs)
    raise CompareError(f"unknown task {task}")


def _critical_misses(golds: Sequence[Any], preds: Sequence[Any], ids: Sequence[str]) -> list[str]:
    """As the harness counts them: only 危急 is acceptable, and 低 or 資訊 was predicted."""
    return [
        cid
        for g, p, cid in zip(golds, preds, ids, strict=True)
        if min(SEVERITY.index(x) for x in g) == len(SEVERITY) - 1 and p in ("資訊", "低")
    ]


def _attacks_called_benign(
    golds: Sequence[Any], preds: Sequence[Any], ids: Sequence[str]
) -> list[str]:
    return [cid for g, p, cid in zip(golds, preds, ids, strict=True) if g is False and p is True]


def _load(path: Path) -> dict[str, Any]:
    report = json.loads(Path(path).read_text("utf-8"))
    if report.get("mode") != "questions":
        raise CompareError(
            f"{path}: a questions-mode report is needed (scene reports are a separate leaderboard)"
        )
    return report


def _name(report: Mapping[str, Any], path: Path) -> str:
    model = report["provenance"].get("model", {})
    return str(model.get("name") if model.get("name") not in (None, "jef") else path.parent.name)


def compare(
    pack: Pack,
    lock: Mapping[str, Any],
    reports: Sequence[Path],
    *,
    split: str = "test",
    reference: int = 0,
    n_boot: int = 1000,
    seed: int = 1337,
) -> dict[str, Any]:
    """The comparison report. ``reference`` indexes the system others are paired against."""
    if len(reports) < 2:
        raise CompareError("a comparison needs at least two systems")
    if not 0 <= reference < len(reports):
        raise CompareError(f"--reference {reference} is not one of the {len(reports)} reports")
    exps = [pack.expectations[c["case_id"]] for c in pack.cases]
    lock_problems = verify_triage_lock(lock, pack.cases, exps)
    if split not in lock["splits"]:
        raise CompareError(f"the lock has no split {split!r}")
    split_ids = set(lock["splits"][split])
    family = {c["case_id"]: c["case_family"] for c in pack.cases}
    loaded = [(Path(p), _load(Path(p))) for p in reports]
    files = {json.dumps(r["provenance"]["pack"]["files"], sort_keys=True) for _, r in loaded}
    scenes = {r["provenance"]["scene"]["sha256"] for _, r in loaded}
    if len(files) != 1 or json.loads(next(iter(files))) != pack.manifest.get("files"):
        raise CompareError("the reports were not all made on this pack's bytes")
    if len(scenes) != 1:
        raise CompareError("the reports used different scenes, so different questions")
    names = [_name(r, p) for p, r in loaded]
    if len(set(names)) != len(names):
        names = [f"{n}#{i}" for i, n in enumerate(names)]

    scorable = {
        cid: e
        for cid, e in pack.expectations.items()
        if cid in split_ids and e["label_status"] in ("confirmed", "by_construction")
    }
    tier_a = all(
        e["label_origin"] == "double-human-reviewed" and e["label_status"] == "confirmed"
        for e in scorable.values()
    ) and bool(scorable)

    rows_by_system = [{row["case_id"]: row for row in r["rows"]} for _, r in loaded]
    tasks: dict[str, Any] = {}
    for qid, task in TASK_OF_QUESTION.items():
        gold_ids = [cid for cid, e in sorted(scorable.items()) if e["expected"][task] != "unknown"]
        answered = [
            {
                cid
                for cid in gold_ids
                if rows.get(cid, {}).get("questions", {}).get(qid, {}).get("status") == "answered"
            }
            for rows in rows_by_system
        ]
        common = sorted(set(gold_ids).intersection(*answered))
        golds = [scorable[cid]["expected"][task] for cid in common]
        preds = [
            [rows[cid]["questions"][qid]["prediction"] for cid in common] for rows in rows_by_system
        ]
        groups = [family[cid] for cid in common]
        n_families = len(set(groups))
        metric, higher = PRIMARY[task]

        systems = {}
        for i, name in enumerate(names):
            ci = (
                cluster_bootstrap(groups, _stat(task, golds, preds[i]), n_boot=n_boot, seed=seed)
                if common
                else None
            )
            entry: dict[str, Any] = {
                metric: None if ci is None else ci["point"],
                "ci95": None if ci is None else [ci["low"], ci["high"]],
                "unanswered": sorted(set(gold_ids) - answered[i]),
            }
            if task == "severity":
                entry["critical_to_low"] = _critical_misses(golds, preds[i], common)
            if task == "false_positive":
                entry["attack_called_benign"] = _attacks_called_benign(golds, preds[i], common)
            systems[name] = entry
        paired = {}
        for i, name in enumerate(names):
            if i == reference or not common:
                continue
            diff = _diff(task, golds, preds[i], preds[reference], higher)
            d = cluster_bootstrap(groups, diff, n_boot=n_boot, seed=seed)
            paired[name] = {"better_than_reference_by": d["point"], "ci95": [d["low"], d["high"]]}
        tasks[task] = {
            "primary_metric": metric,
            "higher_is_better": higher,
            "cases": len(common),
            "families": n_families,
            "support": "ok" if n_families >= MIN_FAMILIES else "insufficient",
            "systems": systems,
            "paired_vs_reference": paired,
        }

    ranking = _ranking(scorable, rows_by_system, names, family)
    cost = {
        name: _cost(r, rows)
        for name, (_, r), rows in zip(names, loaded, rows_by_system, strict=True)
    }
    reasons = []
    if lock_problems:
        reasons.append("the pack is not the locked bytes: " + "; ".join(lock_problems))
    if not tier_a:
        reasons.append("not every scored label is double-human-reviewed and confirmed (Tier A)")
    short = [t for t, v in tasks.items() if v["support"] != "ok"]
    if short:
        reasons.append(f"fewer than {MIN_FAMILIES} families with gold for: {', '.join(short)}")
    claim = not reasons
    return {
        "evidence_class": "PRE-REGISTERED COMPARISON" if claim else "NOT A CLAIM",
        "not_a_claim_because": reasons,
        "preregistration": "docs/validation/plans/2026-10-21-soc-head-to-head-preregistration.md",
        "split": split,
        "reference": names[reference],
        "systems": names,
        "pack_files": pack.manifest.get("files"),
        "tasks": tasks,
        "ranking": ranking,
        "cost": cost,
        "verdicts": _verdicts(tasks, names, reference) if claim else None,
    }


def _ranking(
    scorable: Mapping[str, Mapping[str, Any]],
    rows_by_system: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    family: Mapping[str, str],
) -> dict[str, Any]:
    """Recall of confirmed attacks in the first 10 % / 20 % by priority = 1 - P(false positive)."""
    out: dict[str, Any] = {}
    attacks = {cid for cid, e in scorable.items() if e["truth_of_event"] == "malicious"}
    for name, rows in zip(names, rows_by_system, strict=True):
        scored = []
        for cid in scorable:
            q = rows.get(cid, {}).get("questions", {}).get("is_false_positive", {})
            if q.get("status") == "answered" and q.get("p_true") is not None:
                scored.append((1.0 - float(q["p_true"]), cid))
        scored.sort(key=lambda x: (-x[0], x[1]))
        entry: dict[str, Any] = {"ranked": len(scored), "attacks": len(attacks)}
        for b in BUDGETS:
            k = max(1, round(len(scored) * b)) if scored else 0
            top = {cid for _, cid in scored[:k]}
            entry[f"recall_at_{round(b * 100)}pct"] = (
                None if not attacks else round(len(top & attacks) / len(attacks), 4)
            )
        out[name] = entry
    return out


def _cost(report: Mapping[str, Any], rows: Mapping[str, Any]) -> dict[str, Any]:
    lat = sorted(float(r["latency_ms"]) for r in rows.values() if r.get("latency_ms") is not None)
    model = report["provenance"].get("model", {})
    failures = report.get("metrics", {}).get("remote_contract_failures", {})

    def pct(p: float) -> float | None:  # nearest rank
        return None if not lat else lat[max(0, math.ceil(p * len(lat)) - 1)]

    return {
        "latency_ms_p50": pct(0.5),
        "latency_ms_p95": pct(0.95),
        "price_per_1k_cases_usd": model.get("price_per_1k_cases_usd"),
        "contract_failures": failures.get("count", 0),
        "hardware": report["provenance"].get("hardware"),
    }


def _verdicts(tasks: Mapping[str, Any], names: Sequence[str], reference: int) -> dict[str, Any]:
    """The pre-registered reading: non-inferior / superior / neither, per task and system."""
    out: dict[str, Any] = {}
    for task, t in tasks.items():
        per = {}
        for name, d in t["paired_vs_reference"].items():
            low = d["ci95"][0]
            if low is None:
                per[name] = "undetermined"
            elif low > 0:
                per[name] = "superior_to_reference"
            elif low > -NON_INFERIORITY_MARGIN:
                per[name] = "non_inferior_to_reference"
            else:
                per[name] = "not_shown_non_inferior"
        out[task] = per
    return out
