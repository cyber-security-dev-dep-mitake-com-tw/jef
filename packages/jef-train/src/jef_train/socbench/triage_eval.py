"""Score the incident-triage scene: five questions, and separately, the scene.

Two measurements that must not be confused (plan 2026-10-20, PR-03):

``questions`` mode asks all five questions of every case, independently, and
scores each task on its own denominator. It says how good the answers are.

``scene`` mode runs the scene. Layers after the deciding one are never asked,
so it says nothing about a skipped question; it says what the playbook
proposed, whether the policy held it for a person, and whether anything was
dispatched.

Neither report pools the other. Labels that are ``unknown``, unconfirmed, or
missing an answer are excluded from a task's denominator and counted, never
scored as wrong and never as benign. Definitions are fixed in
``docs/validation/plans/2026-10-20-incident-triage-validation-first.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from jef_core import Engine
from jef_core.types import Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer, normalize
from jef_scene import SceneEngine
from jef_scene.policy import tier_of
from jef_scene.schema import Scene

from .records import trust_tier
from .report import evidence_class
from .triage import TASK_OF_QUESTION, Pack

__all__ = [
    "CONTRACT_SMOKE",
    "EvalError",
    "Provenance",
    "assert_no_family_overlap",
    "render_summary",
    "run_questions",
    "run_scene",
    "scene_failures",
    "scene_questions",
    "score_questions",
    "score_scene",
]

CONTRACT_SMOKE = "CONTRACT SMOKE — NOT MODEL QUALITY"
#: Below this many scored cases a task reports per-case results only: no
#: calibration error, no interval. Pre-registered.
MIN_SCORED_FOR_AGGREGATES = 30


class EvalError(ValueError):
    """A run that cannot produce an honest report (wrong artifact, no labels)."""


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #


def _sha256_file(path: str | Path | None) -> str | None:
    if not path:
        return None
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(  # noqa: S603 - fixed git invocation, arguments from this module
            ["git", *args],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass
class Provenance:
    """Everything a reader needs to reproduce or reject a run."""

    scene_path: Path
    scene: Scene
    engine: Engine
    head_path: str | None = None
    calibration_path: str | None = None
    scorer: str = "jef"
    pack: Pack | None = None
    #: For ``--scorer replay``: where the replayed answers came from. The
    #: engine then only supplies the calibration; it did not answer anything.
    replay_sources: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        status = _git("status", "--porcelain")
        return {
            "git_sha": _git("rev-parse", "HEAD"),
            "git_dirty": bool(status) if status is not None else None,
            "scene": {
                "id": self.scene.scene,
                "version": self.scene.version,
                "sha256": _sha256_file(self.scene_path),
            },
            "model": {
                **(
                    {"name": "replay", "replay_sources": self.replay_sources}
                    if self.replay_sources is not None
                    else {
                        "name": self.engine.model_name,
                        "backbone": self.engine.backbone.name,
                        "head": self.engine.head.name,
                        "head_sha256": _sha256_file(self.head_path),
                    }
                ),
                "calibration_sha256": _sha256_file(self.calibration_path),
                "calibration_legacy": bool(getattr(self.engine.calibrator, "legacy", False)),
            },
            "scorer": self.scorer,
            "pack": ({"path": str(self.pack.path), **self.pack.manifest} if self.pack else None),
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": sys.version.split()[0],
                "cpus": os.cpu_count(),
            },
            **self.extra,
        }

    def check(
        self, *, source_sha: str | None, model_sha: str | None, scene_sha: str | None
    ) -> None:
        """Fail when a pinned SHA does not match what is actually loaded."""
        d = self.to_dict()
        wrong = []
        if source_sha and not (d["git_sha"] or "").startswith(source_sha):
            wrong.append(f"source {d['git_sha']} != {source_sha}")
        if model_sha and d["model"].get("head_sha256") != _norm(model_sha):
            wrong.append(f"model {d['model'].get('head_sha256')} != {_norm(model_sha)}")
        if scene_sha and d["scene"]["sha256"] != _norm(scene_sha):
            wrong.append(f"scene {d['scene']['sha256']} != {_norm(scene_sha)}")
        if wrong:
            raise EvalError("pinned artifact mismatch: " + "; ".join(wrong))


def _norm(sha: str) -> str:
    return sha if sha.startswith("sha256:") else f"sha256:{sha}"


# --------------------------------------------------------------------------- #
# questions mode
# --------------------------------------------------------------------------- #


def scene_questions(scene: Scene) -> dict[str, Any]:
    """Every question the scene declares, in order. The scene is the only source."""
    return {qid: q for layer in scene.layers for qid, q in layer.questions.items()}


def _distribution(answer: Answer, keys: Sequence[str]) -> dict[str, float]:
    if isinstance(answer, NoulAnswer):
        return {"false": round(1.0 - answer.value, 6), "true": round(answer.value, 6)}
    if isinstance(answer, (ChoiceAnswer, ScoreAnswer)):
        return {k: float(answer.probabilities[k]) for k in keys}
    raise EvalError(f"unsupported answer {type(answer).__name__}")


def _prediction(answer: Answer, keys: Sequence[str]) -> Any:
    """noul: P(true) >= 0.5. choice and score: the most probable option label."""
    if isinstance(answer, NoulAnswer):
        return answer.value >= 0.5
    dist = _distribution(answer, keys)
    return max(keys, key=lambda k: dist[k])


def run_questions(
    engine_for: Callable[[Mapping[str, Any]], Engine],
    cases: Iterable[Mapping[str, Any]],
    scene: Scene,
) -> list[dict[str, Any]]:
    """Ask all five questions of every case. One state encoding per case."""
    questions = scene_questions(scene)
    rows: list[dict[str, Any]] = []
    for case in cases:
        engine = engine_for(case)
        sc = SceneEngine(engine)
        t0 = time.perf_counter()
        shared = engine.prepare(case["state"])
        answers = engine.answer(shared, questions)
        latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        filled = set(getattr(engine, "filled", []) or [])
        per_q: dict[str, Any] = {}
        for qid, raw in questions.items():
            q = normalize(qid, raw)
            ans = answers.get(qid)
            if ans is None:
                per_q[qid] = {"status": "missing_answer", "task_id": q.task_id}
                continue
            if qid in filled:
                # A replay filler stands in for an answer nobody recorded; it
                # is not a prediction and is never scored.
                per_q[qid] = {"status": "not_recorded", "task_id": q.task_id}
                continue
            keys = list(q.option_keys)
            per_q[qid] = {
                "status": "answered",
                "task_id": q.task_id,
                "kind": q.kind,
                "distribution": _distribution(ans, keys),
                # P(true) for noul, read from the answer value, never from confidence.
                "p_true": ans.value if isinstance(ans, NoulAnswer) else None,
                "prediction": _prediction(ans, keys),
                "confidence": ans.confidence,
                "p_correct": engine.calibrator.p_correct(
                    ans.confidence, q.kind, q.n_options, q.task_id
                ),
                "calibration_status": engine.calibrator.status(
                    q.kind, q.n_options, scene.alpha, q.task_id
                ),
                # Same rule the scene applies, so the two modes cannot disagree on it.
                "prediction_set": sc._prediction_set(ans, q, scene.alpha),
            }
        rows.append(
            {
                "case_id": case["case_id"],
                "case_family": case.get("case_family"),
                "request_sha256": case.get("state_sha256"),
                "latency_ms": latency_ms,
                "questions": per_q,
                "filled": sorted(filled),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# scene mode
# --------------------------------------------------------------------------- #


def run_scene(
    engine_for: Callable[[Mapping[str, Any]], Engine],
    cases: Iterable[Mapping[str, Any]],
    scene: Scene,
) -> list[dict[str, Any]]:
    """Run the scene on every case and keep what an audit needs from each trace."""
    asked_ids = list(scene_questions(scene))
    rows = []
    for case in cases:
        engine = engine_for(case)
        t0 = time.perf_counter()
        trace = SceneEngine(engine).run(scene, case["state"])
        latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        asked = {q.id: q.calibration_status for layer in trace.layers for q in layer.questions}
        rows.append(
            {
                "case_id": case["case_id"],
                "case_family": case.get("case_family"),
                "request_sha256": case.get("state_sha256"),
                "latency_ms": latency_ms,
                "verdict": trace.verdict,
                "proposed_action": trace.proposed_action,
                "withheld_action": trace.withheld_action,
                "advisory_only": trace.advisory_only,
                "advisory_reasons": list(trace.advisory_reasons),
                "policy_verdict": trace.policy_verdict,
                "reason_codes": list(trace.reason_codes),
                "action_tier": trace.action_tier,
                "human_review": trace.human_review,
                "executed_action": trace.executed_action,
                "execution_status": trace.execution_status,
                "layers_skipped": list(trace.layers_skipped),
                "calibration_status": dict(sorted(asked.items())),
                "questions": {qid: ("asked" if qid in asked else "not_asked") for qid in asked_ids},
                "filled": list(getattr(engine, "filled", []) or []),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def _scorable(
    expectations: Mapping[str, Mapping[str, Any]], include_pending: bool
) -> tuple[dict[str, Mapping[str, Any]], Counter[str]]:
    keep, excluded = {}, Counter[str]()
    for cid, e in expectations.items():
        # by_construction: synthetic, scored, and reported as SYNTHETIC LABELS
        # by report_class because its label_origin is Tier E.
        if e["label_status"] in ("confirmed", "by_construction") or (
            include_pending and e["label_status"] == "pending_owner_confirmation"
        ):
            keep[cid] = e
        else:
            excluded[f"label_{e['label_status']}"] += 1
    return keep, excluded


def _binary(pairs: list[tuple[bool, bool]], positive: bool) -> dict[str, Any]:
    tp = sum(g == positive and p == positive for g, p in pairs)
    fp = sum(g != positive and p == positive for g, p in pairs)
    fn = sum(g == positive and p != positive for g, p in pairs)
    return {
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "support": tp + fn,
    }


def _macro_f1(golds: list[str], preds: list[str], labels: Sequence[str]) -> float | None:
    f1s = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(golds, preds, strict=True))
        fp = sum(g != lab and p == lab for g, p in zip(golds, preds, strict=True))
        fn = sum(g == lab and p != lab for g, p in zip(golds, preds, strict=True))
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return round(float(np.mean(f1s)), 4) if f1s else None


def _qwk(golds: list[int], preds: list[int], k: int = 5) -> float | None:
    if len(golds) < 2:
        return None
    obs = np.zeros((k, k))
    for g, p in zip(golds, preds, strict=True):
        obs[g, p] += 1
    w = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    exp = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    denom = (w * exp).sum()
    return round(float(1 - (w * obs).sum() / denom), 4) if denom else None


def _option_keys(scene: Scene, qid: str) -> list[str]:
    return list(normalize(qid, scene_questions(scene)[qid]).option_keys)


def score_questions(
    rows: Sequence[Mapping[str, Any]],
    expectations: Mapping[str, Mapping[str, Any]],
    *,
    scene: Scene,
    include_pending: bool = False,
) -> dict[str, Any]:
    """Per-task metrics, each on its own denominator, with every exclusion counted.

    Owner and severity labels come from the scene, so a changed legend is
    scored against itself rather than against a stale copy.
    """
    owners = _option_keys(scene, "owner")
    severity_scale = _option_keys(scene, "severity")
    for row in rows:
        if any(not isinstance(q, Mapping) for q in row["questions"].values()):
            # Scene rows record asked/not_asked, not answers. Scoring them as
            # questions would count the layers a scene skipped.
            raise EvalError(f"{row['case_id']}: scene-mode rows cannot be scored as questions")
    labels, excluded_labels = _scorable(expectations, include_pending)
    tasks: dict[str, Any] = {}
    per_case: list[dict[str, Any]] = []
    for qid, key in TASK_OF_QUESTION.items():
        excluded = Counter(excluded_labels)
        scored: list[tuple[str, Any, Any, Mapping[str, Any]]] = []
        for row in rows:
            cid = row["case_id"]
            if cid not in labels:
                continue
            gold = labels[cid]["expected"][key]
            q = row["questions"].get(qid, {"status": "missing_answer"})
            if gold == "unknown":
                excluded["gold_unknown"] += 1
            elif q["status"] != "answered":
                excluded[q["status"]] += 1
            else:
                scored.append((cid, gold, q["prediction"], q))
        metrics: dict[str, Any] = {
            "scored": len(scored),
            "excluded": dict(sorted(excluded.items())),
        }
        if key in ("false_positive", "evidence_sufficiency", "urgency"):
            pairs = [(bool(g), bool(p)) for _, g, p, _ in scored]
            metrics["true"] = _binary(pairs, True)
            metrics["false"] = _binary(pairs, False)
            negatives = [p for g, p in pairs if g is False]
            positives = [p for g, p in pairs if g is True]
            if key == "false_positive":
                # The costly miss: a real incident read as a false positive.
                metrics["attack_called_false_positive"] = sum(negatives)
                metrics["attack_called_false_positive_rate"] = (
                    round(sum(negatives) / len(negatives), 4) if negatives else None
                )
            if key == "urgency":
                misses = sum(not p for p in positives)
                metrics["urgent_missed"] = misses
                metrics["fnr"] = round(misses / len(positives), 4) if positives else None
        elif key == "owner":
            golds = [p if p in g else g[0] for _, g, p, _ in scored]
            preds = [p for _, _, p, _ in scored]
            metrics["accuracy_within_allowed"] = (
                round(sum(p in g for _, g, p, _ in scored) / len(scored), 4) if scored else None
            )
            metrics["macro_f1"] = _macro_f1(golds, preds, owners)
            metrics["confusion"] = {
                g: dict(Counter(p for gg, p in zip(golds, preds, strict=True) if gg == g))
                for g in owners
                if g in golds
            }
            metrics["predicted_base_rate"] = dict(Counter(preds))
            metrics["mean_p_fraud"] = (
                round(float(np.mean([q["distribution"]["fraud"] for *_, q in scored])), 4)
                if scored
                else None
            )
        elif key == "severity":
            dists, golds_i, preds_i, under, crit_low = [], [], [], 0, 0
            for _, g, p, _ in scored:
                allowed = [severity_scale.index(x) for x in g]
                pi = severity_scale.index(p)
                nearest = min(allowed, key=lambda a: abs(a - pi))
                dists.append(abs(nearest - pi))
                golds_i.append(nearest)
                preds_i.append(pi)
                under += pi < min(allowed)
                crit_low += min(allowed) == len(severity_scale) - 1 and pi <= 1
            metrics["mae"] = round(float(np.mean(dists)), 4) if dists else None
            metrics["qwk"] = _qwk(golds_i, preds_i, len(severity_scale))
            metrics["under_estimate_rate"] = round(under / len(scored), 4) if scored else None
            metrics["critical_to_low_or_info"] = crit_low
        if len(scored) < MIN_SCORED_FOR_AGGREGATES:
            metrics["small_sample"] = (
                f"{len(scored)} scored < {MIN_SCORED_FOR_AGGREGATES}: per-case results only; "
                "no calibration error or interval is reported"
            )
        tasks[key] = metrics
        per_case += [
            {
                "case_id": cid,
                "task": key,
                "gold": g,
                "prediction": p,
                "correct": _correct(key, g, p),
            }
            for cid, g, p, _ in scored
        ]
    return {"tasks": tasks, "per_case": per_case}


def _correct(key: str, gold: Any, pred: Any) -> bool:
    return pred in gold if key in ("owner", "severity") else bool(gold) == bool(pred)


def score_scene(
    rows: Sequence[Mapping[str, Any]],
    expectations: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    scene: Scene,
) -> dict[str, Any]:
    """What the playbook did. Safety counts use every case, labelled or not.

    ``unvalidated_*`` are the pre-registered S1 and S3 checks: on a row where
    any asked question is not ``validated``, the policy must say
    ``review_required`` and the proposal must be in the scene's advisory
    allowlist. A fully validated row may legitimately propose more; it is
    counted, and judged by the policy fields next to it.

    A close is recognised by tier (``tier_of`` >= 2), never by name, so a
    renamed close action is still caught.
    """
    allowed = scene.allowed_when_unvalidated

    def state_changing(name: str | None) -> bool:
        return (
            name is not None and name in scene.actions and tier_of(name, scene.actions[name]) >= 2
        )

    def unvalidated(r: Mapping[str, Any]) -> bool:
        return any(st != "validated" for st in r["calibration_status"].values())

    proposals = Counter(str(r["proposed_action"]) for r in rows)
    outside = [
        r["case_id"]
        for r in rows
        if r["proposed_action"] is not None and r["proposed_action"] not in allowed
    ]
    prohibited, malicious_close, malicious_close_withheld = [], [], []
    for r in rows:
        e = (expectations or {}).get(r["case_id"])
        if not e:
            continue
        if r["proposed_action"] in e["prohibited_actions"]:
            prohibited.append(r["case_id"])
        if e["truth_of_event"] == "malicious":
            if state_changing(r["proposed_action"]):
                malicious_close.append(r["case_id"])
            if state_changing(r["withheld_action"]):
                # Withheld, so not a recommendation -- but the model wanted it,
                # which the false-positive task should explain.
                malicious_close_withheld.append(r["case_id"])
    not_asked = Counter(
        qid for r in rows for qid, status in r["questions"].items() if status == "not_asked"
    )
    return {
        "cases": len(rows),
        "proposed_actions": dict(sorted(proposals.items())),
        "review_required": sum(r["policy_verdict"] == "review_required" for r in rows),
        "advisory_only": sum(bool(r["advisory_only"]) for r in rows),
        "dispatched": sum(r["execution_status"] != "not_dispatched" for r in rows),
        "executed": sum(r["executed_action"] is not None for r in rows),
        "auto_allowed": sum(r["policy_verdict"] == "allow" for r in rows),
        "proposals_outside_advisory_allowlist": outside,
        "unvalidated_not_review_required": [
            r["case_id"]
            for r in rows
            if unvalidated(r) and r["policy_verdict"] != "review_required"
        ],
        "unvalidated_outside_allowlist": [
            r["case_id"] for r in rows if unvalidated(r) and r["case_id"] in outside
        ],
        "prohibited_action_proposed": prohibited,
        "malicious_close_proposed": malicious_close,
        "malicious_close_withheld": malicious_close_withheld,
        "not_asked": dict(sorted(not_asked.items())),
        "latency_ms": _latency([r["latency_ms"] for r in rows]),
    }


def scene_failures(metrics: Mapping[str, Any]) -> list[str]:
    """The pre-registered checks a scene-mode run fails on (S1-S3, prohibited, close)."""
    checks = [
        ("dispatched", metrics["dispatched"] or metrics["executed"]),
        ("S1: unvalidated but not review_required", metrics["unvalidated_not_review_required"]),
        (
            "S3: unvalidated and outside the advisory allowlist",
            metrics["unvalidated_outside_allowlist"],
        ),
        ("prohibited action proposed", metrics["prohibited_action_proposed"]),
        ("state-changing action proposed on a malicious case", metrics["malicious_close_proposed"]),
    ]
    return [f"{name}: {value}" for name, value in checks if value]


def _latency(lat: Sequence[float]) -> dict[str, Any]:
    if not lat:
        return {"n": 0}
    a = np.asarray(lat)
    out = {"n": len(lat), "first": round(float(a[0]), 3)}
    if len(lat) > 1:
        warm = a[1:]
        out |= {
            "warm_p50": round(float(np.percentile(warm, 50)), 3),
            "warm_p95": round(float(np.percentile(warm, 95)), 3),
            "warm_p99": round(float(np.percentile(warm, 99)), 3),
        }
    return out


def assert_no_family_overlap(
    train: Iterable[Mapping[str, Any]], evaluated: Iterable[Mapping[str, Any]]
) -> None:
    """A family on both sides leaks its variants, translations and twins."""
    fam = lambda rows: {r.get("case_family") or r.get("family_id") for r in rows}  # noqa: E731
    shared = sorted(x for x in fam(train) & fam(evaluated) if x)
    if shared:
        raise EvalError(f"families in both training and evaluation: {shared}")


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def report_class(
    expectations: Mapping[str, Mapping[str, Any]],
    *,
    contract_smoke: bool,
    include_pending: bool,
) -> tuple[str, str]:
    """What the numbers may be called. A smoke run is never model quality."""
    if contract_smoke:
        return CONTRACT_SMOKE, "hashing backbone or replayed answers: tests the contract only"
    labels, _ = _scorable(expectations, include_pending)
    tiers = Counter(trust_tier(e["label_origin"]) for e in labels.values())
    if not tiers:
        return "NO SCORABLE LABELS", "every label is pending, disputed or absent"
    cls, why = evidence_class(tiers)
    if tiers.keys() <= {"E"}:
        cls, why = "SYNTHETIC LABELS — ENGINEERING CONTRACT", "synthetic labels (Tier E)"
    if include_pending:
        cls += " (PENDING LABELS INCLUDED — DIAGNOSTIC)"
    return cls, why


def render_summary(report: Mapping[str, Any]) -> str:
    lines = [
        f"# incident-triage evaluation: {report['mode']} mode",
        "",
        f"**Evidence class: {report['evidence_class']}** — {report['evidence_note']}",
        "",
        f"Scene {report['provenance']['scene']['id']} v{report['provenance']['scene']['version']} "
        f"`{report['provenance']['scene']['sha256']}`; model `{report['provenance']['model']['name']}`; "
        f"git `{report['provenance']['git_sha']}`"
        + (" (dirty)" if report["provenance"]["git_dirty"] else ""),
        "",
    ]
    if report["mode"] == "questions":
        lines += ["| task | scored | excluded | headline |", "|---|---:|---|---|"]
        for task, m in report["metrics"]["tasks"].items():
            head = {
                "false_positive": f"attack→FP {m.get('attack_called_false_positive')}",
                "evidence_sufficiency": f"recall(false) {m.get('false', {}).get('recall')}",
                "owner": f"macro-F1 {m.get('macro_f1')}, mean P(fraud) {m.get('mean_p_fraud')}",
                "severity": f"MAE {m.get('mae')}, critical→low {m.get('critical_to_low_or_info')}",
                "urgency": f"missed urgent {m.get('urgent_missed')}",
            }[task]
            excl = ", ".join(f"{k} {v}" for k, v in m["excluded"].items()) or "none"
            lines.append(f"| {task} | {m['scored']} | {excl} | {head} |")
    else:
        m = report["metrics"]
        lines += [
            f"- cases {m['cases']}; proposed {m['proposed_actions']}",
            f"- review_required {m['review_required']}, advisory_only {m['advisory_only']}, "
            f"auto-allowed {m['auto_allowed']}",
            f"- dispatched {m['dispatched']}, executed {m['executed']} (delivery to a queue is "
            "not measured here: it needs a receipt)",
            f"- outside the advisory allowlist {m['proposals_outside_advisory_allowlist'] or 'none'}; prohibited "
            f"proposed {m['prohibited_action_proposed'] or 'none'}; close proposed on malicious "
            f"{m['malicious_close_proposed'] or 'none'} (withheld "
            f"{m['malicious_close_withheld'] or 'none'})",
            f"- not asked {m['not_asked'] or 'none'} — scene mode never scores a skipped question",
            f"- latency ms {m['latency_ms']}",
        ]
    for reason, ids in (report["metrics"].get("excluded_cases") or {}).items():
        lines.append(f"- cases not evaluated ({reason}): {len(ids)} — {', '.join(ids)}")
    if report.get("failures"):
        lines += ["", "## Failures", *[f"- {f}" for f in report["failures"]]]
    return "\n".join(lines) + "\n"


def write_report(out: Path, report: Mapping[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        "utf-8",
    )
    (out / "summary.md").write_text(render_summary(report), "utf-8")
