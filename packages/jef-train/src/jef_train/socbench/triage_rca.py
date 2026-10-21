"""The incident-triage root-cause probes (R1-R5, S1, L) as a library.

First written as the one-off study ``docs/validation/research/triage-rca-v1``
(criteria fixed 2026-10-20). The v5 plan asks for the same probes on every
model version, so they live here and the study script calls them; its
``results.json`` is unchanged by the move.

R1 fraud bias, R2 option-order sensitivity, R3 language dependence, R4 length
dependence, R5 label prior on content-free states, S1 the severity errors that
matter (critical read as low, benign read as high), L the Lab canaries
replayed live. Thresholds are the pre-registered ones from the study.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping
from typing import Any

import numpy as np
from jef_core import Engine
from jef_core.types import ChoiceAnswer
from jef_scene import Scene

from .triage import Pack
from .triage_eval import run_questions, scene_questions

__all__ = ["NEUTRAL", "label_prior", "rca_probes", "rotations", "spearman"]

#: Content-free states: anything a model answers here is its label prior.
NEUTRAL = ["告警。", "Alert.", "系統通知。", "Notification."]
SEVERITY_ORDER = ["資訊", "低", "中", "高", "危急"]


def rotations(question: Any) -> list[dict[str, Any]]:
    """The owner question with its options in each of the five rotations."""
    raw = question.model_dump(exclude_none=True)
    items = list(raw["criteria"].items())
    return [{**raw, "criteria": dict(items[i:] + items[:i])} for i in range(len(items))]


def spearman(x: list[float], y: list[float]) -> float:
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _owner(engine: Engine, shared: Any, question: Any) -> ChoiceAnswer:
    a = engine.answer(shared, {"owner": question})["owner"]
    if not isinstance(a, ChoiceAnswer):
        raise TypeError("the owner question must be a choice question")
    return a


def label_prior(engine: Engine, scene: Scene) -> dict[str, Any]:
    """R5: mean P(fraud) on content-free states; a prior towards fraud above 0.30."""
    owner_q = scene_questions(scene)["owner"]
    probes: dict[str, dict[str, Any]] = {}
    for text in NEUTRAL:
        a = _owner(engine, engine.prepare(text), owner_q)
        probes[text] = {
            "p_fraud": round(a.probabilities["fraud"], 4),
            "top": a.choice,
            "dist": a.probabilities,
        }
    r5: dict[str, Any] = {
        "probes": probes,
        "mean_p_fraud": round(statistics.mean(p["p_fraud"] for p in probes.values()), 4),
    }
    r5["prior_towards_fraud"] = r5["mean_p_fraud"] > 0.30
    return r5


def rca_probes(engine: Engine, scene: Scene, dev: Pack, canary: Pack) -> dict[str, Any]:
    """R1-R5, S1 and L on a synthetic dev pack and the canary pack."""
    owner_q = scene_questions(scene)["owner"]

    # Every state once; five rotations of the owner question on the same encoding.
    states = {c["case_id"]: c for c in [*dev.cases, *canary.cases]}
    per_case: dict[str, dict[str, Any]] = {}
    for cid, c in states.items():
        shared = engine.prepare(c["state"])
        rot = []
        for r in rotations(owner_q):
            a = _owner(engine, shared, r)
            rot.append(
                {"order": list(r["criteria"]), "p_fraud": a.probabilities["fraud"], "top": a.choice}
            )
        per_case[cid] = {"length": len(c["state"]), "language": c["language"], "rotations": rot}

    # R1 -- canonical order is rotation 0.
    non_fraud = [
        c["case_id"]
        for c in dev.cases
        if dev.expectations[c["case_id"]]["expected"]["owner"] != "unknown"
        and "fraud" not in dev.expectations[c["case_id"]]["expected"]["owner"]
    ]
    p_fraud = [per_case[c]["rotations"][0]["p_fraud"] for c in non_fraud]
    top_fraud = sum(per_case[c]["rotations"][0]["top"] == "fraud" for c in non_fraud)
    r1: dict[str, Any] = {
        "cases": len(non_fraud),
        "mean_p_fraud": round(statistics.mean(p_fraud), 4),
        "top_fraud_share": round(top_fraud / len(non_fraud), 4),
    }
    r1["present"] = r1["mean_p_fraud"] > 0.30 and r1["top_fraud_share"] > 0.40

    # R2
    spreads, top_changes = [], 0
    for v in per_case.values():
        ps = [r["p_fraud"] for r in v["rotations"]]
        spreads.append(max(ps) - min(ps))
        top_changes += len({r["top"] for r in v["rotations"]}) > 1
    r2: dict[str, Any] = {
        "cases": len(per_case),
        "mean_p_fraud_spread": round(statistics.mean(spreads), 4),
        "top_changes_share": round(top_changes / len(per_case), 4),
    }
    r2["order_sensitive"] = r2["mean_p_fraud_spread"] > 0.10 or r2["top_changes_share"] > 0.25

    # R3
    diffs, disagree, pairs = [], 0, 0
    for c in dev.cases:
        if c["language"] == "en" and c.get("translation_of"):
            zh, en = (
                per_case[c["translation_of"]]["rotations"][0],
                per_case[c["case_id"]]["rotations"][0],
            )
            diffs.append(abs(zh["p_fraud"] - en["p_fraud"]))
            disagree += zh["top"] != en["top"]
            pairs += 1
    r3: dict[str, Any] = {
        "pairs": pairs,
        "mean_abs_p_fraud_diff": round(statistics.mean(diffs), 4),
        "top_disagreement_share": round(disagree / pairs, 4),
    }
    r3["language_dependent"] = (
        r3["mean_abs_p_fraud_diff"] > 0.10 or r3["top_disagreement_share"] > 0.25
    )

    # R4
    lengths = [v["length"] for v in per_case.values()]
    pf = [v["rotations"][0]["p_fraud"] for v in per_case.values()]
    r4: dict[str, Any] = {"cases": len(pf), "spearman_rho": round(spearman(lengths, pf), 4)}
    r4["length_driven"] = abs(r4["spearman_rho"]) > 0.30

    r5 = label_prior(engine, scene)

    # S1 + L: the five questions, every case.
    rows = run_questions(lambda _c: engine, [*dev.cases, *canary.cases], scene)
    by_id = {r["case_id"]: r for r in rows}
    sev_errors = []
    for c in dev.cases:
        gold = dev.expectations[c["case_id"]]["expected"]["severity"]
        if gold == "unknown":
            continue
        pred = by_id[c["case_id"]]["questions"]["severity"]["prediction"]
        crit_low = (
            "危急" in gold
            and max(gold, key=SEVERITY_ORDER.index) == "危急"
            and pred in ("資訊", "低")
        )
        benign_high = set(gold) == {"資訊"} and pred in ("高", "危急")
        if crit_low or benign_high:
            sev_errors.append(
                {
                    "case_id": c["case_id"],
                    "family": c["case_family"],
                    "kind": "critical_to_low" if crit_low else "benign_to_high",
                    "gold": gold,
                    "pred": pred,
                    "distribution": by_id[c["case_id"]]["questions"]["severity"]["distribution"],
                    "state": c["state"],
                }
            )
    lab = []
    for cid, rec in canary.replay.items():
        q = by_id[cid]["questions"]
        lab.append(
            {
                "case_id": cid,
                "family": canary.case(cid)["case_family"],
                "lab_p_fp": rec["observed"].get("is_false_positive"),
                "live_p_fp_on_reconstruction": round(q["is_false_positive"]["p_true"], 4),
                "lab_owner": rec["observed"].get("owner"),
                "live_owner": q["owner"]["prediction"],
                "lab_severity": rec["observed"].get("severity"),
                "live_severity": q["severity"]["prediction"],
            }
        )
    return {
        "verdict": {
            "R1_fraud_bias_present": r1["present"],
            "R2_order_sensitive": r2["order_sensitive"],
            "R3_language_dependent": r3["language_dependent"],
            "R4_length_driven": r4["length_driven"],
            "R5_prior_towards_fraud": r5["prior_towards_fraud"],
        },
        "R1": r1,
        "R2": r2,
        "R3": r3,
        "R4": r4,
        "R5": r5,
        "S1_severity_errors": sev_errors,
        "L_canary_live_vs_lab": lab,
        "per_case_owner_rotations": per_case,
    }


def summary(probes: Mapping[str, Any]) -> dict[str, Any]:
    """The verdicts and headline numbers, without per-case detail."""
    return {
        "verdict": probes["verdict"],
        "R1_mean_p_fraud": probes["R1"]["mean_p_fraud"],
        "R5_mean_p_fraud": probes["R5"]["mean_p_fraud"],
        "S1_errors": len(probes["S1_severity_errors"]),
    }
