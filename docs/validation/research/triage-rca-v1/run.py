#!/usr/bin/env python3
"""SYNTHETIC LABELS RESEARCH: why jef-v0 routes and grades incident-triage alerts badly.

Fixed on 2026-10-20 before R1-R5 were measured. **Already seen** when this
file was written: the B0-B3 comparison on the synthetic test split, run while
building ``jef-bench triage baselines`` (jef-v0 routing macro-F1 0.089
against rules 0.867; mean P(fraud) 0.29; 10 of 22 attacks read as false
positives). The routing decision rule it feeds was pre-registered earlier, in
``docs/validation/plans/2026-10-20-incident-triage-validation-first.md``, and
is unchanged. Nothing below was tuned after seeing it.

**Model.** The deployed artifact: ``jhu-clsp/mmBERT-base`` at the pinned
revision, ``models/jef-v0/head.npz``, ``models/jef-v0/calibration.json``
(schema 1, legacy). ``soar_zh`` was excluded from its training
(``docs/RESULTS.md``: routing 0.193 against 0.200 random when it was tried), so
every routing, severity and urgency answer here is the head's zero-shot
reading of the scene's own label texts.

**Cases.** ``benchmark/incident-triage/synthetic/dev-v1`` (78 cases, 25
families, zh-TW/en pairs, labels by construction, Tier E) and the six K8S Lab
canary reconstructions with their three twins (labels pending; used only
where no gold is needed).

**Questions and criteria, fixed in advance.**

R1 fraud bias: on synthetic cases whose acceptable owners exclude fraud, the
   bias is *present* if mean P(fraud) > 0.30 **and** fraud is the top answer
   for > 40% of them. (Uniform is 0.20.)
R2 option order: rotate the owner options five ways, record P(fraud) and the
   top answer each time. *Order-sensitive* if the mean over cases of
   (max - min) P(fraud) across rotations > 0.10, or the top answer changes
   with the rotation for > 25% of cases.
R3 language: on zh-TW/en pairs, *language-dependent* if mean |P_zh(fraud) -
   P_en(fraud)| > 0.10 or the top owner disagrees on > 25% of pairs.
R4 length: *length-driven* if |Spearman rho(state length, P(fraud))| > 0.30.
R5 label prior: four content-free states ("告警。", "Alert.", "系統通知。",
   "Notification."); a *prior towards fraud* if mean P(fraud) on them > 0.30.
S1 severity (descriptive, no threshold): every case whose most severe
   acceptable level is 危急 and whose prediction is 低 or 資訊, and every case
   whose acceptable levels are only 資訊 and whose prediction is 高 or 危急,
   with the state, so the errors can be read, not just averaged.
B  baselines: B0-B3 on the locked synthetic test split, and the
   pre-registered routing decision.
L  the six canary reconstructions, live: P(FP) next to the Lab's recorded
   value. Descriptive: the reconstructions are not the original text.

Run from the repository root (needs the torch extra and the cached backbone):

    uv run python docs/validation/research/triage-rca-v1/run.py [OUT_DIR]
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
#: Results land next to this file unless a directory is given (the periodic
#: workflow writes elsewhere so the published results stay what they were).
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "packages" / "jef-train" / "src"))

from jef_core import BilinearHead, Engine  # noqa: E402
from jef_core.calibration import Calibrator  # noqa: E402
from jef_scene import load_scene  # noqa: E402
from jef_train.socbench.triage import load_pack  # noqa: E402
from jef_train.socbench.triage_baselines import run_baselines  # noqa: E402
from jef_train.socbench.triage_eval import (  # noqa: E402
    Provenance,
    run_questions,
    scene_questions,
)

BACKBONE = "jhu-clsp/mmBERT-base"
HEAD = ROOT / "models" / "jef-v0" / "head.npz"
CAL = ROOT / "models" / "jef-v0" / "calibration.json"
SCENE_PATH = ROOT / "scenes" / "incident-triage.zh-tw.yaml"
DEV = ROOT / "benchmark" / "incident-triage" / "synthetic" / "dev-v1"
CANARY = ROOT / "benchmark" / "incident-triage" / "canary" / "six-alerts-v1"
NEUTRAL = ["告警。", "Alert.", "系統通知。", "Notification."]


def sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def rotations(question: Any) -> list[dict[str, Any]]:
    """The owner question with its options in each of the five rotations."""
    raw = question.model_dump(exclude_none=True)
    items = list(raw["criteria"].items())
    return [{**raw, "criteria": dict(items[i:] + items[:i])} for i in range(len(items))]


def spearman(x: list[float], y: list[float]) -> float:
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    scene = load_scene(SCENE_PATH)
    engine = Engine(BACKBONE, BilinearHead.load(str(HEAD)), Calibrator.load(str(CAL)))
    owner_q = scene_questions(scene)["owner"]
    dev, canary = load_pack(DEV), load_pack(CANARY)
    lock = json.loads((DEV / "lock.json").read_text("utf-8"))

    # Every state once; five rotations of the owner question on the same encoding.
    states = {c["case_id"]: c for c in [*dev.cases, *canary.cases]}
    per_case: dict[str, dict[str, Any]] = {}
    for cid, c in states.items():
        shared = engine.prepare(c["state"])
        rot = []
        for r in rotations(owner_q):
            a = engine.answer(shared, {"owner": r})["owner"]
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
    r1 = {
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
    r2 = {
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
    r3 = {
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
    r4 = {"cases": len(pf), "spearman_rho": round(spearman(lengths, pf), 4)}
    r4["length_driven"] = abs(r4["spearman_rho"]) > 0.30

    # R5
    probes = {}
    for text in NEUTRAL:
        a = engine.answer(engine.prepare(text), {"owner": owner_q})["owner"]
        probes[text] = {
            "p_fraud": round(a.probabilities["fraud"], 4),
            "top": a.choice,
            "dist": a.probabilities,
        }
    r5 = {
        "probes": probes,
        "mean_p_fraud": round(statistics.mean(p["p_fraud"] for p in probes.values()), 4),
    }
    r5["prior_towards_fraud"] = r5["mean_p_fraud"] > 0.30

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
            and max(gold, key=["資訊", "低", "中", "高", "危急"].index) == "危急"
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

    baselines = run_baselines(dev, lock, scene, jef=engine)
    results = {
        "evidence_class": "SYNTHETIC LABELS RESEARCH (Tier E) on the deployed jef-v0",
        "model": Provenance(SCENE_PATH, scene, engine, str(HEAD), str(CAL), "jef", dev).to_dict()[
            "model"
        ],
        "inputs": {
            "dev_manifest": dev.manifest["files"],
            "canary_manifest": canary.manifest["files"],
            "lock_cases_digest": lock["cases_digest"],
            "scene_sha256": sha(SCENE_PATH),
        },
        "verdict": {
            "R1_fraud_bias_present": r1["present"],
            "R2_order_sensitive": r2["order_sensitive"],
            "R3_language_dependent": r3["language_dependent"],
            "R4_length_driven": r4["length_driven"],
            "R5_prior_towards_fraud": r5["prior_towards_fraud"],
            "routing_decision": baselines["routing_decision"].get("routing"),
        },
        "R1": r1,
        "R2": r2,
        "R3": r3,
        "R4": r4,
        "R5": r5,
        "S1_severity_errors": sev_errors,
        "L_canary_live_vs_lab": lab,
        "B_baselines": {
            "split": baselines["split"],
            "cases": baselines["cases"],
            "fail_closed": baselines["fail_closed"],
            "routing_decision": baselines["routing_decision"],
            "tasks": {k: v["tasks"] for k, v in baselines["baselines"].items()},
        },
        "per_case_owner_rotations": per_case,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        "utf-8",
    )
    print(json.dumps(results["verdict"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
