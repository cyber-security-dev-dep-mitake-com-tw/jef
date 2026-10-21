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

The probes themselves live in ``jef_train.socbench.triage_rca`` (moved there
on 2026-10-21 so every candidate can be probed the same way); this script runs
them on the deployed model and writes the study's ``results.json``, unchanged by
the move.

Run from the repository root (needs the torch extra and the cached backbone):

    uv run python docs/validation/research/triage-rca-v1/run.py [OUT_DIR]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "packages" / "jef-train" / "src"))

from jef_core import BilinearHead, Engine  # noqa: E402
from jef_core.calibration import Calibrator  # noqa: E402
from jef_scene import load_scene  # noqa: E402
from jef_train.socbench.triage import load_pack  # noqa: E402
from jef_train.socbench.triage_baselines import run_baselines  # noqa: E402
from jef_train.socbench.triage_eval import Provenance  # noqa: E402
from jef_train.socbench.triage_rca import rca_probes  # noqa: E402

BACKBONE = "jhu-clsp/mmBERT-base"
HEAD = ROOT / "models" / "jef-v0" / "head.npz"
CAL = ROOT / "models" / "jef-v0" / "calibration.json"
SCENE_PATH = ROOT / "scenes" / "incident-triage.zh-tw.yaml"
DEV = ROOT / "benchmark" / "incident-triage" / "synthetic" / "dev-v1"
CANARY = ROOT / "benchmark" / "incident-triage" / "canary" / "six-alerts-v1"


def sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main(out: Path = OUT) -> None:
    scene = load_scene(SCENE_PATH)
    engine = Engine(BACKBONE, BilinearHead.load(str(HEAD)), Calibrator.load(str(CAL)))
    dev, canary = load_pack(DEV), load_pack(CANARY)
    lock = json.loads((DEV / "lock.json").read_text("utf-8"))

    probes = rca_probes(engine, scene, dev, canary)
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
        **probes,
        "verdict": {
            **probes["verdict"],
            "routing_decision": baselines["routing_decision"].get("routing"),
        },
        "B_baselines": {
            "split": baselines["split"],
            "cases": baselines["cases"],
            "fail_closed": baselines["fail_closed"],
            "routing_decision": baselines["routing_decision"],
            "tasks": {k: v["tasks"] for k, v in baselines["baselines"].items()},
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        "utf-8",
    )
    print(json.dumps(results["verdict"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
