"""The incident-triage harness: two modes that never stand in for each other.

Pre-registered rules under test (plan 2026-10-20): every task has its own
denominator; not_asked, unknown, unrecorded and unconfirmed are excluded and
counted, never scored wrong or benign; noul is scored on P(true), not on
confidence; a pinned artifact that does not match fails the run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jef_core import Engine
from jef_core.calibration import Calibrator
from jef_core.types import Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer
from jef_scene import load_scene, load_scene_text
from jef_train.socbench.cli import main
from jef_train.socbench.triage import ReplayEngine, find_root, load_pack
from jef_train.socbench.triage_eval import (
    CONTRACT_SMOKE,
    EvalError,
    Provenance,
    assert_no_family_overlap,
    report_class,
    run_questions,
    run_scene,
    scene_questions,
    score_questions,
    score_scene,
)

ROOT = find_root(Path(__file__).resolve())
REPO = ROOT.parents[1]
PACK_DIR = ROOT / "canary" / "six-alerts-v1"
SCENE_PATH = REPO / "scenes" / "incident-triage.zh-tw.yaml"
SCENE = load_scene(SCENE_PATH)
LEGACY = Calibrator.load(REPO / "models" / "jef-v0" / "calibration.json")
SEV = ["資訊", "低", "中", "高", "危急"]
OWNERS = ["soc", "appsec", "infra", "identity", "fraud"]


class Scripted(Engine):
    def __init__(self, answers: dict[str, Answer], calibrator: Calibrator | None = None) -> None:
        super().__init__(calibrator=calibrator)
        self._answers = answers

    def answer(self, shared, questions):  # type: ignore[no-untyped-def, override]
        return {q: self._answers[q] for q in questions if q in self._answers}


def answers(fp=0.2, ev=0.9, owner="soc", sev=4, urgent=0.9, fp_conf=None) -> dict[str, Answer]:  # type: ignore[no-untyped-def]
    op = {k: (0.8 if k == owner else 0.05) for k in OWNERS}
    sp = {k: (0.8 if i == sev else 0.05) for i, k in enumerate(SEV)}
    return {
        "is_false_positive": NoulAnswer(
            noul=fp, confidence=abs(2 * fp - 1) if fp_conf is None else fp_conf
        ),
        "has_evidence": NoulAnswer(noul=ev, confidence=abs(2 * ev - 1)),
        "owner": ChoiceAnswer(choice=owner, probabilities=op, confidence=0.75),
        "severity": ScoreAnswer(score=float(sev), legend=SEV, probabilities=sp, confidence=0.75),
        "urgent": NoulAnswer(noul=urgent, confidence=abs(2 * urgent - 1)),
    }


def case(cid: str, fam: str = "f") -> dict:
    return {"case_id": cid, "case_family": fam, "state": f"alert {cid}", "state_sha256": None}


def expectation(cid: str, status: str = "confirmed", **expected) -> dict:  # type: ignore[no-untyped-def]
    base = {
        "false_positive": False,
        "evidence_sufficiency": True,
        "owner": ["soc"],
        "severity": ["危急"],
        "urgency": True,
    }
    return {
        "case_id": cid,
        "label_origin": "single-human-reviewed",
        "label_status": status,
        "reviewers": ["r1"],
        "truth_of_event": "malicious",
        "truth_of_alert": "true_positive",
        "expected": base | expected,
        "prohibited_actions": ["close"],
        "required_context": [],
    }


def scripted(by_case: dict[str, dict[str, Answer]]):  # type: ignore[no-untyped-def]
    return lambda c: Scripted(by_case[c["case_id"]])


# --------------------------------------------------------------------------- #
# the scene is the only source of the questions
# --------------------------------------------------------------------------- #


def test_questions_come_from_the_scene() -> None:
    qs = scene_questions(SCENE)
    assert list(qs) == ["is_false_positive", "has_evidence", "owner", "severity", "urgent"]
    assert qs["owner"].task_id == "soar_zh.routing.v1"


# --------------------------------------------------------------------------- #
# questions mode
# --------------------------------------------------------------------------- #


def test_noul_is_scored_on_p_true_never_on_confidence() -> None:
    """P(FP)=0.2 with confidence 0.9 is 'not a false positive', however sure."""
    rows = run_questions(scripted({"a": answers(fp=0.2, fp_conf=0.9)}), [case("a")], SCENE)
    q = rows[0]["questions"]["is_false_positive"]
    assert q["p_true"] == 0.2 and q["confidence"] == 0.9
    assert q["prediction"] is False


def test_each_task_has_its_own_denominator_and_exclusions_are_counted() -> None:
    rows = run_questions(
        scripted({"a": answers(), "b": answers(), "c": answers(), "d": answers()}),
        [case("a"), case("b"), case("c"), case("d")],
        SCENE,
    )
    exps = {
        "a": expectation("a"),
        "b": expectation("b", false_positive="unknown", severity="unknown"),
        "c": expectation("c", status="pending_owner_confirmation"),
        "d": expectation("d", status="disputed"),
    }
    m = score_questions(rows, exps)["tasks"]
    assert m["false_positive"]["scored"] == 1
    assert m["false_positive"]["excluded"] == {
        "gold_unknown": 1,
        "label_disputed": 1,
        "label_pending_owner_confirmation": 1,
    }
    assert m["owner"]["scored"] == 2, "b's owner is known even though its FP is not"
    assert m["severity"]["scored"] == 1


def test_unknown_is_never_scored_as_benign() -> None:
    """A model saying 'false positive' on an unknown case is neither right nor wrong."""
    rows = run_questions(scripted({"a": answers(fp=0.99)}), [case("a")], SCENE)
    m = score_questions(rows, {"a": expectation("a", false_positive="unknown")})["tasks"]
    assert m["false_positive"]["scored"] == 0
    assert m["false_positive"]["attack_called_false_positive"] == 0


def test_a_missing_answer_is_excluded_not_wrong() -> None:
    partial = {k: v for k, v in answers().items() if k != "urgent"}
    rows = run_questions(scripted({"a": partial}), [case("a")], SCENE)
    assert rows[0]["questions"]["urgent"] == {
        "status": "missing_answer",
        "task_id": "soar_zh.urgency.v1",
    }
    m = score_questions(rows, {"a": expectation("a")})["tasks"]
    assert m["urgency"]["scored"] == 0 and m["urgency"]["excluded"] == {"missing_answer": 1}


def test_metrics_on_known_answers() -> None:
    by_case = {
        "atk": answers(fp=0.9, owner="fraud", sev=1, urgent=0.1),  # every costly mistake
        "ok": answers(fp=0.1, owner="soc", sev=4, urgent=0.9),
    }
    rows = run_questions(scripted(by_case), [case("atk"), case("ok")], SCENE)
    exps = {"atk": expectation("atk"), "ok": expectation("ok")}
    m = score_questions(rows, exps)["tasks"]
    assert m["false_positive"]["attack_called_false_positive"] == 1
    assert m["false_positive"]["attack_called_false_positive_rate"] == 0.5
    assert m["owner"]["accuracy_within_allowed"] == 0.5
    assert m["owner"]["predicted_base_rate"] == {"fraud": 1, "soc": 1}
    assert m["severity"]["critical_to_low_or_info"] == 1
    assert m["severity"]["mae"] == 1.5
    assert m["severity"]["under_estimate_rate"] == 0.5
    assert m["urgency"]["urgent_missed"] == 1 and m["urgency"]["fnr"] == 0.5
    assert "small_sample" in m["owner"], "two cases never get aggregate claims"


def test_an_allowed_alternative_counts_as_correct() -> None:
    rows = run_questions(scripted({"a": answers(owner="infra", sev=3)}), [case("a")], SCENE)
    exp = expectation("a", owner=["soc", "infra"], severity=["高", "危急"])
    out = score_questions(rows, {"a": exp})
    assert out["tasks"]["owner"]["accuracy_within_allowed"] == 1.0
    assert out["tasks"]["severity"]["mae"] == 0.0
    assert all(r["correct"] for r in out["per_case"] if r["task"] in ("owner", "severity"))


def test_uncalibrated_answers_say_so() -> None:
    rows = run_questions(scripted({"a": answers()}), [case("a")], SCENE)
    for q in rows[0]["questions"].values():
        assert q["calibration_status"] == "missing"
        assert q["p_correct"] is None
    assert rows[0]["questions"]["owner"]["prediction_set"] == OWNERS


# --------------------------------------------------------------------------- #
# the two modes never mix
# --------------------------------------------------------------------------- #

SHORT_CIRCUIT = """
scene: sc
fallthrough: escalate_human
advisory: {unvalidated_fallback: escalate_human}
layers:
  - id: L1
    questions:
      is_false_positive: {type: noul, task_id: soar_zh.false_positive.v1, instructions: 誤報？}
      has_evidence: {type: noul, task_id: soar_zh.evidence_sufficiency.v1, instructions: 證據？}
    gates:
      - when: 'is_false_positive.probability < 0.5'
        then: escalate_human
      - else: true
        then: next
  - id: L2
    questions:
      owner:
        type: choice
        task_id: soar_zh.routing.v1
        instructions: 團隊？
        criteria: {soc: s, appsec: a, infra: i, identity: d, fraud: f}
      severity: {type: score, task_id: soar_zh.severity.v1, instructions: 嚴重度？, criteria: [資訊, 低, 中, 高, 危急]}
      urgent: {type: noul, task_id: soar_zh.urgency.v1, instructions: 急？}
    gates:
      - else: true
        then: queue_for_review
actions:
  escalate_human: {human_review: true}
  queue_for_review: {human_review: true}
"""


def test_an_l1_short_circuit_is_not_asked_in_scene_mode_but_answered_in_questions_mode() -> None:
    scene = load_scene_text(SHORT_CIRCUIT)
    engines = scripted({"a": answers(fp=0.1)})
    scene_rows = run_scene(engines, [case("a")], scene)
    q_rows = run_questions(engines, [case("a")], scene)
    assert scene_rows[0]["layers_skipped"] == ["L2"]
    assert scene_rows[0]["questions"]["owner"] == "not_asked"
    assert q_rows[0]["questions"]["owner"]["status"] == "answered"

    sm = score_scene(scene_rows, {"a": expectation("a")})
    assert sm["not_asked"] == {"owner": 1, "severity": 1, "urgent": 1}
    assert "tasks" not in sm, "scene mode never reports per-question accuracy"
    with pytest.raises(EvalError, match="scene-mode rows"):
        score_questions(scene_rows, {"a": expectation("a")})


def test_scene_mode_on_the_canaries_under_legacy() -> None:
    pack = load_pack(PACK_DIR)
    cases = [c for c in pack.cases if c["case_id"] in pack.replay]
    rows = run_scene(lambda c: ReplayEngine(pack.replay[c["case_id"]], LEGACY), cases, SCENE)
    m = score_scene(rows, pack.expectations)
    assert m["cases"] == 6
    assert m["review_required"] == 6 and m["auto_allowed"] == 0
    assert m["dispatched"] == 0 and m["executed"] == 0
    assert m["proposals_outside_escalation"] == []
    assert m["prohibited_action_proposed"] == []
    assert m["malicious_close_proposed"] == []
    assert m["not_asked"] == {"urgent": 6}


def test_replay_fillers_are_never_scored() -> None:
    pack = load_pack(PACK_DIR)
    cases = [c for c in pack.cases if c["case_id"] in pack.replay]
    rows = run_questions(lambda c: ReplayEngine(pack.replay[c["case_id"]], LEGACY), cases, SCENE)
    m = score_questions(rows, pack.expectations, include_pending=True)["tasks"]
    assert m["urgency"]["scored"] == 0 and m["urgency"]["excluded"] == {"not_recorded": 6}
    assert m["false_positive"]["scored"] == 6
    # The Lab's own numbers: ransomware 0.85 and SSH 0.59 read as false positives.
    assert m["false_positive"]["attack_called_false_positive"] == 2
    assert m["severity"]["critical_to_low_or_info"] == 1, "ransomware scored 低"


# --------------------------------------------------------------------------- #
# what the report may be called, and pinned artifacts
# --------------------------------------------------------------------------- #


def test_report_class() -> None:
    exps = {"a": expectation("a")}
    assert report_class(exps, contract_smoke=True, include_pending=False)[0] == CONTRACT_SMOKE
    assert report_class(exps, contract_smoke=False, include_pending=False)[0] == (
        "SINGLE-REVIEW RESEARCH"
    )
    pending = {"a": expectation("a", status="pending_owner_confirmation")}
    assert report_class(pending, contract_smoke=False, include_pending=False)[0] == (
        "NO SCORABLE LABELS"
    )
    synthetic = {"a": expectation("a") | {"label_origin": "synthetic"}}
    cls, _ = report_class(synthetic, contract_smoke=False, include_pending=True)
    assert cls.startswith("SYNTHETIC LABELS") and "DIAGNOSTIC" in cls


def test_a_pinned_sha_that_does_not_match_fails(tmp_path: Path) -> None:
    head = tmp_path / "head.npz"
    head.write_bytes(b"not the head you pinned")
    prov = Provenance(SCENE_PATH, SCENE, Engine(), head_path=str(head))
    with pytest.raises(EvalError, match="model"):
        prov.check(source_sha=None, model_sha="0" * 64, scene_sha=None)
    with pytest.raises(EvalError, match="scene"):
        prov.check(source_sha=None, model_sha=None, scene_sha="0" * 64)
    with pytest.raises(EvalError, match="source"):
        prov.check(source_sha="deadbeef", model_sha=None, scene_sha=None)
    good = prov.to_dict()
    prov.check(
        source_sha=None, model_sha=good["model"]["head_sha256"], scene_sha=good["scene"]["sha256"]
    )


def test_family_overlap_is_refused() -> None:
    assert_no_family_overlap([case("a", "x")], [case("b", "y")])
    with pytest.raises(EvalError, match="x"):
        assert_no_family_overlap([case("a", "x")], [case("b", "x")])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["scene", "questions"])
def test_cli_writes_a_pinned_contract_smoke_report(tmp_path: Path, mode: str) -> None:
    out = tmp_path / mode
    rc = main(
        [
            "triage", "evaluate", "--mode", mode, "--pack", str(PACK_DIR),
            "--scorer", "replay", "--calibration", str(REPO / "models/jef-v0/calibration.json"),
            "--out", str(out),
        ]
    )  # fmt: skip
    assert rc == 0
    report = json.loads((out / "results.json").read_text("utf-8"))
    assert report["evidence_class"] == CONTRACT_SMOKE
    assert report["mode"] == mode
    prov = report["provenance"]
    assert prov["scene"]["sha256"].startswith("sha256:") and prov["scene"]["version"] == 2
    assert prov["model"]["calibration_sha256"].startswith("sha256:")
    assert prov["model"]["calibration_legacy"] is True
    assert prov["pack"]["files"]["cases.jsonl"].startswith("sha256:")
    assert all(r["request_sha256"] for r in report["rows"])
    assert CONTRACT_SMOKE in (out / "summary.md").read_text("utf-8")


def test_cli_refuses_a_mismatched_scene_sha(tmp_path: Path) -> None:
    rc = main(
        [
            "triage", "evaluate", "--mode", "scene", "--pack", str(PACK_DIR),
            "--scene-sha", "0" * 64, "--out", str(tmp_path / "x"),
        ]
    )  # fmt: skip
    assert rc == 2
    assert not (tmp_path / "x").exists()
