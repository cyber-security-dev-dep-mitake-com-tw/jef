"""The policy layer: a gate proposes, the policy decides whether a human is needed.

POL-001 from the v2 review is the headline: an uncalibrated deployment with a
model that is 99.9% sure an alert is a false positive must not get `close`
authorised. The rest pin the tiering, the evidence rule, and that nothing in
the scene or the model can flip `review_required` back to `allow`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from jef_core import Engine
from jef_core.calibration import Calibrator
from jef_core.types import Answer, ChoiceAnswer, NoulAnswer
from jef_scene import SceneEngine, load_scene, load_scene_text
from jef_scene.policy import (
    MAX_AUTO_TIER,
    authorize,
    inferred_tier,
    referenced_questions,
    tier_of,
)
from jef_scene.schema import Action


class ScriptedEngine(Engine):
    """Returns fixed answers, so a test can hand the gate any probability it likes."""

    def __init__(self, answers: dict[str, Answer], calibrator: Calibrator | None = None) -> None:
        super().__init__(calibrator=calibrator)
        self._scripted = answers

    def answer(self, shared, questions):  # type: ignore[override]
        return {qid: self._scripted[qid] for qid in questions}


def sure_false_positive(p: float = 0.999) -> dict[str, Answer]:
    return {"is_fp": NoulAnswer(noul=p, confidence=abs(2 * p - 1))}


TRIAGE = """
scene: t
fallthrough: review
layers:
  - id: L1
    questions:
      is_fp:
        type: noul
        instructions: 是否為誤報？
    gates:
      - when: 'is_fp.probability > 0.90'
        then: {target}
      - else: true
        then: review
actions:
  close: {{}}
  request_context: {{}}
  nuke: {{ tier: 3 }}
  gentle_close: {{ tier: 1 }}
  review: {{ human_review: true }}
"""


def scene_for(target: str):
    return load_scene_text(TRIAGE.format(target=target))


# --------------------------------------------------------------------------- #
# POL-001
# --------------------------------------------------------------------------- #


def test_pol_001_uncalibrated_auto_close_is_not_authorised() -> None:
    """The exact scenario from the shipped scene: raw 0.999, no calibration, `close`."""
    engine = SceneEngine(ScriptedEngine(sure_false_positive()))
    trace = engine.run(scene_for("close"), "IDS 回報連接埠掃描")
    assert trace.action == "close", "the gate fired; the policy must not hide what was proposed"
    assert trace.proposed_action == "close"
    assert trace.policy_verdict == "review_required"
    assert trace.human_review is True
    assert "AUTO_CLOSE_PROHIBITED" in trace.reason_codes
    assert "UNVALIDATED_CALIBRATION:is_fp" in trace.reason_codes
    assert trace.executed_action is None
    assert trace.execution_status == "not_dispatched"


def test_pol_001_a_fabricated_certainty_changes_nothing() -> None:
    """0.99999 is still a model output nobody calibrated."""
    engine = SceneEngine(ScriptedEngine(sure_false_positive(0.99999)))
    trace = engine.run(scene_for("close"), "x")
    assert trace.policy_verdict == "review_required"
    assert trace.human_review is True


def test_close_stays_review_required_even_when_calibrated(calibrated_engine: Engine) -> None:
    """Tier beats calibration: a wrong close is not undone by a better model."""
    engine = SceneEngine(ScriptedEngine(sure_false_positive(), calibrated_engine.calibrator))
    trace = engine.run(scene_for("close"), "x")
    assert trace.policy_verdict == "review_required"
    assert trace.reason_codes == ["AUTO_CLOSE_PROHIBITED"], trace.reason_codes


# --------------------------------------------------------------------------- #
# Tier 1 needs calibrated evidence; with it, automation is allowed
# --------------------------------------------------------------------------- #


def test_tier_one_without_calibration_needs_a_human() -> None:
    engine = SceneEngine(ScriptedEngine(sure_false_positive()))
    trace = engine.run(scene_for("request_context"), "x")
    assert trace.action == "request_context"
    assert trace.policy_verdict == "review_required"
    assert trace.reason_codes == ["UNVALIDATED_CALIBRATION:is_fp"]
    assert trace.human_review is True


def test_tier_one_with_calibrated_evidence_is_allowed(calibrated_engine: Engine) -> None:
    engine = SceneEngine(ScriptedEngine(sure_false_positive(), calibrated_engine.calibrator))
    trace = engine.run(scene_for("request_context"), "x")
    assert trace.policy_verdict == "allow"
    assert trace.reason_codes == []
    assert trace.human_review is False
    assert trace.action_tier == 1


def test_explicit_tier_overrides_the_name_both_ways(calibrated_engine: Engine) -> None:
    engine = SceneEngine(ScriptedEngine(sure_false_positive(), calibrated_engine.calibrator))
    harmless_name_dangerous_tier = engine.run(scene_for("nuke"), "x")
    assert harmless_name_dangerous_tier.policy_verdict == "review_required"
    assert "DISRUPTIVE_ACTION_REQUIRES_APPROVAL" in harmless_name_dangerous_tier.reason_codes
    # The author took responsibility in the YAML: `gentle_close` is declared T1.
    declared_safe = engine.run(scene_for("gentle_close"), "x")
    assert declared_safe.policy_verdict == "allow"


# --------------------------------------------------------------------------- #
# Evidence rule: else gates and fallthrough cannot automate
# --------------------------------------------------------------------------- #

ELSE_AUTOMATES = """
scene: t
layers:
  - id: L1
    questions:
      is_fp:
        type: noul
        instructions: 是否為誤報？
    gates:
      - when: 'is_fp.probability > 0.99999'
        then: review
      - else: true
        then: enrich
actions:
  enrich: {}
  review: {human_review: true}
"""


def test_an_else_gate_cannot_automate(calibrated_engine: Engine) -> None:
    engine = SceneEngine(ScriptedEngine(sure_false_positive(0.5), calibrated_engine.calibrator))
    trace = engine.run(load_scene_text(ELSE_AUTOMATES), "x")
    assert trace.action == "enrich"
    assert trace.policy_verdict == "review_required"
    assert trace.reason_codes == ["NO_EVIDENCE_QUESTION"]


def test_human_review_actions_are_marked_as_such() -> None:
    engine = SceneEngine(ScriptedEngine(sure_false_positive(0.1)))
    trace = engine.run(scene_for("close"), "x")  # 0.1 fails the gate -> else -> review
    assert trace.action == "review"
    assert trace.policy_verdict == "review_required"
    assert trace.reason_codes == ["HUMAN_REVIEW_ACTION"]
    assert trace.human_review is True


def test_fallthrough_is_review_required(scene_engine: SceneEngine) -> None:
    scene = load_scene_text(
        TRIAGE.format(target="close").replace("      - else: true\n        then: review\n", "")
    )
    trace = SceneEngine(ScriptedEngine(sure_false_positive(0.1))).run(scene, "x")
    assert trace.verdict == "fallthrough"
    assert trace.action == "review"
    assert trace.policy_verdict == "review_required"


# --------------------------------------------------------------------------- #
# The pure function and the tiering
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "tier"),
    [
        ("close", 2),
        ("auto_close_and_suppress", 2),
        ("suppress_alert", 2),
        ("isolate_host", 3),
        ("disable_user", 3),
        ("block_ip", 3),
        ("quarantine_mailbox", 3),
        ("assign", 1),
        ("request_context", 1),
        ("page_oncall", 1),
        ("patch_now", 1),
        ("tag", 1),
    ],
)
def test_inferred_tiers(name: str, tier: int) -> None:
    assert inferred_tier(name) == tier


def test_tier_of_prefers_the_explicit_value() -> None:
    assert tier_of("close", Action(tier=1)) == 1
    assert tier_of("enrich", Action(tier=4)) == 4
    assert tier_of("close", Action()) == 2
    assert MAX_AUTO_TIER == 1


def test_referenced_questions_reads_only_known_names() -> None:
    names = referenced_questions("a.p_correct is not None and b.score > 2 and c", {"a", "b"})
    assert names == {"a", "b"}
    assert referenced_questions(None, {"a"}) == set()


def test_authorize_never_consults_the_probability() -> None:
    """A view with a huge probability but no p_correct is still unvalidated."""

    class View:
        p_correct = None
        probability = 0.99999

    verdict = authorize("assign", Action(), referenced={"q"}, views={"q": View()})  # type: ignore[dict-item]
    assert verdict.verdict == "review_required"
    assert verdict.reason_codes == ["UNVALIDATED_CALIBRATION:q"]
    assert verdict.human_review is True


def test_choice_views_work_too(calibrated_engine: Engine) -> None:
    answers: dict[str, Answer] = {
        "owner": ChoiceAnswer(
            choice="soc",
            probabilities={
                "soc": 0.9,
                "appsec": 0.025,
                "infra": 0.025,
                "identity": 0.025,
                "fraud": 0.025,
            },
            confidence=0.875,
        )
    }
    scene = load_scene_text(
        """
scene: r
fallthrough: review
layers:
  - id: L1
    questions:
      owner:
        type: choice
        instructions: 誰處理？
        criteria: {soc: a, appsec: b, infra: c, identity: d, fraud: e}
    gates:
      - when: 'owner.p_correct is not None and owner.choice == "soc"'
        then: assign
actions:
  assign: {}
  review: {human_review: true}
"""
    )
    trace = SceneEngine(ScriptedEngine(answers, calibrated_engine.calibrator)).run(scene, "x")
    assert trace.action == "assign"
    assert trace.policy_verdict == "allow"


# --------------------------------------------------------------------------- #
# The shipped scenes
# --------------------------------------------------------------------------- #


def _repo_root() -> Path:
    return next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())


@pytest.mark.parametrize("name", ["incident-triage.zh-tw.yaml", "vuln-triage.zh-tw.yaml"])
def test_shipped_scenes_never_automate_a_state_changing_action(name: str) -> None:
    scene = load_scene(_repo_root() / "scenes" / name)
    for action_name, action in scene.actions.items():
        if not action.human_review:
            assert tier_of(action_name, action) <= MAX_AUTO_TIER, (
                f"{name}: {action_name} automates a T{tier_of(action_name, action)} action"
            )


def test_incident_triage_closes_only_pending_review() -> None:
    scene = load_scene(_repo_root() / "scenes" / "incident-triage.zh-tw.yaml")
    assert "close" not in scene.actions
    pending = scene.actions["close_pending_review"]
    assert pending.human_review is True
    assert pending.tier == 2
    assert pending.params["queue"] == "closed-pending-review"


def test_trace_dict_carries_the_policy_fields(scene_engine: SceneEngine, minimal_scene) -> None:
    d = scene_engine.run(minimal_scene, "x").to_dict()
    for key in (
        "proposed_action",
        "policy_verdict",
        "action_tier",
        "reason_codes",
        "executed_action",
        "execution_status",
    ):
        assert key in d
    assert d["executed_action"] is None
    assert d["execution_status"] == "not_dispatched"
