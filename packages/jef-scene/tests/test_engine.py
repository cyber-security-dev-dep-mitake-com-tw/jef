"""Gate semantics and the decision trace."""

from __future__ import annotations

import pytest
from jef_core.errors import SceneError
from jef_scene import SceneEngine, load_scene_text
from jef_scene.expr import ExpressionError

ROUTING = """
scene: routing
alpha: 0.10
fallthrough: escalate
layers:
  - id: L1
    questions:
      owner:
        type: choice
        instructions: 誰處理？
        criteria:
          soc: 監控
          appsec: 應用程式
          infra: 基礎設施
          identity: 身分
          fraud: 詐欺
    gates:
      - when: 'owner.set_size > 2'
        then: escalate
        because: 模型無法在承諾覆蓋率下分辨
      - when: 'owner.p_correct is not None and owner.p_correct >= 0.85'
        then: assign
      - else: true
        then: escalate
actions:
  assign: {description: 自動指派, params: {queue: team}}
  escalate: {human_review: true}
"""


def test_uncalibrated_deployment_cannot_automate(scene_engine: SceneEngine) -> None:
    """Without calibration, every path must end at a human.

    This is the design working, not a limitation: p_correct is unavailable and
    the conformal set is the full option set, so no automating gate can fire.
    You cannot get automation out of JEF without calibrating it.
    """
    scene = load_scene_text(ROUTING)
    trace = scene_engine.run(scene, "付款服務連續三天失敗")
    assert trace.action == "escalate"
    assert trace.human_review is True
    assert trace.calibrated is False
    q = trace.layers[0].questions[0]
    assert q.p_correct is None
    assert len(q.prediction_set) == 5, "an uncalibrated conformal set excludes nothing"


def test_calibrated_engine_can_automate(calibrated_engine) -> None:
    scene = load_scene_text(ROUTING)
    engine = SceneEngine(calibrated_engine)
    trace = engine.run(scene, "付款服務連續三天失敗，錯誤碼 502")
    assert trace.calibrated is True
    q = trace.layers[0].questions[0]
    assert q.p_correct is not None
    assert len(q.prediction_set) <= 5


def test_trace_records_every_gate_that_was_evaluated(scene_engine: SceneEngine) -> None:
    trace = scene_engine.run(load_scene_text(ROUTING), "告警")
    gates = trace.layers[0].gates
    assert gates[0].fired is True
    assert gates[0].then == "escalate"
    assert gates[0].because == "模型無法在承諾覆蓋率下分辨"
    # Gates after the one that fired are never evaluated, so never recorded.
    assert len(gates) == 1


def test_trace_records_allowed_answers_and_the_distribution(scene_engine: SceneEngine) -> None:
    """An audit needs to see what was permitted, not just what was chosen."""
    trace = scene_engine.run(load_scene_text(ROUTING), "告警")
    q = trace.layers[0].questions[0]
    assert q.allowed_answers == ["soc", "appsec", "infra", "identity", "fraud"]
    assert set(q.answer["probabilities"]) == set(q.allowed_answers)
    assert q.instructions == "誰處理？"


def test_trace_is_json_serialisable(scene_engine: SceneEngine) -> None:
    import json

    trace = scene_engine.run(load_scene_text(ROUTING), "告警")
    assert json.loads(json.dumps(trace.to_dict(), ensure_ascii=False))["scene"] == "routing"


def test_action_params_reach_the_caller(calibrated_engine) -> None:
    scene = load_scene_text("""
scene: p
layers:
  - id: L1
    questions:
      a: {type: noul, instructions: 甲？}
    gates:
      - else: true
        then: assign
actions:
  assign: {params: {queue: team_inbox, priority: p2}}
""")
    trace = SceneEngine(calibrated_engine).run(scene, "x")
    assert trace.action_params == {"queue": "team_inbox", "priority": "p2"}


def test_fallthrough_is_recorded_as_such(scene_engine: SceneEngine) -> None:
    scene = load_scene_text("""
scene: f
fallthrough: review
layers:
  - id: L1
    questions:
      a: {type: noul, instructions: 甲？}
    gates:
      - when: 'a.probability > 1.5'
        then: close
actions:
  close: {}
  review: {human_review: true}
""")
    trace = scene_engine.run(scene, "x")
    assert trace.verdict == "fallthrough"
    assert trace.action == "review"
    assert trace.human_review is True


def test_runtime_gate_failure_raises_rather_than_reading_as_false(
    scene_engine: SceneEngine,
) -> None:
    """A broken gate must not silently behave like a false one.

    Treating an error as "condition not met" turns a typo into a policy change
    nobody approved -- the alert quietly takes a different branch forever.
    """
    scene = load_scene_text("""
scene: broken
fallthrough: review
layers:
  - id: L1
    questions:
      a: {type: noul, instructions: 甲？}
    gates:
      - when: 'a.p_correct >= 0.9'
        then: close
      - else: true
        then: review
actions:
  close: {}
  review: {}
""")
    with pytest.raises(ExpressionError):
        scene_engine.run(scene, "x")


def test_noul_answers_are_readable_under_both_spellings(scene_engine: SceneEngine) -> None:
    """A gate should not need to know which dialect the question used."""
    for expr in ("a.probability > 0.0", "a.noul > 0.0"):
        scene = load_scene_text(f"""
scene: d
layers:
  - id: L1
    questions:
      a: {{type: boolean, instructions: 甲？}}
    gates:
      - when: '{expr}'
        then: close
      - else: true
        then: review
actions:
  close: {{}}
  review: {{}}
""")
        trace = scene_engine.run(scene, "x")
        assert trace.action == "close", f"{expr} should have fired"


def test_noul_probabilities_are_exposed_as_a_distribution(scene_engine: SceneEngine) -> None:
    scene = load_scene_text("""
scene: d
layers:
  - id: L1
    questions:
      a: {type: noul, instructions: 甲？}
    gates:
      - when: "a.probabilities['true'] + a.probabilities['false'] > 0.99"
        then: close
      - else: true
        then: review
actions:
  close: {}
  review: {}
""")
    assert scene_engine.run(scene, "x").action == "close"


def test_registry_rejects_duplicate_scene_ids() -> None:
    from jef_scene import SceneRegistry

    registry = SceneRegistry()
    registry.add(load_scene_text(ROUTING))
    with pytest.raises(SceneError, match="duplicate scene id"):
        registry.add(load_scene_text(ROUTING))


def test_registry_reports_what_it_knows_when_asked_for_something_it_does_not() -> None:
    from jef_scene import SceneRegistry

    registry = SceneRegistry()
    registry.add(load_scene_text(ROUTING))
    with pytest.raises(SceneError, match="loaded: routing"):
        registry.get("missing")
