"""Scene validation. A wrong scene must stop a deploy, not an incident."""

from __future__ import annotations

import pytest
from jef_core.errors import SceneError
from jef_scene import SceneEngine, load_scene_text

OK = """
scene: t
layers:
  - id: L1
    questions:
      a: {type: noul, instructions: 甲？}
    gates:
      - when: 'a.probability > 0.5'
        then: close
      - else: true
        then: review
actions:
  close: {}
  review: {human_review: true}
"""


def test_valid_scene_loads() -> None:
    scene = load_scene_text(OK)
    assert scene.scene == "t"
    assert scene.question_count == 1


def test_unknown_action_target_is_rejected() -> None:
    with pytest.raises(SceneError, match="neither an action"):
        load_scene_text(OK.replace("then: close", "then: nonexistent"))


def test_gate_needs_exactly_one_condition() -> None:
    with pytest.raises(SceneError, match="exactly one"):
        load_scene_text("""
scene: t
layers:
  - id: L1
    questions:
      a: {type: noul, instructions: 甲？}
    gates:
      - when: 'a.probability > 0.5'
        else: true
        then: close
actions:
  close: {}
""")


def test_else_gate_must_come_last() -> None:
    # An else gate makes every later gate dead code. The author wants to hear
    # about that at load time.
    with pytest.raises(SceneError, match="unreachable"):
        load_scene_text("""
scene: t
layers:
  - id: L1
    questions:
      a: {type: noul, instructions: 甲？}
    gates:
      - else: true
        then: close
      - when: 'a.probability > 0.5'
        then: review
actions:
  close: {}
  review: {}
""")


def test_final_layer_must_decide_something() -> None:
    """A playbook that can run off the end has not made a decision."""
    with pytest.raises(SceneError, match="fall through without deciding"):
        load_scene_text("""
scene: t
layers:
  - id: L1
    questions:
      a: {type: noul, instructions: 甲？}
    gates:
      - when: 'a.probability > 0.5'
        then: close
actions:
  close: {}
""")


def test_fallthrough_satisfies_the_final_layer_requirement() -> None:
    scene = load_scene_text("""
scene: t
fallthrough: review
layers:
  - id: L1
    questions:
      a: {type: noul, instructions: 甲？}
    gates:
      - when: 'a.probability > 0.5'
        then: close
actions:
  close: {}
  review: {human_review: true}
""")
    assert scene.fallthrough == "review"


def test_fallthrough_must_name_a_real_action() -> None:
    with pytest.raises(SceneError, match="not a declared action"):
        load_scene_text(OK.replace("scene: t", "scene: t\nfallthrough: nope"))


def test_duplicate_layer_ids_are_rejected() -> None:
    with pytest.raises(SceneError, match="unique"):
        load_scene_text("""
scene: t
layers:
  - id: L1
    questions: {a: {type: noul, instructions: 甲？}}
    gates: [{else: true, then: close}]
  - id: L1
    questions: {b: {type: noul, instructions: 乙？}}
    gates: [{else: true, then: close}]
actions:
  close: {}
""")


def test_bad_gate_condition_fails_at_compile_not_at_runtime() -> None:
    scene = load_scene_text(OK.replace("a.probability > 0.5", "b.probability > 0.5"))
    with pytest.raises(SceneError, match="unknown name 'b'"):
        SceneEngine().compile(scene)


def test_yaml_boolean_keys_are_normalised() -> None:
    """`true:` and `false:` are booleans in YAML but strings on the wire."""
    scene = load_scene_text("""
scene: t
layers:
  - id: L1
    questions:
      a:
        type: noul
        instructions: 甲？
        criteria:
          true: 是
          false: 否
    gates:
      - else: true
        then: close
actions:
  close: {}
""")
    question = scene.layers[0].questions["a"]
    assert question.criteria is not None
    assert question.criteria.true_ == "是"
    assert question.criteria.false_ == "否"


def test_non_mapping_yaml_is_rejected() -> None:
    with pytest.raises(SceneError, match="must be a mapping"):
        load_scene_text("- just\n- a\n- list\n")


def test_invalid_yaml_is_rejected() -> None:
    with pytest.raises(SceneError, match="not valid YAML"):
        load_scene_text("scene: [unclosed\n")


def test_shipped_triage_scene_is_valid_and_compiles() -> None:
    """The scene in scenes/ is documentation people will copy. It must work."""
    from pathlib import Path

    from jef_scene import load_scene

    path = Path(__file__).resolve().parents[3] / "scenes" / "incident-triage.zh-tw.yaml"
    scene = load_scene(path)
    SceneEngine().compile(scene)
    assert scene.question_count == 5
    assert scene.fallthrough == "escalate_human"
    # Every terminal that a human must see is marked, so the trace can be
    # filtered on it and a scene cannot quietly stop escalating.
    assert scene.actions["escalate_human"].human_review is True
