"""The scene linter.

Most of these defects are ones that produce a *working* scene with silently
wrong behaviour, which is why a linter earns its place: a gate that can never
fire does not raise, it just quietly stops being part of the policy.
"""

from __future__ import annotations

from jef_cli.validate import validate_paths, validate_scene_text

GOOD = """
scene: ok
fallthrough: review
layers:
  - id: L1
    questions:
      owner:
        type: choice
        instructions: 誰處理？
        criteria: { soc: 監控, infra: 網路 }
    gates:
      - when: 'owner.p_correct is not None and owner.p_correct >= 0.95'
        then: assign
      - else: true
        then: review
actions:
  assign: {}
  review: { human_review: true }
"""


def errors(problems):
    return [p for p in problems if p.severity == "error"]


def warnings(problems):
    return [p for p in problems if p.severity == "warning"]


def test_a_good_scene_is_clean() -> None:
    assert validate_scene_text(GOOD, "ok.yaml") == []


def test_percentage_written_as_a_probability_is_an_error() -> None:
    """`confidence > 90` never fires, and nothing at runtime says so."""
    scene = GOOD.replace(
        "'owner.p_correct is not None and owner.p_correct >= 0.95'", "'owner.confidence > 90'"
    )
    found = errors(validate_scene_text(scene, "bad.yaml"))
    assert len(found) == 1
    assert "can never be true" in found[0].message
    # The suggestion matters more than the diagnosis.
    assert "0.9" in found[0].message
    assert found[0].line is not None


def test_negative_threshold_is_an_error() -> None:
    scene = GOOD.replace(
        "'owner.p_correct is not None and owner.p_correct >= 0.95'", "'owner.confidence < -1'"
    )
    assert errors(validate_scene_text(scene, "bad.yaml"))


def test_automating_on_confidence_is_warned() -> None:
    """The project's own thesis, enforced: confidence is not correctness."""
    scene = GOOD.replace(
        "'owner.p_correct is not None and owner.p_correct >= 0.95'", "'owner.confidence >= 0.9'"
    )
    found = warnings(validate_scene_text(scene, "bad.yaml"))
    assert any("not the chance of being correct" in p.message for p in found)


def test_confidence_is_fine_when_it_routes_to_a_human() -> None:
    """Escalating on low confidence is correct; only automating on it is not."""
    scene = GOOD.replace(
        "      - when: 'owner.p_correct is not None and owner.p_correct >= 0.95'\n        then: assign",
        "      - when: 'owner.confidence < 0.5'\n        then: review",
    )
    assert not [
        p for p in validate_scene_text(scene, "ok.yaml") if "chance of being correct" in p.message
    ]


def test_confidence_alongside_p_correct_is_fine() -> None:
    scene = GOOD.replace(
        "'owner.p_correct is not None and owner.p_correct >= 0.95'",
        "'owner.p_correct is not None and owner.p_correct >= 0.95 and owner.confidence > 0.8'",
    )
    assert not [
        p for p in validate_scene_text(scene, "ok.yaml") if "chance of being correct" in p.message
    ]


def test_unread_question_is_warned() -> None:
    scene = GOOD.replace(
        "      owner:",
        "      unused:\n        type: noul\n        instructions: 沒人讀\n      owner:",
    )
    found = warnings(validate_scene_text(scene, "bad.yaml"))
    assert any("'unused' is asked but no gate reads it" in p.message for p in found)


def test_unreachable_action_is_warned() -> None:
    scene = GOOD.replace("actions:\n  assign: {}", "actions:\n  assign: {}\n  orphan: {}")
    found = warnings(validate_scene_text(scene, "bad.yaml"))
    assert any("'orphan' is declared but never reachable" in p.message for p in found)


def test_unknown_question_name_in_a_gate_is_an_error() -> None:
    scene = GOOD.replace("'owner.p_correct", "'nosuch.p_correct")
    found = errors(validate_scene_text(scene, "bad.yaml"))
    assert any("unknown name" in p.message for p in found)


def test_disallowed_attribute_is_an_error() -> None:
    scene = GOOD.replace(
        "'owner.p_correct is not None and owner.p_correct >= 0.95'", "'owner.__class__ == 1'"
    )
    assert errors(validate_scene_text(scene, "bad.yaml"))


def test_schema_failure_is_reported_once_and_stops_there() -> None:
    """A malformed scene cannot be cross-checked, so it reports one clear error."""
    scene = GOOD.replace("then: assign", "then: nonexistent_action")
    found = validate_scene_text(scene, "bad.yaml")
    assert len(found) == 1
    assert found[0].severity == "error"


def test_invalid_yaml_is_reported() -> None:
    found = validate_scene_text("scene: [unclosed\n", "bad.yaml")
    assert errors(found)


def test_missing_path_is_an_error(tmp_path) -> None:
    found = validate_paths([tmp_path / "nope.yaml"])
    assert errors(found)


def test_empty_directory_warns(tmp_path) -> None:
    found = validate_paths([tmp_path])
    assert warnings(found)
    assert not errors(found)


def test_the_shipped_scene_lints_clean() -> None:
    """scenes/ is documentation people copy. It must pass its own linter."""
    from pathlib import Path

    scenes = Path(__file__).resolve().parents[3] / "scenes"
    problems = validate_paths([scenes])
    assert not errors(problems), [p.message for p in errors(problems)]
    assert not warnings(problems), [p.message for p in warnings(problems)]
