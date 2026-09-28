"""The scene linter.

Most of these defects are ones that produce a *working* scene with silently
wrong behaviour, which is why a linter earns its place: a gate that can never
fire does not raise, it just quietly stops being part of the policy.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from jef_cli.validate import validate_paths, validate_scene_text


def _repo_root() -> Path:
    """Walk up to the repository root.

    Not `parents[N]`: these files moved one directory deeper when the four
    light packages merged into one distribution, and every hard-coded depth
    silently started pointing at `packages/scenes`.
    """
    return next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())


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

    scenes = _repo_root() / "scenes"
    problems = validate_paths([scenes])
    assert not errors(problems), [p.message for p in errors(problems)]
    assert not warnings(problems), [p.message for p in warnings(problems)]


# --------------------------------------------------------------------------- #
# JSON Schema
# --------------------------------------------------------------------------- #


def test_the_committed_schema_matches_the_model() -> None:
    """A schema that drifts from the validator is worse than none.

    Editors use it to tell an author their scene is wrong before they run it,
    so it has to agree with what actually rejects the scene.
    """
    import json
    import subprocess
    import sys

    root = _repo_root()
    committed = root / "docs" / "scene.schema.json"
    assert committed.is_file(), "docs/scene.schema.json is missing"

    result = subprocess.run(
        [sys.executable, "-m", "jef_cli", "schema"],
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == json.loads(committed.read_text(encoding="utf-8")), (
        "docs/scene.schema.json is stale; regenerate with `jef schema --out docs/scene.schema.json`"
    )


def test_the_schema_describes_what_a_scene_needs() -> None:
    import json

    schema = json.loads((_repo_root() / "docs" / "scene.schema.json").read_text(encoding="utf-8"))
    assert set(schema["required"]) == {"scene", "layers", "actions"}
    assert "$defs" in schema
    assert {"Layer", "Gate", "Action"} <= set(schema["$defs"])


def test_the_shipped_scenes_validate_against_the_schema() -> None:
    """The schema is only useful if the examples pass it."""
    import json

    jsonschema = pytest.importorskip("jsonschema")
    import yaml

    root = _repo_root()
    schema = json.loads((root / "docs" / "scene.schema.json").read_text(encoding="utf-8"))
    from jef_scene.loader import _normalise_bool_keys

    for path in sorted((root / "scenes").glob("*.yaml")):
        document = _normalise_bool_keys(yaml.safe_load(path.read_text(encoding="utf-8")))
        jsonschema.validate(document, schema)


def test_the_summary_counts_scenes_not_arguments(tmp_path, capsys) -> None:
    """`jef validate scenes/` used to say "1 path(s) checked".

    One argument, twenty files -- and the message was identical to a glob that
    had matched nothing, so a linter silently checking zero scenes looked
    exactly like a clean run.
    """
    from jef_cli.__main__ import main

    for name in ("a.yaml", "b.yaml"):
        (tmp_path / name).write_text(GOOD, encoding="utf-8")

    assert main(["validate", str(tmp_path)]) == 0
    assert "2 scene(s) checked" in capsys.readouterr().out


def test_a_missing_server_says_what_to_do_instead(capsys) -> None:
    """`ConnectError: [Errno 111] Connection refused` names neither the address
    nor the fact that `--local` exists.

    The commands default to a server on localhost, so this is the most likely
    thing a first-time user sees -- and it is what broke the release pipeline's
    smoke job on its first run.
    """
    from jef_cli.__main__ import main

    # Port 9 is discard: reserved, and nothing listens on it.
    assert main(["models", "--url", "http://127.0.0.1:9"]) != 0
    err = capsys.readouterr().err
    assert "http://127.0.0.1:9" in err
    assert "jef models --local" in err
    assert "jef serve" in err
