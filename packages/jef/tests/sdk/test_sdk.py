"""The SDK's promise: embedded and remote have the same shape."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jef_core import Engine
from jef_scene import SceneEngine, SceneRegistry, load_scene
from jef_sdk import Jef, JefClient, JefHTTPError, boolean, choice, noul, score
from jef_server.app import create_app
from jef_server.settings import Settings


def _repo_root() -> Path:
    """Walk up to the repository root.

    Not `parents[N]`: these files moved one directory deeper when the four
    light packages merged into one distribution, and every hard-coded depth
    silently started pointing at `packages/scenes`.
    """
    return next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())


SCENES = _repo_root() / "scenes"
ALERT = "付款服務連續三天失敗，已影響營收。"


# --------------------------------------------------------------------------- #
# Question builders
# --------------------------------------------------------------------------- #


def test_choice_preserves_keyword_order() -> None:
    q = choice("誰處理？", soc="監控", appsec="程式碼", infra="網路")
    assert list(q["criteria"]) == ["soc", "appsec", "infra"]


def test_choice_requires_two_options() -> None:
    with pytest.raises(ValueError, match="at least two options"):
        choice("誰處理？", only="one")


def test_score_keeps_levels_ordered() -> None:
    q = score("嚴重度", "低", "中", "高")
    assert q["criteria"] == ["低", "中", "高"]


def test_score_requires_two_levels() -> None:
    with pytest.raises(ValueError, match="at least two ordered levels"):
        score("嚴重度", "低")


def test_noul_omits_empty_criteria() -> None:
    assert "criteria" not in noul("是否緊急？")
    assert noul("是否緊急？", true="是")["criteria"] == {"true": "是"}


def test_boolean_is_the_same_primitive_in_the_ai_sdk_spelling() -> None:
    a, b = noul("是否緊急？"), boolean("是否緊急？")
    assert a["type"] == "noul"
    assert b["type"] == "boolean"
    assert a["instructions"] == b["instructions"]


# --------------------------------------------------------------------------- #
# Embedded
# --------------------------------------------------------------------------- #


def test_embedded_evaluates_and_warns_about_calibration(caplog) -> None:
    with caplog.at_level("WARNING"):
        jef = Jef()
    assert any("no calibration" in r.message for r in caplog.records), (
        "an uncalibrated SDK must say so; silent uncalibrated automation is the "
        "failure mode this project exists to prevent"
    )
    result = jef.evaluate(ALERT, {"urgent": noul("是否緊急？")})
    assert 0.0 <= result.answers["urgent"].confidence <= 1.0


def test_embedded_runs_scenes_with_one_state_read() -> None:
    jef = Jef(scenes=SCENES)
    trace = jef.run_scene("incident-triage", ALERT)
    assert trace.state_encodes == 1
    assert trace.action is not None


def test_embedded_reports_what_it_is() -> None:
    jef = Jef(scenes=SCENES)
    assert jef.calibrated is False
    assert "incident-triage" in jef.scenes.names()
    assert "hashing" in repr(jef)


def test_add_scene_accepts_a_path() -> None:
    jef = Jef()
    scene = jef.add_scene(SCENES / "incident-triage.zh-tw.yaml")
    assert scene.scene in jef.scenes


def test_run_accepts_an_unregistered_scene() -> None:
    """Ad-hoc composition should not require registering anything."""
    jef = Jef()
    scene = load_scene(SCENES / "incident-triage.zh-tw.yaml")
    SceneEngine(jef.engine).compile(scene)
    assert jef.run(scene, ALERT).action is not None


# --------------------------------------------------------------------------- #
# Remote -- same shape, different transport
# --------------------------------------------------------------------------- #


@pytest.fixture
def remote() -> JefClient:
    engine = Engine()
    scene_engine = SceneEngine(engine)
    registry = SceneRegistry()
    registry.add(load_scene(SCENES / "incident-triage.zh-tw.yaml"), compile_with=scene_engine)
    app = create_app(engine=engine, settings=Settings(), scenes=(scene_engine, registry))
    return JefClient(client=TestClient(app))  # type: ignore[arg-type]


def test_remote_evaluate_returns_the_same_model_as_embedded(remote: JefClient) -> None:
    questions = {"urgent": noul("是否緊急？"), "team": choice("誰處理？", soc="監控", infra="網路")}
    local = Jef().evaluate(ALERT, questions)
    over_http = remote.evaluate(ALERT, questions)
    assert local.answers.keys() == over_http.answers.keys()
    assert local.answers["team"].model_dump() == over_http.answers["team"].model_dump()


def test_remote_scene_trace_is_left_as_a_dict(remote: JefClient) -> None:
    # A newer server may add trace fields; narrowing them into an older
    # dataclass would drop exactly the audit detail the trace exists to carry.
    trace = remote.run_scene("incident-triage", ALERT)
    assert isinstance(trace, dict)
    assert trace["state_encodes"] == 1


def test_remote_surfaces_the_server_s_error_code(remote: JefClient) -> None:
    with pytest.raises(JefHTTPError) as exc:
        remote.evaluate(ALERT, {"q": {"type": "freeform", "instructions": "寫一首詩"}})
    assert exc.value.status == 422
    assert exc.value.code  # branchable without parsing prose


def test_remote_discovery(remote: JefClient) -> None:
    assert any(s["id"] == "incident-triage" for s in remote.scenes())
    assert remote.models()[0]["id"]
    assert remote.limits()["max_questions_per_request"] > 0
    assert remote.health()["status"] == "ok"
