"""The scene HTTP surface -- JEF's own layer, not the compatible one."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jef_core import Engine
from jef_scene import SceneEngine, SceneRegistry, load_scene
from jef_server.app import create_app
from jef_server.settings import Settings

SCENES_DIR = Path(__file__).resolve().parents[3] / "scenes"


@pytest.fixture
def client() -> TestClient:
    engine = Engine()
    scene_engine = SceneEngine(engine)
    registry = SceneRegistry()
    scene = load_scene(SCENES_DIR / "incident-triage.zh-tw.yaml")
    registry.add(scene, compile_with=scene_engine)
    app = create_app(engine=engine, settings=Settings(), scenes=(scene_engine, registry))
    return TestClient(app)


ALERT = (
    "CrowdStrike 於 02:14 偵測到主機 db-core-02 出現大量檔案加密行為，"
    "副檔名遭統一改為 .lockbit，並刪除磁碟區陰影複製。"
)


def test_scenes_are_discoverable(client: TestClient) -> None:
    body = client.get("/v1/scenes").json()
    entry = next(e for e in body["data"] if e["id"] == "incident-triage")
    assert entry["layers"] == 3
    assert entry["questions"] == 5


def test_scene_evaluation_returns_a_full_trace(client: TestClient) -> None:
    r = client.post("/v1/scenes/incident-triage:evaluate", json={"state": ALERT})
    assert r.status_code == 200
    trace = r.json()

    assert trace["scene"] == "incident-triage"
    assert trace["verdict"] in ("decided", "fallthrough")
    assert trace["action"] in (
        "close",
        "request_context",
        "assign",
        "page_oncall",
        "escalate_human",
        "queue_for_review",
    )
    # The whole point: one state read for the entire playbook.
    assert trace["state_encodes"] == 1


def test_trace_carries_the_evidence_an_audit_needs(client: TestClient) -> None:
    trace = client.post("/v1/scenes/incident-triage:evaluate", json={"state": ALERT}).json()
    layer = trace["layers"][0]
    question = layer["questions"][0]
    for key in ("id", "kind", "instructions", "allowed_answers", "answer", "confidence"):
        assert key in question, f"trace must record {key}"
    assert "p_correct" in question, "a reviewer must see whether reliability is known"
    gate = layer["gates"][0]
    for key in ("index", "condition", "fired", "is_else"):
        assert key in gate


def test_uncalibrated_scene_always_reaches_a_human(client: TestClient) -> None:
    """Automation requires calibration. Without it every path ends at a person."""
    trace = client.post("/v1/scenes/incident-triage:evaluate", json={"state": ALERT}).json()
    assert trace["calibrated"] is False
    assert trace["human_review"] is True


def test_unknown_scene_is_rejected_with_the_list_of_known_ones(client: TestClient) -> None:
    r = client.post("/v1/scenes/nope:evaluate", json={"state": "x"})
    assert r.status_code == 422
    assert "incident-triage" in r.json()["error"]["message"]


def test_missing_state_is_rejected(client: TestClient) -> None:
    r = client.post("/v1/scenes/incident-triage:evaluate", json={})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_object_state_is_accepted(client: TestClient) -> None:
    r = client.post(
        "/v1/scenes/incident-triage:evaluate",
        json={"state": {"alert": ALERT, "host": "db-core-02", "vendor": "CrowdStrike"}},
    )
    assert r.status_code == 200
    assert r.json()["state_encodes"] == 1


def test_scene_metrics_are_exported(client: TestClient) -> None:
    client.post("/v1/scenes/incident-triage:evaluate", json={"state": ALERT})
    text = client.get("/metrics").text
    assert "jef_scene_runs_total" in text
    # A deployment where this counter never moves has stopped escalating.
    assert "jef_scene_human_review_total" in text


# --------------------------------------------------------------------------- #
# Debug UI
# --------------------------------------------------------------------------- #

UI_DIR = Path(__file__).resolve().parents[3] / "packages" / "jef-ui"


def _ui_client() -> TestClient:
    engine = Engine()
    scene_engine = SceneEngine(engine)
    registry = SceneRegistry()
    registry.add(load_scene(SCENES_DIR / "incident-triage.zh-tw.yaml"), compile_with=scene_engine)
    app = create_app(
        engine=engine,
        settings=Settings(ui_dir=str(UI_DIR)),
        scenes=(scene_engine, registry),
    )
    return TestClient(app)


def test_ui_is_served_when_configured() -> None:
    r = _ui_client().get("/ui")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_ui_is_self_contained() -> None:
    """No build step, no CDN: a debugger that needs the network during an
    incident is a debugger you cannot use during an incident."""
    html = UI_DIR.joinpath("index.html").read_text(encoding="utf-8")
    assert 'src="http' not in html
    assert 'href="http' not in html
    assert "cdn" not in html.lower()


def test_ui_only_calls_endpoints_the_server_actually_serves(client: TestClient) -> None:
    html = UI_DIR.joinpath("index.html").read_text(encoding="utf-8")
    for path in ("/healthz", "/v1/scenes"):
        assert path in html, f"the UI should read {path}"
        assert client.get(path).status_code == 200


def test_ui_route_absent_when_not_configured(client: TestClient) -> None:
    assert client.get("/ui").status_code == 404


def test_missing_ui_directory_fails_at_startup() -> None:
    # An operator who configured a UI and gets none should find out now.
    with pytest.raises(RuntimeError, match=r"no index\.html"):
        create_app(engine=Engine(), settings=Settings(ui_dir="/nonexistent/jef-ui"))
