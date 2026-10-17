"""NEXT-04 over HTTP: receipts outlive the process, and a dead store fails closed."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jef_core import Engine
from jef_scene import SceneEngine, SceneRegistry, load_scene
from jef_scene.store import StoreUnavailable
from jef_server.app import create_app
from jef_server.settings import Settings

SCENES = Path(__file__).resolve().parents[3] / "scenes"
KEY = {"X-API-Key": "s3cret"}
ALERT = "IDS 回報來自 198.51.100.23 的連接埠掃描，該位址屬於委外弱點掃描服務，掃描視窗已核准。"


def make(dsn: str | None, **overrides) -> TestClient:
    engine = Engine()
    scene_engine = SceneEngine(engine)
    registry = SceneRegistry()
    registry.add(load_scene(SCENES / "incident-triage.zh-tw.yaml"), compile_with=scene_engine)
    settings = Settings(
        api_keys=(("soc", "s3cret"),),
        decisions_enabled=True,
        grant_secret=b"g",
        proposal_secret=b"proposal-secret",
        decision_store=dsn,
        **overrides,
    )
    return TestClient(create_app(engine=engine, settings=settings, scenes=(scene_engine, registry)))


def run(client: TestClient) -> dict:
    r = client.post(
        "/v1/scenes/incident-triage:evaluate",
        json={"state": ALERT, "case_id": "case-42", "asset_id": "host-7"},
        headers=KEY,
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_memory_store_is_reported_as_not_persistent() -> None:
    client = make(None)
    r = client.get("/v1/readiness", headers=KEY).json()
    assert "policy_store:not_persistent" in r["blocking_reasons"]
    assert r["checks"]["policy_store"] == "memory"


def test_a_decision_survives_a_restart(tmp_path: Path) -> None:
    dsn = f"sqlite:///{tmp_path / 'decisions.db'}"
    first = make(dsn)
    assert first.get("/v1/readiness", headers=KEY).json()["checks"]["policy_store"] == "persistent"
    trace = run(first)
    r = first.post(
        "/v1/decisions:authorize",
        json={"proposal_token": trace["proposal_token"], "idempotency_key": "case-42:a:1"},
        headers=KEY,
    )
    assert r.status_code == 200
    recorded = r.json()
    assert recorded["status"] == "not_dispatched", "review_required without a grant"

    second = make(dsn)  # the same database, a new process
    again = second.get("/v1/decisions/case-42:a:1", headers=KEY)
    assert again.status_code == 200
    assert again.json() == recorded


def test_a_bad_dsn_is_a_startup_error() -> None:
    with pytest.raises(StoreUnavailable):
        make("postgresql://nobody@nowhere/jef")


def test_a_dead_store_answers_503_and_authorises_nothing(tmp_path: Path) -> None:
    client = make(f"sqlite:///{tmp_path / 'd.db'}")
    trace = run(client)
    executor = client.app.state.decisions

    def dead(*_: object, **__: object) -> None:
        raise StoreUnavailable("disk gone")

    executor.store.get = dead  # type: ignore[method-assign]
    r = client.post(
        "/v1/decisions:authorize",
        json={"proposal_token": trace["proposal_token"], "idempotency_key": "case-42:b:1"},
        headers=KEY,
    )
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "store_unavailable"
    assert client.get("/v1/decisions/case-42:b:1", headers=KEY).status_code == 503
