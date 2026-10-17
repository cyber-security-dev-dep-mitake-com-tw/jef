"""/v1/decisions: authorize a proposed action, record what the platform did.

End to end through HTTP: run a scene with a case id, redeem the sealed
proposal, and check that forged tokens, forged grants and missing grants are
refused, that a retried step authorizes once, and that a timeout is recorded
as unknown.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jef_core import Engine
from jef_scene import SceneEngine, SceneRegistry, load_scene
from jef_scene.authz import AuthorizationGrant
from jef_server.app import create_app
from jef_server.settings import Settings

SCENES = Path(__file__).resolve().parents[3] / "scenes"
KEY = {"X-API-Key": "s3cret"}
GRANT_SECRET = b"approval-service"
ALERT = "IDS 回報來自 198.51.100.23 的連接埠掃描，該位址屬於委外弱點掃描服務，掃描視窗已核准。"


def make(**overrides) -> TestClient:
    engine = Engine()
    scene_engine = SceneEngine(engine)
    registry = SceneRegistry()
    registry.add(load_scene(SCENES / "incident-triage.zh-tw.yaml"), compile_with=scene_engine)
    settings = Settings(
        api_keys=(("soc", "s3cret"),),
        decisions_enabled=True,
        grant_secret=GRANT_SECRET,
        proposal_secret=b"proposal-secret",
        **overrides,
    )
    return TestClient(create_app(engine=engine, settings=settings, scenes=(scene_engine, registry)))


def run(client: TestClient, **extra) -> dict:
    r = client.post(
        "/v1/scenes/incident-triage:evaluate", json={"state": ALERT, **extra}, headers=KEY
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_disabled_by_default() -> None:
    engine = Engine()
    client = TestClient(create_app(engine=engine, settings=Settings()))
    r = client.post(
        "/v1/decisions:authorize", json={"proposal_token": "x", "idempotency_key": "k" * 8}
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "decisions_disabled"


def test_enabling_needs_api_keys() -> None:
    with pytest.raises(RuntimeError, match="JEF_API_KEYS"):
        create_app(engine=Engine(), settings=Settings(decisions_enabled=True))


def test_a_scene_run_with_a_case_returns_a_sealed_proposal() -> None:
    client = make()
    without = run(client)
    assert "proposal_token" not in without, "no case, no proposal"
    trace = run(client, case_id="case-42", asset_id="host-7")
    assert trace["proposal_token"] and trace["trace_id"]


def test_an_unauthorized_proposal_needs_a_grant() -> None:
    """The uncalibrated scene escalates: review_required, so no grant means no go."""
    client = make()
    trace = run(client, case_id="case-42", asset_id="host-7")
    assert trace["policy_verdict"] == "review_required"
    r = client.post(
        "/v1/decisions:authorize",
        json={"proposal_token": trace["proposal_token"], "idempotency_key": "case-42:a:1"},
        headers=KEY,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "not_dispatched"
    assert "NO_GRANT" in body["reason_codes"]


def test_full_flow_with_a_grant_and_idempotency() -> None:
    client = make()
    trace = run(client, case_id="case-42", asset_id="host-7")
    grant = AuthorizationGrant.issue(
        secret=GRANT_SECRET,
        case_id="case-42",
        action=trace["action"],
        asset_id="host-7",
        approvers=("analyst-anon-01",),
        approver_role="soc-lead",
        issuer="approval-service",
    ).to_dict()
    body = {
        "proposal_token": trace["proposal_token"],
        "idempotency_key": "case-42:act:host-7",
        "grant": grant,
    }
    first = client.post("/v1/decisions:authorize", json=body, headers=KEY).json()
    assert first["status"] == "authorized", first
    for _ in range(5):  # POL-005: a retried step gets the same answer
        assert client.post("/v1/decisions:authorize", json=body, headers=KEY).json() == first

    done = client.post(
        "/v1/decisions/case-42:act:host-7:complete",
        json={"outcome": "timeout", "detail": "EDR did not answer"},
        headers=KEY,
    ).json()
    assert done["status"] == "unknown", "POL-007: a timeout is never success"
    again = client.post(
        "/v1/decisions/case-42:act:host-7:complete", json={"outcome": "success"}, headers=KEY
    )
    assert again.status_code == 409
    got = client.get("/v1/decisions/case-42:act:host-7", headers=KEY).json()
    assert got["status"] == "unknown"


def test_pol_006_forged_grant_and_forged_token_are_refused() -> None:
    client = make()
    trace = run(client, case_id="case-42", asset_id="host-7")
    forged = client.post(
        "/v1/decisions:authorize",
        json={
            "proposal_token": trace["proposal_token"],
            "idempotency_key": "case-42:f:1",
            "grant": {"approved": True},
        },
        headers=KEY,
    ).json()
    assert forged["status"] == "not_dispatched"
    assert forged["reason_codes"] == ["INVALID_GRANT"]

    tampered = trace["proposal_token"][:-4] + "0000"
    r = client.post(
        "/v1/decisions:authorize",
        json={"proposal_token": tampered, "idempotency_key": "case-42:t:1"},
        headers=KEY,
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_proposal"


def test_requests_are_validated_and_authenticated() -> None:
    client = make()
    assert client.post("/v1/decisions:authorize", json={}).status_code == 401
    r = client.post(
        "/v1/decisions:authorize",
        json={"proposal_token": "x", "idempotency_key": "short"},
        headers=KEY,
    )
    assert r.status_code == 422
    assert client.get("/v1/decisions/unknown-key", headers=KEY).status_code == 404
    r = client.post("/v1/decisions/unknown-key:complete", json={"outcome": "success"}, headers=KEY)
    assert r.status_code == 404
