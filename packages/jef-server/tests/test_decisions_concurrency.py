"""Concurrency over HTTP: two server workers, one decision store.

Stands in for ``JEF_WORKERS=2`` on one host with a SQLite store: two app
instances, each with its own executor and connection, share only the
database file. SOAR retry storms and double spends are fired at both at once.
"""

from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path
from typing import Any

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
PER_WORKER = 6


def worker(dsn: str) -> TestClient:
    engine = Engine()
    scene_engine = SceneEngine(engine)
    registry = SceneRegistry()
    registry.add(load_scene(SCENES / "incident-triage.zh-tw.yaml"), compile_with=scene_engine)
    settings = Settings(
        api_keys=(("soc", "s3cret"),),
        decisions_enabled=True,
        grant_secret=GRANT_SECRET,
        proposal_secret=b"shared-proposal-secret",  # shared, as across real replicas
        decision_store=dsn,
    )
    return TestClient(create_app(engine=engine, settings=settings, scenes=(scene_engine, registry)))


def proposal(client: TestClient, case: str = "case-42", asset: str = "host-7") -> dict[str, Any]:
    r = client.post(
        "/v1/scenes/incident-triage:evaluate",
        json={"state": ALERT, "case_id": case, "asset_id": asset},
        headers=KEY,
    )
    assert r.status_code == 200, r.text
    return r.json()


def grant(action: str, case: str = "case-42", asset: str = "host-7") -> dict[str, Any]:
    return AuthorizationGrant.issue(
        secret=GRANT_SECRET,
        case_id=case,
        action=action,
        asset_id=asset,
        approvers=("analyst-anon-01",),
        approver_role="soc-lead",
        issuer="approval-service",
    ).to_dict()


def fire(clients: list[TestClient], calls: list[tuple[int, str, dict[str, Any]]]) -> list[Any]:
    """POST every ``(client index, path, body)`` at once; return ``(status, json)`` in order."""
    barrier = threading.Barrier(len(calls))
    out: list[Any] = [None] * len(calls)
    errors: list[BaseException] = []

    def go(i: int, c: int, path: str, body: dict[str, Any]) -> None:
        try:
            barrier.wait(timeout=30)
            r = clients[c].post(path, json=body, headers=KEY)
            out[i] = (r.status_code, r.json())
        except BaseException as exc:  # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=go, args=(i, *call)) for i, call in enumerate(calls)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert errors == [], errors
    return out


def test_one_grant_two_workers_many_keys_one_authorization(tmp_path: Path) -> None:
    dsn = f"sqlite:///{tmp_path / 'decisions.db'}"
    clients = [worker(dsn), worker(dsn)]
    trace = proposal(clients[0])
    g = grant(trace["action"])
    calls = [
        (
            i % 2,
            "/v1/decisions:authorize",
            {
                "proposal_token": trace["proposal_token"],
                "idempotency_key": f"case-42:storm:{i:02d}",
                "grant": g,
            },
        )
        for i in range(2 * PER_WORKER)
    ]
    results = fire(clients, calls)
    assert all(code == 200 for code, _ in results), results
    statuses = Counter(body["status"] for _, body in results)
    assert statuses == {"authorized": 1, "not_dispatched": 2 * PER_WORKER - 1}, statuses
    refused = [b for _, b in results if b["status"] == "not_dispatched"]
    assert all(b["reason_codes"] == ["GRANT_ALREADY_USED"] for b in refused)


def test_a_retry_storm_on_one_key_across_workers_gets_one_receipt(tmp_path: Path) -> None:
    dsn = f"sqlite:///{tmp_path / 'decisions.db'}"
    clients = [worker(dsn), worker(dsn)]
    trace = proposal(clients[0])
    body = {
        "proposal_token": trace["proposal_token"],
        "idempotency_key": "case-42:retry:1",
        "grant": grant(trace["action"]),
    }
    results = fire(
        clients, [(i % 2, "/v1/decisions:authorize", body) for i in range(2 * PER_WORKER)]
    )
    bodies = [b for _, b in results]
    assert all(b == bodies[0] for b in bodies), "every retry sees the same receipt"
    assert bodies[0]["status"] == "authorized"


def test_racing_completions_across_workers_record_one_outcome(tmp_path: Path) -> None:
    dsn = f"sqlite:///{tmp_path / 'decisions.db'}"
    clients = [worker(dsn), worker(dsn)]
    trace = proposal(clients[0])
    key = "case-42:finish:1"
    first = (
        clients[0]
        .post(
            "/v1/decisions:authorize",
            json={
                "proposal_token": trace["proposal_token"],
                "idempotency_key": key,
                "grant": grant(trace["action"]),
            },
            headers=KEY,
        )
        .json()
    )
    assert first["status"] == "authorized"
    outcomes = ["success", "failed", "timeout"] * 4
    results = fire(
        clients,
        [(i % 2, f"/v1/decisions/{key}:complete", {"outcome": o}) for i, o in enumerate(outcomes)],
    )
    final = clients[1].get(f"/v1/decisions/{key}", headers=KEY).json()["status"]
    assert final in {"success", "failed", "unknown"}
    wanted = {"success": "success", "failed": "failed", "timeout": "unknown"}
    for outcome, (code, body) in zip(outcomes, results, strict=True):
        if wanted[outcome] == final:
            assert code == 200 and body["status"] == final
        else:
            assert code == 409 and body["error"]["code"] == "outcome_conflict"


def test_distinct_cases_across_workers_all_authorise(tmp_path: Path) -> None:
    dsn = f"sqlite:///{tmp_path / 'decisions.db'}"
    clients = [worker(dsn), worker(dsn)]
    calls = []
    for i in range(2 * PER_WORKER):
        case, asset = f"case-{i}", f"host-{i}"
        trace = proposal(clients[i % 2], case, asset)
        calls.append(
            (
                i % 2,
                "/v1/decisions:authorize",
                {
                    "proposal_token": trace["proposal_token"],
                    "idempotency_key": f"{case}:isolate",
                    "grant": grant(trace["action"], case, asset),
                },
            )
        )
    results = fire(clients, calls)
    assert [b["status"] for _, b in results] == ["authorized"] * len(calls)
