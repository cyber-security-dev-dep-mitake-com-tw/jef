"""Ingress guards: unauthenticated is 401/403, oversized is 413 before inference, floods are 429.

Every rejection here must happen without the model running. The tests prove
that with the engine's own encode counter: a rejected request leaves it where
it was.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from jef_core import Engine
from jef_server.app import create_app
from jef_server.security import RateLimiter, authenticate, json_depth, parse_api_keys
from jef_server.settings import Settings

Q = {"urgent": {"type": "noul", "instructions": "緊急？"}}


def make_client(**overrides) -> tuple[TestClient, Engine]:
    engine = Engine()
    settings = Settings(**overrides)
    client = TestClient(create_app(engine=engine, settings=settings), raise_server_exceptions=True)
    return client, engine


# --------------------------------------------------------------------------- #
# API-002: authentication
# --------------------------------------------------------------------------- #


def test_default_has_no_auth_and_says_so() -> None:
    client, _ = make_client()
    assert client.post("/v1/systemone", json={"state": "x", "questions": Q}).status_code == 200
    health = client.get("/healthz").json()
    assert health["auth"] == "none"
    assert health["production_safe"] is False


def test_api_002_protected_endpoints_need_a_key() -> None:
    client, engine = make_client(api_keys=(("soc", "s3cret"),))
    for path, method in [
        ("/v1/systemone", "POST"),
        ("/v1/scenes", "GET"),
        ("/v1/models", "GET"),
        ("/v1/limits", "GET"),
        ("/metrics", "GET"),
    ]:
        r = client.request(method, path, json={"state": "x", "questions": Q})
        assert r.status_code == 401, (path, r.status_code)
        assert r.headers["www-authenticate"] == "Bearer"
        assert r.json()["error"]["code"] == "unauthenticated"
    assert engine.encode_count == 0, "a rejected request must not run the model"


def test_healthz_stays_open_for_probes() -> None:
    client, _ = make_client(api_keys=(("soc", "s3cret"),))
    assert client.get("/healthz").status_code == 200


def test_wrong_key_is_forbidden_right_key_is_accepted() -> None:
    client, _ = make_client(api_keys=(("soc", "s3cret"),))
    body = {"state": "x", "questions": Q}
    assert client.post("/v1/systemone", json=body, headers={"X-API-Key": "nope"}).status_code == 403
    assert (
        client.post(
            "/v1/systemone", json=body, headers={"Authorization": "Bearer s3cret"}
        ).status_code
        == 200
    )
    assert (
        client.post("/v1/systemone", json=body, headers={"X-API-Key": "s3cret"}).status_code == 200
    )
    assert client.get("/metrics", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_metrics_can_be_left_public() -> None:
    client, _ = make_client(api_keys=(("soc", "s3cret"),), metrics_public=True)
    assert client.get("/metrics").status_code == 200
    assert client.get("/v1/models").status_code == 401


def test_require_auth_without_keys_refuses_to_start() -> None:
    with pytest.raises(RuntimeError, match="JEF_REQUIRE_AUTH"):
        create_app(engine=Engine(), settings=Settings(require_auth=True))


def test_parse_api_keys_labels_and_digests() -> None:
    keys = parse_api_keys(" soc-prod:abc , def,, :ghi ")
    assert keys["soc-prod"] == "abc"
    assert keys["key-" + __import__("hashlib").sha256(b"def").hexdigest()[:8]] == "def"
    assert "ghi" in keys.values()
    assert parse_api_keys(None) == {}


def test_authenticate_outcomes() -> None:
    keys = {"a": "one", "b": "two"}
    assert authenticate({}, keys) == ("missing", None)
    assert authenticate({"authorization": "Bearer three"}, keys) == ("invalid", None)
    assert authenticate({"authorization": "Basic one"}, keys) == ("missing", None)
    assert authenticate({"authorization": "Bearer two"}, keys) == ("ok", "b")
    assert authenticate({"x-api-key": "one"}, keys) == ("ok", "a")


# --------------------------------------------------------------------------- #
# API-001: input budgets, enforced before inference
# --------------------------------------------------------------------------- #


def test_api_001_oversized_body_is_413_before_the_model_runs() -> None:
    client, engine = make_client(max_body_bytes=200)
    big = {"state": "x" * 1000, "questions": Q}
    r = client.post("/v1/systemone", json=big)
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "body_too_large"
    assert engine.encode_count == 0


def test_api_001_oversized_state_is_413_before_the_model_runs() -> None:
    client, engine = make_client(max_state_chars=50)
    r = client.post("/v1/systemone", json={"state": "x" * 51, "questions": Q})
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "state_too_large"
    assert engine.encode_count == 0
    # Object states are measured as rendered.
    r = client.post("/v1/systemone", json={"state": {"alert": "y" * 60}, "questions": Q})
    assert r.status_code == 413
    assert client.post("/v1/systemone", json={"state": "x" * 50, "questions": Q}).status_code == 200


def test_api_001_deep_json_is_422_before_the_model_runs() -> None:
    client, engine = make_client(max_json_depth=5)
    nested: dict = {"leaf": 1}
    for _ in range(10):
        nested = {"n": nested}
    r = client.post("/v1/systemone", json={"state": nested, "questions": Q})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "payload_too_deep"
    assert engine.encode_count == 0


def test_json_depth() -> None:
    assert json_depth("x") == 1
    assert json_depth({"a": 1}) == 2
    assert json_depth({"a": [1, {"b": 2}]}) == 4
    assert json_depth([]) == 1


def test_chunked_and_compressed_bodies_are_refused() -> None:
    client, engine = make_client()
    raw = json.dumps({"state": "x", "questions": Q}).encode()
    r = client.post(
        "/v1/systemone",
        content=raw,
        headers={"content-type": "application/json", "content-encoding": "gzip"},
    )
    assert r.status_code == 415

    def chunks():
        yield raw

    r = client.post("/v1/systemone", content=chunks(), headers={"content-type": "application/json"})
    assert r.status_code == 411
    assert engine.encode_count == 0


def test_limits_endpoint_advertises_the_budgets() -> None:
    client, _ = make_client(max_body_bytes=123, rate_limit_per_minute=7)
    limits = client.get("/v1/limits").json()
    assert limits["max_body_bytes"] == 123
    assert limits["rate_limit_per_minute"] == 7
    assert limits["auth"] == "none"
    assert "max_json_depth" in limits


# --------------------------------------------------------------------------- #
# 429
# --------------------------------------------------------------------------- #


def test_rate_limit_returns_429_with_retry_after() -> None:
    client, engine = make_client(rate_limit_per_minute=2)
    body = {"state": "x", "questions": Q}
    assert client.post("/v1/systemone", json=body).status_code == 200
    assert client.post("/v1/systemone", json=body).status_code == 200
    before = engine.encode_count
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 429
    assert int(r.headers["retry-after"]) >= 1
    assert engine.encode_count == before
    # The probe is never rate limited.
    assert client.get("/healthz").status_code == 200


def test_rate_limiter_windows_and_eviction() -> None:
    limiter = RateLimiter(2, max_callers=4)
    assert limiter.check("a", now=0.0) is None
    assert limiter.check("a", now=1.0) is None
    assert limiter.check("a", now=2.0) == 59
    assert limiter.check("a", now=61.0) is None, "a new minute resets the window"
    for i in range(4):
        limiter.check(f"c{i}", now=100.0)
    limiter.check("d", now=200.0)  # overflow: stale windows are evicted first
    assert "d" in limiter._windows
    assert all(k not in limiter._windows for k in ("a", "c0"))
    assert RateLimiter(0).check("anyone") is None


# --------------------------------------------------------------------------- #
# API-003: readiness
# --------------------------------------------------------------------------- #


def test_api_003_test_backbone_is_never_production_safe() -> None:
    client, _ = make_client(api_keys=(("soc", "s3cret"),))
    health = client.get("/healthz").json()
    assert health["status"] == "ok", "liveness is unchanged"
    assert health["test_backbone"] is True
    assert health["auth"] == "api-key"
    assert health["model_loaded"] is True
    assert health["calibration_scope"] == []
    assert health["production_safe"] is False


# --------------------------------------------------------------------------- #
# Model manifest at load
# --------------------------------------------------------------------------- #


def test_a_manifest_mismatch_drops_calibration_and_is_never_production_safe(tmp_path) -> None:
    import json
    import shutil
    from pathlib import Path

    from jef_core.model_manifest import build_manifest
    from jef_server.deps import build_engine, manifest_check

    shipped = (
        next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
        / "models"
        / "jef-v0"
    )
    d = tmp_path / "m"
    d.mkdir()
    for name in ("head.npz", "calibration.json", "report.json"):
        shutil.copy(shipped / name, d / name)
    (d / "manifest.json").write_text(json.dumps(build_manifest(d, model_id="m", version="0")))

    settings = Settings(head_path=str(d / "head.npz"), calibration_path=str(d / "calibration.json"))
    assert manifest_check(settings).status == "verified"
    assert build_engine(settings).calibrator.is_fitted()

    (d / "calibration.json").write_text(
        (d / "calibration.json").read_text().replace("0.", "0.0", 1)
    )
    assert manifest_check(settings).status == "mismatch"
    engine = build_engine(settings)
    assert not engine.calibrator.is_fitted(), "mismatched calibration must be dropped"

    client = TestClient(create_app(engine=engine, settings=settings))
    health = client.get("/healthz").json()
    assert health["model_manifest"] == "mismatch"
    assert health["production_safe"] is False


def test_no_head_means_no_manifest_to_check() -> None:
    client, _ = make_client()
    assert client.get("/healthz").json()["model_manifest"] == "absent"
