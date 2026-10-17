"""NEXT-05: JEF_PRODUCTION refuses to start on an insecure configuration.

Each test removes exactly one safety from an otherwise production-grade
configuration and asserts the server names it and does not start. The last
test is the one configuration that does start, to prove the profile is
satisfiable and not merely strict.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from jef_core import Calibrator, Engine
from jef_core.backends.hashing import HashingBackbone
from jef_core.calibration import Bucket, bucket_key
from jef_core.model_manifest import Verification
from jef_server.app import create_app
from jef_server.profile import ProductionConfigError, production_violations
from jef_server.settings import RequiredTask, Settings

TASK = "soc.alert.verdict.v1"


class RealLookingBackbone(HashingBackbone):
    def __init__(self) -> None:
        super().__init__()
        self.name = "fake-mmbert"


def engine() -> Engine:
    cal = Calibrator(schema=2)
    cal.buckets[bucket_key("choice", 4, TASK)] = Bucket(
        temperature=1.0,
        conformal={"0.1000": 0.5},
        correctness_map=[[0.0, 0.5], [1.0, 0.9]],
        n_samples=300,
        task_id=TASK,
    )
    return Engine(RealLookingBackbone(), calibrator=cal, model_name="jef-soc-test")


GOOD = dict(
    production=True,
    backbone="fake-mmbert",
    head_path="/models/jef-soc/head.npz",
    api_keys=(("soc", "s3cret"),),
    rate_limit_per_minute=60,
    required_task_ids=(RequiredTask(TASK, "choice", 4),),
)


def violations(manifest: str = "verified", **overrides) -> list[str]:
    return production_violations(
        Settings(**{**GOOD, **overrides}), Verification(manifest), engine()
    )


@pytest.mark.parametrize(
    ("overrides", "manifest", "needle"),
    [
        ({"backbone": "hashing"}, "verified", "backbone:"),
        ({"head_path": None}, "verified", "JEF_HEAD_PATH"),
        ({}, "absent", "model_manifest:"),
        ({}, "mismatch", "model_manifest:"),
        ({"api_keys": ()}, "verified", "auth:"),
        ({"rate_limit_per_minute": 0}, "verified", "rate_limit:"),
        ({"metrics_public": True}, "verified", "metrics:"),
        ({"required_task_ids": ()}, "verified", "required_tasks:"),
        ({"decisions_enabled": True}, "verified", "JEF_GRANT_SECRET"),
        (
            {"decisions_enabled": True, "grant_secret": b"g", "proposal_secret": b"p"},
            "verified",
            "persistent store",
        ),
    ],
)
def test_each_missing_safety_is_named(overrides, manifest, needle) -> None:
    found = violations(manifest, **overrides)
    assert any(needle in v for v in found), found


def test_the_good_configuration_has_no_violations() -> None:
    assert violations() == []


def test_trusted_ingress_satisfies_the_auth_check_visibly() -> None:
    assert not any("auth:" in v for v in violations(api_keys=(), trusted_ingress=True))


def test_hashing_backbone_is_caught_from_the_engine_too() -> None:
    found = production_violations(Settings(**GOOD), Verification("verified"), Engine())
    assert any("backbone:" in v for v in found)


def test_startup_refuses_and_lists_every_problem() -> None:
    settings = Settings(**{**GOOD, "api_keys": (), "rate_limit_per_minute": 0})
    with pytest.raises(ProductionConfigError) as e:
        create_app(engine=engine(), settings=settings)
    msg = str(e.value)
    assert "auth:" in msg and "rate_limit:" in msg
    assert "model_manifest:" in msg, "no manifest on disk for this fake head path"


def test_profile_off_keeps_the_lab_defaults_working() -> None:
    client = TestClient(create_app(engine=Engine(), settings=Settings()))
    assert client.get("/healthz").status_code == 200


def test_production_starts_when_everything_is_set(monkeypatch) -> None:
    # The manifest check reads the file system; stand in a verified result so the
    # test exercises the profile, not the loader.
    import jef_server.app as app_module

    monkeypatch.setattr(app_module, "manifest_check", lambda s: Verification("verified"))
    client = TestClient(create_app(engine=engine(), settings=Settings(**GOOD)))
    r = client.get("/v1/readiness", headers={"Authorization": "Bearer s3cret"}).json()
    assert r["decision_support_ready"] is True
    # Decisions are off and the store is memory, so automation stays blocked.
    assert r["automation_ready"] is False
    assert "decisions:disabled" in r["blocking_reasons"]
