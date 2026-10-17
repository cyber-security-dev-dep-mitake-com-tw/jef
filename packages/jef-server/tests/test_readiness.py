"""NEXT-01: readiness is decided per required task, fail-closed.

The old ``production_safe`` flag used ``is_fitted()``, which only says that some
calibration bucket exists. These tests pin the cases where that was wrong:
legacy artifacts, incompatible artifacts, buckets for another task, missing
required tasks, missing auth, the hashing backbone, a manifest mismatch. Each
must block automation, and the report must say why. Only when every fact is
verified does ``automation_ready`` turn true -- and even then the response
says a T2+ action still needs a grant.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from jef_core import Calibrator, Engine
from jef_core.backends.hashing import HashingBackbone
from jef_core.calibration import Bucket, bucket_key
from jef_core.model_manifest import Verification
from jef_server.app import create_app
from jef_server.readiness import assess, task_status
from jef_server.settings import RequiredTask, Settings, parse_required_tasks

ALPHA = 0.10
TASK = "soc.alert.verdict.v1"
OTHER = "soc.ndr.technique.v1"
KEYS = (("soc", "s3cret"),)


class RealLookingBackbone(HashingBackbone):
    """Same maths as the test backbone, but not named ``hashing``."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "fake-mmbert"


def validated_bucket(task_id: str | None, n: int = 4) -> Bucket:
    return Bucket(
        temperature=1.0,
        conformal={f"{ALPHA:.4f}": 0.5},
        correctness_map=[[0.0, 0.5], [1.0, 0.9]],
        n_samples=300,
        task_id=task_id,
    )


def calibrator(schema: int = 2, **buckets: Bucket) -> Calibrator:
    cal = Calibrator(schema=schema)
    for key, bucket in buckets.items():
        cal.buckets[key] = bucket
    return cal


def make(
    *,
    engine: Engine,
    manifest: str = "verified",
    persistent: bool = False,
    **overrides,
) -> TestClient:
    # The settings' backbone name is one of the two facts readiness checks; mirror
    # the engine's so a test that wants a real-looking backbone gets one.
    overrides.setdefault("backbone", engine.backbone.name)
    settings = Settings(**overrides)
    app = create_app(engine=engine, settings=settings)
    app.state.manifest = Verification(manifest)  # the fact under test, not the file system
    app.state.policy_store_persistent = persistent
    return TestClient(app, raise_server_exceptions=True)


def report(client: TestClient) -> dict:
    headers = {"Authorization": "Bearer s3cret"}
    r = client.get("/v1/readiness", headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_required_task_specs_parse_both_forms() -> None:
    tasks = parse_required_tasks(f"{TASK}, {OTHER}:choice:6 ,")
    assert tasks == (RequiredTask(TASK), RequiredTask(OTHER, "choice", 6))
    assert tasks[1].spec == f"{OTHER}:choice:6"
    with pytest.raises(ValueError):
        parse_required_tasks("bad:shape")


# --------------------------------------------------------------------------- #
# The default deployment is honest about itself
# --------------------------------------------------------------------------- #


def test_default_deployment_is_live_but_ready_for_nothing() -> None:
    client = make(engine=Engine(), manifest="absent")
    r = client.get("/v1/readiness").json()
    assert r["liveness"] == "ok"
    assert r["decision_support_ready"] is False
    assert r["automation_ready"] is False
    assert "backbone:test" in r["blocking_reasons"]
    assert "auth:none" in r["blocking_reasons"]
    assert "required_tasks:none_configured" in r["blocking_reasons"]
    assert "policy_store:not_persistent" in r["blocking_reasons"]
    assert r["calibration_artifact"] == "absent"
    assert client.get("/healthz").json()["production_safe"] is False


def test_readyz_is_open_and_fails_on_manifest_mismatch() -> None:
    client = make(engine=Engine(), manifest="verified", api_keys=KEYS)
    assert client.get("/readyz").status_code == 200, "no key needed for a probe"
    client.app.state.manifest = Verification("mismatch", ["calibration.json sha256 differs"])
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["ready"] is False
    assert "calibration.json sha256 differs" in r.json()["problems"]


# --------------------------------------------------------------------------- #
# Every way is_fitted() lied, blocked individually
# --------------------------------------------------------------------------- #


def good_engine(cal: Calibrator) -> Engine:
    return Engine(RealLookingBackbone(), calibrator=cal, model_name="jef-soc-test")


def test_legacy_schema_1_artifact_never_reaches_ready() -> None:
    cal = calibrator(schema=1, **{bucket_key("choice", 4): validated_bucket(None)})
    assert cal.is_fitted(), "the old flag would have passed this"
    client = make(
        engine=good_engine(cal),
        persistent=True,
        api_keys=KEYS,
        decisions_enabled=True,
        required_task_ids=(RequiredTask(TASK, "choice", 4),),
    )
    r = report(client)
    assert r["automation_ready"] is False
    assert r["calibration_artifact"] == "legacy"
    assert f"{TASK}:legacy_calibration" in r["blocking_reasons"]
    assert "calibration:legacy" in r["blocking_reasons"]
    assert r["decision_support_ready"] is True, "advisory use is still fine"


def test_a_bucket_for_another_task_does_not_count() -> None:
    cal = calibrator(**{bucket_key("choice", 4, OTHER): validated_bucket(OTHER)})
    client = make(
        engine=good_engine(cal),
        persistent=True,
        api_keys=KEYS,
        decisions_enabled=True,
        required_task_ids=(RequiredTask(TASK),),
    )
    r = report(client)
    assert r["validated_tasks"] == []
    assert f"{TASK}:missing_calibration" in r["blocking_reasons"]
    assert r["automation_ready"] is False


def test_a_shape_scoped_bucket_does_not_satisfy_a_task() -> None:
    cal = calibrator(**{bucket_key("choice", 4): validated_bucket(None)})
    assert task_status(good_engine(cal), RequiredTask(TASK, "choice", 4), ALPHA) == "missing"


def test_insufficient_bucket_blocks() -> None:
    thin = Bucket(temperature=1.0, conformal={}, correctness_map=[], n_samples=12, task_id=TASK)
    cal = calibrator(**{bucket_key("choice", 4, TASK): thin})
    client = make(
        engine=good_engine(cal),
        persistent=True,
        api_keys=KEYS,
        decisions_enabled=True,
        required_task_ids=(RequiredTask(TASK, "choice", 4),),
    )
    assert f"{TASK}:insufficient_calibration" in report(client)["blocking_reasons"]


def test_incompatible_artifact_blocks() -> None:
    cal = calibrator(**{bucket_key("choice", 4, TASK): validated_bucket(TASK)})
    cal.artifact.model_sha256 = "a" * 64
    cal.runtime_model_sha256 = "b" * 64
    client = make(
        engine=good_engine(cal),
        persistent=True,
        api_keys=KEYS,
        decisions_enabled=True,
        required_task_ids=(RequiredTask(TASK, "choice", 4),),
    )
    r = report(client)
    assert r["calibration_artifact"] == "incompatible"
    assert f"{TASK}:incompatible_calibration" in r["blocking_reasons"]


def test_no_auth_blocks_unless_ingress_is_declared_trusted() -> None:
    cal = calibrator(**{bucket_key("choice", 4, TASK): validated_bucket(TASK)})
    blocked = make(
        engine=good_engine(cal),
        persistent=True,
        required_task_ids=(RequiredTask(TASK, "choice", 4),),
    )
    r = blocked.get("/v1/readiness").json()
    assert "auth:none" in r["blocking_reasons"]
    assert r["decision_support_ready"] is False

    trusted = make(
        engine=good_engine(cal),
        persistent=True,
        trusted_ingress=True,
        required_task_ids=(RequiredTask(TASK, "choice", 4),),
    )
    r = trusted.get("/v1/readiness").json()
    assert "auth:none" not in r["blocking_reasons"]
    assert r["checks"]["auth"] == "trusted-ingress", "the assertion is visible, never silent"


def test_hashing_backbone_blocks_even_when_everything_else_is_right() -> None:
    cal = calibrator(**{bucket_key("choice", 4, TASK): validated_bucket(TASK)})
    engine = Engine(calibrator=cal)  # the default backbone is the hashing stub
    client = make(
        engine=engine,
        persistent=True,
        api_keys=KEYS,
        decisions_enabled=True,
        required_task_ids=(RequiredTask(TASK, "choice", 4),),
    )
    r = report(client)
    assert r["checks"]["test_backbone"] is True
    assert "backbone:test" in r["blocking_reasons"]
    assert r["decision_support_ready"] is False


def test_manifest_mismatch_blocks_decision_support_too() -> None:
    cal = calibrator(**{bucket_key("choice", 4, TASK): validated_bucket(TASK)})
    client = make(
        engine=good_engine(cal),
        manifest="mismatch",
        persistent=True,
        api_keys=KEYS,
        decisions_enabled=True,
        required_task_ids=(RequiredTask(TASK, "choice", 4),),
    )
    r = report(client)
    assert "model_manifest:mismatch" in r["blocking_reasons"]
    assert r["decision_support_ready"] is False


def test_memory_policy_store_and_disabled_decisions_block_automation_only() -> None:
    cal = calibrator(**{bucket_key("choice", 4, TASK): validated_bucket(TASK)})
    client = make(
        engine=good_engine(cal),
        persistent=False,
        api_keys=KEYS,
        required_task_ids=(RequiredTask(TASK, "choice", 4),),
    )
    r = report(client)
    assert r["decision_support_ready"] is True
    assert r["automation_ready"] is False
    assert {"policy_store:not_persistent", "decisions:disabled"} <= set(r["blocking_reasons"])


# --------------------------------------------------------------------------- #
# All conditions met: ready, and still no permission for T2+
# --------------------------------------------------------------------------- #


def test_everything_verified_is_automation_ready_and_still_not_a_grant() -> None:
    cal = calibrator(
        **{
            bucket_key("choice", 4, TASK): validated_bucket(TASK),
            bucket_key("noul", 2, OTHER): validated_bucket(OTHER, 2),
            bucket_key("choice", 6, OTHER): validated_bucket(OTHER, 6),
        }
    )
    client = make(
        engine=good_engine(cal),
        persistent=True,
        api_keys=KEYS,
        decisions_enabled=True,
        required_task_ids=(RequiredTask(TASK, "choice", 4), RequiredTask(OTHER)),
    )
    r = report(client)
    assert r["blocking_reasons"] == []
    assert r["automation_ready"] is True
    assert r["decision_support_ready"] is True
    assert sorted(r["validated_tasks"]) == sorted([TASK, OTHER])
    assert "grant" in r["note"]
    assert client.get("/healthz").json()["production_safe"] is True


def test_one_unvalidated_bucket_of_a_required_task_blocks_that_task() -> None:
    thin = Bucket(temperature=1.0, conformal={}, correctness_map=[], n_samples=5, task_id=OTHER)
    cal = calibrator(
        **{
            bucket_key("noul", 2, OTHER): validated_bucket(OTHER, 2),
            bucket_key("choice", 6, OTHER): thin,
        }
    )
    assert task_status(good_engine(cal), RequiredTask(OTHER), ALPHA) == "insufficient"


def test_assess_is_pure_and_reports_every_reason_at_once() -> None:
    r = assess(
        Engine(),
        Settings(),
        Verification("absent"),
        policy_store_persistent=False,
        decisions_enabled=False,
    )
    assert r.automation_ready is False
    assert len(r.blocking_reasons) >= 5
    assert np.isfinite(len(r.checks["calibration_buckets"]))
