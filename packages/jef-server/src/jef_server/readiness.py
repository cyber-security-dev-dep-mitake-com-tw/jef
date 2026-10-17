"""Readiness, in three strengths, for three different consumers.

``/healthz`` answers one question: does the process answer. A load balancer
needs nothing more, and it must never be read as permission to automate.

``/readyz`` answers whether the model is loaded and its artifacts agree with
their manifest, which is what an orchestrator needs before routing traffic.

``/v1/readiness`` answers the question an operator (or a policy engine) has to
ask before letting any decision leave the advisory lane: is this deployment
ready for decision support, and is it ready for automation. Both are decided
fail-closed from facts the server can verify about itself, and the report
names every reason it said no. Even ``automation_ready: true`` authorises
nothing by itself: a T2+ action still needs a grant through the policy engine.

The rules that matter most are the ones the old ``production_safe`` flag got
wrong. ``is_fitted()`` only says that *some* calibration bucket exists. A
schema-1 artifact (``legacy``), an artifact for another model
(``incompatible``), or a bucket for a different task all satisfy it and none
of them calibrate the question being asked. Readiness therefore checks the
status of each *required task* individually and never infers one task's
readiness from another's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jef_core import Engine
from jef_core.calibration import CalibrationStatus, split_bucket_key
from jef_core.model_manifest import Verification

from .settings import RequiredTask, Settings

__all__ = ["Readiness", "assess", "task_status"]


@dataclass(frozen=True)
class Readiness:
    liveness: str
    decision_support_ready: bool
    automation_ready: bool
    model_id: str
    model_manifest: str
    calibration_artifact: str
    validated_tasks: list[str]
    required_tasks: list[str]
    blocking_reasons: list[str]
    checks: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "liveness": self.liveness,
            "decision_support_ready": self.decision_support_ready,
            "automation_ready": self.automation_ready,
            "model_id": self.model_id,
            "model_manifest": self.model_manifest,
            "calibration_artifact": self.calibration_artifact,
            "validated_tasks": self.validated_tasks,
            "required_tasks": self.required_tasks,
            "blocking_reasons": self.blocking_reasons,
            "checks": self.checks,
            # Stated on every response so no client can forget it.
            "note": "automation_ready never authorises a T2+ action; those need a grant",
        }


def task_status(engine: Engine, task: RequiredTask, alpha: float) -> CalibrationStatus:
    """Calibration status of one required task, decided from its own buckets only.

    With an explicit ``kind:n`` the question is exact. Without one, every bucket
    fitted for that task must be ``validated``; a task with no bucket at all is
    ``missing``. A bucket for another task, or for the same shape without a task
    id, never counts: that is the ``is_fitted()`` mistake this module exists to
    stop.
    """
    cal = engine.calibrator
    if task.kind is not None and task.n_options is not None:
        return cal.status(task.kind, task.n_options, alpha, task_id=task.task_id)
    if cal.legacy:
        return "legacy"
    if cal.incompatible:
        return "incompatible"
    own = [
        (kind, n)
        for key in cal.buckets
        for (tid, kind, n) in [split_bucket_key(key)]
        if tid == task.task_id
    ]
    if not own:
        return "missing"
    statuses = [cal.status(kind, n, alpha, task_id=task.task_id) for kind, n in own]
    return (
        "validated"
        if all(s == "validated" for s in statuses)
        else next(s for s in statuses if s != "validated")
    )


def calibration_artifact_state(engine: Engine) -> str:
    cal = engine.calibrator
    if not cal.buckets:
        return "absent"
    if cal.legacy:
        return "legacy"
    if cal.incompatible:
        return "incompatible"
    return "task-scoped"


def assess(
    engine: Engine,
    settings: Settings,
    manifest: Verification,
    *,
    policy_store_persistent: bool,
    decisions_enabled: bool,
) -> Readiness:
    """Compute the readiness report. Every ``False`` comes with a reason."""
    reasons: list[str] = []
    test_backbone = settings.is_test_backbone or engine.backbone.name == "hashing"
    auth_ok = settings.auth_enabled or settings.trusted_ingress
    artifact = calibration_artifact_state(engine)

    if test_backbone:
        reasons.append("backbone:test")
    if manifest.status != "verified":
        reasons.append(f"model_manifest:{manifest.status}")
    if not auth_ok:
        reasons.append("auth:none")
    if artifact != "task-scoped":
        reasons.append(f"calibration:{artifact}")

    validated: list[str] = []
    if not settings.required_task_ids:
        reasons.append("required_tasks:none_configured")
    for task in settings.required_task_ids:
        status = task_status(engine, task, settings.alpha)
        if status == "validated":
            validated.append(task.task_id)
        else:
            reasons.append(f"{task.task_id}:{status}_calibration")

    if not decisions_enabled:
        reasons.append("decisions:disabled")
    if not policy_store_persistent:
        reasons.append("policy_store:not_persistent")

    # L1 (read-only advisory): a real model whose files are what the manifest
    # says, behind authentication. A manifest that is merely absent does not
    # block advisory use; a mismatch does, because the calibration was dropped.
    decision_support = (not test_backbone) and manifest.status != "mismatch" and auth_ok
    # L2 (T0/T1 automation): everything, and nothing inferred.
    automation = decision_support and not reasons

    return Readiness(
        liveness="ok",
        decision_support_ready=decision_support,
        automation_ready=automation,
        model_id=engine.model_name,
        model_manifest=manifest.status,
        calibration_artifact=artifact,
        validated_tasks=validated,
        required_tasks=[t.spec for t in settings.required_task_ids],
        blocking_reasons=reasons,
        checks={
            "test_backbone": test_backbone,
            "auth": "api-key"
            if settings.auth_enabled
            else ("trusted-ingress" if settings.trusted_ingress else "none"),
            "manifest_problems": list(manifest.problems),
            "calibration_schema": engine.calibrator.schema,
            "calibration_buckets": sorted(engine.calibrator.buckets),
            "decisions_enabled": decisions_enabled,
            "policy_store": "persistent" if policy_store_persistent else "memory",
        },
    )
