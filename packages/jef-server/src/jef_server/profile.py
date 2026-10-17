"""The production profile: refuse to start on a configuration that is unsafe.

``JEF_PRODUCTION=1`` turns every "acceptable in a lab" default into a startup
error. The checks are the same facts ``/v1/readiness`` reports, applied at the
one moment they can still be fixed cheaply: before the process answers a
request. A server that starts in production mode therefore cannot be the
hashing stub, cannot be unauthenticated, cannot be unmetered, cannot serve a
model its manifest does not vouch for, and cannot record authorisations in
memory while claiming to be production.

Nothing here is a readiness substitute. Passing these checks says the
configuration is not insecure; ``/v1/readiness`` still decides, per required
task, whether the model is calibrated for what it is asked.
"""

from __future__ import annotations

from jef_core import Engine
from jef_core.model_manifest import Verification

from .settings import Settings

__all__ = ["ProductionConfigError", "enforce_production", "production_violations"]


class ProductionConfigError(RuntimeError):
    """Raised at startup when ``JEF_PRODUCTION`` is set and the config is unsafe."""


def production_violations(settings: Settings, manifest: Verification, engine: Engine) -> list[str]:
    """Every reason this configuration must not serve as production. Empty is good."""
    v: list[str] = []
    if settings.is_test_backbone or engine.backbone.name == "hashing":
        v.append("backbone: the hashing test backbone cannot serve production (set JEF_BACKBONE)")
    if not settings.head_path:
        v.append("model: JEF_HEAD_PATH is unset; production serves a trained head, not zero-shot")
    if manifest.status != "verified":
        detail = "; ".join(manifest.problems) or manifest.status
        v.append(f"model_manifest: {detail} (production requires a verified manifest.json)")
    if not settings.auth_enabled and not settings.trusted_ingress:
        v.append(
            "auth: JEF_API_KEYS is empty and JEF_TRUSTED_INGRESS is not asserted; "
            "production is never unauthenticated"
        )
    if settings.rate_limit_per_minute <= 0:
        v.append("rate_limit: JEF_RATE_LIMIT_PER_MINUTE is 0; production is never unmetered")
    if settings.metrics_public:
        v.append("metrics: JEF_METRICS_PUBLIC exposes every scene name; keep /metrics behind a key")
    if not settings.required_task_ids:
        v.append(
            "required_tasks: JEF_REQUIRED_TASK_IDS is empty; name the tasks this deployment "
            "must be calibrated for so readiness can be judged"
        )
    if settings.decisions_enabled:
        if not settings.grant_secret:
            v.append("decisions: JEF_GRANT_SECRET is empty; every grant would be refused")
        if not settings.proposal_secret:
            v.append(
                "decisions: JEF_PROPOSAL_SECRET is empty; proposals would not survive a "
                "restart or work across replicas"
            )
        if not settings.decision_store_persistent:
            v.append(
                "decisions: receipts and grants live in process memory; production "
                "requires the persistent store (JEF_DECISION_STORE)"
            )
    return v


def enforce_production(settings: Settings, manifest: Verification, engine: Engine) -> None:
    """Raise :class:`ProductionConfigError` listing every violation, or return."""
    if not settings.production:
        return
    problems = production_violations(settings, manifest, engine)
    if problems:
        bullet = "\n  - ".join(problems)
        raise ProductionConfigError(
            f"JEF_PRODUCTION is set but the configuration is not production-safe:\n  - {bullet}"
        )
