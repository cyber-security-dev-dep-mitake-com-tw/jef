"""``/v1/decisions``: authorize an action, record what the platform did.

The two-phase half of the authorization contract (jef_scene.executor). A
connector runs a scene with a ``case_id`` (and the ``asset_id`` it would act
on) and gets a sealed ``proposal_token`` back. Before performing anything it
POSTs the token, an idempotency key and, for anything beyond an allowed T0/T1
action, a signed AuthorizationGrant to ``/v1/decisions:authorize``. Only on
``authorized`` does it perform the action, with its own platform credentials,
and then it reports the outcome to ``/v1/decisions/{key}:complete``.

JEF executes nothing here and holds no EDR, IAM or firewall credentials. It
decides whether an action may happen, once, and it keeps the receipt.

Receipts live in process memory: one replica, ``JEF_WORKERS=1``. A
multi-replica deployment needs a shared store, which this module does not
pretend to be.
"""

from __future__ import annotations

import logging
import re
import secrets
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from jef_scene.authz import ProposalError, ProposedDecision
from jef_scene.executor import Executor, OutcomeConflict, TransportResult
from prometheus_client import Counter

from .settings import Settings

log = logging.getLogger("jef.server.decisions")

__all__ = ["install_decisions", "seal_for_trace"]

DECISIONS = Counter(
    "jef_policy_decisions_total",
    "Authorization outcomes by action tier and status.",
    ["tier", "status"],
)
UNAUTHORIZED = Counter(
    "jef_unauthorized_action_attempts_total",
    "Authorize calls refused for a missing, invalid or ill-fitting grant. Should be 0.",
    ["reason"],
)

_KEY = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")


class _NoTransport:
    """The server never calls a platform. Present only to satisfy Executor."""

    def execute(self, *_: Any, **__: Any) -> TransportResult:  # pragma: no cover
        raise RuntimeError("jef-server performs no actions; use authorize/complete")


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def install_decisions(app: FastAPI, settings: Settings) -> None:
    """Attach /v1/decisions when enabled; otherwise every route answers 404."""
    if settings.decisions_enabled and not settings.auth_enabled:
        raise RuntimeError(
            "JEF_DECISIONS_ENABLED needs JEF_API_KEYS: these endpoints authorize actions and "
            "record what a platform did, and must not be reachable without a credential"
        )
    secret = settings.proposal_secret
    if settings.decisions_enabled and not secret:
        log.warning(
            "JEF_PROPOSAL_SECRET is empty: using a per-process secret; proposal tokens will not "
            "survive a restart or work across replicas"
        )
        secret = secrets.token_bytes(32)
    if settings.decisions_enabled and not settings.grant_secret:
        log.warning("JEF_GRANT_SECRET is empty: every AuthorizationGrant will be refused")

    executor = Executor("jef-server", _NoTransport(), grant_secret=settings.grant_secret or None)
    app.state.decisions = executor
    app.state.proposal_secret = secret if settings.decisions_enabled else b""

    def _disabled() -> JSONResponse:
        return _error(
            404, "decisions_disabled", "set JEF_DECISIONS_ENABLED=1 to serve /v1/decisions"
        )

    @app.post("/v1/decisions:authorize")
    def authorize(payload: dict[str, Any]) -> Any:
        if not settings.decisions_enabled:
            return _disabled()
        key = payload.get("idempotency_key")
        if not isinstance(key, str) or not _KEY.match(key):
            return _error(
                422, "invalid_request", "idempotency_key must be 8-128 of [A-Za-z0-9._:-]"
            )
        try:
            proposal = ProposedDecision.unseal(
                str(payload.get("proposal_token", "")),
                app.state.proposal_secret,
                max_age_seconds=settings.proposal_ttl_seconds,
            )
        except (ProposalError, ValueError, KeyError, TypeError) as exc:
            return _error(422, "invalid_proposal", str(exc))
        receipt = executor.authorize(
            proposal,
            idempotency_key=key,
            grant=payload.get("grant"),
            dry_run=bool(payload.get("dry_run", False)),
        )
        DECISIONS.labels(tier=str(proposal.action_tier), status=receipt.status).inc()
        if receipt.status == "not_dispatched":
            for code in receipt.reason_codes:
                if code.startswith(("GRANT_", "INVALID_GRANT", "NO_GRANT", "DUAL_")):
                    UNAUTHORIZED.labels(reason=code).inc()
        return receipt.to_dict()

    @app.post("/v1/decisions/{key}:complete")
    def complete(key: str, payload: dict[str, Any]) -> Any:
        if not settings.decisions_enabled:
            return _disabled()
        try:
            receipt = executor.complete(
                key,
                str(payload.get("outcome", "")),
                detail=str(payload.get("detail", ""))[:2000],
                external_ref=(
                    str(payload["external_ref"])[:256] if payload.get("external_ref") else None
                ),
            )
        except OutcomeConflict as exc:
            known = executor.receipt(key) is not None
            return _error(
                409 if known else 404, "outcome_conflict" if known else "unknown_key", str(exc)
            )
        return receipt.to_dict()

    @app.get("/v1/decisions/{key}")
    def receipt(key: str) -> Any:
        if not settings.decisions_enabled:
            return _disabled()
        found = executor.receipt(key)
        if found is None:
            return _error(404, "unknown_key", f"no decision recorded for {key!r}")
        return found.to_dict()


def seal_for_trace(
    app: FastAPI, trace: Any, payload: Mapping[str, Any], model_digest: str | None
) -> dict[str, Any]:
    """``{proposal_token, trace_id}`` for a scene run that named a case, else ``{}``."""
    secret: bytes = getattr(app.state, "proposal_secret", b"")
    case_id = payload.get("case_id")
    if not secret or not isinstance(case_id, str) or not case_id:
        return {}
    asset = payload.get("asset_id")
    proposal = ProposedDecision.from_trace(
        trace,
        case_id=case_id[:128],
        asset_id=str(asset)[:256] if asset else None,
        model_digest=model_digest,
    )
    if proposal is None:
        return {}
    return {"proposal_token": proposal.seal(secret), "trace_id": proposal.trace_id}
