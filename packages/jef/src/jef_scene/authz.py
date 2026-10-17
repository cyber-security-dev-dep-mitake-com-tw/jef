"""The authorization contract: a proposal, a policy decision, a grant, a receipt.

``SceneEngine.run`` tells you what the model would do. Nothing in that output
is permission to do it, and this module is where the two are kept apart with
types rather than discipline:

* :class:`ProposedDecision` is what the scene selected, plus the policy's
  verdict and the identifiers an auditor needs (case, scene version, model
  digest, policy version, trace id). It carries no credential.
* :class:`AuthorizationGrant` is a person's (or a policy service's) permission
  for **one** case, **one** action, **one** asset, for a **short time**, signed
  with a secret the model never sees. ``approved: true`` in a payload is not a
  grant; a grant without a valid signature is not a grant; a grant for a
  different asset or an expired one is not a grant. ``is_valid_for`` is the
  only way to accept one.
* :class:`ExecutionReceipt` is what a connector observed. ``success`` needs a
  positive answer from the target system. A timeout is ``unknown``, never
  ``success``: the EDR may well have isolated the host after the socket closed.
* :class:`AuditEntry` ties the four together in order.

Everything here is pure data plus validation. :mod:`jef_scene.executor` is
the one place that turns a proposal and a grant into a call.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from .policy import MAX_AUTO_TIER
from .trace import SceneTrace

__all__ = [
    "POLICY_VERSION",
    "AuditEntry",
    "AuthorizationGrant",
    "ExecutionReceipt",
    "ExecutionStatus",
    "GrantError",
    "PolicyDecision",
    "ProposalError",
    "ProposedDecision",
]

#: Bumped whenever the rules in jef_scene.policy or this module change what a
#: verdict or a grant means. Recorded on every proposal, decision and receipt
#: so a replayed trace is judged by the rules that produced it.
POLICY_VERSION = "2026.10-p0"

ExecutionStatus = Literal[
    "not_dispatched",  # refused before any call: no grant, wrong grant, verdict
    "dry_run",  # validated end to end, nothing sent
    "authorized",  # two-phase: checks passed, the connector may now perform the action
    "success",  # the target confirmed the change
    "failed",  # the target refused or errored; state is as before
    "timeout",  # reserved: never emitted, see `unknown`
    "unknown",  # no confirmation either way; a human must look
]


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Proposal and decision
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProposedDecision:
    """What the scene selected, with the identifiers an auditor needs.

    No credential lives here. A connector that dispatches on a proposal alone
    is dispatching on a model output, which is the thing this contract exists
    to prevent.
    """

    case_id: str
    action: str
    action_tier: int
    policy_verdict: str
    reason_codes: tuple[str, ...]
    scene: str
    scene_version: int
    model_digest: str
    trace_id: str
    policy_version: str = POLICY_VERSION
    asset_id: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()

    @property
    def automatable(self) -> bool:
        """May run without a grant: the policy allowed it and it is T0/T1."""
        return self.policy_verdict == "allow" and self.action_tier <= MAX_AUTO_TIER

    @classmethod
    def from_trace(
        cls,
        trace: SceneTrace,
        *,
        case_id: str,
        asset_id: str | None = None,
        trace_id: str | None = None,
        model_digest: str | None = None,
    ) -> ProposedDecision | None:
        """Build the proposal a trace implies, or ``None`` when it decided nothing."""
        if trace.action is None:
            return None
        evidence = tuple(f"{layer.id}.{q.id}" for layer in trace.layers for q in layer.questions)
        return cls(
            case_id=case_id,
            action=trace.action,
            action_tier=trace.action_tier if trace.action_tier is not None else 4,
            policy_verdict=trace.policy_verdict,
            reason_codes=tuple(trace.reason_codes),
            scene=trace.scene,
            scene_version=trace.version,
            model_digest=model_digest or trace.model,
            trace_id=trace_id or _digest(trace.to_dict())[:16],
            asset_id=asset_id,
            params=dict(trace.action_params),
            evidence_refs=evidence,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["params"] = dict(self.params)
        return data

    # -- sealing --------------------------------------------------------------

    def seal(self, secret: bytes, *, now: datetime | None = None) -> str:
        """An opaque token a connector carries from the scene run to authorization.

        HMAC-SHA256 over the canonical proposal and its issue time, with a
        secret only the server holds. The connector cannot edit the action,
        the tier, the verdict, the case or the asset without the token failing
        to verify -- which is the point: authorization must judge what the
        scene actually proposed, not what the caller says it proposed.
        """
        if not secret:
            raise ProposalError("a proposal cannot be sealed without a secret")
        body = {"proposal": self.to_dict(), "issued_at": _iso(now or _now())}
        raw = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        mac = hmac.new(secret, raw, hashlib.sha256).hexdigest()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=") + "." + mac

    @classmethod
    def unseal(
        cls,
        token: str,
        secret: bytes,
        *,
        max_age_seconds: int = 3600,
        now: datetime | None = None,
    ) -> ProposedDecision:
        """Verify and decode a sealed proposal, or raise :class:`ProposalError`."""
        if not secret:
            raise ProposalError("no proposal secret configured")
        if not isinstance(token, str) or token.count(".") != 1:
            raise ProposalError("not a proposal token")
        encoded, mac = token.split(".")
        try:
            raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        except (ValueError, TypeError) as exc:
            raise ProposalError("not a proposal token") from exc
        if not hmac.compare_digest(hmac.new(secret, raw, hashlib.sha256).hexdigest(), mac):
            raise ProposalError("proposal token signature does not verify")
        body = json.loads(raw)
        issued = datetime.fromisoformat(body["issued_at"])
        if (now or _now()) - issued > timedelta(seconds=max_age_seconds):
            raise ProposalError("proposal token has expired; run the scene again")
        data = body["proposal"]
        return cls(
            **{
                **data,
                "reason_codes": tuple(data["reason_codes"]),
                "evidence_refs": tuple(data.get("evidence_refs", ())),
                "params": dict(data.get("params", {})),
            }
        )


class ProposalError(ValueError):
    """A token that claimed to be a sealed proposal and was not one."""


@dataclass(frozen=True)
class PolicyDecision:
    """The policy's answer for one proposal, as a record of its own."""

    case_id: str
    trace_id: str
    verdict: str
    reason_codes: tuple[str, ...]
    tier: int
    policy_version: str = POLICY_VERSION
    decided_at: str = field(default_factory=lambda: _iso(_now()))

    @classmethod
    def for_proposal(cls, proposal: ProposedDecision) -> PolicyDecision:
        return cls(
            case_id=proposal.case_id,
            trace_id=proposal.trace_id,
            verdict=proposal.policy_verdict,
            reason_codes=proposal.reason_codes,
            tier=proposal.action_tier,
            policy_version=proposal.policy_version,
        )


# --------------------------------------------------------------------------- #
# Grant
# --------------------------------------------------------------------------- #


class GrantError(ValueError):
    """A payload that claimed to be a grant and was not one."""


_SIGNED_FIELDS = (
    "case_id",
    "action",
    "asset_id",
    "approvers",
    "approver_role",
    "issuer",
    "nonce",
    "issued_at",
    "expires_at",
)


def _digest(data: Any) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _sign(secret: bytes, fields: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {k: fields[k] for k in _SIGNED_FIELDS},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hmac.new(secret, canonical.encode(), hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class AuthorizationGrant:
    """Permission for one case, one action, one asset, until ``expires_at``.

    Issued by :meth:`issue` with a secret held by the approval service, never
    by the model or the scene. ``approvers`` is the identity of whoever
    approved; T4 actions need two distinct ones. The signature covers every
    scoping field, so a grant for host A cannot be edited into one for host B.
    """

    case_id: str
    action: str
    asset_id: str
    approvers: tuple[str, ...]
    approver_role: str
    issuer: str
    nonce: str
    issued_at: str
    expires_at: str
    signature: str

    # -- issuing ------------------------------------------------------------

    @classmethod
    def issue(
        cls,
        *,
        secret: bytes,
        case_id: str,
        action: str,
        asset_id: str,
        approvers: tuple[str, ...] | list[str],
        approver_role: str,
        issuer: str,
        ttl_seconds: int = 900,
        now: datetime | None = None,
    ) -> AuthorizationGrant:
        if not secret:
            raise GrantError("a grant cannot be issued without a signing secret")
        if not approvers or any(not a for a in approvers):
            raise GrantError("a grant needs at least one named approver")
        if ttl_seconds <= 0 or ttl_seconds > 86_400:
            raise GrantError("a grant lives between one second and one day")
        issued = now or _now()
        fields: dict[str, Any] = {
            "case_id": case_id,
            "action": action,
            "asset_id": asset_id,
            "approvers": list(approvers),
            "approver_role": approver_role,
            "issuer": issuer,
            "nonce": secrets.token_hex(16),
            "issued_at": _iso(issued),
            "expires_at": _iso(issued + timedelta(seconds=ttl_seconds)),
        }
        return cls(**{**fields, "approvers": tuple(approvers)}, signature=_sign(secret, fields))

    # -- accepting ----------------------------------------------------------

    @classmethod
    def parse(cls, payload: Any, *, secret: bytes) -> AuthorizationGrant:
        """Accept a grant from an untrusted payload, or raise :class:`GrantError`.

        This is the only entry point for anything that arrived over the wire.
        A dict with ``approved: true`` and no signature is exactly the forgery
        this refuses (POL-006).
        """
        if isinstance(payload, AuthorizationGrant):
            grant = payload
        elif isinstance(payload, Mapping):
            missing = [k for k in (*_SIGNED_FIELDS, "signature") if k not in payload]
            if missing:
                raise GrantError(f"not a grant: missing {', '.join(missing)}")
            try:
                grant = cls(
                    **{k: payload[k] for k in _SIGNED_FIELDS if k != "approvers"},
                    approvers=tuple(payload["approvers"]),
                    signature=str(payload["signature"]),
                )
            except (TypeError, ValueError) as exc:
                raise GrantError(f"not a grant: {exc}") from exc
        else:
            raise GrantError(f"not a grant: {type(payload).__name__}")
        if not grant.verify(secret):
            raise GrantError("grant signature does not verify")
        return grant

    def verify(self, secret: bytes) -> bool:
        if not secret:
            return False
        fields = {**asdict(self), "approvers": list(self.approvers)}
        return hmac.compare_digest(_sign(secret, fields), self.signature)

    def is_valid_for(
        self,
        case_id: str,
        action: str,
        asset_id: str | None,
        *,
        now: datetime | None = None,
        tier: int = 3,
    ) -> tuple[bool, str]:
        """``(True, "ok")`` or ``(False, <reason code>)``. Signature is checked by ``parse``."""
        moment = now or _now()
        if self.case_id != case_id:
            return False, "GRANT_CASE_MISMATCH"
        if self.action != action:
            return False, "GRANT_ACTION_MISMATCH"
        if asset_id is None or self.asset_id != asset_id:
            return False, "GRANT_ASSET_MISMATCH"
        try:
            expires = datetime.fromisoformat(self.expires_at)
            issued = datetime.fromisoformat(self.issued_at)
        except ValueError:
            return False, "GRANT_MALFORMED"
        if moment >= expires:
            return False, "GRANT_EXPIRED"
        if moment < issued - timedelta(seconds=60):
            return False, "GRANT_NOT_YET_VALID"
        if tier >= 4 and len(set(self.approvers)) < 2:
            return False, "DUAL_APPROVAL_REQUIRED"
        return True, "ok"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["approvers"] = list(self.approvers)
        return data


# --------------------------------------------------------------------------- #
# Receipt and audit
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExecutionReceipt:
    """What the connector observed. ``success`` only on the target's say-so."""

    case_id: str
    trace_id: str
    action: str
    asset_id: str | None
    connector: str
    idempotency_key: str
    status: ExecutionStatus
    reason_codes: tuple[str, ...] = ()
    detail: str = ""
    external_ref: str | None = None
    grant_nonce: str | None = None
    policy_version: str = POLICY_VERSION
    started_at: str = field(default_factory=lambda: _iso(_now()))
    finished_at: str | None = None

    @property
    def executed(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuditEntry:
    event: Literal["proposed", "decided", "granted", "refused", "dispatched", "receipt"]
    case_id: str
    trace_id: str
    policy_version: str
    payload_digest: str
    at: str = field(default_factory=lambda: _iso(_now()))

    @classmethod
    def of(cls, event: Any, case_id: str, trace_id: str, payload: Any) -> AuditEntry:
        return cls(
            event=event,
            case_id=case_id,
            trace_id=trace_id,
            policy_version=POLICY_VERSION,
            payload_digest="sha256:" + _digest(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
