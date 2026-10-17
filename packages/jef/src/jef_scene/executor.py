"""Turn a proposal and a grant into exactly one call, and record what came back.

The connector-facing half of the authorization contract. A SOAR connector
(Shuffle, Cortex, n8n) that wants to *act* on a scene result wraps its
platform call in a :class:`Transport` and hands proposal, grant and an
idempotency key to :meth:`Executor.dispatch`. The executor:

1. returns the stored receipt when the idempotency key was seen before, so a
   retried workflow step executes once (POL-005);
2. lets a T0/T1 proposal the policy allowed run without a grant;
3. for anything else requires a grant that parses, verifies against the
   executor's secret, and is valid for this case, this action and this asset
   right now (POL-003, POL-004, POL-006); a T4 action needs two approvers;
4. calls the transport once, and writes a receipt whose ``status`` is
   ``success`` only when the target said so. A timeout or an unparseable
   answer is ``unknown`` (POL-007). ``dry_run`` performs every check and
   sends nothing.

The executor never issues grants and never sees the approval secret's
counterpart on the approval side; it only verifies. Without a secret
configured it refuses every grant, which is the fail-closed default.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from .authz import (
    AuditEntry,
    AuthorizationGrant,
    ExecutionReceipt,
    GrantError,
    PolicyDecision,
    ProposedDecision,
)

__all__ = ["Executor", "Transport", "TransportResult"]


@dataclass(frozen=True)
class TransportResult:
    """What a platform call reported. ``ok`` is the target's confirmation."""

    ok: bool
    detail: str = ""
    external_ref: str | None = None


class Transport(Protocol):
    """One platform operation. Raise ``TimeoutError`` when the outcome was not observed."""

    def execute(
        self, action: str, asset_id: str | None, params: Mapping[str, Any], *, idempotency_key: str
    ) -> TransportResult: ...


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="seconds")


@dataclass
class Executor:
    connector: str
    transport: Transport
    #: Verifies grants. ``None`` means every grant is refused.
    grant_secret: bytes | None = None
    #: Platform operations this executor may perform at all. Empty allows
    #: any action name the scene produced; set it in production.
    allowed_actions: frozenset[str] = frozenset()
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    receipts: dict[str, ExecutionReceipt] = field(default_factory=dict)
    audit: list[AuditEntry] = field(default_factory=list)
    dispatch_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- public ---------------------------------------------------------------

    def dispatch(
        self,
        proposal: ProposedDecision,
        *,
        idempotency_key: str,
        grant: Any = None,
        dry_run: bool = False,
    ) -> ExecutionReceipt:
        with self._lock:
            seen = self.receipts.get(idempotency_key)
            if seen is not None:
                return seen
            receipt = self._dispatch(proposal, idempotency_key, grant, dry_run)
            self.receipts[idempotency_key] = receipt
            return receipt

    # -- internals ------------------------------------------------------------

    def _refuse(
        self, proposal: ProposedDecision, key: str, reasons: list[str], detail: str
    ) -> ExecutionReceipt:
        receipt = ExecutionReceipt(
            case_id=proposal.case_id,
            trace_id=proposal.trace_id,
            action=proposal.action,
            asset_id=proposal.asset_id,
            connector=self.connector,
            idempotency_key=key,
            status="not_dispatched",
            reason_codes=tuple(reasons),
            detail=detail,
            started_at=_iso(self.clock()),
            finished_at=_iso(self.clock()),
        )
        self.audit.append(
            AuditEntry.of("refused", proposal.case_id, proposal.trace_id, receipt.to_dict())
        )
        return receipt

    def _dispatch(
        self, proposal: ProposedDecision, key: str, grant: Any, dry_run: bool
    ) -> ExecutionReceipt:
        self.audit.append(
            AuditEntry.of("proposed", proposal.case_id, proposal.trace_id, proposal.to_dict())
        )
        decision = PolicyDecision.for_proposal(proposal)
        self.audit.append(
            AuditEntry.of("decided", proposal.case_id, proposal.trace_id, decision.__dict__)
        )

        if self.allowed_actions and proposal.action not in self.allowed_actions:
            return self._refuse(
                proposal,
                key,
                ["ACTION_NOT_ALLOWLISTED"],
                f"{self.connector} does not perform {proposal.action!r}",
            )

        grant_nonce: str | None = None
        if not proposal.automatable:
            if grant is None:
                return self._refuse(
                    proposal,
                    key,
                    [*proposal.reason_codes, "NO_GRANT"],
                    f"{proposal.action} is T{proposal.action_tier} with verdict "
                    f"{proposal.policy_verdict}; a signed AuthorizationGrant is required",
                )
            if not self.grant_secret:
                return self._refuse(
                    proposal,
                    key,
                    ["NO_GRANT_SECRET"],
                    "this executor has no grant secret configured and refuses every grant",
                )
            try:
                parsed = AuthorizationGrant.parse(grant, secret=self.grant_secret)
            except GrantError as exc:
                return self._refuse(proposal, key, ["INVALID_GRANT"], str(exc))
            ok, why = parsed.is_valid_for(
                proposal.case_id,
                proposal.action,
                proposal.asset_id,
                now=self.clock(),
                tier=proposal.action_tier,
            )
            if not ok:
                return self._refuse(
                    proposal, key, [why], f"grant {parsed.nonce[:8]}… rejected: {why}"
                )
            grant_nonce = parsed.nonce
            self.audit.append(
                AuditEntry.of("granted", proposal.case_id, proposal.trace_id, parsed.to_dict())
            )

        started = self.clock()
        if dry_run:
            return ExecutionReceipt(
                case_id=proposal.case_id,
                trace_id=proposal.trace_id,
                action=proposal.action,
                asset_id=proposal.asset_id,
                connector=self.connector,
                idempotency_key=key,
                status="dry_run",
                detail="all checks passed; nothing was sent",
                grant_nonce=grant_nonce,
                started_at=_iso(started),
                finished_at=_iso(self.clock()),
            )

        self.dispatch_count += 1
        self.audit.append(
            AuditEntry.of("dispatched", proposal.case_id, proposal.trace_id, {"key": key})
        )
        status: str
        detail = ""
        external: str | None = None
        outcome: tuple[str, ...] = ()
        try:
            result = self.transport.execute(
                proposal.action, proposal.asset_id, proposal.params, idempotency_key=key
            )
        except TimeoutError as exc:
            # The call may well have landed. Saying "failed" here would invite a
            # retry that isolates the host twice; saying "success" would be a lie.
            status, detail, outcome = (
                "unknown",
                f"timeout: outcome not observed ({exc})",
                ("OUTCOME_UNOBSERVED",),
            )
        except Exception as exc:
            status, detail, outcome = "failed", f"{type(exc).__name__}: {exc}", ("TRANSPORT_ERROR",)
        else:
            if isinstance(result, TransportResult):
                status = "success" if result.ok else "failed"
                detail, external = result.detail, result.external_ref
            else:
                status, detail, outcome = (
                    "unknown",
                    "transport returned no TransportResult",
                    ("OUTCOME_UNOBSERVED",),
                )

        receipt = ExecutionReceipt(
            case_id=proposal.case_id,
            trace_id=proposal.trace_id,
            action=proposal.action,
            asset_id=proposal.asset_id,
            connector=self.connector,
            idempotency_key=key,
            status=status,  # type: ignore[arg-type]
            reason_codes=outcome,
            detail=detail,
            external_ref=external,
            grant_nonce=grant_nonce,
            started_at=_iso(started),
            finished_at=_iso(self.clock()),
        )
        self.audit.append(
            AuditEntry.of("receipt", proposal.case_id, proposal.trace_id, receipt.to_dict())
        )
        return receipt
