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
from .store import MemoryStore, ReceiptStore

__all__ = ["Executor", "OutcomeConflict", "Transport", "TransportResult"]


class OutcomeConflict(ValueError):
    """``complete`` was called for a key that cannot take that outcome."""


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
    #: Receipts, spent grant nonces and the audit chain. Memory by default (one
    #: process, one replica); a persistent store makes the once-only guarantees
    #: survive restarts and extra workers. See :mod:`jef_scene.store`.
    store: ReceiptStore = field(default_factory=MemoryStore)
    dispatch_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def receipts(self) -> dict[str, ExecutionReceipt]:
        """Every receipt this executor's store holds, by idempotency key."""
        return self.store.snapshot()

    @property
    def audit(self) -> list[AuditEntry]:
        return self.store.audit_entries()

    def _log(self, entry: AuditEntry) -> None:
        self.store.append_audit(entry)

    # -- public ---------------------------------------------------------------

    def dispatch(
        self,
        proposal: ProposedDecision,
        *,
        idempotency_key: str,
        grant: Any = None,
        dry_run: bool = False,
    ) -> ExecutionReceipt:
        """Check, claim the key, call the platform once, record the outcome.

        The key is claimed in the store (an ``authorized`` receipt) *before*
        the platform is called, and only the worker whose claim landed makes
        the call. Several workers sharing a persistent store therefore cannot
        all pass the "not seen yet" check and each call the platform: the
        losers get the claim back. If the process dies after the claim, the
        key stays ``authorized`` with no outcome, which is the truth: a retry
        returns it without calling again, and someone must reconcile with the
        platform rather than act twice.
        """
        with self._lock:
            seen = self.store.get(idempotency_key)
            if seen is not None:
                return seen
            if dry_run:
                return self._dispatch(proposal, idempotency_key, grant, dry_run=True)
            checked = self._check(proposal, idempotency_key, grant)
            if isinstance(checked, ExecutionReceipt):
                return self.store.put_if_absent(idempotency_key, checked)
            started = self.clock()
            claim = ExecutionReceipt(
                case_id=proposal.case_id,
                trace_id=proposal.trace_id,
                action=proposal.action,
                asset_id=proposal.asset_id,
                connector=self.connector,
                idempotency_key=idempotency_key,
                status="authorized",
                detail="dispatch in progress; no outcome recorded yet",
                grant_nonce=checked,
                started_at=_iso(started),
            )
            stored = self.store.put_if_absent(idempotency_key, claim)
            if stored is not claim:
                # Another worker claimed this key first; it makes the call.
                return stored
            final = self._perform(proposal, idempotency_key, checked, started)
            if self.store.replace(idempotency_key, "authorized", final):
                return final
            # Someone recorded an outcome for the claim meanwhile (``complete``);
            # the store's answer wins, never ours.
            return self.store.get(idempotency_key) or final

    # -- two-phase: for connectors whose platform performs the action ----------
    #
    # A SOAR platform usually performs the isolation itself (CrowdStrike app,
    # Defender app, ...), and JEF must not hold those credentials. So instead of
    # handing JEF a Transport, the connector asks `authorize` first, performs
    # the action only on `authorized`, and reports what the platform said with
    # `complete`. Same checks, same idempotency, same receipts.

    def authorize(
        self,
        proposal: ProposedDecision,
        *,
        idempotency_key: str,
        grant: Any = None,
        dry_run: bool = False,
    ) -> ExecutionReceipt:
        """Every dispatch check, no call. ``authorized`` means "go ahead, once"."""
        with self._lock:
            seen = self.store.get(idempotency_key)
            if seen is not None:
                return seen
            refused = self._check(proposal, idempotency_key, grant)
            if isinstance(refused, ExecutionReceipt):
                return self.store.put_if_absent(idempotency_key, refused)
            receipt = ExecutionReceipt(
                case_id=proposal.case_id,
                trace_id=proposal.trace_id,
                action=proposal.action,
                asset_id=proposal.asset_id,
                connector=self.connector,
                idempotency_key=idempotency_key,
                status="dry_run" if dry_run else "authorized",
                detail="all checks passed; nothing was sent"
                if dry_run
                else "the connector may perform this action once",
                grant_nonce=refused,
                started_at=_iso(self.clock()),
            )
            if not dry_run:
                return self.store.put_if_absent(idempotency_key, receipt)
            return receipt

    def complete(
        self,
        idempotency_key: str,
        outcome: str,
        *,
        detail: str = "",
        external_ref: str | None = None,
    ) -> ExecutionReceipt:
        """Record what the platform reported for an ``authorized`` key.

        ``success`` and ``failed`` need the platform's answer; ``timeout`` and
        ``unknown`` are both stored as ``unknown`` (the action may have landed).
        A repeat with the same outcome returns the stored receipt; a different
        outcome for a finished key, or any outcome for a key that was never
        authorized, raises :class:`OutcomeConflict`.
        """
        normalized = {
            "success": "success",
            "failed": "failed",
            "timeout": "unknown",
            "unknown": "unknown",
        }.get(outcome)
        if normalized is None:
            raise OutcomeConflict(
                f"outcome must be success, failed, timeout or unknown, not {outcome!r}"
            )
        with self._lock:
            current = self.store.get(idempotency_key)
            if current is None:
                raise OutcomeConflict(f"{idempotency_key!r} was never authorized")
            if current.status == "authorized":
                done = ExecutionReceipt(
                    **{
                        **current.to_dict(),
                        "status": normalized,
                        "detail": detail or current.detail,
                        "external_ref": external_ref,
                        "reason_codes": ("OUTCOME_UNOBSERVED",) if normalized == "unknown" else (),
                        "finished_at": _iso(self.clock()),
                    }
                )
                if not self.store.replace(idempotency_key, "authorized", done):
                    # Lost a race with another worker: report what it recorded.
                    current = self.store.get(idempotency_key)
                    if current is not None and current.status == normalized:
                        return current
                    raise OutcomeConflict(
                        f"{idempotency_key!r} is {current.status if current else 'gone'}; "
                        f"it cannot become {normalized}"
                    )
                self.dispatch_count += 1
                self._log(AuditEntry.of("receipt", done.case_id, done.trace_id, done.to_dict()))
                return done
            if current.status == normalized:
                return current
            raise OutcomeConflict(
                f"{idempotency_key!r} is {current.status}; it cannot become {normalized}"
            )

    def receipt(self, idempotency_key: str) -> ExecutionReceipt | None:
        return self.store.get(idempotency_key)

    # -- internals ------------------------------------------------------------

    def _check(
        self, proposal: ProposedDecision, key: str, grant: Any
    ) -> ExecutionReceipt | str | None:
        """The shared checks: a refusal receipt, or the grant nonce (None if no grant needed)."""
        self._log(
            AuditEntry.of("proposed", proposal.case_id, proposal.trace_id, proposal.to_dict())
        )
        decision = PolicyDecision.for_proposal(proposal)
        self._log(AuditEntry.of("decided", proposal.case_id, proposal.trace_id, decision.__dict__))
        if self.allowed_actions and proposal.action not in self.allowed_actions:
            return self._refuse(
                proposal,
                key,
                ["ACTION_NOT_ALLOWLISTED"],
                f"{self.connector} does not perform {proposal.action!r}",
            )
        if proposal.automatable:
            return None
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
            return self._refuse(proposal, key, [why], f"grant {parsed.nonce[:8]}… rejected: {why}")
        # One approval spends once. The same idempotency key may present it
        # again (a retry); any other key is a second command on one signature.
        if not self.store.consume_nonce(
            parsed.nonce,
            key,
            case_id=proposal.case_id,
            action=proposal.action,
            asset_id=proposal.asset_id,
        ):
            return self._refuse(
                proposal,
                key,
                ["GRANT_ALREADY_USED"],
                f"grant {parsed.nonce[:8]}… was already spent by another idempotency key",
            )
        self._log(AuditEntry.of("granted", proposal.case_id, proposal.trace_id, parsed.to_dict()))
        return parsed.nonce

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
        self._log(AuditEntry.of("refused", proposal.case_id, proposal.trace_id, receipt.to_dict()))
        return receipt

    def _dispatch(
        self, proposal: ProposedDecision, key: str, grant: Any, dry_run: bool
    ) -> ExecutionReceipt:
        checked = self._check(proposal, key, grant)
        if isinstance(checked, ExecutionReceipt):
            return checked
        grant_nonce = checked

        started = self.clock()
        if not dry_run:
            return self._perform(proposal, key, grant_nonce, started)
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

    def _perform(
        self, proposal: ProposedDecision, key: str, grant_nonce: str | None, started: datetime
    ) -> ExecutionReceipt:
        """Call the platform once and build the receipt. Callers own the claim."""
        self.dispatch_count += 1
        self._log(AuditEntry.of("dispatched", proposal.case_id, proposal.trace_id, {"key": key}))
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
        self._log(AuditEntry.of("receipt", proposal.case_id, proposal.trace_id, receipt.to_dict()))
        return receipt
