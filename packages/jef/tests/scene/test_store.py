"""NEXT-04: the once-only guarantees survive restarts, workers and crashes.

Every test runs against both stores. The SQLite cases additionally reopen the
file to stand in for a restart, run several executors against one file to
stand in for several workers, and tamper with a row to prove the audit chain
notices.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from jef_scene.authz import AuthorizationGrant, ExecutionReceipt, ProposedDecision
from jef_scene.executor import Executor, OutcomeConflict, TransportResult
from jef_scene.store import MemoryStore, SQLiteStore, StoreUnavailable, open_store

SECRET = b"approval-service-secret"


class Transport:
    def __init__(self, behaviour: str = "ok") -> None:
        self.behaviour = behaviour
        self.calls = 0

    def execute(
        self, action: str, asset_id: str | None, params: Mapping[str, Any], *, idempotency_key: str
    ) -> TransportResult:
        self.calls += 1
        if self.behaviour == "timeout":
            raise TimeoutError("no answer")
        return TransportResult(ok=True, detail="done", external_ref="edr-1")


def proposal(asset: str = "host-01", action: str = "isolate_host") -> ProposedDecision:
    return ProposedDecision(
        case_id="case-1",
        action=action,
        action_tier=3,
        policy_verdict="review_required",
        reason_codes=("DISRUPTIVE_ACTION_REQUIRES_APPROVAL",),
        scene="incident-triage",
        scene_version=1,
        model_digest="sha256:abc",
        trace_id="t-1",
        asset_id=asset,
    )


def grant(asset: str = "host-01", action: str = "isolate_host") -> dict[str, Any]:
    return AuthorizationGrant.issue(
        secret=SECRET,
        case_id="case-1",
        action=action,
        asset_id=asset,
        approvers=("alice",),
        approver_role="incident_commander",
        issuer="approval-svc",
        now=datetime.now(UTC),
    ).to_dict()


@pytest.fixture(params=["memory", "sqlite"])
def store_factory(request, tmp_path: Path):
    """Returns a callable that opens *the same* store again (a restart for SQLite)."""
    if request.param == "memory":
        single = MemoryStore()
        return lambda: single
    path = tmp_path / "decisions.db"
    return lambda: SQLiteStore(path)


def executor(store, behaviour: str = "ok") -> Executor:
    return Executor("fake", Transport(behaviour), grant_secret=SECRET, store=store)


# --------------------------------------------------------------------------- #
# the primitives
# --------------------------------------------------------------------------- #


def receipt(key: str, status: str = "authorized") -> ExecutionReceipt:
    return ExecutionReceipt(
        case_id="case-1",
        trace_id="t-1",
        action="isolate_host",
        asset_id="host-01",
        connector="fake",
        idempotency_key=key,
        status=status,  # type: ignore[arg-type]
    )


def test_put_if_absent_stores_once_and_returns_the_first(store_factory) -> None:
    s = store_factory()
    first = receipt("k", "authorized")
    assert s.put_if_absent("k", first) is first or s.put_if_absent("k", first) == first
    second = receipt("k", "not_dispatched")
    assert s.put_if_absent("k", second).status == "authorized"
    assert s.get("k").status == "authorized"


def test_replace_is_compare_and_set_and_never_overwrites_a_final_outcome(store_factory) -> None:
    s = store_factory()
    s.put_if_absent("k", receipt("k", "authorized"))
    assert s.replace("k", "not_dispatched", receipt("k", "success")) is False, "wrong expectation"
    assert s.replace("k", "authorized", receipt("k", "unknown")) is True
    assert s.replace("k", "unknown", receipt("k", "success")) is False, (
        "unknown never becomes success"
    )
    assert s.get("k").status == "unknown"


def test_a_nonce_belongs_to_the_first_key_that_spends_it(store_factory) -> None:
    s = store_factory()
    assert s.consume_nonce("n1", "key-a", case_id="c", action="a", asset_id="h") is True
    assert s.consume_nonce("n1", "key-a", case_id="c", action="a", asset_id="h") is True, "retry"
    assert s.consume_nonce("n1", "key-b", case_id="c", action="a", asset_id="h") is False


# --------------------------------------------------------------------------- #
# through the executor
# --------------------------------------------------------------------------- #


def test_retried_dispatch_calls_the_platform_once(store_factory) -> None:
    ex = executor(store_factory())
    a = ex.dispatch(proposal(), idempotency_key="k", grant=grant())
    b = ex.dispatch(proposal(), idempotency_key="k", grant=grant())
    assert a.status == "success" and b == a
    assert ex.transport.calls == 1  # type: ignore[attr-defined]


def test_one_grant_cannot_authorise_two_commands(store_factory) -> None:
    ex = executor(store_factory())
    g = grant()
    first = ex.dispatch(proposal(), idempotency_key="k1", grant=g)
    second = ex.dispatch(proposal(), idempotency_key="k2", grant=g)
    assert first.status == "success"
    assert second.status == "not_dispatched"
    assert second.reason_codes == ("GRANT_ALREADY_USED",)
    assert ex.dispatch_count == 1


def test_a_grant_for_one_asset_never_moves_to_another(store_factory) -> None:
    ex = executor(store_factory())
    r = ex.dispatch(proposal(asset="host-02"), idempotency_key="k", grant=grant(asset="host-01"))
    assert r.status == "not_dispatched" and r.reason_codes == ("GRANT_ASSET_MISMATCH",)
    assert ex.dispatch_count == 0


def test_timeout_is_unknown_and_stays_unknown(store_factory) -> None:
    ex = executor(store_factory(), behaviour="timeout")
    r = ex.dispatch(proposal(), idempotency_key="k", grant=grant())
    assert r.status == "unknown" and "OUTCOME_UNOBSERVED" in r.reason_codes
    with pytest.raises(OutcomeConflict):
        ex.complete("k", "success")


def test_audit_chain_is_recorded_and_verifies(store_factory) -> None:
    ex = executor(store_factory())
    ex.dispatch(proposal(), idempotency_key="k", grant=grant())
    ok, n = ex.store.verify_audit()
    assert ok and n == 5
    assert [e.event for e in ex.audit] == [
        "proposed",
        "decided",
        "granted",
        "dispatched",
        "receipt",
    ]


# --------------------------------------------------------------------------- #
# sqlite only: restart, workers, tampering, unavailability
# --------------------------------------------------------------------------- #


def test_receipts_and_spent_nonces_survive_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "d.db"
    g = grant()
    before = executor(SQLiteStore(path))
    r = before.authorize(proposal(), idempotency_key="k", grant=g)
    assert r.status == "authorized"

    after = executor(SQLiteStore(path))  # new process, same file
    assert after.receipt("k") == r
    done = after.complete("k", "success", external_ref="edr-9")
    assert done.status == "success" and done.external_ref == "edr-9"
    # The grant was spent before the restart; a new key after it is refused.
    replay = after.authorize(proposal(), idempotency_key="k-other", grant=g)
    assert replay.reason_codes == ("GRANT_ALREADY_USED",)
    assert after.store.verify_audit()[0]


def test_ten_workers_authorise_the_same_key_exactly_once(tmp_path: Path) -> None:
    path = tmp_path / "d.db"
    g = grant()
    results: list[ExecutionReceipt] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            ex = executor(SQLiteStore(path))  # each worker: its own connection and lock
            r = ex.authorize(proposal(), idempotency_key="k", grant=g)
        except BaseException as exc:  # surfaced by the assertion below
            with lock:
                errors.append(exc)
            return
        with lock:
            results.append(r)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [], f"a worker failed instead of waiting: {errors!r}"
    assert len(results) == 10
    assert all(r == results[0] for r in results), "everyone sees one decision"
    assert results[0].status == "authorized"
    with sqlite3.connect(path) as con:
        assert con.execute("SELECT COUNT(*) FROM receipts").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM grant_nonces").fetchone()[0] == 1


def test_a_tampered_audit_row_breaks_the_chain(tmp_path: Path) -> None:
    path = tmp_path / "d.db"
    ex = executor(SQLiteStore(path))
    ex.dispatch(proposal(), idempotency_key="k", grant=grant())
    assert ex.store.verify_audit() == (True, 5)
    with sqlite3.connect(path) as con:
        con.execute("UPDATE audit_events SET event = 'refused' WHERE event = 'granted'")
    assert ex.store.verify_audit()[0] is False


def test_unavailable_store_is_an_error_not_a_fallback(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    with pytest.raises(StoreUnavailable):
        SQLiteStore(blocker / "d.db")
    with pytest.raises(StoreUnavailable, match="PostgreSQL"):
        open_store("postgresql://user@host/db")
    assert open_store(None).persistent is False
    assert open_store(f"sqlite:///{tmp_path / 'x.db'}").persistent is True
