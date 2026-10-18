"""Concurrency: the once-only guarantees hold when callers race.

``test_store.py`` already races ten threads and four processes on the *same*
idempotency key. These tests race the cases an attacker or a flaky SOAR
retry loop would actually produce:

* one grant, many *different* keys at once (a double spend under load);
* many distinct grants at once (no lost writes, no false refusals);
* several workers completing the same key with *different* outcomes;
* audit appends from many workers (the chain still verifies, nothing lost);
* the same grant spent from several processes with different keys.

Every in-process case runs against both stores; the SQLite cases give each
worker its own ``SQLiteStore`` and ``Executor``, so the only shared thing is
the database file, as with several server workers on one host.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import threading
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from jef_scene.authz import AuthorizationGrant, ExecutionReceipt, ProposedDecision
from jef_scene.executor import Executor, OutcomeConflict, TransportResult
from jef_scene.store import MemoryStore, ReceiptStore, SQLiteStore

SECRET = b"approval-service-secret"
WORKERS = 12


class Transport:
    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    def execute(
        self, action: str, asset_id: str | None, params: Mapping[str, Any], *, idempotency_key: str
    ) -> TransportResult:
        with self._lock:
            self.calls += 1
        return TransportResult(ok=True, detail="done", external_ref="edr-1")


def proposal(case: str = "case-1", asset: str = "host-01") -> ProposedDecision:
    return ProposedDecision(
        case_id=case,
        action="isolate_host",
        action_tier=3,
        policy_verdict="review_required",
        reason_codes=("DISRUPTIVE_ACTION_REQUIRES_APPROVAL",),
        scene="incident-triage",
        scene_version=1,
        model_digest="sha256:abc",
        trace_id=f"t-{case}",
        asset_id=asset,
    )


def grant(case: str = "case-1", asset: str = "host-01") -> dict[str, Any]:
    return AuthorizationGrant.issue(
        secret=SECRET,
        case_id=case,
        action="isolate_host",
        asset_id=asset,
        approvers=("alice",),
        approver_role="incident_commander",
        issuer="approval-svc",
        now=datetime.now(UTC),
    ).to_dict()


@pytest.fixture(params=["memory", "sqlite"])
def opener(request, tmp_path: Path) -> Callable[[], ReceiptStore]:
    """Opens the shared store. SQLite: a new connection per worker, one file."""
    if request.param == "memory":
        shared = MemoryStore()
        return lambda: shared
    path = tmp_path / "decisions.db"
    SQLiteStore(path)  # create the schema before the race
    return lambda: SQLiteStore(path)


def race(n: int, work: Callable[[int], Any]) -> list[Any]:
    """Run ``work(i)`` on ``n`` threads released together; return results in order.

    A barrier makes the threads start at the same moment, so the race is real
    rather than sequential-by-accident. Any exception fails the test.
    """
    barrier = threading.Barrier(n)
    results: list[Any] = [None] * n
    errors: list[BaseException] = []
    lock = threading.Lock()

    def run(i: int) -> None:
        try:
            barrier.wait(timeout=30)
            results[i] = work(i)
        except BaseException as exc:  # surfaced below
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert errors == [], f"a worker failed instead of waiting: {errors!r}"
    return results


def test_one_grant_many_keys_at_once_authorises_exactly_one(opener) -> None:
    """The double spend under load: one approval, twelve commands, one winner."""
    g = grant()

    def spend(i: int) -> ExecutionReceipt:
        ex = Executor("fake", Transport(), grant_secret=SECRET, store=opener())
        return ex.authorize(proposal(), idempotency_key=f"key-{i:02d}", grant=g)

    results = race(WORKERS, spend)
    statuses = Counter(r.status for r in results)
    assert statuses == {"authorized": 1, "not_dispatched": WORKERS - 1}, statuses
    refused = [r for r in results if r.status == "not_dispatched"]
    assert all(r.reason_codes == ("GRANT_ALREADY_USED",) for r in refused)
    store = opener()
    assert sum(r.status == "authorized" for r in store.snapshot().values()) == 1
    assert store.verify_audit()[0]


def test_many_distinct_grants_at_once_are_all_authorised(opener) -> None:
    """No lost writes and no false refusals when unrelated approvals race."""

    def spend(i: int) -> ExecutionReceipt:
        ex = Executor("fake", Transport(), grant_secret=SECRET, store=opener())
        case, asset = f"case-{i}", f"host-{i:02d}"
        return ex.authorize(
            proposal(case, asset), idempotency_key=f"{case}:isolate", grant=grant(case, asset)
        )

    results = race(WORKERS, spend)
    assert [r.status for r in results] == ["authorized"] * WORKERS
    store = opener()
    assert len(store.snapshot()) == WORKERS
    ok, count = store.verify_audit()
    assert ok and count >= WORKERS


def test_racing_completions_record_one_outcome(opener) -> None:
    """Workers report success, failed and timeout for one key: one is final, the rest conflict."""
    ex0 = Executor("fake", Transport(), grant_secret=SECRET, store=opener())
    first = ex0.authorize(proposal(), idempotency_key="k-complete", grant=grant())
    assert first.status == "authorized"
    outcomes = ["success", "failed", "timeout"] * (WORKERS // 3)

    def finish(i: int) -> str:
        ex = Executor("fake", Transport(), grant_secret=SECRET, store=opener())
        try:
            return ex.complete("k-complete", outcomes[i], detail=f"worker {i}").status
        except OutcomeConflict:
            return "conflict"

    results = race(len(outcomes), finish)
    final = opener().get("k-complete")
    assert final is not None and final.status in {"success", "failed", "unknown"}
    # Every worker that did not conflict reported the stored outcome, and
    # every conflict came from a worker that reported a different one.
    winners = {r for r in results if r != "conflict"}
    assert winners == {final.status}, (winners, final.status)
    wanted = {"success": "success", "failed": "failed", "timeout": "unknown"}
    for outcome, result in zip(outcomes, results, strict=True):
        assert (result == "conflict") == (wanted[outcome] != final.status)
    # Exactly one receipt event was appended for the finish.
    receipts = [e for e in opener().audit_entries() if e.event == "receipt"]
    assert len(receipts) == 1


def test_racing_retries_dispatch_once(opener) -> None:
    """One-phase dispatch: a retry storm on one key calls the platform once."""
    transport = Transport()
    g = grant()

    def retry(i: int) -> ExecutionReceipt:
        ex = Executor("fake", transport, grant_secret=SECRET, store=opener())
        return ex.dispatch(proposal(), idempotency_key="k-dispatch", grant=g)

    results = race(WORKERS, retry)
    assert transport.calls == 1, "the platform was called more than once"
    # The worker that claimed the key reports the outcome; a retry that raced
    # it sees the claim ("authorized", in progress) or the finished receipt,
    # never a second call.
    statuses = Counter(r.status for r in results)
    assert set(statuses) <= {"authorized", "success"}, statuses
    assert statuses["success"] >= 1
    final = opener().get("k-dispatch")
    assert final is not None and final.status == "success"
    assert ex_retry_after(opener, g).status == "success", "a later retry gets the outcome"


def ex_retry_after(opener: Callable[[], ReceiptStore], g: dict[str, Any]) -> ExecutionReceipt:
    transport = Transport()
    ex = Executor("fake", transport, grant_secret=SECRET, store=opener())
    receipt = ex.dispatch(proposal(), idempotency_key="k-dispatch", grant=g)
    assert transport.calls == 0
    return receipt


def test_a_crash_after_the_claim_never_dispatches_twice(tmp_path: Path) -> None:
    """The process dies between claiming the key and recording the outcome."""
    path = tmp_path / "d.db"
    g = grant()

    class Dies:
        def execute(self, *a: Any, **k: Any) -> TransportResult:
            raise KeyboardInterrupt("process killed mid-call")

    with pytest.raises(KeyboardInterrupt):
        Executor("fake", Dies(), grant_secret=SECRET, store=SQLiteStore(path)).dispatch(
            proposal(), idempotency_key="k-crash", grant=g
        )
    transport = Transport()
    after = Executor("fake", transport, grant_secret=SECRET, store=SQLiteStore(path))
    again = after.dispatch(proposal(), idempotency_key="k-crash", grant=g)
    assert transport.calls == 0, "a retry after the crash must reconcile, not act again"
    assert again.status == "authorized"
    # The connector reconciles with the platform and records what it found.
    assert after.complete("k-crash", "unknown").status == "unknown"


def test_audit_chain_survives_concurrent_appends(opener) -> None:
    """Twelve workers append at once; the chain verifies and every entry is there."""

    def spend(i: int) -> None:
        ex = Executor("fake", Transport(), grant_secret=SECRET, store=opener())
        for j in range(3):
            case, asset = f"case-{i}-{j}", f"host-{i}-{j}"
            ex.authorize(
                proposal(case, asset), idempotency_key=f"{case}:k", grant=grant(case, asset)
            )

    race(WORKERS, spend)
    store = opener()
    ok, count = store.verify_audit()
    assert ok, "a concurrent append broke the chain"
    assert len(store.snapshot()) == WORKERS * 3
    assert count == len(store.audit_entries())
    assert store.audit_head()["count"] == count


_SPENDER = r"""
import json, sys
from jef_scene.authz import ProposedDecision
from jef_scene.executor import Executor, TransportResult
from jef_scene.store import SQLiteStore

class T:
    def execute(self, *a, **k):
        return TransportResult(ok=True, detail="", external_ref=None)

path, grant, key = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
p = ProposedDecision(case_id="case-1", action="isolate_host", action_tier=3,
    policy_verdict="review_required", reason_codes=("DISRUPTIVE_ACTION_REQUIRES_APPROVAL",),
    scene="incident-triage", scene_version=1, model_digest="sha256:abc", trace_id="t-case-1",
    asset_id="host-01")
ex = Executor("fake", T(), grant_secret=b"approval-service-secret", store=SQLiteStore(path))
print(json.dumps(ex.authorize(p, idempotency_key=key, grant=grant).to_dict(), sort_keys=True))
"""


def test_one_grant_spent_from_four_processes_with_different_keys(tmp_path: Path) -> None:
    """Across processes there is no shared lock but the database's: still one winner."""
    path = str(tmp_path / "d.db")
    SQLiteStore(path)
    g = json.dumps(grant())
    procs = [
        subprocess.Popen(  # noqa: S603 - fixed interpreter and script
            [sys.executable, "-c", _SPENDER, path, g, f"proc-key-{i}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for i in range(4)
    ]
    outputs = [p.communicate(timeout=120) for p in procs]
    assert all(p.returncode == 0 for p in procs), [e.decode()[-400:] for _, e in outputs]
    statuses = Counter(json.loads(o)["status"] for o, _ in outputs)
    assert statuses == {"authorized": 1, "not_dispatched": 3}, statuses
    with sqlite3.connect(path) as con:
        assert con.execute("SELECT COUNT(*) FROM grant_nonces").fetchone()[0] == 1
        assert (
            con.execute("SELECT COUNT(*) FROM receipts WHERE status = 'authorized'").fetchone()[0]
            == 1
        )
