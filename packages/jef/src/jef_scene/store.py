"""Where receipts, consumed grant nonces and the audit trail live.

The executor's guarantees -- a retried step authorises once, a grant is spent
once, every decision leaves an audit entry -- are only as durable as the store
behind them. :class:`MemoryStore` is the original behaviour: one process, one
replica, gone on restart. :class:`SQLiteStore` keeps the same contract on disk
with the database's own atomicity, so a restart, a second worker on the same
host, or a crash between ``authorize`` and ``complete`` cannot turn one
authorisation into two actions.

The contract is deliberately small:

``put_if_absent``
    The idempotency primitive. Exactly one caller stores a receipt for a key;
    everyone else gets that receipt back. Never overwrites.
``replace``
    Compare-and-set on status, for ``complete``: an ``authorized`` receipt
    becomes a finished one once; a finished one is never overwritten and
    ``unknown`` is never upgraded to ``success`` by a later guess.
``consume_nonce``
    A grant nonce is bound to the first idempotency key that spends it. The
    same key may see it again (that is a retry); any other key is refused, so
    one approval cannot authorise two commands, however they are addressed.
``append_audit`` / ``verify_audit`` / ``audit_head``
    Append-only, each entry carrying the digest of the previous one, so an
    edited or deleted row breaks the chain. ``audit_head`` is the value to
    export to independent storage, because a database administrator could
    otherwise recompute the whole chain after an edit.

Crash windows. ``consume_nonce`` and the receipt write are separate
operations, so a process can die after the nonce is bound and before the
receipt exists. That window is safe by construction: the nonce is bound to an
idempotency key, not merely marked spent, so a retry with the same key binds
again and writes the receipt, while any other key is refused with
``GRANT_ALREADY_USED``. One approval still authorises at most one command.
``tests/scene/test_store.py`` simulates the crash.

PostgreSQL is the deployment the execution plan recommends. It needs a driver
this package does not ship, so :func:`open_store` names that honestly instead
of silently falling back to memory.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .authz import AuditEntry, ExecutionReceipt

__all__ = [
    "MemoryStore",
    "ReceiptStore",
    "SQLiteStore",
    "StoreUnavailable",
    "open_store",
]


class StoreUnavailable(RuntimeError):
    """The store cannot be reached or written. Callers must fail closed."""


class ReceiptStore(Protocol):
    persistent: bool

    def get(self, key: str) -> ExecutionReceipt | None: ...
    def put_if_absent(self, key: str, receipt: ExecutionReceipt) -> ExecutionReceipt: ...
    def replace(self, key: str, expected_status: str, receipt: ExecutionReceipt) -> bool: ...
    def consume_nonce(
        self, nonce: str, key: str, *, case_id: str, action: str, asset_id: str | None
    ) -> bool: ...
    def append_audit(self, entry: AuditEntry) -> None: ...
    def audit_entries(self) -> list[AuditEntry]: ...
    def verify_audit(self) -> tuple[bool, int]: ...
    def audit_head(self) -> dict[str, Any]: ...
    def snapshot(self) -> dict[str, ExecutionReceipt]: ...


# --------------------------------------------------------------------------- #
# serialisation
# --------------------------------------------------------------------------- #


def _dump(receipt: ExecutionReceipt) -> str:
    return json.dumps(receipt.to_dict(), sort_keys=True, ensure_ascii=False)


def _load(payload: str) -> ExecutionReceipt:
    d = json.loads(payload)
    d["reason_codes"] = tuple(d.get("reason_codes") or ())
    return ExecutionReceipt(**d)


def _chain_digest(prev: str, entry: dict[str, Any]) -> str:
    raw = prev + "\n" + json.dumps(entry, sort_keys=True, ensure_ascii=False)
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


#: Outcomes that are never rewritten. ``unknown`` is final on purpose: a later
#: guess must not turn "we did not see what happened" into ``success``.
_FINAL = frozenset({"success", "failed", "unknown", "not_dispatched"})


# --------------------------------------------------------------------------- #
# memory
# --------------------------------------------------------------------------- #


@dataclass
class MemoryStore:
    """One process, one replica. The behaviour the executor always had."""

    persistent: bool = False
    _receipts: dict[str, ExecutionReceipt] = field(default_factory=dict)
    _nonces: dict[str, str] = field(default_factory=dict)
    _audit: list[tuple[AuditEntry, str, str]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get(self, key: str) -> ExecutionReceipt | None:
        return self._receipts.get(key)

    def put_if_absent(self, key: str, receipt: ExecutionReceipt) -> ExecutionReceipt:
        with self._lock:
            return self._receipts.setdefault(key, receipt)

    def replace(self, key: str, expected_status: str, receipt: ExecutionReceipt) -> bool:
        with self._lock:
            current = self._receipts.get(key)
            if current is None or current.status != expected_status or current.status in _FINAL:
                return False
            self._receipts[key] = receipt
            return True

    def consume_nonce(
        self, nonce: str, key: str, *, case_id: str, action: str, asset_id: str | None
    ) -> bool:
        with self._lock:
            owner = self._nonces.setdefault(nonce, key)
            return owner == key

    def append_audit(self, entry: AuditEntry) -> None:
        with self._lock:
            prev = self._audit[-1][2] if self._audit else "sha256:genesis"
            self._audit.append((entry, prev, _chain_digest(prev, entry.to_dict())))

    def audit_entries(self) -> list[AuditEntry]:
        return [e for e, _, _ in self._audit]

    def verify_audit(self) -> tuple[bool, int]:
        prev = "sha256:genesis"
        for entry, stored_prev, digest in self._audit:
            if stored_prev != prev or _chain_digest(prev, entry.to_dict()) != digest:
                return False, len(self._audit)
            prev = digest
        return True, len(self._audit)

    def audit_head(self) -> dict[str, Any]:
        return {
            "count": len(self._audit),
            "head": self._audit[-1][2] if self._audit else "sha256:genesis",
        }

    def snapshot(self) -> dict[str, ExecutionReceipt]:
        return dict(self._receipts)


# --------------------------------------------------------------------------- #
# sqlite
# --------------------------------------------------------------------------- #

_SCHEMA = """
CREATE TABLE IF NOT EXISTS receipts (
    idempotency_key TEXT PRIMARY KEY,
    status          TEXT NOT NULL,
    case_id         TEXT NOT NULL,
    trace_id        TEXT NOT NULL,
    action          TEXT NOT NULL,
    asset_id        TEXT,
    grant_nonce     TEXT,
    payload         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS grant_nonces (
    nonce           TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL,
    case_id         TEXT NOT NULL,
    action          TEXT NOT NULL,
    asset_id        TEXT,
    consumed_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_events (
    seq             INTEGER PRIMARY KEY AUTOINCREMENT,
    at              TEXT NOT NULL,
    event           TEXT NOT NULL,
    case_id         TEXT NOT NULL,
    trace_id        TEXT NOT NULL,
    policy_version  TEXT NOT NULL,
    payload_digest  TEXT NOT NULL,
    prev_digest     TEXT NOT NULL,
    event_digest    TEXT NOT NULL
);
"""


class SQLiteStore:
    """Receipts, nonces and audit on disk, with the database doing the atomicity.

    One connection per operation and ``BEGIN IMMEDIATE`` for the read-modify-
    write paths: simple, correct across threads and across processes on the
    same host, and fast enough for a store that sees one row per authorised
    action. WAL mode so readers never block the writer.
    """

    persistent = True
    #: Schema creation from several threads at once (several workers starting
    #: together) is serialised in-process; across processes SQLite's own lock
    #: and the busy timeout cover it.
    _init_lock = threading.Lock()

    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 10_000) -> None:
        self.path = Path(path)
        self.busy_timeout_ms = busy_timeout_ms
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._init_lock, self._connect() as con:
                # WAL needs a moment with no other connection open; if another
                # process holds one, the default journal is still correct, only
                # slower, so a failure here is not an error.
                with suppress(sqlite3.OperationalError):
                    con.execute("PRAGMA journal_mode=WAL")
                con.executescript(_SCHEMA)
        except (sqlite3.Error, OSError) as exc:
            raise StoreUnavailable(f"cannot open decision store at {self.path}: {exc}") from exc

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        try:
            con = sqlite3.connect(
                self.path, timeout=self.busy_timeout_ms / 1000, isolation_level=None
            )
        except sqlite3.Error as exc:
            raise StoreUnavailable(f"decision store at {self.path} is unavailable: {exc}") from exc
        try:
            con.execute(f"PRAGMA busy_timeout={int(self.busy_timeout_ms)}")
            yield con
        except sqlite3.Error as exc:
            with suppress(sqlite3.Error):
                con.execute("ROLLBACK")
            raise StoreUnavailable(f"decision store at {self.path} failed: {exc}") from exc
        finally:
            con.close()

    # -- receipts -------------------------------------------------------------

    def get(self, key: str) -> ExecutionReceipt | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT payload FROM receipts WHERE idempotency_key = ?", (key,)
            ).fetchone()
        return _load(row[0]) if row else None

    def put_if_absent(self, key: str, receipt: ExecutionReceipt) -> ExecutionReceipt:
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT payload FROM receipts WHERE idempotency_key = ?", (key,)
            ).fetchone()
            if row:
                con.execute("COMMIT")
                return _load(row[0])
            now = _now()
            con.execute(
                "INSERT INTO receipts VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    key,
                    receipt.status,
                    receipt.case_id,
                    receipt.trace_id,
                    receipt.action,
                    receipt.asset_id,
                    receipt.grant_nonce,
                    _dump(receipt),
                    now,
                    now,
                ),
            )
            con.execute("COMMIT")
            return receipt

    def replace(self, key: str, expected_status: str, receipt: ExecutionReceipt) -> bool:
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT status FROM receipts WHERE idempotency_key = ?", (key,)
            ).fetchone()
            if not row or row[0] != expected_status or row[0] in _FINAL:
                con.execute("COMMIT")
                return False
            con.execute(
                "UPDATE receipts SET status = ?, payload = ?, updated_at = ? WHERE idempotency_key = ? AND status = ?",
                (receipt.status, _dump(receipt), _now(), key, expected_status),
            )
            con.execute("COMMIT")
            return True

    # -- nonces ---------------------------------------------------------------

    def consume_nonce(
        self, nonce: str, key: str, *, case_id: str, action: str, asset_id: str | None
    ) -> bool:
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT idempotency_key FROM grant_nonces WHERE nonce = ?", (nonce,)
            ).fetchone()
            if row:
                con.execute("COMMIT")
                return row[0] == key
            con.execute(
                "INSERT INTO grant_nonces VALUES (?,?,?,?,?,?)",
                (nonce, key, case_id, action, asset_id, _now()),
            )
            con.execute("COMMIT")
            return True

    # -- audit ----------------------------------------------------------------

    def append_audit(self, entry: AuditEntry) -> None:
        d = entry.to_dict()
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT event_digest FROM audit_events ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            prev = row[0] if row else "sha256:genesis"
            con.execute(
                "INSERT INTO audit_events (at, event, case_id, trace_id, policy_version, payload_digest, prev_digest, event_digest) VALUES (?,?,?,?,?,?,?,?)",
                (
                    d["at"],
                    d["event"],
                    d["case_id"],
                    d["trace_id"],
                    d["policy_version"],
                    d["payload_digest"],
                    prev,
                    _chain_digest(prev, d),
                ),
            )
            con.execute("COMMIT")

    def _audit_rows(self) -> list[tuple[AuditEntry, str, str]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT at, event, case_id, trace_id, policy_version, payload_digest, prev_digest, event_digest FROM audit_events ORDER BY seq"
            ).fetchall()
        out = []
        for at, event, case_id, trace_id, pv, pd, prev, digest in rows:
            entry = AuditEntry(
                event=event,
                case_id=case_id,
                trace_id=trace_id,
                policy_version=pv,
                payload_digest=pd,
                at=at,
            )
            out.append((entry, prev, digest))
        return out

    def audit_entries(self) -> list[AuditEntry]:
        return [e for e, _, _ in self._audit_rows()]

    def verify_audit(self) -> tuple[bool, int]:
        rows = self._audit_rows()
        prev = "sha256:genesis"
        for entry, stored_prev, digest in rows:
            if stored_prev != prev or _chain_digest(prev, entry.to_dict()) != digest:
                return False, len(rows)
            prev = digest
        return True, len(rows)

    def audit_head(self) -> dict[str, Any]:
        """``{count, head}``: export this periodically to storage the DB admin cannot write.

        The chain detects an edited or deleted row, but someone with write
        access to the database could recompute every digest after an edit. A
        head recorded elsewhere before the edit no longer matches afterwards.
        """
        with self._connect() as con:
            row = con.execute(
                "SELECT COUNT(*), (SELECT event_digest FROM audit_events ORDER BY seq DESC LIMIT 1) "
                "FROM audit_events"
            ).fetchone()
        return {"count": int(row[0]), "head": row[1] or "sha256:genesis"}

    def snapshot(self) -> dict[str, ExecutionReceipt]:
        with self._connect() as con:
            rows = con.execute("SELECT idempotency_key, payload FROM receipts").fetchall()
        return {k: _load(p) for k, p in rows}


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #


def open_store(dsn: str | None) -> ReceiptStore:
    """``None``/``memory://`` -> memory; ``sqlite:///path`` or a file path -> SQLite.

    A ``postgresql://`` DSN is recognised and refused: this package ships no
    driver, and a silent fallback to memory would be the one outcome worse than
    an error.
    """
    if not dsn or dsn == "memory://":
        return MemoryStore()
    if dsn.startswith(("postgresql://", "postgres://")):
        raise StoreUnavailable(
            "JEF_DECISION_STORE names PostgreSQL, which this build cannot open (no driver "
            "shipped). Use sqlite:///absolute/path.db for a single host, or leave it unset "
            "and run one replica with JEF_WORKERS=1 while decisions stay memory-only."
        )
    if dsn.startswith("sqlite:///"):
        return SQLiteStore(dsn[len("sqlite:///") :] or ":memory:")
    if dsn.startswith("sqlite://"):
        return SQLiteStore(dsn[len("sqlite://") :])
    return SQLiteStore(dsn)
