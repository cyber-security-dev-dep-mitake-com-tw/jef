"""Ingress guards: who may call, how much they may send, how often.

Every check here runs before the request body is parsed and before any
tokenization or encoding. Rejected requests cost a header read, which is
the whole point: a budget enforced after the model has run is a bill, not a
budget.

What is enforced, in order, for every request except ``/healthz``:

1. **Authentication**, when ``JEF_API_KEYS`` is set. ``Authorization: Bearer
   <key>`` or ``X-API-Key: <key>``, compared in constant time. Missing is
   401, wrong is 403. Keys may be labelled (``soc-prod:<secret>``) so logs
   and rate limits can name a caller without logging the secret.
2. **Size.** ``Content-Length`` over ``JEF_MAX_BODY_BYTES`` is 413. A body
   with no ``Content-Length`` is 411: this API never needs chunked uploads,
   and a limit that can be dodged by omitting a header is not a limit.
   ``Content-Encoding`` is 415: nothing here decompresses, and refusing the
   header keeps a decompression bomb from ever becoming possible.
3. **Rate**, when ``JEF_RATE_LIMIT_PER_MINUTE`` is set: a fixed window per
   caller (the key's label, else the client address), 429 with Retry-After.

Two things this module does not do. It does not replace an ingress: a
deployment behind mTLS or an OIDC gateway may leave ``JEF_API_KEYS`` unset,
and ``/healthz`` says so (``auth: none``) so nobody mistakes the default for
protection. And it does not read the body: the per-request budgets that need
the parsed payload (state length, JSON depth, question count) live in the
handlers, still ahead of inference.
"""

from __future__ import annotations

import hashlib
import hmac
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter

from .settings import Settings

__all__ = [
    "REJECTED",
    "RateLimiter",
    "authenticate",
    "install_guards",
    "json_depth",
    "parse_api_keys",
]

REJECTED = Counter(
    "jef_rejected_requests_total",
    "Requests refused before inference, by reason. No caller identity in the "
    "labels on purpose: a label per client is a cardinality bomb.",
    ["reason"],
)

#: Always reachable: an orchestrator's liveness probe carries no credentials.
OPEN_PATHS: frozenset[str] = frozenset({"/healthz"})


def _reject(status: int, code: str, message: str, **headers: str) -> JSONResponse:
    REJECTED.labels(reason=code).inc()
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message}},
        headers=headers or None,
    )


# --------------------------------------------------------------------------- #
# API keys
# --------------------------------------------------------------------------- #


def parse_api_keys(raw: str | None) -> dict[str, str]:
    """``"soc-prod:abc,def"`` -> ``{"soc-prod": "abc", "key-<digest>": "def"}``.

    An unlabelled key is named by the first eight hex digits of its SHA-256,
    which is enough to tell callers apart in a log and reveals nothing.
    """
    keys: dict[str, str] = {}
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        label, sep, secret = item.partition(":")
        if not sep:
            label, secret = "", item
        secret = secret.strip()
        if not secret:
            continue
        label = label.strip() or f"key-{hashlib.sha256(secret.encode()).hexdigest()[:8]}"
        keys[label] = secret
    return keys


def _presented(headers: Mapping[str, str]) -> str | None:
    auth = headers.get("authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    api_key = headers.get("x-api-key", "")
    return api_key.strip() or None


def authenticate(headers: Mapping[str, str], keys: Mapping[str, str]) -> tuple[str, str | None]:
    """``("ok", label)``, ``("missing", None)`` or ``("invalid", None)``.

    Every configured key is compared, in constant time, whether or not an
    earlier one matched, so the response time does not say how many keys
    exist or which prefix was close.
    """
    token = _presented(headers)
    if token is None:
        return "missing", None
    matched: str | None = None
    for label, secret in keys.items():
        if hmac.compare_digest(token.encode(), secret.encode()):
            matched = label
    return ("ok", matched) if matched is not None else ("invalid", None)


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #


@dataclass
class _Window:
    start: float
    count: int


class RateLimiter:
    """Fixed one-minute window per caller. In-process; one per worker.

    Deliberately simple: the goal is to stop one misbehaving client from
    saturating a CPU-bound service, not to meter a billing plan. The table is
    bounded so a flood of distinct client addresses cannot grow it without
    limit; when it overflows, stale windows are dropped first.
    """

    def __init__(self, per_minute: int, *, max_callers: int = 10_000) -> None:
        self.per_minute = per_minute
        self.max_callers = max_callers
        self._windows: dict[str, _Window] = {}
        self._lock = threading.Lock()

    def check(self, caller: str, now: float | None = None) -> int | None:
        """``None`` when allowed; otherwise seconds until the window resets."""
        if self.per_minute <= 0:
            return None
        now = time.monotonic() if now is None else now
        with self._lock:
            window = self._windows.get(caller)
            if window is None or now - window.start >= 60.0:
                if len(self._windows) >= self.max_callers:
                    self._evict(now)
                window = _Window(start=now, count=0)
                self._windows[caller] = window
            window.count += 1
            if window.count > self.per_minute:
                return max(1, int(60.0 - (now - window.start)) + 1)
            return None

    def _evict(self, now: float) -> None:
        stale = [k for k, w in self._windows.items() if now - w.start >= 60.0]
        for k in stale:
            del self._windows[k]
        if len(self._windows) >= self.max_callers:
            # Still full of live windows: drop the oldest half rather than the
            # whole table, so a burst resets nobody's limit to zero.
            for k in sorted(self._windows, key=lambda k: self._windows[k].start)[
                : self.max_callers // 2
            ]:
                del self._windows[k]


# --------------------------------------------------------------------------- #
# Payload shape
# --------------------------------------------------------------------------- #


def json_depth(value: Any, _depth: int = 1) -> int:
    """Nesting depth of a parsed JSON value. A scalar is 1."""
    if isinstance(value, dict):
        return max((json_depth(v, _depth + 1) for v in value.values()), default=_depth)
    if isinstance(value, list):
        return max((json_depth(v, _depth + 1) for v in value), default=_depth)
    return _depth


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def install_guards(app: FastAPI, settings: Settings) -> None:
    """Attach the ingress middleware to ``app``."""
    keys = dict(settings.api_keys)
    if settings.require_auth and not keys:
        raise RuntimeError(
            "JEF_REQUIRE_AUTH is set but JEF_API_KEYS is empty: refusing to start an "
            "unauthenticated server that was asked to authenticate"
        )
    limiter = RateLimiter(settings.rate_limit_per_minute)
    open_paths = set(OPEN_PATHS)
    if settings.metrics_public:
        open_paths.add("/metrics")
    app.state.rate_limiter = limiter
    app.state.api_key_labels = tuple(sorted(keys))

    @app.middleware("http")
    async def _guards(request: Request, call_next: Any) -> Any:
        path = request.url.path
        if path in open_paths:
            return await call_next(request)

        caller = request.client.host if request.client else "unknown"
        if keys:
            outcome, label = authenticate(request.headers, keys)
            if outcome == "missing":
                return _reject(
                    401,
                    "unauthenticated",
                    "this endpoint requires an API key: Authorization: Bearer <key> or X-API-Key",
                    **{"WWW-Authenticate": "Bearer"},
                )
            if outcome == "invalid":
                return _reject(403, "forbidden", "the presented API key is not valid")
            caller = label or caller
            request.state.auth_subject = label
        else:
            request.state.auth_subject = None

        if request.method in {"POST", "PUT", "PATCH"}:
            if request.headers.get("content-encoding"):
                return _reject(
                    415,
                    "unsupported_encoding",
                    "compressed request bodies are not accepted; send plain JSON",
                )
            length = request.headers.get("content-length")
            if length is None:
                return _reject(
                    411,
                    "length_required",
                    "requests must declare Content-Length; chunked bodies are not accepted",
                )
            try:
                declared = int(length)
            except ValueError:
                return _reject(400, "bad_request", "Content-Length is not an integer")
            if declared > settings.max_body_bytes:
                return _reject(
                    413,
                    "body_too_large",
                    f"request body of {declared} bytes exceeds the limit of "
                    f"{settings.max_body_bytes}",
                )

        retry = limiter.check(caller)
        if retry is not None:
            return _reject(
                429,
                "rate_limited",
                f"more than {settings.rate_limit_per_minute} requests in the current minute",
                **{"Retry-After": str(retry)},
            )

        return await call_next(request)
