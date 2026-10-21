"""Raw Cloudflare security events -> pseudonymous, deduplicated requests.

Input is what ``cloudflare_source.ingest`` wrote. Output keeps what a reviewer
needs to judge an incident and drops what identifies people or leaks secrets:

* client IP, Ray ID and the request key become keyed HMAC pseudonyms
  (``JEF_PSEUDONYM_KEY``, at least 32 bytes, never written). The same IP maps
  to the same actor under one key and to an unrelated one under another, so a
  leaked pack cannot be joined to another pack built with a different key.
  An unsalted hash of an IPv4 address is reversible by enumeration; this is
  why a key is mandatory.
* the path becomes a route template (numeric ids, UUIDs, long hex or
  token-like segments become ``{id}``); the query keeps its parameter names
  and loses every value;
* the user agent becomes a coarse family (``browser:chrome``,
  ``scanner:sqlmap``, ``monitor:uptime`` ...);
* time is bucketed to the minute.

One request can raise several events (one per matching rule). Events sharing
a Ray ID -- or, when the zone does not return Ray IDs, the same client,
second and path -- are folded into one request that lists every event, so
three rules on one request count as one request with three detector outputs.

The detector fields (action, source, rule) are carried for ``weak.jsonl``
and the vendor-signal variant only. They are not labels.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote

__all__ = [
    "MIN_KEY_BYTES",
    "PseudonymError",
    "Pseudonymizer",
    "Request",
    "normalize_events",
    "query_keys",
    "read_raw",
    "route_template",
    "ua_family",
]

MIN_KEY_BYTES = 32

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_HEX = re.compile(r"^[0-9a-f]{8,}$", re.I)
_NUM = re.compile(r"^\d+$")
_TOKENISH = re.compile(r"^(?=.*\d)[A-Za-z0-9_\-=+.]{20,}$")
_EMAILISH = re.compile(r"@|%40")
_IPISH = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")

#: (pattern, family). First match wins; order matters (scanners before browsers).
_UA_FAMILIES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(p, re.I), f)
    for p, f in (
        (r"sqlmap", "scanner:sqlmap"),
        (r"nmap", "scanner:nmap"),
        (r"nikto", "scanner:nikto"),
        (r"nessus", "scanner:nessus"),
        (r"\bzap\b|owasp", "scanner:zap"),
        (r"burp", "scanner:burp"),
        (r"zgrab|masscan|nuclei", "scanner:mass"),
        (r"uptimerobot|pingdom|statuscake|uptime", "monitor:uptime"),
        (r"googlebot|bingbot|duckduckbot|baiduspider|yandex", "crawler:search"),
        (r"curl/", "client:curl"),
        (r"python-requests|aiohttp|httpx|urllib", "client:python"),
        (r"go-http-client", "client:go"),
        (r"java/|okhttp", "client:java"),
        (r"edg/", "browser:edge"),
        (r"chrome/", "browser:chrome"),
        (r"firefox/", "browser:firefox"),
        (r"safari/", "browser:safari"),
        (r"mozilla/", "browser:other"),
    )
)


class PseudonymError(ValueError):
    """No usable pseudonymisation key."""


class Pseudonymizer:
    """Keyed HMAC-SHA256 pseudonyms, one namespace per kind of identifier."""

    def __init__(self, key: bytes) -> None:
        if len(key) < MIN_KEY_BYTES:
            raise PseudonymError(f"the pseudonymisation key must be at least {MIN_KEY_BYTES} bytes")
        self._key = key

    @classmethod
    def from_env(cls, name: str = "JEF_PSEUDONYM_KEY") -> Pseudonymizer:
        value = os.environ.get(name, "")
        if not value:
            raise PseudonymError(f"environment variable {name} is not set")
        return cls(value.encode())

    def __repr__(self) -> str:
        return "Pseudonymizer(key=***)"

    def __call__(self, kind: str, value: str) -> str:
        digest = hmac.new(self._key, f"{kind}\x00{value}".encode(), hashlib.sha256).hexdigest()
        return f"{kind}-{digest[:16]}"


def route_template(path: str) -> str:
    """``/api/orders/123456`` -> ``/api/orders/{id}``. Never keeps an identifier."""
    path = unquote(path or "/").split("?", 1)[0]
    out = []
    for seg in path.split("/"):
        if not seg:
            continue
        if (
            _NUM.match(seg)
            or _UUID.match(seg)
            or _HEX.match(seg)
            or _TOKENISH.match(seg)
            or _EMAILISH.search(seg)
            or _IPISH.match(seg)
        ):
            out.append("{id}")
        else:
            out.append(seg[:64])
    return "/" + "/".join(out)


def route_family(template: str, depth: int = 2) -> str:
    """The first ``depth`` segments: ``/api/orders/{id}`` -> ``/api/orders``."""
    parts = [p for p in template.split("/") if p][:depth]
    return "/" + "/".join(parts)


def query_keys(query: str) -> list[str]:
    """Parameter names only, sorted and deduplicated; every value is dropped."""
    q = (query or "").lstrip("?")
    names = {k[:40] for k, _ in parse_qsl(q, keep_blank_values=True) if k}
    return sorted(n for n in names if re.fullmatch(r"[A-Za-z0-9_.\-\[\]]+", n))


def ua_family(ua: str) -> str:
    if not ua:
        return "none"
    for pattern, family in _UA_FAMILIES:
        if pattern.search(ua):
            return family
    return "other"


def _minute(ts: str) -> str:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%MZ")


@dataclass
class Request:
    """One HTTP request as the edge saw it, with every security event it raised."""

    request_id: str
    minute: str
    actor: str
    asn: str
    country: str
    method: str | None
    status: int | None
    route: str
    query_keys: list[str]
    ua_family: str
    sample_interval: float | None
    detections: list[dict[str, str | None]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "minute": self.minute,
            "actor": self.actor,
            "asn": self.asn,
            "country": self.country,
            "method": self.method,
            "status": self.status,
            "route": self.route,
            "query_keys": self.query_keys,
            "ua_family": self.ua_family,
            "sample_interval": self.sample_interval,
            "detections": self.detections,
        }


def read_raw(raw_dir: Path) -> list[dict[str, Any]]:
    """Every row under ``raw_dir/events/*.jsonl``, in file order."""
    rows: list[dict[str, Any]] = []
    for p in sorted((Path(raw_dir) / "events").glob("*.jsonl")):
        rows += [json.loads(line) for line in p.read_text("utf-8").splitlines() if line.strip()]
    return rows


def normalize_events(rows: Iterable[Mapping[str, Any]], pseud: Pseudonymizer) -> list[Request]:
    """Pseudonymise and fold events into requests (request-level dedup)."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        ray = r.get("rayName")
        raw_key = (
            f"ray:{ray}"
            if ray
            else f"tuple:{r.get('clientIP')}|{r.get('datetime')}|{r.get('clientRequestPath')}"
        )
        grouped[raw_key].append(r)
    out: list[Request] = []
    for raw_key, events in grouped.items():
        first = min(events, key=lambda e: str(e.get("datetime")))
        intervals = [float(e["sampleInterval"]) for e in events if e.get("sampleInterval")]
        status = first.get("edgeResponseStatus")
        detections = sorted(
            {
                (str(e.get("action")), str(e.get("source")), e.get("ruleId") and str(e["ruleId"]))
                for e in events
            },
            key=lambda d: tuple(x or "" for x in d),
        )
        out.append(
            Request(
                request_id=pseud("req", raw_key),
                minute=_minute(str(first["datetime"])),
                actor=pseud("actor", str(first.get("clientIP") or "")),
                asn=str(first.get("clientAsn") or "unknown"),
                country=str(first.get("clientCountryName") or "unknown"),
                method=first.get("clientRequestHTTPMethodName"),
                status=int(status) if status is not None else None,
                route=route_template(str(first.get("clientRequestPath") or "/")),
                query_keys=query_keys(str(first.get("clientRequestQuery") or "")),
                ua_family=ua_family(str(first.get("userAgent") or "")),
                sample_interval=max(intervals) if intervals else None,
                detections=[{"action": a, "source": s, "rule": rl} for a, s, rl in detections],
            )
        )
    out.sort(key=lambda q: (q.minute, q.actor, q.route, q.request_id))
    return out
