"""Read-only Cloudflare Security Events reader (``firewallEventsAdaptive``).

Pulls WAF / security events for one zone the operator owns, window by window,
into a private directory outside any git checkout. It never writes the API
token or the zone id anywhere: not to the output, the run log, an exception
message or a repr. Everything it records about the zone is the alias the
operator chose.

What this data is and is not:

* An event says a Cloudflare detector acted (``block``, ``managed_challenge``,
  ``log`` ...). That is a detector output, Tier D in
  ``benchmark/socbench/sources/sources.yaml``, never a label. An ``allow``
  that raised no event is unlabelled, not benign.
* ``firewallEventsAdaptive`` may be sampled, and one request can raise several
  events. The run log keeps ``sampled`` and the sample interval when the API
  returns one, so nothing downstream can turn event counts into a rate over all
  requests.
* Field availability, history and limits depend on the zone's plan. Only the
  fields in the official query tutorial are required; every other field is
  optional and dropped, with a note in the run log, when the API refuses it.
  Nothing is guessed and no retention period is hard-coded.

The transport is injectable so every path (401, 403, 429, 5xx, GraphQL
``errors`` under HTTP 200, the page boundary, resume) is tested against
recorded mock responses. CI never calls Cloudflare.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

__all__ = [
    "ENDPOINT",
    "OPTIONAL_FIELDS",
    "REQUIRED_FIELDS",
    "CloudflareAuthError",
    "CloudflareError",
    "CloudflareQueryError",
    "GraphQLClient",
    "IngestResult",
    "PrivatePathError",
    "check_alias",
    "check_private_dir",
    "ingest",
    "windows",
]

ENDPOINT = "https://api.cloudflare.com/client/v4/graphql"

#: The fields in Cloudflare's "Query firewall events" tutorial. A zone that
#: refuses one of these cannot be ingested honestly.
REQUIRED_FIELDS: tuple[str, ...] = (
    "action",
    "datetime",
    "source",
    "clientAsn",
    "clientCountryName",
    "clientIP",
    "clientRequestPath",
    "clientRequestQuery",
    "userAgent",
)

#: Asked for, kept when the zone's schema has them, dropped (and reported)
#: when it does not. ``rayName`` lets several events of one request be
#: deduplicated; ``sampleInterval`` says how much was sampled.
OPTIONAL_FIELDS: tuple[str, ...] = (
    "rayName",
    "ruleId",
    "clientRequestHTTPMethodName",
    "edgeResponseStatus",
    "sampleInterval",
)

#: Status codes worth another attempt after a pause.
RETRYABLE = frozenset({429, 500, 502, 503, 504})
MAX_LIMIT = 10_000

# (status, headers, body) for one POST.
Transport = Callable[[str, bytes, Mapping[str, str]], tuple[int, Mapping[str, str], bytes]]

_UNKNOWN_FIELD = re.compile(
    r"(?:unknown field|cannot query field|no such field)\W+\"?([A-Za-z_][A-Za-z0-9_]*)", re.I
)


class CloudflareError(RuntimeError):
    """Anything that stops an ingest. Messages never carry the token or zone id."""


class CloudflareAuthError(CloudflareError):
    """401 or 403: wrong token, wrong scope, or a zone the token cannot read."""


class CloudflareQueryError(CloudflareError):
    """GraphQL ``errors`` (including under HTTP 200), or no ``data``."""


class PrivatePathError(CloudflareError):
    """An output directory inside a git checkout, where it could be committed."""


def _default_transport(timeout: float) -> Transport:
    def post(
        url: str, body: bytes, headers: Mapping[str, str]
    ) -> tuple[int, dict[str, str], bytes]:
        import urllib.error
        import urllib.request

        if not url.startswith("https://"):
            raise CloudflareError("the Cloudflare reader posts only to https")
        req = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")  # noqa: S310
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return resp.status, dict(resp.headers.items()), resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, dict(exc.headers.items()), exc.read()

    return post


def check_private_dir(path: Path) -> Path:
    """``path`` resolved, refused if it (or a parent) is a git working tree."""
    resolved = Path(path).expanduser().resolve()
    for p in [resolved, *resolved.parents]:
        if (p / ".git").exists():
            raise PrivatePathError(
                f"refusing to write Cloudflare data inside the git checkout at {p}; "
                "use a private directory outside any repository"
            )
    return resolved


def _private_dirs(path: Path) -> None:
    """Create ``path`` and any missing parent owner-only (``mkdir(parents=True)``
    gives intermediate directories the umask default, usually world-readable)."""
    missing = [p for p in [path, *path.parents] if not p.exists()]
    for p in reversed(missing):
        p.mkdir(mode=0o700)


def _write_private(path: Path, text: str) -> None:
    _private_dirs(path.parent)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)


def _append_private(path: Path, text: str) -> None:
    _private_dirs(path.parent)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(text)


def _iso(ts: datetime) -> str:
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)


def check_alias(zone_alias: str, zone_id: str | None = None) -> str:
    """A short human name for the zone. Refused if it is, or looks like, a zone id."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", zone_alias):
        raise CloudflareError("zone alias: lowercase letters, digits and dashes, up to 41")
    if re.fullmatch(r"[0-9a-f]{32}", zone_alias) or (zone_id and zone_alias == zone_id.lower()):
        raise CloudflareError("zone alias: choose a name (e.g. site-a), not the zone id")
    return zone_alias


def windows(start: datetime, end: datetime, step: timedelta) -> Iterator[tuple[datetime, datetime]]:
    """Half-open ``[a, b)`` windows covering ``[start, end)``, the last one short."""
    if step <= timedelta(0):
        raise ValueError("window step must be positive")
    if end <= start:
        raise ValueError("end must be after start")
    a = start
    while a < end:
        b = min(a + step, end)
        yield a, b
        a = b


@dataclass
class GraphQLClient:
    """One zone, one token, read-only queries."""

    token: str = field(repr=False)
    zone_id: str = field(repr=False)
    transport: Transport | None = None
    endpoint: str = ENDPOINT
    retries: int = 5
    timeout: float = 60.0
    sleep: Callable[[float], None] = time.sleep
    attempts: int = 0
    failures: int = 0

    def __post_init__(self) -> None:
        if not self.token:
            raise CloudflareError("no API token")
        if not self.zone_id:
            raise CloudflareError("no zone id")
        self._post = self.transport or _default_transport(self.timeout)

    def __repr__(self) -> str:
        return f"GraphQLClient(endpoint={self.endpoint!r}, token=***, zone_id=***)"

    def redact(self, text: str) -> str:
        for secret in (self.token, self.zone_id):
            if secret:
                text = text.replace(secret, "***")
        return text

    def query(self, query: str, variables: Mapping[str, Any]) -> dict[str, Any]:
        """``data`` of one GraphQL call, after retries; raises on any error."""
        body = json.dumps({"query": query, "variables": dict(variables)}).encode()
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        for attempt in range(self.retries + 1):
            self.attempts += 1
            try:
                status, resp_headers, raw = self._post(self.endpoint, body, headers)
            except CloudflareError:
                raise
            except Exception as exc:  # network errors: redact and retry
                self.failures += 1
                if attempt == self.retries:
                    raise CloudflareError(self.redact(f"transport error: {exc}")) from None
                self.sleep(self._backoff(attempt, None))
                continue
            if status in (401, 403):
                self.failures += 1
                raise CloudflareAuthError(
                    f"HTTP {status}: the token cannot read this zone's analytics "
                    "(needs Analytics Read on the zone)"
                )
            if status in RETRYABLE:
                self.failures += 1
                if attempt == self.retries:
                    raise CloudflareError(f"HTTP {status} after {attempt + 1} attempts")
                self.sleep(self._backoff(attempt, resp_headers.get("Retry-After")))
                continue
            if status != 200:
                self.failures += 1
                raise CloudflareError(f"HTTP {status}")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                self.failures += 1
                raise CloudflareQueryError("response is not JSON") from None
            errors = payload.get("errors") or []
            if errors:
                self.failures += 1
                messages = "; ".join(str(e.get("message", e)) for e in errors)
                raise CloudflareQueryError(self.redact(messages))
            data = payload.get("data")
            if not isinstance(data, dict):
                self.failures += 1
                raise CloudflareQueryError("response has no data")
            return data
        raise CloudflareError("unreachable")  # pragma: no cover

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), 120.0)
            except ValueError:
                pass
        return min(2.0**attempt, 60.0) + random.uniform(0, 0.5)  # noqa: S311 -- jitter


def _events_query(fields: Sequence[str], limit: int) -> str:
    selection = "\n          ".join(fields)
    return f"""query JefFirewallEvents($zoneTag: string, $filter: FirewallEventsAdaptiveFilter_InputObject) {{
  viewer {{
    zones(filter: {{ zoneTag: $zoneTag }}) {{
      firewallEventsAdaptive(filter: $filter, limit: {int(limit)}, orderBy: [datetime_ASC]) {{
          {selection}
      }}
    }}
  }}
}}"""


def _rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    zones = (data.get("viewer") or {}).get("zones") or []
    if not zones:
        raise CloudflareQueryError("the token sees no such zone")
    rows = zones[0].get("firewallEventsAdaptive")
    if not isinstance(rows, list):
        raise CloudflareQueryError("no firewallEventsAdaptive in the response")
    return rows


def resolve_fields(
    client: GraphQLClient, probe_window: tuple[datetime, datetime]
) -> tuple[list[str], list[str]]:
    """The fields this zone answers, and the optional ones it refused.

    Asks for everything with ``limit: 1``; each "unknown field" error drops
    that optional field and asks again. A refused required field is an error.
    """
    fields = [*REQUIRED_FIELDS, *OPTIONAL_FIELDS]
    dropped: list[str] = []
    variables = {
        "zoneTag": client.zone_id,
        "filter": {"datetime_geq": _iso(probe_window[0]), "datetime_lt": _iso(probe_window[1])},
    }
    for _ in range(len(OPTIONAL_FIELDS) + 1):
        try:
            _rows(client.query(_events_query(fields, 1), variables))
            return fields, dropped
        except CloudflareQueryError as exc:
            names = [m for m in _UNKNOWN_FIELD.findall(str(exc)) if m in fields]
            if not names:
                raise
            for name in names:
                if name in REQUIRED_FIELDS:
                    raise CloudflareQueryError(
                        f"this zone does not offer required field {name}; cannot ingest"
                    ) from None
                fields.remove(name)
                dropped.append(name)
    raise CloudflareQueryError("could not settle on a field list")  # pragma: no cover


def _row_key(row: Mapping[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def fetch_window(
    client: GraphQLClient,
    fields: Sequence[str],
    start: datetime,
    end: datetime,
    *,
    limit: int,
    max_pages: int = 1000,
) -> tuple[list[dict[str, Any]], bool]:
    """Every event in ``[start, end)``, paging on ``datetime``; and whether it is complete.

    The API has no cursor. A full page moves the lower bound to the last
    event's timestamp (inclusive, because several events share a second) and
    drops rows already seen at that boundary. A page whose events all share
    the boundary second cannot advance: the window is returned incomplete,
    never silently truncated.
    """
    query = _events_query(fields, limit)
    out: list[dict[str, Any]] = []
    seen_at_boundary: set[str] = set()
    lower = start
    for _ in range(max_pages):
        variables = {
            "zoneTag": client.zone_id,
            "filter": {"datetime_geq": _iso(lower), "datetime_lt": _iso(end)},
        }
        page = _rows(client.query(query, variables))
        fresh = [r for r in page if _row_key(r) not in seen_at_boundary]
        out.extend(fresh)
        if len(page) < limit:
            return out, True
        last = _parse(str(page[-1]["datetime"]))
        if last <= lower and not fresh:
            return out, False
        if last > lower:
            seen_at_boundary = set()
        seen_at_boundary |= {_row_key(r) for r in page if _parse(str(r["datetime"])) == last}
        lower = last
    return out, False


@dataclass
class IngestResult:
    windows: int = 0
    skipped: int = 0
    events: int = 0
    incomplete: list[str] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)
    dropped_fields: list[str] = field(default_factory=list)
    sampled: bool | None = None
    attempts: int = 0
    failures: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "windows": self.windows,
            "skipped_already_done": self.skipped,
            "events": self.events,
            "incomplete_windows": self.incomplete,
            "fields": self.fields,
            "dropped_fields": self.dropped_fields,
            "sampled": "unknown" if self.sampled is None else self.sampled,
            "attempts": self.attempts,
            "failures": self.failures,
        }


def _sampled(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bool | None:
    if "sampleInterval" not in fields:
        return None
    intervals = [float(r["sampleInterval"]) for r in rows if r.get("sampleInterval") is not None]
    if not intervals:
        return None
    return any(i > 1 for i in intervals)


def ingest(
    client: GraphQLClient,
    *,
    zone_alias: str,
    start: datetime,
    end: datetime,
    out_dir: Path,
    step: timedelta = timedelta(hours=1),
    limit: int = 1000,
) -> IngestResult:
    """Pull ``[start, end)`` into ``out_dir`` window by window, resuming from the watermark.

    Writes ``events/<window-start>.jsonl`` (raw rows, mode 0600), appends one
    line per window to ``runs.jsonl`` and keeps ``watermark.json`` -- the end
    of the last complete window -- so a rerun skips what is done.
    """
    check_alias(zone_alias, client.zone_id)
    if not 1 <= limit <= MAX_LIMIT:
        raise CloudflareError(f"limit must be between 1 and {MAX_LIMIT}")
    out = check_private_dir(out_dir)
    mark_path = out / "watermark.json"
    watermark = None
    if mark_path.is_file():
        mark = json.loads(mark_path.read_text("utf-8"))
        if mark.get("zone_alias") != zone_alias:
            raise CloudflareError(
                f"{mark_path} belongs to zone alias {mark.get('zone_alias')!r}, not {zone_alias!r}"
            )
        watermark = _parse(mark["complete_until"])

    result = IngestResult()
    spans = list(windows(start, end, step))
    fields, dropped = resolve_fields(client, spans[0])
    result.fields, result.dropped_fields = list(fields), list(dropped)
    sampled_flags: list[bool | None] = []
    for a, b in spans:
        if watermark is not None and b <= watermark:
            result.skipped += 1
            continue
        rows, complete = fetch_window(client, fields, a, b, limit=limit)
        _write_private(
            out / "events" / f"{_iso(a).replace(':', '')}.jsonl",
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
        )
        sampled = _sampled(rows, fields)
        sampled_flags.append(sampled)
        result.windows += 1
        result.events += len(rows)
        if not complete:
            result.incomplete.append(_iso(a))
        _append_private(
            out / "runs.jsonl",
            json.dumps(
                {
                    "zone_alias": zone_alias,
                    "dataset": "firewallEventsAdaptive",
                    "window": [_iso(a), _iso(b)],
                    "events": len(rows),
                    "complete": complete,
                    "sampled": "unknown" if sampled is None else sampled,
                    "fields": list(fields),
                    "dropped_fields": list(dropped),
                    "fetched_at": _iso(datetime.now(UTC)),
                },
                sort_keys=True,
            )
            + "\n",
        )
        # The watermark only moves over complete windows, in order.
        if complete and not result.incomplete:
            _write_private(
                mark_path,
                json.dumps({"zone_alias": zone_alias, "complete_until": _iso(b)}) + "\n",
            )
    known = [s for s in sampled_flags if s is not None]
    result.sampled = (any(known) if known else None) if sampled_flags else None
    result.attempts, result.failures = client.attempts, client.failures
    return result
