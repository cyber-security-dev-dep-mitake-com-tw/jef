"""A recorded stand-in for the Cloudflare GraphQL endpoint, for tests and CI.

``FakeZone`` answers the reader's ``firewallEventsAdaptive`` queries from a
list of synthetic rows (``benchmark/incident-triage/fixtures/cloudflare-synthetic``),
honouring the window filter, ``orderBy: [datetime_ASC]`` and ``limit`` the
way the real API does, so paging and resume are exercised for real. A
``script`` of canned failures (401, 403, 429, 5xx, GraphQL ``errors`` under
HTTP 200) is played before the normal answers, and ``refuse`` makes chosen
fields "unknown", as a zone on a smaller plan would.

It checks the bearer token and zone it was built with, so a test proves the
reader sent them, and records every request for assertions.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["FakeZone", "load_fixture_events"]

_SELECTION = re.compile(r"firewallEventsAdaptive\([^)]*\)\s*\{([^}]*)\}", re.S)
_LIMIT = re.compile(r"limit:\s*(\d+)")


def load_fixture_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text("utf-8").splitlines() if line.strip()]


@dataclass
class FakeZone:
    token: str
    zone_id: str
    events: Sequence[Mapping[str, Any]]
    script: list[tuple[int, Any]] = field(default_factory=list)
    refuse: frozenset[str] = frozenset()
    requests: list[dict[str, Any]] = field(default_factory=list)

    def __call__(
        self, url: str, body: bytes, headers: Mapping[str, str]
    ) -> tuple[int, dict[str, str], bytes]:
        payload = json.loads(body)
        self.requests.append({"url": url, "headers": dict(headers), "payload": payload})
        if self.script:
            status, reply = self.script.pop(0)
            hdrs = {"Retry-After": "0"} if status == 429 else {}
            return status, hdrs, json.dumps(reply).encode()
        if headers.get("Authorization") != f"Bearer {self.token}":
            return 401, {}, b'{"errors":[{"message":"authentication error"}]}'
        query = payload["query"]
        variables = payload["variables"]
        if variables.get("zoneTag") != self.zone_id:
            return 200, {}, json.dumps({"data": {"viewer": {"zones": []}}}).encode()
        selection_match = _SELECTION.search(query)
        limit_match = _LIMIT.search(query)
        if not selection_match or not limit_match:
            return 200, {}, b'{"data":null,"errors":[{"message":"parse error"}]}'
        fields = selection_match.group(1).split()
        refused = [f for f in fields if f in self.refuse]
        if refused:
            message = f'unknown field "{refused[0]}"'
            return 200, {}, json.dumps({"data": None, "errors": [{"message": message}]}).encode()
        flt = variables["filter"]
        lo, hi = flt["datetime_geq"], flt["datetime_lt"]
        rows = sorted(
            (e for e in self.events if lo <= str(e["datetime"]) < hi),
            key=lambda e: (str(e["datetime"]), str(e.get("rayName")), str(e.get("ruleId"))),
        )[: int(limit_match.group(1))]
        data = {
            "viewer": {
                "zones": [{"firewallEventsAdaptive": [{f: r.get(f) for f in fields} for r in rows]}]
            }
        }
        return 200, {}, json.dumps({"data": data, "errors": None}).encode()
