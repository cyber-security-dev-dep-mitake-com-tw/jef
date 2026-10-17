#!/usr/bin/env python3
"""Mock EDR / IAM / firewall targets for the shadow-only lab (NEXT-07).

Stands in for the platforms a SOAR connector would call after JEF says
``authorized``. It performs nothing real: it records every call with its
idempotency key, answers like a platform would, and can be told to misbehave
so the receipt and reconciliation paths get exercised:

* ``POST /edr/isolate``, ``/iam/disable``, ``/fw/block``, ``/siem/close``:
  record the call; reply ``{"ok": true, "ref": "<platform ref>"}``.
* ``X-Mock-Behaviour: timeout`` on a request: the call is **recorded** (the
  action "landed") but the response is delayed past any sane client timeout,
  so the caller must record ``unknown`` and reconcile, never retry blindly.
* ``X-Mock-Behaviour: error``: recorded as attempted, reply 500.
* ``GET /calls``: everything recorded, for the evidence pack. ``GET /calls/<key>``
  answers the reconciliation question "did this land?".
* ``DELETE /calls``: reset between scenarios.

Binds to loopback by default. It has no credentials and no reach beyond its
own process, which is the point of a mock target.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from contextlib import suppress
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

ACTIONS = {
    "/edr/isolate": "isolate_host",
    "/iam/disable": "disable_user",
    "/fw/block": "block_ip",
    "/siem/close": "close_alert",
}


class Ledger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: list[dict[str, Any]] = []

    def record(self, entry: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            # A platform with idempotency of its own returns the first result
            # for a repeated key instead of acting twice.
            for seen in self.calls:
                if seen["idempotency_key"] and seen["idempotency_key"] == entry["idempotency_key"]:
                    seen["repeats"] += 1
                    return seen
            entry["ref"] = f"mock-{len(self.calls) + 1:05d}"
            entry["repeats"] = 0
            self.calls.append(entry)
            return entry

    def find(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            return next((c for c in self.calls if c["idempotency_key"] == key), None)

    def reset(self) -> None:
        with self._lock:
            self.calls.clear()


LEDGER = Ledger()


class Handler(BaseHTTPRequestHandler):
    server_version = "jef-mock-targets/1"

    def _json(self, status: int, body: Any) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter than the default
        return

    def do_GET(self) -> None:  # http.server API
        if self.path == "/calls":
            return self._json(200, {"calls": LEDGER.calls, "count": len(LEDGER.calls)})
        if self.path.startswith("/calls/"):
            found = LEDGER.find(self.path[len("/calls/") :])
            return self._json(200 if found else 404, found or {"landed": False})
        if self.path == "/healthz":
            return self._json(200, {"ok": True, "mock": True})
        return self._json(404, {"error": "unknown route"})

    def do_DELETE(self) -> None:
        if self.path == "/calls":
            LEDGER.reset()
            return self._json(200, {"ok": True})
        return self._json(404, {"error": "unknown route"})

    def do_POST(self) -> None:
        action = ACTIONS.get(self.path)
        if action is None:
            return self._json(404, {"error": f"no mock target at {self.path}"})
        length = int(self.headers.get("content-length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": "body is not JSON"})
        behaviour = (self.headers.get("x-mock-behaviour") or "ok").lower()
        entry = LEDGER.record(
            {
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                "action": action,
                "path": self.path,
                "asset_id": payload.get("asset_id"),
                "idempotency_key": self.headers.get("idempotency-key")
                or payload.get("idempotency_key"),
                "behaviour": behaviour,
                "payload": payload,
            }
        )
        if behaviour == "timeout":
            # The action landed (it is in the ledger) but the caller never hears so.
            time.sleep(float(self.headers.get("x-mock-delay") or 45))
            return self._json(200, {"ok": True, "ref": entry["ref"], "late": True})
        if behaviour == "error":
            return self._json(500, {"ok": False, "error": "platform error (mock)"})
        return self._json(200, {"ok": True, "ref": entry["ref"], "repeats": entry["repeats"]})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--host", default="127.0.0.1", help="loopback by default; the lab vSwitch at most"
    )
    p.add_argument("--port", type=int, default=9009)
    args = p.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock targets on http://{args.host}:{args.port}  (EDR/IAM/firewall/SIEM; records only)")
    with suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
