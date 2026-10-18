#!/usr/bin/env python3
"""Mock SOAR driver for the shadow-only lab: the safety matrix, end to end.

Plays the two roles a real deployment splits between a SOAR platform and an
approval service, against JEF and the mock targets on the lab network:

* as the **SOAR connector** it runs a scene with a ``case_id``, redeems the
  sealed proposal at ``/v1/decisions:authorize``, calls a mock target **only**
  on ``authorized``, and reports the outcome to ``:complete``;
* as the **approval stub** it signs AuthorizationGrants with the lab grant
  secret (``JEF_GRANT_SECRET``), as an approval service would after a human
  said yes. Nothing here is a real approval.

It never calls a target on anything but ``authorized``; the final check proves
that from the mock's own ledger: every recorded platform call has a receipt
with a grant nonce.

Phases (the CI job runs them in this order, restarting JEF in between):

``matrix``         auth, budgets, refusals, double spend, retry storm, the
                   happy path and the unknown timeout. Saves state for later.
``after-restart``  receipts and spent nonces survived ``docker compose restart
                   jef``; then the rate limit, last, because it spends the
                   minute's budget.
``store-down``     the decision store is unreachable: 503, nothing authorised.

Every check is written to ``--out`` as JSON; the exit code is non-zero if
any check failed. Standard library only, plus ``jef_scene.authz`` for grants
(run it in the JEF image).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ALERT = "IDS 回報來自 198.51.100.23 的連接埠掃描，該位址屬於委外弱點掃描服務，掃描視窗已核准。"
SCENE = "incident-triage"


class Http:
    def __init__(self, base: str, api_key: str | None) -> None:
        if not base.startswith(("http://", "https://")):
            raise ValueError(f"lab URLs are http(s) only, not {base!r}")
        self.base = base.rstrip("/")
        self.api_key = api_key

    def call(
        self,
        method: str,
        path: str,
        body: Any = None,
        *,
        headers: dict[str, str] | None = None,
        auth: bool = True,
        timeout: float = 15,
        raw: bytes | None = None,
        retry_429: bool = True,
    ) -> tuple[int, Any, dict[str, str]]:
        """``(status, json-or-text, headers)``. Waits out a 429 unless told not to."""
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
        h = {"content-type": "application/json", **(headers or {})}
        if auth and self.api_key:
            h["X-API-Key"] = self.api_key
        for _ in range(5):
            req = urllib.request.Request(  # noqa: S310 - http(s) checked in __init__
                self.base + path, data=data, method=method, headers=h
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - lab URLs
                    status, payload, rh = r.status, r.read(), dict(r.headers)
            except urllib.error.HTTPError as e:
                status, payload, rh = e.code, e.read(), dict(e.headers)
            if status == 429 and retry_429:
                time.sleep(float(rh.get("Retry-After") or rh.get("retry-after") or 5))
                continue
            try:
                return status, json.loads(payload or b"null"), rh
            except json.JSONDecodeError:
                return status, payload.decode(errors="replace"), rh
        return status, None, rh


def _declared_only(self: Http, path: str, length: int) -> int:
    """Declare a body of ``length`` bytes and read the answer before sending it.

    The server refuses on the declared size and closes the connection, so a
    client that kept uploading would see a reset instead of the 413.
    """
    import http.client
    from urllib.parse import urlsplit

    u = urlsplit(self.base)
    conn = http.client.HTTPConnection(u.hostname or "localhost", u.port or 80, timeout=15)
    try:
        conn.putrequest("POST", path)
        conn.putheader("content-type", "application/json")
        conn.putheader("content-length", str(length))
        if self.api_key:
            conn.putheader("X-API-Key", self.api_key)
        conn.endheaders()
        return conn.getresponse().status
    finally:
        conn.close()


Http.declared_only = _declared_only  # type: ignore[attr-defined]


class Report:
    def __init__(self, phase: str) -> None:
        self.phase = phase
        self.checks: list[dict[str, Any]] = []

    def check(self, case: str, ok: bool, detail: Any = "") -> bool:
        self.checks.append({"case": case, "ok": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {case}  {detail if not ok else ''}".rstrip())
        return ok

    @property
    def failed(self) -> int:
        return sum(not c["ok"] for c in self.checks)

    def write(self, out: Path, **extra: Any) -> None:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{self.phase}.json").write_text(
            json.dumps(
                {"phase": self.phase, "failed": self.failed, "checks": self.checks, **extra},
                indent=2,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )


class Soar:
    """The connector half: scenes, authorize, act on mocks only when authorized, complete."""

    def __init__(self, jef: Http, targets: Http, grant_secret: bytes) -> None:
        self.jef, self.targets, self.grant_secret = jef, targets, grant_secret

    def propose(self, case: str, asset: str) -> dict[str, Any]:
        status, body, _ = self.jef.call(
            "POST",
            f"/v1/scenes/{SCENE}:evaluate",
            {"state": ALERT, "case_id": case, "asset_id": asset},
        )
        if status != 200 or not isinstance(body, dict) or not body.get("proposal_token"):
            raise RuntimeError(f"scene run for {case} gave {status}: {body}")
        return body

    def grant(self, case: str, action: str, asset: str) -> dict[str, Any]:
        from jef_scene.authz import AuthorizationGrant

        return AuthorizationGrant.issue(
            secret=self.grant_secret,
            case_id=case,
            action=action,
            asset_id=asset,
            approvers=("lab-approver-01",),
            approver_role="soc-lead",
            issuer="lab-approval-stub",
        ).to_dict()

    def authorize(self, token: str, key: str, grant: Any = None) -> tuple[int, Any]:
        body: dict[str, Any] = {"proposal_token": token, "idempotency_key": key}
        if grant is not None:
            body["grant"] = grant
        status, payload, _ = self.jef.call("POST", "/v1/decisions:authorize", body)
        return status, payload

    def act(self, key: str, asset: str, behaviour: str = "ok") -> tuple[str, str | None]:
        """Call the mock EDR. ``("success"|"failed"|"timeout", ref)``."""
        headers = {"Idempotency-Key": key, "X-Mock-Behaviour": behaviour, "X-Mock-Delay": "4"}
        try:
            status, body, _ = self.targets.call(
                "POST",
                "/edr/isolate",
                {"asset_id": asset},
                headers=headers,
                auth=False,
                timeout=1.5,
            )
        except (TimeoutError, OSError):
            return "timeout", None
        if status == 200 and isinstance(body, dict) and body.get("ok"):
            return "success", body.get("ref")
        return "failed", None

    def complete(self, key: str, outcome: str, ref: str | None = None) -> tuple[int, Any]:
        body: dict[str, Any] = {"outcome": outcome, "detail": "reported by the mock SOAR"}
        if ref:
            body["external_ref"] = ref
        status, payload, _ = self.jef.call("POST", f"/v1/decisions/{key}:complete", body)
        return status, payload

    def run_case(self, case: str, asset: str, *, behaviour: str = "ok") -> dict[str, Any]:
        """The whole contract for one approved case. Returns what happened."""
        trace = self.propose(case, asset)
        key = f"{case}:{trace['action']}:{asset}"
        g = self.grant(case, trace["action"], asset)
        status, receipt = self.authorize(trace["proposal_token"], key, g)
        out: dict[str, Any] = {"key": key, "authorize": receipt, "action": trace["action"]}
        if status != 200 or receipt.get("status") != "authorized":
            return out  # never act on anything else
        outcome, ref = self.act(key, asset, behaviour)
        out["platform"] = outcome
        out["complete"] = self.complete(key, outcome, ref)[1]
        return out


# --------------------------------------------------------------------------- #
# phases
# --------------------------------------------------------------------------- #


def phase_matrix(soar: Soar, rep: Report, state: dict[str, Any]) -> None:
    jef = soar.jef
    run = state["run_id"] = time.strftime("%Y%m%dT%H%M%S")
    status, ready, _ = jef.call("GET", "/v1/readiness")
    rep.check("readiness answers", status == 200, status)
    rep.check(
        "readiness: the test backbone is not decision support",
        isinstance(ready, dict) and ready.get("decision_support_ready") is False,
        ready.get("blocking_reasons") if isinstance(ready, dict) else ready,
    )
    rep.check(
        "readiness: the decision store is persistent",
        isinstance(ready, dict) and ready.get("checks", {}).get("policy_store") == "persistent",
        ready.get("checks") if isinstance(ready, dict) else ready,
    )
    soar.targets.call("DELETE", "/calls", auth=False)

    # S01 authentication
    q1 = {"urgent": {"type": "noul", "instructions": "Is this urgent?"}}
    s, _, _ = jef.call("POST", "/v1/systemone", {"state": "x", "questions": q1}, auth=False)
    rep.check("S01 no API key: 401", s == 401, s)
    s, _, _ = jef.call("POST", "/v1/decisions:authorize", {}, auth=False)
    rep.check("S01 decisions without a key: 401", s == 401, s)

    # S02 request budgets, before inference
    s = jef.declared_only("/v1/systemone", 3_000_000)
    rep.check("S02 3 MB body: 413 before it is read", s == 413, s)
    deep: Any = "x"
    for _ in range(20):
        deep = {"d": deep}
    s, _, _ = jef.call("POST", "/v1/systemone", {"state": deep, "questions": q1})
    rep.check("S02 20-deep JSON: 422", s == 422, s)
    qs = {f"q{i}": {"type": "noul", "instructions": f"Question {i}?"} for i in range(30)}
    s, _, _ = jef.call("POST", "/v1/systemone", {"state": "x", "questions": qs})
    rep.check("S02 30 questions: 413", s == 413, s)

    # S03 no grant
    t = soar.propose(f"{run}-lab-case-nogrant", "host-a")
    s, r = soar.authorize(t["proposal_token"], f"{run}-lab-case-nogrant:k:1")
    rep.check(
        "S03 T2+ without a grant: not_dispatched NO_GRANT",
        s == 200
        and r.get("status") == "not_dispatched"
        and "NO_GRANT" in r.get("reason_codes", []),
        r,
    )

    # S04 forged grant, forged token
    s, r = soar.authorize(t["proposal_token"], f"{run}-lab-case-nogrant:k:2", {"approved": True})
    rep.check(
        "S04 forged grant: INVALID_GRANT",
        r.get("reason_codes") == ["INVALID_GRANT"] if isinstance(r, dict) else False,
        r,
    )
    g = soar.grant(f"{run}-lab-case-nogrant", t["action"], "host-a")
    g_edit = {**g, "asset_id": "host-z"}
    s, r = soar.authorize(t["proposal_token"], f"{run}-lab-case-nogrant:k:3", g_edit)
    rep.check(
        "S04 edited grant payload: refused",
        isinstance(r, dict) and r.get("status") == "not_dispatched",
        r,
    )
    s, r = soar.authorize(t["proposal_token"][:-4] + "0000", f"{run}-lab-case-nogrant:k:4", g)
    rep.check(
        "S04 forged proposal token: 422 invalid_proposal",
        s == 422 and isinstance(r, dict) and r.get("error", {}).get("code") == "invalid_proposal",
        (s, r),
    )

    # S05 cross-asset replay
    tb = soar.propose(f"{run}-lab-case-cross", "host-b")
    ga = soar.grant(f"{run}-lab-case-cross", tb["action"], "host-a")
    s, r = soar.authorize(tb["proposal_token"], f"{run}-lab-case-cross:k:1", ga)
    rep.check(
        "S05 grant for host A, proposal for host B: GRANT_ASSET_MISMATCH",
        isinstance(r, dict) and "GRANT_ASSET_MISMATCH" in r.get("reason_codes", []),
        r,
    )

    # Happy path: authorized, act once, success
    happy = soar.run_case(f"{run}-lab-case-happy", "host-c")
    rep.check(
        "S06 approved case: authorized, platform called, receipt success",
        happy.get("authorize", {}).get("status") == "authorized"
        and happy.get("platform") == "success"
        and happy.get("complete", {}).get("status") == "success",
        happy,
    )

    # S07 double spend: the happy grant with another key
    th = soar.propose(f"{run}-lab-case-happy", "host-c")
    gh = soar.grant(f"{run}-lab-case-happy", th["action"], "host-c")
    _, r1 = soar.authorize(th["proposal_token"], f"{run}-lab-case-ds:k:1", gh)
    _, r2 = soar.authorize(th["proposal_token"], f"{run}-lab-case-ds:k:2", gh)
    rep.check(
        "S07 one grant, two keys: second GRANT_ALREADY_USED",
        r1.get("status") == "authorized" and r2.get("reason_codes") == ["GRANT_ALREADY_USED"],
        (r1, r2),
    )
    # The SOAR performs the first one, as it is allowed to, so the ledger stays consistent.
    outcome, ref = soar.act(f"{run}-lab-case-ds:k:1", "host-c")
    soar.complete(f"{run}-lab-case-ds:k:1", outcome, ref)

    # S08 retry storm on one key, concurrently
    ts = soar.propose(f"{run}-lab-case-storm", "host-d")
    gs = soar.grant(f"{run}-lab-case-storm", ts["action"], "host-d")
    key = f"{run}-lab-case-storm:k:1"
    results: list[Any] = []
    lock = threading.Lock()

    def storm() -> None:
        _, r = soar.authorize(ts["proposal_token"], key, gs)
        with lock:
            results.append(json.dumps(r, sort_keys=True))

    threads = [threading.Thread(target=storm) for _ in range(10)]
    for th_ in threads:
        th_.start()
    for th_ in threads:
        th_.join()
    rep.check(
        "S08 ten concurrent retries: one receipt", len(set(results)) == 1, sorted(set(results))
    )
    first = json.loads(results[0])
    if first.get("status") == "authorized":
        outcome, ref = soar.act(key, "host-d")
        soar.complete(key, outcome, ref)

    # S09 the platform acted but did not answer
    late = soar.run_case(f"{run}-lab-case-timeout", "host-e", behaviour="timeout")
    rep.check(
        "S09 timeout: receipt unknown",
        late.get("platform") == "timeout" and late.get("complete", {}).get("status") == "unknown",
        late,
    )
    time.sleep(5)  # let the mock finish its delayed answer
    s, landed, _ = soar.targets.call("GET", f"/calls/{late['key']}", auth=False)
    rep.check("S09 reconciliation: the mock shows it landed", s == 200, landed)
    s, again = soar.complete(late["key"], "success")
    rep.check("S09 unknown is never upgraded to success: 409", s == 409, (s, again))

    state["receipts"] = {}
    for k in (happy["key"], late["key"], key, f"{run}-lab-case-ds:k:1"):
        s, r, _ = jef.call("GET", f"/v1/decisions/{k}")
        state["receipts"][k] = r
    state["spent_grant"] = {"token": th["proposal_token"], "grant": gh}
    ledger_check(soar, rep)


def ledger_check(soar: Soar, rep: Report) -> None:
    """Every platform call in the mock ledger has a receipt with a grant nonce."""
    s, ledger, _ = soar.targets.call("GET", "/calls", auth=False)
    calls = ledger.get("calls", []) if isinstance(ledger, dict) else []
    bad = []
    for c in calls:
        _, r, _ = soar.jef.call("GET", f"/v1/decisions/{c['idempotency_key']}")
        if (
            not isinstance(r, dict)
            or not r.get("grant_nonce")
            or r.get("status") == "not_dispatched"
        ):
            bad.append({"call": c["idempotency_key"], "receipt": r})
    rep.check(
        "S10 every mock platform call has an authorizing receipt with a grant nonce",
        s == 200 and calls and not bad,
        {"calls": len(calls), "without_grant": bad},
    )
    rep.check(
        "S10 no platform call was repeated",
        all(c.get("repeats", 0) == 0 for c in calls),
        [c["idempotency_key"] for c in calls if c.get("repeats")],
    )


def phase_after_restart(soar: Soar, rep: Report, state: dict[str, Any]) -> None:
    for key, before in state.get("receipts", {}).items():
        s, after, _ = soar.jef.call("GET", f"/v1/decisions/{key}")
        rep.check(
            f"S11 receipt for {key} survived the restart", s == 200 and after == before, after
        )
    spent = state.get("spent_grant")
    if spent:
        s, r = soar.authorize(
            spent["token"], f"{state['run_id']}-after-restart:k:1", spent["grant"]
        )
        rep.check(
            "S11 a grant spent before the restart is still spent",
            isinstance(r, dict) and r.get("reason_codes") == ["GRANT_ALREADY_USED"],
            r,
        )
    ledger_check(soar, rep)

    # S12 rate limit: last, because it spends the minute's budget.
    got = None
    for i in range(120):
        s, _, h = soar.jef.call("GET", "/v1/readiness", retry_429=False)
        if s == 429:
            got = (i + 1, h.get("Retry-After") or h.get("retry-after"))
            break
    rep.check("S12 flood from one caller: 429 with Retry-After", bool(got and got[1]), got)


def phase_store_down(soar: Soar, rep: Report, state: dict[str, Any]) -> None:
    t = state.get("store_down_proposal")
    s, r = soar.authorize(t["token"], f"{t['case']}:k:1", t["grant"]) if t else (0, None)
    rep.check(
        "S13 store unreachable: 503 store_unavailable, nothing authorised",
        s == 503 and isinstance(r, dict) and r.get("error", {}).get("code") == "store_unavailable",
        (s, r),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("phase", choices=["matrix", "after-restart", "prepare-store-down", "store-down"])
    p.add_argument("--jef", default=os.environ.get("JEF_URL", "http://jef:8080"))
    p.add_argument("--targets", default=os.environ.get("TARGETS_URL", "http://mock-targets:9009"))
    p.add_argument("--out", type=Path, default=Path(os.environ.get("SOAR_OUT", "/out")))
    args = p.parse_args(argv)
    api_key = os.environ.get("JEF_API_KEY", "")
    secret = os.environ.get("JEF_GRANT_SECRET", "").encode()
    if not api_key or not secret:
        print("JEF_API_KEY and JEF_GRANT_SECRET must be set (lab placeholders)", file=sys.stderr)
        return 2
    soar = Soar(Http(args.jef, api_key), Http(args.targets, None), secret)
    for _ in range(60):  # wait for JEF and the mocks
        try:
            if soar.jef.call("GET", "/readyz", auth=False)[0] == 200 and (
                soar.targets.call("GET", "/healthz", auth=False)[0] == 200
            ):
                break
        except OSError:
            pass
        time.sleep(1)
    state_path = args.out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if args.phase == "prepare-store-down":
        case = f"{state.get('run_id', 'run')}-lab-case-down"
        t = soar.propose(case, "host-f")
        state["store_down_proposal"] = {
            "case": case,
            "token": t["proposal_token"],
            "grant": soar.grant(case, t["action"], "host-f"),
        }
        args.out.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        print("prepared a proposal and grant for the store-down check")
        return 0
    rep = Report(args.phase)
    {"matrix": phase_matrix, "after-restart": phase_after_restart, "store-down": phase_store_down}[
        args.phase
    ](soar, rep, state)
    rep.write(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    print(f"{args.phase}: {len(rep.checks) - rep.failed} passed, {rep.failed} failed")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
