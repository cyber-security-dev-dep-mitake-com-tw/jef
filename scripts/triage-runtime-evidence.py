#!/usr/bin/env python3
"""Runtime evidence for incident-triage on an existing deployment (plan 2026-10-20, PR-08).

    JEF_API_KEY=... python scripts/triage-runtime-evidence.py \\
        --base-url http://jef.lab.internal:8080 \\
        --pack benchmark/incident-triage/canary/six-alerts-v1 \\
        --levels 1,2,4,8 --requests 40 --warmup 5 \\
        --kube-namespace jef --kube-selector app=jef \\
        [--mock-ledger-url http://mock-targets:9009] \\
        --out evidence/triage-runtime/<date>

Run by an operator against the existing K8S lab, never from public CI: the
output names hosts and may carry alert text. What it records, and keeps apart:

* **Service**: cold first request and warm latency per concurrency level
  (p50/p95/p99 with their sample sizes), throughput, 5xx and timeouts. A
  percentile is reported only when the level has enough samples for it (20
  for p95, 100 for p99); otherwise it says so instead of printing a number.
* **Safety**: every response's policy verdict, proposal and execution status.
  Requests never carry a ``case_id``, so the server seals no proposal and
  nothing can be authorised from this run.
* **Delivery**: ``review_required`` is a policy verdict, not a ticket. With
  ``--mock-ledger-url`` the mock target's call count is read before and after
  and must not move; without it, delivery is reported as *not measured*.
* **Platform**: with ``--kube-namespace``, pod image digests, node, restarts
  before and after, ServiceAccount, NetworkPolicy names and ``kubectl top``.
* **Provenance**: git SHA and scene sha256 of this checkout, the model the
  server reports, and its readiness with addresses and hostnames masked.

The API key is read from the environment variable named by ``--api-key-env``
and is never written anywhere.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: (method, path, body or None) -> (status, parsed JSON or None, seconds)
Send = Callable[[str, str, Any], tuple[int, Any, float]]
#: argv -> stdout; raises on failure
Kubectl = Callable[[Sequence[str]], str]

SCENE = "incident-triage"
MIN_FOR = {"p50": 1, "p95": 20, "p99": 100}
_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_HOST = re.compile(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", re.IGNORECASE)


def http_sender(base_url: str, api_key: str | None, timeout: float) -> Send:
    def send(method: str, path: str, body: Any) -> tuple[int, Any, float]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(base_url.rstrip("/") + path, data=data, method=method)  # noqa: S310
        req.add_header("Content-Type", "application/json")
        if api_key:
            req.add_header("X-API-Key", api_key)
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 -- operator-given URL
                raw = resp.read()
                status = resp.status
        except urllib.error.HTTPError as exc:
            raw, status = exc.read(), exc.code
        except (TimeoutError, urllib.error.URLError):
            return -1, None, time.perf_counter() - t0
        elapsed = time.perf_counter() - t0
        try:
            return status, json.loads(raw or b"null"), elapsed
        except json.JSONDecodeError:
            return status, None, elapsed

    return send


def redact(value: Any) -> Any:
    """Mask addresses and hostnames anywhere in a JSON value."""
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return _HOST.sub("<host>", _IP.sub("<ip>", value))
    return value


def percentiles(samples: Sequence[float]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(samples)}
    if not samples:
        return out
    ordered = sorted(samples)
    for name, need in MIN_FOR.items():
        q = int(name[1:]) / 100
        if len(samples) >= need:
            # Nearest rank: ceil(q * n), 1-based.
            idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
            out[f"{name}_ms"] = round(ordered[idx] * 1000, 1)
        else:
            out[f"{name}_ms"] = f"n<{need}: not reported"
    out["max_ms"] = round(ordered[-1] * 1000, 1)
    return out


def _evaluate(send: Send, state: str) -> tuple[int, Any, float]:
    # No case_id: the server seals no proposal, so nothing here can be authorised.
    return send("POST", f"/v1/scenes/{SCENE}:evaluate", {"state": state})


def _safety(trace: Any) -> dict[str, Any]:
    if not isinstance(trace, dict):
        return {"ok": False}
    return {
        "ok": True,
        "policy_verdict": trace.get("policy_verdict"),
        "proposed_action": trace.get("proposed_action"),
        "advisory_only": trace.get("advisory_only"),
        "execution_status": trace.get("execution_status"),
        "executed_action": trace.get("executed_action"),
    }


def run_load(
    send: Send, states: Sequence[str], levels: Sequence[int], requests: int, warmup: int
) -> dict[str, Any]:
    cold_status, _cold_trace, cold_s = _evaluate(send, states[0])
    for i in range(warmup):
        _evaluate(send, states[i % len(states)])
    out: dict[str, Any] = {
        "cold_first_request": {"status": cold_status, "ms": round(cold_s * 1000, 1)},
        "warmup_requests": warmup,
        "levels": {},
    }
    safety: Counter[str] = Counter()
    for level in levels:
        jobs = [states[i % len(states)] for i in range(requests)]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=level) as pool:
            results = list(pool.map(lambda s: _evaluate(send, s), jobs))
        wall = time.perf_counter() - t0
        ok = [sec for status, _, sec in results if status == 200]
        for status, trace, _ in results:
            s = _safety(trace) if status == 200 else {"ok": False}
            if s["ok"]:
                safety[f"policy_verdict:{s['policy_verdict']}"] += 1
                safety[f"execution_status:{s['execution_status']}"] += 1
                safety[f"proposed:{s['proposed_action']}"] += 1
                safety["executed"] += s["executed_action"] is not None
        out["levels"][str(level)] = {
            "requests": requests,
            "ok": len(ok),
            "status_5xx": sum(500 <= st < 600 for st, _, _ in results),
            "status_4xx": sum(400 <= st < 500 for st, _, _ in results),
            "timeouts_or_unreachable": sum(st == -1 for st, _, _ in results),
            "throughput_rps": round(len(ok) / wall, 3) if wall else None,
            "latency": percentiles(ok),
        }
    out["safety"] = dict(sorted(safety.items()))
    # A run with no successful response has shown nothing about safety: every
    # level and the cold request must have answered at least once.
    answered = cold_status == 200 and all(m["ok"] > 0 for m in out["levels"].values())
    out["answered"] = answered
    out["safety_ok"] = (
        answered
        and safety.get("executed", 0) == 0
        and all(
            k == "execution_status:not_dispatched"
            for k in safety
            if k.startswith("execution_status:")
        )
    )
    return out


def ledger_calls(send_ledger: Send | None) -> int | None:
    if send_ledger is None:
        return None
    status, body, _ = send_ledger("GET", "/calls", None)
    if status != 200:
        return None
    return (
        len(body)
        if isinstance(body, list)
        else len(body.get("calls", []))
        if isinstance(body, dict)
        else None
    )


def kube_facts(kubectl: Kubectl, namespace: str, selector: str) -> dict[str, Any]:
    facts: dict[str, Any] = {"namespace": namespace, "selector": selector}
    pods = json.loads(kubectl(["get", "pods", "-n", namespace, "-l", selector, "-o", "json"]))
    facts["pods"] = [
        {
            "name": p["metadata"]["name"],
            "node": p["spec"].get("nodeName"),
            "service_account": p["spec"].get("serviceAccountName"),
            "images": [c.get("imageID") for c in p.get("status", {}).get("containerStatuses", [])],
            "restarts": sum(
                c.get("restartCount", 0) for c in p.get("status", {}).get("containerStatuses", [])
            ),
        }
        for p in pods.get("items", [])
    ]
    try:
        nps = json.loads(kubectl(["get", "networkpolicy", "-n", namespace, "-o", "json"]))
        facts["network_policies"] = sorted(n["metadata"]["name"] for n in nps.get("items", []))
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError):
        facts["network_policies"] = "unavailable"
    try:
        facts["top"] = kubectl(
            ["top", "pod", "-n", namespace, "-l", selector, "--no-headers"]
        ).strip()
    except (subprocess.SubprocessError, OSError):
        facts["top"] = "unavailable (metrics-server?)"
    return facts


def pod_changes(before: Any, after: Any) -> dict[str, Any] | None:
    """Restarts per pod by name, plus pods that disappeared or appeared.

    Summing restart counts across two pod sets misses a pod that was replaced
    (its successor starts at 0) and can go negative; comparing by name does
    not.
    """
    if before is None:
        return None
    if not after or "pods" not in after:
        return {"clean": False, "error": (after or {}).get("error", "no pod list after the run")}
    b = {p["name"]: p["restarts"] for p in before["pods"]}
    a = {p["name"]: p["restarts"] for p in after["pods"]}
    restarted = {n: a[n] - b[n] for n in sorted(b.keys() & a.keys()) if a[n] != b[n]}
    gone, new = sorted(b.keys() - a.keys()), sorted(a.keys() - b.keys())
    return {
        "restarted": restarted,
        "disappeared": gone,
        "appeared": new,
        "clean": not restarted and not gone and not new,
    }


def _kubectl(argv: Sequence[str]) -> str:
    return subprocess.run(  # noqa: S603 - fixed kubectl verbs, operator-given namespace
        ["kubectl", *argv],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout


def collect(
    send: Send,
    states: Sequence[str],
    *,
    levels: Sequence[int],
    requests: int,
    warmup: int,
    send_ledger: Send | None = None,
    kubectl: Kubectl | None = None,
    namespace: str | None = None,
    selector: str = "app=jef",
    scene_path: Path | None = None,
) -> dict[str, Any]:
    status, readiness, _ = send("GET", "/v1/readiness", None)
    before_calls = ledger_calls(send_ledger)
    kube_before = kube_facts(kubectl, namespace, selector) if kubectl and namespace else None
    load = run_load(send, states, levels, requests, warmup)
    after_calls = ledger_calls(send_ledger)
    # After minutes of load, a failed kubectl call must not lose the evidence.
    try:
        kube_after = kube_facts(kubectl, namespace, selector) if kubectl and namespace else None
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError, KeyError) as exc:
        kube_after = {"error": f"{type(exc).__name__}: {exc}"}

    if send_ledger is not None and (before_calls is None or after_calls is None):
        # Asked for, and could not be performed: that is a failure, not "clean".
        delivery = {
            "measured": False,
            "requested": True,
            "error": "the mock ledger could not be read before or after the run",
        }
    elif before_calls is None or after_calls is None:
        delivery = {
            "measured": False,
            "note": "review_required is a policy verdict, not a delivery; pass --mock-ledger-url "
            "or record the queue's own receipts",
        }
    else:
        delivery = {
            "measured": True,
            "ledger_calls_before": before_calls,
            "ledger_calls_after": after_calls,
        }
        delivery["side_effects"] = after_calls - before_calls
    restarts = pod_changes(kube_before, kube_after)
    try:
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        git = ""
    scene_sha = None
    if scene_path is not None and scene_path.is_file():
        scene_sha = "sha256:" + hashlib.sha256(scene_path.read_bytes()).hexdigest()
    # Delivery: clean when not asked for, or measured with no side effect.
    delivery_ok = not delivery.get("requested") and delivery.get("side_effects", 0) == 0
    # Platform: clean when not asked for, or every pod kept its name and count.
    platform_ok = restarts is None or bool(restarts["clean"])
    return {
        "evidence_class": "RUNTIME EVIDENCE — not an SLA; sample sizes are next to every number",
        "collected_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "provenance": {
            "git_sha": git or None,
            "scene_sha256": scene_sha,
            "client": {"platform": platform.platform(), "python": sys.version.split()[0]},
        },
        "readiness": {"status": status, "body": redact(readiness)},
        "load": load,
        "delivery": delivery,
        "platform": {"before": kube_before, "after": kube_after, "restarts_during_run": restarts},
        "ok": load["safety_ok"] and delivery_ok and platform_ok,
    }


def render(e: dict[str, Any]) -> str:
    lines = [
        "# incident-triage runtime evidence",
        "",
        f"**{e['evidence_class']}** · collected {e['collected_at']} · git `{e['provenance']['git_sha']}`",
        "",
        f"- readiness: HTTP {e['readiness']['status']}; automation_ready "
        f"{(e['readiness']['body'] or {}).get('automation_ready')}",
        f"- cold first request: {e['load']['cold_first_request']}",
        "",
        "| concurrency | ok / requests | 5xx | timeouts | rps | p50 | p95 | p99 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level, m in e["load"]["levels"].items():
        lat = m["latency"]
        lines.append(
            f"| {level} | {m['ok']}/{m['requests']} | {m['status_5xx']} | {m['timeouts_or_unreachable']} "
            f"| {m['throughput_rps']} | {lat.get('p50_ms')} | {lat.get('p95_ms')} | {lat.get('p99_ms')} |"
        )
    lines += [
        "",
        f"- safety: {e['load']['safety']} → {'ok' if e['load']['safety_ok'] else '**NOT OK**'}",
        f"- delivery: {e['delivery']}",
        f"- pods during run: {e['platform']['restarts_during_run']}",
        "",
        f"**Overall: {'ok' if e['ok'] else 'NOT OK'}**",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key-env", default="JEF_API_KEY")
    p.add_argument(
        "--pack", type=Path, default=Path("benchmark/incident-triage/canary/six-alerts-v1")
    )
    p.add_argument("--levels", default="1,2,4,8")
    p.add_argument("--requests", type=int, default=40, help="per concurrency level")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--mock-ledger-url")
    p.add_argument("--kube-namespace")
    p.add_argument("--kube-selector", default="app=jef")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    states = [
        json.loads(line)["state"]
        for line in (args.pack / "cases.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    send = http_sender(args.base_url, os.environ.get(args.api_key_env), args.timeout)
    ledger = http_sender(args.mock_ledger_url, None, args.timeout) if args.mock_ledger_url else None
    evidence = collect(
        send,
        states,
        levels=[int(x) for x in args.levels.split(",")],
        requests=args.requests,
        warmup=args.warmup,
        send_ledger=ledger,
        kubectl=_kubectl if args.kube_namespace else None,
        namespace=args.kube_namespace,
        selector=args.kube_selector,
        scene_path=Path(__file__).resolve().parents[1] / "scenes" / "incident-triage.zh-tw.yaml",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    (args.out / "summary.md").write_text(render(evidence), "utf-8")
    print(render(evidence))
    return 0 if evidence["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
