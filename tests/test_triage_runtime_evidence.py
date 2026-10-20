"""scripts/triage-runtime-evidence.py against the real app, in process.

The operator runs it against the K8S lab; here the same code drives the
FastAPI app through TestClient, with kubectl and the mock ledger faked.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jef_core import Engine
from jef_scene import SceneEngine, SceneRegistry, load_scene
from jef_server.app import create_app
from jef_server.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "benchmark" / "incident-triage" / "canary" / "six-alerts-v1"


def _load():
    spec = importlib.util.spec_from_file_location(
        "triage_runtime_evidence", ROOT / "scripts" / "triage-runtime-evidence.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rte = _load()


@pytest.fixture(scope="module")
def send():
    engine = Engine()
    scene_engine = SceneEngine(engine)
    registry = SceneRegistry()
    registry.add(
        load_scene(ROOT / "scenes" / "incident-triage.zh-tw.yaml"), compile_with=scene_engine
    )
    client = TestClient(
        create_app(engine=engine, settings=Settings(), scenes=(scene_engine, registry))
    )

    def send(method, path, body):  # type: ignore[no-untyped-def]
        r = client.request(method, path, json=body)
        return (
            r.status_code,
            r.json(),
            r.elapsed.total_seconds() if hasattr(r, "elapsed") else 0.001,
        )

    return send


STATES = [
    json.loads(line)["state"] for line in (PACK / "cases.jsonl").read_text("utf-8").splitlines()
]


def test_collect_against_the_app(send) -> None:
    e = rte.collect(
        send,
        STATES,
        levels=[1, 2],
        requests=6,
        warmup=1,
        scene_path=ROOT / "scenes" / "incident-triage.zh-tw.yaml",
    )
    assert e["load"]["safety_ok"] is True
    assert e["load"]["levels"]["2"]["ok"] == 6
    lat = e["load"]["levels"]["1"]["latency"]
    assert isinstance(lat["p50_ms"], float)
    assert lat["p95_ms"] == "n<20: not reported", "no percentile without the samples for it"
    assert e["delivery"]["measured"] is False
    assert "not a delivery" in e["delivery"]["note"]
    assert e["readiness"]["status"] == 200
    assert e["ok"] is True
    assert "execution_status:not_dispatched" in e["load"]["safety"]
    assert "**Overall: ok**" in rte.render(e)


def test_a_moving_ledger_is_a_side_effect(send) -> None:
    calls = {"n": 0}

    def ledger(method, path, body):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return 200, [{}] * (calls["n"] - 1), 0.001

    e = rte.collect(send, STATES, levels=[1], requests=2, warmup=0, send_ledger=ledger)
    assert e["delivery"]["side_effects"] == 1
    assert e["ok"] is False


def test_kube_facts_and_restarts(send) -> None:
    restarts = iter([0, 1])

    def kubectl(argv):  # type: ignore[no-untyped-def]
        if argv[:2] == ["get", "pods"]:
            return json.dumps(
                {
                    "items": [
                        {
                            "metadata": {"name": "jef-0"},
                            "spec": {"nodeName": "w1", "serviceAccountName": "jef"},
                            "status": {
                                "containerStatuses": [
                                    {
                                        "imageID": "ghcr.io/x/jef@sha256:abc",
                                        "restartCount": next(restarts),
                                    }
                                ]
                            },
                        }
                    ]
                }
            )
        if argv[:2] == ["get", "networkpolicy"]:
            return json.dumps({"items": [{"metadata": {"name": "jef-ingress"}}]})
        return "jef-0   250m   826Mi"

    e = rte.collect(
        send, STATES, levels=[1], requests=2, warmup=0, kubectl=kubectl, namespace="jef"
    )
    before = e["platform"]["before"]
    assert before["pods"][0]["images"] == ["ghcr.io/x/jef@sha256:abc"]
    assert before["network_policies"] == ["jef-ingress"]
    assert e["platform"]["restarts_during_run"]["restarted"] == {"jef-0": 1}
    assert e["ok"] is False, "a restart during the run is not a clean run"


def test_redaction_masks_addresses_and_hosts() -> None:
    out = rte.redact({"a": ["10.1.2.3 via jef.lab.internal"], "b": 3})
    assert out == {"a": ["<ip> via <host>"], "b": 3}


def test_percentiles_need_their_samples() -> None:
    p = rte.percentiles([0.1] * 25)
    assert p["p95_ms"] == 100.0 and p["p99_ms"] == "n<100: not reported"


def test_no_successful_response_is_not_ok() -> None:
    def unauthorised(method, path, body):  # type: ignore[no-untyped-def]
        return 401, {"error": "auth"}, 0.001

    e = rte.collect(unauthorised, STATES, levels=[1], requests=3, warmup=0)
    assert e["load"]["answered"] is False
    assert e["ok"] is False


def test_a_requested_ledger_that_cannot_be_read_fails(send) -> None:
    def broken(method, path, body):  # type: ignore[no-untyped-def]
        return 500, None, 0.001

    e = rte.collect(send, STATES, levels=[1], requests=2, warmup=0, send_ledger=broken)
    assert e["delivery"]["requested"] is True and e["delivery"]["measured"] is False
    assert e["ok"] is False


def test_a_replaced_pod_is_caught() -> None:
    before = {"pods": [{"name": "jef-a", "restarts": 0}]}
    after = {"pods": [{"name": "jef-b", "restarts": 0}]}
    out = rte.pod_changes(before, after)
    assert out == {"restarted": {}, "disappeared": ["jef-a"], "appeared": ["jef-b"], "clean": False}
    assert rte.pod_changes(before, {"error": "timeout"})["clean"] is False
    assert rte.pod_changes(None, None) is None


def test_nearest_rank_percentile() -> None:
    samples = [i / 1000 for i in range(1, 21)]  # 1..20 ms
    assert rte.percentiles(samples)["p95_ms"] == 19.0, "ceil(0.95*20)=19th value, not the max"
