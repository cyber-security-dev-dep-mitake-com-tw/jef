"""NEXT-06 evaluation manifest and egress guard; NEXT-07 mock targets."""

from __future__ import annotations

import importlib.util
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from jef_train.socbench import cli
from jef_train.socbench.baselines import HttpScorer, RuleScorer, questions_digest
from jef_train.socbench.records import write_jsonl
from jef_train.socbench.report import evaluate, evaluation_manifest
from jef_train.socbench.splits import make_lock

ROOT = Path(__file__).resolve().parents[3]
TASK = "soc.alert.verdict.v1"


def case(n: int, family: str) -> dict:
    return {
        "case_id": f"socb-t-{n:06d}",
        "family_id": family,
        "source_dataset": "jef-analyst-reviewed",
        "source_hash": "sha256:" + "0" * 64,
        "telemetry_origin": "lab-reproduction",
        "language": "en",
        "task_id": TASK,
        "state": {"alert": f"alert {n}"},
        "evidence": [{"id": "e1", "field": "alert", "value": f"alert {n}"}],
    }


def gold(cid: str, verdict: str) -> dict:
    return {
        "case_id": cid,
        "task_id": TASK,
        "label_origin": "double-human-reviewed",
        "gold": {"verdict": verdict, "confidence_of_gold": "high"},
        "annotators": ["analyst-anon-01", "analyst-anon-02"],
        "adjudication_status": "agreed",
    }


def fixture() -> tuple[list[dict], list[dict], dict]:
    cases = [case(n, f"fam-{n // 2}") for n in range(1, 25)]
    verdicts = ["confirmed_malicious", "suspicious", "confirmed_benign", "unknown"]
    golds = [gold(c["case_id"], verdicts[i % 4]) for i, c in enumerate(cases)]
    lock = make_lock(cases, golds, seed="t", pinned={"fam-11": "calibration", "fam-12": "test"})
    return cases, golds, lock


# --------------------------------------------------------------------------- #
# NEXT-06
# --------------------------------------------------------------------------- #


def test_report_carries_a_manifest_that_pins_what_ran() -> None:
    cases, golds, lock = fixture()
    report = evaluate(cases, golds, lock, [RuleScorer()], split="test")
    m = report["manifest"]
    assert m["lock"]["cases_digest"] == lock["cases_digest"] and m["lock"]["split"] == "test"
    assert m["scorers"]["rules"] == {"type": "rules", "egress": "none"}
    assert m["egress"] == ["none"]
    assert m["python"] and m["platform"] and m["jef_core"]


def test_http_scorer_discloses_host_egress_and_prompt() -> None:
    local = HttpScorer("http://127.0.0.1:4141", post=lambda u, b: {"answers": {}})
    remote = HttpScorer("https://api.example.com", post=lambda u, b: {"answers": {}})
    assert local.describe()["egress"] == "loopback"
    assert (
        remote.describe()["egress"] == "external" and remote.describe()["host"] == "api.example.com"
    )
    # The digest covers every task's instructions, which change only with the
    # questions JEF heads are trained on.
    assert remote.describe()["prompt_digest"] == questions_digest()
    assert questions_digest().startswith("sha256:")
    m = evaluation_manifest([local, remote], {"cases_digest": "x", "gold_digest": "y"}, "test")
    assert m["egress"] == ["external", "loopback"]


def test_cli_refuses_external_scorer_unless_egress_is_allowed(tmp_path: Path, capsys) -> None:
    cases, golds, lock = fixture()
    write_jsonl(tmp_path / "cases.jsonl", cases)
    write_jsonl(tmp_path / "gold.jsonl", golds)
    (tmp_path / "lock.json").write_text(json.dumps(lock), "utf-8")
    base = [
        "evaluate",
        "--lock",
        str(tmp_path / "lock.json"),
        "--cases",
        str(tmp_path / "cases.jsonl"),
        "--gold",
        str(tmp_path / "gold.jsonl"),
        "--out",
        str(tmp_path / "report.json"),
    ]
    rc = cli.main([*base, "--scorer", "http=https://api.example.com"])
    assert rc == 2
    assert "--allow-egress api.example.com" in capsys.readouterr().err
    assert not (tmp_path / "report.json").exists(), "no case left before the refusal"

    rc = cli.main([*base, "--scorer", "rules"])
    assert rc == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["manifest"]["egress_allowed"] == []


# --------------------------------------------------------------------------- #
# NEXT-07: the mock targets behave like a platform with its own idempotency
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_url():
    spec = importlib.util.spec_from_file_location(
        "mock_targets", ROOT / "deploy" / "lab" / "mock_targets.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    mod.LEDGER.reset()
    server = ThreadingHTTPServer(("127.0.0.1", 0), mod.Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


def post(url: str, body: dict, headers: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(  # noqa: S310 - loopback test server
        url,
        data=json.dumps(body).encode(),
        method="POST",
        headers={"content-type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:  # noqa: S310
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def get(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:  # noqa: S310
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_mock_records_once_per_idempotency_key(mock_url: str) -> None:
    body = {"asset_id": "host-7", "idempotency_key": "case-1:isolate:1"}
    s1, r1 = post(f"{mock_url}/edr/isolate", body)
    s2, r2 = post(f"{mock_url}/edr/isolate", body)
    assert s1 == s2 == 200 and r1["ref"] == r2["ref"]
    assert r2["repeats"] == 1
    _, calls = get(f"{mock_url}/calls")
    assert calls["count"] == 1 and calls["calls"][0]["action"] == "isolate_host"


def test_mock_timeout_lands_but_does_not_answer_in_time(mock_url: str) -> None:
    body = {"asset_id": "host-7", "idempotency_key": "case-2:disable:1"}
    status, reply = post(
        f"{mock_url}/iam/disable", body, {"X-Mock-Behaviour": "timeout", "X-Mock-Delay": "0.2"}
    )
    assert status == 200 and reply["late"] is True
    status, found = get(f"{mock_url}/calls/case-2:disable:1")
    assert status == 200 and found["action"] == "disable_user", "reconciliation sees it landed"
    assert get(f"{mock_url}/calls/never-sent")[0] == 404


def test_mock_error_is_recorded_as_attempted(mock_url: str) -> None:
    status, reply = post(
        f"{mock_url}/fw/block",
        {"asset_id": "10.0.0.9", "idempotency_key": "k3"},
        {"X-Mock-Behaviour": "error"},
    )
    assert status == 500 and reply["ok"] is False
    assert get(f"{mock_url}/calls")[1]["count"] == 1
    assert post(f"{mock_url}/edr/wipe", {})[0] == 404, "no mock target means no action"
