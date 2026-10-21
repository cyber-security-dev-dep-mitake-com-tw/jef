"""Remote systems answer the five scene questions under JEF's rules, or fail visibly.

No network: every reply comes from an in-process fake.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jef_scene import load_scene
from jef_train.socbench import cli, remote
from jef_train.socbench.remote import GatewayScorer, SystemOneScorer
from jef_train.socbench.triage import load_pack
from jef_train.socbench.triage_eval import run_questions, scene_questions, score_questions
from jef_train.socbench.triage_remote import EgressRefused, RemoteTriageEngine, egress_check

ROOT = Path(__file__).resolve().parents[3]
CANARY = ROOT / "benchmark" / "incident-triage" / "canary" / "six-alerts-v1"
DEV = ROOT / "benchmark" / "incident-triage" / "synthetic" / "dev-v1"
SCENE = load_scene(ROOT / "scenes" / "incident-triage.zh-tw.yaml")
OWNERS = ["soc", "appsec", "infra", "identity", "fraud"]
LEVELS = ["資訊", "低", "中", "高", "危急"]


def good_answers() -> dict[str, Any]:
    return {
        "is_false_positive": {"type": "noul", "noul": 0.2},
        "has_evidence": {"type": "boolean", "probability": 0.9},
        # Deliberately not in the scene's option order.
        "owner": {"type": "choice", "probabilities": dict(zip(reversed(OWNERS), [0.05, 0.05, 0.1, 0.2, 0.6], strict=True))},
        "severity": {"type": "score", "probabilities": dict(zip(LEVELS, [0.0, 0.1, 0.2, 0.5, 0.2], strict=True))},
        "urgent": {"type": "noul", "noul": 0.7},
    }  # fmt: skip


class Fake:
    def __init__(self, answers: dict[str, Any] | None = None, *, raise_: Exception | None = None):
        self.answers = answers if answers is not None else good_answers()
        self.raise_ = raise_
        self.calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def __call__(self, url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        self.calls.append((url, body, headers))
        if self.raise_:
            raise self.raise_
        asked = body["questions"]
        return {"answers": {q: a for q, a in self.answers.items() if q in asked}}


def engine(fake: Fake, **kw: Any) -> RemoteTriageEngine:
    scorer = SystemOneScorer("http://127.0.0.1:9", name="mock", model="m-1", post=fake, **kw)
    return RemoteTriageEngine(scorer)


def run(fake: Fake, n: int = 3) -> tuple[RemoteTriageEngine, list[dict[str, Any]]]:
    cases = load_pack(DEV).cases[:n]
    eng = engine(fake)
    return eng, run_questions(eng.for_case, cases, SCENE)


def test_a_full_reply_is_scored_like_jef_but_never_calibrated() -> None:
    eng, rows = run(Fake())
    q = rows[0]["questions"]
    assert {v["status"] for v in q.values()} == {"answered"}
    assert q["owner"]["prediction"] == "soc"  # 0.6, despite the reversed order
    assert q["severity"]["prediction"] == "高"
    assert q["is_false_positive"]["p_true"] == pytest.approx(0.2)
    assert {v["calibration_status"] for v in q.values()} == {"missing"}
    assert {v["p_correct"] for v in q.values()} == {None}
    assert eng.failures == []
    score_questions(rows, load_pack(DEV).expectations, scene=SCENE)


def test_the_request_carries_the_scene_questions_and_the_state() -> None:
    fake = Fake()
    run(fake, n=1)
    url, body, _ = fake.calls[0]
    assert url.endswith("/v1/systemone") and body["model"] == "m-1"
    assert body["state"] == load_pack(DEV).cases[0]["state"]
    sent = body["questions"]
    expected = scene_questions(SCENE)
    assert list(sent) == list(expected)
    assert list(sent["owner"]["criteria"]) == OWNERS
    assert sent["owner"]["instructions"] == expected["owner"].instructions


@pytest.mark.parametrize(
    ("qid", "bad", "kind"),
    [
        ("owner", {"type": "choice", "probabilities": {"soc": 0.7, "infra": 0.3}}, "MissingLabelsError"),
        ("owner", {"type": "choice", "choice": "soc"}, "NoDistributionError"),
        ("severity", {"type": "score", "probabilities": dict(zip(LEVELS, [0.5] * 5, strict=True))}, "NotADistributionError"),
        ("urgent", {"type": "noul", "noul": 1.4}, "NotADistributionError"),
        ("urgent", {"type": "refusal"}, "RefusalError"),
    ],
)  # fmt: skip
def test_a_bad_answer_fails_that_question_only(qid: str, bad: Any, kind: str) -> None:
    eng, rows = run(Fake({**good_answers(), qid: bad}), n=2)
    for row in rows:
        assert row["questions"][qid]["status"] == "missing_answer"
        assert all(v["status"] == "answered" for k, v in row["questions"].items() if k != qid)
    assert {f["kind"] for f in eng.failures} == {kind}
    assert len(eng.failures) == 2


def test_a_dead_endpoint_fails_every_question_and_is_counted() -> None:
    eng, rows = run(Fake(raise_=ConnectionError("refused")), n=2)
    assert all(v["status"] == "missing_answer" for r in rows for v in r["questions"].values())
    assert [f["kind"] for f in eng.failures] == ["TransportError", "TransportError"]


def test_gateway_dialect_sends_state_parts() -> None:
    fake = Fake()
    scorer = GatewayScorer("https://gw.example.invalid", name="gw", model="vendor/m", post=fake)
    run_questions(RemoteTriageEngine(scorer).for_case, load_pack(DEV).cases[:1], SCENE)
    url, body, headers = fake.calls[0]
    assert url.endswith("/decision-model") and body["stateParts"][0]["type"] == "text"
    assert headers["ai-model-id"] == "vendor/m"


def test_egress_needs_an_allowed_host_and_never_carries_private_cases() -> None:
    external = SystemOneScorer("https://api.example.invalid", name="ext", model="m", post=Fake())
    public = load_pack(DEV).cases
    with pytest.raises(EgressRefused, match=r"--allow-egress api\.example\.invalid"):
        egress_check(external, public, [])
    egress_check(external, public, ["api.example.invalid"])
    for origin in ("own-telemetry", "reviewer-sourced"):
        private = [{**public[0], "origin": origin}]
        with pytest.raises(EgressRefused, match=origin):
            egress_check(external, private, ["api.example.invalid"])
    loopback = SystemOneScorer("http://127.0.0.1:9", name="lo", model="m", post=Fake())
    egress_check(loopback, [{**public[0], "origin": "own-telemetry"}], [])


def test_cli_remote_run_reports_the_system_and_no_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sekrit-value-0123456789"
    monkeypatch.setenv("MOCK_SCORER_KEY", secret)
    fake = Fake({**good_answers(), "urgent": {"type": "refusal"}})
    monkeypatch.setattr(remote, "_default_post", lambda timeout: fake)
    config = tmp_path / "scorers.json"
    config.write_text(json.dumps({"scorers": [{
        "name": "mock-llm", "type": "systemone", "url": "http://127.0.0.1:9", "model": "llm-x",
        "auth_env": "MOCK_SCORER_KEY", "provider": "test", "version": "0",
    }]}))  # fmt: skip
    out = tmp_path / "out"
    rc = cli.main(["triage", "evaluate", "--mode", "questions", "--scorer", "remote",
                   "--scorer-config", str(config), "--scorer-name", "mock-llm",
                   "--pack", str(DEV), "--out", str(out)])  # fmt: skip
    assert rc == 0
    blob = "".join(p.read_text() for p in out.rglob("*") if p.is_file())
    assert secret not in blob
    assert fake.calls[0][2]["Authorization"] == f"Bearer {secret}"
    report = json.loads((out / "results.json").read_text())
    assert report["provenance"]["model"]["name"] == "remote"
    assert report["provenance"]["model"]["auth"] == "env:MOCK_SCORER_KEY"
    assert "CONTRACT SMOKE" not in report["evidence_class"]
    failures = report["metrics"]["remote_contract_failures"]
    assert failures["by_kind"] == {"RefusalError": len(load_pack(DEV).cases)}


def test_cli_refuses_an_external_scorer_without_permission(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "scorers.json"
    config.write_text(json.dumps([{"name": "ext", "type": "systemone",
                                   "url": "https://api.example.invalid", "model": "m"}]))  # fmt: skip
    rc = cli.main(["triage", "evaluate", "--mode", "questions", "--scorer", "remote",
                   "--scorer-config", str(config), "--scorer-name", "ext",
                   "--pack", str(CANARY), "--out", str(tmp_path / "o")])  # fmt: skip
    assert rc == 2 and "--allow-egress" in capsys.readouterr().err
