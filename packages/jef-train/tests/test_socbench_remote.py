"""NEXT-V3-02/08: remote scorers that can be compared fairly."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from jef_train.socbench import cli
from jef_train.socbench.annotate import TASK_LABELS
from jef_train.socbench.baselines import JefScorer, case_text, verdict_question
from jef_train.socbench.records import write_jsonl
from jef_train.socbench.remote import (
    GatewayScorer,
    MissingLabelsError,
    NotADistributionError,
    RefusalError,
    ScorerConfigError,
    SystemOneScorer,
    load_scorer_config,
    strict_distribution,
)
from jef_train.socbench.report import evaluate
from jef_train.socbench.splits import make_lock

TASK = "soc.ndr.technique.v1"
LABELS = TASK_LABELS[TASK]
SECRET = "sk-test-do-not-leak-0123456789"


def case(n: int, family: str = "fam-a") -> dict:
    return {
        "case_id": f"socb-t-{n:06d}",
        "family_id": family,
        "source_dataset": "ctu-13",
        "source_hash": "sha256:" + "0" * 64,
        "telemetry_origin": "lab-reproduction",
        "language": "n/a",
        "task_id": TASK,
        "state": {"flow": f"beacon every 60s #{n}"},
        "evidence": [{"id": "e1", "field": "flow", "value": "beacon"}],
    }


def uniform(_: str, body: dict, __: dict) -> dict:
    labels = body["questions"]["verdict"]["criteria"]
    return {
        "answers": {
            "verdict": {"type": "choice", "probabilities": {k: 1 / len(labels) for k in labels}}
        }
    }


# --------------------------------------------------------------------------- #
# one question for every system
# --------------------------------------------------------------------------- #


class CapturingEngine:
    model_name = "fake"

    def __init__(self) -> None:
        self.questions: list[dict] = []

    def evaluate(self, state, questions):
        self.questions.append(questions["verdict"])

        from types import SimpleNamespace

        answer = SimpleNamespace(probabilities={lab: 1 / len(LABELS) for lab in LABELS})
        return SimpleNamespace(answers={"verdict": answer})


def test_jef_systemone_and_gateway_are_asked_the_same_question() -> None:
    eng = CapturingEngine()
    JefScorer(eng).predict(case(1))
    seen: dict[str, dict] = {}

    def capture(kind: str):
        def post(url: str, body: dict, headers: dict) -> dict:
            seen[kind] = body["questions"]["verdict"]
            return uniform(url, body, headers)

        return post

    SystemOneScorer("http://127.0.0.1:1", name="s1", model="m", post=capture("s1")).predict(case(1))
    GatewayScorer("http://127.0.0.1:1", name="gw", model="m", post=capture("gw")).predict(case(1))
    jef_q = {k: v for k, v in eng.questions[0].items() if k != "task_id"}
    assert jef_q == seen["s1"] == seen["gw"] == verdict_question(TASK)
    assert list(seen["gw"]["criteria"]) == list(LABELS), "options in TASK_LABELS order"


# --------------------------------------------------------------------------- #
# credentials: from the environment, never into a report
# --------------------------------------------------------------------------- #


def test_a_missing_credential_fails_before_any_case_is_sent() -> None:
    calls = []
    with pytest.raises(ScorerConfigError, match="LAYA_KEY"):
        SystemOneScorer(
            "http://127.0.0.1:1",
            name="laya",
            model="m",
            auth_env="LAYA_KEY",
            environ={},
            post=lambda *a: calls.append(a) or {},
        )
    assert calls == []


def test_the_secret_is_sent_but_never_recorded() -> None:
    sent: list[dict] = []

    def post(url: str, body: dict, headers: dict) -> dict:
        sent.append(headers)
        return uniform(url, body, headers)

    s = SystemOneScorer(
        "http://127.0.0.1:1",
        name="jev-1.14.0",
        model="jev-1.14.0",
        auth_env="TYPESAFE_API_KEY",
        environ={"TYPESAFE_API_KEY": SECRET},
        post=post,
        meta={"provider": "TypeSafe", "version": "1.14.0", "price_per_1k_cases_usd": 0.02},
    )
    cases = [case(n, f"fam-{n % 4}") for n in range(1, 13)]
    golds = [
        {
            "case_id": c["case_id"],
            "task_id": TASK,
            "label_origin": "flow-label",
            "gold": {"verdict": "c2", "confidence_of_gold": "medium"},
        }
        for c in cases
    ]
    lock = make_lock(cases, golds, seed="t", ratios={"test": 1.0})
    report = evaluate(cases, golds, lock, [s], split="test")
    assert sent[0]["Authorization"] == f"Bearer {SECRET}"
    dumped = json.dumps(report)
    assert SECRET not in dumped, "the credential must never reach the report"
    d = report["manifest"]["scorers"]["jev-1.14.0"]
    assert (
        d["auth"] == "env:TYPESAFE_API_KEY"
        and d["provider"] == "TypeSafe"
        and d["version"] == "1.14.0"
    )
    assert report["evidence_class"] == "SOURCE-LABEL RESEARCH"


# --------------------------------------------------------------------------- #
# a reply that is not a full distribution is a failure, with its reason
# --------------------------------------------------------------------------- #


def test_strict_distribution_rejects_what_the_old_scorer_smoothed() -> None:
    good = {lab: 1 / len(LABELS) for lab in LABELS}
    assert sum(strict_distribution(LABELS, good).values()) == pytest.approx(1.0)
    with pytest.raises(MissingLabelsError):
        strict_distribution(LABELS, {"c2": 1.0})
    with pytest.raises(NotADistributionError):
        strict_distribution(LABELS, {**good, "c2": -0.1})
    with pytest.raises(NotADistributionError):
        strict_distribution(LABELS, {lab: 0.5 for lab in LABELS})


def test_contract_failures_are_counted_by_reason() -> None:
    replies = iter(
        [
            {"answers": {"verdict": {"type": "choice", "choice": "c2"}}},  # no probabilities
            {"answers": {"verdict": {"type": "refusal"}}},
            {"answers": {"verdict": {"type": "choice", "probabilities": {"c2": 1.0}}}},
            None,  # a well-formed reply
        ]
    )

    def post(url: str, body: dict, headers: dict) -> dict:
        r = next(replies)
        return r if r is not None else uniform(url, body, headers)

    gw = GatewayScorer("http://127.0.0.1:1", name="gw", model="typesafe-ai/jev", post=post)
    cases = [case(n, f"fam-{n}") for n in range(1, 5)]
    golds = [
        {
            "case_id": c["case_id"],
            "task_id": TASK,
            "label_origin": "double-human-reviewed",
            "gold": {"verdict": "c2", "confidence_of_gold": "high"},
            "annotators": ["analyst-anon-01", "analyst-anon-02"],
            "adjudication_status": "agreed",
        }
        for c in cases
    ]
    lock = make_lock(cases, golds, seed="t", ratios={"test": 1.0})
    report = evaluate(cases, golds, lock, [gw], split="test")
    f = report["scorers"]["gw"]
    assert f["cases"] == 4 and f["answered"] == 1
    assert f["failures"]["by_error"] == {
        "NoDistributionError": 1,
        "RefusalError": 1,
        "MissingLabelsError": 1,
    }
    assert report["evidence_class"] == "TIER-A GOLD"


def test_gateway_wire_format() -> None:
    seen: list[tuple[str, dict, dict]] = []

    def post(url: str, body: dict, headers: dict) -> dict:
        seen.append((url, body, headers))
        return uniform(url, body, headers)

    GatewayScorer(
        "https://ai-gateway.example/v4/ai",
        name="gw",
        model="typesafe-ai/jev",
        auth_env="GW",
        environ={"GW": SECRET},
        post=post,
    ).predict(case(1))
    url, body, headers = seen[0]
    assert url == "https://ai-gateway.example/v4/ai/decision-model"
    assert headers["ai-decision-model-specification-version"] == "4"
    assert headers["ai-model-id"] == "typesafe-ai/jev"
    # Every scorer sees the same rendering of the case: state, then evidence.
    assert body["stateParts"] == [{"type": "text", "text": case_text(case(1))}]
    assert set(body) == {"stateParts", "questions"}


def test_retries_are_bounded_and_counted() -> None:
    attempts = {"n": 0}

    def flaky(url: str, body: dict, headers: dict) -> dict:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TimeoutError("first try times out")
        return uniform(url, body, headers)

    s = SystemOneScorer("http://127.0.0.1:1", name="s", model="m", retries=1, post=flaky)
    assert sum(s.predict(case(1)).values()) == pytest.approx(1.0)
    assert s.attempts == 2
    with pytest.raises(RefusalError):
        SystemOneScorer(
            "http://127.0.0.1:1",
            name="s2",
            model="m",
            retries=3,
            post=lambda *a: {"answers": {"verdict": {"type": "refusal"}}},
        ).predict(case(1))


# --------------------------------------------------------------------------- #
# config files
# --------------------------------------------------------------------------- #


def write_config(tmp_path: Path, entries: object, name: str = "scorers.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(entries))
    return p


def test_config_loads_named_pinned_scorers(tmp_path: Path) -> None:
    p = write_config(
        tmp_path,
        {
            "scorers": [
                {
                    "name": "laya-r1",
                    "type": "systemone",
                    "url": "http://127.0.0.1:9001",
                    "model": "laya",
                    "checkpoint_sha256": "sha256:" + "a" * 64,
                    "hardware": "16 vCPU",
                    "data_flow": "local",
                },
                {
                    "name": "jev-via-gateway",
                    "type": "gateway",
                    "url": "https://ai-gateway.vercel.sh/v4/ai",
                    "model": "typesafe-ai/jev",
                    "auth_env": "AI_GATEWAY_API_KEY",
                    "retries": 1,
                },
            ]
        },
    )
    scorers = load_scorer_config(p, environ={"AI_GATEWAY_API_KEY": SECRET})
    assert [s.name for s in scorers] == ["laya-r1", "jev-via-gateway"]
    assert isinstance(scorers[1], GatewayScorer) and scorers[1].egress == "external"
    assert scorers[0].describe()["checkpoint_sha256"].startswith("sha256:")


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        (
            [{"name": "a", "type": "systemone", "url": "http://127.0.0.1:1", "model": "m"}] * 2,
            "used twice",
        ),
        (
            [
                {
                    "name": "a",
                    "type": "systemone",
                    "url": "http://127.0.0.1:1",
                    "model": "m",
                    "api_key": "x",
                }
            ],
            "inline secret",
        ),
        (
            [
                {
                    "name": "a",
                    "type": "systemone",
                    "url": "http://127.0.0.1:1",
                    "model": "m",
                    "mdoel": "x",
                }
            ],
            "unknown keys",
        ),
        (
            [{"name": "a", "type": "grpc", "url": "http://127.0.0.1:1", "model": "m"}],
            "type must be",
        ),
        ([{"name": "a", "type": "systemone", "url": "http://127.0.0.1:1"}], "model is required"),
        ([], "non-empty"),
    ],
)
def test_bad_configs_are_refused(tmp_path: Path, entries: object, message: str) -> None:
    with pytest.raises(ScorerConfigError, match=message):
        load_scorer_config(write_config(tmp_path, entries), environ={})


# --------------------------------------------------------------------------- #
# CLI end to end, against a real loopback server
# --------------------------------------------------------------------------- #


class SystemOne(BaseHTTPRequestHandler):
    def log_message(self, *a) -> None:
        return

    def do_POST(self) -> None:  # http.server API
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        reply = uniform(self.path, body, {})
        raw = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def server_url():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), SystemOne)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()


def _files(tmp_path: Path) -> list[str]:
    cases = [case(n, f"fam-{n % 5}") for n in range(1, 16)]
    golds = [
        {
            "case_id": c["case_id"],
            "task_id": TASK,
            "label_origin": "flow-label",
            "gold": {"verdict": "c2" if i % 2 else "benign", "confidence_of_gold": "medium"},
        }
        for i, c in enumerate(cases)
    ]
    write_jsonl(tmp_path / "cases.jsonl", cases)
    write_jsonl(tmp_path / "gold.jsonl", golds)
    (tmp_path / "lock.json").write_text(
        json.dumps(make_lock(cases, golds, seed="t", ratios={"test": 1.0}))
    )
    return [
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


def test_cli_runs_named_scorers_and_labels_the_evidence(
    tmp_path: Path, server_url: str, capsys
) -> None:
    cfg = write_config(
        tmp_path,
        [
            {"name": "laya-local", "type": "systemone", "url": server_url, "model": "laya"},
            {"name": "llm-baseline", "type": "systemone", "url": server_url, "model": "llm-x"},
        ],
    )
    rc = cli.main([*_files(tmp_path), "--scorer", "rules", "--scorer-config", str(cfg)])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert out.startswith("[SOURCE-LABEL RESEARCH]")
    report = json.loads((tmp_path / "report.json").read_text())
    assert set(report["scorers"]) == {"rules", "laya-local", "llm-baseline"}, (
        "no name overwrote another"
    )
    assert report["manifest"]["families"] == 5


def test_cli_refuses_duplicate_names_and_unapproved_egress(tmp_path: Path, capsys) -> None:
    base = _files(tmp_path)
    dup = write_config(
        tmp_path,
        [{"name": "rules", "type": "systemone", "url": "http://127.0.0.1:1", "model": "m"}],
    )
    assert cli.main([*base, "--scorer", "rules", "--scorer-config", str(dup)]) == 2
    assert "unique" in capsys.readouterr().err
    ext = write_config(
        tmp_path,
        [
            {
                "name": "far",
                "type": "gateway",
                "url": "https://gw.example/v4/ai",
                "model": "typesafe-ai/jev",
            }
        ],
        name="ext.json",
    )
    assert cli.main([*base, "--scorer-config", str(ext)]) == 2
    assert "--allow-egress gw.example" in capsys.readouterr().err
    assert not (tmp_path / "report.json").exists()


def test_jef_is_scored_on_the_question_and_text_it_is_trained_on() -> None:
    """A head trained by export_corpus must be asked the same thing at evaluation."""
    from jef_train.socbench.corpus_export import samples_from_gold

    c = {**case(1), "family_id": "f1", "source_dataset": "unit", "language": "en"}
    gold = {
        "case_id": c["case_id"],
        "task_id": TASK,
        "label_origin": "two-reviewer",
        "gold": {"verdict": TASK_LABELS[TASK][0]},
    }
    [sample], _ = samples_from_gold([c], [gold], {c["case_id"]: "train"}, min_tier="E")
    seen: dict[str, Any] = {}

    class Engine:
        def evaluate(self, state: Any, questions: dict[str, Any]) -> Any:
            seen["state"], seen["q"] = state, questions["verdict"]
            probs = {lab: 1 / len(TASK_LABELS[TASK]) for lab in TASK_LABELS[TASK]}
            return SimpleNamespace(answers={"verdict": SimpleNamespace(probabilities=probs)})

    JefScorer(Engine()).predict(c)
    assert seen["state"] == sample.state
    assert seen["q"]["instructions"] == sample.instructions
    assert list(seen["q"]["criteria"]) == sample.option_keys
    assert list(seen["q"]["criteria"].values()) == sample.option_labels
