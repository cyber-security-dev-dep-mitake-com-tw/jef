"""``/v1/systemone`` conformance.

These are the tests that decide whether an existing TypeSafe or Vercel AI SDK
client can point at JEF without changing a line.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from jef_core import Engine
from jef_server.app import create_app
from jef_server.settings import Settings

TEAM = {
    "type": "choice",
    "instructions": "應由哪個團隊處理？",
    "criteria": {"billing": "付款、發票、退款", "infra": "基礎設施", "appsec": "應用程式漏洞"},
}
SEV = {"type": "score", "instructions": "嚴重度", "criteria": ["資訊", "低", "中", "高", "危急"]}


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(engine=Engine(), settings=Settings()))


def post(client: TestClient, body: dict) -> tuple[int, dict]:
    r = client.post("/v1/systemone", json=body)
    return r.status_code, r.json()


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_mixed_primitives_in_one_request(client: TestClient) -> None:
    status, body = post(client, {
        "state": "付款服務連續三天失敗",
        "questions": {
            "urgent": {"type": "noul", "instructions": "是否緊急？"},
            "team": TEAM,
            "sev": SEV,
        },
    })
    assert status == 200
    assert set(body["answers"]) == {"urgent", "team", "sev"}
    assert body["answers"]["team"]["type"] == "choice"
    assert body["answers"]["sev"]["type"] == "score"
    assert body["answers"]["urgent"]["type"] == "noul"
    assert body["model"]
    assert body["usage"]["outputTokens"] == 0


def test_answers_are_keyed_by_the_caller_s_ids(client: TestClient) -> None:
    _, body = post(client, {
        "state": "x",
        "questions": {"refund_requested": {"type": "noul", "instructions": "要退款嗎？"}},
    })
    assert "refund_requested" in body["answers"]


def test_choice_answer_fields(client: TestClient) -> None:
    _, body = post(client, {"state": "x", "questions": {"team": TEAM}})
    a = body["answers"]["team"]
    assert set(a) == {"type", "choice", "probabilities", "confidence"}
    assert set(a["probabilities"]) == set(TEAM["criteria"])
    assert sum(a["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)


def test_score_answer_carries_its_legend(client: TestClient) -> None:
    _, body = post(client, {"state": "x", "questions": {"sev": SEV}})
    a = body["answers"]["sev"]
    assert a["legend"] == SEV["criteria"]
    assert 0.0 <= a["score"] <= 4.0


# --------------------------------------------------------------------------- #
# The noul / boolean dialect split
# --------------------------------------------------------------------------- #


def test_noul_dialect_returns_noul_field(client: TestClient) -> None:
    _, body = post(client, {"state": "x", "questions": {"q": {"type": "noul", "instructions": "是否緊急？"}}})
    a = body["answers"]["q"]
    assert a["type"] == "noul"
    assert "noul" in a and "probability" not in a


def test_boolean_dialect_returns_probability_field(client: TestClient) -> None:
    _, body = post(client, {"state": "x", "questions": {"q": {"type": "boolean", "instructions": "是否緊急？"}}})
    a = body["answers"]["q"]
    assert a["type"] == "boolean"
    assert "probability" in a and "noul" not in a


def test_boolean_criteria_uses_true_false_keys(client: TestClient) -> None:
    status, body = post(client, {
        "state": "客服已確認退款完成",
        "questions": {"refunded": {
            "type": "boolean",
            "instructions": "是否已退款給客戶？",
            "criteria": {"true": "已確認退款", "false": "未退款或遭拒"},
        }},
    })
    assert status == 200
    assert 0.0 <= body["answers"]["refunded"]["probability"] <= 1.0


# --------------------------------------------------------------------------- #
# State shapes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("state", [
    "純文字",
    {"alert": "payout failed", "count": 3},
    ["事件一", "事件二", "事件三"],
])
def test_accepted_state_shapes(client: TestClient, state: object) -> None:
    status, body = post(client, {"state": state, "questions": {"q": {"type": "noul", "instructions": "緊急？"}}})
    assert status == 200
    # An array is one shared state, never a batch: still exactly one answer.
    assert len(body["answers"]) == 1


def test_numeric_state_is_rejected(client: TestClient) -> None:
    status, _ = post(client, {"state": 42, "questions": {"q": {"type": "noul", "instructions": "x"}}})
    assert status == 422


# --------------------------------------------------------------------------- #
# Rejections
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("questions", [
    {"q": {"type": "freeform", "instructions": "寫一首詩"}},
    {"q": {"type": "choice", "instructions": "x", "criteria": {"only": "one"}}},
    {"q": {"type": "score", "instructions": "x", "criteria": ["only"]}},
    {"q": {"type": "noul", "instructions": ""}},
    {},
])
def test_contract_violations_return_422(client: TestClient, questions: dict) -> None:
    status, body = post(client, {"state": "x", "questions": questions})
    assert status == 422
    assert "error" in body and "code" in body["error"]


def test_missing_state_is_rejected(client: TestClient) -> None:
    status, _ = post(client, {"questions": {"q": {"type": "noul", "instructions": "x"}}})
    assert status == 422


def test_too_many_questions_returns_413(client: TestClient) -> None:
    app = create_app(engine=Engine(), settings=Settings(max_questions=2))
    with TestClient(app) as c:
        r = c.post("/v1/systemone", json={
            "state": "x",
            "questions": {f"q{i}": {"type": "noul", "instructions": "x"} for i in range(3)},
        })
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "too_many_questions"


# --------------------------------------------------------------------------- #
# Discovery and observability
# --------------------------------------------------------------------------- #


def test_models_endpoint(client: TestClient) -> None:
    body = client.get("/v1/models").json()
    entry = body["data"][0]
    assert {"choice", "score", "noul", "boolean"} <= set(entry["question_types"])


def test_limits_endpoint(client: TestClient) -> None:
    body = client.get("/v1/limits").json()
    assert body["max_questions_per_request"] > 0


def test_healthz_flags_the_test_backbone(client: TestClient) -> None:
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    # A deployment must never quietly serve meaningless vectors.
    assert body["test_backbone"] is True
    assert body["calibrated"] is False


def _metric(text: str, name: str, labels: str = "") -> float:
    """Read one exported metric value, defaulting to 0 when not yet emitted."""
    for line in text.splitlines():
        if line.startswith(f"{name}{labels} "):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def test_metrics_prove_the_shared_state_guarantee(client: TestClient) -> None:
    """state_encodes / requests must stay at 1.0 regardless of question count.

    The counters are process-wide by design (that is what Prometheus counters
    are), so this measures the delta across one request rather than absolutes.
    """
    ok = '{endpoint="systemone",status="200"}'
    before = client.get("/metrics").text
    e0, r0 = _metric(before, "jef_state_encodes_total"), _metric(before, "jef_requests_total", ok)

    client.post(
        "/v1/systemone",
        json={
            "state": "付款失敗",
            "questions": {f"q{i}": {"type": "noul", "instructions": "緊急？"} for i in range(25)},
        },
    )

    after = client.get("/metrics").text
    encodes = _metric(after, "jef_state_encodes_total") - e0
    requests = _metric(after, "jef_requests_total", ok) - r0
    assert requests == 1.0
    assert encodes == 1.0, f"D3 regression: 25 questions caused {encodes} state encodes"


def test_question_kind_metric_folds_boolean_into_noul(client: TestClient) -> None:
    before = _metric(client.get("/metrics").text, "jef_questions_total", '{kind="noul"}')
    client.post("/v1/systemone", json={
        "state": "x",
        "questions": {
            "a": {"type": "noul", "instructions": "緊急？"},
            "b": {"type": "boolean", "instructions": "緊急？"},
        },
    })
    after = _metric(client.get("/metrics").text, "jef_questions_total", '{kind="noul"}')
    assert after - before == 2.0
