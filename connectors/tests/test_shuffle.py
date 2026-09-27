"""The Shuffle app's request shaping.

What can silently go wrong here is not the HTTP call -- it is option order, how
a state gets typed, and whether an error reaches the workflow as something it
can branch on. Those are what these test.
"""

from __future__ import annotations

import json
from typing import Any

import pytest


@pytest.fixture
def app(shuffle_module: Any) -> Any:
    return shuffle_module.JEF()


@pytest.fixture
def captured(app: Any, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Intercept outgoing requests instead of making them."""
    calls: list[dict[str, Any]] = []

    def fake(url: str, apikey: str, path: str, payload: dict[str, Any] | None) -> str:
        calls.append({"url": url, "apikey": apikey, "path": path, "payload": payload})
        return json.dumps({"ok": True}, ensure_ascii=False)

    monkeypatch.setattr(app, "_request", fake)
    return calls


# --------------------------------------------------------------------------- #
# State typing
# --------------------------------------------------------------------------- #


def test_plain_text_state_stays_text(app: Any) -> None:
    assert app._parse_state("付款服務連續三天失敗") == "付款服務連續三天失敗"


def test_json_object_state_is_sent_as_structure(app: Any) -> None:
    """A model reading {"host": x, "severity": 4} sees the fields a human sees."""
    assert app._parse_state('{"host": "db-01", "severity": 4}') == {
        "host": "db-01",
        "severity": 4,
    }


def test_json_array_state_is_sent_as_structure(app: Any) -> None:
    assert app._parse_state('["甲", "乙"]') == ["甲", "乙"]


def test_text_that_merely_starts_with_a_brace_is_not_mangled(app: Any) -> None:
    text = "{這不是 JSON，是一則以大括號開頭的告警文字"
    assert app._parse_state(text) == text


# --------------------------------------------------------------------------- #
# Question shaping
# --------------------------------------------------------------------------- #


def test_choice_preserves_option_order(app: Any, captured: list[dict[str, Any]]) -> None:
    app.ask_choice(
        "http://jef",
        "",
        "告警",
        "誰處理？",
        "soc=一般資安監控事件\nappsec=應用程式漏洞\ninfra=基礎設施",
    )
    criteria = captured[0]["payload"]["questions"]["answer"]["criteria"]
    # Order is the option index in the returned distribution.
    assert list(criteria) == ["soc", "appsec", "infra"]
    assert criteria["soc"] == "一般資安監控事件"


def test_choice_option_without_description_becomes_null(
    app: Any, captured: list[dict[str, Any]]
) -> None:
    app.ask_choice("http://jef", "", "x", "誰處理？", "soc\ninfra=網路")
    criteria = captured[0]["payload"]["questions"]["answer"]["criteria"]
    assert criteria["soc"] is None
    assert criteria["infra"] == "網路"


def test_choice_rejects_a_single_option(app: Any) -> None:
    result = json.loads(app.ask_choice("http://jef", "", "x", "誰？", "only=one"))
    assert result["success"] is False
    assert "at least two" in result["error"]


def test_score_keeps_levels_in_the_order_given(app: Any, captured: list[dict[str, Any]]) -> None:
    app.ask_score("http://jef", "", "x", "嚴重度", "資訊\n低\n中\n高\n危急")
    criteria = captured[0]["payload"]["questions"]["answer"]["criteria"]
    # The answer is the expectation over this ordering; reordering inverts it.
    assert criteria == ["資訊", "低", "中", "高", "危急"]


def test_score_rejects_duplicate_levels(app: Any) -> None:
    result = json.loads(app.ask_score("http://jef", "", "x", "嚴重度", "低\n低"))
    assert result["success"] is False
    assert "distinct" in result["error"]


def test_score_rejects_a_single_level(app: Any) -> None:
    result = json.loads(app.ask_score("http://jef", "", "x", "嚴重度", "低"))
    assert result["success"] is False


def test_yes_no_omits_empty_criteria(app: Any, captured: list[dict[str, Any]]) -> None:
    app.ask_yes_no("http://jef", "", "x", "是否緊急？")
    assert "criteria" not in captured[0]["payload"]["questions"]["answer"]


def test_yes_no_carries_descriptions_when_given(app: Any, captured: list[dict[str, Any]]) -> None:
    app.ask_yes_no("http://jef", "", "x", "是否緊急？", if_true="正在造成損害")
    assert captured[0]["payload"]["questions"]["answer"]["criteria"] == {"true": "正在造成損害"}


def test_ask_many_passes_the_questions_through(app: Any, captured: list[dict[str, Any]]) -> None:
    questions = {
        "urgent": {"type": "noul", "instructions": "緊急？"},
        "team": {"type": "choice", "instructions": "誰？", "criteria": {"a": "甲", "b": "乙"}},
    }
    app.ask_many("http://jef", "", "告警", json.dumps(questions, ensure_ascii=False))
    assert captured[0]["payload"]["questions"] == questions


def test_ask_many_rejects_malformed_json(app: Any) -> None:
    result = json.loads(app.ask_many("http://jef", "", "x", "{not json"))
    assert result["success"] is False
    assert "JSON object" in result["error"]


def test_ask_many_rejects_a_list(app: Any) -> None:
    result = json.loads(app.ask_many("http://jef", "", "x", "[]"))
    assert result["success"] is False


# --------------------------------------------------------------------------- #
# Routing and errors
# --------------------------------------------------------------------------- #


def test_run_scene_targets_the_scene_endpoint(app: Any, captured: list[dict[str, Any]]) -> None:
    app.run_scene("http://jef", "", "incident-triage", "告警")
    assert captured[0]["path"] == "/v1/scenes/incident-triage:evaluate"
    assert captured[0]["payload"] == {"state": "告警"}


def test_health_is_a_get(app: Any, captured: list[dict[str, Any]]) -> None:
    app.health("http://jef", "")
    assert captured[0]["path"] == "/healthz"
    assert captured[0]["payload"] is None


def test_missing_url_fails_without_making_a_request(app: Any) -> None:
    result = json.loads(app._request("", "", "/healthz", None))
    assert result["success"] is False
    assert "no JEF url" in result["error"]
