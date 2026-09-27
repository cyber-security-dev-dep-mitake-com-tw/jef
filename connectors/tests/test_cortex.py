"""The Cortex analyzer and responder.

The taxonomy is what an analyst reads first, so these check the mapping from a
decision to a verdict chip -- and especially that "we cannot say how reliable
this is" never renders as a number.
"""

from __future__ import annotations

from typing import Any

import pytest


def _analyzer(module: Any, config: dict[str, Any], data: Any = "告警內容") -> Any:
    analyzer = module.JefAnalyzer.__new__(module.JefAnalyzer)
    analyzer._config = {"config.url": "http://jef", **config}
    analyzer._data = data
    analyzer.reported = None
    analyzer.service = analyzer._config.get("config.service", "scene")
    analyzer.base_url = "http://jef"
    analyzer.timeout = 60
    analyzer.api_key = None
    return analyzer


def _trace(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "scene": "incident-triage",
        "action": "assign",
        "human_review": False,
        "calibrated": True,
        "layers": [{"questions": [{"p_correct": 0.93}, {"p_correct": 0.88}]}],
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# Taxonomies
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "action,level",
    [
        ("close", "safe"),
        ("page_oncall", "malicious"),
        ("escalate_human", "suspicious"),
        ("assign", "info"),
        ("some_custom_action", "info"),
    ],
)
def test_action_maps_to_a_verdict_level(
    cortex_analyzer_module: Any, action: str, level: str
) -> None:
    analyzer = _analyzer(cortex_analyzer_module, {"config.service": "scene"})
    taxonomies = analyzer.summary(_trace(action=action))["taxonomies"]
    action_tax = next(t for t in taxonomies if t["predicate"] == "action")
    assert action_tax["level"] == level
    assert action_tax["value"] == action


def test_human_review_gets_its_own_taxonomy(cortex_analyzer_module: Any) -> None:
    analyzer = _analyzer(cortex_analyzer_module, {"config.service": "scene"})
    taxonomies = analyzer.summary(_trace(human_review=True))["taxonomies"]
    assert any(t["predicate"] == "review" and t["value"] == "human" for t in taxonomies)


def test_worst_reliability_is_reported_not_the_best(cortex_analyzer_module: Any) -> None:
    """A chain is as reliable as its least reliable link."""
    analyzer = _analyzer(cortex_analyzer_module, {"config.service": "scene"})
    taxonomies = analyzer.summary(_trace())["taxonomies"]
    p = next(t for t in taxonomies if t["predicate"] == "p_correct")
    assert p["value"] == "0.88"


def test_unknown_reliability_never_renders_as_a_number(cortex_analyzer_module: Any) -> None:
    """ "Unknown" and "low" are different and lead to different decisions."""
    analyzer = _analyzer(cortex_analyzer_module, {"config.service": "scene"})
    trace = _trace(layers=[{"questions": [{"p_correct": None}]}])
    taxonomies = analyzer.summary(trace)["taxonomies"]
    p = next(t for t in taxonomies if t["predicate"] == "p_correct")
    assert p["value"] == "unknown"


def test_absent_calibration_is_surfaced(cortex_analyzer_module: Any) -> None:
    # Without it, every case routing to a human looks like caution rather than
    # a server that cannot automate anything.
    analyzer = _analyzer(cortex_analyzer_module, {"config.service": "scene"})
    taxonomies = analyzer.summary(_trace(calibrated=False))["taxonomies"]
    assert any(t["predicate"] == "calibration" and t["value"] == "absent" for t in taxonomies)


def test_question_service_summarises_each_answer_type(cortex_analyzer_module: Any) -> None:
    analyzer = _analyzer(cortex_analyzer_module, {"config.service": "question"})
    cases = [
        ({"type": "choice", "choice": "soc", "confidence": 0.9}, "soc"),
        ({"type": "score", "score": 3.25, "confidence": 0.7}, "3.25"),
        ({"type": "noul", "noul": 0.81, "confidence": 0.62}, "0.81"),
    ]
    for answer, expected in cases:
        taxonomies = analyzer.summary({"answers": {"q": answer}})["taxonomies"]
        assert taxonomies[0]["value"] == expected
        assert any(t["predicate"] == "confidence" for t in taxonomies)


# --------------------------------------------------------------------------- #
# Question building
# --------------------------------------------------------------------------- #


def test_choice_question_preserves_option_order(cortex_analyzer_module: Any) -> None:
    analyzer = _analyzer(
        cortex_analyzer_module,
        {
            "config.service": "question",
            "config.question_type": "choice",
            "config.instructions": "誰處理？",
            "config.options": ["soc=監控", "appsec=程式碼", "infra=網路"],
        },
    )
    question = analyzer._build_question()
    assert list(question["criteria"]) == ["soc", "appsec", "infra"]


def test_choice_question_rejects_one_option(
    cortex_analyzer_module: Any, analyzer_exit: type[Exception]
) -> None:
    analyzer = _analyzer(
        cortex_analyzer_module,
        {
            "config.service": "question",
            "config.question_type": "choice",
            "config.instructions": "誰？",
            "config.options": ["only=one"],
        },
    )
    with pytest.raises(analyzer_exit, match="at least two"):
        analyzer._build_question()


def test_score_question_keeps_level_order(cortex_analyzer_module: Any) -> None:
    analyzer = _analyzer(
        cortex_analyzer_module,
        {
            "config.service": "question",
            "config.question_type": "score",
            "config.instructions": "嚴重度",
            "config.options": ["低", "中", "高"],
        },
    )
    assert analyzer._build_question()["criteria"] == ["低", "中", "高"]


def test_unknown_question_type_is_rejected(
    cortex_analyzer_module: Any, analyzer_exit: type[Exception]
) -> None:
    analyzer = _analyzer(
        cortex_analyzer_module,
        {
            "config.service": "question",
            "config.question_type": "freeform",
            "config.instructions": "寫一首詩",
        },
    )
    with pytest.raises(analyzer_exit, match="unknown question_type"):
        analyzer._build_question()


# --------------------------------------------------------------------------- #
# Responder
# --------------------------------------------------------------------------- #


def _responder(module: Any, data: dict[str, Any]) -> Any:
    responder = module.JefTriage.__new__(module.JefTriage)
    responder._config = {}
    responder._data = data
    responder.base_url = "http://jef"
    responder.scene = "incident-triage"
    responder.timeout = 60
    responder.api_key = None
    return responder


def test_responder_sends_case_structure_not_flattened_text(
    cortex_responder_module: Any,
) -> None:
    responder = _responder(
        cortex_responder_module,
        {
            "title": "勒索軟體疑似活動",
            "description": "db-core-02 出現大量檔案加密行為",
            "severity": 3,
            "tags": ["ransomware"],
            "observables": [{"dataType": "ip", "data": "198.51.100.23"}],
        },
    )
    state = responder._state()
    assert state["title"] == "勒索軟體疑似活動"
    # TheHive's own severity is evidence the scene may disagree with, not the
    # answer, so it is passed under its own key.
    assert state["thehive_severity"] == 3
    assert state["observables"] == [{"type": "ip", "value": "198.51.100.23"}]


def test_responder_never_sends_an_empty_state(cortex_responder_module: Any) -> None:
    state = _responder(cortex_responder_module, {"unexpected": "shape"})._state()
    assert state


def test_responder_caps_observables(cortex_responder_module: Any) -> None:
    data = {
        "title": "x",
        "observables": [{"dataType": "ip", "data": f"10.0.0.{i}"} for i in range(200)],
    }
    assert len(_responder(cortex_responder_module, data)._state()["observables"]) == 50
