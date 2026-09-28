"""The MCP server.

The caller is an agent, and an agent reads only what the tool returns and what
its description says. So these check the two things that actually protect it:
that the descriptions explain what `confidence` is not, and that an untrustworthy
server says so in the payload rather than in a log the agent never sees.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

import pytest

mcp = pytest.importorskip("mcp", reason="needs the mcp extra")

from jef_cli.mcp_server import build_server  # noqa: E402

SCENES = "scenes"


def make_args(**overrides: Any) -> argparse.Namespace:
    base = {
        "local": True,
        "url": "",
        "backbone": "hashing",
        "head": None,
        "calibration": None,
        "scenes": SCENES,
        "threads": None,
        "timeout": 60.0,
        "dialect": "noul",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture(scope="module")
def server() -> Any:
    return build_server(make_args())


@pytest.fixture(scope="module")
def tools(server: Any) -> dict[str, Any]:
    listed = asyncio.run(server.list_tools())
    return {tool.name: tool for tool in listed}


def call(server: Any, name: str, **arguments: Any) -> dict[str, Any]:
    """Invoke a tool the way a client would, returning its structured result."""
    result = asyncio.run(server.call_tool(name, arguments))
    assert not result.is_error, f"{name} failed: {result.content}"
    assert result.structured_content is not None, f"{name} returned no structured content"
    return dict(result.structured_content)


def refusal(server: Any, name: str, **arguments: Any) -> dict[str, Any]:
    """Invoke a tool that should refuse, returning the refusal the agent sees.

    Refusals come back as structured payloads rather than exceptions: MCP masks
    exception messages, so a raised ValueError reaches the agent as "Error
    executing tool jef_choice" with nothing to correct.
    """
    result = asyncio.run(server.call_tool(name, arguments))
    assert not result.is_error, f"{name} raised instead of refusing: {result.content}"
    payload = dict(result.structured_content or {})
    assert "error" in payload, f"{name} returned no refusal: {payload}"
    return payload


# --------------------------------------------------------------------------- #
# Surface
# --------------------------------------------------------------------------- #


def test_every_expected_tool_is_registered(tools: dict[str, Any]) -> None:
    assert {
        "jef_choice",
        "jef_score",
        "jef_yes_no",
        "jef_ask_many",
        "jef_run_scene",
        "jef_list_scenes",
        "jef_health",
    } <= set(tools)


def test_descriptions_explain_what_confidence_is_not(tools: dict[str, Any]) -> None:
    """An agent seeing `confidence: 0.94` will read it as 94% correct."""
    description = tools["jef_choice"].description or ""
    assert "sharpness" in description or "peaked" in description
    assert "p_correct" in description


def test_null_p_correct_is_explained_as_unknown(server: Any) -> None:
    """Null must not read as zero: they lead to different decisions."""
    instructions = (server.instructions or "").lower()
    assert "null means unknown" in instructions


def test_instructions_state_the_knowledge_boundary(server: Any) -> None:
    instructions = (server.instructions or "").lower()
    assert "does not know things" in instructions


def test_ask_many_advertises_that_extra_questions_are_cheap(tools: dict[str, Any]) -> None:
    description = tools["jef_ask_many"].description or ""
    assert "read once" in description or "parallel" in description


def test_scene_tool_points_at_human_review(tools: dict[str, Any]) -> None:
    assert "human_review" in (tools["jef_run_scene"].description or "")


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #


def test_choice_returns_a_distribution_over_the_given_options(server: Any) -> None:
    result = call(
        server,
        "jef_choice",
        state="付款服務連續三天失敗",
        question="誰處理？",
        options={"soc": "監控", "infra": "網路"},
    )
    answer = result["answer"]
    assert answer["type"] == "choice"
    assert set(answer["probabilities"]) == {"soc", "infra"}
    assert answer["choice"] in answer["probabilities"]


def test_score_returns_a_continuous_score(server: Any) -> None:
    result = call(server, "jef_score", state="x", question="嚴重度", levels=["低", "中", "高"])
    assert result["answer"]["type"] == "score"
    assert 0.0 <= result["answer"]["score"] <= 2.0


def test_yes_no_returns_a_probability(server: Any) -> None:
    result = call(server, "jef_yes_no", state="x", question="是否緊急？")
    answer = result["answer"]
    assert 0.0 <= answer.get("noul", answer.get("probability", -1)) <= 1.0


def test_ask_many_answers_every_question(server: Any) -> None:
    result = call(
        server,
        "jef_ask_many",
        state="付款失敗",
        questions={
            "urgent": {"type": "noul", "instructions": "是否緊急？"},
            "team": {
                "type": "choice",
                "instructions": "誰處理？",
                "criteria": {"soc": "監控", "infra": "網路"},
            },
        },
    )
    assert set(result["answers"]) == {"urgent", "team"}


def test_an_untrustworthy_server_says_so_in_the_payload(server: Any) -> None:
    """Not in a log. The agent never reads the server's logs."""
    result = call(server, "jef_yes_no", state="x", question="是否緊急？")
    caveats = " ".join(result.get("caveats", []))
    assert "no meaning" in caveats, "the test backbone must be declared to the caller"
    assert "p_correct is unavailable" in caveats


def test_health_reports_both_trust_flags(server: Any) -> None:
    result = call(server, "jef_health")
    assert result["test_backbone"] is True
    assert result["calibrated"] is False
    assert result.get("caveats")


def test_scenes_are_listable(server: Any) -> None:
    result = call(server, "jef_list_scenes")
    ids = {entry["id"] for entry in result["scenes"]}
    assert "vuln-triage" in ids


def test_running_a_scene_returns_the_trace(server: Any) -> None:
    result = call(server, "jef_run_scene", scene="vuln-triage", state="一段漏洞描述" * 20)
    assert result["state_encodes"] == 1
    assert "layers" in result
    assert result["action"] is not None


def test_a_single_option_choice_is_refused_with_a_usable_reason(server: Any) -> None:
    """The agent must learn what to fix, not just that something failed."""
    payload = refusal(server, "jef_choice", state="x", question="誰？", options={"only": "one"})
    assert "at least two" in payload["error"]
    assert payload["hint"]


def test_an_empty_question_set_is_refused(server: Any) -> None:
    payload = refusal(server, "jef_ask_many", state="x", questions={})
    assert "not be empty" in payload["error"]
    assert payload["hint"]


def test_duplicate_score_levels_are_refused(server: Any) -> None:
    payload = refusal(server, "jef_score", state="x", question="嚴重度", levels=["低", "低"])
    assert "distinct" in payload["error"]
