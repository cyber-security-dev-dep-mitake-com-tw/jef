"""Wire-contract conformance: question shapes, both noul dialects, rejections."""

from __future__ import annotations

import pytest
from jef_core.errors import InvalidQuestionError
from jef_core.types import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    coerce_question,
    normalize,
)


def test_choice_normalizes_preserving_option_order() -> None:
    q = normalize(
        "team",
        {
            "type": "choice",
            "instructions": "誰處理？",
            "criteria": {"billing": "付款", "infra": "網路", "sales": None},
        },
    )
    assert q.kind == "choice"
    assert q.option_keys == ["billing", "infra", "sales"]
    assert q.option_labels == ["付款", "網路", None]
    assert q.n_options == 3


def test_score_uses_levels_as_both_keys_and_labels() -> None:
    q = normalize(
        "sev", {"type": "score", "instructions": "嚴重度", "criteria": ["低", "中", "高"]}
    )
    assert q.kind == "score"
    assert q.option_keys == ["低", "中", "高"]


@pytest.mark.parametrize("dialect", ["noul", "boolean"])
def test_both_noul_dialects_normalize_identically(dialect: str) -> None:
    q = normalize("x", {"type": dialect, "instructions": "是否緊急？"})
    assert q.kind == "noul"
    # Order is always (false, true) so index 1 is P(true) everywhere downstream.
    assert q.option_keys == ["false", "true"]
    assert q.dialect == dialect


def test_noul_criteria_uses_reserved_words_as_aliases() -> None:
    q = normalize(
        "x",
        {
            "type": "noul",
            "instructions": "退款了嗎？",
            "criteria": {"true": "已退款", "false": "未退款"},
        },
    )
    assert q.option_labels == ["未退款", "已退款"]


def test_choice_rejects_single_option() -> None:
    with pytest.raises(InvalidQuestionError):
        coerce_question("q", {"type": "choice", "instructions": "x", "criteria": {"only": "one"}})


def test_score_rejects_single_level() -> None:
    with pytest.raises(InvalidQuestionError):
        coerce_question("q", {"type": "score", "instructions": "x", "criteria": ["only"]})


def test_score_rejects_duplicate_levels() -> None:
    with pytest.raises(InvalidQuestionError):
        coerce_question("q", {"type": "score", "instructions": "x", "criteria": ["低", "低"]})


def test_unknown_type_is_rejected() -> None:
    with pytest.raises(InvalidQuestionError):
        coerce_question("q", {"type": "freeform", "instructions": "寫一首詩"})


def test_empty_instructions_rejected() -> None:
    with pytest.raises(InvalidQuestionError):
        coerce_question("q", {"type": "noul", "instructions": ""})


def test_unknown_field_rejected() -> None:
    # extra="forbid" keeps typos from silently becoming no-ops.
    with pytest.raises(InvalidQuestionError):
        coerce_question("q", {"type": "noul", "instructions": "x", "critera": {}})


def test_already_typed_questions_pass_through() -> None:
    for model in (
        ChoiceQuestion(type="choice", instructions="a", criteria={"x": None, "y": None}),
        ScoreQuestion(type="score", instructions="b", criteria=["l", "h"]),
        NoulQuestion(type="noul", instructions="c"),
    ):
        assert coerce_question("q", model) is model
