"""Engine semantics -- above all, the shared-state guarantee (D3).

If ``encode_count`` ever exceeds 1 for a single request, JEF has silently become
"call an encoder N times", the per-question marginal cost claim is false, and a
20-gate scene costs 20x what it should. These tests exist to make that
regression impossible to merge.
"""

from __future__ import annotations

import json

import pytest
from jef_core import Engine
from jef_core.engine import render_state
from jef_core.errors import InvalidStateError
from jef_core.types import ChoiceAnswer, NoulAnswer, ScoreAnswer

TEAM = {
    "type": "choice",
    "instructions": "應由哪個團隊處理？",
    "criteria": {"billing": "付款、發票、退款", "infra": "基礎設施", "appsec": "應用程式漏洞"},
}
SEV = {"type": "score", "instructions": "嚴重度", "criteria": ["資訊", "低", "中", "高", "危急"]}
URGENT = {"type": "noul", "instructions": "是否緊急？"}


@pytest.fixture
def engine() -> Engine:
    return Engine()


# --------------------------------------------------------------------------- #
# D3: the state is encoded exactly once
# --------------------------------------------------------------------------- #


def test_single_question_encodes_state_once(engine: Engine) -> None:
    engine.evaluate("付款失敗", {"a": URGENT})
    assert engine.encode_count == 1


def test_twenty_questions_still_encode_state_once(engine: Engine) -> None:
    questions = {f"q{i}": dict(URGENT) for i in range(20)}
    engine.evaluate("付款失敗", questions)
    assert engine.encode_count == 1, "D3 violated: state re-encoded per question"


def test_prepare_then_many_answer_calls_encode_once(engine: Engine) -> None:
    # This is the path a multi-layer scene takes: prepare once, answer per layer.
    shared = engine.prepare("付款失敗")
    for _ in range(5):
        engine.answer(shared, {"a": URGENT, "b": TEAM})
    assert engine.encode_count == 1


# --------------------------------------------------------------------------- #
# Independence: questions cannot contaminate each other
# --------------------------------------------------------------------------- #


def test_answer_is_unchanged_by_unrelated_sibling_questions(engine: Engine) -> None:
    alone = engine.evaluate("付款失敗", {"team": TEAM}).answers["team"]
    crowded = engine.evaluate(
        "付款失敗",
        {"team": TEAM, "sev": SEV, "urgent": URGENT, "x": dict(URGENT)},
    ).answers["team"]
    assert alone.model_dump() == crowded.model_dump(), "questions must be evaluated in isolation"


def test_question_order_does_not_matter(engine: Engine) -> None:
    a = engine.evaluate("x", {"team": TEAM, "sev": SEV}).answers["sev"]
    b = engine.evaluate("x", {"sev": SEV, "team": TEAM}).answers["sev"]
    assert a.model_dump() == b.model_dump()


def test_evaluation_is_deterministic(engine: Engine) -> None:
    a = engine.evaluate("同樣的輸入", {"team": TEAM}).answers["team"]
    b = engine.evaluate("同樣的輸入", {"team": TEAM}).answers["team"]
    assert a.model_dump() == b.model_dump()


# --------------------------------------------------------------------------- #
# Answer shapes
# --------------------------------------------------------------------------- #


def test_choice_answer_shape(engine: Engine) -> None:
    a = engine.evaluate("x", {"team": TEAM}).answers["team"]
    assert isinstance(a, ChoiceAnswer)
    assert set(a.probabilities) == set(TEAM["criteria"])
    assert a.choice in a.probabilities
    assert sum(a.probabilities.values()) == pytest.approx(1.0, abs=1e-4)
    # The reported choice must be the argmax of the reported distribution.
    assert a.choice == max(a.probabilities, key=lambda k: a.probabilities[k])
    assert 0.0 <= a.confidence <= 1.0


def test_score_answer_shape_and_range(engine: Engine) -> None:
    a = engine.evaluate("x", {"sev": SEV}).answers["sev"]
    assert isinstance(a, ScoreAnswer)
    assert a.legend == SEV["criteria"]
    assert 0.0 <= a.score <= len(SEV["criteria"]) - 1
    assert sum(a.probabilities.values()) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize(
    "dialect,present,absent",
    [("noul", "noul", "probability"), ("boolean", "probability", "noul")],
)
def test_noul_dialect_is_echoed_back(
    engine: Engine, dialect: str, present: str, absent: str
) -> None:
    a = engine.evaluate("x", {"q": {"type": dialect, "instructions": "是否緊急？"}}).answers["q"]
    assert isinstance(a, NoulAnswer)
    d = a.model_dump(exclude_none=True)
    assert d["type"] == dialect
    assert present in d and absent not in d
    assert 0.0 <= a.value <= 1.0


def test_both_dialects_give_the_same_number(engine: Engine) -> None:
    r = engine.evaluate(
        "x",
        {
            "n": {"type": "noul", "instructions": "是否緊急？"},
            "b": {"type": "boolean", "instructions": "是否緊急？"},
        },
    )
    assert r.answers["n"].value == r.answers["b"].value  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# State shapes and usage
# --------------------------------------------------------------------------- #


def test_object_and_array_states_are_accepted(engine: Engine) -> None:
    for state in ({"alert": "payout failed", "count": 3}, ["a", "b", "c"]):
        r = engine.evaluate(state, {"q": URGENT})
        assert r.answers["q"] is not None


def test_array_state_is_one_state_not_a_batch(engine: Engine) -> None:
    # TypeSafe is explicit that an array is one shared state. One array of three
    # items must produce one answer, never three.
    r = engine.evaluate(["事件一", "事件二", "事件三"], {"q": URGENT})
    assert len(r.answers) == 1


def test_render_state_preserves_non_ascii(engine: Engine) -> None:
    assert "緊急" in render_state({"msg": "緊急"})
    assert json.loads(render_state({"a": 1})) == {"a": 1}


def test_invalid_state_rejected(engine: Engine) -> None:
    with pytest.raises(InvalidStateError):
        engine.evaluate(42, {"q": URGENT})


def test_output_tokens_are_structurally_zero(engine: Engine) -> None:
    # A System One model generates nothing. This is not an optimisation.
    r = engine.evaluate("付款失敗", {"q": URGENT, "t": TEAM})
    assert r.usage.outputTokens == 0
    assert r.usage.inputTokens > 0
    assert r.usage.totalTokens == r.usage.inputTokens


def test_uncalibrated_engine_warns(engine: Engine) -> None:
    r = engine.evaluate("x", {"q": URGENT})
    assert r.warnings is not None
    assert any("calibrat" in w for w in r.warnings)


def test_option_encodes_are_batched_across_questions(engine: Engine) -> None:
    """Twenty questions cost one option encode, not twenty.

    encode_count alone does not catch this: a shared state encoding followed by
    a per-question option encode reports one state read while costing linearly.
    Against a real backbone that was the difference between eleven questions
    costing 189% more than one and costing 46% more.
    """
    engine.evaluate("付款失敗", {f"q{i}": dict(URGENT) for i in range(20)})
    assert engine.encode_count == 1
    assert engine.query_encode_count == 1


def test_a_scene_style_run_batches_per_layer_not_per_question(engine: Engine) -> None:
    shared = engine.prepare("付款失敗")
    for _ in range(3):
        engine.answer(shared, {"a": URGENT, "b": TEAM, "c": SEV})
    assert engine.encode_count == 1
    # One batched option encode per layer, not one per question.
    assert engine.query_encode_count == 3


def test_batched_and_single_question_answers_are_identical(engine: Engine) -> None:
    """Batching must be a performance change, never an answer change.

    Two things could break it: a wrong span offset when slicing the batched
    query vectors, and a backbone that is not padding-invariant. The first is
    the likely regression and this catches it; the second was verified against
    mmBERT, where option texts of very different lengths batched together give
    bit-identical answers.
    """
    from jef_core.types import normalize

    questions = {"urgent": URGENT, "team": TEAM, "sev": SEV}
    shared = engine.prepare("付款服務連續三天失敗")

    batched = engine.answer(shared, questions)
    one_at_a_time = {
        qid: engine.answer_one(shared, normalize(qid, q)) for qid, q in questions.items()
    }
    for qid in questions:
        assert batched[qid].model_dump() == one_at_a_time[qid].model_dump()


def test_batching_slices_in_question_order(engine: Engine) -> None:
    """A span offset error would silently give each question another's options."""
    questions = {"a": TEAM, "b": SEV, "c": URGENT}
    answers = engine.evaluate("x", questions).answers
    assert set(answers["a"].probabilities) == set(TEAM["criteria"])  # type: ignore[union-attr]
    assert answers["b"].legend == SEV["criteria"]  # type: ignore[union-attr]
    assert answers["c"].type == "noul"


def test_marginal_question_cost_is_sublinear(engine: Engine) -> None:
    """One extra question must not re-read the state."""
    engine.evaluate("x" * 5000, {"q": URGENT})
    one_state_encode = engine.encode_count
    engine.reset_counters()
    engine.evaluate("x" * 5000, {f"q{i}": dict(URGENT) for i in range(50)})
    assert engine.encode_count == one_state_encode == 1
