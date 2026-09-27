"""The gate expression sandbox.

Gate conditions arrive as YAML, and YAML in a SOAR deployment arrives from
wherever playbooks arrive from. These tests are the security boundary: anything
outside the whitelist must fail when the scene is *loaded*, not when an incident
is being triaged, and certainly not by executing.
"""

from __future__ import annotations

import pytest
from jef_scene.expr import (
    ExpressionError,
    compile_condition,
    evaluate_condition,
    validate_condition,
)

NAMES = {"a", "b"}


class _View:
    def __init__(self, **kw: object) -> None:
        self.__dict__.update(kw)


# --------------------------------------------------------------------------- #
# What must be rejected
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os').system('id')",
        "open('/etc/passwd').read()",
        "a.__class__.__mro__",
        "a.__init__.__globals__",
        "(lambda: 1)()",
        "[x for x in range(10)]",
        "a.confidence.__class__",
        "exec('x=1')",
        "eval('1')",
        "print(1)",
        "globals()",
        "getattr(a, 'confidence')",
        "a.probabilities.keys()",
        "f'{a}'",
        "(c := 1)",
        "a if b else 0",
    ],
)
def test_dangerous_expressions_are_rejected(source: str) -> None:
    with pytest.raises(ExpressionError):
        validate_condition(source, NAMES)


def test_unknown_names_are_rejected_with_a_useful_message() -> None:
    with pytest.raises(ExpressionError, match="unknown name 'zzz'"):
        validate_condition("zzz.confidence > 0.5", NAMES)


def test_undeclared_attributes_are_rejected() -> None:
    with pytest.raises(ExpressionError, match="not readable"):
        validate_condition("a.secret > 0.5", NAMES)


def test_syntax_errors_are_rejected_at_load_time() -> None:
    with pytest.raises(ExpressionError, match="cannot parse"):
        validate_condition("a.confidence >", NAMES)


def test_rejection_happens_before_any_execution(tmp_path) -> None:
    """Validation must not run the expression to decide whether to allow it."""
    marker = tmp_path / "written"
    source = f"open({str(marker)!r}, 'w')"
    with pytest.raises(ExpressionError):
        validate_condition(source, NAMES)
    assert not marker.exists()


# --------------------------------------------------------------------------- #
# What must work
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "source,expected",
    [
        ("a.confidence > 0.5", True),
        ("a.confidence < 0.5", False),
        ("a.confidence >= 0.9 and b.score >= 3", True),
        ("a.confidence > 0.99 or b.score >= 3", True),
        ("not a.confidence > 0.99", True),
        ("a.p_correct is not None and a.p_correct > 0.8", True),
        ("a.probabilities['soc'] > 0.4", True),
        ("a.choice == 'soc'", True),
        ("a.choice in ['soc', 'infra']", True),
        ("b.score - 1 >= 2", True),
        ("a.set_size == 1", True),
        ("'soc' in a.prediction_set", True),
    ],
)
def test_supported_conditions(source: str, expected: bool) -> None:
    a = _View(
        confidence=0.9,
        p_correct=0.95,
        choice="soc",
        probabilities={"soc": 0.7, "infra": 0.3},
        prediction_set=["soc"],
        set_size=1,
    )
    b = _View(score=3.2, confidence=0.6)
    code = compile_condition(source, NAMES)
    assert evaluate_condition(code, {"a": a, "b": b}) is expected


def test_p_correct_none_short_circuits_safely() -> None:
    """`is not None` must guard, not crash, when no correctness map exists."""
    a = _View(p_correct=None, confidence=0.99)
    code = compile_condition("a.p_correct is not None and a.p_correct >= 0.9", NAMES)
    assert evaluate_condition(code, {"a": a, "b": _View()}) is False


def test_comparing_none_directly_raises_rather_than_silently_passing() -> None:
    """An unguarded comparison against None is an author error, surfaced loudly.

    Returning False instead would turn "we cannot tell you how reliable this is"
    into "this is unreliable", which are different and lead to different gates.
    """
    a = _View(p_correct=None)
    code = compile_condition("a.p_correct >= 0.9", NAMES)
    with pytest.raises(ExpressionError, match="runtime"):
        evaluate_condition(code, {"a": a, "b": _View()})


def test_builtins_are_absent_at_runtime() -> None:
    code = compile_condition("a.confidence > 0.5", NAMES)
    ns: dict[str, object] = {"a": _View(confidence=0.9), "b": _View()}
    assert evaluate_condition(code, ns) is True
    # The namespace handed in must not be mutated with __builtins__.
    assert "__builtins__" not in ns
