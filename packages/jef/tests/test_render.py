"""Terminal rendering. Mostly about not lying and not misaligning CJK."""

from __future__ import annotations

import pytest
from jef_cli.render import Style, display_width, pad, render_answer, render_trace

PLAIN = Style(enabled=False)


def test_display_width_counts_columns_not_characters() -> None:
    assert display_width("soc") == 3
    # Two characters, four columns.
    assert display_width("資訊") == 4
    assert display_width("危急") == 4
    assert display_width("a資b") == 4


def test_pad_aligns_by_display_width() -> None:
    assert display_width(pad("資訊", 10)) == 10
    assert display_width(pad("soc", 10)) == 10
    assert display_width(pad("資安應變小組", 10)) == 10


def test_pad_truncates_on_a_column_boundary() -> None:
    out = pad("資安應變小組非常長", 8)
    assert display_width(out) == 8
    assert out.rstrip().endswith("…")


def test_percentage_column_aligns_across_mixed_scripts() -> None:
    """The reason display_width exists: a zh-TW label must not shift the column."""
    answer = {
        "type": "choice",
        "choice": "資安應變小組",
        "probabilities": {"soc": 0.225, "資安應變小組": 0.444, "infra": 0.331},
        "confidence": 0.167,
    }
    columns = {
        display_width(line[: line.rindex("%") + 1])
        for line in render_answer("q", answer, PLAIN).splitlines()
        if "%" in line
    }
    assert len(columns) == 1, f"percentages landed in different columns: {columns}"


def test_small_probabilities_stay_visible() -> None:
    """Rounding 3% to zero blocks would render it as indistinguishable from zero."""
    answer = {
        "type": "choice",
        "choice": "a",
        "probabilities": {"a": 0.97, "b": 0.03},
        "confidence": 0.94,
    }
    line = next(line for line in render_answer("q", answer, PLAIN).splitlines() if " b " in line)
    assert any(block in line for block in "▏▎▍▌▋▊▉█")


def test_unknown_reliability_is_never_a_number() -> None:
    answer = {"type": "noul", "noul": 0.6, "confidence": 0.2}
    out = render_answer("q", answer, PLAIN, p_correct=None)
    assert "P(correct) unknown" in out
    assert "P(correct) 0.000" not in out


def test_known_reliability_is_shown() -> None:
    answer = {"type": "noul", "noul": 0.6, "confidence": 0.2}
    assert "P(correct) 0.910" in render_answer("q", answer, PLAIN, p_correct=0.91)


def test_conformal_exclusions_are_shown_not_hidden() -> None:
    """A reader needs to see what was ruled out, not just what survived."""
    answer = {
        "type": "choice",
        "choice": "a",
        "probabilities": {"a": 0.8, "b": 0.15, "c": 0.05},
        "confidence": 0.7,
    }
    out = render_answer("q", answer, PLAIN, prediction_set=["a"])
    for key in ("a", "b", "c"):
        assert any(line.strip().startswith(key) for line in out.splitlines())
    assert "conformal set 1/3" in out


def test_style_disables_itself_without_a_tty() -> None:
    assert Style(enabled=False).red("x") == "x"
    assert "\033[" in Style(enabled=True).red("x")


def test_trace_marks_skipped_layers_as_never_asked() -> None:
    trace = {
        "scene": "t",
        "version": 1,
        "verdict": "decided",
        "action": "close",
        "human_review": False,
        "calibrated": True,
        "state_encodes": 1,
        "questions_asked": 1,
        "layers_skipped": ["L2", "L3"],
        "layers": [
            {
                "id": "L1",
                "outcome": "close",
                "questions": [
                    {
                        "id": "q",
                        "answer": {"type": "noul", "noul": 0.9, "confidence": 0.8},
                        "p_correct": 0.95,
                        "prediction_set": ["true"],
                    }
                ],
                "gates": [
                    {
                        "index": 0,
                        "condition": "q.noul > 0.5",
                        "fired": True,
                        "is_else": False,
                        "then": "close",
                        "because": "確信",
                    }
                ],
            }
        ],
    }
    out = render_trace(trace, PLAIN)
    assert "not evaluated" in out
    assert "L2" in out and "L3" in out
    assert "FIRED" in out


def test_trace_flags_an_uncalibrated_run() -> None:
    trace = {
        "scene": "t",
        "version": 1,
        "verdict": "fallthrough",
        "action": "escalate_human",
        "human_review": True,
        "calibrated": False,
        "state_encodes": 1,
        "questions_asked": 0,
        "layers_skipped": [],
        "layers": [],
    }
    out = render_trace(trace, PLAIN)
    assert "UNCALIBRATED" in out
    assert "needs human review" in out


@pytest.mark.parametrize("fraction", [0.0, 0.001, 0.5, 0.999, 1.0])
def test_bars_never_exceed_their_width(fraction: float) -> None:
    from jef_cli.render import _bar

    assert display_width(_bar(fraction, 20)) == 20
