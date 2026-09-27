"""Maths that the whole contract rests on -- and that the Go port must match."""

from __future__ import annotations

import math

import numpy as np
import pytest
from jef_core.mathx import (
    brier_score,
    confidence,
    expected_calibration_error,
    score_expectation,
    softmax,
)


def test_softmax_sums_to_one() -> None:
    p = softmax(np.array([1.0, 2.0, 3.0]))
    assert p.sum() == pytest.approx(1.0)
    assert np.all(p > 0)


def test_softmax_is_shift_invariant() -> None:
    a = softmax(np.array([1.0, 2.0, 3.0]))
    b = softmax(np.array([101.0, 102.0, 103.0]))
    assert np.allclose(a, b)


def test_softmax_survives_extreme_logits() -> None:
    p = softmax(np.array([1000.0, -1000.0]))
    assert math.isfinite(p.sum())
    assert p.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("t,expect_sharper", [(0.5, True), (2.0, False)])
def test_temperature_direction(t: float, expect_sharper: bool) -> None:
    base = softmax(np.array([2.0, 1.0, 0.0]))
    scaled = softmax(np.array([2.0, 1.0, 0.0]), temperature=t)
    # bool() matters: numpy's bool_ is not Python's bool under `is`.
    assert bool(scaled.max() > base.max()) is expect_sharper


def test_softmax_rejects_nonpositive_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        softmax(np.array([1.0, 2.0]), temperature=0.0)


def test_confidence_matches_jev_formula() -> None:
    # Jev: (n * peak - 1) / (n - 1). For n=3, peak=0.5 -> (1.5-1)/2 = 0.25.
    assert confidence([0.5, 0.3, 0.2]) == pytest.approx(0.25)


def test_confidence_bounds() -> None:
    assert confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0)
    assert confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_confidence_binary_reduces_to_abs_2p_minus_1() -> None:
    for p in (0.5, 0.7, 0.9, 1.0):
        assert confidence([1 - p, p]) == pytest.approx(abs(2 * p - 1))


def test_confidence_needs_two_options() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        confidence([1.0])


def test_score_lands_between_levels() -> None:
    # Mass split evenly between level 1 and level 2 -> score 1.5, not 1 or 2.
    assert score_expectation([0.0, 0.5, 0.5, 0.0]) == pytest.approx(1.5)


def test_score_endpoints() -> None:
    assert score_expectation([1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert score_expectation([0.0, 0.0, 1.0]) == pytest.approx(2.0)


def test_brier_score_perfect_and_worst() -> None:
    assert brier_score(np.array([1.0, 0.0]), 0) == pytest.approx(0.0)
    assert brier_score(np.array([0.0, 1.0]), 0) == pytest.approx(2.0)


def test_ece_zero_for_perfectly_calibrated() -> None:
    conf = np.array([0.5] * 100)
    correct = np.array([True] * 50 + [False] * 50)
    assert expected_calibration_error(conf, correct) == pytest.approx(0.0, abs=1e-9)


def test_ece_detects_overconfidence() -> None:
    conf = np.array([0.99] * 100)
    correct = np.array([True] * 50 + [False] * 50)
    assert expected_calibration_error(conf, correct) == pytest.approx(0.49, abs=1e-6)


def test_ece_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        expected_calibration_error(np.array([0.5]), np.array([True, False]))
