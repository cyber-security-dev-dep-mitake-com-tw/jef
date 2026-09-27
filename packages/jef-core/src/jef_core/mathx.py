"""Pure probability maths shared by every JEF backend.

Kept dependency-light (numpy only) so the contract semantics can be tested
without loading a model, and so the Go/ONNX port has an unambiguous reference.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "brier_score",
    "confidence",
    "expected_calibration_error",
    "score_expectation",
    "softmax",
]


def softmax(logits: NDArray[np.floating], temperature: float = 1.0) -> NDArray[np.float64]:
    """Numerically stable softmax with optional temperature scaling.

    ``temperature`` > 1 flattens the distribution, < 1 sharpens it. It is the
    single scalar fitted during calibration (see :mod:`jef_core.calibration`).
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - np.max(z)
    e = np.exp(z)
    total = e.sum()
    if total == 0 or not np.isfinite(total):
        # Degenerate input: fall back to a uniform distribution rather than NaN.
        return np.full(z.shape, 1.0 / z.size, dtype=np.float64)
    return e / total


def confidence(probabilities: NDArray[np.floating] | list[float]) -> float:
    """Jev's confidence statistic: ``(n * peak - 1) / (n - 1)``.

    This measures how *peaked* the distribution is, not how *correct* it is.
    A uniform distribution scores 0; a one-hot distribution scores 1. For n == 2
    it reduces to ``|2p - 1|``.

    Calibration (temperature + conformal) is what makes this number actionable
    as a gate threshold -- see :mod:`jef_core.calibration`.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    n = p.size
    if n < 2:
        raise ValueError("confidence requires at least 2 options")
    peak = float(np.max(p))
    return float(np.clip((n * peak - 1.0) / (n - 1.0), 0.0, 1.0))


def score_expectation(probabilities: NDArray[np.floating] | list[float]) -> float:
    """Continuous score as the expected level index: ``sum(i * p_i)``.

    Jev's ``score`` answers may fall *between* levels; taking the expectation
    over the ordered level distribution is what produces that behaviour. The
    result lies in ``[0, K-1]`` for K levels.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if p.size < 2:
        raise ValueError("score requires at least 2 ordered levels")
    idx = np.arange(p.size, dtype=np.float64)
    return float(np.dot(idx, p))


def brier_score(probabilities: NDArray[np.floating], correct_index: int) -> float:
    """Multiclass Brier score for a single prediction (lower is better)."""
    p = np.asarray(probabilities, dtype=np.float64)
    target = np.zeros_like(p)
    target[correct_index] = 1.0
    return float(np.sum((p - target) ** 2))


def expected_calibration_error(
    confidences: NDArray[np.floating],
    correct: NDArray[np.bool_],
    n_bins: int = 15,
) -> float:
    """Equal-width-binned ECE.

    This is the headline number the ecosystem refuses to publish. We publish it.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    hit = np.asarray(correct, dtype=bool)
    if conf.shape != hit.shape:
        raise ValueError("confidences and correct must have the same shape")
    if conf.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in pairwise(edges):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        weight = mask.mean()
        ece += weight * abs(hit[mask].mean() - conf[mask].mean())
    return float(ece)
