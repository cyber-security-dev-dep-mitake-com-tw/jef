"""Calibration must actually calibrate -- otherwise gate thresholds are theatre."""

from __future__ import annotations

import numpy as np
import pytest
from jef_core.calibration import (
    Calibrator,
    bucket_key,
    fit_conformal_quantile,
    fit_temperature,
)
from jef_core.mathx import expected_calibration_error, softmax


def _overconfident(n: int = 600, k: int = 4, seed: int = 0):
    """Logits that are right ~60% of the time but scream ~99% certainty."""
    rng = np.random.default_rng(seed)
    logits, labels = [], []
    for _ in range(n):
        y = int(rng.integers(k))
        lg = rng.normal(0.0, 0.3, size=k)
        # Put a huge margin on a label that is only sometimes the true one.
        claimed = y if rng.random() < 0.6 else int(rng.integers(k))
        lg[claimed] += 9.0
        logits.append(lg)
        labels.append(y)
    return logits, labels


def test_fit_temperature_softens_overconfidence() -> None:
    logits, labels = _overconfident()
    t = fit_temperature(logits, labels)
    assert t > 1.0, "an overconfident model needs temperature > 1"


def test_temperature_scaling_reduces_ece() -> None:
    logits, labels = _overconfident()
    t = fit_temperature(logits, labels)

    def ece_at(temp: float) -> float:
        probs = [softmax(lg, temperature=temp) for lg in logits]
        conf = np.array([p.max() for p in probs])
        hit = np.array([int(np.argmax(p)) == y for p, y in zip(probs, labels, strict=True)])
        return expected_calibration_error(conf, hit)

    assert ece_at(t) < ece_at(1.0), "calibration must improve ECE, not just move it"


def test_fit_temperature_on_empty_input_is_identity() -> None:
    assert fit_temperature([], []) == 1.0


def test_conformal_quantile_delivers_nominal_coverage() -> None:
    rng = np.random.default_rng(7)
    k, alpha = 5, 0.10
    probs, labels = [], []
    for _ in range(1000):
        y = int(rng.integers(k))
        lg = rng.normal(0.0, 1.0, size=k)
        lg[y] += 1.5
        probs.append(softmax(lg))
        labels.append(y)

    split = 500
    qhat = fit_conformal_quantile(probs[:split], labels[:split], alpha)
    covered = sum(
        1 for p, y in zip(probs[split:], labels[split:], strict=True) if p[y] >= 1.0 - qhat
    )
    coverage = covered / len(labels[split:])
    # Split conformal guarantees >= 1 - alpha marginally; allow finite-sample slack.
    assert coverage >= 1 - alpha - 0.04, f"coverage {coverage:.3f} below nominal {1 - alpha}"


def test_conformal_falls_back_to_trivial_set_when_undersampled() -> None:
    # With 3 points you cannot certify 99% coverage; returning 1.0 (always cover)
    # is the honest answer, not a confidently wrong small quantile.
    q = fit_conformal_quantile([softmax(np.array([1.0, 0.0]))] * 3, [0, 0, 0], alpha=0.01)
    assert q == 1.0


def test_calibrator_roundtrips_through_json(tmp_path) -> None:
    logits, labels = _overconfident(n=200)
    c = Calibrator(backbone="hashing", head="zeroshot")
    c.fit_bucket("choice", 4, logits, labels)
    path = tmp_path / "calib.json"
    c.save(path)
    back = Calibrator.load(path)
    assert back.temperature("choice", 4) == pytest.approx(c.temperature("choice", 4))
    assert back.backbone == "hashing"
    assert back.is_fitted()


def test_buckets_are_keyed_by_kind_and_option_count() -> None:
    c = Calibrator()
    logits, labels = _overconfident(n=120, k=4)
    c.fit_bucket("choice", 4, logits, labels)
    assert bucket_key("choice", 4) in c.buckets
    # An unfitted bucket must not borrow another bucket's temperature.
    assert c.temperature("choice", 3) == 1.0
    assert c.temperature("noul", 2) == 1.0


def test_unfitted_calibrator_is_identity() -> None:
    c = Calibrator()
    lg = np.array([2.0, 1.0, 0.0])
    assert np.allclose(c.apply(lg, "choice", 3), softmax(lg))
    assert not c.is_fitted()


def test_prediction_set_widens_when_the_model_cannot_separate() -> None:
    logits, labels = _overconfident(n=400)
    c = Calibrator()
    c.fit_bucket("choice", 4, logits, labels)
    flat = np.array([0.25, 0.25, 0.25, 0.25])
    peaked = np.array([0.97, 0.01, 0.01, 0.01])
    assert len(c.prediction_set(flat, "choice", 4)) >= len(c.prediction_set(peaked, "choice", 4))


def test_prediction_set_is_never_empty() -> None:
    logits, labels = _overconfident(n=400)
    c = Calibrator()
    c.fit_bucket("choice", 4, logits, labels)
    s = c.prediction_set(np.array([0.9, 0.05, 0.03, 0.02]), "choice", 4, alpha=0.01)
    assert len(s) >= 1


# --------------------------------------------------------------------------- #
# Confidence -> P(correct): the answer to "confidence is not correctness"
# --------------------------------------------------------------------------- #


def test_isotonic_output_is_non_decreasing() -> None:
    from jef_core.calibration import isotonic_fit

    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, 400)
    y = (rng.uniform(0, 1, 400) < x).astype(float)  # P(correct) rises with x
    points = isotonic_fit(list(x), list(y))
    ys = [p[1] for p in points]
    assert ys == sorted(ys), "isotonic regression must be monotone by construction"
    assert all(0.0 <= v <= 1.0 for v in ys)


def test_isotonic_on_empty_input() -> None:
    from jef_core.calibration import isotonic_fit

    assert isotonic_fit([], []) == []


def test_correctness_map_refuses_to_fit_on_too_little_data() -> None:
    from jef_core.calibration import fit_correctness_map

    # An isotonic curve from a handful of points memorises. Returning nothing
    # lets a gate react to "unavailable" instead of to a confidently wrong map.
    assert fit_correctness_map([0.5] * 10, [True] * 10) == []


def _linked_confidence(n: int = 1200, k: int = 3, seed: int = 11):
    """Logits whose peakedness genuinely tracks correctness."""
    rng = np.random.default_rng(seed)
    logits, labels = [], []
    for _ in range(n):
        y = int(rng.integers(k))
        margin = rng.uniform(0.0, 4.0)
        lg = rng.normal(0, 0.2, size=k)
        # A bigger margin means both a peakier distribution and a better chance
        # the peak is on the right answer.
        target = y if rng.random() < 0.5 + margin / 10 else int(rng.integers(k))
        lg[target] += margin
        logits.append(lg)
        labels.append(y)
    return logits, labels


def test_p_correct_is_better_calibrated_than_raw_confidence() -> None:
    from jef_core.mathx import confidence as conf_of

    logits, labels = _linked_confidence()
    split = 600
    c = Calibrator()
    c.fit_bucket("choice", 3, logits[:split], labels[:split])

    probs = [c.apply(lg, "choice", 3) for lg in logits[split:]]
    hits = np.array([int(np.argmax(p)) == y for p, y in zip(probs, labels[split:], strict=True)])
    raw = np.array([conf_of(p) for p in probs])
    mapped = np.array([c.p_correct(conf_of(p), "choice", 3) for p in probs], dtype=float)

    ece_raw = expected_calibration_error(raw, hits)
    ece_mapped = expected_calibration_error(mapped, hits)
    assert ece_mapped < ece_raw, (
        f"the correctness map must beat raw confidence: {ece_mapped:.4f} vs {ece_raw:.4f}"
    )


def test_p_correct_is_none_without_a_map() -> None:
    c = Calibrator()
    assert c.p_correct(0.9, "choice", 3) is None
    assert not c.has_correctness_map("choice", 3)


def test_p_correct_is_clamped_to_the_fitted_range() -> None:
    logits, labels = _linked_confidence()
    c = Calibrator()
    c.fit_bucket("choice", 3, logits, labels)
    assert c.has_correctness_map("choice", 3)
    for value in (-5.0, 0.0, 0.5, 1.0, 5.0):
        p = c.p_correct(value, "choice", 3)
        assert p is not None and 0.0 <= p <= 1.0


def test_correctness_map_survives_json_round_trip(tmp_path) -> None:
    logits, labels = _linked_confidence()
    c = Calibrator()
    c.fit_bucket("choice", 3, logits, labels)
    path = tmp_path / "c.json"
    c.save(path)
    back = Calibrator.load(path)
    assert back.p_correct(0.7, "choice", 3) == pytest.approx(c.p_correct(0.7, "choice", 3))
