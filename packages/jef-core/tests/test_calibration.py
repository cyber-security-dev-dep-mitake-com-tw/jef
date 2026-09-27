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
