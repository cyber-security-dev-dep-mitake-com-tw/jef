"""Calibration must actually calibrate -- otherwise gate thresholds are theatre."""

from __future__ import annotations

import json

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
# M0-01: prediction_set fails closed. A missing quantile must never collapse
# to [argmax] -- a singleton reads as "safe to automate" to every set_size gate.
# --------------------------------------------------------------------------- #


def test_cal_001_missing_bucket_returns_full_set() -> None:
    """CAL-001: a fitted calibrator queried for a bucket it never saw excludes nothing."""
    logits, labels = _overconfident(n=400, k=4)
    c = Calibrator()
    c.fit_bucket("choice", 4, logits, labels)
    assert c.is_fitted()
    peaked = np.array([0.97, 0.01, 0.01, 0.01])
    # Same width, different kind.
    assert c.prediction_set(peaked, "score", 4) == [0, 1, 2, 3]
    # Same kind, different width.
    assert c.prediction_set(np.array([0.98, 0.01, 0.01]), "choice", 3) == [0, 1, 2]
    # Entirely unfitted calibrator.
    assert Calibrator().prediction_set(peaked, "choice", 4) == [0, 1, 2, 3]


def test_cal_002_missing_alpha_returns_full_set() -> None:
    """CAL-002: a bucket that exists but has no quantile for this alpha excludes nothing."""
    logits, labels = _overconfident(n=400, k=4)
    c = Calibrator()
    c.fit_bucket("choice", 4, logits, labels, alphas=(0.10,))
    peaked = np.array([0.97, 0.01, 0.01, 0.01])
    # The fitted alpha still produces a real conformal set...
    assert len(c.prediction_set(peaked, "choice", 4, alpha=0.10)) < 4
    # ...but an alpha nobody fitted must not borrow argmax.
    assert c.prediction_set(peaked, "choice", 4, alpha=0.05) == [0, 1, 2, 3]
    assert c.prediction_set(peaked, "choice", 4, alpha=0.01) == [0, 1, 2, 3]


def test_cal_003_empty_keep_set_returns_full_set() -> None:
    """CAL-003: when nothing clears the threshold the honest set is everything, not argmax."""
    from jef_core.calibration import Bucket

    # A tiny quantile demands p >= 0.95; a split distribution clears it nowhere.
    c = Calibrator(buckets={bucket_key("choice", 2): Bucket(conformal={"0.1000": 0.05})})
    assert c.prediction_set(np.array([0.6, 0.4]), "choice", 2, alpha=0.10) == [0, 1]
    # Sanity: a distribution that does clear it yields the singleton.
    assert c.prediction_set(np.array([0.99, 0.01]), "choice", 2, alpha=0.10) == [0]


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


# --------------------------------------------------------------------------- #
# Per-bucket calibration status: validated | insufficient | missing
# --------------------------------------------------------------------------- #
#
# is_fitted() answers "is anything fitted". A scene gate that asked it would
# automate a noul:2 question on the strength of a choice:5 bucket. status()
# answers for one bucket, from that bucket alone.


def test_cal_status_001_unfitted_is_missing() -> None:
    assert Calibrator().status("noul", 2) == "missing"


def test_cal_status_002_another_shape_never_lends_its_status() -> None:
    logits, labels = _overconfident(n=400, k=4)
    c = Calibrator()
    c.fit_bucket("choice", 4, logits, labels)
    assert c.status("choice", 4) == "validated"
    assert c.status("choice", 3) == "missing"
    assert c.status("noul", 2) == "missing"
    assert c.status("score", 4) == "missing"


def test_cal_status_003_too_few_samples_is_insufficient() -> None:
    """A bucket exists, but below MIN_CORRECTNESS_SAMPLES no correctness map is fitted."""
    logits, labels = _overconfident(n=120, k=4)
    c = Calibrator()
    c.fit_bucket("choice", 4, logits, labels)
    assert bucket_key("choice", 4) in c.buckets
    assert c.p_correct(0.9, "choice", 4) is None
    assert c.status("choice", 4) == "insufficient"


def test_cal_status_004_missing_alpha_is_insufficient() -> None:
    logits, labels = _overconfident(n=400, k=4)
    c = Calibrator()
    c.fit_bucket("choice", 4, logits, labels, alphas=(0.10,))
    assert c.status("choice", 4, alpha=0.10) == "validated"
    assert c.status("choice", 4, alpha=0.05) == "insufficient"


def test_cal_status_validated_implies_p_correct_available() -> None:
    logits, labels = _overconfident(n=400, k=4)
    c = Calibrator()
    c.fit_bucket("choice", 4, logits, labels)
    assert c.status("choice", 4) == "validated"
    assert c.p_correct(0.9, "choice", 4) is not None
    assert len(c.prediction_set(np.array([0.97, 0.01, 0.01, 0.01]), "choice", 4)) < 4


# --------------------------------------------------------------------------- #
# Task-scoped calibration (schema 2): identity is task + shape, never shape alone
# --------------------------------------------------------------------------- #


def test_bucket_key_round_trips_with_and_without_task() -> None:
    from jef_core.calibration import split_bucket_key

    assert bucket_key("noul", 2) == "noul:2"
    assert bucket_key("noul", 2, "cve.user_interaction.v1") == "cve.user_interaction.v1|noul:2"
    assert split_bucket_key("cve.user_interaction.v1|noul:2") == (
        "cve.user_interaction.v1",
        "noul",
        2,
    )
    assert split_bucket_key("noul:2") == (None, "noul", 2)


def test_pol_002_same_shape_other_task_never_borrows() -> None:
    """Task A is calibrated; task B is also noul:2. B gets nothing from A."""
    logits, labels = _overconfident(n=400, k=2)
    c = Calibrator()
    c.fit_bucket("noul", 2, logits, labels, task_id="cve.user_interaction.v1")
    assert c.status("noul", 2, task_id="cve.user_interaction.v1") == "validated"
    assert c.status("noul", 2, task_id="soar_zh.false_positive.v1") == "missing"
    assert c.p_correct(0.9, "noul", 2, "soar_zh.false_positive.v1") is None
    assert c.temperature("noul", 2, "soar_zh.false_positive.v1") == 1.0
    assert c.prediction_set(
        np.array([0.99, 0.01]), "noul", 2, 0.10, "soar_zh.false_positive.v1"
    ) == [0, 1]
    # And a question that names no task does not get the task's bucket either.
    assert c.status("noul", 2) == "missing"
    assert c.tasks() == ["cve.user_interaction.v1"]


def test_ml_001_schema_one_artifact_loads_as_legacy() -> None:
    """The shipped models/jef-v0/calibration.json shape: no schema field, shape keys."""
    logits, labels = _overconfident(n=400, k=4)
    fitted = Calibrator()
    fitted.fit_bucket("choice", 4, logits, labels)
    v1 = {k: v for k, v in fitted.to_dict().items() if k not in ("calibration_schema", "artifact")}
    for b in v1["buckets"].values():  # type: ignore[union-attr]
        b.pop("task_id")
        b.pop("metrics")

    legacy = Calibrator.from_dict(v1)
    assert legacy.legacy is True
    assert legacy.schema == 1
    assert legacy.is_fitted()
    # Numbers are still there for analysis...
    assert legacy.p_correct(0.9, "choice", 4) is not None
    # ...but the status never unlocks anything, with or without a task.
    assert legacy.status("choice", 4) == "legacy"
    assert legacy.status("choice", 4, task_id="anything.v1") == "legacy"
    assert legacy.status("noul", 2) == "legacy"


def test_the_shipped_artifact_is_legacy_and_says_so() -> None:
    from pathlib import Path

    root = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
    shipped = Calibrator.load(root / "models" / "jef-v0" / "calibration.json")
    assert shipped.legacy
    assert shipped.status("choice", 4) == "legacy"


def test_incompatible_model_digest_disables_everything() -> None:
    logits, labels = _overconfident(n=400, k=4)
    c = Calibrator()
    c.fit_bucket("choice", 4, logits, labels, task_id="t.v1")
    c.artifact.model_sha256 = "sha256:aaaa"
    c.bind_model("sha256:aaaa")
    assert not c.incompatible
    assert c.status("choice", 4, task_id="t.v1") == "validated"
    c.bind_model("sha256:bbbb")
    assert c.incompatible
    assert c.status("choice", 4, task_id="t.v1") == "incompatible"
    assert c.p_correct(0.9, "choice", 4, "t.v1") is None
    assert c.temperature("choice", 4, "t.v1") == 1.0
    assert c.prediction_set(np.array([0.97, 0.01, 0.01, 0.01]), "choice", 4, 0.10, "t.v1") == [
        0,
        1,
        2,
        3,
    ]
    # No digest recorded on either side: nothing to compare, nothing to refuse.
    c.artifact.model_sha256 = None
    assert not c.incompatible


def test_schema_two_round_trip_keeps_tasks_and_provenance(tmp_path) -> None:
    from jef_core.calibration import ArtifactInfo

    logits, labels = _overconfident(n=400, k=4)
    c = Calibrator(backbone="b", head="h")
    c.fit_bucket("choice", 4, logits, labels, task_id="cve.attack_vector.v1", metrics={"ece": 0.05})
    c.artifact = ArtifactInfo(
        model_sha256="sha256:abc",
        language="en,zh-TW",
        dataset_digest="sha256:def",
        sample_count=400,
    )
    path = tmp_path / "c.json"
    c.save(path)
    raw = json.loads(path.read_text())
    assert raw["calibration_schema"] == 2
    assert raw["artifact"]["model_sha256"] == "sha256:abc"
    back = Calibrator.load(path)
    assert not back.legacy
    assert back.tasks() == ["cve.attack_vector.v1"]
    assert back.buckets["cve.attack_vector.v1|choice:4"].task_id == "cve.attack_vector.v1"
    assert back.buckets["cve.attack_vector.v1|choice:4"].metrics == {"ece": 0.05}
    assert back.artifact.language == "en,zh-TW"
    assert back.status("choice", 4, task_id="cve.attack_vector.v1") == "validated"


def test_from_dict_refuses_the_impossible() -> None:
    with pytest.raises(ValueError, match="newer"):
        Calibrator.from_dict({"calibration_schema": 99, "buckets": {}})
    with pytest.raises(ValueError, match="task-scoped bucket"):
        Calibrator.from_dict({"buckets": {"t.v1|noul:2": {"temperature": 1.0}}})


def test_migrate_is_explicit_whole_and_non_destructive() -> None:
    logits, labels = _overconfident(n=400, k=4)
    fitted = Calibrator(backbone="b", head="h")
    fitted.fit_bucket("choice", 4, logits, labels)
    legacy = Calibrator.from_dict(
        {k: v for k, v in fitted.to_dict().items() if k not in ("calibration_schema", "artifact")}
    )
    assert legacy.legacy

    migrated = legacy.migrate("cve.severity.v1", model_sha256="sha256:abc")
    assert not migrated.legacy
    assert set(migrated.buckets) == {"cve.severity.v1|choice:4"}
    assert migrated.status("choice", 4, task_id="cve.severity.v1") == "validated"
    assert migrated.status("choice", 4) == "missing", "a migrated bucket answers only its task"
    assert migrated.artifact.model_sha256 == "sha256:abc"
    assert "Migrated" in migrated.notes
    # The original is untouched and still legacy.
    assert legacy.legacy and set(legacy.buckets) == {"choice:4"}
    with pytest.raises(ValueError, match="already carries task ids"):
        migrated.migrate("other.v1")
    with pytest.raises(ValueError, match="task_id"):
        legacy.migrate("")


def test_question_task_ids_are_validated_and_normalised() -> None:
    from jef_core.types import InvalidQuestionError, normalize

    q = normalize("q", {"type": "noul", "instructions": "x", "task_id": "cve.user_interaction.v1"})
    assert q.task_id == "cve.user_interaction.v1"
    assert normalize("q", {"type": "noul", "instructions": "x"}).task_id is None
    for bad in ("CVE.v1", "cve", "cve.user interaction.v1", "cve.user_interaction", "1cve.v1"):
        with pytest.raises(InvalidQuestionError):
            normalize("q", {"type": "noul", "instructions": "x", "task_id": bad})
