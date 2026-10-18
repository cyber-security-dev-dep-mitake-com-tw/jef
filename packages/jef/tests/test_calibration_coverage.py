"""coverage_check: `validated` describes the artifact; this measures the environment."""

from __future__ import annotations

import numpy as np
from jef_core.calibration import Calibrator


def fitted() -> Calibrator:
    cal = Calibrator()
    rng = np.random.default_rng(0)
    logits = [rng.normal(size=3) * 3 for _ in range(400)]
    labels = [int(np.argmax(z)) if i % 5 else int(rng.integers(3)) for i, z in enumerate(logits)]
    cal.fit_bucket("choice", 3, logits, labels, task_id="t")
    return cal


def probs(n: int, correct_share: float, seed: int = 1) -> tuple[list[np.ndarray], list[int]]:
    rng = np.random.default_rng(seed)
    ps, ys = [], []
    for _ in range(n):
        p = np.array([0.96, 0.02, 0.02])
        ps.append(p)
        ys.append(0 if rng.random() < correct_share else 1)
    return ps, ys


def test_coverage_holds_on_exchangeable_data_and_is_flagged_under_shift() -> None:
    cal = fitted()
    ok = cal.coverage_check(*probs(400, 0.97), "choice", 3, 0.10, "t")
    assert ok["status"] == "holds", ok
    shifted = cal.coverage_check(*probs(400, 0.60), "choice", 3, 0.10, "t")
    assert shifted["status"] == "violated", shifted
    assert shifted["wilson_95"][1] < 0.90


def test_too_few_and_unfitted_say_so() -> None:
    cal = fitted()
    assert cal.coverage_check(*probs(20, 0.5), "choice", 3, 0.10, "t")["status"] == "too_few"
    assert cal.coverage_check(*probs(200, 0.5), "choice", 4, 0.10, "t")["status"] == "unfitted"
    assert cal.coverage_check(*probs(200, 0.5), "choice", 3, 0.10, "other")["status"] == "unfitted"
