"""The benchmark runner's contract with a scorer.

`run_benchmark` used to take an `Engine`, which made "the number" and "JEF's
number" the same thing. P1.5d needs comparators over *identical* cases, so the
runner now depends on the `Scorer` protocol instead. These tests use a scripted
scorer: no backbone, no Hub, and the expected metrics are arithmetic rather
than whatever the model happens to do today.
"""

from __future__ import annotations

import numpy as np
from jef_core.calibration import Calibrator
from jef_train.bench import BenchCase, run_benchmark


def case(label: int, task: str = "t") -> BenchCase:
    return BenchCase(
        state="付款服務連續三天失敗",
        question={
            "type": "choice",
            "instructions": "誰處理？",
            "criteria": {"billing": None, "infra": None},
        },
        option_keys=["billing", "infra"],
        label=label,
        task=task,
    )


class ScriptedScorer:
    """Returns a queued distribution per call, in order."""

    name = "scripted"

    def __init__(self, rows: list[list[float]], calibrator: Calibrator | None = None) -> None:
        self._rows = list(rows)
        self._calibrator = calibrator or Calibrator()
        self.seen: list[BenchCase] = []

    @property
    def calibrator(self) -> Calibrator:
        return self._calibrator

    def probs(self, case: BenchCase) -> np.ndarray:
        self.seen.append(case)
        return np.array(self._rows.pop(0))


def test_scripted_scorer_satisfies_the_protocol() -> None:
    # Structural, so this is the check that a comparator needs nothing from
    # Engine -- no backbone, no head, no fitted calibrator.
    from jef_train.bench import Scorer

    scorer: Scorer = ScriptedScorer([[1.0, 0.0]])
    assert scorer.probs(case(0)).tolist() == [1.0, 0.0]


def test_accuracy_and_random_baseline_come_from_the_scorer() -> None:
    # Three right, one wrong, against a two-option task.
    scorer = ScriptedScorer([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.7, 0.3]])
    results = run_benchmark(scorer, [case(0), case(0), case(1), case(1)])

    entry = results["t"]
    assert entry["n"] == 4
    assert entry["n_options"] == 2
    assert entry["random_baseline"] == 0.5
    assert entry["accuracy"] == 0.75


def test_every_case_is_scored_exactly_once() -> None:
    scorer = ScriptedScorer([[1.0, 0.0]] * 3)
    run_benchmark(scorer, [case(0), case(0), case(0)])
    assert len(scorer.seen) == 3


def test_tasks_are_reported_separately() -> None:
    scorer = ScriptedScorer([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    results = run_benchmark(
        scorer,
        [case(0, "a"), case(0, "a"), case(0, "b"), case(0, "b")],
    )
    assert set(results) == {"a", "b"}
    assert results["a"]["accuracy"] == 1.0
    assert results["b"]["accuracy"] == 0.0


def test_an_unfitted_calibrator_reports_no_conformal_columns() -> None:
    """A baseline has nothing fitted, and the runner must say so rather than
    emit a coverage number that no calibration supports."""
    scorer = ScriptedScorer([[0.9, 0.1], [0.1, 0.9]])
    assert not scorer.calibrator.is_fitted()

    entry = run_benchmark(scorer, [case(0), case(1)])["t"]
    assert entry["ece_p_correct"] is None
    assert "conformal_coverage" not in entry
    assert "mean_set_size" not in entry


def test_probabilities_follow_option_key_order() -> None:
    # The runner indexes the vector positionally, so a scorer that returns
    # options in a different order would silently score the wrong class.
    scorer = ScriptedScorer([[0.2, 0.8]])
    entry = run_benchmark(scorer, [case(1)])["t"]
    assert entry["accuracy"] == 1.0
