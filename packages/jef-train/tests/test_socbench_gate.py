"""The close gate: fitted on calibration only, proven clean there, frozen for test."""

from __future__ import annotations

from jef_train.socbench.gate import clopper_pearson_upper, fit_close_gate


def test_clopper_pearson_rule_of_three() -> None:
    # Zero failures in n trials: the 95% upper bound is about 3/n.
    assert abs(clopper_pearson_upper(0, 300) - 0.00994) < 0.0005
    assert clopper_pearson_upper(0, 0) == 1.0
    assert clopper_pearson_upper(5, 5) == 1.0


def test_the_gate_picks_the_lowest_threshold_with_a_clean_enough_bound() -> None:
    # 400 confident benign (score 0.99), 10 attacks at 0.95, 200 benign at 0.95.
    scores = [0.99] * 400 + [0.95] * 210
    attack = [False] * 400 + [True] * 10 + [False] * 200
    gate = fit_close_gate(scores, attack, max_attack_rate=0.01)
    assert gate.threshold == 0.99, "lowering to 0.95 would close attacks"
    assert gate.calibration["attacks"] == 0 and gate.calibration["closed"] == 400


def test_too_little_clean_data_means_the_gate_closes_nothing() -> None:
    gate = fit_close_gate([0.99] * 100, [False] * 100, max_attack_rate=0.01)
    assert gate.threshold is None  # 0 of 100: the bound (~3%) is above 1%
    result = gate.evaluate([0.999] * 5, [True] * 5, [False] * 5)
    assert result["closed"] == 0 and result["abstained"] == 5


def test_an_attack_in_the_conformal_set_forces_abstention() -> None:
    scores = [0.99] * 400
    attack = [False] * 400
    gate = fit_close_gate(scores, attack, set_closing_only=[True] * 400)
    result = gate.evaluate(
        [0.999, 0.999], [True, False], [False, True], set_closing_only=[False, True]
    )
    assert result["closed"] == 1 and result["closed_attacks"] == 0
