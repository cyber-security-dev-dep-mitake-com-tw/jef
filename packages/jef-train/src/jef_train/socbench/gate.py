"""A measured risk gate for closing cases: fitted on calibration, scored on test.

The product never auto-closes (``jef_scene.policy``: closing is T2 and needs a
person). This gate answers the question a SOC would ask before ever relaxing
that: *if* cases above some score were closed without a look, how many attacks
would be closed with them, and how many benign cases would a human be spared?

It is fitted on a calibration split and then frozen, so the test split never
chooses its own threshold:

* score = probability on the closing options (benign, normal automation);
* a case is close-eligible when its score clears the threshold **and**, when
  prediction sets are given, its conformal set contains closing options only
  (any attack option in the set means "cannot rule it out": abstain);
* the threshold is the lowest one at which the calibration cases it would
  close contain attacks at a rate whose one-sided 95% Clopper-Pearson upper
  bound is at most ``max_attack_rate``. If no threshold gets there, the gate
  closes nothing.

Everything else is abstention: the case goes to a person.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["CloseGate", "clopper_pearson_upper", "fit_close_gate"]


def clopper_pearson_upper(failures: int, n: int, confidence: float = 0.95) -> float:
    """One-sided upper bound on a rate after ``failures`` in ``n`` trials."""
    if n <= 0:
        return 1.0
    if failures >= n:
        return 1.0
    from scipy.stats import beta

    return float(beta.ppf(confidence, failures + 1, n - failures))


@dataclass(frozen=True)
class CloseGate:
    threshold: float | None  # None: closes nothing
    max_attack_rate: float
    require_set: bool
    calibration: dict[str, Any]

    def eligible(self, score: float, set_is_closing_only: bool = True) -> bool:
        if self.threshold is None:
            return False
        return score >= self.threshold and (set_is_closing_only or not self.require_set)

    def evaluate(
        self,
        scores: Sequence[float],
        is_attack: Sequence[bool],
        is_closing_truth: Sequence[bool],
        set_closing_only: Sequence[bool] | None = None,
    ) -> dict[str, Any]:
        """What the frozen gate does to a labelled split."""
        sets = set_closing_only if set_closing_only is not None else [True] * len(scores)
        closed = [self.eligible(s, k) for s, k in zip(scores, sets, strict=True)]
        n_closed = sum(closed)
        attacks = sum(c and a for c, a in zip(closed, is_attack, strict=True))
        benign_total = sum(is_closing_truth)
        benign_closed = sum(c and b for c, b in zip(closed, is_closing_truth, strict=True))
        return {
            "cases": len(scores),
            "closed": n_closed,
            "closed_attacks": attacks,
            "closed_benign": benign_closed,
            "attack_rate_among_closed": (attacks / n_closed) if n_closed else None,
            "benign_yield": (benign_closed / benign_total) if benign_total else None,
            "abstained": len(scores) - n_closed,
        }


def fit_close_gate(
    scores: Sequence[float],
    is_attack: Sequence[bool],
    *,
    set_closing_only: Sequence[bool] | None = None,
    max_attack_rate: float = 0.01,
    confidence: float = 0.95,
) -> CloseGate:
    """The lowest threshold whose calibration closures are provably clean enough."""
    s = np.asarray(scores, dtype=np.float64)
    a = np.asarray(is_attack, dtype=bool)
    k = np.asarray(
        set_closing_only if set_closing_only is not None else [True] * len(s), dtype=bool
    )
    best: dict[str, Any] | None = None
    for t in sorted(set(s[k].tolist()), reverse=True):  # highest first, keep lowering
        mask = k & (s >= t)
        n, fails = int(mask.sum()), int((mask & a).sum())
        upper = clopper_pearson_upper(fails, n, confidence)
        if upper <= max_attack_rate:
            best = {"threshold": float(t), "closed": n, "attacks": fails, "upper_bound": upper}
    info = best or {"threshold": None, "closed": 0, "attacks": 0, "upper_bound": None}
    return CloseGate(
        threshold=info["threshold"],
        max_attack_rate=max_attack_rate,
        require_set=set_closing_only is not None,
        calibration={**info, "cases": len(s), "attacks_total": int(a.sum())},
    )
