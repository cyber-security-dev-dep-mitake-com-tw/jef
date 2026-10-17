"""SOC metrics: what a detection costs at the base rate a SOC actually sees.

Accuracy on a balanced benchmark says little about a queue where one alert in
a hundred is real (review 11.3, 11.4). These are the numbers the report
leads with:

* ``average_precision`` (PR-AUC) and ``fnr_at_fpr``: detection at a fixed
  false-positive budget;
* ``precision_at_budget``: of the N alerts a shift can look at, how many are
  real;
* ``ppv_at_prevalence``: the PPV the measured TPR and FPR imply at 1, 3 and 5
  percent prevalence, because balanced-set precision does not transfer;
* ``ece``, ``brier``, ``nll``: whether probabilities mean what they say;
* ``risk_coverage``: whether abstaining on low confidence buys lower error;
* ``cluster_bootstrap``: confidence intervals resampled by family, because a
  thousand alerts from one incident are one sample, not a thousand.

Pure numpy; no plotting, no sklearn.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "average_precision",
    "brier",
    "cluster_bootstrap",
    "ece",
    "fnr_at_fpr",
    "nll",
    "ppv_at_prevalence",
    "precision_at_budget",
    "risk_coverage",
    "tpr_fpr_at",
]


def _arr(x: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(x, dtype=np.float64)


def average_precision(scores: ArrayLike, positive: ArrayLike) -> float | None:
    """Area under the precision-recall curve (step interpolation). ``None`` without positives."""
    s, y = _arr(scores), np.asarray(positive, dtype=bool)
    if not y.any():
        return None
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def tpr_fpr_at(
    scores: ArrayLike, positive: ArrayLike, threshold: float
) -> tuple[float | None, float | None]:
    s, y = _arr(scores), np.asarray(positive, dtype=bool)
    flagged = s >= threshold
    tpr = float(flagged[y].mean()) if y.any() else None
    fpr = float(flagged[~y].mean()) if (~y).any() else None
    return tpr, fpr


def fnr_at_fpr(
    scores: ArrayLike, positive: ArrayLike, max_fpr: float = 0.05
) -> dict[str, float | None]:
    """The lowest miss rate achievable while keeping FPR at or under ``max_fpr``."""
    s, y = _arr(scores), np.asarray(positive, dtype=bool)
    if not y.any() or not (~y).any():
        return {"fnr": None, "fpr": None, "threshold": None}
    best: dict[str, float | None] = {"fnr": 1.0, "fpr": 0.0, "threshold": math.inf}
    for t in np.unique(s)[::-1]:
        tpr, fpr = tpr_fpr_at(s, y, float(t))
        if (
            fpr is not None
            and tpr is not None
            and fpr <= max_fpr
            and 1 - tpr < (best["fnr"] or 1.0) + 1e-12
        ):
            best = {"fnr": 1 - tpr, "fpr": fpr, "threshold": float(t)}
    return best


def precision_at_budget(scores: ArrayLike, positive: ArrayLike, budget: int) -> float | None:
    """Of the ``budget`` highest-scored items, the share that are positive."""
    s, y = _arr(scores), np.asarray(positive, dtype=bool)
    if budget <= 0 or len(s) == 0:
        return None
    top = np.argsort(-s, kind="mergesort")[: min(budget, len(s))]
    return float(y[top].mean())


def ppv_at_prevalence(tpr: float, fpr: float, prevalence: float) -> float | None:
    """PPV = TPR*pi / (TPR*pi + FPR*(1-pi)). Review 11.4: TPR .9, FPR .05, pi .01 -> 0.154."""
    denom = tpr * prevalence + fpr * (1 - prevalence)
    return None if denom == 0 else tpr * prevalence / denom


def ece(confidence: ArrayLike, correct: ArrayLike, bins: int = 15) -> float | None:
    c, y = _arr(confidence), np.asarray(correct, dtype=np.float64)
    if len(c) == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(c, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(c[m].mean() - y[m].mean())
    return float(total)


def brier(probabilities: NDArray[np.floating], labels: Sequence[int]) -> float | None:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.size == 0:
        return None
    onehot = np.zeros_like(p)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def nll(
    probabilities: NDArray[np.floating], labels: Sequence[int], eps: float = 1e-12
) -> float | None:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.size == 0:
        return None
    return float(-np.log(np.clip(p[np.arange(len(labels)), labels], eps, 1.0)).mean())


def risk_coverage(confidence: ArrayLike, correct: ArrayLike) -> dict[str, object]:
    """Selective risk at each coverage when abstaining on the least confident first."""
    c, y = _arr(confidence), np.asarray(correct, dtype=bool)
    if len(c) == 0:
        return {"aurc": None, "points": []}
    order = np.argsort(-c, kind="mergesort")
    errors = np.cumsum(~y[order])
    n = np.arange(1, len(c) + 1)
    risk = errors / n
    coverage = n / len(c)
    marks = sorted({max(1, round(len(c) * q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)})
    return {
        "aurc": float(risk.mean()),
        "points": [
            {"coverage": round(float(coverage[k - 1]), 4), "risk": round(float(risk[k - 1]), 4)}
            for k in marks
        ],
    }


def cluster_bootstrap(
    groups: Sequence[str],
    statistic: Callable[[NDArray[np.int64]], float | None],
    *,
    n_boot: int = 500,
    seed: int = 1337,
    alpha: float = 0.05,
) -> dict[str, float | None]:
    """Percentile CI for ``statistic(indices)``, resampling whole groups (families)."""
    g = np.asarray(groups)
    uniq = np.unique(g)
    members = {u: np.flatnonzero(g == u) for u in uniq}
    point = statistic(np.arange(len(g)))
    if len(uniq) < 2 or point is None:
        return {"point": point, "low": None, "high": None, "groups": len(uniq)}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[u] for u in pick])
        v = statistic(idx)
        if v is not None:
            values.append(v)
    if not values:
        return {"point": point, "low": None, "high": None, "groups": len(uniq)}
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return {"point": point, "low": float(lo), "high": float(hi), "groups": len(uniq)}
