"""Calibration: the deliverable the rest of the ecosystem disclaims.

Nearly every open System One project ships the line *"confidence measures
concentration, not correctness"* and stops there. That is honest but useless:
a SOAR gate threshold built on an uncalibrated distribution is a coin flip with
extra steps. JEF therefore treats calibration as a first-class artifact (D4):

1. **Temperature scaling** -- one scalar per (kind, n_options) bucket, fitted by
   minimising NLL on a held-out split. Buckets are separate because a 2-option
   noul and a 5-level score have genuinely different sharpness profiles.
2. **Split conformal prediction** -- a per-bucket nonconformity quantile that
   turns a distribution into a *prediction set* with a finite-sample coverage
   guarantee. This is what gives "low confidence -> escalate to a human" an
   actual statistical meaning instead of a vibe.

3. **A confidence-to-correctness map** -- isotonic regression from Jev's
   confidence statistic onto the observed probability of being right.

   This one is the direct answer to the line the whole ecosystem ships:
   *"confidence measures concentration, not correctness"*. That is true of the
   raw statistic, and it makes a gate threshold meaningless -- ``confidence >=
   0.9`` says the distribution is peaked, not that the answer is right 90% of
   the time. With a held-out split you can simply measure the relationship and
   fit it. A gate can then threshold on P(correct), which is the quantity an
   operator actually meant all along.

Both artifacts are tiny JSON and ship alongside the head weights.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .mathx import confidence as _confidence
from .mathx import softmax

__all__ = [
    "Bucket",
    "Calibrator",
    "bucket_key",
    "fit_conformal_quantile",
    "fit_temperature",
]


def bucket_key(kind: str, n_options: int) -> str:
    """Calibration bucket identity: question kind plus option count."""
    return f"{kind}:{n_options}"


@dataclass
class Bucket:
    """Fitted calibration parameters for one (kind, n_options) bucket."""

    temperature: float = 1.0
    #: alpha -> nonconformity quantile. Keys are stringified floats for JSON.
    conformal: dict[str, float] = field(default_factory=dict)
    #: Breakpoints of the isotonic confidence -> P(correct) map, as
    #: ``[[confidence, p_correct], ...]`` sorted by confidence. Empty when
    #: unfitted, in which case P(correct) is simply not available.
    correctness_map: list[list[float]] = field(default_factory=list)
    n_samples: int = 0


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #


def _nll(logits: Sequence[NDArray[np.floating]], labels: Sequence[int], t: float) -> float:
    total = 0.0
    for lg, y in zip(logits, labels, strict=True):
        p = softmax(np.asarray(lg, dtype=np.float64), temperature=t)
        total -= math.log(max(float(p[y]), 1e-12))
    return total / max(len(labels), 1)


def fit_temperature(
    logits: Sequence[NDArray[np.floating]],
    labels: Sequence[int],
    *,
    lo: float = 0.05,
    hi: float = 20.0,
    iterations: int = 60,
) -> float:
    """Fit one temperature by minimising NLL.

    Uses a coarse grid followed by golden-section refinement -- the objective is
    one-dimensional and smooth, so this avoids a scipy dependency in the core
    package (which must stay importable on the Go/ONNX serving path's toolchain).
    """
    if not labels:
        return 1.0

    grid = np.geomspace(lo, hi, 24)
    best_t = float(grid[int(np.argmin([_nll(logits, labels, float(t)) for t in grid]))])

    a, b = max(lo, best_t / 3.0), min(hi, best_t * 3.0)
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - inv_phi * (b - a), a + inv_phi * (b - a)
    fc, fd = _nll(logits, labels, c), _nll(logits, labels, d)
    for _ in range(iterations):
        if b - a < 1e-4:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = _nll(logits, labels, c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = _nll(logits, labels, d)
    return float((a + b) / 2.0)


def fit_conformal_quantile(
    probabilities: Sequence[NDArray[np.floating]],
    labels: Sequence[int],
    alpha: float,
) -> float:
    """Split-conformal nonconformity quantile for miscoverage ``alpha``.

    Nonconformity score is ``1 - p[true label]``. The returned ``qhat`` yields
    prediction sets ``{j : p_j >= 1 - qhat}`` whose marginal coverage is at least
    ``1 - alpha`` on exchangeable data.
    """
    n = len(labels)
    if n == 0:
        return 1.0
    scores = np.array(
        [
            1.0 - float(np.asarray(p, dtype=np.float64)[y])
            for p, y in zip(probabilities, labels, strict=True)
        ]
    )
    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    if level >= 1.0:
        # Too few calibration points to certify this alpha; fall back to the
        # trivial (always-covering) set rather than silently under-covering.
        return 1.0
    return float(np.quantile(scores, level, method="higher"))


def isotonic_fit(x: Sequence[float], y: Sequence[float]) -> list[list[float]]:
    """Pool-adjacent-violators isotonic regression, returning breakpoints.

    Implemented here rather than pulled from scikit-learn because jef-core is a
    serving dependency: it ships in the container and is the reference the Go
    port follows, so it stays on numpy alone.

    Args:
        x: Predictor values (confidence).
        y: Targets in [0, 1] (1.0 for a correct answer, 0.0 otherwise).

    Returns:
        ``[[x, y], ...]`` sorted and non-decreasing in y.
    """
    if not x:
        return []
    order = sorted(range(len(x)), key=lambda i: x[i])
    xs = [float(x[i]) for i in order]
    ys = [float(y[i]) for i in order]

    # Each block is [sum, count, right_edge_index].
    blocks: list[list[float]] = []
    for i, value in enumerate(ys):
        blocks.append([value, 1.0, float(i)])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
            merged_sum = blocks[-2][0] + blocks[-1][0]
            merged_count = blocks[-2][1] + blocks[-1][1]
            right = blocks[-1][2]
            blocks[-2:] = [[merged_sum, merged_count, right]]

    out: list[list[float]] = []
    for block in blocks:
        mean = block[0] / block[1]
        out.append([xs[int(block[2])], mean])
    return out


#: Below this many calibration points, an isotonic curve memorises rather than
#: fits. Measured on this project's own corpus: a bucket with 62 calibration
#: samples produced a map whose out-of-sample ECE was 0.235, against 0.046 for a
#: bucket with 290. Returning nothing is the honest outcome -- a gate can react
#: to "P(correct) unavailable", but not to a confidently wrong one.
MIN_CORRECTNESS_SAMPLES = 150


def fit_correctness_map(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    min_samples: int = MIN_CORRECTNESS_SAMPLES,
) -> list[list[float]]:
    """Fit confidence -> P(correct), or return nothing if the data cannot support it."""
    if len(confidences) < min_samples:
        return []
    return isotonic_fit(list(confidences), [1.0 if c else 0.0 for c in correct])


# --------------------------------------------------------------------------- #
# Calibrator
# --------------------------------------------------------------------------- #

DEFAULT_ALPHAS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)


@dataclass
class Calibrator:
    """Applies fitted temperatures and conformal quantiles at inference time."""

    buckets: dict[str, Bucket] = field(default_factory=dict)
    backbone: str = ""
    head: str = ""
    notes: str = ""

    # -- inference --------------------------------------------------------- #

    def temperature(self, kind: str, n_options: int) -> float:
        b = self.buckets.get(bucket_key(kind, n_options))
        return b.temperature if b else 1.0

    def apply(self, logits: NDArray[np.floating], kind: str, n_options: int) -> NDArray[np.float64]:
        """Logits -> calibrated probabilities."""
        return softmax(logits, temperature=self.temperature(kind, n_options))

    def prediction_set(
        self,
        probabilities: NDArray[np.floating],
        kind: str,
        n_options: int,
        alpha: float = 0.10,
    ) -> list[int]:
        """Indices whose probability clears the conformal threshold.

        An empty-or-singleton set means the answer is safe to act on; a larger
        set is the statistically honest signal that the model cannot separate
        the candidates and the decision belongs to a human.
        """
        b = self.buckets.get(bucket_key(kind, n_options))
        p = np.asarray(probabilities, dtype=np.float64)
        if b is None:
            return [int(np.argmax(p))]
        qhat = b.conformal.get(_akey(alpha))
        if qhat is None:
            return [int(np.argmax(p))]
        keep = [int(i) for i in np.flatnonzero(p >= 1.0 - qhat)]
        return keep or [int(np.argmax(p))]

    def p_correct(self, confidence: float, kind: str, n_options: int) -> float | None:
        """Calibrated probability that the top answer is correct.

        ``None`` when no correctness map was fitted for this bucket -- the
        honest answer, and the one a gate must be able to distinguish from a
        low probability. Scene gates surface this as ``p_correct`` so a
        threshold means what an operator thought it meant.
        """
        bucket = self.buckets.get(bucket_key(kind, n_options))
        if bucket is None or not bucket.correctness_map:
            return None
        points = bucket.correctness_map
        if confidence <= points[0][0]:
            return float(points[0][1])
        if confidence >= points[-1][0]:
            return float(points[-1][1])
        for (x0, y0), (x1, y1) in pairwise(points):
            if x0 <= confidence <= x1:
                if x1 == x0:
                    return float(y1)
                ratio = (confidence - x0) / (x1 - x0)
                return float(y0 + ratio * (y1 - y0))
        return float(points[-1][1])  # pragma: no cover -- covered by the bounds above

    def is_fitted(self) -> bool:
        return bool(self.buckets)

    def has_correctness_map(self, kind: str, n_options: int) -> bool:
        bucket = self.buckets.get(bucket_key(kind, n_options))
        return bool(bucket and bucket.correctness_map)

    # -- fitting ----------------------------------------------------------- #

    def fit_bucket(
        self,
        kind: str,
        n_options: int,
        logits: Sequence[NDArray[np.floating]],
        labels: Sequence[int],
        alphas: Sequence[float] = DEFAULT_ALPHAS,
    ) -> Bucket:
        """Fit temperature then conformal quantiles for one bucket.

        Conformal quantiles are computed from *temperature-scaled* probabilities
        because that is what inference will produce; fitting them on raw softmax
        output would silently break the coverage guarantee.
        """
        t = fit_temperature(logits, labels)
        probs = [softmax(np.asarray(lg, dtype=np.float64), temperature=t) for lg in logits]
        confidences = [_confidence(p) for p in probs]
        hits = [int(np.argmax(p)) == y for p, y in zip(probs, labels, strict=True)]
        bucket = Bucket(
            temperature=t,
            conformal={_akey(a): fit_conformal_quantile(probs, labels, a) for a in alphas},
            correctness_map=fit_correctness_map(confidences, hits),
            n_samples=len(labels),
        )
        self.buckets[bucket_key(kind, n_options)] = bucket
        return bucket

    # -- persistence ------------------------------------------------------- #

    def to_dict(self) -> dict[str, object]:
        return {
            "backbone": self.backbone,
            "head": self.head,
            "notes": self.notes,
            "buckets": {k: asdict(v) for k, v in self.buckets.items()},
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Calibrator:
        raw = d.get("buckets") or {}
        if not isinstance(raw, dict):
            raise TypeError("calibration 'buckets' must be an object")
        buckets = {str(k): Bucket(**v) for k, v in raw.items()}
        return cls(
            buckets=buckets,
            backbone=str(d.get("backbone", "")),
            head=str(d.get("head", "")),
            notes=str(d.get("notes", "")),
        )

    @classmethod
    def load(cls, path: str | Path) -> Calibrator:
        """Load from a local path or an ``hf://owner/repo/calibration.json``."""
        from .artifacts import resolve_artifact

        return cls.from_dict(json.loads(resolve_artifact(path).read_text(encoding="utf-8")))


def _akey(alpha: float) -> str:
    return f"{alpha:.4f}"
