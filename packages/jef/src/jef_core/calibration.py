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
from typing import Literal

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


def bucket_key(kind: str, n_options: int, task_id: str | None = None) -> str:
    """Calibration bucket identity.

    ``task_id|kind:n`` when the task is known, ``kind:n`` otherwise. The two
    never alias: a question that names its task is answered only by a bucket
    fitted for that task, and a question that does not is answered only by a
    shape-scoped bucket. "CVE user interaction" and "phishing true positive"
    are both ``noul:2``; their ground truth, base rate and error cost are not.
    """
    shape = f"{kind}:{n_options}"
    return f"{task_id}|{shape}" if task_id else shape


def split_bucket_key(key: str) -> tuple[str | None, str, int]:
    """Inverse of :func:`bucket_key`: ``(task_id, kind, n_options)``."""
    task, _, shape = key.rpartition("|")
    kind, _, n = shape.partition(":")
    return (task or None), kind, int(n)


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
    #: The task this bucket was fitted for; ``None`` for a shape-scoped bucket.
    #: Redundant with the key on purpose: a bucket pulled out of the dict still
    #: knows what it is for.
    task_id: str | None = None
    #: Measured on the calibration split when the bucket was fitted (ece, brier,
    #: nll, coverage_at_90). Absent when the fitter did not compute them.
    metrics: dict[str, float] | None = None


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

    Implemented here rather than pulled from scikit-learn because the engine is a
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

#: What a gate or a policy may conclude about one bucket's calibration.
#:
#: ``validated``    a correctness map and a conformal quantile for the requested
#:                  alpha were both fitted on enough data; ``p_correct`` and
#:                  ``prediction_set`` mean what they say.
#: ``insufficient`` the bucket exists but something is missing: too few samples
#:                  for a correctness map (``MIN_CORRECTNESS_SAMPLES``), or no
#:                  quantile for this alpha. ``p_correct`` is ``None``.
#: ``missing``      no bucket for this (kind, n_options) at all.
#:
#: ``legacy``       the artifact predates task-scoped calibration (schema 1).
#:                  Its numbers are still shown for analysis, but nothing in
#:                  it was fitted for a named task, so it never unlocks
#:                  automation. ``jef calibration migrate`` is the explicit
#:                  way out; there is no silent one.
#: ``incompatible`` the artifact names a model digest and the running model
#:                  has a different one. Temperatures, quantiles and maps for
#:                  another model are noise: ``p_correct`` is ``None`` and the
#:                  prediction set is the full set until this is fixed.
#:
#: Only ``validated`` may unlock automation. Everything else is "cannot say",
#: and a UI must render it as such, not as a wide-but-trustworthy set.
CalibrationStatus = Literal["validated", "insufficient", "missing", "legacy", "incompatible"]

#: The artifact format this code writes. 1 is the shape-keyed format that
#: ``models/jef-v0/calibration.json`` still carries; it loads as ``legacy``.
CALIBRATION_SCHEMA = 2


@dataclass
class ArtifactInfo:
    """Provenance of a calibration artifact: what, from what, when.

    Every field is recorded as what it is. ``model_sha256`` is the digest of
    the head weights the buckets were fitted against, and the only basis for
    ``incompatible``. ``dataset_digest`` is the corpus manifest the
    calibration split came from. ``calibration_window`` is the data's time
    range when the corpus has one, else ``None`` -- not today's date.
    """

    model_sha256: str | None = None
    label_schema_version: str = "1"
    language: str = ""
    dataset_digest: str | None = None
    calibration_window: list[str] | None = None
    split_policy: str = "group"
    sample_count: int = 0
    min_validated_samples: int = MIN_CORRECTNESS_SAMPLES


@dataclass
class Calibrator:
    """Applies fitted temperatures and conformal quantiles at inference time."""

    buckets: dict[str, Bucket] = field(default_factory=dict)
    backbone: str = ""
    head: str = ""
    notes: str = ""
    #: Artifact format. In-memory calibrators are current; a loaded schema-1
    #: file keeps 1 and reports every bucket as ``legacy``.
    schema: int = CALIBRATION_SCHEMA
    artifact: ArtifactInfo = field(default_factory=ArtifactInfo)
    #: Digest of the head actually running, set by the loader (``bind_model``).
    #: Compared with ``artifact.model_sha256`` for ``incompatible``.
    runtime_model_sha256: str | None = None

    # -- identity ---------------------------------------------------------- #

    @property
    def legacy(self) -> bool:
        return self.schema < CALIBRATION_SCHEMA

    @property
    def incompatible(self) -> bool:
        a, r = self.artifact.model_sha256, self.runtime_model_sha256
        return bool(a and r and a != r)

    def bind_model(self, model_sha256: str | None) -> None:
        """Tell the calibrator which head is running, so a mismatch can be seen."""
        self.runtime_model_sha256 = model_sha256

    def _bucket(self, kind: str, n_options: int, task_id: str | None) -> Bucket | None:
        """The bucket for exactly this identity, or ``None``. Never a neighbour's.

        Also ``None`` when the artifact is incompatible with the running model:
        numbers fitted for another head are not calibration, and every
        inference path below falls back to its uncalibrated behaviour.
        """
        if self.incompatible:
            return None
        return self.buckets.get(bucket_key(kind, n_options, task_id))

    # -- inference --------------------------------------------------------- #

    def temperature(self, kind: str, n_options: int, task_id: str | None = None) -> float:
        b = self._bucket(kind, n_options, task_id)
        return b.temperature if b else 1.0

    def status(
        self, kind: str, n_options: int, alpha: float = 0.10, task_id: str | None = None
    ) -> CalibrationStatus:
        """Per-bucket calibration status; see :data:`CalibrationStatus`.

        Decided from this bucket alone. A fitted bucket of another shape, or of
        the same shape for another task, never makes this one look calibrated.
        ``legacy`` and ``incompatible`` are artifact-wide and come first: a
        schema-1 file has no task identity to check, and an artifact for a
        different model has nothing to say about this one.
        """
        if self.legacy:
            return "legacy"
        if self.incompatible:
            return "incompatible"
        b = self.buckets.get(bucket_key(kind, n_options, task_id))
        if b is None:
            return "missing"
        if not b.correctness_map or b.conformal.get(_akey(alpha)) is None:
            return "insufficient"
        return "validated"

    def apply(
        self,
        logits: NDArray[np.floating],
        kind: str,
        n_options: int,
        task_id: str | None = None,
    ) -> NDArray[np.float64]:
        """Logits -> calibrated probabilities."""
        return softmax(logits, temperature=self.temperature(kind, n_options, task_id))

    def prediction_set(
        self,
        probabilities: NDArray[np.floating],
        kind: str,
        n_options: int,
        alpha: float = 0.10,
        task_id: str | None = None,
    ) -> list[int]:
        """Indices whose probability clears the conformal threshold.

        A singleton set means the answer is safe to act on; a larger set is the
        statistically honest signal that the model cannot separate the
        candidates and the decision belongs to a human.

        The method fails closed. Whenever no fitted quantile applies -- the
        bucket for ``(kind, n_options)`` was never fitted, the bucket has no
        quantile for this ``alpha``, or no probability clears the threshold --
        it returns the full index set ``list(range(len(probabilities)))``.
        That is the only set whose coverage is actually guaranteed without a
        quantile, and it keeps ``set_size`` gates from automating on a bucket
        nobody calibrated. Returning ``[argmax]`` in those cases would dress an
        uncalibrated guess up as a certified singleton.
        """
        p = np.asarray(probabilities, dtype=np.float64)
        full = list(range(len(p)))
        b = self._bucket(kind, n_options, task_id)
        if b is None:
            return full
        qhat = b.conformal.get(_akey(alpha))
        if qhat is None:
            return full
        keep = [int(i) for i in np.flatnonzero(p >= 1.0 - qhat)]
        return keep or full

    def p_correct(
        self, confidence: float, kind: str, n_options: int, task_id: str | None = None
    ) -> float | None:
        """Calibrated probability that the top answer is correct.

        ``None`` when no correctness map was fitted for this bucket -- the
        honest answer, and the one a gate must be able to distinguish from a
        low probability. Scene gates surface this as ``p_correct`` so a
        threshold means what an operator thought it meant. A ``legacy``
        artifact still answers here, for analysis; ``status()`` is what says
        whether the answer may drive anything.
        """
        bucket = self._bucket(kind, n_options, task_id)
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

    def has_correctness_map(self, kind: str, n_options: int, task_id: str | None = None) -> bool:
        bucket = self._bucket(kind, n_options, task_id)
        return bool(bucket and bucket.correctness_map)

    def tasks(self) -> list[str]:
        """Distinct task ids this artifact was fitted for (empty for shape-only)."""
        return sorted({b.task_id for b in self.buckets.values() if b.task_id})

    # -- fitting ----------------------------------------------------------- #

    def fit_bucket(
        self,
        kind: str,
        n_options: int,
        logits: Sequence[NDArray[np.floating]],
        labels: Sequence[int],
        alphas: Sequence[float] = DEFAULT_ALPHAS,
        *,
        task_id: str | None = None,
        metrics: dict[str, float] | None = None,
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
            task_id=task_id,
            metrics=metrics,
        )
        self.buckets[bucket_key(kind, n_options, task_id)] = bucket
        return bucket

    # -- migration --------------------------------------------------------- #

    def migrate(self, task_id: str, *, model_sha256: str | None = None) -> Calibrator:
        """A schema-2 copy of a legacy artifact with every bucket scoped to ``task_id``.

        Deliberately explicit and deliberately whole: the caller asserts that
        every bucket in the file was fitted on data for this one task. A file
        that mixes tasks has to be refitted, not relabelled. The original is
        not touched.
        """
        if not task_id:
            raise ValueError("migrate needs a task_id")
        if any(b.task_id for b in self.buckets.values()):
            raise ValueError("artifact already carries task ids; refit instead of migrating")
        out = Calibrator(
            backbone=self.backbone,
            head=self.head,
            notes=(self.notes + " " if self.notes else "")
            + f"Migrated from schema 1 with task_id={task_id}; provenance fields unknown.",
            artifact=ArtifactInfo(
                model_sha256=model_sha256, sample_count=self.artifact.sample_count
            ),
        )
        for key, bucket in self.buckets.items():
            _, kind, n = split_bucket_key(key)
            moved = Bucket(**{**asdict(bucket), "task_id": task_id})
            out.buckets[bucket_key(kind, n, task_id)] = moved
        out.artifact.sample_count = sum(b.n_samples for b in out.buckets.values())
        return out

    # -- persistence ------------------------------------------------------- #

    def to_dict(self) -> dict[str, object]:
        """Always the current schema. A legacy calibrator saved here becomes a
        schema-2 file with shape-only buckets, which is still ``missing`` for
        any question that names a task -- saving does not launder it."""
        return {
            "calibration_schema": CALIBRATION_SCHEMA,
            "backbone": self.backbone,
            "head": self.head,
            "notes": self.notes,
            "artifact": asdict(self.artifact),
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
        schema = int(d.get("calibration_schema", 1))  # type: ignore[call-overload]
        if schema > CALIBRATION_SCHEMA:
            raise ValueError(
                f"calibration_schema {schema} is newer than this code understands "
                f"({CALIBRATION_SCHEMA}); upgrade jef"
            )
        buckets: dict[str, Bucket] = {}
        for k, v in raw.items():
            task, _, _ = split_bucket_key(str(k))
            fields = {**v, "task_id": v.get("task_id") or task}
            if schema < CALIBRATION_SCHEMA and fields["task_id"]:
                raise ValueError(f"schema-1 artifact carries a task-scoped bucket {k!r}")
            buckets[str(k)] = Bucket(**fields)
        info = d.get("artifact") or {}
        if not isinstance(info, dict):
            raise TypeError("calibration 'artifact' must be an object")
        return cls(
            buckets=buckets,
            backbone=str(d.get("backbone", "")),
            head=str(d.get("head", "")),
            notes=str(d.get("notes", "")),
            schema=schema,
            artifact=ArtifactInfo(**info),
        )

    @classmethod
    def load(cls, path: str | Path) -> Calibrator:
        """Load from a local path or an ``hf://owner/repo/calibration.json``."""
        from .artifacts import resolve_artifact

        return cls.from_dict(json.loads(resolve_artifact(path).read_text(encoding="utf-8")))


def _akey(alpha: float) -> str:
    return f"{alpha:.4f}"


def file_sha256(path: str | Path) -> str:
    """``sha256:<hex>`` of a file, streamed. Used to fingerprint head weights."""
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
