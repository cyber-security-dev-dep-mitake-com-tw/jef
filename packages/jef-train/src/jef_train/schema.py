"""Training sample schema shared by every corpus builder.

A sample is one (state, question, correct option) triple. That is deliberately
the same shape the engine consumes at inference time, so nothing has to be
translated between training and serving -- the head sees exactly what it will
see in production.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

__all__ = ["Sample", "read_samples", "split_samples", "write_samples"]

Kind = Literal["choice", "score", "noul"]


@dataclass(frozen=True)
class Sample:
    """One labelled decision.

    Attributes:
        state: The text the model reads.
        kind: Question primitive.
        instructions: The question asked of the state.
        option_keys: Allowed answers, in order. For ``noul`` always
            ``["false", "true"]`` so index 1 is P(true), matching the engine.
        option_labels: Per-option descriptions; ``None`` where absent.
        label: Index into ``option_keys`` of the correct answer.
        source: Which builder produced this, for provenance and for stratifying
            the eval split so a benchmark cannot be dominated by one generator.
        lang: BCP-47-ish tag. ``zh-TW`` is a first-class citizen here.
    """

    state: str
    kind: Kind
    instructions: str
    option_keys: list[str]
    option_labels: list[str | None]
    label: int
    source: str
    lang: str = "zh-TW"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.option_keys) < 2:
            raise ValueError(f"{self.source}: need >= 2 options, got {self.option_keys}")
        if len(self.option_labels) != len(self.option_keys):
            raise ValueError(f"{self.source}: option_labels must align with option_keys")
        if not 0 <= self.label < len(self.option_keys):
            raise ValueError(f"{self.source}: label {self.label} out of range")
        if self.kind == "noul" and self.option_keys != ["false", "true"]:
            raise ValueError("noul samples must use option_keys ['false', 'true']")

    @property
    def n_options(self) -> int:
        return len(self.option_keys)

    @property
    def bucket(self) -> str:
        """Calibration bucket this sample belongs to."""
        return f"{self.kind}:{self.n_options}"

    def to_question(self) -> dict[str, Any]:
        """Render as a wire question, for round-tripping through the engine."""
        if self.kind == "choice":
            return {
                "type": "choice",
                "instructions": self.instructions,
                "criteria": dict(zip(self.option_keys, self.option_labels, strict=True)),
            }
        if self.kind == "score":
            return {
                "type": "score",
                "instructions": self.instructions,
                "criteria": self.option_keys,
            }
        criteria = {
            k: v for k, v in zip(self.option_keys, self.option_labels, strict=True) if v is not None
        }
        q: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if criteria:
            q["criteria"] = criteria
        return q


def write_samples(samples: Iterable[Sample], path: str | Path) -> int:
    """Write JSONL. Returns the count written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_samples(path: str | Path) -> Iterator[Sample]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Sample(**json.loads(line))


def split_samples(
    samples: list[Sample],
    *,
    train: float = 0.65,
    # 20%, not the more conventional 15%: the confidence-to-correctness map is
    # fitted here and needs ~150 points *per bucket* to be honest, and the
    # narrowest bucket is a fraction of the corpus. Undersizing this split means
    # shipping a model whose gates have no P(correct) to threshold on.
    calibration: float = 0.20,
    seed: int = 1337,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Three-way split: train / calibration / test.

    Calibration gets its own slice on purpose. Fitting temperature or conformal
    quantiles on data the head was trained on produces a calibrator that looks
    excellent and generalises not at all -- which is precisely the failure this
    project claims to fix, so it would be an embarrassing one to commit.

    The split is stratified by ``(source, bucket)`` so every generator and every
    question shape is represented in all three slices.
    """
    import random

    if not 0 < train < 1 or not 0 < calibration < 1 or train + calibration >= 1:
        raise ValueError("train and calibration must be positive and sum to < 1")

    strata: dict[tuple[str, str], list[Sample]] = {}
    for s in samples:
        strata.setdefault((s.source, s.bucket), []).append(s)

    # Seeded and reproducible by design; this splits a dataset, not keys.
    rng = random.Random(seed)  # noqa: S311
    tr: list[Sample] = []
    cal: list[Sample] = []
    te: list[Sample] = []
    for key in sorted(strata):
        group = strata[key][:]
        rng.shuffle(group)
        n = len(group)
        i = int(n * train)
        j = i + int(n * calibration)
        tr.extend(group[:i])
        cal.extend(group[i:j])
        te.extend(group[j:])
    rng.shuffle(tr)
    rng.shuffle(cal)
    rng.shuffle(te)
    return tr, cal, te
