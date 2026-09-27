"""Training sample schema shared by every corpus builder.

A sample is one (state, question, correct option) triple. That is deliberately
the same shape the engine consumes at inference time, so nothing has to be
translated between training and serving -- the head sees exactly what it will
see in production.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "Sample",
    "assert_no_group_leakage",
    "read_samples",
    "split_samples",
    "write_samples",
]

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
        group: The piece of evidence this sample derives from -- an ATT&CK
            technique id, a CVE id, a SOAR scenario template. Splitting is done
            by group, never by sample, because one CVE yields both a severity
            question and an attack-vector question **over identical state text**.
            Splitting per sample puts that text in train and in test at once, and
            every metric computed afterwards is measuring memorisation.
        lang: BCP-47-ish tag. ``zh-TW`` is a first-class citizen here.
    """

    state: str
    kind: Kind
    instructions: str
    option_keys: list[str]
    option_labels: list[str | None]
    label: int
    source: str
    group: str = ""
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

    @property
    def uid(self) -> str:
        """Stable identity, so a feature cache built over the whole corpus can
        be sliced into splits without re-running the backbone."""
        digest = hashlib.sha1(  # noqa: S324 -- an index key, not a security boundary
            "\x00".join(
                [self.source, self.instructions, self.state, str(self.label), *self.option_keys]
            ).encode()
        )
        return digest.hexdigest()[:20]

    @property
    def group_key(self) -> str:
        """Split unit. Falls back to the state text when no group was declared.

        The fallback is deliberate rather than permissive: two samples sharing
        state text must never straddle a split, so an un-grouped builder still
        gets the minimum correct behaviour instead of silently leaking.
        """
        return self.group or f"state:{hash(self.state)}"

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
    """Three-way split: train / calibration / test, **grouped by evidence**.

    Grouping is the load-bearing part. One CVE produces a severity question and
    an attack-vector question over the same description; one SOAR scenario
    produces near-identical alerts that differ only in hostname and timestamp.
    Splitting per sample puts that text on both sides of the line, and the
    resulting accuracy measures recall of the training set. The first run of this
    pipeline scored 1.000 on SOAR routing for exactly that reason.

    Calibration also gets its own slice. Fitting temperature or conformal
    quantiles on data the head trained on produces a calibrator that looks
    excellent and generalises not at all -- which would be an embarrassing
    failure for a project whose pitch is calibration.

    Groups are assigned stratified by their dominant source, so every generator
    appears in all three slices.
    """
    import random

    if not 0 < train < 1 or not 0 < calibration < 1 or train + calibration >= 1:
        raise ValueError("train and calibration must be positive and sum to < 1")

    groups: dict[str, list[Sample]] = {}
    for s in samples:
        groups.setdefault(s.group_key, []).append(s)

    # Stratify group assignment by the source that dominates each group.
    by_source: dict[str, list[str]] = {}
    for key, members in groups.items():
        counts = Counter(m.source for m in members)
        by_source.setdefault(counts.most_common(1)[0][0], []).append(key)

    rng = random.Random(seed)  # noqa: S311 -- reproducible split, not crypto
    tr: list[Sample] = []
    cal: list[Sample] = []
    te: list[Sample] = []
    for source in sorted(by_source):
        keys = sorted(by_source[source])
        rng.shuffle(keys)
        n = len(keys)
        i = int(n * train)
        j = i + int(n * calibration)
        for key in keys[:i]:
            tr.extend(groups[key])
        for key in keys[i:j]:
            cal.extend(groups[key])
        for key in keys[j:]:
            te.extend(groups[key])

    rng.shuffle(tr)
    rng.shuffle(cal)
    rng.shuffle(te)
    return tr, cal, te


def assert_no_group_leakage(*splits: list[Sample]) -> None:
    """Raise if any evidence group appears in more than one split.

    Called by the corpus builder so contamination fails the build rather than
    quietly inflating a published number.
    """
    seen: dict[str, int] = {}
    for index, split in enumerate(splits):
        for s in split:
            previous = seen.setdefault(s.group_key, index)
            if previous != index:
                raise ValueError(
                    f"group {s.group_key!r} appears in split {previous} and "
                    f"split {index}: the same evidence is on both sides of the "
                    "split and every metric downstream is contaminated"
                )
