"""The ``/v1/systemone`` wire contract, plus JEF's normalized internal form.

Two dialects exist in the wild for the same primitive:

* TypeSafe's native API calls the yes/no primitive ``noul`` and returns it as
  ``{"noul": 0.83}``.
* Vercel AI SDK's ``experimental_evaluate`` calls it ``boolean`` and returns
  ``{"type": "boolean", "probability": 0.83}``.

JEF accepts both spellings and echoes back whichever the caller used, so the
official TypeSafe SDK and the AI SDK can both point at a JEF server unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

from .errors import InvalidQuestionError

__all__ = [
    "Answer",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "EvaluateRequest",
    "EvaluateResponse",
    "NormalizedQuestion",
    "NoulAnswer",
    "NoulQuestion",
    "Question",
    "RawQuestion",
    "ScoreAnswer",
    "ScoreQuestion",
    "Usage",
    "coerce_question",
    "normalize",
]

# ``noul`` is the native spelling; ``boolean`` is the AI SDK spelling.
NoulDialect = Literal["noul", "boolean"]

#: An unvalidated question straight off the wire. Typed as ``Mapping[str, Any]``
#: rather than ``Mapping[str, object]`` on purpose: a JSON question is
#: heterogeneous, and the narrower type would reject an ordinary
#: ``{"type": "noul", "instructions": "..."}`` literal because dict is
#: invariant in its value type.
RawQuestion = Mapping[str, Any]

State = str | dict[str, Any] | list[Any]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Questions
# --------------------------------------------------------------------------- #


class ChoiceQuestion(_Base):
    """Pick exactly one of ``criteria``. Values describe each option to the model."""

    type: Literal["choice"]
    instructions: str = Field(min_length=1)
    criteria: dict[str, str | None] = Field(min_length=2)


class ScoreQuestion(_Base):
    """Rate the state against ordered levels, lowest first."""

    type: Literal["score"]
    instructions: str = Field(min_length=1)
    criteria: list[str] = Field(min_length=2)

    @field_validator("criteria")
    @classmethod
    def _distinct_levels(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("score levels must be distinct")
        return v


class NoulCriteria(_Base):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    true_: str | None = Field(default=None, alias="true")
    false_: str | None = Field(default=None, alias="false")


class NoulQuestion(_Base):
    """Yes/no. Returns the probability that the statement holds against the state."""

    type: NoulDialect
    instructions: str = Field(min_length=1)
    criteria: NoulCriteria | None = None


Question = Annotated[
    ChoiceQuestion | ScoreQuestion | NoulQuestion,
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------- #
# Normalized internal form
# --------------------------------------------------------------------------- #

Kind = Literal["choice", "score", "noul"]


class NormalizedQuestion(_Base):
    """A question reduced to "score these N options against the state".

    All three primitives collapse to the same shape, which is what lets a single
    decision head serve them: ``choice`` scores its options, ``score`` scores its
    ordered levels, and ``noul`` scores exactly two -- ``(false, true)`` in that
    order, so ``probabilities[1]`` is always P(true).
    """

    id: str
    kind: Kind
    instructions: str
    option_keys: list[str]
    option_labels: list[str | None]
    dialect: NoulDialect | None = None

    @property
    def n_options(self) -> int:
        return len(self.option_keys)


_QUESTION_ADAPTER: TypeAdapter[Question] = TypeAdapter(Question)


def coerce_question(qid: str, q: Question | RawQuestion) -> Question:
    """Validate a raw wire dict into a typed question.

    Raises :class:`InvalidQuestionError` rather than pydantic's ``ValidationError``
    so every caller -- HTTP server, SDK, scene engine -- maps failures to one
    contract error with a consistent 422.
    """
    if isinstance(q, BaseModel):
        return q
    try:
        return _QUESTION_ADAPTER.validate_python(q)
    except ValidationError as exc:
        raise InvalidQuestionError(f"question {qid!r}: {exc.errors()[0].get('msg', exc)}") from exc


def normalize(qid: str, q: Question | RawQuestion) -> NormalizedQuestion:
    """Reduce any wire question to the shared option-scoring form."""
    q = coerce_question(qid, q)
    if isinstance(q, ChoiceQuestion):
        keys = list(q.criteria.keys())
        if len(keys) < 2:
            raise InvalidQuestionError(f"question {qid!r}: choice needs >= 2 options")
        return NormalizedQuestion(
            id=qid,
            kind="choice",
            instructions=q.instructions,
            option_keys=keys,
            option_labels=[q.criteria[k] for k in keys],
        )

    if isinstance(q, ScoreQuestion):
        return NormalizedQuestion(
            id=qid,
            kind="score",
            instructions=q.instructions,
            option_keys=list(q.criteria),
            option_labels=list(q.criteria),
        )

    crit = q.criteria or NoulCriteria()
    return NormalizedQuestion(
        id=qid,
        kind="noul",
        instructions=q.instructions,
        option_keys=["false", "true"],
        option_labels=[crit.false_, crit.true_],
        dialect=q.type,
    )


# --------------------------------------------------------------------------- #
# Answers
# --------------------------------------------------------------------------- #


class ChoiceAnswer(_Base):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(_Base):
    type: Literal["score"] = "score"
    score: float
    legend: list[str]
    probabilities: dict[str, float]
    confidence: float


class NoulAnswer(_Base):
    """Carries the caller's dialect.

    ``noul`` callers get ``{"type": "noul", "noul": p}``; ``boolean`` callers get
    ``{"type": "boolean", "probability": p}``. ``confidence`` is a JEF extension
    (``|2p - 1|``) so scene gates can reference one field name across all types.
    """

    type: NoulDialect = "noul"
    noul: float | None = None
    probability: float | None = None
    confidence: float

    @property
    def value(self) -> float:
        v = self.noul if self.noul is not None else self.probability
        if v is None:  # pragma: no cover -- constructor always sets one
            raise ValueError("NoulAnswer has neither noul nor probability set")
        return v


Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer


# --------------------------------------------------------------------------- #
# Envelope
# --------------------------------------------------------------------------- #


class Usage(_Base):
    inputTokens: int = 0
    outputTokens: int = 0
    totalTokens: int = 0


class EvaluateRequest(_Base):
    state: State
    questions: dict[str, Question] = Field(min_length=1)
    model: str | None = None


class EvaluateResponse(_Base):
    model: str
    answers: dict[str, Answer]
    usage: Usage
    warnings: list[str] | None = None
