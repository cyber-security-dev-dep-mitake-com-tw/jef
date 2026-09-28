"""Scene definitions: layered typed questions with confidence gates.

A scene is the layer Jev deliberately leaves to the caller -- *"Your application
defines the possible answers and uses the results in its logic"* -- and which
every open reimplementation therefore also omits. In a SOAR playbook that logic
is layers of gated decisions, and today it is hand-written if/else inside Splunk
SOAR, Shuffle or TheHive.

The shape is deliberately small:

    layers: ordered. Each asks a set of typed questions and evaluates its gates
            in order; the first matching gate decides what happens.
    gates:  `when` is an expression over every answer produced so far -- this
            layer's and any earlier layer's -- and `then` is either `next` or
            the name of a terminal action.

Everything in a scene sees the same state, encoded exactly once for the whole
run. That is the entire performance argument: a twenty-gate playbook costs one
state read, not twenty.
"""

from __future__ import annotations

from typing import Any, Literal

from jef_core.types import Question
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["NEXT", "Action", "Gate", "Layer", "Scene"]

#: Reserved ``then`` value meaning "fall through to the next layer".
NEXT = "next"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Gate(_Base):
    """One branch of a layer's decision.

    Exactly one of ``when`` or ``else_`` is set. An ``else`` gate always fires
    and must therefore come last -- a scene with an unreachable gate is a bug
    the author wants to hear about at load time, not at 3am.
    """

    when: str | None = None
    else_: bool = Field(default=False, alias="else")
    then: str
    #: Free-form note carried into the decision trace, for the human who reads it.
    because: str | None = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def _exactly_one_condition(self) -> Gate:
        if bool(self.when) == bool(self.else_):
            raise ValueError("a gate needs exactly one of 'when' or 'else'")
        return self


class Layer(_Base):
    id: str = Field(min_length=1)
    description: str | None = None
    questions: dict[str, Question] = Field(min_length=1)
    gates: list[Gate] = Field(min_length=1)

    @model_validator(mode="after")
    def _else_comes_last(self) -> Layer:
        for gate in self.gates[:-1]:
            if gate.else_:
                raise ValueError(
                    f"layer {self.id!r}: an 'else' gate makes every later gate "
                    "unreachable, so it must be last"
                )
        return self


class Action(_Base):
    """A terminal outcome a gate can select."""

    description: str | None = None
    #: Whether reaching this action means a human must look at the case. Kept
    #: explicit rather than inferred from the name so the trace can be filtered
    #: on it and so a scene cannot quietly stop escalating.
    human_review: bool = False
    #: Arbitrary payload for the caller's own dispatcher (queue names, severities).
    params: dict[str, Any] = Field(default_factory=dict)


class Scene(_Base):
    scene: str = Field(min_length=1)
    version: int = 1
    description: str | None = None
    layers: list[Layer] = Field(min_length=1)
    actions: dict[str, Action] = Field(min_length=1)
    #: Conformal miscoverage level used by ``prediction_set`` in expressions.
    alpha: float = Field(default=0.10, gt=0.0, lt=1.0)
    #: What happens when every layer falls through without a terminal action.
    #: Defaults to escalating rather than silently doing nothing, because a
    #: playbook that runs off the end has not made a decision.
    fallthrough: str | None = None

    @model_validator(mode="after")
    def _targets_resolve(self) -> Scene:
        known = set(self.actions) | {NEXT}
        for layer in self.layers:
            for gate in layer.gates:
                if gate.then not in known:
                    raise ValueError(
                        f"layer {layer.id!r}: gate targets {gate.then!r}, which is "
                        f"neither an action nor {NEXT!r}"
                    )
        if self.fallthrough is not None and self.fallthrough not in self.actions:
            raise ValueError(f"fallthrough {self.fallthrough!r} is not a declared action")

        ids = [layer.id for layer in self.layers]
        if len(set(ids)) != len(ids):
            raise ValueError("layer ids must be unique")

        # Question ids are unique across the whole scene, not just within a
        # layer, because a gate may reference any question answered so far.
        # Without this, `owner` in layer three would be ambiguous.
        seen: dict[str, str] = {}
        for layer in self.layers:
            for qid in layer.questions:
                if qid in seen:
                    raise ValueError(
                        f"question id {qid!r} appears in both layer {seen[qid]!r} and "
                        f"layer {layer.id!r}; ids must be unique across a scene so a "
                        "gate can reference an earlier layer's answer unambiguously"
                    )
                seen[qid] = layer.id

        last = self.layers[-1]
        if self.fallthrough is None and not any(g.else_ for g in last.gates):
            raise ValueError(
                f"the final layer {last.id!r} can fall through without deciding "
                "anything; give it an 'else' gate or set scene.fallthrough"
            )
        return self

    @property
    def question_count(self) -> int:
        return sum(len(layer.questions) for layer in self.layers)


Verdict = Literal["decided", "fallthrough"]
