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

__all__ = ["NEXT", "Action", "Advisory", "Gate", "Layer", "Scene"]

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
    #: Action tier, T0 (read-only) to T4 (destructive or wide-blast). When unset
    #: the runtime policy infers it from the name (``close`` is T2, ``isolate``
    #: T3, anything else T1). Only T0 and T1 may run without a human, so a
    #: scene that wants to automate an action whose name sounds disruptive
    #: must say so here, in the YAML, where a reviewer sees it.
    tier: int | None = Field(default=None, ge=0, le=4)
    #: Arbitrary payload for the caller's own dispatcher (queue names, severities).
    params: dict[str, Any] = Field(default_factory=dict)


class Advisory(_Base):
    """What a gate may still select when the evidence it read is not validated.

    A gate that fires on an answer whose calibration is ``legacy``, ``missing``
    or ``insufficient`` has measured nothing it can stand behind, so the engine
    does not let it *recommend* anything beyond these actions: whatever else it
    selected is withheld and ``unvalidated_fallback`` is proposed instead. The
    allowlist is by action key, never by name pattern, so renaming
    ``close_pending_review`` to ``normal_maintenance`` does not get past it.
    """

    #: Proposed in place of a withheld action. Defaults to ``scene.fallthrough``;
    #: with neither, a withheld gate proposes no action at all.
    unvalidated_fallback: str | None = None
    #: Actions a gate on unvalidated evidence may still select. Each must be a
    #: human-review action of tier 1 or below. Defaults to the fallback alone.
    allowed_when_unvalidated: list[str] = Field(default_factory=list)
    #: Questions asked only so the trace records the model's view, with no gate
    #: reading them -- typically because no gate on them is safe until their
    #: task is calibrated. Declared so the validator does not flag them.
    observe_only: list[str] = Field(default_factory=list)


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
    #: Fail-closed handling of gates that fire on unvalidated calibration.
    advisory: Advisory = Field(default_factory=Advisory)

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
        self._check_advisory()
        return self

    def _check_advisory(self) -> None:
        asked = {qid for layer in self.layers for qid in layer.questions}
        for qid in self.advisory.observe_only:
            if qid not in asked:
                raise ValueError(f"advisory.observe_only names {qid!r}, which no layer asks")
        # Only names the author wrote under ``advisory`` are checked here. The
        # implicit fallback (``fallthrough``) may be any action; when it is not
        # a safe one, ``unvalidated_fallback`` is None and a withheld gate
        # proposes nothing, which is the honest answer when nothing measured
        # supports a choice.
        names = set(self.advisory.allowed_when_unvalidated)
        if self.advisory.unvalidated_fallback is not None:
            names.add(self.advisory.unvalidated_fallback)
        for name in sorted(names):
            action = self.actions.get(name)
            if action is None:
                raise ValueError(f"advisory names {name!r}, which is not a declared action")
            if not _is_advisory_safe(name, action):
                raise ValueError(
                    f"advisory action {name!r} must be human_review: true and tier 1 or "
                    "below: it is what a gate on unvalidated evidence may still propose"
                )

    @property
    def unvalidated_fallback(self) -> str | None:
        if self.advisory.unvalidated_fallback is not None:
            return self.advisory.unvalidated_fallback
        if self.fallthrough is not None and _is_advisory_safe(
            self.fallthrough, self.actions[self.fallthrough]
        ):
            return self.fallthrough
        return None

    @property
    def allowed_when_unvalidated(self) -> frozenset[str]:
        fallback = self.unvalidated_fallback
        allowed = set(self.advisory.allowed_when_unvalidated)
        if fallback is not None:
            allowed.add(fallback)
        return frozenset(allowed)

    @property
    def question_count(self) -> int:
        return sum(len(layer.questions) for layer in self.layers)


def _is_advisory_safe(name: str, action: Action) -> bool:
    """A human-review action of tier 1 or below: safe to propose on no evidence."""
    # Imported here: policy imports this module for its type hints.
    from .policy import tier_of

    return action.human_review and tier_of(name, action) <= 1


Verdict = Literal["decided", "fallthrough", "withheld"]
