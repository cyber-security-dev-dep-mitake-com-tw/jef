"""Decision traces.

A SOAR audit does not ask "what did the model say"; it asks what evidence was
read, what was asked of it, what answers were permitted, how the probability
mass fell, and which gate fired as a result. Existing tooling cannot answer that
because the decision lives in hand-written branching that keeps no record.

Every scene run emits one of these, whether it decided or fell through.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["GateTrace", "LayerTrace", "QuestionTrace", "SceneTrace"]


@dataclass
class QuestionTrace:
    """One question, its permitted answers, and where the mass landed."""

    id: str
    kind: str
    instructions: str
    allowed_answers: list[str]
    answer: dict[str, Any]
    confidence: float
    #: Calibrated probability the answer is right. ``None`` means no correctness
    #: map was fitted for this bucket -- which a reviewer needs to see, because
    #: it is the difference between "unlikely" and "unknown".
    p_correct: float | None = None
    #: Conformal prediction set at the scene's alpha. More than one entry means
    #: the model could not separate the candidates at the promised coverage.
    #: On an uncalibrated deployment this is every option -- the only set that
    #: honestly covers at any level without a fitted quantile.
    prediction_set: list[str] = field(default_factory=list)


@dataclass
class GateTrace:
    index: int
    condition: str | None
    is_else: bool
    fired: bool
    then: str | None = None
    because: str | None = None
    #: Present when the condition could not be evaluated.
    error: str | None = None


@dataclass
class LayerTrace:
    id: str
    description: str | None
    questions: list[QuestionTrace]
    gates: list[GateTrace]
    outcome: str  # 'next' or an action name


@dataclass
class SceneTrace:
    scene: str
    version: int
    model: str
    #: 'decided' when a gate selected an action, 'fallthrough' otherwise.
    verdict: str
    action: str | None
    human_review: bool
    action_params: dict[str, Any]
    layers: list[LayerTrace]
    #: Layers after the deciding one are never evaluated. Recorded so a reader
    #: can tell "not asked" from "asked and inconclusive".
    layers_skipped: list[str]
    state_encodes: int
    questions_asked: int
    calibrated: bool
    warnings: list[str] = field(default_factory=list)
    #: What the scene selected. Always equal to ``action``; named separately so
    #: a reader never confuses "the model proposed close" with "close happened".
    proposed_action: str | None = None
    #: ``allow`` or ``review_required``, from :mod:`jef_scene.policy`. When
    #: ``review_required``, ``human_review`` is true regardless of what the
    #: action declared. Nothing downstream may flip it back.
    policy_verdict: str = "review_required"
    #: Tier the policy applied (explicit on the action, or inferred from its name).
    action_tier: int | None = None
    #: Why the verdict is what it is; empty when allowed.
    reason_codes: list[str] = field(default_factory=list)
    #: The scene engine executes nothing. These exist so the trace shape already
    #: has the slot a connector's receipt will fill, and so no reader mistakes
    #: a proposal for an outcome.
    executed_action: str | None = None
    execution_status: str = "not_dispatched"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
