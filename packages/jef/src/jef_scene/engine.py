"""Run a scene against one state.

The load-bearing property (D3): :meth:`SceneEngine.run` calls
``Engine.prepare`` exactly once and every layer reuses that encoding. A
twenty-gate playbook therefore costs one state read, not twenty, and the tests
assert ``encode_count == 1`` across a full multi-layer run.

Layers after the deciding one are never evaluated -- their questions are never
asked and cost nothing -- which is the other half of why a deep playbook is
cheap.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from jef_core import Engine
from jef_core.errors import SceneError
from jef_core.types import (
    Answer,
    ChoiceAnswer,
    NormalizedQuestion,
    NoulAnswer,
    ScoreAnswer,
    normalize,
)

from .expr import ExpressionError, compile_condition, evaluate_condition
from .schema import NEXT, Scene
from .trace import GateTrace, LayerTrace, QuestionTrace, SceneTrace

log = logging.getLogger("jef.scene")

__all__ = ["AnswerView", "SceneEngine"]


class AnswerView:
    """What a gate condition can see of one answer.

    A thin, explicit surface rather than the pydantic model itself: gates should
    not be able to reach into internals, and every readable name here is also in
    the expression whitelist, so the two cannot drift apart silently.
    """

    __slots__ = (
        "choice",
        "confidence",
        "noul",
        "p_correct",
        "prediction_set",
        "probabilities",
        "probability",
        "score",
        "type",
    )

    def __init__(
        self,
        answer: Answer,
        question: NormalizedQuestion,
        p_correct: float | None,
        prediction_set: list[str],
    ) -> None:
        self.type = answer.type
        self.confidence = answer.confidence
        self.p_correct = p_correct
        self.prediction_set = prediction_set
        self.choice = getattr(answer, "choice", None)
        self.score = getattr(answer, "score", None)
        self.probabilities = dict(getattr(answer, "probabilities", {}) or {})
        if isinstance(answer, NoulAnswer):
            # Both spellings are readable so a gate does not have to know which
            # dialect the question was written in.
            self.probability: float | None = answer.value
            self.noul: float | None = answer.value
            self.probabilities = {
                "false": round(1.0 - answer.value, 6),
                "true": round(answer.value, 6),
            }
        else:
            self.probability = None
            self.noul = None

    @property
    def set_size(self) -> int:
        return len(self.prediction_set)


class SceneEngine:
    """Evaluates scenes. One instance per process; scenes are data."""

    def __init__(self, engine: Engine | None = None) -> None:
        self.engine = engine or Engine()
        self._compiled: dict[tuple[str, int, str, int], Any] = {}

    # -- compilation -------------------------------------------------------- #

    def compile(self, scene: Scene) -> None:
        """Compile every gate condition up front.

        Called by the loader so a scene with a bad condition fails when it is
        loaded, not while an incident is being triaged.

        A layer's gates may read any question answered so far, so the name set
        accumulates: layer three can combine its own answer with one from layer
        one, which is how a real playbook reasons.
        """
        names: set[str] = set()
        for layer in scene.layers:
            names |= set(layer.questions)
            for i, gate in enumerate(layer.gates):
                if gate.when is None:
                    continue
                key = (scene.scene, scene.version, layer.id, i)
                self._compiled[key] = compile_condition(gate.when, names)

    def _condition(
        self, scene: Scene, layer_id: str, index: int, source: str, names: set[str]
    ) -> Any:
        key = (scene.scene, scene.version, layer_id, index)
        if key not in self._compiled:
            self._compiled[key] = compile_condition(source, names)
        return self._compiled[key]

    # -- execution ---------------------------------------------------------- #

    def run(self, scene: Scene, state: object) -> SceneTrace:
        """Evaluate ``scene`` against ``state`` and return the full trace."""
        engine = self.engine
        engine.reset_counters()

        # The one expensive call in the whole run. Everything below reuses it.
        shared = engine.prepare(state)

        layer_traces: list[LayerTrace] = []
        decided: str | None = None
        questions_asked = 0
        available: dict[str, AnswerView] = {}

        for position, layer in enumerate(scene.layers):
            answers = engine.answer(shared, layer.questions)
            questions_asked += len(answers)

            views: dict[str, AnswerView] = {}
            question_traces: list[QuestionTrace] = []
            for qid, raw in layer.questions.items():
                question = normalize(qid, raw)
                answer = answers[qid]
                p_correct = engine.calibrator.p_correct(
                    answer.confidence, question.kind, question.n_options
                )
                pset = self._prediction_set(answer, question, scene.alpha)
                views[qid] = AnswerView(answer, question, p_correct, pset)
                question_traces.append(
                    QuestionTrace(
                        id=qid,
                        kind=question.kind,
                        instructions=question.instructions,
                        allowed_answers=list(question.option_keys),
                        answer=answer.model_dump(exclude_none=True),
                        confidence=answer.confidence,
                        p_correct=p_correct,
                        prediction_set=pset,
                    )
                )

            # Answers accumulate: a later gate may combine its own layer's
            # answer with an earlier one.
            available.update(views)
            gate_traces, outcome = self._run_gates(scene, layer, available)
            layer_traces.append(
                LayerTrace(
                    id=layer.id,
                    description=layer.description,
                    questions=question_traces,
                    gates=gate_traces,
                    outcome=outcome,
                )
            )

            if outcome != NEXT:
                decided = outcome
                skipped = [layer.id for layer in scene.layers[position + 1 :]]
                break
        else:
            skipped = []

        if decided is None:
            decided = scene.fallthrough
            verdict = "fallthrough"
        else:
            verdict = "decided"

        action = scene.actions.get(decided) if decided else None
        warnings = list(engine._warnings() or [])
        if action is None and decided is not None:  # pragma: no cover -- schema-checked
            raise SceneError(f"scene {scene.scene!r} selected unknown action {decided!r}")

        return SceneTrace(
            scene=scene.scene,
            version=scene.version,
            model=engine.model_name,
            verdict=verdict,
            action=decided,
            human_review=bool(action and action.human_review),
            action_params=dict(action.params) if action else {},
            layers=layer_traces,
            layers_skipped=skipped,
            state_encodes=engine.encode_count,
            questions_asked=questions_asked,
            calibrated=engine.calibrator.is_fitted(),
            warnings=warnings,
        )

    # -- internals ---------------------------------------------------------- #

    def _run_gates(
        self, scene: Scene, layer: Any, available: dict[str, AnswerView]
    ) -> tuple[list[GateTrace], str]:
        """Evaluate a layer's gates against every answer produced so far."""
        traces: list[GateTrace] = []
        names = set(available)
        namespace: dict[str, Any] = dict(available)

        for i, gate in enumerate(layer.gates):
            if gate.else_:
                traces.append(
                    GateTrace(
                        index=i,
                        condition=None,
                        is_else=True,
                        fired=True,
                        then=gate.then,
                        because=gate.because,
                    )
                )
                return traces, gate.then

            assert gate.when is not None  # noqa: S101 -- schema guarantees this
            try:
                code = self._condition(scene, layer.id, i, gate.when, names)
                fired = evaluate_condition(code, namespace)
            except ExpressionError as exc:
                # A broken gate must not silently behave like a false one: that
                # would turn a typo into a policy change nobody approved.
                traces.append(
                    GateTrace(
                        index=i,
                        condition=gate.when,
                        is_else=False,
                        fired=False,
                        because=gate.because,
                        error=str(exc),
                    )
                )
                raise

            traces.append(
                GateTrace(
                    index=i,
                    condition=gate.when,
                    is_else=False,
                    fired=fired,
                    then=gate.then if fired else None,
                    because=gate.because,
                )
            )
            if fired:
                return traces, gate.then

        return traces, NEXT

    def _prediction_set(
        self, answer: Answer, question: NormalizedQuestion, alpha: float
    ) -> list[str]:
        calibrator = self.engine.calibrator
        if not calibrator.is_fitted():
            # Without a fitted quantile the honest conformal set is the full set
            # of options: coverage 1.0 requires excluding nothing. Returning an
            # empty list instead would make `set_size` read as 0, so a gate like
            # `owner.set_size <= 2` would fire on an uncalibrated deployment and
            # automate a decision nobody could stand behind.
            return list(question.option_keys)
        probs = np.asarray(self._distribution(answer, question), dtype=np.float64)
        indices = calibrator.prediction_set(probs, question.kind, question.n_options, alpha)
        return [question.option_keys[i] for i in indices]

    @staticmethod
    def _distribution(answer: Answer, question: NormalizedQuestion) -> list[float]:
        if isinstance(answer, NoulAnswer):
            return [1.0 - answer.value, answer.value]
        if isinstance(answer, (ChoiceAnswer, ScoreAnswer)):
            return [answer.probabilities[k] for k in question.option_keys]
        raise SceneError(f"unsupported answer type {type(answer).__name__}")
