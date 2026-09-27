"""The JEF engine: evaluate typed questions against one shared state encoding.

The whole point of a System One model is that the state is read once and every
question is answered independently against it, in parallel and in isolation.
:meth:`Engine.prepare` does the expensive part exactly once; :meth:`Engine.answer`
is cheap and may be called repeatedly -- which is what lets a multi-layer scene
run 20 gates for the price of one state encode (D3).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from .backbone import Backbone, InstrumentedBackbone, StateEncoding
from .backends import load_backbone
from .calibration import Calibrator
from .errors import InvalidStateError
from .head import DecisionHead, ZeroShotHead, build_query_texts
from .mathx import confidence as confidence_of
from .mathx import score_expectation
from .types import (
    Answer,
    ChoiceAnswer,
    EvaluateResponse,
    NormalizedQuestion,
    NoulAnswer,
    Question,
    RawQuestion,
    ScoreAnswer,
    Usage,
    normalize,
)

__all__ = ["DEFAULT_BACKBONE", "Engine", "SharedState", "render_state"]

#: Engine() with no arguments must never silently download 300M parameters, so
#: the default is the deterministic test backbone. Production callers pass a
#: model id explicitly (see jef_core.backends.mmbert.DEFAULT_MODEL_ID).
DEFAULT_BACKBONE = "hashing"

_PROB_DECIMALS = 6


def render_state(state: object) -> str:
    """Flatten an accepted state shape to the text the backbone sees.

    A JSON array is *one shared state*, not a batch of independent inputs -- the
    same distinction TypeSafe draws. Rendering it as a single document preserves
    that: every question sees the whole array.
    """
    if isinstance(state, str):
        return state
    if isinstance(state, (dict, list)):
        return json.dumps(state, ensure_ascii=False, indent=2, sort_keys=False)
    raise InvalidStateError(f"state must be str, object or array, got {type(state).__name__}")


@dataclass
class SharedState:
    """One state, encoded once, reusable by any number of questions."""

    text: str
    encoding: StateEncoding
    n_tokens: int = 0
    meta: dict[str, object] = field(default_factory=dict)


class Engine:
    """Stateless evaluator over a frozen backbone, a head, and a calibrator."""

    def __init__(
        self,
        backbone: Backbone | str | None = None,
        head: DecisionHead | None = None,
        calibrator: Calibrator | None = None,
        *,
        model_name: str | None = None,
        alpha: float = 0.10,
    ) -> None:
        if backbone is None:
            raw: Backbone = load_backbone(DEFAULT_BACKBONE)
        elif isinstance(backbone, str):
            raw = load_backbone(backbone)
        else:
            raw = backbone
        # Always instrumented: the D3 guarantee is only real if tests can see it.
        self.backbone = InstrumentedBackbone(raw)
        self.head: DecisionHead = head or ZeroShotHead()
        self.calibrator = calibrator or Calibrator()
        self.alpha = alpha
        self.model_name = model_name or f"jef/{self.backbone.name}+{self.head.name}"

    # -- instrumentation ---------------------------------------------------- #

    @property
    def encode_count(self) -> int:
        """Number of full state encodes performed since the last reset."""
        return self.backbone.encode_count

    def reset_counters(self) -> None:
        self.backbone.reset_counters()

    # -- the two-phase API -------------------------------------------------- #

    def prepare(self, state: object) -> SharedState:
        """Encode the state. This is the only expensive call in the pipeline."""
        text = render_state(state)
        encoding = self.backbone.encode_state(text)
        return SharedState(text=text, encoding=encoding, n_tokens=encoding.n_tokens)

    def answer_one(self, shared: SharedState, q: NormalizedQuestion) -> Answer:
        """Answer a single normalized question against an already-encoded state."""
        queries = self.backbone.encode_queries(build_query_texts(q))
        logits = self.head.logits(shared.encoding, q, queries)
        probs = self.calibrator.apply(logits, q.kind, q.n_options)
        return _build_answer(q, probs)

    def answer(
        self, shared: SharedState, questions: Mapping[str, Question | RawQuestion]
    ) -> dict[str, Answer]:
        """Answer every question against the same encoding, independently.

        No question can influence another: each one only ever reads ``shared``.
        """
        out: dict[str, Answer] = {}
        for qid, q in questions.items():
            out[qid] = self.answer_one(shared, normalize(qid, q))
        return out

    def evaluate(
        self, state: object, questions: Mapping[str, Question | RawQuestion]
    ) -> EvaluateResponse:
        """Full ``/v1/systemone`` evaluation: one state encode, N answers."""
        shared = self.prepare(state)
        answers = self.answer(shared, questions)
        question_tokens = sum(
            self.backbone.count_tokens(t)
            for qid, q in questions.items()
            for t in build_query_texts(normalize(qid, q))
        )
        input_tokens = shared.n_tokens + question_tokens
        return EvaluateResponse(
            model=self.model_name,
            answers=answers,
            usage=Usage(
                inputTokens=input_tokens,
                # A System One model generates nothing. This is structurally 0.
                outputTokens=0,
                totalTokens=input_tokens,
            ),
            warnings=self._warnings(),
        )

    def _warnings(self) -> list[str] | None:
        w: list[str] = []
        if self.backbone.name == "hashing":
            w.append(
                "backbone='hashing' produces deterministic but semantically "
                "meaningless vectors; for testing only"
            )
        if not self.calibrator.is_fitted():
            w.append(
                "calibrator is unfitted: probabilities are uncalibrated and "
                "confidence reflects distribution concentration only"
            )
        return w or None


def _round(x: float) -> float:
    return round(float(x), _PROB_DECIMALS)


def _build_answer(q: NormalizedQuestion, probs: np.ndarray) -> Answer:
    """Project one calibrated distribution into the answer type the caller asked for."""
    conf = confidence_of(probs)
    dist = {k: _round(p) for k, p in zip(q.option_keys, probs, strict=True)}

    if q.kind == "choice":
        return ChoiceAnswer(
            choice=q.option_keys[int(np.argmax(probs))],
            probabilities=dist,
            confidence=_round(conf),
        )

    if q.kind == "score":
        # Expectation over ordered levels, so the score may land between levels.
        return ScoreAnswer(
            score=_round(score_expectation(probs)),
            legend=list(q.option_keys),
            probabilities=dist,
            confidence=_round(conf),
        )

    # noul: option order is (false, true), so index 1 is P(true).
    p_true = _round(float(probs[1]))
    dialect = q.dialect or "noul"
    if dialect == "boolean":
        return NoulAnswer(type="boolean", probability=p_true, confidence=_round(conf))
    return NoulAnswer(type="noul", noul=p_true, confidence=_round(conf))
