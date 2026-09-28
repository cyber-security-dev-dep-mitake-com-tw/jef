"""JEF core -- an open System One decision engine.

Evaluate typed questions (``choice`` / ``score`` / ``noul``) against a shared
state and get back probability distributions with calibrated confidence. No text
generation, no parsing.

    >>> from jef_core import Engine
    >>> engine = Engine()  # deterministic test backbone
    >>> r = engine.evaluate(
    ...     "付款服務連續三天失敗",
    ...     {"urgent": {"type": "noul", "instructions": "這是否緊急？"}},
    ... )
    >>> engine.encode_count  # the state is read exactly once
    1
"""

from __future__ import annotations

from .artifacts import is_hf_uri, resolve_artifact
from .backbone import Backbone, InstrumentedBackbone, StateEncoding
from .backends import load_backbone
from .calibration import Bucket, Calibrator
from .engine import Engine, SharedState, render_state
from .errors import (
    BackendUnavailableError,
    InvalidQuestionError,
    InvalidStateError,
    JefError,
    SceneError,
)
from .head import BilinearHead, DecisionHead, ZeroShotHead, attention_pool
from .mathx import confidence, expected_calibration_error, score_expectation, softmax
from .types import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    EvaluateRequest,
    EvaluateResponse,
    NormalizedQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    RawQuestion,
    ScoreAnswer,
    ScoreQuestion,
    Usage,
    coerce_question,
    normalize,
)

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "Backbone",
    "BackendUnavailableError",
    "BilinearHead",
    "Bucket",
    "Calibrator",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "DecisionHead",
    "Engine",
    "EvaluateRequest",
    "EvaluateResponse",
    "InstrumentedBackbone",
    "InvalidQuestionError",
    "InvalidStateError",
    "JefError",
    "NormalizedQuestion",
    "NoulAnswer",
    "NoulQuestion",
    "Question",
    "RawQuestion",
    "SceneError",
    "ScoreAnswer",
    "ScoreQuestion",
    "SharedState",
    "StateEncoding",
    "Usage",
    "ZeroShotHead",
    "__version__",
    "attention_pool",
    "coerce_question",
    "confidence",
    "expected_calibration_error",
    "is_hf_uri",
    "load_backbone",
    "normalize",
    "render_state",
    "resolve_artifact",
    "score_expectation",
    "softmax",
]
