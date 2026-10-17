"""JEF scenes: layered typed questions with calibrated confidence gates.

This is the layer Jev leaves to the caller and every open reimplementation
therefore omits. A scene reads its state once and asks each layer's questions
against that single encoding, so a deep playbook costs one state read.
"""

from __future__ import annotations

from .authz import (
    POLICY_VERSION,
    AuditEntry,
    AuthorizationGrant,
    ExecutionReceipt,
    GrantError,
    PolicyDecision,
    ProposedDecision,
)
from .engine import AnswerView, SceneEngine
from .executor import Executor, Transport, TransportResult
from .expr import ExpressionError, validate_condition
from .loader import SceneRegistry, load_scene, load_scene_text, load_scenes
from .policy import MAX_AUTO_TIER, PolicyVerdict, authorize, tier_of
from .schema import NEXT, Action, Gate, Layer, Scene
from .trace import GateTrace, LayerTrace, QuestionTrace, SceneTrace

__version__ = "0.1.0"

__all__ = [
    "MAX_AUTO_TIER",
    "NEXT",
    "POLICY_VERSION",
    "Action",
    "AnswerView",
    "AuditEntry",
    "AuthorizationGrant",
    "ExecutionReceipt",
    "Executor",
    "ExpressionError",
    "Gate",
    "GateTrace",
    "GrantError",
    "Layer",
    "LayerTrace",
    "PolicyDecision",
    "PolicyVerdict",
    "ProposedDecision",
    "QuestionTrace",
    "Scene",
    "SceneEngine",
    "SceneRegistry",
    "SceneTrace",
    "Transport",
    "TransportResult",
    "__version__",
    "authorize",
    "load_scene",
    "load_scene_text",
    "load_scenes",
    "tier_of",
    "validate_condition",
]
