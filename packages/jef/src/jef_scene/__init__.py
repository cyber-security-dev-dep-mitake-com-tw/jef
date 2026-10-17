"""JEF scenes: layered typed questions with calibrated confidence gates.

This is the layer Jev leaves to the caller and every open reimplementation
therefore omits. A scene reads its state once and asks each layer's questions
against that single encoding, so a deep playbook costs one state read.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

from .engine import AnswerView, SceneEngine
from .executor import Executor, Transport, TransportResult
from .expr import ExpressionError, validate_condition
from .loader import SceneRegistry, load_scene, load_scene_text, load_scenes
from .policy import MAX_AUTO_TIER, PolicyVerdict, authorize, tier_of
from .schema import NEXT, Action, Gate, Layer, Scene
from .store import MemoryStore, ReceiptStore, SQLiteStore, StoreUnavailable, open_store
from .trace import GateTrace, LayerTrace, QuestionTrace, SceneTrace

try:
    __version__ = _dist_version("jef")
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0+unknown"

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
    "MemoryStore",
    "PolicyDecision",
    "PolicyVerdict",
    "ProposedDecision",
    "QuestionTrace",
    "ReceiptStore",
    "SQLiteStore",
    "Scene",
    "SceneEngine",
    "SceneRegistry",
    "SceneTrace",
    "StoreUnavailable",
    "Transport",
    "TransportResult",
    "__version__",
    "authorize",
    "load_scene",
    "load_scene_text",
    "load_scenes",
    "open_store",
    "tier_of",
    "validate_condition",
]
