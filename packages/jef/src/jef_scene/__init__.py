"""JEF scenes: layered typed questions with calibrated confidence gates.

This is the layer Jev leaves to the caller and every open reimplementation
therefore omits. A scene reads its state once and asks each layer's questions
against that single encoding, so a deep playbook costs one state read.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

from .engine import AnswerView, SceneEngine
from .expr import ExpressionError, validate_condition
from .loader import SceneRegistry, load_scene, load_scene_text, load_scenes
from .policy import MAX_AUTO_TIER, PolicyVerdict, authorize, tier_of
from .schema import NEXT, Action, Gate, Layer, Scene
from .trace import GateTrace, LayerTrace, QuestionTrace, SceneTrace

try:
    __version__ = _dist_version("jef")
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "MAX_AUTO_TIER",
    "NEXT",
    "Action",
    "AnswerView",
    "ExpressionError",
    "Gate",
    "GateTrace",
    "Layer",
    "LayerTrace",
    "PolicyVerdict",
    "QuestionTrace",
    "Scene",
    "SceneEngine",
    "SceneRegistry",
    "SceneTrace",
    "__version__",
    "authorize",
    "load_scene",
    "load_scene_text",
    "load_scenes",
    "tier_of",
    "validate_condition",
]
