"""JEF scenes: layered typed questions with calibrated confidence gates.

This is the layer Jev leaves to the caller and every open reimplementation
therefore omits. A scene reads its state once and asks each layer's questions
against that single encoding, so a deep playbook costs one state read.
"""

from __future__ import annotations

from .engine import AnswerView, SceneEngine
from .expr import ExpressionError, validate_condition
from .loader import SceneRegistry, load_scene, load_scene_text, load_scenes
from .schema import NEXT, Action, Gate, Layer, Scene
from .trace import GateTrace, LayerTrace, QuestionTrace, SceneTrace

__version__ = "0.1.0"

__all__ = [
    "NEXT",
    "Action",
    "AnswerView",
    "ExpressionError",
    "Gate",
    "GateTrace",
    "Layer",
    "LayerTrace",
    "QuestionTrace",
    "Scene",
    "SceneEngine",
    "SceneRegistry",
    "SceneTrace",
    "__version__",
    "load_scene",
    "load_scene_text",
    "load_scenes",
    "validate_condition",
]
