"""Run JEF in-process.

For a SOAR worker that already has the alert in memory, the HTTP hop is pure
overhead: the same engine object serves both, so embedding it skips a
serialisation round trip per decision without changing a single answer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from jef_core import BilinearHead, Calibrator, Engine, EvaluateResponse, ZeroShotHead
from jef_core.backends import load_backbone
from jef_core.head import DecisionHead
from jef_scene import Scene, SceneEngine, SceneRegistry, SceneTrace, load_scene

log = logging.getLogger("jef.sdk")

__all__ = ["Jef"]


class Jef:
    """An in-process JEF: evaluate questions, or run scenes.

    Example:
        >>> from jef_sdk import Jef, choice, noul
        >>> jef = Jef()                       # deterministic test backbone
        >>> r = jef.evaluate("付款失敗", {"urgent": noul("是否緊急？")})
        >>> r.answers["urgent"].confidence <= 1.0
        True
    """

    def __init__(
        self,
        backbone: str = "hashing",
        *,
        head: str | Path | DecisionHead | None = None,
        calibration: str | Path | Calibrator | None = None,
        scenes: str | Path | None = None,
        threads: int | None = None,
        alpha: float = 0.10,
        model_name: str | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if threads and backbone != "hashing":
            kwargs["threads"] = threads

        resolved_head: DecisionHead
        if head is None:
            resolved_head = ZeroShotHead()
        elif isinstance(head, (str, Path)):
            resolved_head = BilinearHead.load(head)
        else:
            resolved_head = head

        if calibration is None:
            calibrator = Calibrator()
        elif isinstance(calibration, (str, Path)):
            calibrator = Calibrator.load(calibration)
        else:
            calibrator = calibration

        if not calibrator.is_fitted():
            log.warning(
                "no calibration loaded: p_correct will be unavailable and scene "
                "gates that depend on it cannot fire. This is deliberate -- "
                "automating on an uncalibrated distribution is a coin flip."
            )

        self.engine = Engine(
            load_backbone(backbone, **kwargs),
            resolved_head,
            calibrator,
            model_name=model_name,
            alpha=alpha,
        )
        self.scenes = SceneRegistry()
        self._scene_engine = SceneEngine(self.engine)
        if scenes:
            self.load_scenes(scenes)

    # -- questions ---------------------------------------------------------- #

    def evaluate(self, state: object, questions: dict[str, Any]) -> EvaluateResponse:
        """Evaluate typed questions against one state."""
        return self.engine.evaluate(state, questions)

    # -- scenes ------------------------------------------------------------- #

    def load_scenes(self, directory: str | Path) -> int:
        """Load and compile every scene under ``directory``."""
        return self.scenes.load_dir(directory, compile_with=self._scene_engine)

    def add_scene(self, scene: Scene | str | Path) -> Scene:
        """Register one scene, from an object or a path."""
        resolved = scene if isinstance(scene, Scene) else load_scene(scene)
        self.scenes.add(resolved, compile_with=self._scene_engine)
        return resolved

    def run_scene(self, name: str, state: object) -> SceneTrace:
        """Run a registered scene and return its decision trace."""
        return self._scene_engine.run(self.scenes.get(name), state)

    def run(self, scene: Scene, state: object) -> SceneTrace:
        """Run an unregistered scene object directly, for ad-hoc composition."""
        return self._scene_engine.run(scene, state)

    # -- introspection ------------------------------------------------------ #

    @property
    def model(self) -> str:
        return self.engine.model_name

    @property
    def calibrated(self) -> bool:
        return self.engine.calibrator.is_fitted()

    def __repr__(self) -> str:
        return (
            f"Jef(model={self.model!r}, calibrated={self.calibrated}, scenes={self.scenes.names()})"
        )
