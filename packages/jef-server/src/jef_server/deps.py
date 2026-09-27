"""Engine construction and process-wide lifecycle.

The backbone is loaded once at startup, never per request: loading 300M frozen
parameters per call would be absurd, and the whole design assumes the encoder is
a long-lived, shared, read-only resource.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from jef_core import BilinearHead, Calibrator, Engine, ZeroShotHead
from jef_core.backends import load_backbone
from jef_core.head import DecisionHead
from jef_scene import SceneEngine, SceneRegistry

from .settings import Settings, load_settings

log = logging.getLogger("jef.server")

__all__ = ["build_engine", "build_scenes", "get_engine", "get_scenes", "get_settings"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def build_engine(settings: Settings) -> Engine:
    """Assemble backbone + head + calibrator from configuration."""
    kwargs: dict[str, object] = {}
    if settings.threads and not settings.is_test_backbone:
        kwargs["threads"] = settings.threads
    backbone = load_backbone(settings.backbone, **kwargs)

    head: DecisionHead = (
        BilinearHead.load(settings.head_path) if settings.head_path else ZeroShotHead()
    )
    calibrator = (
        Calibrator.load(settings.calibration_path) if settings.calibration_path else Calibrator()
    )

    if not calibrator.is_fitted():
        log.warning(
            "no calibration loaded: probabilities are uncalibrated and confidence "
            "reflects distribution concentration only -- do not gate automated "
            "actions on it (set JEF_CALIBRATION_PATH)"
        )
    if settings.is_test_backbone:
        log.warning(
            "JEF_BACKBONE=hashing: deterministic but semantically meaningless "
            "vectors. Set a model id for anything real."
        )

    return Engine(
        backbone,
        head,
        calibrator,
        model_name=settings.model_name,
        alpha=settings.alpha,
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return build_engine(get_settings())


def build_scenes(settings: Settings, engine: Engine) -> tuple[SceneEngine, SceneRegistry]:
    """Load and compile every scene at startup.

    Compiling here rather than lazily means a scene with a bad gate condition
    fails the deployment, not an incident. A scene directory that does not exist
    is a configuration error worth failing on for the same reason -- silently
    serving zero scenes looks identical to serving the wrong ones.
    """
    scene_engine = SceneEngine(engine)
    registry = SceneRegistry()
    if settings.scenes_dir:
        count = registry.load_dir(settings.scenes_dir, compile_with=scene_engine)
        log.info("loaded %d scene(s) from %s", count, settings.scenes_dir)
    return scene_engine, registry


@lru_cache(maxsize=1)
def get_scenes() -> tuple[SceneEngine, SceneRegistry]:
    return build_scenes(get_settings(), get_engine())
