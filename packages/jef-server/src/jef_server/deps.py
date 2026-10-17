"""Engine construction and process-wide lifecycle.

The backbone is loaded once at startup, never per request: loading 300M frozen
parameters per call would be absurd, and the whole design assumes the encoder is
a long-lived, shared, read-only resource.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from jef_core import BilinearHead, Calibrator, Engine, ZeroShotHead
from jef_core.backends import load_backbone
from jef_core.calibration import file_sha256
from jef_core.head import DecisionHead
from jef_core.model_manifest import Verification, check_model_dir
from jef_scene import SceneEngine, SceneRegistry

from .settings import Settings, load_settings

log = logging.getLogger("jef.server")

__all__ = [
    "build_engine",
    "build_scenes",
    "get_engine",
    "get_scenes",
    "get_settings",
    "manifest_check",
]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def manifest_check(settings: Settings) -> Verification:
    """Verify the loaded artifacts against their manifest, if there is one.

    The manifest is ``JEF_MODEL_MANIFEST`` or ``manifest.json`` beside the head.
    No head path means there is nothing to verify (``absent``).
    """
    if not settings.head_path:
        return Verification("absent")
    head = Path(settings.head_path)
    explicit = Path(settings.model_manifest) if settings.model_manifest else None
    if explicit is not None and not explicit.is_file():
        return Verification("mismatch", [f"JEF_MODEL_MANIFEST={explicit} does not exist"])
    return check_model_dir(head.parent, explicit)


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
    verification = manifest_check(settings)
    if verification.status == "mismatch":
        # Fail closed. Numbers fitted for other bytes are not calibration: drop
        # them, so every question reports `missing` and no gate automates.
        log.error(
            "model manifest mismatch; serving WITHOUT calibration: %s",
            "; ".join(verification.problems),
        )
        calibrator = Calibrator()
    if settings.head_path and Path(settings.head_path).is_file():
        # So a calibration fitted for another head is seen as `incompatible`
        # rather than silently applied.
        calibrator.bind_model(file_sha256(settings.head_path))
    if calibrator.legacy:
        log.warning(
            "calibration %s is schema 1 (legacy): shape-scoped, no task identity. Every "
            "question reports calibration_status=legacy and no scene gate automates. "
            "Migrate explicitly with `jef calibration migrate` or refit.",
            settings.calibration_path,
        )
    elif calibrator.incompatible:
        log.error(
            "calibration %s was fitted for head %s but %s is loaded: treated as incompatible, "
            "p_correct unavailable, prediction sets are the full set",
            settings.calibration_path,
            calibrator.artifact.model_sha256,
            calibrator.runtime_model_sha256,
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
