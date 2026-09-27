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

from .settings import Settings, load_settings

log = logging.getLogger("jef.server")

__all__ = ["build_engine", "get_engine", "get_settings"]


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
