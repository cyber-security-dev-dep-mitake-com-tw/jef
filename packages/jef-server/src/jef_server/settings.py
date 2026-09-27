"""Server configuration, read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = ["Settings", "load_settings"]


@dataclass(frozen=True)
class Settings:
    """Runtime configuration.

    Defaults are deliberately safe for a cold start: the hashing backbone loads
    instantly and downloads nothing, so `docker run` works offline. Production
    deployments set ``JEF_BACKBONE`` to a model id.
    """

    backbone: str = "hashing"
    head_path: str | None = None
    calibration_path: str | None = None
    model_name: str | None = None
    scenes_dir: str | None = None
    #: torch intra-op threads. The target is a 16 vCPU no-GPU Proxmox VM, where
    #: this is the single biggest throughput lever.
    threads: int | None = None
    max_questions: int = 256
    max_state_chars: int = 1_000_000
    alpha: float = 0.10

    @property
    def is_test_backbone(self) -> bool:
        return self.backbone == "hashing"


def _int(name: str) -> int | None:
    raw = os.environ.get(name)
    return int(raw) if raw else None


def load_settings() -> Settings:
    return Settings(
        backbone=os.environ.get("JEF_BACKBONE", "hashing"),
        head_path=os.environ.get("JEF_HEAD_PATH") or None,
        calibration_path=os.environ.get("JEF_CALIBRATION_PATH") or None,
        model_name=os.environ.get("JEF_MODEL_NAME") or None,
        scenes_dir=os.environ.get("JEF_SCENES_DIR") or None,
        threads=_int("JEF_THREADS"),
        max_questions=_int("JEF_MAX_QUESTIONS") or 256,
        max_state_chars=_int("JEF_MAX_STATE_CHARS") or 1_000_000,
        alpha=float(os.environ.get("JEF_ALPHA", "0.10")),
    )
