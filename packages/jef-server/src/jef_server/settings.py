"""Server configuration, read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = ["RequiredTask", "Settings", "load_settings", "parse_required_tasks"]


@dataclass(frozen=True)
class RequiredTask:
    """One entry of ``JEF_REQUIRED_TASK_IDS``: ``task_id`` or ``task_id:kind:n``.

    With a shape, readiness checks that exact bucket. Without one, every bucket
    fitted for the task must be validated. Alpha comes from ``JEF_ALPHA``.
    """

    task_id: str
    kind: str | None = None
    n_options: int | None = None

    @property
    def spec(self) -> str:
        if self.kind is None:
            return self.task_id
        return f"{self.task_id}:{self.kind}:{self.n_options}"


def parse_required_tasks(raw: str | None) -> tuple[RequiredTask, ...]:
    out: list[RequiredTask] = []
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) == 1:
            out.append(RequiredTask(parts[0]))
        elif len(parts) == 3 and parts[1] in {"noul", "choice", "score"} and parts[2].isdigit():
            out.append(RequiredTask(parts[0], parts[1], int(parts[2])))
        else:
            raise ValueError(
                f"JEF_REQUIRED_TASK_IDS entry {item!r} is not task_id or task_id:kind:n"
            )
    return tuple(out)


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
    #: manifest.json to verify the head and calibration against. Defaults to
    #: ``manifest.json`` next to the head when that file exists.
    model_manifest: str | None = None
    model_name: str | None = None
    scenes_dir: str | None = None
    #: Directory holding the debug UI. Empty disables the route entirely.
    ui_dir: str | None = None
    #: torch intra-op threads. The target is a 16 vCPU no-GPU Proxmox VM, where
    #: this is the single biggest throughput lever.
    threads: int | None = None
    max_questions: int = 256
    max_state_chars: int = 1_000_000
    alpha: float = 0.10

    # -- ingress --------------------------------------------------------------
    #: ``(label, secret)`` pairs from ``JEF_API_KEYS``. Empty means no app-level
    #: authentication, which is only acceptable behind an ingress that does it;
    #: ``/healthz`` reports ``auth: none`` so nobody mistakes the default for
    #: protection.
    api_keys: tuple[tuple[str, str], ...] = ()
    #: ``JEF_REQUIRE_AUTH``: refuse to start when ``api_keys`` is empty. For
    #: deployments that chose app-level auth and must not silently lose it.
    require_auth: bool = False
    #: Largest request body accepted, by Content-Length. The default is
    #: comfortably above a 1M-character state as UTF-8 JSON (at most ~4 MB).
    max_body_bytes: int = 8_000_000
    #: Deepest JSON nesting a payload may have. Real requests are three deep.
    max_json_depth: int = 32
    #: Per-caller requests per minute; 0 disables the limiter.
    rate_limit_per_minute: int = 0
    #: Leave ``/metrics`` reachable without a key, for a scraper that cannot
    #: send one. Off by default: the metrics name every scene.
    metrics_public: bool = False
    #: ``JEF_TRUSTED_INGRESS``: the operator asserts that an ingress in front of
    #: this server (mTLS, OIDC gateway) authenticates every caller, so an empty
    #: ``api_keys`` does not block readiness. Reported as ``auth:
    #: trusted-ingress`` so the assertion is visible, never silent.
    trusted_ingress: bool = False

    # -- readiness ------------------------------------------------------------
    #: ``JEF_REQUIRED_TASK_IDS``: the tasks this deployment must be calibrated
    #: for before ``/v1/readiness`` reports ``automation_ready``. Empty means
    #: nothing is required, which readiness reports as a blocking reason: a
    #: deployment that names no task cannot claim to be ready for any.
    required_task_ids: tuple[RequiredTask, ...] = ()

    # -- decisions (authorize / receipt) --------------------------------------
    #: ``JEF_DECISIONS_ENABLED``: serve /v1/decisions. Off by default; when on,
    #: JEF_API_KEYS is required, because these endpoints record what a SOAR
    #: platform did. JEF still executes nothing and holds no platform
    #: credentials: it authorizes, and it records.
    decisions_enabled: bool = False
    #: Verifies AuthorizationGrants. Shared with the approval service that
    #: issues them. Empty: every grant is refused, so only allowed T0/T1
    #: proposals can be authorized.
    grant_secret: bytes = b""
    #: Seals proposal tokens. Empty: a random per-process secret, so tokens do
    #: not survive a restart and do not work across replicas.
    proposal_secret: bytes = b""
    #: How long a sealed proposal stays redeemable.
    proposal_ttl_seconds: int = 3600

    @property
    def is_test_backbone(self) -> bool:
        return self.backbone == "hashing"

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)


def _int(name: str) -> int | None:
    raw = os.environ.get(name)
    return int(raw) if raw else None


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _api_keys(raw: str | None) -> tuple[tuple[str, str], ...]:
    # Parsed here rather than in security.py so Settings stays the one place
    # configuration is read; security.py only consumes the result.
    from .security import parse_api_keys

    return tuple(sorted(parse_api_keys(raw).items()))


def load_settings() -> Settings:
    return Settings(
        backbone=os.environ.get("JEF_BACKBONE", "hashing"),
        head_path=os.environ.get("JEF_HEAD_PATH") or None,
        calibration_path=os.environ.get("JEF_CALIBRATION_PATH") or None,
        model_manifest=os.environ.get("JEF_MODEL_MANIFEST") or None,
        model_name=os.environ.get("JEF_MODEL_NAME") or None,
        scenes_dir=os.environ.get("JEF_SCENES_DIR") or None,
        ui_dir=os.environ.get("JEF_UI_DIR") or None,
        threads=_int("JEF_THREADS"),
        max_questions=_int("JEF_MAX_QUESTIONS") or 256,
        max_state_chars=_int("JEF_MAX_STATE_CHARS") or 1_000_000,
        alpha=float(os.environ.get("JEF_ALPHA", "0.10")),
        api_keys=_api_keys(os.environ.get("JEF_API_KEYS")),
        require_auth=_flag("JEF_REQUIRE_AUTH"),
        max_body_bytes=_int("JEF_MAX_BODY_BYTES") or 8_000_000,
        max_json_depth=_int("JEF_MAX_JSON_DEPTH") or 32,
        rate_limit_per_minute=_int("JEF_RATE_LIMIT_PER_MINUTE") or 0,
        metrics_public=_flag("JEF_METRICS_PUBLIC"),
        trusted_ingress=_flag("JEF_TRUSTED_INGRESS"),
        required_task_ids=parse_required_tasks(os.environ.get("JEF_REQUIRED_TASK_IDS")),
        decisions_enabled=_flag("JEF_DECISIONS_ENABLED"),
        grant_secret=os.environ.get("JEF_GRANT_SECRET", "").encode(),
        proposal_secret=os.environ.get("JEF_PROPOSAL_SECRET", "").encode(),
        proposal_ttl_seconds=_int("JEF_PROPOSAL_TTL_SECONDS") or 3600,
    )
