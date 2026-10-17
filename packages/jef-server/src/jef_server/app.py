"""JEF HTTP API.

Two surfaces:

* ``POST /v1/systemone`` -- the compatible contract, so existing TypeSafe / AI
  SDK clients can point at a JEF deployment unchanged.
* ``POST /v1/scenes/{id}:evaluate`` -- JEF's own layer, added in P2.

Handlers are deliberately plain ``def`` rather than ``async def``: evaluation is
CPU-bound, so FastAPI runs them in its threadpool instead of blocking the event
loop. On the 16 vCPU CPU-only target that is the difference between serving
concurrent requests and serialising them.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from jef_core import Engine, EvaluateRequest
from jef_core.engine import render_state
from jef_core.errors import JefError
from jef_scene import SceneEngine, SceneRegistry
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import ValidationError

from ._version import __version__
from .deps import build_scenes, get_engine, get_settings, manifest_check
from .security import install_guards, json_depth
from .settings import Settings

log = logging.getLogger("jef.server")

__all__ = ["create_app"]

# --------------------------------------------------------------------------- #
# Metrics
#
# state_encodes / requests is the production proof of D3: it must stay at 1.0 no
# matter how many questions per request, and a rising ratio means the shared
# state encoding has regressed into per-question encoding.
# --------------------------------------------------------------------------- #

REQUESTS = Counter("jef_requests_total", "Evaluation requests", ["endpoint", "status"])
QUESTIONS = Counter("jef_questions_total", "Questions evaluated", ["kind"])
STATE_ENCODES = Counter("jef_state_encodes_total", "Full state encodes performed")
LATENCY = Histogram(
    "jef_evaluate_seconds",
    "Evaluation wall time",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
CALIBRATED = Gauge("jef_calibration_loaded", "1 if a fitted calibrator is loaded")
SCENE_RUNS = Counter("jef_scene_runs_total", "Scene evaluations", ["scene", "action"])
SCENE_HUMAN = Counter(
    "jef_scene_human_review_total",
    "Scene runs that ended at a human. A deployment where this never moves is "
    "one that has stopped escalating.",
    ["scene"],
)


def _error(exc: JefError | Exception, status: int, code: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": str(exc)}})


def create_app(
    engine: Engine | None = None,
    settings: Settings | None = None,
    scenes: tuple[SceneEngine, SceneRegistry] | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    engine = engine or get_engine()
    scene_engine, registry = scenes or build_scenes(settings, engine)

    app = FastAPI(
        title="JEF",
        version=__version__,
        description="Open System One decision engine. Typed questions, calibrated confidence.",
    )
    app.state.engine = engine
    app.state.settings = settings
    app.state.scene_engine = scene_engine
    app.state.scenes = registry
    app.state.manifest = manifest_check(settings)
    CALIBRATED.set(1 if engine.calibrator.is_fitted() else 0)

    # Auth, size and rate checks: header-only, ahead of body parsing. Raises
    # at startup if JEF_REQUIRE_AUTH is set without keys.
    install_guards(app, settings)
    if not settings.auth_enabled:
        log.warning(
            "JEF_API_KEYS is empty: the API has no authentication of its own. Only "
            "acceptable behind an ingress that authenticates; /healthz reports auth=none"
        )

    def _budget(payload: dict[str, Any]) -> None:
        """Per-request budgets that need the parsed body, still ahead of inference."""
        depth = json_depth(payload)
        if depth > settings.max_json_depth:
            raise _TooDeep(f"payload nests {depth} levels; the limit is {settings.max_json_depth}")
        state = payload.get("state")
        if state is not None:
            chars = len(state) if isinstance(state, str) else len(render_state(state))
            if chars > settings.max_state_chars:
                raise _StateTooLarge(
                    f"state is {chars} characters; the limit is {settings.max_state_chars}"
                )

    # -- error handling ---------------------------------------------------- #

    @app.exception_handler(JefError)
    def _jef_error(_: Request, exc: JefError) -> JSONResponse:
        REQUESTS.labels(endpoint="systemone", status=str(exc.status)).inc()
        return _error(exc, exc.status, exc.code)

    @app.exception_handler(ValidationError)
    def _validation_error(_: Request, exc: ValidationError) -> JSONResponse:
        REQUESTS.labels(endpoint="systemone", status="422").inc()
        return _error(exc, 422, "invalid_request")

    # -- the compatible contract ------------------------------------------- #

    @app.post("/v1/systemone")
    def systemone(payload: dict[str, Any]) -> Any:
        """Evaluate typed questions against one state.

        The request is parsed leniently into a dict first so that contract
        violations surface as JEF's own 422 with a stable error code, rather
        than FastAPI's generic body-validation envelope which clients would
        have to special-case.
        """
        started = time.perf_counter()
        eng: Engine = app.state.engine

        _budget(payload)
        req = EvaluateRequest.model_validate(payload)

        n = len(req.questions)
        if n > settings.max_questions:
            raise _TooMany(f"{n} questions exceeds the limit of {settings.max_questions}")

        before = eng.encode_count
        result = eng.evaluate(req.state, req.questions)
        STATE_ENCODES.inc(eng.encode_count - before)

        for q in req.questions.values():
            kind = "noul" if q.type in ("noul", "boolean") else q.type
            QUESTIONS.labels(kind=kind).inc()
        REQUESTS.labels(endpoint="systemone", status="200").inc()
        LATENCY.labels(endpoint="systemone").observe(time.perf_counter() - started)

        # exclude_none keeps the noul/boolean dialect clean: a `noul` answer must
        # not carry a null `probability` field and vice versa.
        return result.model_dump(exclude_none=True)

    # -- scenes: the layer Jev leaves to the caller ------------------------ #

    @app.get("/v1/scenes")
    def list_scenes() -> dict[str, Any]:
        reg: SceneRegistry = app.state.scenes
        return {
            "data": [
                {
                    "id": name,
                    "version": reg.get(name).version,
                    "description": reg.get(name).description,
                    "layers": len(reg.get(name).layers),
                    "questions": reg.get(name).question_count,
                }
                for name in reg.names()
            ]
        }

    @app.post("/v1/scenes/{scene_id}:evaluate")
    def evaluate_scene(scene_id: str, payload: dict[str, Any]) -> Any:
        """Run a scene against one state and return the full decision trace.

        The trace is the product here, not the verdict. An auditor needs the
        evidence, the questions, the permitted answers, where the probability
        mass fell and which gate fired -- which is precisely what hand-written
        playbook branching cannot produce.
        """
        started = time.perf_counter()
        scene = app.state.scenes.get(scene_id)
        if "state" not in payload:
            raise _MissingState("request body must contain 'state'")
        _budget(payload)

        trace = app.state.scene_engine.run(scene, payload["state"])

        SCENE_RUNS.labels(scene=scene_id, action=trace.action or "none").inc()
        if trace.human_review:
            SCENE_HUMAN.labels(scene=scene_id).inc()
        STATE_ENCODES.inc(trace.state_encodes)
        for layer in trace.layers:
            for question in layer.questions:
                QUESTIONS.labels(kind=question.kind).inc()
        REQUESTS.labels(endpoint="scene", status="200").inc()
        LATENCY.labels(endpoint="scene").observe(time.perf_counter() - started)

        return trace.to_dict()

    # -- discovery --------------------------------------------------------- #

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        eng: Engine = app.state.engine
        return {
            "data": [
                {
                    "id": eng.model_name,
                    "object": "model",
                    "backbone": eng.backbone.name,
                    "head": eng.head.name,
                    "calibrated": eng.calibrator.is_fitted(),
                    "question_types": ["choice", "score", "noul", "boolean"],
                }
            ]
        }

    @app.get("/v1/limits")
    def limits() -> dict[str, Any]:
        return {
            "max_questions_per_request": settings.max_questions,
            "max_state_chars": settings.max_state_chars,
            "max_body_bytes": settings.max_body_bytes,
            "max_json_depth": settings.max_json_depth,
            "rate_limit_per_minute": settings.rate_limit_per_minute,
            "auth": "api-key" if settings.auth_enabled else "none",
        }

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        """Liveness, plus the facts a deployer needs to judge readiness.

        ``status: ok`` means the process answers. ``production_safe`` means
        something stricter and is false by default: a real backbone, a
        fitted calibrator, app-level authentication, and a model manifest
        the loaded files match. A deployment that
        installs only the hashing stub, or no keys, is told so here rather
        than discovering it from a confident-looking answer.
        """
        eng: Engine = app.state.engine
        test_backbone = settings.is_test_backbone or eng.backbone.name == "hashing"
        calibrated = eng.calibrator.is_fitted()
        return {
            "status": "ok",
            "model": eng.model_name,
            "model_loaded": True,
            "calibrated": calibrated,
            "calibration_scope": sorted(eng.calibrator.buckets),
            # Surfaced so a deployment cannot quietly serve the test backbone.
            "test_backbone": test_backbone,
            "auth": "api-key" if settings.auth_enabled else "none",
            # verified | mismatch | absent. On mismatch the calibration was
            # dropped at load, so `calibrated` is false as well.
            "model_manifest": app.state.manifest.status,
            "production_safe": (not test_backbone)
            and calibrated
            and settings.auth_enabled
            and app.state.manifest.status == "verified",
        }

    # -- debug UI ----------------------------------------------------------- #

    if settings.ui_dir:
        ui_index = Path(settings.ui_dir) / "index.html"
        if not ui_index.is_file():
            # A UI directory that is configured but empty means the operator
            # expects a UI and will not get one. Fail at startup, not at 3am.
            raise RuntimeError(f"JEF_UI_DIR={settings.ui_dir!r} has no index.html")

        @app.get("/ui", include_in_schema=False)
        def ui() -> FileResponse:
            """The decision viewer.

            One self-contained file: a debugging tool that needs a build step
            before it runs is one you cannot open during an incident.
            """
            return FileResponse(ui_index, media_type="text/html; charset=utf-8")

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)

    return app


class _TooMany(JefError):
    code = "too_many_questions"
    status = 413


class _MissingState(JefError):
    code = "invalid_request"
    status = 422


class _StateTooLarge(JefError):
    code = "state_too_large"
    status = 413


class _TooDeep(JefError):
    code = "payload_too_deep"
    status = 422
