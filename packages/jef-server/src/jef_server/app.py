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
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from jef_core import Engine, EvaluateRequest
from jef_core.errors import JefError
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import ValidationError

from .deps import get_engine, get_settings
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


def _error(exc: JefError | Exception, status: int, code: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": str(exc)}})


def create_app(engine: Engine | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    engine = engine or get_engine()

    app = FastAPI(
        title="JEF",
        version="0.1.0",
        description="Open System One decision engine. Typed questions, calibrated confidence.",
    )
    app.state.engine = engine
    app.state.settings = settings
    CALIBRATED.set(1 if engine.calibrator.is_fitted() else 0)

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
        }

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        eng: Engine = app.state.engine
        return {
            "status": "ok",
            "model": eng.model_name,
            "calibrated": eng.calibrator.is_fitted(),
            # Surfaced so a deployment cannot quietly serve the test backbone.
            "test_backbone": settings.is_test_backbone,
        }

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)

    return app


class _TooMany(JefError):
    code = "too_many_questions"
    status = 413
