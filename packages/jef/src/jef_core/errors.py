"""JEF error types, mapped to HTTP status codes by jef-server."""

from __future__ import annotations


class JefError(Exception):
    """Base class for all JEF errors."""

    code: str = "jef_error"
    status: int = 500


class InvalidQuestionError(JefError):
    """A question definition violates the /v1/systemone contract."""

    code = "invalid_question"
    status = 422


class InvalidStateError(JefError):
    """The supplied state is not an accepted shape."""

    code = "invalid_state"
    status = 422


class BackendUnavailableError(JefError):
    """The requested inference backend could not be loaded."""

    code = "backend_unavailable"
    status = 503


class SceneError(JefError):
    """A scene definition or evaluation failed."""

    code = "scene_error"
    status = 422
