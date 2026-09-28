"""Talk to a JEF server over HTTP.

Mirrors :class:`jef_sdk.Jef` so that moving a workload between in-process and
remote is a change of constructor, not a change of code.
"""

from __future__ import annotations

from typing import Any

import httpx
from jef_core import EvaluateResponse

__all__ = ["JefClient", "JefHTTPError"]

DEFAULT_TIMEOUT = 30.0


class JefHTTPError(RuntimeError):
    """A JEF server returned an error.

    Carries the server's own error code so a caller can branch on
    ``invalid_question`` versus ``too_many_questions`` without parsing prose.
    """

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"{code} ({status}): {message}")
        self.status = status
        self.code = code
        self.message = message


class JefClient:
    """HTTP client for ``/v1/systemone`` and ``/v1/scenes``.

    Example:
        >>> from jef_sdk import JefClient, choice          # doctest: +SKIP
        >>> with JefClient("http://localhost:8080") as jef:
        ...     r = jef.evaluate("付款失敗", {"team": choice("誰處理？", soc="監控", infra="網路")})
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        *,
        timeout: float = DEFAULT_TIMEOUT,
        headers: dict[str, str] | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self.base_url, timeout=timeout, headers=headers or {}
        )

    # -- lifecycle ---------------------------------------------------------- #

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> JefClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- requests ----------------------------------------------------------- #

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        response = self._client.post(path, json=payload)
        if response.status_code >= 400:
            self._raise(response)
        return response.json()

    @staticmethod
    def _raise(response: httpx.Response) -> None:
        try:
            error = response.json().get("error", {})
        except ValueError:
            error = {}
        raise JefHTTPError(
            response.status_code,
            error.get("code", "http_error"),
            error.get("message", response.text[:500]),
        )

    # -- API ---------------------------------------------------------------- #

    def evaluate(self, state: object, questions: dict[str, Any]) -> EvaluateResponse:
        """Evaluate typed questions. Returns the same model the embedded API does."""
        body = self._post("/v1/systemone", {"state": state, "questions": questions})
        return EvaluateResponse.model_validate(body)

    def run_scene(self, name: str, state: object) -> dict[str, Any]:
        """Run a scene and return its decision trace as a dict.

        Left as a dict rather than reconstructed into SceneTrace on purpose: a
        server may be running a newer JEF with extra trace fields, and dropping
        them on the floor to satisfy an older dataclass would lose exactly the
        audit detail the trace exists to carry.
        """
        result = self._post(f"/v1/scenes/{name}:evaluate", {"state": state})
        return dict(result)

    def scenes(self) -> list[dict[str, Any]]:
        return list(self._get("/v1/scenes")["data"])

    def models(self) -> list[dict[str, Any]]:
        return list(self._get("/v1/models")["data"])

    def limits(self) -> dict[str, Any]:
        return dict(self._get("/v1/limits"))

    def health(self) -> dict[str, Any]:
        return dict(self._get("/healthz"))

    def _get(self, path: str) -> Any:
        response = self._client.get(path)
        if response.status_code >= 400:
            self._raise(response)
        return response.json()
