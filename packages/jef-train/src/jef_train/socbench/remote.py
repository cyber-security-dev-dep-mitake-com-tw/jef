"""Remote scorers that can be compared fairly (plan v3, NEXT-V3-02 / NEXT-V3-08).

A comparison between JEF, Laya, the official TypeSafe Jev and an LLM baseline
is only worth reporting if every system was asked the same question about the
same case, if each one is named and pinned so the report can say exactly what
was measured, if credentials never end up in the report, and if a system that
fails to answer properly is counted as failing rather than quietly scored.
This module provides that:

``SystemOneScorer``
    Any ``POST /v1/systemone`` service: TypeSafe's own API, a JEF server, Laya
    (Jev-wire-compatible), an LLM wrapper, or research/jev-local.
``GatewayScorer``
    The Vercel AI Gateway decision-model dialect (``POST {base}/decision-model``,
    spec v4), for reaching the official Jev through the gateway.

Every scorer:

* asks :func:`baselines.verdict_question` -- the same instructions and the same
  options in the same order that ``JefScorer`` asks;
* has a unique ``name`` and records provider, model id, version, checkpoint
  digest, hardware, price and data flow in ``describe()``;
* reads its credential from the environment variable named in ``auth_env`` at
  construction (failing before any case is sent if it is unset) and never
  writes the value anywhere;
* raises a :class:`ScorerContractError` subclass when a reply is not a full,
  valid distribution over the task's labels, so ``evaluate`` counts it as a
  failure with a reason instead of filling in zeros.

``load_scorer_config`` reads a JSON file of such scorers; inline secrets are
rejected.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .annotate import TASK_LABELS
from .baselines import VERDICT_INSTRUCTIONS, case_text, prompt_digest, verdict_question

__all__ = [
    "GatewayScorer",
    "HttpScorer",
    "MissingLabelsError",
    "NoDistributionError",
    "NotADistributionError",
    "RefusalError",
    "RemoteScorer",
    "ScorerConfigError",
    "ScorerContractError",
    "SystemOneScorer",
    "load_scorer_config",
    "strict_distribution",
]

Post = Callable[[str, dict[str, Any], dict[str, str]], dict[str, Any]]
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
META_KEYS = (
    "provider",
    "version",
    "checkpoint_sha256",
    "fine_tuned",
    "hardware",
    "precision",
    "context_limit",
    "price_per_1k_cases_usd",
    "data_flow",
    "notes",
)


class ScorerConfigError(ValueError):
    """A scorer cannot be built honestly (missing credential, bad config)."""


class ScorerContractError(RuntimeError):
    """A reply that is not a usable distribution. Counted as a failure."""


class NoDistributionError(ScorerContractError):
    """The system picked an answer but gave no probabilities."""


class RefusalError(ScorerContractError):
    """The system refused to answer."""


class MissingLabelsError(ScorerContractError):
    """The distribution does not cover every allowed label."""


class NotADistributionError(ScorerContractError):
    """Negative, non-finite, or not summing to 1."""


def strict_distribution(
    labels: tuple[str, ...] | list[str], probs: Mapping[str, Any]
) -> dict[str, float]:
    """``probs`` over exactly ``labels``, renormalised, or a ScorerContractError."""
    missing = [lab for lab in labels if lab not in probs]
    if missing:
        raise MissingLabelsError(f"no probability for {missing}")
    values = []
    for lab in labels:
        v = probs[lab]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise NotADistributionError(f"{lab}: {v!r} is not a probability")
        values.append(float(v))
    total = sum(values)
    if abs(total - 1.0) > 0.02:
        raise NotADistributionError(f"probabilities sum to {total:.4f}, not 1")
    return {lab: v / total for lab, v in zip(labels, values, strict=True)}


def _default_post(timeout: float) -> Post:
    def post(url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        import urllib.request

        if not url.startswith(("http://127.0.0.1", "http://localhost", "http://[::1]", "https://")):
            raise ValueError("remote scorers post only to https or loopback http")
        req = urllib.request.Request(  # noqa: S310 -- scheme checked above
            url,
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json", **headers},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read())

    return post


class RemoteScorer:
    """Shared plumbing: naming, pinning, auth from env, retries, egress."""

    kind = "remote"

    def __init__(
        self,
        url: str,
        *,
        name: str,
        model: str,
        auth_env: str | None = None,
        auth_header: str = "Authorization",
        auth_scheme: str = "Bearer",
        timeout: float = 120.0,
        retries: int = 0,
        meta: Mapping[str, Any] | None = None,
        post: Post | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if not name:
            raise ScorerConfigError("every remote scorer needs a unique name")
        self.url = url.rstrip("/")
        self.name = name
        self.model = model
        self.auth_env = auth_env
        self.auth_header = auth_header
        self.auth_scheme = auth_scheme
        self.timeout = timeout
        self.retries = max(0, int(retries))
        self.meta = {k: v for k, v in (meta or {}).items() if k in META_KEYS}
        self._post = post or _default_post(timeout)
        env = os.environ if environ is None else environ
        self._secret: str | None = None
        if auth_env:
            value = env.get(auth_env)
            if not value:
                raise ScorerConfigError(
                    f"scorer {name!r}: environment variable {auth_env} is not set"
                )
            self._secret = value
        self.attempts = 0

    @property
    def host(self) -> str:
        return urlparse(self.url).hostname or ""

    @property
    def egress(self) -> str:
        """``loopback`` or ``external``: whether a case leaves this machine."""
        return "loopback" if self.host in LOOPBACK else "external"

    def _headers(self) -> dict[str, str]:
        if not self._secret:
            return {}
        value = f"{self.auth_scheme} {self._secret}" if self.auth_scheme else self._secret
        return {self.auth_header: value}

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.kind,
            "url": self.url,
            "host": self.host,
            "egress": self.egress,
            "model": self.model,
            # The variable's name only. The value is never recorded.
            "auth": f"env:{self.auth_env}" if self.auth_env else "none",
            "timeout_s": self.timeout,
            "retries": self.retries,
            "prompt_digest": prompt_digest(VERDICT_INSTRUCTIONS),
            **self.meta,
        }

    def predict(self, case: Mapping[str, Any]) -> dict[str, float]:
        raise NotImplementedError

    def _call(
        self, url: str, body: dict[str, Any], extra: Mapping[str, str] | None = None
    ) -> dict[str, Any]:
        headers = {**self._headers(), **(extra or {})}
        last: Exception | None = None
        for _ in range(self.retries + 1):
            self.attempts += 1
            started = time.perf_counter()
            try:
                reply = self._post(url, body, headers)
                self.last_latency = time.perf_counter() - started
                return reply
            except ScorerContractError:
                raise
            except Exception as exc:  # timeouts, connection errors, HTTP errors
                last = exc
        if last is None:  # pragma: no cover - the loop runs at least once
            raise RuntimeError("no attempt was made")
        raise last


class SystemOneScorer(RemoteScorer):
    """``POST {url}/v1/systemone`` with the shared verdict question."""

    kind = "systemone"

    def predict(self, case: Mapping[str, Any]) -> dict[str, float]:
        task = case["task_id"]
        body = {
            "model": self.model,
            "state": case["state"],
            "questions": {"verdict": verdict_question(task)},
        }
        reply = self._call(f"{self.url}/v1/systemone", body)
        answer = (reply.get("answers") or {}).get("verdict")
        if not isinstance(answer, Mapping):
            raise NoDistributionError("reply has no answer for the verdict question")
        if answer.get("type") == "refusal":
            raise RefusalError("the system refused")
        probs = answer.get("probabilities")
        if not isinstance(probs, Mapping):
            raise NoDistributionError("the answer carries no probabilities")
        return strict_distribution(TASK_LABELS[task], probs)


class GatewayScorer(RemoteScorer):
    """Vercel AI Gateway decision-model dialect, spec v4 (``POST {url}/decision-model``)."""

    kind = "gateway"

    def predict(self, case: Mapping[str, Any]) -> dict[str, float]:
        task = case["task_id"]
        state = case["state"]
        part = (
            {"type": "json", "value": state}
            if isinstance(state, Mapping)
            else {"type": "text", "text": str(state) if state is not None else case_text(case)}
        )
        body = {"stateParts": [part], "questions": {"verdict": verdict_question(task)}}
        reply = self._call(
            f"{self.url}/decision-model",
            body,
            {"ai-decision-model-specification-version": "4", "ai-model-id": self.model},
        )
        answer = (reply.get("answers") or {}).get("verdict")
        if not isinstance(answer, Mapping):
            raise NoDistributionError("reply has no answer for the verdict question")
        if answer.get("type") == "refusal":
            raise RefusalError("the system refused")
        probs = answer.get("probabilities")
        if not isinstance(probs, Mapping):
            # The gateway schema makes probabilities optional. Without them there
            # is nothing to rank or calibrate, so it is a failure, not a guess.
            raise NoDistributionError("the gateway returned a choice without probabilities")
        return strict_distribution(TASK_LABELS[task], probs)


class HttpScorer(SystemOneScorer):
    """Backward-compatible ``http=URL`` scorer: an unauthenticated SystemOneScorer."""

    def __init__(
        self,
        base_url: str,
        *,
        name: str = "http",
        model: str = "jev-latest",
        post: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        timeout: float = 120.0,
    ) -> None:
        adapted: Post | None = (lambda u, b, h: post(u, b)) if post else None
        super().__init__(base_url, name=name, model=model, timeout=timeout, post=adapted)
        self.base_url = self.url

    def _default_post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        return _default_post(self.timeout)(url, body, {})


_KINDS: dict[str, type[RemoteScorer]] = {"systemone": SystemOneScorer, "gateway": GatewayScorer}
_ALLOWED = {
    "name",
    "type",
    "url",
    "model",
    "auth_env",
    "auth_header",
    "auth_scheme",
    "timeout_s",
    "retries",
    *META_KEYS,
}
_SECRET_HINTS = ("key", "token", "secret", "password", "bearer")


def load_scorer_config(
    path: str | Path, *, environ: Mapping[str, str] | None = None, post: Post | None = None
) -> list[RemoteScorer]:
    """Remote scorers from a JSON file: a list, or ``{"scorers": [...]}``."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = raw.get("scorers") if isinstance(raw, Mapping) else raw
    if not isinstance(entries, list) or not entries:
        raise ScorerConfigError(f"{path}: expected a non-empty list of scorers")
    out: list[RemoteScorer] = []
    seen: set[str] = set()
    for i, e in enumerate(entries):
        if not isinstance(e, Mapping):
            raise ScorerConfigError(f"{path}[{i}]: each scorer is an object")
        unknown = sorted(set(e) - _ALLOWED)
        leaked = [k for k in unknown if any(h in k.lower() for h in _SECRET_HINTS)]
        if leaked:
            raise ScorerConfigError(
                f"{path}[{i}]: {leaked} looks like an inline secret; name an environment "
                "variable in auth_env instead, so the value never enters a file or report"
            )
        if unknown:
            raise ScorerConfigError(f"{path}[{i}]: unknown keys {unknown}")
        for req in ("name", "type", "url", "model"):
            if not e.get(req):
                raise ScorerConfigError(f"{path}[{i}]: {req} is required")
        if e["name"] in seen:
            raise ScorerConfigError(f"{path}: scorer name {e['name']!r} is used twice")
        seen.add(e["name"])
        cls = _KINDS.get(e["type"])
        if cls is None:
            raise ScorerConfigError(f"{path}[{i}]: type must be one of {sorted(_KINDS)}")
        out.append(
            cls(
                e["url"],
                name=e["name"],
                model=e["model"],
                auth_env=e.get("auth_env"),
                auth_header=e.get("auth_header", "Authorization"),
                auth_scheme=e.get("auth_scheme", "Bearer"),
                timeout=float(e.get("timeout_s", 120)),
                retries=int(e.get("retries", 0)),
                meta={k: e[k] for k in META_KEYS if k in e},
                post=post,
                environ=environ,
            )
        )
    return out
