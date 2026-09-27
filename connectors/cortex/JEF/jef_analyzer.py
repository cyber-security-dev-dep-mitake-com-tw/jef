#!/usr/bin/env python3
"""Cortex analyzer for JEF.

Two flavors share this file, selected by the ``service`` config:

``scene``
    Run a named JEF scene over the observable and report the decision trace.
``question``
    Ask one ad-hoc typed question, for cases where a scene would be overkill.

The taxonomy level is derived from what the decision *means* rather than from a
score, because that chip is what an analyst reads first and a number there
invites them to eyeball a threshold that the scene already applied properly.
"""

from __future__ import annotations

from typing import Any

import requests
from cortexutils.analyzer import Analyzer

#: Scene actions mapped to Cortex taxonomy levels. Anything unlisted is "info":
#: a scene may define its own actions, and guessing a severity for one we have
#: never seen would be worse than declining to.
_ACTION_LEVEL = {
    "close": "safe",
    "request_context": "info",
    "assign": "info",
    "queue_for_review": "suspicious",
    "escalate_human": "suspicious",
    "page_oncall": "malicious",
}


class JefAnalyzer(Analyzer):
    """Runs a scene or a single question against a JEF server."""

    def __init__(self) -> None:
        super().__init__()
        self.service = self.get_param("config.service", "scene")
        self.base_url = self.get_param("config.url", None, "JEF server url is missing").rstrip("/")
        self.timeout = int(self.get_param("config.timeout", 60))
        self.api_key = self.get_param("config.api_key", None)

    # -- transport ---------------------------------------------------------- #

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = requests.post(
                f"{self.base_url}{path}",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            self.error(f"JEF server unreachable at {self.base_url}: {exc}")

        if response.status_code >= 400:
            try:
                detail = response.json().get("error", {})
                self.error(
                    f"JEF rejected the request: {detail.get('code')} -- {detail.get('message')}"
                )
            except ValueError:
                self.error(f"JEF returned HTTP {response.status_code}: {response.text[:400]}")
        return dict(response.json())

    # -- question building -------------------------------------------------- #

    def _build_question(self) -> dict[str, Any]:
        kind = self.get_param("config.question_type", "choice")
        instructions = self.get_param("config.instructions", None, "instructions are required")
        raw_options = self.get_param("config.options", []) or []
        options = [line.strip() for line in raw_options if line and line.strip()]

        if kind == "choice":
            criteria: dict[str, str | None] = {}
            for line in options:
                key, _, description = line.partition("=")
                criteria[key.strip()] = description.strip() or None
            if len(criteria) < 2:
                self.error("a choice question needs at least two 'key=description' options")
            return {"type": "choice", "instructions": instructions, "criteria": criteria}

        if kind == "score":
            if len(options) < 2:
                self.error("a score question needs at least two ordered levels, lowest first")
            return {"type": "score", "instructions": instructions, "criteria": options}

        if kind in ("noul", "boolean"):
            return {"type": kind, "instructions": instructions}

        self.error(f"unknown question_type {kind!r} (expected choice, score or noul)")
        raise AssertionError("unreachable")  # pragma: no cover -- self.error exits

    # -- reporting ---------------------------------------------------------- #

    def summary(self, raw: dict[str, Any]) -> dict[str, Any]:
        taxonomies = []

        if self.service == "scene":
            action = raw.get("action") or "none"
            level = _ACTION_LEVEL.get(action, "info")
            taxonomies.append(self.build_taxonomy(level, "JEF", "action", action))
            if raw.get("human_review"):
                taxonomies.append(self.build_taxonomy("suspicious", "JEF", "review", "human"))
            if not raw.get("calibrated", False):
                # Without calibration nothing can be automated, so this is not a
                # footnote -- it explains why every case looks cautious.
                taxonomies.append(self.build_taxonomy("info", "JEF", "calibration", "absent"))

            reliabilities = [
                q.get("p_correct")
                for layer in raw.get("layers", [])
                for q in layer.get("questions", [])
            ]
            known = [p for p in reliabilities if p is not None]
            if known:
                taxonomies.append(
                    self.build_taxonomy("info", "JEF", "p_correct", f"{min(known):.2f}")
                )
            elif reliabilities:
                # "unknown" and "low" are different, and an analyst deciding
                # whether to trust a verdict needs to see which this is.
                taxonomies.append(self.build_taxonomy("info", "JEF", "p_correct", "unknown"))
        else:
            answer = raw.get("answers", {}).get("q", {})
            kind = answer.get("type", "?")
            if kind == "choice":
                value = answer.get("choice", "?")
            elif kind == "score":
                value = f"{answer.get('score', 0):.2f}"
            else:
                value = f"{answer.get('noul', answer.get('probability', 0)):.2f}"
            taxonomies.append(self.build_taxonomy("info", "JEF", kind, value))
            taxonomies.append(
                self.build_taxonomy(
                    "info", "JEF", "confidence", f"{answer.get('confidence', 0):.2f}"
                )
            )

        return {"taxonomies": taxonomies}

    # -- entry point -------------------------------------------------------- #

    def run(self) -> None:
        super().run()

        observable = self.get_data()
        if isinstance(observable, dict) and "attachment" in observable:
            self.error("JEF reads text, JSON objects and arrays; it cannot read files")
        if isinstance(observable, (dict, list)):
            state: Any = observable
        else:
            state = str(observable)

        if self.service == "scene":
            scene = self.get_param("config.scene", None, "scene id is missing")
            result = self._post(f"/v1/scenes/{scene}:evaluate", {"state": state})
            # Surfaced in the report so an analyst can see the shared-state
            # guarantee holding rather than take it on faith.
            result.setdefault("_jef", {})["state_encodes"] = result.get("state_encodes")
        else:
            result = self._post(
                "/v1/systemone", {"state": state, "questions": {"q": self._build_question()}}
            )

        self.report(result)


if __name__ == "__main__":
    JefAnalyzer().run()
