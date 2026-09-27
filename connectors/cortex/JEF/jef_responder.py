#!/usr/bin/env python3
"""Cortex responder: triage a TheHive case or alert with a JEF scene.

The case's title and description are sent as a JSON object rather than
concatenated text, so the model sees the structure a human sees. The verdict
comes back as tags, including an explicit one when a human must review -- an
automated triage that can silently stop escalating is worse than no automated
triage.
"""

from __future__ import annotations

from typing import Any

import requests
from cortexutils.responder import Responder


class JefTriage(Responder):
    """Runs a scene over a case/alert and returns tags for TheHive to apply."""

    def __init__(self) -> None:
        super().__init__()
        self.base_url = self.get_param("config.url", None, "JEF server url is missing").rstrip("/")
        self.scene = self.get_param("config.scene", "incident-triage")
        self.timeout = int(self.get_param("config.timeout", 60))
        self.api_key = self.get_param("config.api_key", None)

    def _state(self) -> dict[str, Any]:
        data = self.get_data()
        state: dict[str, Any] = {}
        for key in ("title", "description", "summary", "source", "sourceRef", "type"):
            value = data.get(key)
            if value:
                state[key] = value
        if data.get("tags"):
            state["tags"] = data["tags"]
        if data.get("severity") is not None:
            # TheHive's own severity is evidence, not the answer: the scene is
            # being asked whether to agree with it.
            state["thehive_severity"] = data["severity"]
        observables = data.get("observables") or data.get("artifacts") or []
        if observables:
            state["observables"] = [
                {"type": o.get("dataType"), "value": o.get("data")}
                for o in observables
                if o.get("data")
            ][:50]
        return state or {"raw": str(data)[:4000]}

    def run(self) -> None:
        super().run()

        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                f"{self.base_url}/v1/scenes/{self.scene}:evaluate",
                json={"state": self._state()},
                headers=headers,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            self.error(f"JEF server unreachable at {self.base_url}: {exc}")

        if response.status_code >= 400:
            self.error(f"JEF returned HTTP {response.status_code}: {response.text[:400]}")

        trace = response.json()
        action = trace.get("action") or "none"

        tags = [f"JEF:{action}"]
        if trace.get("human_review"):
            tags.append("JEF:human-review")
        if not trace.get("calibrated", False):
            # Explicit, because an uncalibrated server routes everything to a
            # human and that must not be mistaken for a considered judgement.
            tags.append("JEF:uncalibrated")
        for key, value in (trace.get("action_params") or {}).items():
            tags.append(f"JEF:{key}={value}")

        self.report({"message": f"scene {self.scene} -> {action}", "tags": tags})

    def operations(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        return [self.build_operation("AddTagToCase", tag=tag) for tag in raw.get("tags", [])]


if __name__ == "__main__":
    JefTriage().run()
