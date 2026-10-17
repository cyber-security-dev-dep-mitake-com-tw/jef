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

        body: dict[str, Any] = {"state": self._state()}
        case_id = self._case_id()
        if case_id:
            # Lets a server with decisions enabled seal a proposal for this case.
            body["case_id"] = case_id
        try:
            response = requests.post(
                f"{self.base_url}/v1/scenes/{self.scene}:evaluate",
                json=body,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            self.error(f"JEF server unreachable at {self.base_url}: {exc}")

        if response.status_code >= 400:
            self.error(f"JEF returned HTTP {response.status_code}: {response.text[:400]}")

        trace = response.json()
        self.report(self._report(trace))

    def _report(self, trace: dict[str, Any]) -> dict[str, Any]:
        """Tags that describe a proposal, in words that cannot be read as an outcome.

        The responder applies tags; it executes nothing. `JEF:proposed=close`
        plus `JEF:policy=review_required` tells the analyst what the model
        would do and that nobody has authorised it. Dispatching a T2+ action
        needs a signed AuthorizationGrant and an ExecutionReceipt
        (jef_scene.authz); a tag is neither.
        """
        action = trace.get("action") or "none"
        verdict = trace.get("policy_verdict") or "review_required"

        tags = [f"JEF:proposed={action}", f"JEF:policy={verdict}"]
        if trace.get("human_review"):
            tags.append("JEF:human-review")
        if not trace.get("calibrated", False):
            # Explicit, because an uncalibrated server routes everything to a
            # human and that must not be mistaken for a considered judgement.
            tags.append("JEF:uncalibrated")
        for code in trace.get("reason_codes") or []:
            if code != "HUMAN_REVIEW_ACTION":
                tags.append(f"JEF:reason={code}")
        for key, value in (trace.get("action_params") or {}).items():
            tags.append(f"JEF:{key}={value}")

        message = f"scene {self.scene} proposed {action} ({verdict}); nothing was executed"
        report: dict[str, Any] = {"message": message, "tags": tags}
        if trace.get("proposal_token"):
            # Not a tag: it is long and opaque. A workflow that wants to act on
            # the proposal redeems it at /v1/decisions:authorize.
            report["proposal_token"] = trace["proposal_token"]
            report["trace_id"] = trace.get("trace_id")
        return report

    def _case_id(self) -> str | None:
        data = self.get_data() or {}
        for key in ("_id", "id", "caseId"):
            value = data.get(key)
            if value:
                return str(value)
        return None

    def operations(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        return [self.build_operation("AddTagToCase", tag=tag) for tag in raw.get("tags", [])]


if __name__ == "__main__":
    JefTriage().run()
