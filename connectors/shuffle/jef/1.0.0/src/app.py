"""Shuffle app for JEF.

Every action returns JSON as a string, which is what Shuffle's Liquid templating
consumes cleanly: a workflow can then branch on
``$jef.action`` or ``$jef.answers.team.choice`` without a parsing step.

The actions are deliberately shaped around what a playbook does rather than
around the HTTP API. ``ask_many`` exists because JEF evaluates every question in
a request in parallel against the same state, so asking five speculative
questions costs barely more than asking one -- and a workflow that asks
everything up front branches on results instead of round-tripping per branch.
"""

from __future__ import annotations

import json
from typing import Any

import requests
from shuffle_sdk import AppBase


class JEF(AppBase):
    __version__ = "1.0.0"
    app_name = "JEF"

    def __init__(self, redis: Any = None, logger: Any = None, console_logger: Any = None) -> None:
        super().__init__(redis, logger, console_logger)

    # -- transport ---------------------------------------------------------- #

    def _request(self, url: str, apikey: str, path: str, payload: dict[str, Any] | None) -> str:
        base = (url or "").rstrip("/")
        if not base:
            return self._fail("no JEF url configured")

        headers = {"content-type": "application/json"}
        if apikey:
            headers["authorization"] = f"Bearer {apikey}"

        try:
            if payload is None:
                response = requests.get(f"{base}{path}", headers=headers, timeout=60)
            else:
                response = requests.post(f"{base}{path}", json=payload, headers=headers, timeout=60)
        except requests.RequestException as exc:
            return self._fail(f"JEF unreachable at {base}: {exc}")

        if response.status_code >= 400:
            try:
                detail = response.json().get("error", {})
                return self._fail(
                    f"{detail.get('code', 'http_error')}: {detail.get('message', '')}"
                )
            except ValueError:
                return self._fail(f"HTTP {response.status_code}: {response.text[:400]}")

        return json.dumps(response.json(), ensure_ascii=False)

    @staticmethod
    def _fail(message: str) -> str:
        return json.dumps({"success": False, "error": message}, ensure_ascii=False)

    @staticmethod
    def _parse_state(state: str) -> Any:
        """Text stays text; JSON objects and arrays are sent as structure.

        Structure matters: a model reading ``{"host": "x", "severity": 4}`` sees
        the fields a human sees, where a flattened string does not.
        """
        text = (state or "").strip()
        if text.startswith(("{", "[")):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Genuinely text that happens to start with a brace.
                return text
        return text

    @staticmethod
    def _lines(raw: str) -> list[str]:
        return [line.strip() for line in (raw or "").splitlines() if line.strip()]

    # -- actions ------------------------------------------------------------ #

    def run_scene(self, url: str, apikey: str, scene: str, state: str) -> str:
        return self._request(
            url, apikey, f"/v1/scenes/{scene}:evaluate", {"state": self._parse_state(state)}
        )

    def ask_choice(self, url: str, apikey: str, state: str, instructions: str, options: str) -> str:
        criteria: dict[str, str | None] = {}
        for line in self._lines(options):
            key, _, description = line.partition("=")
            criteria[key.strip()] = description.strip() or None
        if len(criteria) < 2:
            return self._fail("a choice needs at least two 'key=description' options")

        return self._request(
            url,
            apikey,
            "/v1/systemone",
            {
                "state": self._parse_state(state),
                "questions": {
                    "answer": {
                        "type": "choice",
                        "instructions": instructions,
                        "criteria": criteria,
                    }
                },
            },
        )

    def ask_score(self, url: str, apikey: str, state: str, instructions: str, levels: str) -> str:
        ordered = self._lines(levels)
        if len(ordered) < 2:
            return self._fail("a score needs at least two ordered levels, lowest first")
        if len(set(ordered)) != len(ordered):
            return self._fail("score levels must be distinct")

        return self._request(
            url,
            apikey,
            "/v1/systemone",
            {
                "state": self._parse_state(state),
                "questions": {
                    "answer": {
                        "type": "score",
                        "instructions": instructions,
                        "criteria": ordered,
                    }
                },
            },
        )

    def ask_yes_no(
        self,
        url: str,
        apikey: str,
        state: str,
        instructions: str,
        if_true: str = "",
        if_false: str = "",
    ) -> str:
        question: dict[str, Any] = {"type": "noul", "instructions": instructions}
        criteria = {k: v for k, v in (("true", if_true), ("false", if_false)) if v}
        if criteria:
            question["criteria"] = criteria

        return self._request(
            url,
            apikey,
            "/v1/systemone",
            {"state": self._parse_state(state), "questions": {"answer": question}},
        )

    def ask_many(self, url: str, apikey: str, state: str, questions: str) -> str:
        try:
            parsed = json.loads(questions)
        except json.JSONDecodeError as exc:
            return self._fail(f"questions must be a JSON object: {exc}")
        if not isinstance(parsed, dict) or not parsed:
            return self._fail("questions must be a non-empty JSON object of id -> question")

        return self._request(
            url,
            apikey,
            "/v1/systemone",
            {"state": self._parse_state(state), "questions": parsed},
        )

    def health(self, url: str, apikey: str) -> str:
        return self._request(url, apikey, "/healthz", None)


if __name__ == "__main__":
    JEF.run()
