"""Remote systems asked the five incident-triage questions, on the same terms as JEF.

``RemoteTriageEngine`` puts a ``remote.RemoteScorer`` (a ``/v1/systemone``
service or the AI Gateway decision-model dialect: TypeSafe Jev, Laya, an LLM
wrapper, another JEF) behind the engine interface ``triage_eval`` already
drives, so a remote system is scored by exactly the code that scores JEF:

* it receives the scene's own questions -- instructions, options, option
  order -- with the same state, in one request per case (per layer in scene
  mode, as JEF is asked);
* every answer must be a full, valid distribution over the question's
  options. A refusal, a missing question, a partial or non-normalised
  distribution is a *contract failure*: the question is left unanswered, so
  the harness excludes and counts it, and ``failures`` says why. Nothing is
  filled in;
* it has no calibration, so every question reports ``missing`` and no gate
  that needs a validated bucket can fire on its answers;
* the credential is read from the environment by the scorer and never
  written; ``describe()`` names provider, model, version and data flow.

``egress_check`` is the gate in front of all this: an external endpoint needs
an explicit ``--allow-egress <host>``, and private cases (own telemetry,
reviewer-sourced alerts) never go to an external endpoint at all.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from jef_core import Engine
from jef_core.engine import SharedState
from jef_core.types import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    Question,
    RawQuestion,
    ScoreAnswer,
    normalize,
)

from .remote import (
    GatewayScorer,
    NoDistributionError,
    NotADistributionError,
    RefusalError,
    RemoteScorer,
    ScorerContractError,
    strict_distribution,
)

__all__ = ["PRIVATE_ORIGINS", "EgressRefused", "RemoteTriageEngine", "egress_check"]

#: Case origins that never leave the machine for an external model.
PRIVATE_ORIGINS = frozenset({"own-telemetry", "reviewer-sourced"})


class EgressRefused(RuntimeError):
    """Cases would leave the machine without the approval that requires."""


def egress_check(
    scorer: RemoteScorer, cases: Iterable[Mapping[str, Any]], allow_egress: Iterable[str]
) -> None:
    """Raise unless sending ``cases`` to ``scorer`` is allowed."""
    if scorer.egress == "loopback":
        return
    allowed = set(allow_egress)
    if scorer.host not in allowed:
        raise EgressRefused(
            f"{scorer.name} is external ({scorer.host}); pass --allow-egress {scorer.host} "
            "only for public or approved cases"
        )
    private = sorted({str(c.get("origin")) for c in cases} & PRIVATE_ORIGINS)
    if private:
        raise EgressRefused(
            f"refusing to send {', '.join(private)} cases to external {scorer.host}: private "
            "telemetry and reviewer-sourced alerts need a separate data-processing and egress "
            "approval, and this tool does not take one"
        )


def _wire(question: Question | RawQuestion) -> dict[str, Any]:
    if hasattr(question, "model_dump"):
        return question.model_dump(exclude_none=True)
    return dict(question)


def _parse(qid: str, question: Question | RawQuestion, reply: Any) -> Answer:
    if not isinstance(reply, Mapping):
        raise NoDistributionError(f"{qid}: no answer")
    if reply.get("type") == "refusal":
        raise RefusalError(f"{qid}: the system refused")
    q = normalize(qid, question)
    keys = list(q.option_keys)
    if q.kind == "noul":
        p = reply.get("noul", reply.get("probability"))
        if p is None and isinstance(reply.get("probabilities"), Mapping):
            p = strict_distribution(["false", "true"], reply["probabilities"])["true"]
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p):
            raise NoDistributionError(f"{qid}: no probability of true")
        if not 0.0 <= float(p) <= 1.0:
            raise NotADistributionError(f"{qid}: {p!r} is not a probability")
        p = float(p)
        return NoulAnswer(noul=p, confidence=abs(2 * p - 1))
    probs = reply.get("probabilities")
    if not isinstance(probs, Mapping):
        raise NoDistributionError(f"{qid}: a choice without probabilities")
    dist = strict_distribution(keys, probs)  # keyed by label: the reply's order does not matter
    top = max(keys, key=lambda k: dist[k])
    if q.kind == "choice":
        return ChoiceAnswer(choice=top, probabilities=dist, confidence=dist[top])
    expected = sum(i * dist[k] for i, k in enumerate(keys))
    return ScoreAnswer(score=expected, legend=keys, probabilities=dist, confidence=dist[top])


class RemoteTriageEngine(Engine):
    """Engine interface over a remote scorer. No calibration: every bucket is ``missing``."""

    def __init__(self, scorer: RemoteScorer) -> None:
        super().__init__()
        self.scorer = scorer
        self.filled: list[str] = []
        self.failures: list[dict[str, str]] = []
        self._state: Any = None
        self._case: str | None = None

    def for_case(self, case: Mapping[str, Any]) -> RemoteTriageEngine:
        self._case = str(case.get("case_id"))
        return self

    def prepare(self, state: Any) -> SharedState:
        self._state = state
        return super().prepare(state)

    def answer(
        self, shared: SharedState, questions: Mapping[str, Question | RawQuestion]
    ) -> dict[str, Answer]:
        wire = {qid: _wire(q) for qid, q in questions.items()}
        try:
            reply = self._post(wire)
        except ScorerContractError as exc:
            self._fail("*", type(exc).__name__, str(exc))
            return {}
        except Exception as exc:  # transport: counted, never raised into the harness
            self._fail("*", "TransportError", type(exc).__name__)
            return {}
        answers = reply.get("answers") if isinstance(reply, Mapping) else None
        if not isinstance(answers, Mapping):
            self._fail("*", "NoDistributionError", "reply has no answers")
            return {}
        out: dict[str, Answer] = {}
        for qid, question in questions.items():
            try:
                out[qid] = _parse(qid, question, answers.get(qid))
            except ScorerContractError as exc:
                self._fail(qid, type(exc).__name__, str(exc))
        return out

    def _post(self, wire: Mapping[str, Any]) -> Any:
        s = self.scorer
        state = self._state if isinstance(self._state, str) else str(self._state)
        if isinstance(s, GatewayScorer):
            return s._call(
                f"{s.url}/decision-model",
                {"stateParts": [{"type": "text", "text": state}], "questions": dict(wire)},
                {"ai-decision-model-specification-version": "4", "ai-model-id": s.model},
            )
        return s._call(
            f"{s.url}/v1/systemone", {"model": s.model, "state": state, "questions": dict(wire)}
        )

    def _fail(self, qid: str, kind: str, detail: str) -> None:
        self.failures.append(
            {"case_id": self._case or "?", "question": qid, "kind": kind, "detail": detail[:200]}
        )
