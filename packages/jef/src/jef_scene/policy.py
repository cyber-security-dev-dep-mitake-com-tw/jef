"""Runtime policy: may the action a gate selected be carried out without a human?

A scene gate expresses what the *model* thinks. ``human_review: false`` on an
action expresses what the *scene author* is willing to automate. Neither of
those is an authorization. The gap this module closed was concrete: an
earlier ``scenes/incident-triage.zh-tw.yaml`` selected ``close``
(``human_review: false``) on ``is_false_positive.probability > 0.90``, a raw
model probability from a bucket nobody had calibrated for that task, and a
connector reading ``human_review`` would have closed the alert. (The engine's
advisory guard now also keeps such a gate from *proposing* the action; see
:class:`jef_scene.schema.Advisory`. This module is the layer underneath it.)

The verdict here is deterministic and sits between the scene and whatever
dispatches actions. It can turn an automating action into ``review_required``;
nothing in a scene, a model output or a request body can turn
``review_required`` back into ``allow``. Two rules:

1. **Tier.** Actions are classed T0 to T4 (see ``tier_of``). Only T0 and T1
   (read-only, routing, enrichment, ticketing) may run without a human. Closing,
   suppressing, isolating, disabling or blocking never does, whatever the
   probability, because a wrong one is not undone by a better model.
2. **Evidence.** An automating action must rest on questions whose calibration
   is ``validated`` (``Calibrator.status``: a correctness map and a conformal
   quantile at the scene's alpha, both fitted on enough data). ``missing`` and
   ``insufficient`` are both "cannot say" and give ``review_required``. A gate
   that read nothing (an ``else``, or the scene fallthrough) cannot automate at
   all.

This is the first, intentionally small, policy layer: upgrade plan item 03
and the v2 review's P0-A. Task-scoped calibration status, signed
authorization grants and execution receipts build on the verdict shape here
rather than replacing it.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .engine import AnswerView
    from .schema import Action

__all__ = [
    "MAX_AUTO_TIER",
    "PolicyVerdict",
    "Verdict",
    "authorize",
    "inferred_tier",
    "referenced_questions",
    "tier_of",
]

Verdict = Literal["allow", "review_required"]

#: The highest tier a scene may carry out without a human. T0 reads, T1 routes.
MAX_AUTO_TIER = 1

#: Name patterns that mark an action as state-changing. Deliberately broad:
#: a false positive here costs a human a look, a false negative closes an
#: alert nobody saw. An explicit ``tier`` can raise an action above its
#: name's tier, never lower it: a scene that declares ``close_alert`` T1 still
#: gets T2 and ``AUTO_CLOSE_PROHIBITED``. Closing or suppressing is never
#: automated, whatever the YAML says; an action wrongly caught by a pattern is
#: renamed, not declared safe.
_TIER_PATTERNS: tuple[tuple[int, re.Pattern[str]], ...] = (
    (
        3,
        re.compile(
            r"isolat|disabl|block|quarantin|delet|revok|kill|terminat|wipe|lock|"
            r"reset_password|contain|shutdown|purge|ban"
        ),
    ),
    (2, re.compile(r"close|suppress|dismiss|resolve|mute|silence|snooze|ignore|archive")),
)


def inferred_tier(name: str) -> int:
    """Tier guessed from an action's name alone. T1 when nothing matches."""
    lowered = name.lower()
    for tier, pattern in _TIER_PATTERNS:
        if pattern.search(lowered):
            return tier
    return 1


def tier_of(name: str, action: Action) -> int:
    """The higher of the action's explicit ``tier`` and the tier its name implies.

    A declared tier is a floor raise, not an override: lowering a close,
    suppress, isolate or block action below its name's tier would let a scene
    automate it by editing one line of YAML.
    """
    inferred = inferred_tier(name)
    return inferred if action.tier is None else max(action.tier, inferred)


def referenced_questions(condition: str | None, available: Iterable[str]) -> set[str]:
    """Question ids a gate condition reads. Empty for an ``else`` gate."""
    if not condition:
        return set()
    names = set(available)
    try:
        tree = ast.parse(condition, mode="eval")
    except SyntaxError:  # pragma: no cover -- the scene loader compiled it already
        return set()
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id in names}


@dataclass
class PolicyVerdict:
    """Whether the selected action may be dispatched without a person."""

    verdict: Verdict
    tier: int
    #: Machine-readable reasons, stable strings a dashboard can count on.
    #: ``UNVALIDATED_CALIBRATION:<question>`` names the question whose bucket
    #: is not ``validated`` (the question's trace says whether it is
    #: ``missing`` or ``insufficient``); the others are fixed.
    reason_codes: list[str] = field(default_factory=list)

    @property
    def human_review(self) -> bool:
        return self.verdict != "allow"


def authorize(
    name: str,
    action: Action,
    *,
    referenced: set[str],
    views: Mapping[str, AnswerView],
) -> PolicyVerdict:
    """Decide whether ``action`` (selected by a gate that read ``referenced``) may run.

    Pure: it reads the answers' calibration state and the action's tier and
    returns a verdict. It never consults the probability a gate compared
    against -- a 0.99999 that nobody calibrated is exactly the input this
    exists to stop.
    """
    tier = tier_of(name, action)
    if action.human_review:
        return PolicyVerdict("review_required", tier, ["HUMAN_REVIEW_ACTION"])

    reasons: list[str] = []
    if tier > MAX_AUTO_TIER:
        reasons.append(
            "AUTO_CLOSE_PROHIBITED" if tier == 2 else "DISRUPTIVE_ACTION_REQUIRES_APPROVAL"
        )
    if not referenced:
        # An else gate or the fallthrough: the scene reached this action by
        # elimination, not by reading evidence it can stand behind.
        reasons.append("NO_EVIDENCE_QUESTION")
    for qid in sorted(referenced):
        view = views.get(qid)
        if view is None or view.calibration_status != "validated":
            reasons.append(f"UNVALIDATED_CALIBRATION:{qid}")

    return PolicyVerdict("allow" if not reasons else "review_required", tier, reasons)
