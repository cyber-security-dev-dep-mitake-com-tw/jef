"""Question builders.

Writing questions as dict literals works and is what the wire contract is, but
it puts the two things most worth getting right -- option order and the yes/no
dialect -- in the author's hands every single time. These helpers make the
common case short and the ordering explicit.
"""

from __future__ import annotations

from typing import Any

__all__ = ["boolean", "choice", "noul", "score"]


def choice(instructions: str, **options: str | None) -> dict[str, Any]:
    """A choice question. Keyword order is the option order the model sees.

    choice("誰處理？", soc="監控事件", appsec="程式碼相關")
    """
    if len(options) < 2:
        raise ValueError("a choice question needs at least two options")
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


def score(instructions: str, *levels: str) -> dict[str, Any]:
    """A score question. Levels are ordinal and must be given lowest first.

        score("嚴重度", "低", "中", "高")

    Order matters more than it looks: the answer is the expectation over this
    ordering, so reversing it silently inverts the scale rather than erroring.
    """
    if len(levels) < 2:
        raise ValueError("a score question needs at least two ordered levels")
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


def noul(instructions: str, *, true: str | None = None, false: str | None = None) -> dict[str, Any]:
    """A yes/no question in TypeSafe's native spelling."""
    q: dict[str, Any] = {"type": "noul", "instructions": instructions}
    criteria = {k: v for k, v in (("true", true), ("false", false)) if v is not None}
    if criteria:
        q["criteria"] = criteria
    return q


def boolean(
    instructions: str, *, true: str | None = None, false: str | None = None
) -> dict[str, Any]:
    """The same primitive, in the Vercel AI SDK's spelling.

    Provided so code ported from ``experimental_evaluate`` reads the same here.
    The answer comes back under ``probability`` rather than ``noul``, matching
    whichever spelling was asked.
    """
    q = noul(instructions, true=true, false=false)
    q["type"] = "boolean"
    return q
