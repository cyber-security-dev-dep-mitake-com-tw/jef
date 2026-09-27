"""Human-readable rendering of answers and traces.

The terminal output deliberately mirrors the web viewer in `packages/jef-ui`:
probability bars, `p_correct` next to `confidence`, and options outside the
conformal set greyed rather than hidden. Someone who has read one should be able
to read the other without relearning anything.
"""

from __future__ import annotations

import os
import shutil
import sys
import unicodedata
from typing import Any

__all__ = [
    "Style",
    "display_width",
    "pad",
    "render_answer",
    "render_trace",
    "warn_if_untrustworthy",
]


class Style:
    """ANSI styling that switches itself off when it would be noise.

    Honours NO_COLOR and disables itself when stdout is not a terminal, so
    `jef ask ... > file` and `jef ask ... | jq` produce clean text.
    """

    def __init__(self, enabled: bool | None = None) -> None:
        if enabled is None:
            enabled = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)


def _bar(fraction: float, width: int) -> str:
    """A probability bar using eighth-blocks, so small values stay visible.

    Rounding a 3% slice to zero full blocks would make it look like zero, which
    is exactly the distinction a distribution is meant to show.
    """
    eighths = round(max(0.0, min(1.0, fraction)) * width * 8)
    full, remainder = divmod(eighths, 8)
    partial = " ▏▎▍▌▋▊▉"[remainder] if remainder else ""
    return ("█" * full + partial).ljust(width)


def _terminal_width(default: int = 88) -> int:
    return min(shutil.get_terminal_size((default, 24)).columns, 110)


def display_width(text: str) -> int:
    """Columns a string occupies in a terminal, not characters.

    Full-width and wide East Asian characters take two columns. Padding by
    character count leaves every CJK label misaligned -- `資訊` and `低` are one
    and two characters but four and two columns -- which is not acceptable in a
    project where zh-TW is a first-class language.
    """
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)


def pad(text: str, width: int) -> str:
    """Left-align to a display width, truncating on a column boundary."""
    current = display_width(text)
    if current <= width:
        return text + " " * (width - current)

    out, used = "", 0
    for char in text:
        char_width = 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        if used + char_width > width - 1:
            break
        out += char
        used += char_width
    return out + "…" + " " * max(0, width - used - 1)


def render_answer(
    qid: str,
    answer: dict[str, Any],
    style: Style,
    *,
    prediction_set: list[str] | None = None,
    p_correct: float | None = None,
    indent: str = "",
) -> str:
    """One answer, as a distribution with its statistics underneath."""
    lines: list[str] = []
    kind = answer.get("type", "?")

    probabilities: dict[str, float]
    if "probabilities" in answer:
        probabilities = dict(answer["probabilities"])
    else:
        p = float(answer.get("noul", answer.get("probability", 0.0)))
        probabilities = {"false": round(1.0 - p, 6), "true": round(p, 6)}

    chosen = answer.get("choice")
    if chosen is None and probabilities:
        chosen = max(probabilities, key=lambda k: probabilities[k])

    key_width = min(max((display_width(k) for k in probabilities), default=6), 24)
    bar_width = max(12, _terminal_width() - key_width - len(indent) - 22)

    lines.append(f"{indent}{style.cyan(qid)} {style.dim(f'· {kind}')}")
    in_set = set(prediction_set) if prediction_set else None
    for key, value in probabilities.items():
        excluded = in_set is not None and key not in in_set
        bar = _bar(value, bar_width)
        row = f"{indent}  {pad(key, key_width)} {bar} {value * 100:5.1f}%"
        if key == chosen:
            row = style.bold(row)
        elif excluded:
            # Excluded by the conformal set: shown, not hidden. A reader needs
            # to see what was ruled out as well as what was kept.
            row = style.dim(row)
        lines.append(row)

    stats = [f"confidence {answer.get('confidence', 0):.3f}"]
    if p_correct is None:
        # Never a number, never zero: "we cannot say how reliable this is" and
        # "this is unreliable" lead to different decisions.
        stats.append(style.yellow("P(correct) unknown"))
    else:
        stats.append(f"P(correct) {p_correct:.3f}")
    if "score" in answer:
        stats.append(f"score {answer['score']:.3f}")
    if prediction_set is not None:
        stats.append(f"conformal set {len(prediction_set)}/{len(probabilities)}")
    lines.append(f"{indent}  {style.dim('  '.join(stats))}")
    return "\n".join(lines)


def render_trace(trace: dict[str, Any], style: Style) -> str:
    """A scene run: verdict first, then each layer's evidence and gates."""
    lines: list[str] = []
    action = trace.get("action") or "(no decision)"
    decided = trace.get("verdict") == "decided"

    heading = style.bold(action)
    if trace.get("human_review"):
        heading += "  " + style.yellow("[needs human review]")
    else:
        heading += "  " + style.green("[automated]")
    lines.append(heading)

    badges = [
        f"scene {trace.get('scene')} v{trace.get('version')}",
        f"state encoded {trace.get('state_encodes')}×",
        f"{trace.get('questions_asked')} question(s) asked",
        "gate decided" if decided else "fell through",
        "calibrated" if trace.get("calibrated") else style.red("UNCALIBRATED"),
    ]
    lines.append(style.dim("  " + "  ·  ".join(badges)))
    lines.append("")

    for layer in trace.get("layers", []):
        lines.append(f"  {style.bold(layer['id'])} {style.dim('→ ' + layer['outcome'])}")
        for question in layer.get("questions", []):
            lines.append(
                render_answer(
                    question["id"],
                    question["answer"],
                    style,
                    prediction_set=question.get("prediction_set"),
                    p_correct=question.get("p_correct"),
                    indent="    ",
                )
            )
        for gate in layer.get("gates", []):
            mark = "else" if gate.get("is_else") else ("FIRED" if gate.get("fired") else "  -  ")
            marker = (
                style.green(mark) if gate.get("fired") or gate.get("is_else") else style.dim(mark)
            )
            condition = gate.get("condition") or "(default)"
            target = f" → {gate['then']}" if gate.get("then") else ""
            lines.append(f"    [{marker}] {style.dim(condition)}{target}")
            if gate.get("because"):
                lines.append(f"            {style.dim(gate['because'])}")
        lines.append("")

    for skipped in trace.get("layers_skipped", []):
        # Shown, not omitted: "never asked" is different from "asked and
        # inconclusive", and only one of them means the model saw the evidence.
        lines.append(style.dim(f"  {skipped} — not evaluated; an earlier layer decided"))

    return "\n".join(lines).rstrip()


def warn_if_untrustworthy(info: dict[str, Any], style: Style) -> None:
    """Print the two conditions under which every number above is meaningless."""
    if info.get("test_backbone"):
        print(
            style.red(
                "! This server runs the hashing test backbone: answers are "
                "well-formed and carry no meaning."
            ),
            file=sys.stderr,
        )
    if info.get("calibrated") is False:
        print(
            style.yellow(
                "! No calibration loaded: P(correct) is unavailable, the conformal "
                "set excludes nothing, and every scene gate will route to a human."
            ),
            file=sys.stderr,
        )
