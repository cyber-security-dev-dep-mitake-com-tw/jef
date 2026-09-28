"""Scene linting, without loading a model.

`jef_scene` already rejects a lot at load time -- an `else` gate that makes
later gates unreachable, a gate targeting an action that does not exist, a final
layer that can fall through without deciding. This adds the checks that need to
look across a whole scene, and the one check that encodes this project's actual
thesis: a gate that automates on `confidence` is thresholding on how decisive
the model sounded, not on how likely it is to be right.

Runs with no backbone, so it belongs in CI on every commit.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jef_core.errors import SceneError
from jef_scene import Scene, load_scene_text
from jef_scene.expr import ExpressionError, validate_condition

__all__ = ["Problem", "discover_scenes", "validate_files", "validate_paths", "validate_scene_text"]

#: Fields whose value cannot exceed 1.0, so a comparison above it never fires.
_UNIT_FIELDS = {"confidence", "p_correct", "probability", "noul"}


@dataclass
class Problem:
    path: str
    severity: str  # 'error' | 'warning'
    message: str
    line: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "line": self.line,
            "severity": self.severity,
            "message": self.message,
        }


def _line_of(text: str, needle: str) -> int | None:
    """Best-effort line number for a fragment, so errors are navigable."""
    if not needle:
        return None
    for number, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return number
    return None


def _referenced_names(condition: str) -> set[str]:
    """Question ids a condition reads."""
    try:
        tree = ast.parse(condition, mode="eval")
    except SyntaxError:
        return set()
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _attributes_on(condition: str, name: str) -> set[str]:
    """Attributes read from one question in a condition."""
    try:
        tree = ast.parse(condition, mode="eval")
    except SyntaxError:
        return set()
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == name
    }


def _asserts_high_confidence(condition: str, name: str) -> bool:
    """Whether a condition acts *because* confidence is high.

    The distinction matters. `confidence < 0.15` is detecting indecision, which
    is exactly what a sharpness statistic measures and a correct use of it.
    `confidence >= 0.9` is claiming the answer is probably right, which is a
    different quantity -- that is the one worth flagging.
    """
    try:
        tree = ast.parse(condition, mode="eval")
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        left, op, right = node.left, node.ops[0], node.comparators[0]
        reads_confidence = (
            isinstance(left, ast.Attribute)
            and left.attr == "confidence"
            and isinstance(left.value, ast.Name)
            and left.value.id == name
        )
        if reads_confidence and isinstance(op, (ast.Gt, ast.GtE)):
            return True
        # The mirrored form: `0.9 <= x.confidence`.
        reads_confidence_right = (
            isinstance(right, ast.Attribute)
            and right.attr == "confidence"
            and isinstance(right.value, ast.Name)
            and right.value.id == name
        )
        if reads_confidence_right and isinstance(op, (ast.Lt, ast.LtE)):
            return True
    return False


def _numeric_literal(node: ast.expr) -> float | None:
    """A numeric constant, including a negated one.

    `-1` parses as UnaryOp(USub, Constant(1)), not Constant(-1), so matching
    only on Constant silently skips every negative threshold.
    """
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
        value = node.operand.value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(node.op, ast.USub):
                return -float(value)
            if isinstance(node.op, ast.UAdd):
                return float(value)
    return None


def _impossible_comparisons(condition: str) -> list[str]:
    """Comparisons that can never be true, e.g. `x.confidence > 1.0`.

    A gate that can never fire is dead config, and the usual cause is a
    percentage written where a probability was meant -- `confidence > 90`
    rather than `> 0.9`. That one silently disables a gate forever.
    """
    try:
        tree = ast.parse(condition, mode="eval")
    except SyntaxError:
        return []

    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        left, op, right = node.left, node.ops[0], node.comparators[0]
        if not (isinstance(left, ast.Attribute) and left.attr in _UNIT_FIELDS):
            continue
        value = _numeric_literal(right)
        if value is None:
            continue
        rendered = f"{left.attr} {type(op).__name__} {value}"
        if isinstance(op, (ast.Gt, ast.GtE)) and value > 1.0:
            findings.append(
                f"`{rendered}` can never be true ({left.attr} is 0..1; did you mean {value / 100:g}?)"
            )
        elif isinstance(op, (ast.Lt, ast.LtE)) and value < 0.0:
            findings.append(f"`{rendered}` can never be true ({left.attr} is 0..1)")
    return findings


def validate_scene_text(text: str, source: str) -> list[Problem]:
    """Lint one scene document."""
    problems: list[Problem] = []

    try:
        scene: Scene = load_scene_text(text, source=source)
    except SceneError as exc:
        # Schema failures are fatal for this file; nothing below can run.
        message = str(exc).replace(f"{source}: ", "", 1)
        return [Problem(source, "error", message.splitlines()[0], _line_of(text, "scene:"))]
    except yaml.YAMLError as exc:  # pragma: no cover -- loader wraps these
        return [Problem(source, "error", f"invalid YAML: {exc}", None)]

    targeted: set[str] = set()
    if scene.fallthrough:
        targeted.add(scene.fallthrough)

    # Names accumulate across layers, exactly as the engine does it: a gate may
    # read any question answered so far. Scoping per layer here would report
    # errors on scenes that run correctly.
    names: set[str] = set()
    used_anywhere: set[str] = set()
    defined_in: dict[str, str] = {}

    for layer in scene.layers:
        names |= set(layer.questions)
        for qid in layer.questions:
            defined_in[qid] = layer.id

        for index, gate in enumerate(layer.gates):
            targeted.add(gate.then)
            if gate.when is None:
                continue

            try:
                validate_condition(gate.when, names)
            except ExpressionError as exc:
                problems.append(
                    Problem(source, "error", f"layer {layer.id}: {exc}", _line_of(text, gate.when))
                )
                continue

            used_anywhere |= _referenced_names(gate.when)

            for finding in _impossible_comparisons(gate.when):
                problems.append(
                    Problem(
                        source,
                        "error",
                        f"layer {layer.id} gate {index}: {finding}",
                        _line_of(text, gate.when),
                    )
                )

            # The thesis check. `confidence` is distribution sharpness; a gate
            # that acts *because* it is high is saying "the model sounded sure",
            # which is not the same as "the model is likely right". Flagged only
            # when the gate both asserts high confidence and leads somewhere no
            # human will see -- escalating on low confidence is correct use.
            action = scene.actions.get(gate.then)
            automating = action is not None and not action.human_review
            if automating:
                for name in _referenced_names(gate.when):
                    attributes = _attributes_on(gate.when, name)
                    if (
                        "confidence" in attributes
                        and "p_correct" not in attributes
                        and _asserts_high_confidence(gate.when, name)
                    ):
                        problems.append(
                            Problem(
                                source,
                                "warning",
                                f"layer {layer.id} gate {index} automates to "
                                f"'{gate.then}' on {name}.confidence. confidence "
                                "measures how peaked the distribution is, not the "
                                "chance of being correct — consider p_correct",
                                _line_of(text, gate.when),
                            )
                        )

    # Checked after every layer, because a question may be read by a gate in a
    # later layer than the one that asks it.
    for unused in sorted(names - used_anywhere):
        problems.append(
            Problem(
                source,
                "warning",
                f"layer {defined_in[unused]}: question '{unused}' is asked but no gate reads it",
                _line_of(text, f"{unused}:"),
            )
        )

    for orphan in sorted(set(scene.actions) - targeted):
        problems.append(
            Problem(
                source,
                "warning",
                f"action '{orphan}' is declared but never reachable",
                _line_of(text, f"{orphan}:"),
            )
        )

    return problems


def discover_scenes(paths: list[Path]) -> tuple[list[Path], list[Problem]]:
    """Expand files and directories into scene files, and say what was missing.

    Split out from `validate_paths` so the CLI can report how many *files* it
    checked. It used to print the number of path arguments, so linting a
    directory of twenty scenes said "1 path(s) checked" -- indistinguishable
    from a glob that matched nothing.
    """
    files: list[Path] = []
    problems: list[Problem] = []
    for path in paths:
        if path.is_dir():
            found = sorted([*path.rglob("*.yaml"), *path.rglob("*.yml")])
            if not found:
                problems.append(Problem(str(path), "warning", "no scene files found"))
            files.extend(found)
        elif path.is_file():
            files.append(path)
        else:
            problems.append(Problem(str(path), "error", "no such file or directory"))
    return files, problems


def validate_files(files: list[Path]) -> list[Problem]:
    """Lint scene files that have already been discovered."""
    problems: list[Problem] = []
    for file in files:
        try:
            text = file.read_text(encoding="utf-8")
        except OSError as exc:
            problems.append(Problem(str(file), "error", f"cannot read: {exc}"))
            continue
        problems.extend(validate_scene_text(text, str(file)))
    return problems


def validate_paths(paths: list[Path]) -> list[Problem]:
    """Lint every scene under the given files or directories."""
    files, problems = discover_scenes(paths)
    return problems + validate_files(files)
