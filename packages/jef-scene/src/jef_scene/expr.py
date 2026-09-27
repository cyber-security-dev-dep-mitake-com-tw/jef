"""A deliberately small expression language for gate conditions.

Gate conditions come from YAML, and YAML in a SOAR deployment comes from
wherever playbooks come from -- a repo, a UI, a colleague. So this is a
whitelist over Python's own parser, never ``eval`` with a blocklist:

* the AST is walked before execution and **any** node type not explicitly
  permitted is a load-time error;
* there are no function calls, no imports, no attribute access on anything but
  the injected answer objects, no subscripting except into ``probabilities``;
* names resolve only from the namespace built for that layer.

The result is that a malformed or malicious condition fails when the scene is
loaded, not when an incident is being triaged.

Supported:

    refund.probability > 0.85
    owner.confidence >= 0.9 and severity.score >= 3
    owner.p_correct is not None and owner.p_correct > 0.95
    owner.probabilities["soc"] > 0.5
    not is_false_positive.probability > 0.5
    severity.score >= 3 or (urgent.probability > 0.7 and owner.confidence > 0.8)
"""

from __future__ import annotations

import ast
from typing import Any

from jef_core.errors import SceneError

__all__ = ["ExpressionError", "compile_condition", "evaluate_condition", "validate_condition"]


class ExpressionError(SceneError):
    """A gate condition is malformed or uses something outside the whitelist."""

    code = "invalid_expression"


#: Every node type a condition may contain. Absent by design: Call, Import,
#: Lambda, comprehensions, assignment expressions, f-strings, starred args.
_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Attribute,
    ast.Subscript,
    ast.Tuple,
    ast.List,
)

#: Attributes a gate may read off an answer. Anything else -- including every
#: dunder -- is rejected at load time.
ALLOWED_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "choice",
        "score",
        "probability",
        "noul",
        "confidence",
        "p_correct",
        "probabilities",
        "prediction_set",
        "set_size",
        "type",
    }
)


def _check(tree: ast.AST, names: set[str]) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(f"{type(node).__name__} is not allowed in a gate condition")
        if isinstance(node, ast.Attribute) and node.attr not in ALLOWED_ATTRIBUTES:
            raise ExpressionError(
                f"attribute {node.attr!r} is not readable from a gate condition; "
                f"allowed: {', '.join(sorted(ALLOWED_ATTRIBUTES))}"
            )
        if isinstance(node, ast.Name) and node.id not in names:
            raise ExpressionError(
                f"unknown name {node.id!r} in gate condition; this layer defines: "
                f"{', '.join(sorted(names)) or '(nothing)'}"
            )


def validate_condition(source: str, names: set[str]) -> None:
    """Parse and whitelist-check without executing. Raises on any violation."""
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"cannot parse gate condition {source!r}: {exc.msg}") from exc
    _check(tree, names)


def compile_condition(source: str, names: set[str]) -> Any:
    """Validate then compile. Scenes compile every condition at load time."""
    validate_condition(source, names)
    return compile(ast.parse(source, mode="eval"), filename="<gate>", mode="eval")


def evaluate_condition(code: Any, namespace: dict[str, Any]) -> bool:
    """Run a compiled condition.

    ``__builtins__`` is emptied so that even if the whitelist were bypassed
    there is nothing reachable to call.
    """
    try:
        result = eval(code, {"__builtins__": {}}, dict(namespace))  # noqa: S307
    except Exception as exc:
        raise ExpressionError(f"gate condition failed at runtime: {exc}") from exc
    return bool(result)
