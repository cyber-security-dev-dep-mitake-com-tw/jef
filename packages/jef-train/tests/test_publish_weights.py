"""The release job runs this module without installing the training stack.

`uv run` at the workspace root syncs a virtual root package with no members,
so the weights job used to die on `ModuleNotFoundError: jef_train` after two
minutes of installing dependencies it did not need. It now runs with
`--no-project` and `PYTHONPATH=packages/jef-train/src`, which works only
because this module imports nothing heavier than the standard library and,
lazily, huggingface_hub. Nothing else would notice that changing: every other
caller has the whole workspace installed.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "src" / "jef_train" / "publish_weights.py"

#: Imported inside the function that uses it, so it is not needed to load the
#: module -- but the job does install it.
ALLOWED_THIRD_PARTY = {"huggingface_hub"}


def _top_level_imports(path: Path) -> set[str]:
    """Module names imported at import time, ignoring imports inside functions."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:  # top level only
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_publishing_needs_neither_torch_nor_the_rest_of_jef() -> None:
    imported = _top_level_imports(MODULE)
    outside = imported - sys.stdlib_module_names - ALLOWED_THIRD_PARTY
    assert not outside, (
        f"{MODULE.name} gained a heavy import at module level: {sorted(outside)}. "
        "The release workflow runs it with --no-project, so this would fail the "
        "weights job -- move the import into the function that needs it, or "
        "change the job to install the package."
    )
