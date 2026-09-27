"""Robot library that drives the SOAR connectors against a live JEF server.

The pytest suite in connectors/tests covers request shaping with the SDKs
stubbed out. This covers the other half: that the shapes those connectors build
are ones a real server actually accepts, and that what comes back is what a
playbook would branch on.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SHUFFLE_APP = ROOT / "connectors" / "shuffle" / "jef" / "1.0.0" / "src" / "app.py"


def _install_shuffle_stub() -> None:
    if "shuffle_sdk" in sys.modules:
        return
    module = types.ModuleType("shuffle_sdk")

    class AppBase:
        def __init__(
            self, redis: Any = None, logger: Any = None, console_logger: Any = None
        ) -> None:
            pass

        @classmethod
        def run(cls) -> None:
            return None

    module.AppBase = AppBase  # type: ignore[attr-defined]
    sys.modules["shuffle_sdk"] = module


def _load_app() -> Any:
    _install_shuffle_stub()
    spec = importlib.util.spec_from_file_location("jef_shuffle_live", SHUFFLE_APP)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["jef_shuffle_live"] = module
    spec.loader.exec_module(module)
    return module.JEF()


class ConnectorLibrary:
    """Keywords for exercising the connectors end to end."""

    ROBOT_LIBRARY_SCOPE = "SUITE"

    def __init__(self) -> None:
        self._app = _load_app()

    def shuffle_run_scene(self, base_url: str, scene: str, state: str) -> dict[str, Any]:
        return json.loads(self._app.run_scene(base_url, "", scene, state))

    def shuffle_ask_choice(
        self, base_url: str, state: str, instructions: str, options: str
    ) -> dict[str, Any]:
        return json.loads(self._app.ask_choice(base_url, "", state, instructions, options))

    def shuffle_ask_score(
        self, base_url: str, state: str, instructions: str, levels: str
    ) -> dict[str, Any]:
        return json.loads(self._app.ask_score(base_url, "", state, instructions, levels))

    def shuffle_ask_yes_no(self, base_url: str, state: str, instructions: str) -> dict[str, Any]:
        return json.loads(self._app.ask_yes_no(base_url, "", state, instructions))

    def shuffle_ask_many(self, base_url: str, state: str, questions: str) -> dict[str, Any]:
        return json.loads(self._app.ask_many(base_url, "", state, questions))

    def shuffle_health(self, base_url: str) -> dict[str, Any]:
        return json.loads(self._app.health(base_url, ""))

    def cortex_scene_taxonomies(self, trace: dict[str, Any]) -> list[dict[str, Any]]:
        """Run a real trace through the Cortex summary, as Cortex would.

        Loaded by path rather than by name: Robot puts the suite directory on
        sys.path, not this resources directory, so a plain import resolves only
        by accident of how the suite was invoked.
        """
        bridge_path = Path(__file__).with_name("connector_cortex_bridge.py")
        spec = importlib.util.spec_from_file_location("connector_cortex_bridge", bridge_path)
        assert spec and spec.loader
        bridge = importlib.util.module_from_spec(spec)
        sys.modules["connector_cortex_bridge"] = bridge
        spec.loader.exec_module(bridge)
        return list(bridge.summarise_scene(trace))
