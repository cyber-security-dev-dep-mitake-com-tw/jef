"""Bridge the Cortex analyzer's summary() into Robot without cortexutils.

Cortex injects config, owns stdout and exits the process on error, so importing
its SDK to check a taxonomy would test Cortex. This loads the analyzer with a
minimal stand-in and calls the one method that decides what an analyst sees.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ANALYZER = ROOT / "connectors" / "cortex" / "JEF" / "jef_analyzer.py"


def _install_cortex_stub() -> None:
    if "cortexutils.analyzer" in sys.modules:
        return
    cortexutils = types.ModuleType("cortexutils")
    analyzer_mod = types.ModuleType("cortexutils.analyzer")

    class Analyzer:
        def __init__(self) -> None:
            self._config: dict[str, Any] = {}

        def get_param(self, name: str, default: Any = None, message: str | None = None) -> Any:
            return self._config.get(name, default)

        def error(self, message: str) -> None:
            raise RuntimeError(message)

        @staticmethod
        def build_taxonomy(
            level: str, namespace: str, predicate: str, value: Any
        ) -> dict[str, Any]:
            return {"level": level, "namespace": namespace, "predicate": predicate, "value": value}

    analyzer_mod.Analyzer = Analyzer  # type: ignore[attr-defined]
    cortexutils.analyzer = analyzer_mod  # type: ignore[attr-defined]
    sys.modules["cortexutils"] = cortexutils
    sys.modules["cortexutils.analyzer"] = analyzer_mod


def summarise_scene(trace: dict[str, Any]) -> list[dict[str, Any]]:
    _install_cortex_stub()
    spec = importlib.util.spec_from_file_location("jef_analyzer_bridge", ANALYZER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["jef_analyzer_bridge"] = module
    spec.loader.exec_module(module)

    analyzer = module.JefAnalyzer.__new__(module.JefAnalyzer)
    analyzer._config = {}
    analyzer.service = "scene"
    return list(analyzer.summary(trace)["taxonomies"])
