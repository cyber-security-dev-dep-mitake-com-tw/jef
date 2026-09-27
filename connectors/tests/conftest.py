"""Stub the SOAR SDKs so connector logic can be tested without them.

Cortex and Shuffle both ship SDKs that assume a platform around them -- config
injected by the runner, stdout treated as the report channel, process exit on
error. Installing them to test request shaping would test the platforms, not the
connector. These stubs provide the surface each connector actually uses, so the
parts that carry real risk -- option order, state typing, error mapping -- are
exercised directly.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
CORTEX_DIR = ROOT / "connectors" / "cortex" / "JEF"
SHUFFLE_DIR = ROOT / "connectors" / "shuffle" / "jef" / "1.0.0" / "src"


class _AnalyzerExit(Exception):
    """Raised in place of Cortex's process exit, so tests can assert on it."""


def _install_cortexutils() -> None:
    cortexutils = types.ModuleType("cortexutils")
    analyzer_mod = types.ModuleType("cortexutils.analyzer")
    responder_mod = types.ModuleType("cortexutils.responder")

    class _Base:
        def __init__(self) -> None:
            self._config: dict[str, Any] = {}
            self._data: Any = None
            self.reported: Any = None

        # -- surface the connectors use ----------------------------------- #

        def get_param(self, name: str, default: Any = None, message: str | None = None) -> Any:
            value = self._config.get(name, default)
            if value is None and message:
                self.error(message)
            return value

        def get_data(self) -> Any:
            return self._data

        def error(self, message: str) -> None:
            raise _AnalyzerExit(message)

        def report(self, payload: Any) -> None:
            self.reported = payload

        def run(self) -> None:
            return None

        @staticmethod
        def build_taxonomy(
            level: str, namespace: str, predicate: str, value: Any
        ) -> dict[str, Any]:
            return {
                "level": level,
                "namespace": namespace,
                "predicate": predicate,
                "value": value,
            }

        @staticmethod
        def build_operation(op_type: str, **kwargs: Any) -> dict[str, Any]:
            return {"type": op_type, **kwargs}

    class Analyzer(_Base):
        pass

    class Responder(_Base):
        pass

    analyzer_mod.Analyzer = Analyzer  # type: ignore[attr-defined]
    responder_mod.Responder = Responder  # type: ignore[attr-defined]
    cortexutils.analyzer = analyzer_mod  # type: ignore[attr-defined]
    cortexutils.responder = responder_mod  # type: ignore[attr-defined]
    sys.modules.setdefault("cortexutils", cortexutils)
    sys.modules["cortexutils.analyzer"] = analyzer_mod
    sys.modules["cortexutils.responder"] = responder_mod


def _install_shuffle_sdk() -> None:
    module = types.ModuleType("shuffle_sdk")

    class AppBase:
        def __init__(
            self, redis: Any = None, logger: Any = None, console_logger: Any = None
        ) -> None:
            self.redis = redis
            self.logger = logger
            self.console_logger = console_logger

        @classmethod
        def run(cls) -> None:  # pragma: no cover -- entry point only
            return None

    module.AppBase = AppBase  # type: ignore[attr-defined]
    sys.modules.setdefault("shuffle_sdk", module)


def _load(path: Path, name: str) -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_install_cortexutils()
_install_shuffle_sdk()


@pytest.fixture(scope="session")
def analyzer_exit() -> type[Exception]:
    return _AnalyzerExit


@pytest.fixture(scope="session")
def cortex_analyzer_module() -> Any:
    return _load(CORTEX_DIR / "jef_analyzer.py", "jef_analyzer_under_test")


@pytest.fixture(scope="session")
def cortex_responder_module() -> Any:
    return _load(CORTEX_DIR / "jef_responder.py", "jef_responder_under_test")


@pytest.fixture(scope="session")
def shuffle_module() -> Any:
    return _load(SHUFFLE_DIR / "app.py", "jef_shuffle_under_test")
