"""The version tool rewrites pyproject files, so it can corrupt them.

It already did once: a naive search-and-replace turned `name = "jef"` into
`name = "jef==0.1.0"`, which builds with an obscure "not a valid package name"
error far from the cause. These tests pin the boundaries of what it may touch.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "set_version", ROOT / "scripts" / "set-version.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["set_version"] = module
    spec.loader.exec_module(module)
    return module


set_version = _load()


PYPROJECT = """[project]
name = "jef-server"
version = "0.1.0"
description = "test"
dependencies = ["jef-core", "jef-scene==0.0.9", "fastapi>=0.115"]

[project.optional-dependencies]
torch = ["jef-core[torch]"]
extra = ["jef-train==0.0.1", "requests>=2"]

[tool.hatch.build.targets.wheel]
packages = ["src/jef_server"]
"""


def test_version_and_dependencies_are_rewritten() -> None:
    out = set_version._rewrite_dependencies(PYPROJECT, "0.2.0")
    assert '"jef-core==0.2.0"' in out
    assert '"jef-scene==0.2.0"' in out
    assert '"jef-train==0.2.0"' in out


def test_extras_survive_the_rewrite() -> None:
    """`jef-core[torch]` must not become `jef-core`, or torch stops installing."""
    out = set_version._rewrite_dependencies(PYPROJECT, "0.2.0")
    assert '"jef-core[torch]==0.2.0"' in out


def test_the_name_field_is_never_touched() -> None:
    """The bug this file exists for."""
    out = set_version._rewrite_dependencies(PYPROJECT, "0.2.0")
    assert 'name = "jef-server"' in out
    assert '"jef-server==' not in out


def test_third_party_dependencies_are_left_alone() -> None:
    out = set_version._rewrite_dependencies(PYPROJECT, "0.2.0")
    assert '"fastapi>=0.115"' in out
    assert '"requests>=2"' in out


def test_build_backend_config_is_left_alone() -> None:
    out = set_version._rewrite_dependencies(PYPROJECT, "0.2.0")
    assert '"src/jef_server"' in out


def test_rewriting_is_idempotent() -> None:
    once = set_version._rewrite_dependencies(PYPROJECT, "0.2.0")
    assert set_version._rewrite_dependencies(once, "0.2.0") == once


def test_rewriting_an_already_pinned_version_updates_it() -> None:
    once = set_version._rewrite_dependencies(PYPROJECT, "0.2.0")
    twice = set_version._rewrite_dependencies(once, "0.3.0")
    assert '"jef-core==0.3.0"' in twice
    assert "0.2.0" not in twice


@pytest.mark.parametrize("version", ["0.2.0", "1.0.0", "0.2.0rc1", "0.2.0a1", "0.2.0b2"])
def test_accepted_versions(version: str) -> None:
    assert set_version.VERSION_RE.match(version)


@pytest.mark.parametrize("version", ["0.2", "v0.2.0", "0.2.0-rc1", "latest", "0.2.0.post1"])
def test_rejected_versions(version: str) -> None:
    assert not set_version.VERSION_RE.match(version)


def test_check_reports_a_mismatch_with_a_nonzero_exit() -> None:
    assert set_version.main(["--check", "9.9.9"]) == 1


def test_check_passes_on_the_current_version() -> None:
    current = set_version.read_versions()
    assert len(set(current.values())) == 1, f"repo is not in lockstep: {current}"
    assert set_version.main(["--check", next(iter(current.values()))]) == 0


def test_the_repo_is_in_lockstep() -> None:
    """Every package, python and npm, carries the same version."""
    versions = set_version.read_versions()
    assert len(versions) == 8, f"expected 8 packages, found {sorted(versions)}"
    assert len(set(versions.values())) == 1, versions


def test_npm_packages_keep_their_formatting(tmp_path: Path) -> None:
    """npm rewrites package.json on install; matching its format avoids churn."""
    for rel in set_version.NPM_PACKAGES:
        raw = (ROOT / rel).read_text(encoding="utf-8")
        assert raw.endswith("\n")
        assert json.loads(raw)  # parses
        assert '\n  "' in raw, f"{rel} should use two-space indent"
