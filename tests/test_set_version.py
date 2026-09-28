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
dependencies = ["jef", "fastapi>=0.115"]

[project.optional-dependencies]
torch = ["jef[torch]"]
extra = ["jef-train==0.0.1", "requests>=2"]

[tool.hatch.build.targets.wheel]
packages = ["src/jef_server"]
"""


def test_version_and_dependencies_are_rewritten() -> None:
    out = set_version._rewrite_dependencies(PYPROJECT, "0.2.0")
    assert '"jef==0.2.0"' in out
    assert '"jef-train==0.2.0"' in out


def test_extras_survive_the_rewrite() -> None:
    """`jef[torch]` must not become `jef`, or torch stops installing."""
    out = set_version._rewrite_dependencies(PYPROJECT, "0.2.0")
    assert '"jef[torch]==0.2.0"' in out


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
    assert '"jef==0.3.0"' in twice
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
    assert len(versions) == 5, f"expected 5 packages, found {sorted(versions)}"
    assert set_version._in_lockstep(versions), versions


def test_npm_packages_keep_their_formatting(tmp_path: Path) -> None:
    """npm rewrites package.json on install; matching its format avoids churn."""
    for rel in set_version.NPM_PACKAGES:
        raw = (ROOT / rel).read_text(encoding="utf-8")
        assert raw.endswith("\n")
        assert json.loads(raw)  # parses
        assert '\n  "' in raw, f"{rel} should use two-space indent"


# --------------------------------------------------------------------------- #
# Against the real files, not a synthetic string
# --------------------------------------------------------------------------- #
#
# The tests above exercise the rewrite function in isolation, and they passed
# while five real pyproject.toml files sat in the repo with `name =
# "jef-core==0.1.0"`. Testing a function is not testing the artifact it
# produces. These check the files themselves.

EXPECTED_NAMES = {
    "jef": "jef",
    "jef-server": "jef-server",
    "jef-train": "jef-train",
}


@pytest.mark.parametrize(("directory", "distribution"), sorted(EXPECTED_NAMES.items()))
def test_real_pyproject_names_are_intact(directory: str, distribution: str) -> None:
    import re

    path = ROOT / "packages" / directory / "pyproject.toml"
    match = re.search(r'(?m)^name\s*=\s*"([^"]+)"', path.read_text(encoding="utf-8"))
    assert match, f"{path} has no name field"
    assert match.group(1) == distribution, (
        f"{path} declares name {match.group(1)!r}; a version suffix here fails the "
        "build with a message that names neither the file nor the cause"
    )


@pytest.mark.parametrize("directory", sorted(EXPECTED_NAMES))
def test_every_package_builds(directory: str, tmp_path: Path) -> None:
    """The only check that would have caught the corrupted name fields."""
    import shutil
    import subprocess

    # Resolved rather than looked up from PATH at exec time: a partial path is
    # the kind of thing the bandit rules exist to catch, and this repo has them
    # on for a reason.
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")

    result = subprocess.run(  # noqa: S603
        [uv, "build", "--wheel", str(ROOT / "packages" / directory), "-o", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert list(tmp_path.glob("*.whl")), "no wheel produced"


@pytest.mark.parametrize("directory", sorted(EXPECTED_NAMES))
def test_every_package_declares_pypi_metadata(directory: str) -> None:
    """Missing metadata publishes fine and looks abandoned."""
    text = (ROOT / "packages" / directory / "pyproject.toml").read_text(encoding="utf-8")
    for field in ("readme", "classifiers", "keywords", "authors", "[project.urls]"):
        assert field in text, f"{directory} is missing {field}"
    assert (ROOT / "packages" / directory / "README.md").is_file(), f"{directory} has no README"


def test_the_jef_wheel_carries_all_four_modules(tmp_path: Path) -> None:
    """`jef` is one distribution holding four modules, not a metapackage.

    Nothing else would notice if a module fell out of the wheel: the workspace
    install puts all four on the path from source regardless, so the tests --
    and every developer -- would keep passing while `pip install jef` shipped
    an engine with no scenes.
    """
    import shutil
    import subprocess
    import zipfile

    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")

    result = subprocess.run(  # noqa: S603
        [uv, "build", "--wheel", str(ROOT / "packages" / "jef"), "-o", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-1500:]

    (wheel,) = tmp_path.glob("jef-*.whl")
    top_level = {name.split("/")[0] for name in zipfile.ZipFile(wheel).namelist()}
    assert {"jef_core", "jef_scene", "jef_sdk", "jef_cli"} <= top_level, sorted(top_level)


# --------------------------------------------------------------------------- #
# PEP 440 is not semver
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("pep440", "semver"),
    [
        ("0.2.0", "0.2.0"),
        ("0.2.0rc1", "0.2.0-rc.1"),
        ("0.2.0a1", "0.2.0-alpha.1"),
        ("0.2.0b2", "0.2.0-beta.2"),
        ("1.0.0rc10", "1.0.0-rc.10"),
    ],
)
def test_npm_versions_are_translated(pep440: str, semver: str) -> None:
    assert set_version.npm_version(pep440) == semver


def test_the_npm_prerelease_is_dotted() -> None:
    """`rc.9 < rc.10`, but `rc10 < rc9`.

    Semver compares dot-separated numeric identifiers numerically and
    alphanumeric ones as strings, so dropping the dot silently reverses the
    order of the tenth release candidate and the ninth.
    """
    assert set_version.npm_version("1.0.0rc10").endswith("-rc.10")


def test_a_prerelease_writes_different_strings_to_each_ecosystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """npm would otherwise publish a version matching neither tag nor pyproject.

    `0.2.0rc1` fails npm's strict semver, but its loose parser reads it as
    `0.2.0-rc1` and the publish path normalises -- so the registry ends up
    holding a third spelling, and `--check` cannot see it because it reads the
    files we wrote rather than the registry.

    PY_PACKAGES is emptied as well as redirecting NPM_PACKAGES. `apply` writes
    to both lists, and an earlier draft of this test left the real pyproject
    files sitting at 0.2.0rc1 -- the same shape of accident this file exists to
    guard against.
    """
    pkg = tmp_path / "package.json"
    pkg.write_text(json.dumps({"name": "x", "version": "0.0.0"}, indent=2) + "\n")

    monkeypatch.setattr(set_version, "PY_PACKAGES", [])
    monkeypatch.setattr(set_version, "NPM_PACKAGES", [pkg])

    assert set_version.apply("0.2.0rc1", check=False) == []
    assert json.loads(pkg.read_text())["version"] == "0.2.0-rc.1"


def test_a_release_writes_the_same_string_to_both() -> None:
    assert set_version.npm_version("1.2.3") == "1.2.3"
