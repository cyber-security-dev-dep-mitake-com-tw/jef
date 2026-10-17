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


# --------------------------------------------------------------------------- #
# Tag spelling is not version identity
# --------------------------------------------------------------------------- #
#
# Release #10 was tagged `v0.2.0rc.4` -- the semver spelling, the one the npm
# packages carry -- against pyproject files at `0.2.0rc4`. The strict check
# failed the release at the first step over a dot. The workflow now
# canonicalises the tag first; these pin what that accepts and refuses.


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.2.0", "0.2.0"),
        ("v0.2.0", "0.2.0"),
        ("0.2.0rc4", "0.2.0rc4"),
        ("v0.2.0rc4", "0.2.0rc4"),
        ("v0.2.0rc.4", "0.2.0rc4"),  # Release #10
        ("v0.2.0-rc.4", "0.2.0rc4"),  # npm's own spelling
        ("0.2.0-rc4", "0.2.0rc4"),
        ("0.2.0_rc4", "0.2.0rc4"),
        ("0.2.0.rc4", "0.2.0rc4"),
        ("0.2.0RC4", "0.2.0rc4"),
        ("V0.2.0-RC.4", "0.2.0rc4"),
        ("0.2.0-alpha.1", "0.2.0a1"),
        ("0.2.0-beta.2", "0.2.0b2"),
        ("0.2.0c1", "0.2.0rc1"),  # PEP 440 synonyms for rc
        ("0.2.0-pre.1", "0.2.0rc1"),
        ("0.2.0preview1", "0.2.0rc1"),
        ("1.0.0-rc.10", "1.0.0rc10"),
        ("0.2.0rc04", "0.2.0rc4"),  # PEP 440 drops leading zeros
        (" v0.2.0rc4\n", "0.2.0rc4"),
    ],
)
def test_canonical_accepts_every_reasonable_spelling(raw: str, expected: str) -> None:
    assert set_version.canonical(raw) == expected
    assert set_version.VERSION_RE.match(set_version.canonical(raw))


@pytest.mark.parametrize(
    "raw",
    [
        "0.2",
        "0.2.0.1",
        "latest",
        "0.2.0.post1",
        "0.2.0.dev1",
        "1!0.2.0",
        "0.2.0rc",  # a label with no number
        "0.2.0-rc.4.1",
        "0.2.0rc4-final",
        "",
        "v",
    ],
)
def test_canonical_refuses_what_is_not_a_release_version(raw: str) -> None:
    with pytest.raises(ValueError, match="not a release version"):
        set_version.canonical(raw)


def test_canonical_flag_prints_the_pep440_spelling(capsys: pytest.CaptureFixture[str]) -> None:
    assert set_version.main(["--canonical", "v0.2.0rc.4"]) == 0
    assert capsys.readouterr().out.strip() == "0.2.0rc4"


def test_canonical_flag_fails_loudly_on_garbage(capsys: pytest.CaptureFixture[str]) -> None:
    assert set_version.main(["--canonical", "latest"]) == 2
    assert "not a release version" in capsys.readouterr().err


def test_check_accepts_the_tag_spelling_of_the_current_version() -> None:
    """The exact failure of Release #10, against the real files."""
    current = set_version.read_versions()
    python = next(v for label, v in current.items() if label.startswith("py:"))
    match = set_version.VERSION_RE.match(python)
    assert match, python
    base, kind, number = match.groups()
    if kind is None:
        pytest.skip("the repo is at a final release; no prerelease spelling to vary")
    for spelling in (
        f"v{base}{kind}.{number}",
        f"v{base}-{kind}.{number}",
        f"{base}-{kind}{number}",
    ):
        assert set_version.main(["--check", spelling]) == 0, spelling


def test_check_passes_on_the_current_version() -> None:
    """The python version is the one to check against, not any of them.

    This compared `set(values)` for a single element, which quietly assumed
    npm and PyPI spell a version identically. On a prerelease they do not --
    `0.2.0rc1` against `0.2.0-rc.1` -- and the release workflow runs exactly
    this check before it builds anything, so an rc tag would have failed
    `verify` rather than publishing.
    """
    current = set_version.read_versions()
    assert set_version._in_lockstep(current), f"repo is not in lockstep: {current}"
    python = {v for label, v in current.items() if label.startswith("py:")}
    assert set_version.main(["--check", next(iter(python))]) == 0


def test_the_repo_is_in_lockstep() -> None:
    """Every package, python and npm, carries the same version."""
    versions = set_version.read_versions()
    assert len(versions) == 6, f"expected 5 packages + 1 source constant, found {sorted(versions)}"
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
    # And the source constants: without this the test rewrote the real
    # packages/jef-sdk-ts/src/version.ts to 0.2.0-rc.1 the day that file was
    # added -- caught by test_sdk_version_constant_matches_package_json.
    monkeypatch.setattr(set_version, "TS_VERSION_FILES", {})

    assert set_version.apply("0.2.0rc1", check=False) == []
    assert json.loads(pkg.read_text())["version"] == "0.2.0-rc.1"


def test_a_release_writes_the_same_string_to_both() -> None:
    assert set_version.npm_version("1.2.3") == "1.2.3"


# --------------------------------------------------------------------------- #
# REL-001: a version constant in source must not drift from its package
# --------------------------------------------------------------------------- #


def test_sdk_version_constant_matches_package_json() -> None:
    """`@jef-ai/sdk` exported VERSION = '0.1.0' while package.json said 0.2.0-rc.4."""
    pkg = json.loads((ROOT / "packages/jef-sdk-ts/package.json").read_text(encoding="utf-8"))
    src = (ROOT / "packages/jef-sdk-ts/src/version.ts").read_text(encoding="utf-8")
    match = set_version._TS_VERSION_RE.search(src)
    assert match, "version.ts must export VERSION"
    assert match.group(1) == pkg["version"]


def test_check_fails_when_the_ts_constant_drifts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg = tmp_path / "package.json"
    pkg.write_text(json.dumps({"name": "x", "version": "0.2.0-rc.1"}, indent=2) + "\n")
    ts = tmp_path / "version.ts"
    ts.write_text("export const VERSION = '0.1.0';\n")
    monkeypatch.setattr(set_version, "PY_PACKAGES", [])
    monkeypatch.setattr(set_version, "NPM_PACKAGES", [pkg])
    monkeypatch.setattr(set_version, "TS_VERSION_FILES", {ts: pkg})
    monkeypatch.setattr(set_version, "ROOT", Path("/"))

    problems = set_version.apply("0.2.0rc1", check=True)
    assert any("exports VERSION '0.1.0'" in p for p in problems), problems

    # Writing fixes it, and the file is regenerated whole with its header.
    assert set_version.apply("0.2.0rc1", check=False) == []
    assert "export const VERSION = '0.2.0-rc.1';" in ts.read_text()
    assert ts.read_text().startswith("// Written by scripts/set-version.py")
    assert set_version.apply("0.2.0rc1", check=True) == []


def test_python_dunder_versions_read_the_installed_distribution() -> None:
    """No more hand-maintained `__version__ = "0.1.0"` four minor versions behind."""
    import importlib.metadata

    import jef_cli
    import jef_core
    import jef_scene
    import jef_sdk
    import jef_server

    jef_dist = importlib.metadata.version("jef")
    assert jef_core.__version__ == jef_dist
    assert jef_scene.__version__ == jef_dist
    assert jef_cli.__version__ == jef_dist
    assert jef_sdk.__version__ == jef_dist
    assert jef_server.__version__ == importlib.metadata.version("jef-server")
    python = {v for label, v in set_version.read_versions().items() if label.startswith("py:")}
    assert python == {jef_dist}


@pytest.mark.parametrize(
    ("lower", "higher"),
    [
        ("0.2.0a1", "0.2.0b1"),
        ("0.2.0b2", "0.2.0rc1"),
        ("0.2.0rc9", "0.2.0rc10"),  # numeric, not string, comparison
        ("0.2.0rc10", "0.2.0"),  # a final release sorts after its rcs
        ("0.2.0", "0.2.1rc1"),
        ("0.9.9", "0.10.0"),
    ],
)
def test_version_order(lower: str, higher: str) -> None:
    assert set_version.version_key(lower) < set_version.version_key(higher)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.2.0rc4", "0.2.0rc5"),
        ("0.2.0rc9", "0.2.0rc10"),
        ("0.2.0a1", "0.2.0a2"),
        ("0.2.0", "0.2.1"),
        ("1.9.9", "1.9.10"),
    ],
)
def test_bump_stays_on_its_track(version: str, expected: str) -> None:
    assert set_version.bump(version) == expected


def test_next_version_bumps_the_newest_published_spelling() -> None:
    """npm's `0.2.0-rc.4` and a tag's `v0.2.0rc3` are versions; `0.0.0-stage` is not."""
    published = ["0.0.0-stage", "0.2.0-rc.4", "v0.2.0rc3", "0.2.0rc1"]
    assert set_version.next_version("0.2.0rc4", published) == "0.2.0rc5"


def test_next_version_never_goes_below_what_is_published() -> None:
    """Files behind the registry (a release that never committed its bump)."""
    assert set_version.next_version("0.2.0rc2", ["0.2.0rc4"]) == "0.2.0rc5"


def test_next_version_keeps_files_that_are_ahead() -> None:
    """`set-version.py 0.3.0rc1` on purpose is the next release, not 0.2.0rc5."""
    assert set_version.next_version("0.3.0rc1", ["0.2.0rc4"]) == "0.3.0rc1"
    assert set_version.next_version("0.2.0", ["0.2.0rc4"]) == "0.2.0"


def test_next_version_with_nothing_published_is_the_files() -> None:
    assert set_version.next_version("0.1.0", []) == "0.1.0"
    assert set_version.next_version("0.1.0", ["0.0.0-stage"]) == "0.1.0"
