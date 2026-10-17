"""The docs drift checker: it must catch what it claims to, and pass on this repo."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("check_docs", ROOT / "scripts" / "check-docs.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_docs"] = module
    spec.loader.exec_module(module)
    return module


cd = _load()


def test_the_repository_has_no_docs_drift() -> None:
    assert cd.main() == 0


@pytest.mark.parametrize(
    ("heading", "slug"),
    [
        ("Release status", "release-status"),
        ("Preflight: why the four rc runs failed", "preflight-why-the-four-rc-runs-failed"),
        ("`confidence` is not correctness", "confidence-is-not-correctness"),
        ("7.1 Benchmark v0.1 的 6 個 Task", "71-benchmark-v01-的-6-個-task"),
        ("zh-TW: two tracks, reported apart", "zh-tw-two-tracks-reported-apart"),
    ],
)
def test_github_slugs(heading: str, slug: str) -> None:
    assert cd.github_slug(heading) == slug


def test_broken_links_and_anchors_are_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cd, "ROOT", tmp_path)
    (tmp_path / "a.md").write_text("# Title\n\n## Second part\n")
    (tmp_path / "b.md").write_text(
        "[ok](a.md) [ok](a.md#second-part) [web](https://x.example) [self](#here)\n"
        "[gone](missing.md) [bad anchor](a.md#nope)\n"
        "```\n[in code](ignored.md)\n```\n"
        "`[inline](ignored.md)`\n\n## Here\n"
    )
    problems = cd.check_links([tmp_path / "a.md", tmp_path / "b.md"])
    assert len(problems) == 2, problems
    assert any("missing.md does not resolve" in p for p in problems)
    assert any("a.md#nope names no heading" in p for p in problems)


def test_prerelease_install_lines_need_a_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cd, "ROOT", tmp_path)
    monkeypatch.setattr(cd, "_version", lambda: "0.2.0rc4")
    md = tmp_path / "x.md"
    md.write_text(
        "npm install @jef-ai/sdk\nnpm install @jef-ai/sdk@next\nnpm i -D n8n-nodes-jef@0.2.0-rc.4\nnpm install n8n-nodes-jef\n"
    )
    problems = cd.check_install_lines([md])
    assert [p.split(":")[1] for p in problems] == ["1", "4"]
    monkeypatch.setattr(cd, "_version", lambda: "0.2.0")
    assert cd.check_install_lines([md]) == []


def test_shipped_capabilities_must_cite_existing_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cd, "ROOT", tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "real.py").write_text("")
    (tmp_path / "docs" / "CAPABILITIES.md").write_text(
        "| Capability | Status | Evidence |\n|---|---|---|\n"
        "| Good | shipped | `real.py` |\n"
        "| No evidence | shipped | |\n"
        "| Gone | shipped | `src/gone.py` |\n"
        "| Later | in review (#9) | |\n"
    )
    problems = cd.check_capabilities()
    assert len(problems) == 2
    assert any("'No evidence' cites no path" in p for p in problems)
    assert any("src/gone.py, which does not exist" in p for p in problems)


def test_undocumented_server_settings_are_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cd, "ROOT", tmp_path)
    src = tmp_path / "packages/jef-server/src/jef_server"
    src.mkdir(parents=True)
    (src / "settings.py").write_text('os.environ.get("JEF_ONE")\n_int("JEF_TWO")\n')
    (tmp_path / "packages/jef-server/README.md").write_text("| `JEF_ONE` | documented |\n")
    assert cd.check_server_settings() == [
        "packages/jef-server/README.md does not document JEF_TWO (read in settings.py)"
    ]
