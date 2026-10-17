"""NEXT-V3-01: nightly results are an artifact with a verifiable manifest, never a push to main."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "benchmark_manifest", ROOT / "scripts" / "benchmark-manifest.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_manifest"] = module
    spec.loader.exec_module(module)
    return module


bm = _load()


def _workflow(name: str) -> dict:
    return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text())


def test_the_nightly_job_cannot_write_to_the_repository() -> None:
    wf = _workflow("benchmark.yml")
    job = wf["jobs"]["third-party"]
    assert job["permissions"] == {"contents": "read"}
    runs = "\n".join(str(step.get("run", "")) for step in job["steps"])
    assert "git push" not in runs and "git commit" not in runs
    uploads = [
        s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/upload-artifact")
    ]
    assert uploads and "benchmark-manifest.json" in uploads[0]["with"]["path"]
    assert "${{ github.sha }}" in uploads[0]["with"]["name"]


def test_the_publisher_opens_a_pull_request_and_never_pushes_main() -> None:
    wf = _workflow("benchmark-publish.yml")
    runs = "\n".join(str(s.get("run", "")) for s in wf["jobs"]["publish"]["steps"])
    assert "benchmark-manifest.py verify" in runs
    assert "gh pr create --base main" in runs
    assert "git push origin main" not in runs and "--force" not in runs
    assert 'git push origin "$branch"' in runs


def _results(tmp: Path) -> None:
    (tmp / "docs" / "badges").mkdir(parents=True)
    (tmp / "docs" / "benchmarks.json").write_text(
        json.dumps(
            {"backbone": "b", "tasks": {"cti-mcq": {"n": 300}, "cti-vsp/attack_vector": {"n": 250}}}
        )
    )
    (tmp / "docs" / "badges" / "x.json").write_text('{"message": "0.8"}')


def test_write_then_verify_round_trips_and_catches_tampering(tmp_path: Path) -> None:
    _results(tmp_path)
    out = tmp_path / "docs" / "benchmark-manifest.json"
    rc = bm.main(
        [
            "write",
            "docs/benchmarks.json",
            "docs/badges",
            "--out",
            str(out),
            "--source-sha",
            "abc123",
            "--run-id",
            "42",
            "--root",
            str(tmp_path),
            "--model-dir",
            str(ROOT / "models" / "jef-v0"),
        ]
    )
    assert rc == 0
    m = json.loads(out.read_text())
    assert m["source_sha"] == "abc123" and m["workflow_run_id"] == "42"
    assert set(m["files"]) == {"docs/benchmarks.json", "docs/badges/x.json"}
    assert m["datasets"] == ["cti-mcq", "cti-vsp"] and m["model"]["calibration_schema"] == 1
    assert m["model"]["head_sha256"].startswith("sha256:")

    assert bm.main(["verify", str(out), "--root", str(tmp_path), "--source-sha", "abc123"]) == 0
    assert bm.main(["verify", str(out), "--root", str(tmp_path), "--source-sha", "other"]) == 1
    (tmp_path / "docs" / "badges" / "x.json").write_text('{"message": "0.99"}')
    assert bm.main(["verify", str(out), "--root", str(tmp_path)]) == 1


def test_write_refuses_a_run_without_results(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    assert (
        bm.main(
            [
                "write",
                "docs",
                "--out",
                str(tmp_path / "m.json"),
                "--source-sha",
                "a",
                "--run-id",
                "1",
                "--root",
                str(tmp_path),
            ]
        )
        == 1
    )


def test_verify_works_on_an_artifact_restaged_under_docs(tmp_path: Path) -> None:
    """upload-artifact drops the common docs/ prefix; the publisher puts it back first."""
    import shutil

    _results(tmp_path)
    m = tmp_path / "docs" / "benchmark-manifest.json"
    bm.main(
        [
            "write",
            "docs/benchmarks.json",
            "docs/badges",
            "--out",
            str(m),
            "--source-sha",
            "s",
            "--run-id",
            "1",
            "--root",
            str(tmp_path),
            "--model-dir",
            str(ROOT / "models" / "jef-v0"),
        ]
    )
    artifact = tmp_path / "artifact"  # what the download looks like: no docs/ prefix
    shutil.copytree(tmp_path / "docs", artifact)
    stage = tmp_path / "stage"
    shutil.copytree(artifact, stage / "docs")
    assert (
        bm.main(["verify", str(stage / "docs" / "benchmark-manifest.json"), "--root", str(stage)])
        == 0
    )
    assert (
        bm.main(["verify", str(artifact / "benchmark-manifest.json"), "--root", str(artifact)]) == 1
    )
