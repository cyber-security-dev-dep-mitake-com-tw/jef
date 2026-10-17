"""The model manifest: REL-003, and failing closed when the bytes disagree."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from jef_core.model_manifest import build_manifest, check_model_dir, verify_manifest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
SHIPPED = ROOT / "models" / "jef-v0"


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "jef-v0"
    d.mkdir()
    for name in ("head.npz", "calibration.json", "report.json"):
        shutil.copy(SHIPPED / name, d / name)
    manifest = build_manifest(d, model_id="jef-v0", version="0.0.0")
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


def test_the_shipped_weights_match_their_manifest() -> None:
    """REL-003 against the real files: a weights change without a manifest change fails here."""
    result = check_model_dir(SHIPPED)
    assert result.status == "verified", result.problems


def test_a_fresh_manifest_verifies(model_dir: Path) -> None:
    assert check_model_dir(model_dir).ok


def test_rel_003_one_changed_byte_is_a_mismatch(model_dir: Path) -> None:
    head = model_dir / "head.npz"
    data = bytearray(head.read_bytes())
    data[-1] ^= 0x01
    head.write_bytes(bytes(data))
    result = check_model_dir(model_dir)
    assert result.status == "mismatch"
    assert any(p.startswith("head.npz is sha256:") for p in result.problems)


def test_missing_and_unlisted_files_are_mismatches(model_dir: Path) -> None:
    (model_dir / "report.json").unlink()
    assert any(
        "report.json is listed but missing" in p for p in check_model_dir(model_dir).problems
    )

    manifest = json.loads((model_dir / "manifest.json").read_text())
    shutil.copy(SHIPPED / "report.json", model_dir / "report.json")
    del manifest["artifacts"]["report.json"]
    assert any(
        "present but not in the manifest" in p
        for p in verify_manifest(model_dir, manifest).problems
    )


def test_calibration_fitted_for_another_head_is_a_mismatch(model_dir: Path) -> None:
    cal = json.loads((model_dir / "calibration.json").read_text())
    cal["calibration_schema"] = 2
    cal["artifact"] = {"model_sha256": "sha256:" + "0" * 64}
    (model_dir / "calibration.json").write_text(json.dumps(cal))
    manifest = build_manifest(model_dir, model_id="jef-v0", version="0.0.0")
    result = verify_manifest(model_dir, manifest)
    assert result.status == "mismatch"
    assert any("was fitted for head" in p for p in result.problems)


def test_no_manifest_is_absent_not_ok(tmp_path: Path) -> None:
    assert check_model_dir(tmp_path).status == "absent"
    (tmp_path / "manifest.json").write_text("{not json")
    assert check_model_dir(tmp_path).status == "mismatch"


def test_build_needs_a_head(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_manifest(tmp_path, model_id="x", version="0")


def test_the_cli_exit_codes(model_dir: Path) -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "model_manifest_cli", ROOT / "scripts" / "model-manifest.py"
    )
    assert spec and spec.loader
    cli = importlib.util.module_from_spec(spec)
    sys.modules["model_manifest_cli"] = cli
    spec.loader.exec_module(cli)
    assert cli.main(["check", str(model_dir)]) == 0
    (model_dir / "report.json").write_text("{}")
    assert cli.main(["check", str(model_dir)]) == 1
    empty = model_dir.parent / "empty"
    empty.mkdir()
    assert cli.main(["check", str(empty)]) == 0
    assert cli.main(["check", str(empty), "--require"]) == 1
