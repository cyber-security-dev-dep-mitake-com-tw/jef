"""The model manifest: which bytes a release says it ships, and whether these are they.

A model directory (``models/jef-v0``) holds a head, a calibration artifact and a
report. ``manifest.json`` next to them records each file's SHA-256 and size,
the backbone the head was trained against, the calibration's schema and the
head digest the calibration declares, and the commit it was built from.

Two consumers:

* the release workflow refuses to publish weights whose files do not match
  the manifest (REL-003), and attests the manifest with the weights;
* the server and embedded SDK verify it at load and, on a mismatch, drop the
  calibration rather than serve numbers fitted for other bytes. That fails
  closed: every question is then ``missing`` and no gate automates.

Stdlib only, so the release step can run it before any dependency is
installed: ``scripts/model-manifest.py`` loads this file directly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA",
    "ManifestStatus",
    "build_manifest",
    "check_model_dir",
    "file_digest",
    "verify_manifest",
]

MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA = 1

#: Files a model directory may carry, in the order they are listed. Anything
#: else in the directory is not part of the model and is ignored.
ARTIFACT_NAMES = ("head.npz", "calibration.json", "report.json")

ManifestStatus = Literal["verified", "mismatch", "absent"]


def file_digest(path: Path) -> str:
    """``sha256:<hex>``, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _calibration_summary(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"readable": False}
    schema = int(raw.get("calibration_schema", 1))
    buckets = raw.get("buckets") or {}
    tasks = sorted({k.rpartition("|")[0] for k in buckets if "|" in k})
    return {
        "schema": schema,
        "legacy": schema < 2,
        "buckets": len(buckets),
        "tasks": tasks,
        "declared_head_sha256": (raw.get("artifact") or {}).get("model_sha256"),
        "backbone": raw.get("backbone", ""),
        "head": raw.get("head", ""),
    }


def build_manifest(
    model_dir: Path,
    *,
    model_id: str,
    version: str,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Describe the artifacts in ``model_dir`` as they are now."""
    artifacts: dict[str, dict[str, Any]] = {}
    for name in ARTIFACT_NAMES:
        path = model_dir / name
        if path.is_file():
            artifacts[name] = {"sha256": file_digest(path), "size": path.stat().st_size}
    if "head.npz" not in artifacts:
        raise FileNotFoundError(f"{model_dir}/head.npz is missing; nothing to describe")
    calibration = (
        _calibration_summary(model_dir / "calibration.json")
        if "calibration.json" in artifacts
        else None
    )
    return {
        "manifest_schema": MANIFEST_SCHEMA,
        "model_id": model_id,
        "version": version,
        "git_commit": git_commit,
        "backbone": (calibration or {}).get("backbone", ""),
        "artifacts": artifacts,
        "calibration": calibration,
    }


@dataclass
class Verification:
    status: ManifestStatus
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "verified"


def verify_manifest(model_dir: Path, manifest: dict[str, Any]) -> Verification:
    """Compare ``model_dir`` with ``manifest``. Every disagreement is listed."""
    problems: list[str] = []
    if manifest.get("manifest_schema") != MANIFEST_SCHEMA:
        problems.append(
            f"manifest_schema {manifest.get('manifest_schema')!r} is not {MANIFEST_SCHEMA}"
        )
    listed: dict[str, Any] = manifest.get("artifacts") or {}
    if "head.npz" not in listed:
        problems.append("manifest lists no head.npz")
    for name, expected in sorted(listed.items()):
        path = model_dir / name
        if not path.is_file():
            problems.append(f"{name} is listed but missing")
            continue
        actual = file_digest(path)
        if actual != expected.get("sha256"):
            problems.append(f"{name} is {actual}, manifest says {expected.get('sha256')}")
        elif path.stat().st_size != expected.get("size"):
            problems.append(
                f"{name} is {path.stat().st_size} bytes, manifest says {expected.get('size')}"
            )
    for name in ARTIFACT_NAMES:
        if name not in listed and (model_dir / name).is_file():
            problems.append(f"{name} is present but not in the manifest")

    # The calibration names the head it was fitted for. If it names one, it
    # must be the head this manifest ships; otherwise every P(correct) would be
    # describing a different model.
    head = listed.get("head.npz", {}).get("sha256")
    calibration = model_dir / "calibration.json"
    if head and calibration.is_file():
        declared = _calibration_summary(calibration).get("declared_head_sha256")
        if declared and declared != head:
            problems.append(
                f"calibration.json was fitted for head {declared}, manifest ships {head}"
            )

    return Verification("mismatch" if problems else "verified", problems)


def check_model_dir(model_dir: Path, manifest_path: Path | None = None) -> Verification:
    """Verify ``model_dir`` against its manifest; ``absent`` when there is none."""
    path = manifest_path or (model_dir / MANIFEST_NAME)
    if not path.is_file():
        return Verification("absent")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return Verification("mismatch", [f"{path} is not JSON: {exc}"])
    return verify_manifest(model_dir, manifest)
