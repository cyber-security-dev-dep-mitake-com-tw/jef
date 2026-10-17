"""`jef calibration`: look at an artifact; migrate a legacy one on purpose."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from jef_cli.__main__ import EXIT_INVALID, EXIT_OK, main
from jef_core.calibration import Calibrator


def _legacy_file(path: Path) -> Path:
    rng = np.random.default_rng(0)
    c = Calibrator(backbone="b", head="h")
    logits, labels = [], []
    for _ in range(400):
        y = int(rng.integers(4))
        lg = rng.normal(0, 0.5, size=4)
        lg[y] += 2.0
        logits.append(lg)
        labels.append(y)
    c.fit_bucket("choice", 4, logits, labels)
    d = {k: v for k, v in c.to_dict().items() if k not in ("calibration_schema", "artifact")}
    for b in d["buckets"].values():  # type: ignore[union-attr]
        b.pop("task_id")
        b.pop("metrics")
    path.write_text(json.dumps(d))
    return path


def test_show_marks_legacy(tmp_path: Path, capsys) -> None:
    path = _legacy_file(tmp_path / "v1.json")
    assert main(["calibration", "show", str(path)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "(legacy)" in out
    assert "choice:4" in out and "legacy" in out.splitlines()[-1]


def test_migrate_writes_a_task_scoped_schema_two_file(tmp_path: Path, capsys) -> None:
    path = _legacy_file(tmp_path / "v1.json")
    out = tmp_path / "v2.json"
    rc = main(
        [
            "calibration",
            "migrate",
            str(path),
            str(out),
            "--task-id",
            "cve.severity.v1",
            "--model-sha256",
            "sha256:abc",
            "--language",
            "en",
        ]
    )
    assert rc == EXIT_OK
    migrated = Calibrator.load(out)
    assert not migrated.legacy
    assert set(migrated.buckets) == {"cve.severity.v1|choice:4"}
    assert migrated.artifact.model_sha256 == "sha256:abc"
    assert migrated.artifact.language == "en"
    assert migrated.status("choice", 4, task_id="cve.severity.v1") == "validated"
    # Already migrated: refuse rather than relabel again.
    assert (
        main(["calibration", "migrate", str(out), str(tmp_path / "v3.json"), "--task-id", "x.v1"])
        == EXIT_INVALID
    )
    assert "already schema" in capsys.readouterr().err
    # Show on the result lists the task.
    assert main(["calibration", "show", str(out)]) == EXIT_OK
    assert "cve.severity.v1" in capsys.readouterr().out
