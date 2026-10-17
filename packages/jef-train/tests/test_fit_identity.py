"""`fit` writes artifacts that say which backbone weights they belong to.

Training itself needs torch, which CI does not install; what is under test is
the bookkeeping around it, so the optimiser is replaced by a fixed head.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jef_core import Calibrator, Engine
from jef_core.backends import HashingBackbone
from jef_core.errors import ArtifactMismatchError
from jef_core.head import BilinearHead
from jef_train import fit
from jef_train.schema import Sample, split_samples, write_samples

REV = "c" * 40


def _corpus(root: Path, n_groups: int = 60) -> Path:
    samples: list[Sample] = []
    for g in range(n_groups):
        samples.append(
            Sample(
                state=f"漏洞描述編號 {g}：說明文字。",
                kind="noul",
                instructions="是否緊急？",
                option_keys=["false", "true"],
                option_labels=[None, None],
                label=g % 2,
                source="a",
                group=f"evidence:{g}",
            )
        )
    write_samples(samples, root / "all.jsonl")
    for name, part in zip(("train", "calibration", "test"), split_samples(samples), strict=True):
        write_samples(part, root / f"{name}.jsonl")
    return root


@pytest.fixture(autouse=True)
def _no_torch(monkeypatch) -> None:
    monkeypatch.setattr(
        fit,
        "train_head",
        lambda cache, *_a, **_k: BilinearHead.init_identity(cache.ctx.shape[1], 4),
    )


def _run(tmp_path: Path, backbone: str) -> Path:
    corpus = _corpus(tmp_path / "corpus")
    out = tmp_path / "model"
    argv = ["--corpus", str(corpus), "--out", str(out), "--cache-dir", str(tmp_path / "cache")]
    assert fit.main([*argv, "--backbone", backbone, "--epochs", "1", "--rank", "4"]) == 0
    return out


def test_fit_records_the_backbone_in_every_artifact(tmp_path) -> None:
    out = _run(tmp_path, "hashing")
    head = BilinearHead.load(out / "head.npz")
    calibrator = Calibrator.load(out / "calibration.json")
    report = json.loads((out / "report.json").read_text())
    assert head.backbone == calibrator.backbone == report["backbone"] == "hashing"
    assert head.backbone_revision is None and report["backbone_revision"] is None
    # And the result loads back onto the backbone it came from.
    Engine("hashing", head, calibrator)


def test_fit_keys_features_by_revision_and_records_it(tmp_path, monkeypatch) -> None:
    def fake(spec: str, **_: object) -> HashingBackbone:
        bb = HashingBackbone()
        bb.name = spec.partition("@")[0]
        bb.revision = REV  # type: ignore[attr-defined]
        return bb

    monkeypatch.setattr(fit, "load_backbone", fake)
    out = _run(tmp_path, f"org/model@{REV}")

    assert [p.name for p in (tmp_path / "cache").iterdir()] == [f"all.org__model@{REV}.npz"]
    head = BilinearHead.load(out / "head.npz")
    assert (head.backbone, head.backbone_revision) == ("org/model", REV)
    assert Calibrator.load(out / "calibration.json").backbone_revision == REV
    assert json.loads((out / "report.json").read_text())["backbone_revision"] == REV

    other = fake("org/model")
    other.revision = "d" * 40  # type: ignore[attr-defined]
    with pytest.raises(ArtifactMismatchError):
        Engine(other, head)
    assert np.load(out / "head.npz")["backbone_revision"].item() == REV
