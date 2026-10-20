"""The schema-2 candidate pipeline for incident-triage: it works, and it refuses.

Until task-matched human labels exist the correct outcome is ``insufficient``.
The dry run proves the pipeline end to end on synthetic labels and that the
check says so; the refusals prove nothing weaker can be promoted.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from jef_scene import load_scene
from jef_train import fit
from jef_train.schema import read_samples
from jef_train.socbench.cli import main
from jef_train.socbench.triage import find_root, load_pack
from jef_train.socbench.triage_candidate import (
    CandidateError,
    check_candidate,
    export_triage_corpus,
    required_buckets,
)

ROOT = find_root(Path(__file__).resolve())
REPO = ROOT.parents[1]
DEV = ROOT / "synthetic" / "dev-v1"
SCENE = load_scene(REPO / "scenes" / "incident-triage.zh-tw.yaml")


@pytest.fixture(scope="module")
def pack():
    return load_pack(DEV)


@pytest.fixture(scope="module")
def lock():
    return json.loads((DEV / "lock.json").read_text("utf-8"))


@pytest.fixture(scope="module")
def dry_run(tmp_path_factory, pack, lock):
    # Training needs torch, which the per-PR CI job does not install; the
    # periodic triage-quality workflow runs this end to end with it.
    pytest.importorskip("torch")
    base = tmp_path_factory.mktemp("cand")
    corpus = base / "corpus"
    export_triage_corpus(pack, lock, SCENE, corpus, dry_run=True)
    model = base / "model"
    rc = fit.main(
        [
            "--corpus", str(corpus), "--out", str(model), "--backbone", "hashing",
            "--cache-dir", str(base / "cache"), "--epochs", "2",
        ]
    )  # fmt: skip
    assert rc == 0
    return corpus, model


def test_the_buckets_are_the_scenes() -> None:
    assert set(required_buckets(SCENE)) == {
        "soar_zh.false_positive.v1|noul:2",
        "soar_zh.evidence_sufficiency.v1|noul:2",
        "soar_zh.routing.v1|choice:5",
        "soar_zh.severity.v1|score:5",
        "soar_zh.urgency.v1|noul:2",
    }


def test_synthetic_labels_never_export_without_dry_run(tmp_path: Path, pack, lock) -> None:
    with pytest.raises(CandidateError, match="no labels at tier B"):
        export_triage_corpus(pack, lock, SCENE, tmp_path / "c")
    with pytest.raises(CandidateError, match="cannot train a deployable"):
        export_triage_corpus(pack, lock, SCENE, tmp_path / "c", min_tier="E")
    assert not (tmp_path / "c").exists(), "refused before writing anything"


def test_an_unlocked_pack_is_refused(tmp_path: Path, pack, lock) -> None:
    bad = dict(lock, cases_digest="sha256:" + "0" * 64)
    with pytest.raises(CandidateError, match="locked bytes"):
        export_triage_corpus(pack, bad, SCENE, tmp_path / "c", dry_run=True)


def test_dry_run_export_is_task_scoped_family_locked_and_marked(dry_run) -> None:
    corpus, _ = dry_run
    manifest = json.loads((corpus / "manifest.json").read_text("utf-8"))
    assert manifest["dry_run"] is True and manifest["deployable"] is False
    samples = list(read_samples(corpus / "all.jsonl"))
    assert {s.bucket for s in samples} == set(required_buckets(SCENE))
    families = {
        name: {s.group for s in read_samples(corpus / f"{name}.jsonl")}
        for name in ("train", "calibration", "test")
    }
    assert not families["train"] & families["test"]
    assert not families["train"] & families["calibration"]


def test_the_dry_run_candidate_is_insufficient(dry_run) -> None:
    corpus, model = dry_run
    report = check_candidate(model, corpus, SCENE, backbone="hashing", repo_root=REPO)
    assert report["status"] == "insufficient"
    assert report["status_ceiling"] == "research"
    assert report["problems"] == [], "artifacts and splits are consistent; data is the gap"
    for b in report["buckets"].values():
        assert b["status"] == "insufficient"
        assert b["calibration_records"] < b["min_records"] == 150


def test_a_calibration_for_another_head_is_caught(tmp_path: Path, dry_run) -> None:
    """The calibration names the head it was fitted for; a mismatch is a problem."""
    corpus, model = dry_run
    other = tmp_path / "model"
    shutil.copytree(model, other)
    cal = json.loads((other / "calibration.json").read_text("utf-8"))
    cal["artifact"]["model_sha256"] = "sha256:" + "0" * 64
    (other / "calibration.json").write_text(json.dumps(cal), "utf-8")
    report = check_candidate(other, corpus, SCENE, backbone="hashing")
    assert "calibration.json was not fitted for this head.npz" in report["problems"]
    assert report["status"] == "insufficient"


def test_the_shipped_model_is_never_a_candidate(dry_run) -> None:
    corpus, _ = dry_run
    with pytest.raises(CandidateError, match="shipped"):
        check_candidate(
            REPO / "models" / "jef-v0", corpus, SCENE, backbone="hashing", repo_root=REPO
        )


def test_cli_candidate_check(tmp_path: Path, dry_run) -> None:
    corpus, model = dry_run
    out = tmp_path / "report.json"
    rc = main(
        [
            "triage", "candidate-check", "--candidate", str(model), "--corpus", str(corpus),
            "--backbone", "hashing", "--out", str(out),
        ]
    )  # fmt: skip
    assert rc == 0
    assert json.loads(out.read_text("utf-8"))["status"] == "insufficient"
