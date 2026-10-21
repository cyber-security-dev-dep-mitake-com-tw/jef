"""S1-S6 differ in what they train on, never in what they are judged on."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jef_scene import load_scene
from jef_train.schema import read_samples
from jef_train.socbench.cloudflare_cases import build_cases
from jef_train.socbench.cloudflare_mock import load_fixture_events
from jef_train.socbench.cloudflare_normalize import Pseudonymizer, normalize_events
from jef_train.socbench.triage import load_pack, pack_manifest
from jef_train.socbench.triage_ablation import ARMS, AblationError, build_ablation
from jef_train.socbench.triage_synth import triage_lock

ROOT = Path(__file__).resolve().parents[3]
DEV = ROOT / "benchmark" / "incident-triage" / "synthetic" / "dev-v1"
FIXTURE = ROOT / "benchmark" / "incident-triage" / "fixtures" / "cloudflare-synthetic"
SCENE = load_scene(ROOT / "scenes" / "incident-triage.zh-tw.yaml")


def held(arm_dir: Path) -> set[tuple[str, str]]:
    return {
        (str(s.meta["case_id"]), str(s.task))
        for split in ("calibration", "test")
        for s in read_samples(arm_dir / f"{split}.jsonl")
    }


@pytest.fixture(scope="module")
def dev_arms(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    out = tmp_path_factory.mktemp("abl")
    lock = json.loads((DEV / "lock.json").read_text())
    return out, build_ablation(load_pack(DEV), lock, SCENE, out, dry_run=True)


def test_every_arm_is_built_and_marked_a_dry_run(dev_arms: tuple[Path, dict[str, Any]]) -> None:
    out, report = dev_arms
    assert list(report["arms"]) == list(ARMS)
    assert report["selection"]["select_on"] == "calibration"
    for name in ARMS:
        m = json.loads((out / name / "manifest.json").read_text())
        assert m["dry_run"] is True and m["deployable"] is False
        assert m["fit_args"] == ([] if ARMS[name].class_weighting else ["--no-class-weighting"])


def test_s1_to_s4_hold_out_the_same_cases(dev_arms: tuple[Path, dict[str, Any]]) -> None:
    out, _ = dev_arms
    assert held(out / "S1") == held(out / "S2") == held(out / "S3") == held(out / "S4")


def test_hard_negatives_leave_train_only_and_only_in_s1_s2(
    dev_arms: tuple[Path, dict[str, Any]],
) -> None:
    out, report = dev_arms
    pack = load_pack(DEV)
    hard = {c["case_id"] for c in pack.cases if c.get("hard_negative_of")}
    s1_train = {str(s.meta["case_id"]) for s in read_samples(out / "S1" / "train.jsonl")}
    s3_train = {str(s.meta["case_id"]) for s in read_samples(out / "S3" / "train.jsonl")}
    assert not s1_train & hard and s3_train & hard
    assert report["arms"]["S1"]["left_out"].get("hard_negative_in_train", 0) > 0


def test_s2_renders_evidence_and_s1_does_not(tmp_path: Path) -> None:
    # The synthetic dev cases carry no evidence items; give some of them one.
    dev = load_pack(DEV)
    cases = [
        {**c, "evidence": [{"id": "e1", "field": "host", "value": f"nb-{n:03d}"},
                           {"id": "e2", "field": "detector_output", "value": "block"}]}
        if n % 2 == 0 else c
        for n, c in enumerate(dev.cases)
    ]  # fmt: skip
    exps = [dev.expectations[c["case_id"]] for c in dev.cases]
    pack_dir = tmp_path / "with-evidence"
    pack_dir.mkdir()
    (pack_dir / "cases.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases)
    )
    (pack_dir / "expectations.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in exps)
    )
    (pack_dir / "manifest.json").write_text(json.dumps(pack_manifest(pack_dir)))
    lock = triage_lock(cases, exps)
    build_ablation(
        load_pack(pack_dir), lock, SCENE, tmp_path / "abl", arms=["S1", "S2"], dry_run=True
    )
    s1 = list(read_samples(tmp_path / "abl" / "S1" / "test.jsonl"))
    s2 = list(read_samples(tmp_path / "abl" / "S2" / "test.jsonl"))
    assert not any("\nEvidence: " in s.state for s in s1)
    rendered = [s.state for s in s2 if "\nEvidence: " in s.state]
    assert rendered and all("host=nb-" in s for s in rendered)
    assert not any(
        "detector_output" in s or "block" in s.split("\nEvidence: ")[1] for s in rendered
    )


def test_unknown_arm_is_refused(tmp_path: Path) -> None:
    with pytest.raises(AblationError, match="unknown arms"):
        build_ablation(
            load_pack(DEV),
            json.loads((DEV / "lock.json").read_text()),
            SCENE,
            tmp_path,
            arms=["S9"],
        )


def test_telemetry_arms_with_and_without_vendor_signals(tmp_path: Path) -> None:
    pseud = Pseudonymizer(b"k" * 40)
    reqs = normalize_events(load_fixture_events(FIXTURE / "events.jsonl"), pseud)
    cf_cases, cf_exps, _ = build_cases(
        reqs, zone_alias="site-a", pseud=pseud, with_vendor_signals=True
    )
    labelled = {"false_positive": "unknown", "evidence_sufficiency": False, "owner": ["appsec"],
                "severity": "unknown", "urgency": "unknown"}  # fmt: skip
    cf_exps = [
        {**e, "label_origin": "single-human-reviewed", "label_status": "confirmed",
         "reviewers": ["r1"], "expected": labelled}
        for e in cf_exps
    ]  # fmt: skip
    dev = load_pack(DEV)
    cases = dev.cases + cf_cases
    exps = [dev.expectations[c["case_id"]] for c in dev.cases] + cf_exps
    pack_dir = tmp_path / "mixed"
    pack_dir.mkdir()
    (pack_dir / "cases.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases)
    )
    (pack_dir / "expectations.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in exps)
    )
    (pack_dir / "manifest.json").write_text(json.dumps(pack_manifest(pack_dir)))
    lock = triage_lock(cases, exps, training=[])
    report = build_ablation(load_pack(pack_dir), lock, SCENE, tmp_path / "abl", dry_run=True)
    vendor = {c["case_id"] for c in cf_cases if c["vendor_signals"]}
    telemetry = {c["case_id"] for c in cf_cases}

    def ids(arm: str) -> set[str]:
        return {str(s.meta["case_id"]) for s in read_samples(tmp_path / "abl" / arm / "all.jsonl")}

    assert not ids("S1") & telemetry and not ids("S3") & telemetry
    assert ids("S5") & vendor and not ids("S6") & vendor
    assert (ids("S6") & telemetry) == telemetry - vendor
    assert report["arms"]["S6"]["left_out"]["vendor_signals"] > 0
    # S2 never re-renders own-telemetry states: they already carry their evidence.
    s5 = {
        str(s.meta["case_id"]): s.state for s in read_samples(tmp_path / "abl" / "S5" / "all.jsonl")
    }
    assert all("\nEvidence: " not in s5[c] for c in telemetry if c in s5)


def test_an_arm_corpus_fits_and_checks_insufficient(
    dev_arms: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    out, _ = dev_arms
    model = tmp_path / "S1-model"
    from jef_train import fit

    rc = fit.main(["--corpus", str(out / "S1"), "--out", str(model), "--backbone", "hashing",
                   "--epochs", "1", "--cache-dir", str(tmp_path / "cache"), "--no-class-weighting"])  # fmt: skip
    assert rc == 0
    from jef_train.socbench.triage_candidate import check_candidate

    report = check_candidate(model, out / "S1", SCENE, backbone="hashing", repo_root=ROOT)
    assert report["status"] == "insufficient" and report["dry_run"] is True
    assert "label_prior_R5" in report
