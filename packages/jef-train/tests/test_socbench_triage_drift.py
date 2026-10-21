"""Drift: stable on identical runs, flags a shifted task, never assumes coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jef_train.socbench import cli
from jef_train.socbench.triage import load_pack, pack_manifest
from jef_train.socbench.triage_drift import PSI_SHIFTED, DriftError, drift_report, psi

ROOT = Path(__file__).resolve().parents[3]
DEV = ROOT / "benchmark" / "incident-triage" / "synthetic" / "dev-v1"
CAL = ROOT / "models" / "jef-v0" / "calibration.json"


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("drift") / "ref"
    assert (
        cli.main(
            ["triage", "evaluate", "--mode", "questions", "--pack", str(DEV), "--out", str(out)]
        )
        == 0
    )
    return out / "results.json"


def rewrite(report: Path, out: Path, qid: str, prediction: object) -> Path:
    data = json.loads(report.read_text())
    for row in data["rows"]:
        q = row["questions"][qid]
        if q["status"] == "answered":
            q["prediction"] = prediction
            if q["kind"] == "noul":
                q["p_true"] = 0.99
    out.write_text(json.dumps(data, ensure_ascii=False))
    return out


def test_psi_is_zero_on_identical_and_large_on_disjoint() -> None:
    assert psi({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) == 0
    assert psi({"a": 1.0}, {"b": 1.0}) > PSI_SHIFTED


def test_identical_runs_are_stable_and_change_nothing(report: Path) -> None:
    r = drift_report(report, report)
    assert {t["flag"] for t in r["tasks"].values()} == {"stable"}
    assert r["recommendation"].startswith("no shift flagged")
    assert {t["coverage"]["status"] for t in r["tasks"].values()} == {"NOT_MEASURED"}


def test_a_shifted_task_is_flagged_with_a_human_recommendation(
    report: Path, tmp_path: Path
) -> None:
    cur = rewrite(report, tmp_path / "cur.json", "owner", "fraud")
    r = drift_report(report, cur)
    assert r["tasks"]["owner"]["flag"] == "shifted"
    assert r["tasks"]["severity"]["flag"] == "stable"
    assert "answers shifted for owner" in r["recommendation"]
    assert "human-reviewed" in r["recommendation"]
    noul = rewrite(report, tmp_path / "noul.json", "urgent", True)
    assert drift_report(report, noul)["tasks"]["urgency"]["flag"] == "shifted"


def test_coverage_is_never_assumed(report: Path, tmp_path: Path) -> None:
    # Synthetic labels are by construction, not confirmed: nothing to measure.
    r = drift_report(report, report, pack=load_pack(DEV), calibration=CAL)
    assert r["tasks"]["owner"]["coverage"]["status"] == "NOT_MEASURED"
    # Confirmed labels against the shipped legacy calibration: no fitted
    # quantile applies, so the check says unfitted -- not "holds".
    dev = load_pack(DEV)
    pack_dir = tmp_path / "confirmed"
    pack_dir.mkdir()
    (pack_dir / "cases.jsonl").write_text(
        "".join(
            json.dumps({**c, "origin": "reviewer-sourced"}, ensure_ascii=False) + "\n"
            for c in dev.cases
        )
    )
    (pack_dir / "expectations.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    **dev.expectations[c["case_id"]],
                    "label_origin": "single-human-reviewed",
                    "label_status": "confirmed",
                    "reviewers": ["r1"],
                },
                ensure_ascii=False,
            )
            + "\n"
            for c in dev.cases
        )
    )
    (pack_dir / "manifest.json").write_text(json.dumps(pack_manifest(pack_dir)))
    r = drift_report(report, report, pack=load_pack(pack_dir), calibration=CAL)
    assert r["tasks"]["owner"]["coverage"]["status"] == "unfitted"


def test_other_scenes_scene_mode_and_private_strings_are_refused(
    report: Path, tmp_path: Path
) -> None:
    data = json.loads(report.read_text())
    data["provenance"]["scene"]["sha256"] = "sha256:" + "0" * 64
    (tmp_path / "s.json").write_text(json.dumps(data))
    with pytest.raises(DriftError, match="different scenes"):
        drift_report(report, tmp_path / "s.json")
    (tmp_path / "m.json").write_text(
        json.dumps({**json.loads(report.read_text()), "mode": "scene"})
    )
    with pytest.raises(DriftError, match="questions-mode"):
        drift_report(report, tmp_path / "m.json")
    leaky = rewrite(report, tmp_path / "leak.json", "owner", "10.1.2.3")
    with pytest.raises(DriftError, match="private IP"):
        drift_report(report, leaky)


def test_cli(report: Path, tmp_path: Path) -> None:
    out = tmp_path / "drift.json"
    assert (
        cli.main(
            [
                "triage",
                "drift",
                "--reference",
                str(report),
                "--current",
                str(report),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert json.loads(out.read_text())["evidence_class"].startswith("DRIFT AGGREGATES")


def test_the_workflow_never_runs_on_hosted_runners_or_pull_requests() -> None:
    import yaml

    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "triage-drift.yml").read_text())
    triggers = wf.get("on", wf.get(True))
    assert set(triggers) == {"workflow_dispatch"}
    job = wf["jobs"]["drift"]
    assert job["runs-on"] == ["self-hosted", "jef-private"]
    assert "JEF_PRIVATE_RUNNER_ENABLED" in job["if"]
