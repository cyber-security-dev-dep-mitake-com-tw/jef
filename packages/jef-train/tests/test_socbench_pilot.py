"""NEXT-V3-06: the single-reviewer pilot report says what one reviewer can show, and no more."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jef_train.socbench import cli
from jef_train.socbench.annotate import parse_annotations
from jef_train.socbench.pilot import pilot_report
from jef_train.socbench.records import write_jsonl

TASK = "soc.ndr.technique.v1"
PINNED = "sha256:" + "1f8daeca146131a368b432b2d8625de5e4429bce833f3e4cf4bea8312344ab7f"


def case(n: int, source_hash: str = PINNED) -> dict:
    return {
        "case_id": f"socb-ctu13-{n:06d}",
        "family_id": f"ctu13-scenario-{n % 3}",
        "source_dataset": "ctu-13",
        "source_hash": source_hash,
        "telemetry_origin": "lab-reproduction",
        "language": "n/a",
        "task_id": TASK,
        "state": {"flow": f"flow {n}"},
        "evidence": [
            {"id": "e1", "field": "flow", "value": f"flow {n}"},
            {"id": "e2", "field": "dst", "value": "x"},
        ],
    }


def label(cid: str, verdict: str, **extra) -> dict:
    row = {
        "case_id": cid,
        "annotator": "analyst-anon-01",
        "verdict": verdict,
        "confidence": "medium",
        "evidence_ids": ["e1"],
        "note": "periodic beacon to one host",
        "seconds_spent": 240,
    }
    row.update(extra)
    return row


def fixture() -> tuple[dict, list, list]:
    cases = {c["case_id"]: c for c in (case(n) for n in range(1, 9))}
    cases["socb-ctu13-000099"] = case(99, source_hash="sha256:" + "0" * 64)
    ids = sorted(cases)
    rows = [
        label(ids[0], "c2"),
        label(ids[1], "c2", seconds_spent=600),
        label(ids[2], "benign"),
        label(ids[3], "unknown", evidence_ids=[]),
        label(ids[4], "botnet", note=""),
        label(ids[5], "benign", evidence_ids=[]),
        label(ids[6], "c2", seconds_spent=None),
        label(ids[7], "unknown", evidence_ids=[]),
        label(ids[8], "benign"),
    ]
    weak = [
        {
            "case_id": i,
            "task_id": TASK,
            "label_origin": "flow-label",
            "gold": {"verdict": v, "confidence_of_gold": "medium"},
        }
        for i, v in zip(
            ids,
            ["c2", "botnet", "benign", "unknown", "botnet", "benign", "c2", "benign", "benign"],
            strict=True,
        )
    ]
    return cases, parse_annotations(rows, cases), weak


def test_report_is_tier_b_and_measures_completeness_field_by_field() -> None:
    cases, anns, weak = fixture()
    r = pilot_report(
        anns, cases, weak_golds=weak, pinned_hashes={"ctu-13": {PINNED}}, hourly_rate=60.0
    )
    assert r["trust_tier"] == "B" and r["label_origin"] == "single-human-reviewed"
    assert r["cases"] == 9
    by = r["completeness"]["by_field"]
    assert by["note"] == pytest.approx(8 / 9, abs=1e-3), "one empty note"
    assert by["evidence"] == pytest.approx(8 / 9, abs=1e-3), (
        "a benign with no evidence; unknowns are exempt"
    )
    assert by["timed"] == pytest.approx(8 / 9, abs=1e-3)
    assert r["completeness"]["complete_share"] == pytest.approx(6 / 9, abs=1e-3)
    assert r["undetermined"]["count"] == 2


def test_traceability_time_cost_and_the_source_check_are_labelled() -> None:
    cases, anns, weak = fixture()
    r = pilot_report(
        anns, cases, weak_golds=weak, pinned_hashes={"ctu-13": {PINNED}}, hourly_rate=60.0
    )
    assert r["traceability"]["traceable_share"] == pytest.approx(8 / 9, abs=1e-3)
    assert list(r["traceability"]["untraceable"]) == [
        "ctu-13: source_hash is not a pinned artifact"
    ]
    assert r["time"]["median_seconds"] == 240.0 and r["time"]["p90_seconds"] == 600.0
    s = r["source_label_check"]
    assert s["compared"] == 9 and s["matches_source_label"] == 7
    assert "NOT inter-annotator" in s["caveat"] and "kappa" not in json.dumps(
        {k: v for k, v in s.items() if k != "caveat"}
    )
    # mean of the timed reviews: (7 x 240 + 600) / 8 = 285 s -> 285 / 3600 x 60
    assert r["reviewer_cost"]["per_case"] == pytest.approx(4.75, abs=0.01)
    assert (
        "single-reviewer" in r["may_be_described_as"]
        and "not a double-reviewed" in r["may_be_described_as"]
    )


def test_two_reviewers_are_refused() -> None:
    cases, anns, _ = fixture()
    second = parse_annotations(
        [{**label(sorted(cases)[0], "c2"), "annotator": "analyst-anon-02"}], cases
    )
    with pytest.raises(ValueError, match="one reviewer"):
        pilot_report([*anns, *second], cases)


def test_cli_pilot_report_reads_real_pins(tmp_path: Path, capsys) -> None:
    cases, _, weak = fixture()
    rows = [label(cid, "c2") for cid in sorted(cases)]
    write_jsonl(tmp_path / "cases.jsonl", cases.values())
    write_jsonl(tmp_path / "labels.jsonl", rows)
    write_jsonl(tmp_path / "weak.jsonl", weak)
    rc = cli.main(
        [
            "annotate",
            "pilot-report",
            "--cases",
            str(tmp_path / "cases.jsonl"),
            "--labels",
            str(tmp_path / "labels.jsonl"),
            "--weak-gold",
            str(tmp_path / "weak.jsonl"),
            "--hourly-rate",
            "50",
            "--out",
            str(tmp_path / "pilot.json"),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    assert out.startswith("Tier B pilot by analyst-anon-01: 9 cases")
    r = json.loads((tmp_path / "pilot.json").read_text())
    # The repository's own ctu-13 pin is the real archive digest, so 8 of 9 trace.
    assert r["traceability"]["traceable_share"] == pytest.approx(8 / 9, abs=1e-3)
