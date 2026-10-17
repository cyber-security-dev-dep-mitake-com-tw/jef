"""NEXT-B: the sampler that produces sampled-cases.jsonl for blind annotation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jef_train.socbench import cli
from jef_train.socbench.records import write_jsonl
from jef_train.socbench.report import evaluate
from jef_train.socbench.sample import MIN_FAMILIES_FOR_CI, SampleError, sample_cases
from jef_train.socbench.splits import make_lock

TASK = "soc.ndr.technique.v1"


def case(n: int, family: str, task: str = TASK, **extra) -> dict:
    return {
        "case_id": f"socb-t-{n:06d}",
        "family_id": family,
        "source_dataset": "ctu-13",
        "source_hash": "sha256:" + "0" * 64,
        "telemetry_origin": "lab-reproduction",
        "language": "n/a",
        "task_id": task,
        "state": {"flow": f"flow {n}"},
        "evidence": [{"id": "e1", "field": "flow", "value": f"flow {n}"}],
        **extra,
    }


def weak(cid: str, verdict: str) -> dict:
    return {
        "case_id": cid,
        "task_id": TASK,
        "label_origin": "flow-label",
        "gold": {"verdict": verdict, "confidence_of_gold": "medium"},
    }


def corpus() -> tuple[list[dict], list[dict]]:
    """3 families; each 200 benign, 6 c2, 30 unknown -- c2 is rare, as in CTU-13."""
    cases, golds, n = [], [], 0
    for f in range(3):
        for verdict, count in (("benign", 200), ("c2", 6), ("unknown", 30)):
            for _ in range(count):
                n += 1
                c = case(n, f"ctu13-scenario-{f}")
                cases.append(c)
                golds.append(weak(c["case_id"], verdict))
    return cases, golds


def test_rare_classes_get_their_minimum_and_weights_undo_it() -> None:
    cases, golds = corpus()
    chosen, m = sample_cases(cases, golds, per_family=30, min_per_stratum=5)
    assert m["families"] == 3 and m["sampled"] == len(chosen)
    for fam in m["by_family"].values():
        strata = fam["strata"]
        assert strata["c2"]["sampled"] == 6, "all rare positives kept (fewer than the floor allows)"
        assert strata["unknown"]["sampled"] >= 5
    w = m["inverse_inclusion_weights"]
    c2_ids = {g["case_id"] for g in golds if g["gold"]["verdict"] == "c2"}
    assert all(w[cid] == 1.0 for cid in c2_ids if cid in w), "every c2 was taken: weight 1"
    benign_w = {w[c["case_id"]] for c in chosen if c["case_id"] not in c2_ids} - {1.0}
    assert benign_w and min(benign_w) > 1.0, "benign was subsampled, so its weight is > 1"


def test_output_is_blind_deterministic_and_family_whole() -> None:
    cases, golds = corpus()
    a, ma = sample_cases(cases, golds, per_family=20, seed="s1")
    b, _ = sample_cases(cases, golds, per_family=20, seed="s1")
    c, _ = sample_cases(cases, golds, per_family=20, seed="s2")
    assert [x["case_id"] for x in a] == [x["case_id"] for x in b], "same seed, same draw"
    assert [x["case_id"] for x in a] != [x["case_id"] for x in c], "another seed, another draw"
    assert all("gold" not in x and "verdict" not in x for x in a), "reviewers stay blind"
    assert ma["weak_labels_used_for_stratification_only"] is True


def test_exclude_makes_a_second_batch_disjoint_and_task_filters() -> None:
    cases, golds = corpus()
    cases.append(case(9999, "phish-fam", task="soc.phishing.verdict.v1"))
    first, _ = sample_cases(cases, golds, task=TASK, per_family=10)
    second, _ = sample_cases(
        cases, golds, task=TASK, per_family=10, exclude=[c["case_id"] for c in first]
    )
    assert not {c["case_id"] for c in first} & {c["case_id"] for c in second}
    assert all(c["task_id"] == TASK for c in first + second)
    with pytest.raises(SampleError):
        sample_cases(cases, golds, task="soc.identity.abuse.v1")


def test_too_few_families_is_said_out_loud() -> None:
    cases, golds = corpus()
    _, m = sample_cases(cases, golds, per_family=10)
    assert m["families"] < MIN_FAMILIES_FOR_CI
    assert any("independent families" in w for w in m["warnings"])


def test_cli_sample_writes_cases_and_manifest(tmp_path: Path, capsys) -> None:
    cases, golds = corpus()
    write_jsonl(tmp_path / "cases.jsonl", cases)
    write_jsonl(tmp_path / "gold.jsonl", golds)
    rc = cli.main(
        [
            "sample",
            "--cases",
            str(tmp_path / "cases.jsonl"),
            "--weak-gold",
            str(tmp_path / "gold.jsonl"),
            "--task",
            TASK,
            "--per-family",
            "15",
            "--out",
            str(tmp_path / "sampled-cases.jsonl"),
        ]
    )
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "families" in out.out and "warning:" in out.err
    m = json.loads((tmp_path / "sampled-cases.manifest.json").read_text())
    assert m["sampled"] == sum(1 for _ in (tmp_path / "sampled-cases.jsonl").open())


# --------------------------------------------------------------------------- #
# report: failures are counted, hard negatives are their own track
# --------------------------------------------------------------------------- #


class Flaky:
    name = "flaky"

    def __init__(self) -> None:
        self.calls = 0

    def describe(self) -> dict:
        return {"type": "test", "egress": "none"}

    def predict(self, case: dict) -> dict[str, float]:
        self.calls += 1
        if self.calls % 3 == 0:
            raise TimeoutError("upstream did not answer")
        return {
            "c2": 0.6,
            "botnet": 0.1,
            "dns_tunnel": 0.05,
            "normal_automation": 0.05,
            "benign": 0.15,
            "unknown": 0.05,
        }


def test_scorer_failures_are_reported_not_fatal() -> None:
    cases = [case(n, f"fam-{n % 6}", hard_negative=(n % 2 == 0)) for n in range(1, 31)]
    golds = [
        {
            "case_id": c["case_id"],
            "task_id": TASK,
            "label_origin": "double-human-reviewed",
            "gold": {"verdict": "c2" if i % 3 else "benign", "confidence_of_gold": "high"},
            "annotators": ["analyst-anon-01", "analyst-anon-02"],
            "adjudication_status": "agreed",
        }
        for i, c in enumerate(cases)
    ]
    lock = make_lock(cases, golds, seed="t", ratios={"test": 1.0})
    report = evaluate(cases, golds, lock, [Flaky()], split="test")
    s = report["scorers"]["flaky"]
    assert s["failures"]["count"] == 10 and s["failures"]["by_error"] == {"TimeoutError": 10}
    assert abs(s["failures"]["rate"] - 10 / 30) < 1e-3
    assert "p99" in s["latency_s"]
    block = s["tasks"][TASK]
    assert block["n"] == 20, "metrics cover the answered cases"
    assert set(block["by_track"]) == {"hard_negative", "natural"}
    assert report["comparison"][0]["failure_rate"] == pytest.approx(10 / 30, abs=1e-3)


# --------------------------------------------------------------------------- #
# pilot throughput: seconds_spent is optional, validated and summarised
# --------------------------------------------------------------------------- #


def test_agreement_reports_measured_reviewer_throughput() -> None:
    from jef_train.socbench.annotate import AnnotationError, agreement, parse_annotations

    cases = {c["case_id"]: c for c in (case(n, "fam-a") for n in range(1, 5))}
    rows = []
    for cid in cases:
        for who, secs in (("analyst-anon-01", 300), ("analyst-anon-02", 420)):
            rows.append(
                {
                    "case_id": cid,
                    "annotator": who,
                    "verdict": "benign",
                    "confidence": "high",
                    "seconds_spent": secs,
                }
            )
    rows.append(
        {
            "case_id": "socb-t-000001",
            "annotator": "analyst-anon-03",
            "verdict": "c2",
            "confidence": "low",
        }
    )  # untimed reviews are allowed
    report = agreement(parse_annotations(rows, cases), cases)
    t = report["timing"][TASK]
    assert t["reviews_timed"] == 8
    assert t["median_seconds"] == 420.0
    assert t["reviewer_hours_per_100_cases_double_reviewed"] == pytest.approx(2 * 100 * 360 / 3600)
    with pytest.raises(AnnotationError, match="seconds_spent"):
        parse_annotations([{**rows[0], "annotator": "analyst-anon-09", "seconds_spent": -1}], cases)
