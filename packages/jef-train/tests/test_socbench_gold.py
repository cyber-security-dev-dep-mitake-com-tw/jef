"""Two-reviewer gold and the split lock (item 08).

DATA-003 (paired and family cases share a split) and DATA-005 (no
double-human-reviewed label without two reviewers) are the review's tests;
the rest pin blindness, kappa, adjudication rules and tamper detection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jef_train.socbench import SocbenchRepo, cli
from jef_train.socbench.annotate import (
    AnnotationError,
    adjudicate,
    agreement,
    cohen_kappa,
    export_tasks,
    parse_annotations,
)
from jef_train.socbench.records import write_jsonl
from jef_train.socbench.splits import (
    LockError,
    assign_splits,
    check_leakage,
    make_lock,
    verify_lock,
)

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def repo() -> SocbenchRepo:
    return SocbenchRepo(ROOT / "benchmark" / "socbench")


def case(n: int, family: str, text: str | None = None, **extra) -> dict:
    return {
        "case_id": f"socb-t-{n:06d}",
        "family_id": family,
        "source_dataset": "jef-analyst-reviewed",
        "source_hash": "sha256:" + "0" * 64,
        "telemetry_origin": "lab-reproduction",
        "language": "en",
        "task_id": "soc.alert.verdict.v1",
        "state": {"alert": text or f"alert {n}"},
        "evidence": [
            {"id": "e1", "field": "alert", "value": text or f"alert {n}"},
            {"id": "e2", "field": "host", "value": "h"},
        ],
        **extra,
    }


def label(cid: str, who: str, verdict: str, **extra) -> dict:
    return {"case_id": cid, "annotator": who, "verdict": verdict, "confidence": "high", **extra}


CASES = [case(i, f"fam-{i % 4}") for i in range(1, 9)]
BY_ID = {c["case_id"]: c for c in CASES}
A, B, C = "analyst-anon-01", "analyst-anon-02", "analyst-anon-03"


# --------------------------------------------------------------------------- #
# Blind export
# --------------------------------------------------------------------------- #


def test_exported_tasks_are_blind_and_reviewer_ordered() -> None:
    weak = [{**c, "gold": {"verdict": "suspicious"}, "hard_negative": True} for c in CASES]
    ta, tb = export_tasks(weak, A), export_tasks(weak, B)
    blob = json.dumps(ta)
    assert 'suspicious"' not in blob.replace('"suspicious",', ""), "weak gold must not leak"
    assert all("gold" not in t and "hard_negative" not in t for t in ta)
    assert all(t["verdict"] is None for t in ta)
    assert [t["case_id"] for t in ta] != [t["case_id"] for t in tb], (
        "each reviewer has their own order"
    )
    assert sorted(t["case_id"] for t in ta) == sorted(BY_ID)
    with pytest.raises(AnnotationError, match="handle"):
        export_tasks(CASES, "alice")


def test_annotations_are_validated() -> None:
    cid = CASES[0]["case_id"]
    with pytest.raises(AnnotationError, match="not one of"):
        parse_annotations([label(cid, A, "malicious")], BY_ID)
    with pytest.raises(AnnotationError, match="unknown case"):
        parse_annotations([label("socb-t-999999", A, "suspicious")], BY_ID)
    with pytest.raises(AnnotationError, match="evidence ids"):
        parse_annotations([label(cid, A, "suspicious", evidence_ids=["e9"])], BY_ID)
    with pytest.raises(AnnotationError, match="twice"):
        parse_annotations([label(cid, A, "suspicious"), label(cid, A, "unknown")], BY_ID)


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #


def test_cohen_kappa_matches_the_textbook_case() -> None:
    # 50 items: 20 yes/yes, 15 no/no, 5 yes/no, 10 no/yes -> po 0.7, pe 0.5, kappa 0.4
    a = ["y"] * 20 + ["n"] * 15 + ["y"] * 5 + ["n"] * 10
    b = ["y"] * 20 + ["n"] * 15 + ["n"] * 5 + ["y"] * 10
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    assert cohen_kappa(["x", "x"], ["x", "x"]) is None
    assert cohen_kappa([], []) is None


def _reviews() -> list:
    rows = []
    for c in CASES:
        rows.append(
            label(
                c["case_id"],
                A,
                "suspicious",
                evidence_ids=["e1", "e2"],
                prohibited_actions=["close"],
            )
        )
        rows.append(
            label(
                c["case_id"],
                B,
                "suspicious" if c["case_id"] != CASES[0]["case_id"] else "confirmed_benign",
                evidence_ids=["e1"],
            )
        )
    return parse_annotations(rows, BY_ID)


def test_agreement_reports_kappa_and_queues_disagreements() -> None:
    report = agreement(_reviews(), BY_ID)
    task = report["by_task"]["soc.alert.verdict.v1"]
    assert task["pairs"] == 8
    assert task["percent_agreement"] == 0.875
    assert [q["case_id"] for q in report["queue"]] == [CASES[0]["case_id"]]
    assert report["reviewers"] == [A, B]


# --------------------------------------------------------------------------- #
# Adjudication (DATA-005)
# --------------------------------------------------------------------------- #


def test_agreed_reviews_become_double_reviewed_gold(repo: SocbenchRepo) -> None:
    golds = {g["case_id"]: g for g in adjudicate(_reviews(), BY_ID)}
    agreed = golds[CASES[1]["case_id"]]
    assert agreed["label_origin"] == "double-human-reviewed"
    assert agreed["adjudication_status"] == "agreed"
    assert agreed["annotators"] == [A, B]
    assert agreed["gold"]["correct_evidence_ids"] == ["e1"], "only evidence both reviewers cited"
    assert agreed["gold"]["prohibited_actions"] == ["close"]
    disputed = golds[CASES[0]["case_id"]]
    assert disputed["adjudication_status"] == "disputed"
    assert disputed["gold"]["verdict"] == "unknown"
    for g in golds.values():
        repo.validate("gold", g, where=g["case_id"])


def test_a_third_person_adjudicates_with_a_reason(repo: SocbenchRepo) -> None:
    cid = CASES[0]["case_id"]
    ok = adjudicate(
        _reviews(),
        BY_ID,
        [
            {
                "case_id": cid,
                "adjudicator": C,
                "verdict": "confirmed_benign",
                "note": "approved scanner",
            }
        ],
    )
    g = next(x for x in ok if x["case_id"] == cid)
    assert g["adjudication_status"] == "adjudicated"
    assert g["gold"]["verdict"] == "confirmed_benign"
    assert g["gold"]["allowed_alternatives"] == ["suspicious"]
    assert g["adjudication_note"].startswith(C)
    repo.validate("gold", g, where=cid)
    with pytest.raises(AnnotationError, match="cannot adjudicate"):
        adjudicate(
            _reviews(),
            BY_ID,
            [{"case_id": cid, "adjudicator": A, "verdict": "suspicious", "note": "x"}],
        )
    with pytest.raises(AnnotationError, match="written reason"):
        adjudicate(_reviews(), BY_ID, [{"case_id": cid, "adjudicator": C, "verdict": "suspicious"}])


def test_data_005_one_reviewer_is_single_human_reviewed(repo: SocbenchRepo) -> None:
    only = parse_annotations([label(CASES[2]["case_id"], A, "suspicious")], BY_ID)
    (g,) = adjudicate(only, BY_ID)
    assert g["label_origin"] == "single-human-reviewed"
    assert g["annotators"] == [A]
    repo.validate("gold", g, where="single")
    forged = {**g, "label_origin": "double-human-reviewed"}
    with pytest.raises(ValueError, match=r"gold\.schema\.json"):
        repo.validate("gold", forged, where="forged")


# --------------------------------------------------------------------------- #
# Splits and the lock (DATA-003)
# --------------------------------------------------------------------------- #


def test_splits_are_per_family_and_deterministic() -> None:
    a = assign_splits(CASES, seed="s")
    assert a == assign_splits(CASES, seed="s")
    assert set(a) == {f"fam-{i}" for i in range(4)}
    assert assign_splits(CASES, seed="s", pinned={"fam-0": "test"})["fam-0"] == "test"


def test_data_003_a_pair_across_splits_is_leakage() -> None:
    pair = [case(1, "fam-a", "same alert"), case(2, "fam-b", "other", paired_with="socb-t-000001")]
    problems = check_leakage(pair, {"fam-a": "train", "fam-b": "test"})
    assert any("its pair" in p for p in problems)
    dupes = [case(1, "fam-a", "same"), case(2, "fam-b", "same")]
    assert any(
        "identical state" in p for p in check_leakage(dupes, {"fam-a": "train", "fam-b": "test"})
    )
    assert check_leakage(pair, {"fam-a": "test", "fam-b": "test"}) == []


def test_lock_refuses_missing_gold_and_detects_tampering() -> None:
    golds = adjudicate(_reviews(), BY_ID)
    with pytest.raises(LockError, match="no gold"):
        make_lock(CASES, golds[:-1])
    lock = make_lock(
        CASES,
        golds,
        pinned={"fam-0": "test", "fam-1": "train", "fam-2": "train", "fam-3": "calibration"},
    )
    assert sum(len(v) for v in lock["splits"].values()) == len(CASES)
    assert lock["counts"]["test"]["tier"] == {"A": 2}
    assert verify_lock(lock, CASES, golds) == []
    edited = [dict(g) for g in golds]
    edited[1] = {**edited[1], "gold": {**edited[1]["gold"], "verdict": "confirmed_malicious"}}
    assert "gold differs from the locked gold" in verify_lock(lock, CASES, edited)


def test_hard_negative_share_is_reported() -> None:
    cs = [case(1, "f", "a", hard_negative=True), case(2, "f", "b"), case(3, "f", "c")]
    gs = [
        {
            "case_id": c["case_id"],
            "task_id": "soc.alert.verdict.v1",
            "label_origin": "flow-label",
            "gold": {"verdict": v, "confidence_of_gold": "high"},
            "adjudication_status": "not-applicable",
        }
        for c, v in zip(
            cs, ["confirmed_benign", "confirmed_benign", "confirmed_malicious"], strict=True
        )
    ]
    lock = make_lock(cs, gs, pinned={"f": "test"})
    assert lock["hard_negative_share_of_negatives"]["test"] == 0.5


# --------------------------------------------------------------------------- #
# CLI round trip
# --------------------------------------------------------------------------- #


def test_cli_export_agree_adjudicate_lock(tmp_path: Path, capsys) -> None:
    root = str(ROOT / "benchmark" / "socbench")
    cases = tmp_path / "cases.jsonl"
    write_jsonl(cases, CASES)
    assert (
        cli.main(
            [
                "--root",
                root,
                "annotate",
                "export",
                "--cases",
                str(cases),
                "--annotator",
                A,
                "--out",
                str(tmp_path / "ta.jsonl"),
            ]
        )
        == 0
    )
    a_rows = [
        {**t, "verdict": "suspicious", "confidence": "high"}
        for t in (json.loads(x) for x in (tmp_path / "ta.jsonl").read_text().splitlines())
    ]
    b_rows = [{**r, "annotator": B} for r in a_rows]
    b_rows[0]["verdict"] = "confirmed_benign"
    write_jsonl(tmp_path / "a.jsonl", a_rows)
    write_jsonl(tmp_path / "b.jsonl", b_rows)
    labels = ["--labels", str(tmp_path / "a.jsonl"), "--labels", str(tmp_path / "b.jsonl")]
    assert (
        cli.main(
            [
                "--root",
                root,
                "annotate",
                "agree",
                "--cases",
                str(cases),
                *labels,
                "--out",
                str(tmp_path / "r.json"),
                "--queue",
                str(tmp_path / "q.jsonl"),
            ]
        )
        == 0
    )
    queued = json.loads((tmp_path / "q.jsonl").read_text().splitlines()[0])["case_id"]
    write_jsonl(
        tmp_path / "d.jsonl",
        [
            {
                "case_id": queued,
                "adjudicator": C,
                "verdict": "suspicious",
                "note": "scan matches the attack window",
            }
        ],
    )
    gold = tmp_path / "gold.jsonl"
    assert (
        cli.main(
            [
                "--root",
                root,
                "annotate",
                "adjudicate",
                "--cases",
                str(cases),
                *labels,
                "--decisions",
                str(tmp_path / "d.jsonl"),
                "--out",
                str(gold),
            ]
        )
        == 0
    )
    lock = tmp_path / "lock.json"
    pins = [
        x for fam in range(4) for x in ("--pin", f"fam-{fam}={'test' if fam == 0 else 'train'}")
    ]
    assert (
        cli.main(
            [
                "--root",
                root,
                "lock",
                "--cases",
                str(cases),
                "--gold",
                str(gold),
                "--out",
                str(lock),
                *pins,
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "--root",
                root,
                "lock",
                "--verify",
                str(lock),
                "--cases",
                str(cases),
                "--gold",
                str(gold),
            ]
        )
        == 0
    )
    assert "0 still disputed" in capsys.readouterr().out
