"""NEXT-03 bridge: locked SOCBench gold becomes a fit() corpus without leakage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jef_train.schema import read_samples
from jef_train.socbench import cli
from jef_train.socbench.corpus_export import ExportError, export_corpus, samples_from_gold
from jef_train.socbench.records import write_jsonl
from jef_train.socbench.splits import LockError, make_lock

TASK = "soc.alert.verdict.v1"


def case(n: int, family: str, lang: str = "en") -> dict:
    return {
        "case_id": f"socb-t-{n:06d}",
        "family_id": family,
        "source_dataset": "jef-analyst-reviewed",
        "source_hash": "sha256:" + "0" * 64,
        "telemetry_origin": "lab-reproduction",
        "language": lang,
        "task_id": TASK,
        "state": {"alert": f"alert {n} in family {family}", "host": f"h{n}"},
        "evidence": [{"id": "e1", "field": "alert", "value": f"alert {n}"}],
        "hard_negative": n % 3 == 0,
    }


def gold(cid: str, verdict: str, origin: str = "double-human-reviewed") -> dict:
    return {
        "case_id": cid,
        "task_id": TASK,
        "label_origin": origin,
        "gold": {"verdict": verdict, "confidence_of_gold": "high"},
        "annotators": ["analyst-anon-01", "analyst-anon-02"],
        "adjudication_status": "agreed",
    }


VERDICTS = ["confirmed_malicious", "suspicious", "confirmed_benign", "insufficient_evidence"]


def fixture(weak: int = 0) -> tuple[list[dict], list[dict], dict]:
    cases = [case(n, f"fam-{n // 2}", "zh-TW" if n % 4 == 0 else "en") for n in range(1, 25)]
    golds = []
    for i, c in enumerate(cases):
        origin = "attack-phase-window" if i < weak else "double-human-reviewed"
        golds.append(gold(c["case_id"], VERDICTS[i % 4], origin))
    # 13 families is small enough that a seeded hash can leave a split empty;
    # pin one family into each so the fixture always exercises all three.
    lock = make_lock(
        cases, golds, seed="t", pinned={"fam-11": "calibration", "fam-12": "test", "fam-0": "train"}
    )
    return cases, golds, lock


def test_samples_follow_the_lock_and_carry_task_scoped_buckets() -> None:
    cases, golds, lock = fixture()
    split_of = {cid: s for s, ids in lock["splits"].items() for cid in ids}
    samples, excluded = samples_from_gold(cases, golds, split_of)
    assert len(samples) == 24 and not excluded
    s = samples[0]
    assert s.kind == "choice" and s.task == TASK
    assert s.bucket == f"{TASK}|choice:5"
    assert s.option_keys[s.label] == golds[0]["gold"]["verdict"]
    assert s.group == cases[0]["family_id"]
    assert "alert 1 in family" in s.state and "host: h1" in s.state
    assert s.meta["split"] == split_of[s.meta["case_id"]]


def test_weak_labels_are_excluded_and_counted_not_trained_on() -> None:
    cases, golds, lock = fixture(weak=5)
    split_of = {cid: s for s, ids in lock["splits"].items() for cid in ids}
    samples, excluded = samples_from_gold(cases, golds, split_of)
    assert len(samples) == 19 and excluded == {"D": 5}
    looser, excluded2 = samples_from_gold(cases, golds, split_of, min_tier="D")
    assert len(looser) == 24 and not excluded2


def test_export_writes_fit_layout_without_family_leakage(tmp_path: Path) -> None:
    cases, golds, lock = fixture(weak=3)
    manifest = export_corpus(cases, golds, lock, tmp_path / "corpus")
    for name in ("train", "calibration", "test", "all"):
        assert (tmp_path / "corpus" / f"{name}.jsonl").is_file()
    assert manifest["splits"]["all"] == 21
    assert manifest["excluded_by_tier"] == {"D": 3}
    assert manifest["lock"]["cases_digest"] == lock["cases_digest"]
    assert "evaluation only" in manifest["test_split"]
    families = {
        name: {s.group for s in read_samples(tmp_path / "corpus" / f"{name}.jsonl")}
        for name in ("train", "calibration", "test")
    }
    assert not (families["train"] & families["test"])
    assert not (families["train"] & families["calibration"])
    assert not (families["calibration"] & families["test"])
    assert manifest["summary"]["by_bucket"] == {f"{TASK}|choice:5": 21}
    assert set(manifest["summary"]["by_state_lang"]) <= {"en", "zh-TW"}


def test_tampered_gold_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    cases, golds, lock = fixture()
    golds[0] = {**golds[0], "gold": {**golds[0]["gold"], "verdict": "confirmed_benign"}}
    with pytest.raises(LockError):
        export_corpus(cases, golds, lock, tmp_path / "corpus")
    assert not (tmp_path / "corpus").exists()


def test_a_verdict_outside_the_task_labels_is_an_error() -> None:
    cases, golds, lock = fixture()
    golds[1] = {**golds[1], "gold": {"verdict": "nope", "confidence_of_gold": "high"}}
    lock = make_lock(cases, golds, seed="t", pinned={"fam-11": "calibration", "fam-12": "test"})
    with pytest.raises(ExportError, match=r"not a soc\.alert\.verdict\.v1 label"):
        export_corpus(cases, golds, lock, Path("/nonexistent"))


def test_zh_tw_questions_are_native_not_translated_in_place() -> None:
    cases, golds, lock = fixture()
    split_of = {cid: s for s, ids in lock["splits"].items() for cid in ids}
    zh, _ = samples_from_gold(cases, golds, split_of, question_lang="zh-TW")
    en, _ = samples_from_gold(cases, golds, split_of, question_lang="en")
    assert zh[0].instructions != en[0].instructions
    assert "判定" in zh[0].instructions


def test_cli_export_corpus(tmp_path: Path, capsys) -> None:
    cases, golds, lock = fixture(weak=2)
    write_jsonl(tmp_path / "cases.jsonl", cases)
    write_jsonl(tmp_path / "gold.jsonl", golds)
    (tmp_path / "lock.json").write_text(json.dumps(lock), "utf-8")
    rc = cli.main(
        [
            "export-corpus",
            "--lock",
            str(tmp_path / "lock.json"),
            "--cases",
            str(tmp_path / "cases.jsonl"),
            "--gold",
            str(tmp_path / "gold.jsonl"),
            "--out",
            str(tmp_path / "corpus"),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "all=22" in out and "excluded below tier A" in out
    assert "jef_train.fit --corpus" in out
    assert json.loads((tmp_path / "corpus" / "manifest.json").read_text())["min_trust_tier"] == "A"
