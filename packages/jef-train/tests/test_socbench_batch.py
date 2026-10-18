"""Annotation batches in git: structure, data hygiene, agreement and independence."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from jef_train.socbench.batch import BatchError, check_batch, git_authors, new_batch
from jef_train.socbench.records import write_jsonl

TASK = "soc.ndr.technique.v1"
A, B, ADJ = "analyst-anon-01", "analyst-anon-02", "analyst-anon-03"


def cases(n: int = 4) -> list[dict[str, Any]]:
    return [
        {
            "case_id": f"c{i}",
            "task_id": TASK,
            "family_id": f"f{i % 2}",
            "source_dataset": "unit",
            "state": {"flows": i},
            "evidence": [{"id": "e1", "field": "flow", "value": f"flow {i}"}],
        }
        for i in range(n)
    ]


def label(cid: str, who: str, verdict: str) -> dict[str, Any]:
    return {
        "case_id": cid,
        "annotator": who,
        "verdict": verdict,
        "confidence": "high",
        "evidence_ids": ["e1"],
    }


def git(repo: Path, *args: str, email: str = "x@example.org") -> None:
    subprocess.run(  # noqa: S603
        ["git", "-c", f"user.email={email}", "-c", "user.name=t", *args],  # noqa: S607
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    return tmp_path


def make(repo: Path, purpose: str = "gold", reviewers: tuple[str, ...] = (A, B)) -> Path:
    d = repo / "batch-1"
    d.mkdir()
    meta = new_batch(
        cases(),
        batch_id="batch-1",
        purpose=purpose,
        reviewers=reviewers,
        recipe="see README",
        adjudicator=ADJ if purpose == "gold" else None,
    )
    (d / "batch.json").write_text(json.dumps(meta))
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "batch", email="lead@example.org")
    return d


def commit_labels(repo: Path, d: Path, who: str, verdicts: list[str], email: str) -> None:
    write_jsonl(d / f"labels-{who}.jsonl", [label(f"c{i}", who, v) for i, v in enumerate(verdicts)])
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", f"labels {who}", email=email)


def test_two_independent_reviewers_and_an_adjudicator_make_tier_a(repo: Path) -> None:
    d = make(repo)
    commit_labels(repo, d, A, ["c2", "botnet", "benign", "unknown"], "a@example.org")
    commit_labels(repo, d, B, ["c2", "botnet", "benign", "benign"], "b@example.org")
    report = check_batch(d, cases=cases(), authors=git_authors(d))
    assert report["problems"] == []
    assert report["disagreements"] == 1 and report["open_disagreements"] == 1
    assert report["tier_if_merged"] is None, "an open disagreement blocks tier A"
    write_jsonl(
        d / "decisions.jsonl", [{"case_id": "c3", "adjudicator": ADJ, "verdict": "unknown"}]
    )
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "decide", email="adj@example.org")
    report = check_batch(d, cases=cases(), authors=git_authors(d))
    assert report["problems"] == [] and report["tier_if_merged"] == "A"
    assert report["kappa"] is not None


def test_one_author_for_both_reviewers_is_refused(repo: Path) -> None:
    d = make(repo)
    commit_labels(repo, d, A, ["c2", "botnet", "benign", "unknown"], "same@example.org")
    commit_labels(repo, d, B, ["c2", "botnet", "benign", "unknown"], "same@example.org")
    report = check_batch(d, authors=git_authors(d))
    assert any("same author" in p for p in report["problems"])
    assert report["tier_if_merged"] is None


def test_one_commit_with_both_reviewers_files_is_refused(repo: Path) -> None:
    d = make(repo)
    write_jsonl(d / f"labels-{A}.jsonl", [label("c0", A, "c2")])
    write_jsonl(d / f"labels-{B}.jsonl", [label("c0", B, "c2")])
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "both", email="a@example.org")
    report = check_batch(d, authors=git_authors(d))
    assert any("more than one reviewer" in p for p in report["problems"])


def test_case_text_never_enters_the_repository(repo: Path) -> None:
    d = make(repo)
    write_jsonl(d / f"labels-{A}.jsonl", [{**label("c0", A, "c2"), "state": {"flows": 0}}])
    report = check_batch(d)
    assert any("not allowed in the repository" in p for p in report["problems"])


def test_decisions_only_by_the_adjudicator_and_only_on_disagreements(repo: Path) -> None:
    d = make(repo)
    commit_labels(repo, d, A, ["c2", "botnet", "benign", "unknown"], "a@example.org")
    commit_labels(repo, d, B, ["c2", "botnet", "benign", "benign"], "b@example.org")
    write_jsonl(
        d / "decisions.jsonl",
        [
            {"case_id": "c0", "adjudicator": ADJ, "verdict": "c2"},  # agreed case
            {"case_id": "c3", "adjudicator": A, "verdict": "unknown"},  # a reviewer
        ],
    )
    problems = check_batch(d)["problems"]
    assert any("not a disagreement" in p for p in problems)
    assert any("not the batch's adjudicator" in p for p in problems)


def test_a_pilot_has_one_reviewer_and_reaches_tier_b(repo: Path) -> None:
    d = make(repo, "pilot", (A,))
    commit_labels(repo, d, A, ["c2", "botnet", "benign", "unknown"], "a@example.org")
    report = check_batch(d, cases=cases(), authors=git_authors(d))
    assert report["problems"] == [] and report["tier_if_merged"] == "B"


def test_malformed_batches_are_refused(repo: Path) -> None:
    d = repo / "batch-1"
    d.mkdir()
    meta = new_batch(cases(), batch_id="batch-1", purpose="gold", reviewers=[A, A], recipe="r")
    (d / "batch.json").write_text(json.dumps(meta))
    with pytest.raises(BatchError, match="distinct"):
        check_batch(d)
    meta = new_batch(
        cases(), batch_id="batch-1", purpose="gold", reviewers=[A, B], recipe="r", adjudicator=A
    )
    (d / "batch.json").write_text(json.dumps(meta))
    with pytest.raises(BatchError, match="third person"):
        check_batch(d)


def test_a_wrong_case_file_is_caught(repo: Path) -> None:
    d = make(repo)
    other = cases()
    other[0]["state"] = {"flows": 99}
    assert any("digest" in p for p in check_batch(d, cases=other)["problems"])
