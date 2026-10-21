"""The interactive labeller: what it saves is what `review import` accepts.

Keystrokes are scripted through the same ``ask`` hook the terminal uses. The
worksheet is a synthetic dry-run batch; none of these answers is a label.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from jef_scene import load_scene
from jef_train.socbench import cli
from jef_train.socbench.triage import load_pack
from jef_train.socbench.triage_label import Labeller, consistency_warnings, load_answers
from jef_train.socbench.triage_review import (
    ReviewError,
    export_worksheet,
    import_answers,
    sample_batch,
)

ROOT = Path(__file__).resolve().parents[3]
DEV = ROOT / "benchmark" / "incident-triage" / "synthetic" / "dev-v1"
SCENE = load_scene(ROOT / "scenes" / "incident-triage.zh-tw.yaml")

# event, alert, is_false_positive, has_evidence, owner, severity, urgent, context, notes, confirm
CONSISTENT = ["1", "1", "2", "1", "1", "4,5", "1", "", "", ""]


def worksheet() -> tuple[list[Any], dict[str, Any], list[dict[str, Any]]]:
    packs = [load_pack(DEV)]
    batch = sample_batch(
        packs, batch_id="dry", strata={"edr": 2, "identity": 1}, allow_synthetic=True
    )
    return packs, batch, export_worksheet(packs, batch, SCENE, reviewer="r1")


def keys(*seqs: list[str]) -> Iterator[str]:
    for seq in seqs:
        yield from seq


def labeller(rows: list[dict[str, Any]], path: Path, script: Iterator[str]) -> Labeller:
    tick = iter(range(0, 10_000, 30))
    out: list[str] = []

    def ask(prompt: str) -> str:
        out.append(prompt)
        try:
            return next(script)
        except StopIteration:
            raise EOFError from None

    lab = Labeller(
        rows, path, ask=ask, say=out.append, clock=lambda: float(next(tick)), color=False
    )
    lab.screen = out  # type: ignore[attr-defined]
    return lab


def test_a_full_session_is_accepted_by_import_as_tier_b(tmp_path: Path) -> None:
    packs, batch, rows = worksheet()
    path = tmp_path / "a.jsonl"
    left = labeller(rows, path, keys(CONSISTENT, CONSISTENT, CONSISTENT)).run()
    assert left == 0
    saved = [json.loads(x) for x in path.read_text().splitlines()]
    assert [r["case_id"] for r in saved] == [r["case_id"] for r in rows]
    assert saved[0]["answers"]["severity"] == ["高", "危急"]
    assert saved[0]["answers"]["owner"] == ["soc"]
    assert saved[0]["minutes_spent"] > 0
    assert path.stat().st_mode & 0o077 == 0
    _, exps, _ = import_answers(packs, batch, SCENE, [saved])
    assert {e["label_origin"] for e in exps} == {"single-human-reviewed"}


def test_quit_keeps_what_was_confirmed_and_the_next_run_resumes(tmp_path: Path) -> None:
    _, _, rows = worksheet()
    path = tmp_path / "a.jsonl"
    first = labeller(rows, path, keys(CONSISTENT, ["1", "q"]))
    assert first.run() == 2
    assert len(load_answers(path, rows)) == 1
    second = labeller(rows, path, keys(CONSISTENT, CONSISTENT))
    assert second.run() == 0
    assert any(rows[1]["case_id"] in line for line in second.screen)  # type: ignore[attr-defined]
    assert not any(  # the confirmed first case is not asked again
        f"Case 1/3   {rows[0]['case_id']}" in line
        for line in second.screen  # type: ignore[attr-defined]
    )


def test_back_changes_an_earlier_answer(tmp_path: Path) -> None:
    _, _, rows = worksheet()
    path = tmp_path / "a.jsonl"
    # answer the event as benign, go back from the alert question, change it to malicious
    script = ["2", "b", "1", *CONSISTENT[1:]]
    labeller(rows[:1], path, keys(script)).run()
    assert load_answers(path, rows[:1])[rows[0]["case_id"]]["truth_of_event"] == "malicious"


def test_bad_input_is_asked_again_not_guessed(tmp_path: Path) -> None:
    _, _, rows = worksheet()
    path = tmp_path / "a.jsonl"
    script = ["9", "x", "1,1", "1", *CONSISTENT[1:]]
    lab = labeller(rows[:1], path, keys(script))
    lab.run()
    assert load_answers(path, rows[:1])[rows[0]["case_id"]]["truth_of_event"] == "malicious"
    assert sum("請輸入列出的號碼" in line for line in lab.screen) == 3  # type: ignore[attr-defined]


def test_a_single_answer_question_refuses_two(tmp_path: Path) -> None:
    _, _, rows = worksheet()
    script = ["1,2", "1", *CONSISTENT[1:]]
    lab = labeller(rows[:1], tmp_path / "a.jsonl", keys(script))
    lab.run()
    assert any("只能選一個" in line for line in lab.screen)  # type: ignore[attr-defined]


def test_unknown_is_one_key_and_reaches_import(tmp_path: Path) -> None:
    packs, batch, rows = worksheet()
    path = tmp_path / "a.jsonl"
    unknowns = ["u", "u", "u", "2", "u", "u", "u", "1,4", "", ""]
    labeller(rows, path, keys(unknowns, unknowns, unknowns)).run()
    saved = [json.loads(x) for x in path.read_text().splitlines()]
    assert saved[0]["answers"]["owner"] == "unknown"
    assert saved[0]["required_context"] == ["change_ticket", "asset_owner"]
    import_answers(packs, batch, SCENE, [saved])


def test_contradictions_are_warned_and_need_a_second_yes(tmp_path: Path) -> None:
    _, _, rows = worksheet()
    path = tmp_path / "a.jsonl"
    # alert = false positive but is_false_positive = no; refuse once (-> edit), then fix it
    bad = ["1", "2", "2", "1", "1", "4", "1", "", "", "", "n"]
    fixed = ["", "1", "", "", "", "", "", "", "", ""]  # Enter keeps the earlier answer
    lab = labeller(rows[:1], path, keys(bad, fixed))
    lab.run()
    assert any("兩者應一致" in line for line in lab.screen)  # type: ignore[attr-defined]
    saved = load_answers(path, rows[:1])[rows[0]["case_id"]]
    assert saved["truth_of_alert"] == "true_positive"
    assert consistency_warnings(saved) == []


def test_an_edit_that_is_then_quit_is_not_saved(tmp_path: Path) -> None:
    _, _, rows = worksheet()
    path = tmp_path / "a.jsonl"
    labeller(rows[:1], path, keys(CONSISTENT[:-1], ["e", "q"])).run()
    assert not path.exists()


def test_relabel_one_case(tmp_path: Path) -> None:
    _, _, rows = worksheet()
    path = tmp_path / "a.jsonl"
    labeller(rows, path, keys(CONSISTENT, CONSISTENT, CONSISTENT)).run()
    again = ["", "", "", "", "", "3", "2", "", "", ""]
    labeller(rows, path, keys(again)).run(only=rows[1]["case_id"])
    saved = load_answers(path, rows)
    assert saved[rows[1]["case_id"]]["answers"]["severity"] == ["中"]
    assert saved[rows[0]["case_id"]]["answers"]["severity"] == ["高", "危急"]


def test_an_answer_file_from_another_reviewer_is_refused(tmp_path: Path) -> None:
    _, _, rows = worksheet()
    path = tmp_path / "a.jsonl"
    path.write_text(json.dumps({**rows[0], "reviewer": "someone-else"}) + "\n")
    with pytest.raises(ReviewError, match="another batch or reviewer"):
        load_answers(path, rows)


def test_the_cli_runs_the_labeller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, _, rows = worksheet()
    ws = tmp_path / "ws.jsonl"
    ws.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    script = keys(CONSISTENT, ["q"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(script))
    assert cli.main(["triage", "review", "label", "--worksheet", str(ws), "--no-color"]) == 0
    assert (tmp_path / "dry-r1.answers.jsonl").exists()
    assert "事件人工標註" in capsys.readouterr().out
