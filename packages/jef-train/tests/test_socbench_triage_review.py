"""The reviewer queue: who sees what, what comes back, and what tier it earns.

Packs here are copies of the synthetic dev set relabelled as reviewer-sourced
so the queue treats them as real; the answers are written by the tests. None
of this is a label of anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jef_scene import load_scene
from jef_train.socbench import cli
from jef_train.socbench.triage import Pack, load_pack, pack_manifest
from jef_train.socbench.triage_review import (
    DEFAULT_STRATA,
    ReviewError,
    check_review,
    cross_check_sample,
    export_worksheet,
    import_answers,
    sample_batch,
    worksheet_html,
    worksheet_questions,
    write_reviewed_pack,
)

ROOT = Path(__file__).resolve().parents[3]
DEV = ROOT / "benchmark" / "incident-triage" / "synthetic" / "dev-v1"
SCENE = load_scene(ROOT / "scenes" / "incident-triage.zh-tw.yaml")
FORBIDDEN_KEYS = {"expected", "label_origin", "label_status", "truth_of_event_gold", "observed",
                  "probability", "p_correct", "weak", "detector", "actions", "rules"}  # fmt: skip


def as_real(tmp_path: Path, name: str = "real") -> Pack:
    """The dev pack's cases as reviewer-sourced, with no labels carried over."""
    dev = load_pack(DEV)
    out = tmp_path / name
    out.mkdir()
    cases = [{**c, "origin": "reviewer-sourced"} for c in dev.cases]
    exps = [
        {**dev.expectations[c["case_id"]], "label_origin": "single-human-reviewed",
         "label_status": "pending_owner_confirmation", "reviewers": []}
        for c in dev.cases
    ]  # fmt: skip
    (out / "cases.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases)
    )
    (out / "expectations.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in exps)
    )
    (out / "manifest.json").write_text(json.dumps(pack_manifest(out)))
    return load_pack(out)


def answer(row: dict[str, Any], **override: Any) -> dict[str, Any]:
    answers = {
        "is_false_positive": False,
        "has_evidence": True,
        "owner": ["soc"],
        "severity": ["高", "危急"],
        "urgent": True,
    }
    answers.update(override)
    return {
        **row,
        "answers": answers,
        "truth_of_event": "malicious",
        "truth_of_alert": "true_positive",
        "minutes_spent": 3,
    }


def answered(rows: list[dict[str, Any]], **override: Any) -> list[dict[str, Any]]:
    return [answer(r, **override) for r in rows]


@pytest.fixture
def real(tmp_path: Path) -> Pack:
    return as_real(tmp_path)


@pytest.fixture
def batch(real: Pack) -> dict[str, Any]:
    return sample_batch([real], batch_id="b1", strata={"edr": 4, "identity": 3})


# --------------------------------------------------------------------------- #
# sample
# --------------------------------------------------------------------------- #


def test_the_default_strata_are_the_v5_pilot() -> None:
    assert sum(DEFAULT_STRATA.values()) == 100


def test_a_short_stratum_is_reported_never_filled(real: Pack) -> None:
    b = sample_batch([real], batch_id="b", strata={"edr": 500, "web": 25})
    assert b["shortfall"]["web"] == 25  # no web cases exist at all
    assert 0 < b["strata_selected"]["edr"] < 500 and b["shortfall"]["edr"] > 0
    assert b["shortfall_note"].startswith("left short on purpose")


def test_synthetic_cases_are_not_eligible_unless_asked() -> None:
    dev = load_pack(DEV)
    b = sample_batch([dev], batch_id="b", strata={"edr": 4})
    assert b["case_ids"] == [] and b["skipped"]["synthetic"] > 0 and not b["dry_run"]
    dry = sample_batch([dev], batch_id="b", strata={"edr": 4}, allow_synthetic=True)
    assert len(dry["case_ids"]) == 4 and dry["dry_run"]


def test_family_cap_and_twins_follow(real: Pack) -> None:
    b = sample_batch([real], batch_id="b", strata={"edr": 30, "identity": 30}, max_per_family=1)
    fams = [
        real.case(cid)["case_family"] for cid in b["case_ids"] if not real.case(cid).get("twin_of")
    ]
    assert len(fams) == len(set(fams))
    for cid in b["case_ids"]:
        twin_of = real.case(cid).get("twin_of")
        assert twin_of is None or twin_of in b["case_ids"]


def test_sampling_is_deterministic_and_text_free(real: Pack) -> None:
    a = sample_batch([real], batch_id="b", strata={"edr": 5})
    b = sample_batch([real], batch_id="b", strata={"edr": 5})
    assert a["case_ids"] == b["case_ids"]
    assert "state" not in json.dumps(a)


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #


def test_questions_come_from_the_scene_with_unknown_always_offered() -> None:
    qs = worksheet_questions(SCENE)
    assert list(qs) == ["is_false_positive", "has_evidence", "owner", "severity", "urgent"]
    assert all("unknown" in q["options"] for q in qs.values())
    assert qs["owner"]["options"][:5] == ["soc", "appsec", "infra", "identity", "fraud"]
    assert qs["is_false_positive"]["meanings"]["true"].startswith("為既有排程")


def test_a_worksheet_carries_no_label_score_or_detector(real: Pack, batch: dict[str, Any]) -> None:
    rows = export_worksheet([real], batch, SCENE, reviewer="analyst-a")

    def keys(x: Any) -> set[str]:
        if isinstance(x, dict):
            return set(x) | {k for v in x.values() for k in keys(v)}
        if isinstance(x, list):
            return {k for v in x for k in keys(v)}
        return set()

    for r in rows:
        assert not keys({k: v for k, v in r.items() if k != "questions"}) & FORBIDDEN_KEYS
        assert all(v is None for v in r["answers"].values())


def test_detector_evidence_is_hidden_and_vendor_variants_refused(
    real: Pack, batch: dict[str, Any]
) -> None:
    cid = batch["case_ids"][0]
    case = real.case(cid)
    case["evidence"].append({"id": "e99", "field": "detector_output", "value": "actions block×3"})
    row = export_worksheet([real], batch, SCENE, reviewer="a", case_ids=[cid])[0]
    assert all(e["field"] != "detector_output" for e in row["evidence"])
    case["vendor_signals"] = True
    with pytest.raises(ReviewError, match="detector's own output"):
        export_worksheet([real], batch, SCENE, reviewer="a", case_ids=[cid])


def test_each_reviewer_gets_their_own_order(real: Pack, batch: dict[str, Any]) -> None:
    a = [r["case_id"] for r in export_worksheet([real], batch, SCENE, reviewer="a")]
    b = [r["case_id"] for r in export_worksheet([real], batch, SCENE, reviewer="b")]
    assert sorted(a) == sorted(b) and a != b


def test_the_html_form_is_self_contained(real: Pack, batch: dict[str, Any]) -> None:
    page = worksheet_html(export_worksheet([real], batch, SCENE, reviewer="a"))
    # The case data (which may quote URLs from an alert) sits on one line;
    # the rest of the page must load nothing from anywhere.
    shell = "\n".join(line for line in page.splitlines() if not line.startswith("const rows = "))
    assert "<script src" not in shell and "<link" not in shell
    assert "http://" not in shell and "https://" not in shell
    assert "</script>" not in page.split("const rows = ", 1)[1].split("\n", 1)[0]
    assert "prefers-color-scheme" in page


# --------------------------------------------------------------------------- #
# import
# --------------------------------------------------------------------------- #


def test_one_reviewer_gives_tier_b(real: Pack, batch: dict[str, Any]) -> None:
    rows = answered(export_worksheet([real], batch, SCENE, reviewer="analyst-a"))
    cases, exps, review = import_answers([real], batch, SCENE, [rows])
    assert {e["label_origin"] for e in exps} == {"single-human-reviewed"}
    assert {e["label_status"] for e in exps} == {"confirmed"}
    assert exps[0]["expected"]["owner"] == ["soc"] and exps[0]["reviewers"] == ["analyst-a"]
    assert review["reviewers"] == ["analyst-a"] and len(cases) == len(batch["case_ids"])


def test_a_missing_task_or_an_unoffered_option_fails(real: Pack, batch: dict[str, Any]) -> None:
    rows = answered(export_worksheet([real], batch, SCENE, reviewer="a"))
    rows[0]["answers"]["urgent"] = None
    with pytest.raises(ReviewError, match="urgent is not answered"):
        import_answers([real], batch, SCENE, [rows])
    rows = answered(export_worksheet([real], batch, SCENE, reviewer="a"), owner=["helpdesk"])
    with pytest.raises(ReviewError, match="does not offer"):
        import_answers([real], batch, SCENE, [rows])
    rows = answered(export_worksheet([real], batch, SCENE, reviewer="a"))
    rows[1]["truth_of_alert"] = "maybe"
    with pytest.raises(ReviewError, match="truth_of_alert"):
        import_answers([real], batch, SCENE, [rows])


def test_an_unanswered_case_fails(real: Pack, batch: dict[str, Any]) -> None:
    rows = answered(export_worksheet([real], batch, SCENE, reviewer="a"))[1:]
    with pytest.raises(ReviewError, match="did not answer"):
        import_answers([real], batch, SCENE, [rows])


def test_the_same_handle_twice_is_not_a_second_review(real: Pack, batch: dict[str, Any]) -> None:
    rows = answered(export_worksheet([real], batch, SCENE, reviewer="a"))
    with pytest.raises(ReviewError, match="answered twice"):
        import_answers([real], batch, SCENE, [rows, rows])


def test_two_reviewers_agreeing_give_double_review(real: Pack, batch: dict[str, Any]) -> None:
    a = answered(export_worksheet([real], batch, SCENE, reviewer="a"))
    b = answered(export_worksheet([real], batch, SCENE, reviewer="b"))
    _, exps, _ = import_answers([real], batch, SCENE, [a, b])
    assert {(e["label_origin"], e["label_status"]) for e in exps} == {
        ("double-human-reviewed", "confirmed")
    }


def test_disagreement_is_disputed_until_a_third_person_decides(
    real: Pack, batch: dict[str, Any]
) -> None:
    a = answered(export_worksheet([real], batch, SCENE, reviewer="a"))
    b = answered(export_worksheet([real], batch, SCENE, reviewer="b"))
    target = b[0]["case_id"]
    b[0]["answers"]["owner"] = ["infra"]
    _, exps, _ = import_answers([real], batch, SCENE, [a, b])
    disputed = next(e for e in exps if e["case_id"] == target)
    assert disputed["label_status"] == "disputed"
    assert set(disputed["expected"].values()) == {"unknown"}  # nobody's answer stands in
    with pytest.raises(ReviewError, match="third person"):
        import_answers(
            [real], batch, SCENE, [a, b], adjudications=[{**answer(b[0]), "adjudicator": "a"}]
        )
    decision = {
        **answer(a[0] if a[0]["case_id"] == target else b[0]),
        "case_id": target,
        "adjudicator": "c",
    }
    _, exps, review = import_answers([real], batch, SCENE, [a, b], adjudications=[decision])
    decided = next(e for e in exps if e["case_id"] == target)
    assert decided["label_status"] == "confirmed" and decided["reviewers"] == ["a", "b", "c"]
    assert review["adjudicators"] == ["c"]


def test_vendor_variants_copy_labels_but_twins_do_not(real: Pack, tmp_path: Path) -> None:
    base = next(c for c in real.cases if c.get("domain") == "edr" and not c.get("twin_of"))
    variant = {**base, "case_id": base["case_id"].rsplit("-", 1)[0] + "-999", "variant_of": base["case_id"],
               "vendor_signals": True}  # fmt: skip
    real.cases.append(variant)
    b = {"batch_id": "bv", "case_ids": [base["case_id"]], "packs": {}}
    rows = answered(export_worksheet([real], b, SCENE, reviewer="a"))
    _, exps, _ = import_answers([real], b, SCENE, [rows])
    by = {e["case_id"]: e for e in exps}
    assert by[variant["case_id"]]["expected"] == by[base["case_id"]]["expected"]
    report = check_review(exps)
    assert report["cases"] == 1 and report["copied_to_vendor_variants"] == 1
    twin = next(c for c in real.cases if c.get("twin_of"))
    bt = {"batch_id": "bt", "case_ids": [twin["twin_of"]], "packs": {}}
    rows = answered(export_worksheet([real], bt, SCENE, reviewer="a"))
    _, exps, _ = import_answers([real], bt, SCENE, [rows])
    assert twin["case_id"] not in {e["case_id"] for e in exps}


def test_the_reviewed_pack_loads_and_private_origins_stay_out_of_the_checkout(
    real: Pack, batch: dict[str, Any], tmp_path: Path
) -> None:
    rows = answered(export_worksheet([real], batch, SCENE, reviewer="a"))
    out = write_reviewed_pack(tmp_path / "reviewed", *import_answers([real], batch, SCENE, [rows]))
    pack = load_pack(out)
    assert all(p.stat().st_mode & 0o077 == 0 for p in out.iterdir())
    assert len(pack.cases) == len(batch["case_ids"])
    with pytest.raises(ReviewError, match="inside the git checkout"):
        write_reviewed_pack(
            ROOT / "out" / "reviewed", *import_answers([real], batch, SCENE, [rows])
        )


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #


def test_check_reports_tiers_unknowns_kappa_and_time(real: Pack, batch: dict[str, Any]) -> None:
    a = answered(export_worksheet([real], batch, SCENE, reviewer="a"), severity="unknown")
    b = answered(export_worksheet([real], batch, SCENE, reviewer="b"), severity="unknown")
    b[0]["answers"]["owner"] = ["infra"]
    _, exps, _ = import_answers([real], batch, SCENE, [a, b])
    report = check_review(exps, [a, b])
    assert report["evidence_class"].startswith("TIER A CANDIDATE")
    # Everyone said unknown for severity; the disputed case is all-unknown too.
    assert report["unknown_per_task"]["severity"] == len(exps)
    assert report["kappa_per_task"]["owner"] is not None
    assert report["minutes_spent"]["total"] == 3 * 2 * len(exps)


def test_cross_check_is_a_fifth_of_the_batch(batch: dict[str, Any]) -> None:
    sample = cross_check_sample(batch)
    assert 1 <= len(sample) <= max(1, round(len(batch["case_ids"]) * 0.2))
    assert set(sample) <= set(batch["case_ids"])


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #


def test_cli_round_trip(real: Pack, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pack, batch_path, ws = str(real.path), tmp_path / "batch.json", tmp_path / "ws.jsonl"
    assert cli.main(["triage", "review", "sample", "--pack", pack, "--batch-id", "b1",
                     "--stratum", "edr=3", "--stratum", "web=2", "--out", str(batch_path)]) == 0  # fmt: skip
    assert "shortfall {'web': 2}" in capsys.readouterr().out
    assert cli.main(["triage", "review", "export", "--pack", pack, "--batch", str(batch_path),
                     "--reviewer", "a", "--out", str(ws), "--html", str(tmp_path / "ws.html")]) == 0  # fmt: skip
    rows = [answer(json.loads(x)) for x in ws.read_text().splitlines()]
    ans = tmp_path / "a.jsonl"
    ans.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    out = tmp_path / "reviewed"
    assert cli.main(["triage", "review", "import", "--pack", pack, "--batch", str(batch_path),
                     "--answers", str(ans), "--out", str(out)]) == 0  # fmt: skip
    capsys.readouterr()
    assert cli.main(["triage", "review", "check", "--pack", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["evidence_class"] == "TIER B — SINGLE REVIEWER"
