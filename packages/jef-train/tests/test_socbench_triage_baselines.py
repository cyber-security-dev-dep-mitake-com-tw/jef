"""Incident-triage baselines: same cases, same harness, a pre-registered decision."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jef_core import Engine
from jef_core.types import ChoiceAnswer, NoulAnswer, ScoreAnswer
from jef_scene import load_scene
from jef_train.socbench.cli import main
from jef_train.socbench.triage import find_root, load_pack
from jef_train.socbench.triage_baselines import (
    LogisticTriageEngine,
    RuleTriageEngine,
    answer_from,
    fail_closed_summary,
    routing_decision,
    run_baselines,
)
from jef_train.socbench.triage_eval import EvalError, scene_questions

ROOT = find_root(Path(__file__).resolve())
REPO = ROOT.parents[1]
SCENE = load_scene(REPO / "scenes" / "incident-triage.zh-tw.yaml")
QS = scene_questions(SCENE)
DEV = ROOT / "synthetic" / "dev-v1"


def ask(engine: Engine, text: str) -> dict:
    return engine.answer(engine.prepare(text), QS)


def test_answer_from_builds_each_kind() -> None:
    assert isinstance(answer_from(QS["urgent"], "urgent", {"true": 0.8}), NoulAnswer)
    owner = answer_from(QS["owner"], "owner", {"infra": 3, "soc": 1})
    assert isinstance(owner, ChoiceAnswer) and owner.choice == "infra"
    assert abs(sum(owner.probabilities.values()) - 1) < 1e-4
    sev = answer_from(QS["severity"], "severity", {"危急": 1.0})
    assert (
        isinstance(sev, ScoreAnswer) and max(sev.probabilities, key=sev.probabilities.get) == "危急"
    )


def test_rules_follow_the_approval_rule() -> None:
    rules = RuleTriageEngine()
    approved = ask(
        rules, "IDS 偵測 198.51.100.23 掃描；為登記的掃描器，已核准（工單 CHG-2026-1234）。"
    )
    assert approved["is_false_positive"].value > 0.5
    unexplained = ask(rules, "IDS 偵測 198.51.100.23 掃描。告警未說明是否為核准的掃描器。")
    assert unexplained["is_false_positive"].value < 0.5
    assert unexplained["has_evidence"].value < 0.5


def test_rules_do_not_route_trading_hosts_to_fraud() -> None:
    rules = RuleTriageEngine()
    malware = ask(rules, "EDR 在下單前台伺服器 web-02 偵測到 QakBot 木馬。")
    assert malware["owner"].choice == "soc"
    ato = ask(rules, "風控系統標記客戶帳戶：更換綁定手機後申請將全部餘額轉出至新約定帳戶。")
    assert ato["owner"].choice == "fraud"


def test_logreg_trains_on_the_train_split_only_and_refuses_overlap() -> None:
    pack = load_pack(DEV)
    lock = json.loads((DEV / "lock.json").read_text("utf-8"))
    train = [c for c in pack.cases if c["case_id"] in set(lock["splits"]["train"])]
    test = [c for c in pack.cases if c["case_id"] in set(lock["splits"]["test"])]
    model = LogisticTriageEngine(train, pack.expectations)
    model.check_no_leak(test)
    with pytest.raises(EvalError, match="families in both"):
        model.check_no_leak(train[:1])


def test_fail_closed_is_the_floor() -> None:
    s = fail_closed_summary([{"case_id": "a"}, {"case_id": "b"}])
    assert s["review_required"] == 2 and s["auto_allowed"] == 0 and s["workload_share"] == 1.0


@pytest.mark.parametrize(
    ("rules", "jef", "routing"),
    [
        (0.8, 0.2, "human_or_rule_assisted"),
        (0.5, 0.5, "may_proceed_to_human_labelled_test"),
        (0.4, 0.6, "may_proceed_to_human_labelled_test"),
    ],
)
def test_routing_decision_is_the_pre_registered_rule(
    rules: float, jef: float, routing: str
) -> None:
    by = {
        "rules": {"tasks": {"owner": {"macro_f1": rules}}},
        "jef": {"tasks": {"owner": {"macro_f1": jef}}},
    }
    assert routing_decision(by)["routing"] == routing
    assert routing_decision({"rules": by["rules"]})["decided"] is False


def test_baselines_on_the_test_split_with_a_hashing_b3(tmp_path: Path) -> None:
    pack = load_pack(DEV)
    lock = json.loads((DEV / "lock.json").read_text("utf-8"))
    report = run_baselines(pack, lock, SCENE, jef=Engine())
    assert report["cases"] == len(lock["splits"]["test"])
    assert set(report["baselines"]) == {"rules", "logreg", "jef"}
    assert report["routing_decision"]["decided"] is True
    assert report["fail_closed"]["review_required"] == report["cases"]


def test_cli_baselines(tmp_path: Path) -> None:
    out = tmp_path / "b"
    rc = main(
        [
            "triage", "baselines", "--pack", str(DEV), "--lock", str(DEV / "lock.json"),
            "--backbone", "hashing", "--out", str(out),
        ]
    )  # fmt: skip
    assert rc == 0
    report = json.loads((out / "results.json").read_text("utf-8"))
    assert report["evidence_class"] == "CONTRACT SMOKE — NOT MODEL QUALITY"
    assert "Routing decision (pre-registered)" in (out / "summary.md").read_text("utf-8")


def test_cli_refuses_a_lock_that_does_not_match(tmp_path: Path) -> None:
    lock = json.loads((DEV / "lock.json").read_text("utf-8"))
    lock["cases_digest"] = "sha256:" + "0" * 64
    bad = tmp_path / "lock.json"
    bad.write_text(json.dumps(lock), "utf-8")
    rc = main(
        [
            "triage",
            "baselines",
            "--pack",
            str(DEV),
            "--lock",
            str(bad),
            "--out",
            str(tmp_path / "o"),
        ]
    )
    assert rc == 2


def test_logreg_constant_false_noul_predicts_false() -> None:
    """A train split with only `false` for a noul task must not predict True."""
    pack = load_pack(DEV)
    lock = json.loads((DEV / "lock.json").read_text("utf-8"))
    train = [c for c in pack.cases if c["case_id"] in set(lock["splits"]["train"])]
    exps = {
        cid: dict(e, expected=dict(e["expected"], urgency=False))
        for cid, e in pack.expectations.items()
    }
    model = LogisticTriageEngine(train, exps)
    assert model.constant["urgency"] == "false"
    assert ask(model, "任何告警")["urgent"].value == 0.0


def test_logreg_ignores_unscorable_labels() -> None:
    pack = load_pack(DEV)
    lock = json.loads((DEV / "lock.json").read_text("utf-8"))
    train = [c for c in pack.cases if c["case_id"] in set(lock["splits"]["train"])]
    exps = {cid: dict(e, label_status="disputed") for cid, e in pack.expectations.items()}
    with pytest.raises(EvalError, match="no training labels"):
        LogisticTriageEngine(train, exps)
