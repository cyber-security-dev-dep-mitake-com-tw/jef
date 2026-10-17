"""SOC metrics, baselines and the comparable report (item 09)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jef_core import Engine
from jef_train.socbench import cli
from jef_train.socbench import metrics as m
from jef_train.socbench.annotate import TASK_LABELS
from jef_train.socbench.baselines import HttpScorer, JefScorer, LogisticScorer, RuleScorer
from jef_train.socbench.records import write_jsonl
from jef_train.socbench.report import evaluate
from jef_train.socbench.splits import make_lock

ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------- #
# Metrics, against hand-computed values
# --------------------------------------------------------------------------- #


def test_average_precision() -> None:
    assert m.average_precision([0.9, 0.8, 0.7, 0.6], [True, False, True, False]) == pytest.approx(
        (1 + 2 / 3) / 2
    )
    assert m.average_precision([0.1, 0.2], [False, False]) is None


def test_ppv_at_low_prevalence_matches_the_review_example() -> None:
    """Review 11.4: TPR 90%, FPR 5%, 1% prevalence -> PPV about 15.4%."""
    assert m.ppv_at_prevalence(0.9, 0.05, 0.01) == pytest.approx(0.1538, abs=1e-4)
    assert m.ppv_at_prevalence(0.0, 0.0, 0.01) is None


def test_fnr_at_fpr_and_precision_at_budget() -> None:
    scores = [0.95, 0.9, 0.8, 0.4, 0.3, 0.2]
    truth = [True, True, False, True, False, False]
    r = m.fnr_at_fpr(scores, truth, 0.0)
    assert r["fpr"] == 0.0 and r["fnr"] == pytest.approx(1 / 3)
    assert m.fnr_at_fpr(scores, truth, 0.34)["fnr"] == 0.0
    assert m.precision_at_budget(scores, truth, 2) == 1.0
    assert m.precision_at_budget(scores, truth, 3) == pytest.approx(2 / 3)


def test_ece_brier_nll() -> None:
    assert m.ece([0.8] * 10, [True] * 8 + [False] * 2) == pytest.approx(0.0)
    assert m.ece([0.99] * 10, [True] * 5 + [False] * 5) == pytest.approx(0.49)
    p = np.array([[1.0, 0.0], [0.5, 0.5]])
    assert m.brier(p, [0, 0]) == pytest.approx((0 + 0.5) / 2)
    assert m.nll(p, [0, 1]) == pytest.approx(-np.log(0.5) / 2)


def test_risk_coverage_rewards_good_abstention() -> None:
    good = m.risk_coverage([0.9, 0.8, 0.2, 0.1], [True, True, False, False])
    bad = m.risk_coverage([0.1, 0.2, 0.8, 0.9], [True, True, False, False])
    assert good["aurc"] < bad["aurc"]
    assert good["points"][-1] == {"coverage": 1.0, "risk": 0.5}


def test_cluster_bootstrap_is_deterministic_and_brackets_the_point() -> None:
    values = np.array([1, 1, 0, 1, 0, 0, 1, 1, 1, 0], dtype=float)
    groups = ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"]
    r1 = m.cluster_bootstrap(groups, lambda i: float(values[i].mean()))
    r2 = m.cluster_bootstrap(groups, lambda i: float(values[i].mean()))
    assert r1 == r2
    assert r1["low"] <= r1["point"] <= r1["high"]
    assert r1["groups"] == 5
    assert m.cluster_bootstrap(["a", "a"], lambda i: 1.0)["low"] is None


# --------------------------------------------------------------------------- #
# Scorers
# --------------------------------------------------------------------------- #


def _case(n: int, family: str, task: str, state: dict, language: str = "en") -> dict:
    return {
        "case_id": f"socb-r-{n:06d}",
        "family_id": family,
        "source_dataset": "jef-analyst-reviewed",
        "source_hash": "sha256:" + "0" * 64,
        "telemetry_origin": "synthetic",
        "language": language,
        "task_id": task,
        "state": state,
        "evidence": [{"id": "e1", "field": k, "value": v} for k, v in list(state.items())[:3]],
    }


ALERT = "soc.alert.verdict.v1"
NDR = "soc.ndr.technique.v1"
MAL = {
    "alerts": "03:01:02 wazuh L12 Multiple authentication failures followed by success; mimikatz detected"
}
OK = {"alerts": "03:01:02 wazuh L3 ClamAV database update"}
BEACON = {"proto": "tcp", "dport": "80", "flows": 40, "interval_cv": 0.02, "states": "CON:40"}
DNS = {"proto": "udp", "dport": "53", "flows": 3, "interval_cv": 0.9, "states": "CON:3"}


@pytest.mark.parametrize("state,task", [(MAL, ALERT), (OK, ALERT), (BEACON, NDR), (DNS, NDR)])
def test_rule_scorer_returns_a_distribution_over_the_task_labels(state, task) -> None:
    probs = RuleScorer().predict(_case(1, "f", task, state))
    assert set(probs) == set(TASK_LABELS[task])
    assert sum(probs.values()) == pytest.approx(1.0)


def test_rule_scorer_orders_obvious_cases() -> None:
    r = RuleScorer()
    mal = r.predict(_case(1, "f", ALERT, MAL))
    ok = r.predict(_case(2, "f", ALERT, OK))
    assert (
        mal["confirmed_malicious"] + mal["suspicious"]
        > ok["confirmed_malicious"] + ok["suspicious"]
    )
    assert max(r.predict(_case(3, "f", NDR, BEACON)).items(), key=lambda kv: kv[1])[0] == "c2"


def test_jef_and_http_scorers_speak_the_same_shape() -> None:
    jef = JefScorer(Engine()).predict(_case(1, "f", ALERT, MAL))
    assert set(jef) == set(TASK_LABELS[ALERT]) and sum(jef.values()) == pytest.approx(1.0, abs=1e-3)
    seen = []

    def post(url: str, body: dict) -> dict:
        seen.append((url, body))
        labels = body["questions"]["verdict"]["criteria"]
        return {"answers": {"verdict": {"probabilities": {k: 1 / len(labels) for k in labels}}}}

    http = HttpScorer("http://127.0.0.1:4141", post=post).predict(_case(1, "f", ALERT, MAL))
    assert seen[0][0] == "http://127.0.0.1:4141/v1/systemone"
    assert sum(http.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="https or loopback"):
        HttpScorer("http://example.com")._default_post("http://example.com/v1/systemone", {})


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def _dataset():
    cases, golds = [], []
    n = 0
    for fam in range(8):
        for k in range(6):
            n += 1
            malicious = k % 3 == 0
            unknown = k == 5
            state = MAL if malicious else OK
            if unknown:
                state = {"alerts": f"03:0{k}:00 wazuh L5 unclassified event {fam}"}
            c = _case(
                n,
                f"fam-{fam}",
                ALERT,
                {**state, "host": f"h{fam}-{k}"},
                language="zh-TW" if fam % 2 else "en",
            )
            cases.append(c)
            verdict = (
                "unknown"
                if unknown
                else ("confirmed_malicious" if malicious else "confirmed_benign")
            )
            golds.append(
                {
                    "case_id": c["case_id"],
                    "task_id": ALERT,
                    "label_origin": "double-human-reviewed",
                    "annotators": ["analyst-anon-01", "analyst-anon-02"],
                    "adjudication_status": "agreed",
                    "gold": {"verdict": verdict, "confidence_of_gold": "high"},
                }
            )
    pins = {f"fam-{f}": ("test" if f >= 5 else "train") for f in range(8)}
    return cases, golds, make_lock(cases, golds, pinned=pins)


def test_logreg_fits_on_train_only() -> None:
    cases, golds, lock = _dataset()
    train = [c for c in cases if c["case_id"] in set(lock["splits"]["train"])]
    scorer = LogisticScorer().fit(train, {g["case_id"]: g for g in golds})
    assert scorer.trained_on == set(lock["splits"]["train"])
    assert scorer.trained_on.isdisjoint(lock["splits"]["test"])


def test_evaluate_produces_comparable_blocks_for_every_scorer() -> None:
    cases, golds, lock = _dataset()
    train = [c for c in cases if c["case_id"] in set(lock["splits"]["train"])]
    scorers = [
        RuleScorer(),
        LogisticScorer().fit(train, {g["case_id"]: g for g in golds}),
        JefScorer(Engine()),
    ]
    report = evaluate(cases, golds, lock, scorers)
    assert report["lock"]["n"] == len(lock["splits"]["test"])
    assert set(report["scorers"]) == {"rules", "logreg", "jef"}
    block = report["scorers"]["rules"]["tasks"][ALERT]
    for key in (
        "accuracy",
        "macro_f1",
        "accuracy_ci",
        "macro_f1_ci",
        "ece",
        "brier",
        "nll",
        "risk_coverage",
        "confusion",
    ):
        assert key in block
    binary = block["binary"]
    assert binary["n_known"] == block["n"] - block["gold_distribution"]["unknown"]
    assert set(binary["ppv_at_prevalence"]) == {"0.01", "0.03", "0.05"}
    assert "pr_auc_ci" in binary and "fnr_at_fpr_5pct" in binary
    assert block["unknown_handling"]["undetermined_gold"] == block["gold_distribution"]["unknown"]
    assert set(block["by_language"]) == {"en", "zh-TW"}
    assert set(block["by_tier"]) == {"A"}
    assert {r["scorer"] for r in report["comparison"]} == {"rules", "logreg", "jef"}
    assert block["binary"]["pr_auc"] > 0.9, "the rules separate this toy set"


def test_evaluate_refuses_files_that_are_not_the_locked_ones() -> None:
    cases, golds, lock = _dataset()
    golds[0] = {**golds[0], "gold": {"verdict": "confirmed_benign", "confidence_of_gold": "high"}}
    with pytest.raises(ValueError, match="not the locked ones"):
        evaluate(cases, golds, lock, [RuleScorer()])


def test_cli_evaluate(tmp_path: Path, capsys) -> None:
    cases, golds, lock = _dataset()
    write_jsonl(tmp_path / "c.jsonl", cases)
    write_jsonl(tmp_path / "g.jsonl", golds)
    (tmp_path / "lock.json").write_text(json.dumps(lock))
    rc = cli.main(
        [
            "--root",
            str(ROOT / "benchmark" / "socbench"),
            "evaluate",
            "--lock",
            str(tmp_path / "lock.json"),
            "--cases",
            str(tmp_path / "c.jsonl"),
            "--gold",
            str(tmp_path / "g.jsonl"),
            "--scorer",
            "rules",
            "--scorer",
            "logreg",
            "--scorer",
            "jef",
            "--out",
            str(tmp_path / "r.json"),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "rules" in out and "logreg" in out and "jef" in out
    assert json.loads((tmp_path / "r.json").read_text())["lock"]["split"] == "test"
