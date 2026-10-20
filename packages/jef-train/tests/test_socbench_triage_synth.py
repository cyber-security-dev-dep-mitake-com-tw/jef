"""The synthetic incident-triage set: rules from LABELBOOK.md, locked by family.

These labels are true by construction, so the tests check the construction:
the approval rule, the two-truths split, translation pairs carrying the same
labels, coverage of every owner and severity level with counter-examples,
family-grouped splits, and that no case is a near copy of the training corpus.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from jef_train.socbench.cli import main
from jef_train.socbench.triage import find_root, load_pack
from jef_train.socbench.triage_synth import (
    LockError,
    build_families,
    char_ngrams,
    jaccard,
    triage_lock,
    verify_triage_lock,
)

ROOT = find_root(Path(__file__).resolve())
FAMILIES = ROOT / "synthetic" / "families-v1.yaml"
PACK = ROOT / "synthetic" / "dev-v1"


@pytest.fixture(scope="module")
def built():
    return build_families(FAMILIES)


def test_the_committed_pack_is_what_the_families_render(built) -> None:
    cases, exps = built
    pack = load_pack(PACK)
    assert pack.cases == cases
    assert [pack.expectations[c["case_id"]] for c in cases] == exps
    assert pack.manifest["evidence_class"] == "SYNTHETIC LABELS — ENGINEERING CONTRACT"


def test_rendering_is_deterministic() -> None:
    assert build_families(FAMILIES) == build_families(FAMILIES)


def test_every_label_is_synthetic_by_construction(built) -> None:
    _, exps = built
    assert {e["label_origin"] for e in exps} == {"synthetic"}
    assert {e["label_status"] for e in exps} == {"by_construction"}
    assert all(e["reviewers"] == [] for e in exps)


def test_the_approval_rule(built) -> None:
    """Benign only with the evidence in the state; without it, unknown and insufficient."""
    cases, exps = built
    for c, e in zip(cases, exps, strict=True):
        x = e["expected"]
        if x["false_positive"] is True:
            assert x["evidence_sufficiency"] is True, c["case_id"]
            assert e["truth_of_alert"] == "false_positive", c["case_id"]
            assert e["truth_of_event"] == "benign", c["case_id"]
        if x["false_positive"] == "unknown":
            assert x["evidence_sufficiency"] is False, c["case_id"]
            assert e["required_context"], f"{c['case_id']}: say what is missing"


def test_anything_not_benign_prohibits_closing(built) -> None:
    for c, e in zip(*built, strict=True):
        if e["truth_of_event"] != "benign":
            assert "close" in e["prohibited_actions"], c["case_id"]


def test_two_truths_are_kept_apart(built) -> None:
    """A real expiry is a true-positive alert about a benign event."""
    _, exps = built
    pairs = Counter((e["truth_of_event"], e["truth_of_alert"]) for e in exps)
    assert pairs[("benign", "true_positive")] > 0
    assert pairs[("benign", "false_positive")] > 0
    assert pairs[("malicious", "true_positive")] > 0


def test_translations_carry_the_same_labels_and_family(built) -> None:
    cases, exps = built
    by_id = {c["case_id"]: (c, e) for c, e in zip(cases, exps, strict=True)}
    en = [c for c in cases if c["language"] == "en"]
    assert len(en) == len(cases) // 2
    for c in en:
        src, src_exp = by_id[c["translation_of"]]
        assert src["language"] == "zh-TW"
        assert src["case_family"] == c["case_family"]
        assert src_exp["expected"] == by_id[c["case_id"]][1]["expected"]


def test_variants_stay_in_their_family(built) -> None:
    cases, _ = built
    by_id = {c["case_id"]: c for c in cases}
    for c in cases:
        for link in ("hard_negative_of", "twin_of", "translation_of"):
            if c.get(link):
                assert by_id[c[link]]["case_family"] == c["case_family"]


def test_coverage_of_the_label_space(built) -> None:
    cases, exps = built
    owners = Counter(e["expected"]["owner"][0] for e in exps if e["expected"]["owner"] != "unknown")
    assert set(owners) == {"soc", "appsec", "infra", "identity", "fraud"}
    levels = {
        s for e in exps if e["expected"]["severity"] != "unknown" for s in e["expected"]["severity"]
    }
    assert levels == {"資訊", "低", "中", "高", "危急"}
    assert sum(bool(c.get("hard_negative_of")) for c in cases) >= 20
    assert {c["domain"] for c in cases} == {"edr", "ndr", "identity", "routing", "severity"}


def test_trading_vocabulary_has_counter_examples_outside_fraud(built) -> None:
    """The Lab's fraud bias: trading words on a host must not imply fraud."""
    cases, exps = built
    trading = [
        e["expected"]["owner"]
        for c, e in zip(cases, exps, strict=True)
        if ("交易" in c["state"] or "trading" in c["state"] or "order" in c["state"])
    ]
    assert any("fraud" not in owners for owners in trading)


def test_no_case_is_a_near_copy_of_the_training_corpus(built) -> None:
    lock = json.loads((PACK / "lock.json").read_text("utf-8"))
    dups = lock["near_duplicates"]
    assert dups["to_training"] == [] and dups["across_families"] == []
    assert dups["max_jaccard_to_training"] < dups["thresholds"]["to_training"]


# --------------------------------------------------------------------------- #
# lock
# --------------------------------------------------------------------------- #


def test_the_committed_lock_matches_and_splits_by_family(built) -> None:
    cases, exps = built
    lock = json.loads((PACK / "lock.json").read_text("utf-8"))
    assert verify_triage_lock(lock, cases, exps) == []
    split_of = {cid: s for s, ids in lock["splits"].items() for cid in ids}
    for c in cases:
        assert split_of[c["case_id"]] == lock["families"][c["case_family"]]


def test_a_near_copy_of_training_is_refused(built) -> None:
    cases, exps = built
    template = cases[0]["state"]
    with pytest.raises(LockError, match="near copy of training"):
        triage_lock(cases, exps, training=[("planted", template)])


def test_a_near_duplicate_across_families_is_refused(built) -> None:
    cases, exps = built
    clone = dict(cases[0], case_id="itc-synth-v1-999", case_family="another-family")
    clone_exp = dict(exps[0], case_id="itc-synth-v1-999")
    with pytest.raises(LockError, match="across families"):
        triage_lock([*cases, clone], [*exps, clone_exp], training=[])


def test_jaccard_masks_slot_values() -> None:
    a = char_ngrams("主機 app-07.corp.example 於 02:09 連線 192.0.2.14")
    b = char_ngrams("主機 db-11.corp.example 於 23:51 連線 203.0.113.99")
    assert jaccard(a, b) == 1.0


def test_cli_lock_verifies(tmp_path: Path) -> None:
    assert main(["triage", "lock", "--pack", str(PACK), "--verify", str(PACK / "lock.json")]) == 0
    out = tmp_path / "lock.json"
    assert main(["triage", "lock", "--pack", str(PACK), "--out", str(out)]) == 0
    assert (
        json.loads(out.read_text("utf-8"))["cases_digest"]
        == json.loads((PACK / "lock.json").read_text("utf-8"))["cases_digest"]
    )


def test_synthetic_scores_say_synthetic(tmp_path: Path) -> None:
    out = tmp_path / "q"
    assert (
        main(["triage", "evaluate", "--mode", "questions", "--pack", str(PACK), "--out", str(out)])
        == 0
    )
    report = json.loads((out / "results.json").read_text("utf-8"))
    assert report["evidence_class"] == "CONTRACT SMOKE — NOT MODEL QUALITY"
    assert report["metrics"]["tasks"]["owner"]["scored"] > 0, "by_construction labels are scored"
