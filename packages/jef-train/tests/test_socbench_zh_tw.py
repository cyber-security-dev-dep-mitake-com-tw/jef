"""The zh-TW seed set and paired cases (item 10)."""

from __future__ import annotations

from pathlib import Path

import pytest
from jef_train.socbench import SocbenchRepo
from jef_train.socbench.bilingual import check_pairs, pair_translation
from jef_train.socbench.records import read_jsonl, trust_tier
from jef_train.socbench.splits import make_lock

ROOT = Path(__file__).resolve().parents[3]
SEEDS = ROOT / "benchmark" / "socbench" / "seeds" / "zh-TW"

#: Characters that exist only in Simplified Chinese. One of them in a zh-TW
#: case means it was written in, or converted from, the wrong script.
SIMPLIFIED_ONLY = set(
    "们这国说时发对实现么为后门问题观应让讲认识话们个这还进过动长东车马鸟龙书买卖钱银网络帐号码验证务账户"
)


@pytest.fixture(scope="module")
def seeds() -> tuple[list[dict], list[dict]]:
    return read_jsonl(SEEDS / "cases.jsonl"), read_jsonl(SEEDS / "gold.jsonl")


def test_seeds_validate_against_the_schemas(seeds) -> None:
    repo = SocbenchRepo(ROOT / "benchmark" / "socbench")
    cases, golds = seeds
    for c in cases:
        repo.validate("case", c, where=c["case_id"])
    for g in golds:
        repo.validate("gold", g, where=g["case_id"])
    assert {c["case_id"] for c in cases} == {g["case_id"] for g in golds}


def test_seeds_are_honestly_labelled(seeds) -> None:
    cases, golds = seeds
    assert {c["telemetry_origin"] for c in cases} == {"synthetic"}
    assert {g["label_origin"] for g in golds} == {"llm-authored"}
    assert {trust_tier(g["label_origin"]) for g in golds} == {"E"}
    assert {g["gold"]["confidence_of_gold"] for g in golds} == {"low"}
    assert all(
        "Awaiting two-reviewer review" in g["adjudication_note"]
        or "translation" in g["adjudication_note"]
        for g in golds
    )


def test_native_cases_are_traditional_chinese(seeds) -> None:
    cases, _ = seeds
    native = [c for c in cases if c["language"] == "zh-TW"]
    assert len(native) == 24
    for c in native:
        text = "".join(str(v) for v in c["state"].values())
        found = SIMPLIFIED_ONLY & set(text)
        assert not found, f"{c['case_id']} contains Simplified-only characters {sorted(found)}"


def test_no_real_domains_and_no_live_links(seeds) -> None:
    cases, _ = seeds
    for c in cases:
        for v in c["state"].values():
            s = str(v)
            assert "http://" not in s and "https://" not in s, f"{c['case_id']}: live link"
            for token in s.replace("[.]", ".").split():
                if "@" in token or token.startswith("hxxp"):
                    assert ".invalid" in token or ".example" in token, f"{c['case_id']}: {token}"


def test_every_pair_is_consistent_and_twelve_families_hold_them(seeds) -> None:
    cases, golds = seeds
    assert check_pairs(cases, golds) == []
    assert len({c["family_id"] for c in cases}) == 12
    kinds = [c.get("pair_kind") for c in cases]
    assert kinds.count("hard-negative") == 24 and kinds.count("translation") == 4
    negatives = [c for c in cases if c["hard_negative"]]
    gold_by = {g["case_id"]: g for g in golds}
    assert {gold_by[c["case_id"]]["gold"]["verdict"] for c in negatives} == {
        "legitimate",
        "confirmed_benign",
    }


def test_data_003_the_lock_keeps_twins_and_translations_together(seeds) -> None:
    cases, golds = seeds
    lock = make_lock(cases, golds, seed="zh")
    split_of = {cid: s for s, ids in lock["splits"].items() for cid in ids}
    for c in cases:
        if c.get("paired_with"):
            assert split_of[c["case_id"]] == split_of[c["paired_with"]]
    assert set(lock["counts"][next(iter(lock["counts"]))]["language"]) <= {"zh-TW", "en"}


def test_pair_translation_carries_the_verdict_and_the_family(seeds) -> None:
    cases, golds = seeds
    src = next(
        c for c in cases if c["family_id"] == "zhtw-seed-parcel-fee" and not c["hard_negative"]
    )
    src_gold = next(g for g in golds if g["case_id"] == src["case_id"])
    en_state = {
        "channel": "sms",
        "from": "+886 9xx-xxx-073",
        "subject": "",
        "body": "[Delivery] Your parcel could not be delivered because the address is incomplete. Pay NT$25 today and update the address.",
        "url": "hxxps://tw-parcel-fee[.]invalid/pay",
    }
    twin, twin_gold = pair_translation(
        src,
        src_gold,
        state=en_state,
        new_case_id="socb-zhtw-900001",
        language="en",
        translator="analyst-anon-05",
    )
    assert twin["family_id"] == src["family_id"] and twin["paired_with"] == src["case_id"]
    assert twin["pair_kind"] == "translation" and twin["language"] == "en"
    assert twin_gold["gold"] == src_gold["gold"]
    assert check_pairs([*cases, twin], [*golds, twin_gold]) == []
    with pytest.raises(ValueError, match="same fields"):
        pair_translation(
            src,
            src_gold,
            state={"body": "x"},
            new_case_id="socb-zhtw-900002",
            language="en",
            translator="t",
        )
    with pytest.raises(ValueError, match="already"):
        pair_translation(
            src,
            src_gold,
            state=src["state"],
            new_case_id="socb-zhtw-900003",
            language="zh-TW",
            translator="t",
        )


def test_check_pairs_finds_broken_pairs(seeds) -> None:
    cases, golds = seeds
    broken = [dict(c) for c in cases]
    broken[0]["family_id"] = "zhtw-seed-elsewhere"
    broken[2] = {**broken[2], "pair_kind": None}
    problems = check_pairs(broken, golds)
    assert any("different families" in p for p in problems)
    assert any("no pair_kind" in p for p in problems)
