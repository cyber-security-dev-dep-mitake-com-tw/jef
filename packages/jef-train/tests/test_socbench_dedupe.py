"""dedupe_states: identical states cannot reach two splits, and no label is chosen arbitrarily."""

from __future__ import annotations

from jef_train.socbench.splits import check_leakage, dedupe_states, make_lock

TASK = "soc.ndr.technique.v1"


def case(cid: str, family: str, state: dict) -> dict:
    return {
        "case_id": cid,
        "family_id": family,
        "source_dataset": "ctu-13",
        "source_hash": "sha256:" + "0" * 64,
        "telemetry_origin": "lab-reproduction",
        "language": "n/a",
        "task_id": TASK,
        "state": state,
        "evidence": [{"id": "e1", "field": "f", "value": "v"}],
    }


def gold(cid: str, verdict: str) -> dict:
    return {
        "case_id": cid,
        "task_id": TASK,
        "label_origin": "flow-label",
        "gold": {"verdict": verdict, "confidence_of_gold": "medium"},
    }


def test_same_label_duplicates_keep_one_and_conflicts_drop_all() -> None:
    icmp = {"proto": "icmp", "bytes": 135}
    odd = {"proto": "udp", "bytes": 60}
    cases = [
        case("socb-ctu13-000003", "s2", icmp),
        case("socb-ctu13-000001", "s1", icmp),
        case("socb-ctu13-000005", "s1", odd),
        case("socb-ctu13-000007", "s3", odd),
        case("socb-ctu13-000009", "s3", {"proto": "tcp"}),
    ]
    golds = [
        gold("socb-ctu13-000003", "benign"),
        gold("socb-ctu13-000001", "benign"),
        gold("socb-ctu13-000005", "benign"),
        gold("socb-ctu13-000007", "botnet"),
        gold("socb-ctu13-000009", "c2"),
    ]
    kept, kept_gold, report = dedupe_states(cases, golds)
    assert [c["case_id"] for c in kept] == ["socb-ctu13-000001", "socb-ctu13-000009"]
    assert report["dropped_same_label"] == ["socb-ctu13-000003"]
    assert report["dropped_conflicting_labels"] == ["socb-ctu13-000005", "socb-ctu13-000007"]
    assert {g["case_id"] for g in kept_gold} == {c["case_id"] for c in kept}
    assert check_leakage(kept, {"s1": "train", "s2": "test", "s3": "test"}) == []


def test_a_lock_that_failed_on_duplicates_succeeds_after_dedupe() -> None:
    shared = {"proto": "icmp"}
    cases = [
        case(f"socb-ctu13-{i:06d}", f"s{i}", shared if i < 3 else {"n": i}) for i in range(1, 9)
    ]
    golds = [gold(c["case_id"], "benign") for c in cases]
    pins = {"s1": "train", "s2": "test"}
    try:
        make_lock(cases, golds, seed="t", pinned=pins)
        raised = False
    except ValueError:
        raised = True
    assert raised, "identical state across splits must be refused"
    kept, kept_gold, _ = dedupe_states(cases, golds)
    assert make_lock(kept, kept_gold, seed="t", pinned=pins)["cases_digest"]
