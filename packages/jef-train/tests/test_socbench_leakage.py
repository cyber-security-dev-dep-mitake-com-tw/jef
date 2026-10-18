"""Leakage audit: host reuse, near-duplicates across scenarios, window label basis."""

from __future__ import annotations

from typing import Any

from jef_train.socbench.leakage import (
    audit,
    host_of,
    host_reuse,
    near_duplicates,
    window_label_basis,
)

TASK = "soc.ndr.host.v1"


def case(cid: str, fam: str, flows: int, host: str, verdict: str, labels: str) -> tuple[dict, dict]:
    c: dict[str, Any] = {
        "case_id": cid,
        "family_id": fam,
        "task_id": TASK,
        "state": {"flows": flows, "protocols": "udp:1"},
    }
    g = {
        "case_id": cid,
        "gold": {"verdict": verdict},
        "adjudication_note": f"host {host}, scenario {fam[-1]}, window 1 (300s); host labels in window: {labels}; flows: e1=x#1",
    }
    return c, g


def data() -> tuple[list[dict], list[dict]]:
    rows = [
        case("c1", "s-1", 10, "10.0.0.1", "c2", "{'botnet': 4, 'c2': 1}"),
        case(
            "c2", "s-2", 11, "10.0.0.1", "c2", "{'c2': 6}"
        ),  # same host, other scenario, near-duplicate flows
        case("c3", "s-1", 500, "10.0.0.9", "benign", "{'benign': 500}"),
        case("c4", "s-2", 3, "10.0.0.7", "unknown", "none"),
        case("c5", "s-3", 10, "10.0.0.5", "benign", "{'benign': 10}"),  # looks like c1, other label
    ]
    return [r[0] for r in rows], [r[1] for r in rows]


def test_host_of_reads_the_gold_note_then_the_state() -> None:
    c, g = case("x", "s-1", 1, "192.0.2.4", "c2", "{}")
    assert host_of(c, g) == "192.0.2.4"
    assert host_of({"state": {"src": "198.51.100.1"}}, {"adjudication_note": ""}) == "198.51.100.1"


def test_host_reuse_counts_hosts_seen_in_several_scenarios() -> None:
    cases, golds = data()
    r = host_reuse(cases, golds)
    assert r["per_class"]["c2"]["in_more_than_one_scenario"] == 1
    assert r["per_class"]["benign"]["in_more_than_one_scenario"] == 0
    assert r["per_held_out_scenario"]["s-2"]["host_also_in_training"] == 1  # c2's host is in s-1


def test_near_duplicates_split_same_label_from_ambiguity() -> None:
    cases, golds = data()
    r = near_duplicates(cases, golds)
    s1 = r["per_held_out_scenario"]["s-1"]
    assert (
        s1["near_duplicate_same_label_elsewhere"] == 1
    )  # c1 ~ c2 (10 and 11 flows share a log2 bucket)
    s3 = r["per_held_out_scenario"]["s-3"]
    assert s3["near_duplicate_other_label_only"] == 1  # c5 looks like the c2 windows


def test_window_label_basis_counts_single_flow_decisions() -> None:
    _, golds = data()
    r = window_label_basis(golds)
    assert r is not None
    assert r["c2_windows"] == 2 and r["c2_flows_per_c2_window"]["exactly_one"] == 1
    assert r["decided_by_a_single_flow"] == 1
    assert window_label_basis([{"gold": {"verdict": "c2"}, "adjudication_note": "no note"}]) is None


def test_audit_bundles_everything() -> None:
    cases, golds = data()
    r = audit(cases, golds)
    assert r["cases"] == 5 and "not a leak" in r["window_overlap"]
