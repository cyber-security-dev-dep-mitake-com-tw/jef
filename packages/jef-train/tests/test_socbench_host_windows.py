"""CTU-13 host windows: host-level labels, masked addresses, traceable evidence."""

from __future__ import annotations

import json
import re

from jef_train.socbench.ctu13 import Flow
from jef_train.socbench.ctu13_hosts import HOST_TASK_ID, build_host_windows, host_label
from jef_train.socbench.manifest import SocbenchRepo
from jef_train.socbench.records import validate_records

BOT, NORMAL, BG = "147.32.84.165", "147.32.84.170", "203.0.113.9"
IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def flow(
    src: str, label: str, start: float, dport: str = "53", dst: str = "8.8.8.8", row: int = 1
) -> Flow:
    return Flow(
        scenario="1",
        start=start,
        duration=0.1,
        proto="udp",
        src=src,
        sport="1234",
        dst=dst,
        dport=dport,
        state="CON",
        packets=2,
        bytes=150,
        label=label,
        source_file="CTU-13-Dataset/1/capture.binetflow",
        row=row,
    )


def flows() -> list[Flow]:
    out = []
    for i in range(12):  # window 0: bot DNS; window 1: bot with one C&C flow
        out.append(flow(BOT, "flow=From-Botnet-V42-UDP-DNS", 10 + i, row=i + 1))
        out.append(flow(BOT, "flow=From-Botnet-V42-UDP-DNS", 310 + i, row=100 + i))
    out.append(
        flow(
            BOT, "flow=From-Botnet-V42-TCP-CC6-Plain-HTTP-Encrypted-Data", 320, dport="80", row=200
        )
    )
    for i in range(5):
        out.append(
            flow(
                NORMAL,
                "flow=From-Normal-V42-Grill",
                20 + i,
                dport="443",
                dst="147.32.80.9",
                row=300 + i,
            )
        )
        out.append(flow(BG, "flow=Background-UDP-Established", 30 + i, row=400 + i))
    return out


def test_label_semantics() -> None:
    assert host_label("flow=From-Botnet-V42-UDP-DNS") == "botnet"
    assert host_label("flow=From-Botnet-V42-TCP-CC6-Plain-HTTP") == "c2"
    assert host_label("flow=From-Normal-V42-Grill") == "benign"
    assert host_label("flow=To-Botnet-V42-UDP") is None
    assert host_label("flow=Background-TCP-Established") is None


def test_windows_carry_host_level_gold_and_masked_evidence() -> None:
    cases, golds, summary = build_host_windows(
        flows(), source_hash="sha256:" + "0" * 64, background_host_rate=1.0
    )
    verdicts = {g["gold"]["verdict"] for g in golds}
    assert verdicts == {"botnet", "c2", "benign", "unknown"}
    assert all(c["task_id"] == HOST_TASK_ID for c in cases)
    assert summary["distinct_hosts_by_verdict"]["botnet"] == 1
    for c in cases:
        text = json.dumps({"state": c["state"], "evidence": c["evidence"]})
        assert not IP.search(text), f"an address leaked into what the scorer sees: {text}"
    by_id = {g["case_id"]: g for g in golds}
    c2 = next(c for c in cases if by_id[c["case_id"]]["gold"]["verdict"] == "c2")
    note = by_id[c2["case_id"]]["adjudication_note"]
    assert BOT in note and "capture.binetflow#" in note, "the gold keeps the trace"
    dns = next(c for c in cases if by_id[c["case_id"]]["gold"]["verdict"] == "botnet")
    assert dns["state"]["dns_flows"] == 12 and dns["state"]["flows"] == 12


def test_background_hosts_are_sampled_by_host_and_tiny_windows_dropped() -> None:
    _, golds, _ = build_host_windows(
        flows(), source_hash="sha256:" + "0" * 64, background_host_rate=0.0
    )
    assert "unknown" not in {g["gold"]["verdict"] for g in golds}
    _, golds, _ = build_host_windows(
        flows(), source_hash="sha256:" + "0" * 64, min_flows=6, background_host_rate=1.0
    )
    assert {g["gold"]["verdict"] for g in golds} == {"botnet", "c2"}


def test_records_validate_against_the_schemas() -> None:
    repo = SocbenchRepo()
    cases, golds, _ = build_host_windows(
        flows(), source_hash="sha256:" + "0" * 64, background_host_rate=1.0
    )
    assert validate_records(repo, "case", cases) == len(cases)
    assert validate_records(repo, "gold", golds) == len(golds)


def fusion_flows() -> list[Flow]:
    out = flows()
    for i in range(6):  # window 2: the bot sends only DNS -> unknown under behaviour gold
        out.append(flow(BOT, "flow=From-Botnet-V42-UDP-DNS", 620 + i * 30, row=500 + i))
    for i in range(5):  # window 3: spam attempts -> botnet
        out.append(
            flow(BOT, "flow=From-Botnet-V42-TCP-Attempt-SPAM", 920 + i, dport="25", row=600 + i)
        )
    return out


def test_fusion_gold_follows_behaviour_and_tags_never_leak() -> None:
    cases, golds, summary = build_host_windows(
        fusion_flows(),
        source_hash="sha256:" + "0" * 64,
        background_host_rate=1.0,
        gold="behaviour",
        fusion=True,
    )
    by_id = {g["case_id"]: g for g in golds}
    bot = sorted(
        (c for c in cases if BOT in by_id[c["case_id"]]["adjudication_note"]),
        key=lambda c: c["case_id"],
    )
    verdicts = [by_id[c["case_id"]]["gold"]["verdict"] for c in bot]
    assert verdicts == ["unknown", "c2", "unknown", "botnet"], verdicts
    assert {g["label_origin"] for g in golds} == {"host-behaviour-label"}
    assert summary["gold"] == "behaviour" and summary["fusion"] is True
    normal = [c for c in cases if NORMAL in by_id[c["case_id"]]["adjudication_note"]]
    assert by_id[normal[0]["case_id"]]["gold"]["verdict"] == "benign"
    for c in cases:
        text = json.dumps({"state": c["state"], "evidence": c["evidence"]})
        assert not IP.search(text)
        assert not re.search(r"Botnet|SPAM|CC\d|Normal|Background", text), text
        assert c["evidence"][0]["field"] == "conversation"
        assert len(c["state"]["flows_by_minute"].split(",")) == 5
    note = by_id[bot[1]["case_id"]]["adjudication_note"]
    assert "host role: infected" in note and "conversations: e1=" in note


def test_host_mode_is_unchanged_by_the_fusion_options() -> None:
    a = build_host_windows(flows(), source_hash="sha256:" + "0" * 64, background_host_rate=1.0)
    b = build_host_windows(
        flows(),
        source_hash="sha256:" + "0" * 64,
        background_host_rate=1.0,
        gold="host",
        fusion=False,
    )
    assert a[0] == b[0] and a[1] == b[1]
    assert all("flows_by_minute" not in c["state"] for c in a[0])
