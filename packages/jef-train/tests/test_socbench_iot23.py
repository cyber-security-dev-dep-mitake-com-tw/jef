"""IoT-23 adapter: both column layouts, label case, host roles, fusion cases."""

from __future__ import annotations

import json
import re
from pathlib import Path

from jef_train.socbench.build import build_source
from jef_train.socbench.ctu13_hosts import build_host_windows
from jef_train.socbench.iot23 import classify, host_roles, internal, iter_flows
from jef_train.socbench.manifest import SocbenchRepo

FIELDS = (
    "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tservice\tduration\t"
    "orig_bytes\tresp_bytes\tconn_state\tlocal_orig\tlocal_resp\tmissed_bytes\thistory\torig_pkts\t"
    "orig_ip_bytes\tresp_pkts\tresp_ip_bytes\ttunnel_parents   label   detailed-label\n"
)
ROOT = Path(__file__).resolve().parents[3]


def row(ts: float, src: str, dst: str, dport: str, tail: str, *, tabs: bool = False) -> str:
    base = (
        f"{ts}\tC1\t{src}\t4000\t{dst}\t{dport}\ttcp\t-\t0.5\t10\t20\tS0\t-\t-\t0\tS\t1\t60\t0\t0"
    )
    return base + ("\t" + tail.replace("   ", "\t") if tabs else "\t" + tail) + "\n"


def write_captures(d: Path) -> Path:
    d.mkdir(parents=True)
    bot, pot = "192.168.1.5", "192.168.1.9"
    lines = [FIELDS]
    for i in range(6):
        lines.append(
            row(100 + i, bot, "203.0.113.7", "23", "-   Malicious   PartOfAHorizontalPortScan")
        )
    for i in range(4):
        lines.append(row(500 + i, bot, "203.0.113.8", "6667", "-   Malicious   C&C"))
    for i in range(5):
        lines.append(row(900 + i, bot, "198.51.100.2", "123", "-   Benign   -"))
    (d / "CTU-IoT-Malware-Capture-9-1.conn.log.labeled").write_text("".join(lines))
    tabbed = [FIELDS] + [
        row(100 + i, pot, "198.51.100.3", "443", "-   benign   -", tabs=True) for i in range(5)
    ]
    (d / "CTU-Honeypot-Capture-4-1.conn.log.labeled").write_text("".join(tabbed))
    return d


def test_both_layouts_and_label_case_parse(tmp_path: Path) -> None:
    d = write_captures(tmp_path / "iot")
    flows = list(iter_flows(d))
    assert {f.scenario for f in flows} == {"honeypot-4-1", "malware-9-1"}
    assert {f.label for f in flows} >= {
        "Malicious|C&C",
        "Benign|-",
        "Malicious|PartOfAHorizontalPortScan",
    }
    assert all(f.row > 0 and f.source_file.endswith(".conn.log.labeled") for f in flows)
    assert classify(next(f for f in flows if "C&C" in f.label)) == ("c2", "c2")
    assert internal("192.168.1.5") and not internal("203.0.113.7")


def test_roles_and_behaviour_gold(tmp_path: Path) -> None:
    d = write_captures(tmp_path / "iot")
    roles = host_roles(d)
    assert roles[("malware-9-1", "192.168.1.5")] == "infected"
    assert roles[("honeypot-4-1", "192.168.1.9")] == "normal"
    cases, golds, _ = build_host_windows(
        iter_flows(d),
        source_hash="sha256:" + "0" * 64,
        gold="behaviour",
        fusion=True,
        classify=classify,
        host_roles=roles,
        internal=internal,
        source_id="iot-23",
        case_tag="iot23f",
        family_prefix="iot23-",
    )
    verdicts = sorted(g["gold"]["verdict"] for g in golds)
    assert verdicts == ["benign", "botnet", "c2", "unknown"], verdicts
    for c in cases:
        text = json.dumps({"s": c["state"], "e": c["evidence"]})
        assert not re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b|Malicious|Benign|C&C|PortScan", text)


def test_build_source_reads_a_directory_and_validates(tmp_path: Path) -> None:
    d = write_captures(tmp_path / "iot-23")
    summary = build_source(
        SocbenchRepo(ROOT / "benchmark" / "socbench"),
        "iot-23",
        input_path=d,
        out_dir=tmp_path / "out",
        options={},
    )
    assert summary["source"] == "iot-23" and summary["fusion"] is True
    assert (
        summary["source_hash"].startswith("sha256:")
        and summary["source_hash"] != "sha256:" + "0" * 64
    )
