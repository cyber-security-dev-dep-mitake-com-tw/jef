"""AIT-ADS and CTU-13 adapters, on small fixtures shaped like the real files.

The fixtures follow the formats the dataset authors document and their own
analysis code reads (ait-aecid/alert-data-set analyze.py; Argus binetflow
columns). The labels these tests pin are the review's DATA-001 and DATA-002:
CTU-13 Background is never benign, and a legitimate alert inside an AIT attack
window is never a confirmed true positive.
"""

from __future__ import annotations

import io
import json
import tarfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from jef_train.socbench import SocbenchRepo, ait_ads, ctu13
from jef_train.socbench.build import build_source
from jef_train.socbench.records import read_jsonl, trust_tier

ROOT = Path(__file__).resolve().parents[3]


def ts(s: str) -> float:
    return datetime.fromisoformat(s).replace(tzinfo=UTC).timestamp()


@pytest.fixture
def repo() -> SocbenchRepo:
    return SocbenchRepo(ROOT / "benchmark" / "socbench")


# --------------------------------------------------------------------------- #
# AIT-ADS
# --------------------------------------------------------------------------- #

LABELS = (
    "scenario,attack,start,end\n"
    f"fox,network_scans,{ts('2022-01-18T11:00:00')},{ts('2022-01-18T11:10:00')}\n"
    f"fox,service_scans,{ts('2022-01-18T11:10:00')},{ts('2022-01-18T11:20:00')}\n"
)

FW, INTRA, MAIL = "10.35.32.1", "10.35.35.206", "172.17.131.81"

# Shaped like the published records (checked 2026-10-17): one replaying agent
# named wazuh-client for every host, the host identified by agent.ip, an
# occasional predecoder.hostname, and an @timestamp carrying event time.
WAZUH = [
    # Suricata via Wazuh, event time in data.timestamp. A stray top-level
    # timestamp (raw Wazuh's replay time) must be ignored.
    {
        "timestamp": "2022-06-01T00:00:00.000+0000",
        "@timestamp": "2022-01-18T11:00:30.000000Z",
        "rule": {"description": "Suricata: ET SCAN Nmap Scripting Engine", "level": 3},
        "agent": {"name": "wazuh-client", "ip": FW},
        "data": {
            "timestamp": "2022-01-18T11:00:30.000000+0000",
            "src_ip": "192.168.1.5",
            "dest_ip": "10.35.35.2",
            "dest_port": "80",
            "alert": {"severity": 1},
        },
    },
    # Suricata protocol noise at severity 3, outside every window, and the Wazuh
    # counting rule built on it at level 10: neither is a hard-negative candidate.
    {
        "rule": {
            "description": "Suricata: Alert - SURICATA TLS invalid record/traffic",
            "level": 3,
        },
        "agent": {"name": "wazuh-client", "ip": "10.35.35.9"},
        "data": {"timestamp": "2022-01-18T17:00:00.000000+0000", "alert": {"severity": 3}},
    },
    {
        "rule": {"description": "Multiple IDS alerts for same id.", "level": 10},
        "agent": {"name": "wazuh-client", "ip": "10.35.35.9"},
        "data": {"timestamp": "2022-01-18T17:00:05.000000+0000"},
    },
    # Benign housekeeping inside the window, same host and bucket: DATA-002.
    {
        "rule": {"description": "ClamAV database update", "level": 3},
        "agent": {"name": "wazuh-client", "ip": FW},
        "predecoder": {"hostname": "inet-firewall", "timestamp": "Jan 18 11:02:00"},
    },
    # Syslog-only time on another host, inside the window.
    {
        "rule": {"description": "sshd: authentication failed.", "level": 5},
        "agent": {"name": "wazuh-client", "ip": INTRA},
        "predecoder": {"hostname": "intranet-server", "timestamp": "Jan 18 11:05:00"},
    },
    # Apache access log time from full_log, hours after any window: unknown.
    # Wazuh's own web fields are srcip / dstip / url, not Suricata's names.
    {
        "rule": {"description": "Web server 400 error code.", "level": 5},
        "agent": {"name": "wazuh-client", "ip": INTRA},
        "data": {"srcip": "10.1.1.1", "url": "/wp-login.php", "protocol": "GET"},
        "location": "/var/log/apache2/intranet-access.log",
        "full_log": '10.1.1.1 - - [18/Jan/2022:15:00:00 +0000] "GET /wp-login.php HTTP/1.1" 400 0',
    },
    # High severity outside every window: a hard-negative candidate.
    {
        "rule": {
            "description": "Multiple authentication failures followed by a success.",
            "level": 12,
        },
        "agent": {"name": "wazuh-client", "ip": MAIL},
        "predecoder": {"hostname": "mail", "timestamp": "Jan 18 16:00:00"},
    },
    # Only the replay timestamp: untimed, dropped.
    {
        "timestamp": "2022-06-01T00:00:00.000+0000",
        "rule": {"description": "x", "level": 3},
        "agent": {"name": "wazuh-client", "ip": FW},
    },
]
AMINER = [
    {
        "AnalysisComponent": {"AnalysisComponentName": "AMiner: New event type."},
        "AMiner": {"ID": INTRA},
        "LogData": {"DetectionTimestamp": [ts("2022-01-18T11:06:00")], "RawLogData": ["x"]},
    }
]


@pytest.fixture
def ait_files(tmp_path: Path) -> tuple[Path, Path]:
    archive = tmp_path / "ait-ads" / "ait_ads.zip"
    archive.parent.mkdir(parents=True)
    lines = [json.dumps(x) for x in WAZUH] + ["The average number of alerts is 3", "{not json"]
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("ait_ads/fox_wazuh.json", "\n".join(lines) + "\n")
        zf.writestr("ait_ads/fox_aminer.json", "\n".join(json.dumps(x) for x in AMINER) + "\n")
        zf.writestr("ait_ads/README.txt", "ignored")
    labels = archive.parent / "labels.csv"
    labels.write_text(LABELS)
    return archive, labels


def test_phase_windows_read_the_published_columns() -> None:
    windows = ait_ads.read_phase_windows(LABELS)
    assert [w.attack for w in windows["fox"]] == ["network_scans", "service_scans"]
    with pytest.raises(ValueError, match="columns"):
        ait_ads.read_phase_windows("scenario,phase\nfox,x\n")


def test_event_time_follows_the_authors_order_and_ignores_replay_time() -> None:
    assert ait_ads.event_time(WAZUH[0], 2022) == ts("2022-01-18T11:00:30")
    assert ait_ads.event_time(WAZUH[1], 2022) == ts("2022-01-18T17:00:00")
    assert ait_ads.event_time(WAZUH[4], 2022) == ts("2022-01-18T11:05:00")
    assert ait_ads.event_time(WAZUH[5], 2022) == ts("2022-01-18T15:00:00")
    assert ait_ads.event_time(WAZUH[7], 2022) is None, "the replay timestamp is not event time"
    fast = {"predecoder": {"timestamp": "01/18/2022-11:00:30.123456"}}
    assert ait_ads.event_time(fast, 2022) == pytest.approx(ts("2022-01-18T11:00:30.123456"))
    audit = {
        "location": "/var/log/audit/audit.log",
        "full_log": "type=USER_AUTH msg=audit(1642503630.123:42): x",
    }
    assert ait_ads.event_time(audit, 2022) == 1642503630.0


def test_alerts_parse_and_stats_count_what_was_dropped(ait_files) -> None:
    archive, labels = ait_files
    windows = ait_ads.read_phase_windows(labels.read_text())
    stats = ait_ads.ParseStats()
    alerts = list(ait_ads.iter_alerts(archive, windows, stats))
    assert stats.alerts == 8 and len(alerts) == 8
    assert stats.untimed == 1 and stats.skipped == 1 and stats.malformed == 1
    sensors = {a.sensor for a in alerts}
    assert sensors == {"wazuh", "suricata", "aminer"}
    suricata = next(a for a in alerts if a.sensor == "suricata")
    assert suricata.name.startswith("ET SCAN") and suricata.dest_port == 80


def test_data_002_benign_alerts_in_a_window_are_suspicious_not_malicious(ait_files) -> None:
    archive, labels = ait_files
    windows = ait_ads.read_phase_windows(labels.read_text())
    alerts = list(ait_ads.iter_alerts(archive, windows))
    cases, golds, summary = ait_ads.build_cases(alerts, windows, source_hash="sha256:" + "0" * 64)
    by_id = {g["case_id"]: g for g in golds}
    firewall = next(c for c in cases if FW in c["state"]["host"])
    gold = by_id[firewall["case_id"]]
    assert "ClamAV database update" in firewall["state"]["alerts"]
    assert gold["gold"]["verdict"] == "suspicious"
    assert gold["gold"]["attack_stage"] == "network_scans"
    assert "confirmed_malicious" not in {g["gold"]["verdict"] for g in golds}
    assert gold["label_origin"] == "attack-phase-window"
    assert trust_tier(gold["label_origin"]) == "D"
    assert summary["trust_tier"] == "D"


def test_outside_every_window_is_unknown_and_severe_ones_are_candidates(ait_files) -> None:
    archive, labels = ait_files
    windows = ait_ads.read_phase_windows(labels.read_text())
    cases, golds, summary = ait_ads.build_cases(
        ait_ads.iter_alerts(archive, windows), windows, source_hash="sha256:" + "0" * 64
    )
    verdicts = {
        (c["state"]["window_start"][11:16], c["state"]["host"]): g["gold"]["verdict"]
        for c, g in zip(cases, golds, strict=True)
    }
    assert verdicts[("15:00", f"intranet-server ({INTRA})")] == "unknown"
    assert verdicts[("16:00", f"mail ({MAIL})")] == "unknown"
    mail = next(c for c in cases if MAIL in c["state"]["host"])
    assert mail["case_id"] in summary["hard_negative_candidates"]
    noise = next(c for c in cases if "10.35.35.9" in c["state"]["host"])
    assert noise["case_id"] not in summary["hard_negative_candidates"], (
        "counting rules over severity-3 noise are not alarming"
    )
    assert "suricata S3 Alert - SURICATA TLS" in noise["state"]["alerts"]
    assert "benign" not in {g["gold"]["verdict"] for g in golds}


def test_hosts_are_agent_ips_shared_with_aminer(ait_files) -> None:
    """agent.name is one replaying collector; the IP is the host, as in AMiner.ID."""
    archive, labels = ait_files
    windows = ait_ads.read_phase_windows(labels.read_text())
    cases, _, _ = ait_ads.build_cases(
        ait_ads.iter_alerts(archive, windows), windows, source_hash="sha256:" + "0" * 64
    )
    assert all("wazuh-client" not in c["state"]["host"] for c in cases)
    intranet = next(
        c
        for c in cases
        if c["state"]["host"] == f"intranet-server ({INTRA})"
        and c["state"]["window_start"].endswith("11:00:00Z")
    )
    assert intranet["state"]["sensors"] == "aminer:1, wazuh:1", (
        "AMiner and Wazuh alerts for one IP share a case"
    )
    web = next(c for c in cases if c["state"]["window_start"].endswith("15:00:00Z"))
    assert "/wp-login.php" in web["state"]["alerts"] and "src 10.1.1.1" in web["state"]["alerts"]


def test_the_state_never_carries_the_label(ait_files) -> None:
    archive, labels = ait_files
    windows = ait_ads.read_phase_windows(labels.read_text())
    cases, _, _ = ait_ads.build_cases(
        list(ait_ads.iter_alerts(archive, windows)), windows, source_hash="sha256:" + "0" * 64
    )
    blob = json.dumps([c["state"] for c in cases]) + json.dumps([c["evidence"] for c in cases])
    for leak in ("network_scans", "service_scans", "fox", "attack", "phase", "window_label"):
        if leak == "fox":
            # Host names carry the scenario prefix in AIT; the scenario name itself is not a field.
            assert all("scenario" not in c["state"] for c in cases)
            continue
        assert leak not in blob, leak


def test_build_writes_validated_files(repo: SocbenchRepo, ait_files, tmp_path: Path) -> None:
    assert ait_files[0].is_file()
    summary = build_source(repo, "ait-ads", base=tmp_path)
    out = Path(summary["out_dir"])
    cases, golds = read_jsonl(out / "cases.jsonl"), read_jsonl(out / "gold.jsonl")
    assert len(cases) == len(golds) == summary["written"]["cases"] > 0
    assert all(c["case_id"].startswith("socb-ait-") for c in cases)
    assert {c["family_id"] for c in cases} == {"ait-fox"}
    assert json.loads((out / "summary.json").read_text())["parse"]["untimed"] == 1
    assert (
        cases[0]["source_hash"].startswith("sha256:")
        and cases[0]["source_hash"] != "sha256:" + "0" * 64
    )


def test_build_says_how_to_get_missing_inputs(repo: SocbenchRepo, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="jef-bench fetch --source ait-ads"):
        build_source(repo, "ait-ads", base=tmp_path)


# --------------------------------------------------------------------------- #
# CTU-13
# --------------------------------------------------------------------------- #

HEADER = "StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label\n"
BOT, CNC = "147.32.84.165", "195.88.191.59"


def row(t: str, src: str, dst: str, dport: str, label: str, sport: str = "1025") -> str:
    return (
        f"2011/08/10 {t},1.0,tcp,{src},{sport},   ->,{dst},{dport},CON,0,0,4,276,156,flow={label}\n"
    )


@pytest.fixture
def ctu_archive(tmp_path: Path) -> Path:
    lines = [HEADER]
    for i in range(6):  # a periodic C&C beacon
        lines.append(
            row(
                f"10:{i:02d}:00.000000",
                BOT,
                CNC,
                "80",
                "From-Botnet-V42-TCP-CC6-Plain-HTTP-Encrypted-Data",
                str(2000 + i),
            )
        )
    lines.append(
        row("10:30:00.000000", BOT, "64.12.168.40", "25", "From-Botnet-V42-TCP-Attempt-SPAM")
    )
    lines.append(
        row("10:31:00.000000", "147.32.84.170", "147.32.80.9", "53", "From-Normal-V42-Stribrek")
    )
    lines.append(row("10:32:00.000000", "147.32.86.1", BOT, "445", "To-Botnet-V42-TCP-Attempt"))
    for i in range(30):
        lines.append(
            row(
                f"11:{i:02d}:00.000000",
                f"10.0.{i}.1",
                "147.32.80.9",
                "53",
                "Background-UDP-Established",
            )
        )
    # One conversation whose labelled flows disagree.
    lines.append(
        row("12:00:00.000000", "147.32.84.170", "8.8.8.8", "53", "From-Normal-V42-Stribrek")
    )
    lines.append(
        row("12:01:00.000000", "147.32.84.170", "8.8.8.8", "53", "From-Botnet-V42-UDP-DNS")
    )
    raw = "".join(lines).encode()
    archive = tmp_path / "ctu-13" / "CTU-13-Dataset.tar.bz2"
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w:bz2") as tf:
        info = tarfile.TarInfo("CTU-13-Dataset/1/capture20110810.binetflow")
        info.size = len(raw)
        tf.addfile(info, io.BytesIO(raw))
    return archive


@pytest.mark.parametrize(
    ("label", "behaviour", "host"),
    [
        ("flow=From-Botnet-V43-UDP-DNS", "unknown", "botnet"),
        ("flow=From-Botnet-V42-UDP-Attempt-DNS", "unknown", "botnet"),
        ("flow=From-Botnet-V42-1-UDP-DNS", "unknown", "botnet"),
        ("flow=From-Botnet-V42-TCP-Established", "unknown", "botnet"),
        ("flow=From-Botnet-V42-TCP-WEB-Established", "unknown", "botnet"),
        ("flow=From-Botnet-V42-TCP-Attempt", "botnet", "botnet"),
        ("flow=From-Botnet-V42-TCP-Established-HTTP-Ad", "botnet", "botnet"),
        ("flow=From-Botnet-V45-TCP-Not-Encrypted-SMTP-Private-Proxy", "botnet", "botnet"),
        ("flow=From-Botnet-V51-2-ICMP", "botnet", "botnet"),
        ("flow=From-Botnet-V42-TCP-CC74-HTTP-Custom-Port-Not-Encrypted", "c2", "c2"),
        ("flow=From-Normal-V42-Stribrek", "benign", "benign"),
    ],
)
def test_ctu_flow_label_semantics(label: str, behaviour: str, host: str) -> None:
    """Behaviour labels follow the labelbook; host labels mark every bot flow."""
    assert ctu13.map_label(label)[0] == behaviour, "behaviour is the default"
    assert ctu13.map_label(label, "behaviour")[0] == behaviour
    assert ctu13.map_label(label, "host")[0] == host


def test_ctu_unknown_semantics_is_refused() -> None:
    with pytest.raises(ValueError, match="semantics"):
        ctu13.map_label("flow=From-Botnet-V42-UDP-DNS", "flow")


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("flow=From-Botnet-V42-TCP-CC6-Plain-HTTP-Encrypted-Data", "c2"),
        ("flow=To-Botnet-V42-TCP-Attempt", "unknown"),
        ("flow=From-Normal-V42-Stribrek", "benign"),
        ("flow=To-Normal-V45-UDP-NTP-server", "unknown"),
        ("flow=Normal-V45-HTTP-windowsupdate", "benign"),
        ("flow=From-Botnet-V45-TCP-CC106-IRC-Not-Encrypted", "c2"),
        ("flow=From-Botnet-V45-TCP-Attempt-SPAM", "botnet"),
        ("flow=Background-UDP-Established", "unknown"),
        ("flow=To-Background-UDP-CVUT-DNS-Server", "unknown"),
        ("", "unknown"),
    ],
)
def test_ctu_label_mapping(label: str, expected: str) -> None:
    assert ctu13.map_label(label)[0] == expected


def test_data_001_background_is_never_benign(ctu_archive: Path) -> None:
    cases, golds, summary = ctu13.build_cases(
        ctu13.iter_flows(ctu_archive), source_hash="sha256:" + "0" * 64, background_rate=1.0
    )
    for case, gold in zip(cases, golds, strict=True):
        if case["state"]["dst"] == "147.32.80.9" and case["state"]["src"].startswith("10.0."):
            assert gold["gold"]["verdict"] == "unknown"
    assert summary["unknown_label_cases"] >= 30
    assert summary["known_label_cases"] == 3  # c2, botnet (spam), benign


def test_ctu_conversations_and_their_verdicts(ctu_archive: Path) -> None:
    cases, golds, _ = ctu13.build_cases(
        ctu13.iter_flows(ctu_archive), source_hash="sha256:" + "0" * 64, background_rate=1.0
    )
    by_dst = {
        (c["state"]["src"], c["state"]["dst"]): (c, g) for c, g in zip(cases, golds, strict=True)
    }
    beacon, gold = by_dst[(BOT, CNC)]
    assert gold["gold"]["verdict"] == "c2"
    assert beacon["state"]["flows"] == 6
    assert beacon["state"]["mean_interval_s"] == 60.0 and beacon["state"]["interval_cv"] == 0.0
    assert by_dst[(BOT, "64.12.168.40")][1]["gold"]["verdict"] == "botnet"
    assert by_dst[("147.32.84.170", "147.32.80.9")][1]["gold"]["verdict"] == "benign"
    assert by_dst[("147.32.86.1", BOT)][1]["gold"]["verdict"] == "unknown", (
        "To-Botnet is not malicious by label"
    )
    assert by_dst[("147.32.84.170", "8.8.8.8")][1]["gold"]["verdict"] == "unknown", (
        "disagreeing labels"
    )
    assert {c["family_id"] for c in cases} == {"ctu13-scenario-1"}
    assert all("Botnet" not in json.dumps(c) for c in cases), (
        "the label never reaches the state or evidence"
    )


def test_ctu_sampling_is_deterministic_and_keeps_every_labelled_conversation(
    ctu_archive: Path,
) -> None:
    first = ctu13.build_cases(
        ctu13.iter_flows(ctu_archive), source_hash="sha256:" + "0" * 64, background_rate=0.2
    )
    again = ctu13.build_cases(
        ctu13.iter_flows(ctu_archive), source_hash="sha256:" + "0" * 64, background_rate=0.2
    )
    assert first[0] == again[0]
    assert first[2]["known_label_cases"] == 3
    assert first[2]["unknown_label_cases"] < 30


def test_ctu_build_writes_validated_files(
    repo: SocbenchRepo, ctu_archive: Path, tmp_path: Path
) -> None:
    summary = build_source(repo, "ctu-13", base=tmp_path, options={"background_rate": 1.0})
    golds = read_jsonl(Path(summary["out_dir"]) / "gold.jsonl")
    assert summary["trust_tier"] == "C"
    assert {g["label_origin"] for g in golds} == {"flow-label"}
    assert all(c.startswith("socb-ctu13-") for c in (g["case_id"] for g in golds))


def test_ctu_build_honours_the_flow_label_option(
    repo: SocbenchRepo, ctu_archive: Path, tmp_path: Path
) -> None:
    """The CLI option reaches the adapter: host and behaviour builds differ on bot DNS."""
    host = build_source(
        repo,
        "ctu-13",
        base=tmp_path / "h",
        input_path=ctu_archive,
        options={"background_rate": 1.0, "flow_labels": "host"},
    )
    behaviour = build_source(
        repo,
        "ctu-13",
        base=tmp_path / "b",
        input_path=ctu_archive,
        options={"background_rate": 1.0, "flow_labels": "behaviour"},
    )
    assert host["flow_labels"] == "host" and behaviour["flow_labels"] == "behaviour"
    assert host["flows_seen_by_class"].get("botnet", 0) > behaviour["flows_seen_by_class"].get(
        "botnet", 0
    )
