"""AIT Alert Data Set (Zenodo 8263181) -> SOCBench alert cases with weak labels.

What the source labels, and therefore all this adapter may claim: for each of
eight scenarios, the start and end of each attack phase (``labels.csv``:
``scenario,attack,start,end`` in epoch seconds). It does **not** label
individual alerts. An alert inside a phase window may be the attacker, or a
user who happened to be active then; an alert outside every window was raised
while the testbed's attacker was idle, which the dataset authors call a false
positive by time but nobody reviewed.

So the gold this produces is tier D (``label_origin: attack-phase-window``):

* a case that contains an alert inside a phase window is ``suspicious``, with
  ``attack_stage`` set to the phase; never ``confirmed_malicious`` (DATA-002);
* a case with no alert inside any window is ``unknown``. High-severity cases
  among these are listed as hard-negative *candidates* in the summary, for a
  human to review, not marked as hard negatives.

A case is every alert on one host within one time bucket (default 10
minutes) of one scenario. ``family_id`` is the scenario, so a split grouped
by family never puts one attack chain on both sides (review 9.3, step 5).

Event times follow the dataset authors' ``analyze.py`` exactly. Wazuh's own
top-level ``timestamp`` is **not** used: AIT-ADS was produced by replaying
logs, so in raw Wazuh output that field is the replay time. The order is
``data.timestamp`` (Suricata eve), then ``predecoder.timestamp``, then the
timestamp inside ``full_log`` chosen by the log file in ``location``; AMiner
alerts use the last ``LogData.DetectionTimestamp``. Every time is taken as
UTC, as the authors do; syslog lines have no year and take the scenario's.
(Checked against the published archive, 2026-10-17: the records carry no
top-level ``timestamp`` but an ``@timestamp``, and the derived event time
equals it for every one of 340,000 sampled alerts across three scenarios.)

Hosts. Every Wazuh record names the same agent (``agent.name`` is
``wazuh-client``: one collector replayed every host's logs), so the host is
``agent.ip``, which is also what AMiner records carry as ``AMiner.ID``. That is
the key the authors use too. A host's display name is the most common
``predecoder.hostname`` seen for its IP, when any is.
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .records import case_id

__all__ = [
    "Alert",
    "PhaseWindow",
    "build_cases",
    "event_time",
    "iter_alerts",
    "read_phase_windows",
]

SOURCE_ID = "ait-ads"
TASK_ID = "soc.alert.verdict.v1"
_MEMBER = re.compile(r"(?:^|/)(?P<scenario>[a-z]+)_(?P<kind>wazuh|aminer)\.json$")


@dataclass(frozen=True)
class PhaseWindow:
    scenario: str
    attack: str
    start: float
    end: float

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end


@dataclass(frozen=True)
class Alert:
    scenario: str
    sensor: Literal["wazuh", "suricata", "aminer"]
    time: float
    name: str
    host: str
    level: int | None = None
    src_ip: str | None = None
    dest_ip: str | None = None
    dest_port: int | None = None
    url: str | None = None
    host_name: str | None = None
    #: Suricata's own alert severity (1 high .. 3 low); Wazuh files every
    #: Suricata alert at rule level 3, so the level says nothing for them.
    severity: int | None = None
    ref: str = ""


def read_phase_windows(text: str) -> dict[str, list[PhaseWindow]]:
    """``labels.csv`` -> windows per scenario, sorted by start."""
    reader = csv.DictReader(io.StringIO(text))
    missing = {"scenario", "attack", "start", "end"} - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"labels.csv lacks columns {sorted(missing)}")
    out: dict[str, list[PhaseWindow]] = defaultdict(list)
    for row in reader:
        out[row["scenario"]].append(
            PhaseWindow(row["scenario"], row["attack"], float(row["start"]), float(row["end"]))
        )
    return {k: sorted(v, key=lambda w: w.start) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Event time (port of the authors' analyze.py / timestampExtractor.py)
# --------------------------------------------------------------------------- #


def _utc(dt: datetime) -> float:
    return dt.replace(tzinfo=UTC).timestamp()


_SYSLOG_FILES = (
    "/auth.log",
    "/mail.log",
    "/mail.info",
    "/mail.warn",
    "/messages",
    "/daemon.log",
    "logs/syslog",
    "/user.log",
    "/kern.log",
    "/dnsmasq.log",
)
_ACCESS = re.compile(r"\[(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} [+-]\d{4})\]")
_ERROR = re.compile(r"\[\w{3} (\w{3} \d{2} \d{2}:\d{2}:\d{2}\.\d+ \d{4})\]")
_AUDIT = re.compile(r"audit\((\d+)(?:\.\d+)?:")
_SURICATA_LOG = re.compile(r"(\d{1,2}/\d{1,2}/\d{4}) -- (\d{2}:\d{2}:\d{2})")
_ISO_Z = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)Z")
_EXIM = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def _from_full_log(location: str, line: str, year: int) -> float | None:
    try:
        if location.endswith(_SYSLOG_FILES):
            return _utc(datetime.strptime(f"{year} {line[:15]}", "%Y %b %d %H:%M:%S"))
        if location.endswith(("-access.log", "/access.log", "/other_vhosts_access.log")):
            m = _ACCESS.search(line)
            return datetime.strptime(m.group(1), "%d/%b/%Y:%H:%M:%S %z").timestamp() if m else None
        if location.endswith(("-error.log", "/error.log")):
            m = _ERROR.search(line)
            return _utc(datetime.strptime(m.group(1), "%b %d %H:%M:%S.%f %Y")) if m else None
        if location.endswith("/audit.log"):
            m = _AUDIT.search(line)
            return float(m.group(1)) if m else None
        if location.endswith(("/mainlog", "/openvpn.log")):
            m = _EXIM.match(line)
            return _utc(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")) if m else None
        if location.endswith("/fast.log"):
            return _utc(datetime.strptime(line[:26], "%m/%d/%Y-%H:%M:%S.%f"))
        if location.endswith("/eve.json"):
            ts = json.loads(line).get("timestamp")
            return _utc(datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f%z")) if ts else None
        if location.endswith("/suricata.log"):
            m = _SURICATA_LOG.search(line)
            return (
                _utc(datetime.strptime(f"{m.group(1)} {m.group(2)}", "%d/%m/%Y %H:%M:%S"))
                if m
                else None
            )
        if "/logstash/" in location:
            m = _ISO_Z.search(line)
            return _utc(datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S.%f")) if m else None
    except (ValueError, AttributeError, json.JSONDecodeError):
        return None
    return None


def event_time(alert: dict[str, Any], year: int) -> float | None:
    """When the logged event happened, in UTC epoch seconds, or ``None``.

    Never Wazuh's top-level ``timestamp``: that is when the replayed log was
    ingested, days after the scenario ran.
    """
    data = alert.get("data") or {}
    try:
        if isinstance(data, dict) and data.get("timestamp"):
            # The authors parse the offset and then overwrite it with UTC; so do we.
            return _utc(
                datetime.strptime(data["timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z").replace(tzinfo=None)
            )
        pre = alert.get("predecoder") or {}
        if isinstance(pre, dict) and pre.get("timestamp"):
            ts = pre["timestamp"]
            if "/" in ts:
                return _utc(datetime.strptime(ts, "%m/%d/%Y-%H:%M:%S.%f"))
            return _utc(datetime.strptime(f"{year} {ts}", "%Y %b %d %H:%M:%S"))
    except ValueError:
        return None
    full, location = alert.get("full_log"), alert.get("location")
    if isinstance(full, str) and isinstance(location, str):
        return _from_full_log(location, full, year)
    return None


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #


def _wazuh(j: dict[str, Any], scenario: str, ref: str, year: int) -> Alert | None:
    rule = j.get("rule") or {}
    description = rule.get("description")
    if not isinstance(description, str):
        return None
    t = event_time(j, year)
    if t is None:
        return None
    agent = j.get("agent") or {}
    raw_data = j.get("data")
    data: dict[str, Any] = raw_data if isinstance(raw_data, dict) else {}
    pre = j.get("predecoder") if isinstance(j.get("predecoder"), dict) else {}
    suricata = description.startswith("Suricata: ")
    port = data.get("dest_port") or data.get("dstport")
    level = rule.get("level")
    url = data.get("url")
    alert_block = data.get("alert") if isinstance(data.get("alert"), dict) else {}
    sev = (alert_block or {}).get("severity")
    return Alert(
        scenario=scenario,
        sensor="suricata" if suricata else "wazuh",
        time=t,
        name=description.removeprefix("Suricata: "),
        # agent.name is the one replaying collector for every record; the IP
        # is the host (and matches AMiner.ID).
        host=str(agent.get("ip") or agent.get("name") or "unknown"),
        host_name=(pre or {}).get("hostname") or None,
        level=int(level) if isinstance(level, int | str) and str(level).isdigit() else None,
        src_ip=data.get("src_ip") or data.get("srcip"),
        dest_ip=data.get("dest_ip") or data.get("dstip"),
        dest_port=int(port) if isinstance(port, int | str) and str(port).isdigit() else None,
        url=str(url)[:160] if url else None,
        severity=int(sev)
        if suricata and isinstance(sev, int | str) and str(sev).isdigit()
        else None,
        ref=ref,
    )


def _aminer(j: dict[str, Any], scenario: str, ref: str) -> Alert | None:
    try:
        t = float(j["LogData"]["DetectionTimestamp"][-1])
        name = str(j["AnalysisComponent"]["AnalysisComponentName"])
        host = str(j["AMiner"]["ID"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return Alert(scenario=scenario, sensor="aminer", time=t, name=name, host=host, ref=ref)


@dataclass
class ParseStats:
    lines: int = 0
    alerts: int = 0
    untimed: int = 0
    malformed: int = 0
    skipped: int = 0
    by_member: dict[str, int] = field(default_factory=dict)


def iter_alerts(
    archive: Path, windows: dict[str, list[PhaseWindow]], stats: ParseStats | None = None
) -> Iterator[Alert]:
    """Every alert in the zip whose event time can be established."""
    stats = stats if stats is not None else ParseStats()
    with zipfile.ZipFile(archive) as zf:
        for member in sorted(zf.namelist()):
            m = _MEMBER.search(member)
            if not m:
                continue
            scenario, kind = m.group("scenario"), m.group("kind")
            first = windows.get(scenario)
            year = datetime.fromtimestamp(first[0].start, UTC).year if first else 2022
            count = 0
            with zf.open(member) as raw:
                for lineno, line in enumerate(
                    io.TextIOWrapper(raw, encoding="utf-8", errors="replace"), 1
                ):
                    stats.lines += 1
                    if "The average number" in line:  # skipped by the authors too
                        stats.skipped += 1
                        continue
                    try:
                        j = json.loads(line)
                    except json.JSONDecodeError:
                        stats.malformed += 1
                        continue
                    ref = f"{member}:{lineno}"
                    alert = (
                        _aminer(j, scenario, ref)
                        if kind == "aminer"
                        else _wazuh(j, scenario, ref, year)
                    )
                    if alert is None:
                        stats.untimed += 1
                        continue
                    stats.alerts += 1
                    count += 1
                    yield alert
            stats.by_member[member] = count


def phase_of(t: float, windows: Iterable[PhaseWindow]) -> str | None:
    for w in windows:
        if w.contains(t):
            return w.attack
    return None


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _line(a: Alert, repeat: int) -> str:
    parts = [_iso(a.time)[11:], a.sensor]
    if a.sensor == "suricata" and a.severity is not None:
        parts.append(f"S{a.severity}")
    elif a.level is not None:
        parts.append(f"L{a.level}")
    parts.append(a.name)
    net = " ".join(
        x
        for x in (
            f"src {a.src_ip}" if a.src_ip else "",
            f"dst {a.dest_ip}" + (f":{a.dest_port}" if a.dest_port else "") if a.dest_ip else "",
        )
        if x
    )
    if net:
        parts.append(f"[{net}]")
    if a.url:
        parts.append(a.url)
    if repeat > 1:
        parts.append(f"x{repeat}")
    return " ".join(parts)


@dataclass
class _Distinct:
    first: Alert
    count: int = 0
    in_window: bool = False


@dataclass
class _Group:
    count: int = 0
    max_level: int = 0
    sensors: Counter[str] = field(default_factory=Counter)
    phases: Counter[str] = field(default_factory=Counter)
    in_window: int = 0
    alarming: set[str] = field(default_factory=set)
    distinct: dict[tuple[Any, ...], _Distinct] = field(default_factory=dict)


#: Wazuh correlation rules whose level comes from counting other IDS alerts,
#: not from what happened. On the published data they are almost all built on
#: Suricata's severity-3 TLS protocol anomalies, so they say "noisy", not
#: "alarming".
_COUNTING_RULES = ("Multiple IDS alerts", "Multiple IDS events")


def _alarming(a: Alert, level: int) -> bool:
    """An alert an analyst would look at twice: worth a hard-negative review when unlabelled."""
    if a.sensor == "suricata":
        return a.severity is not None and a.severity <= 2
    return (a.level or 0) >= level and not a.name.startswith(_COUNTING_RULES)


def build_cases(
    alerts: Iterable[Alert],
    windows: dict[str, list[PhaseWindow]],
    *,
    source_hash: str,
    bucket_seconds: int = 600,
    max_evidence: int = 25,
    hard_negative_level: int = 10,
    candidates_per_signature: int = 20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Group alerts into cases and attach weak gold. Returns (cases, gold, summary).

    Hard-negative candidates are unlabelled cases with an alarming alert
    (Wazuh level >= ``hard_negative_level`` that is not a counting rule, or
    Suricata severity 1-2), at most ``candidates_per_signature`` per alarming
    signature per scenario. On the published data one informational rule
    (DNS queries to .biz) fires in about 3,000 buckets; uncapped, it would be
    most of the review list.

    Streams: each group keeps counts and one representative per distinct
    alert signature, never the alerts themselves, so memory follows the number
    of groups and signatures rather than the 2.6 million alerts.
    """
    groups: dict[tuple[str, str, int], _Group] = defaultdict(_Group)
    names: dict[str, Counter[str]] = defaultdict(Counter)
    for a in alerts:
        g = groups[(a.scenario, a.host, int(a.time // bucket_seconds))]
        g.count += 1
        g.max_level = max(g.max_level, a.level or 0)
        g.sensors[a.sensor] += 1
        phase = phase_of(a.time, windows.get(a.scenario, ()))
        if phase:
            g.in_window += 1
            g.phases[phase] += 1
        if _alarming(a, hard_negative_level):
            g.alarming.add(f"{a.sensor}: {a.name}")
        sig = (a.sensor, a.name, a.level, a.src_ip, a.dest_ip, a.dest_port, a.url)
        d = g.distinct.get(sig)
        if d is None:
            d = g.distinct[sig] = _Distinct(first=a)
        elif a.time < d.first.time:
            d.first = a
        d.count += 1
        d.in_window = d.in_window or bool(phase)
        if a.host_name:
            names[a.host][a.host_name] += 1

    cases: list[dict[str, Any]] = []
    golds: list[dict[str, Any]] = []
    verdicts: Counter[str] = Counter()
    stages: Counter[str] = Counter()
    candidates: list[str] = []
    signature_cases: Counter[tuple[str, str]] = Counter()
    signature_total: Counter[str] = Counter()

    for n, key in enumerate(sorted(groups), start=1):
        scenario, host, bucket = key
        g = groups[key]
        cid = case_id("ait", n)
        distinct = sorted(g.distinct.values(), key=lambda d: (d.first.time, d.first.name))
        shown = distinct[:max_evidence]
        evidence = [
            {"id": f"e{i}", "field": "alert", "value": _line(d.first, d.count)}
            for i, d in enumerate(shown, start=1)
        ]
        start, end = bucket * bucket_seconds, (bucket + 1) * bucket_seconds
        label = names[host].most_common(1)[0][0] if names.get(host) else None

        # The state is what a model sees: alerts only. No phase, no window
        # membership, no scenario name -- each of those would be the label.
        cases.append(
            {
                "case_id": cid,
                "family_id": f"ait-{scenario}",
                "source_dataset": SOURCE_ID,
                "source_hash": source_hash,
                "telemetry_origin": "lab-reproduction",
                "language": "en",
                "task_id": TASK_ID,
                "state": {
                    "host": f"{label} ({host})" if label else host,
                    "window_start": _iso(start),
                    "window_end": _iso(end),
                    "alert_count": g.count,
                    "distinct_alerts": len(distinct),
                    "sensors": ", ".join(f"{k}:{v}" for k, v in sorted(g.sensors.items())),
                    "alerts": "\n".join(e["value"] for e in evidence),
                },
                "evidence": evidence,
                "split_group": f"ait-{scenario}",
                "license_reviewed": False,
                "pii_reviewed": False,
            }
        )

        if g.in_window:
            stage = g.phases.most_common(1)[0][0]
            gold: dict[str, Any] = {
                "verdict": "suspicious",
                "confidence_of_gold": "low",
                "allowed_alternatives": ["confirmed_malicious", "confirmed_benign"],
                "attack_stage": stage,
                "correct_evidence_ids": [
                    e["id"] for e, d in zip(evidence, shown, strict=True) if d.in_window
                ],
                "prohibited_actions": ["close", "suppress"],
            }
            stages[stage] += 1
        else:
            gold = {
                "verdict": "unknown",
                "confidence_of_gold": "low",
                "allowed_alternatives": ["confirmed_benign", "suspicious"],
                "attack_stage": None,
            }
            if g.alarming:
                signature_total.update(g.alarming)
                if any(
                    signature_cases[(scenario, sig)] < candidates_per_signature
                    for sig in g.alarming
                ):
                    candidates.append(cid)
                    signature_cases.update((scenario, sig) for sig in g.alarming)
        verdicts[gold["verdict"]] += 1
        golds.append(
            {
                "case_id": cid,
                "task_id": TASK_ID,
                "label_origin": "attack-phase-window",
                "gold": gold,
                "adjudication_status": "not-applicable",
                "adjudication_note": (
                    f"{g.in_window} of {g.count} alerts fall inside an attack-phase window"
                    if g.in_window
                    else "no alert falls inside an attack-phase window; unlabelled, not benign"
                ),
            }
        )

    summary = {
        "source": SOURCE_ID,
        "task_id": TASK_ID,
        "label_origin": "attack-phase-window",
        "trust_tier": "D",
        "cases": len(cases),
        "by_verdict": dict(verdicts),
        "by_attack_stage": dict(stages),
        "families": sorted({c["family_id"] for c in cases}),
        "hosts": {
            fam: n
            for fam, n in sorted(
                Counter(f"ait-{k[0]}" for k in {(s, h) for s, h, _ in groups}).items()
            )
        },
        "hard_negative_candidates": candidates,
        "hard_negative_signatures": dict(signature_total.most_common()),
        "candidates_per_signature": candidates_per_signature,
        "bucket_seconds": bucket_seconds,
        "caveat": (
            "suspicious means 'inside an attack-phase window', not 'malicious'; unknown means "
            "'outside every window', not 'benign'. Promote either only through analyst review."
        ),
    }
    return cases, golds, summary
