"""CTU-13 (Stratosphere Lab) -> SOCBench NDR cases from labelled bidirectional flows.

The source labels every Argus bidirectional flow in its ``*.binetflow`` files
(``StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,
TotPkts,TotBytes,SrcBytes,Label``). Labels name the direction and the class:

=========================  ==========  ====================================
label contains             maps to     why
=========================  ==========  ====================================
``From-Botnet`` + ``CC``   ``c2``      the bot's command-and-control channel
``From-Botnet``            ``botnet``  other traffic the bot originated
``To-Botnet``              ``unknown`` sent *to* the bot by others; the
                                       capture README says not necessarily
                                       malicious
``To-Normal``              ``unknown`` sent *to* a verified normal host by an
                                       unverified source: the mirror of
                                       To-Botnet, and just as unverified
``Normal`` (From-, bare)   ``benign``  traffic from hosts the authors
                                       verified; their README calls server
                                       labels less reliable, so confidence
                                       is medium
``Background`` / other     ``unknown`` unverified university traffic; never
                                       benign (DATA-001)
=========================  ==========  ====================================

A case is one conversation: every flow with the same (scenario, source,
destination, destination port, protocol). Its gold is the class every
labelled flow in it agrees on; a conversation whose labelled flows disagree,
or that has none, is ``unknown``. ``unknown`` is kept, not dropped, so a
report can say how often a model acted on traffic nobody verified (review
9.3 B2). ``family_id`` is the scenario: captures are split whole, so one
botnet's flows never sit on both sides of a split.

Background traffic dominates the captures (tens of millions of flows). Every
conversation with a Botnet or Normal flow is kept; Background-only ones are
sampled deterministically by a hash of their key, so a rebuild picks the
same ones. Times are the capture's local wall clock (the source does not
state a zone); only intervals between flows are used.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import tarfile
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from .records import case_id

__all__ = ["Flow", "build_cases", "iter_flows", "map_label"]

SOURCE_ID = "ctu-13"
TASK_ID = "soc.ndr.technique.v1"
_CC = re.compile(r"(?:^|-)CC\d*(?:-|$)|C&C")
_TIME_FORMATS = ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S")


#: Behaviour tags in CTU-13 ``From-Botnet`` labels that match the labelbook's
#: ``botnet`` (spam relay, click fraud, scanning or connection attempts, DDoS).
_BOT_BEHAVIOUR = re.compile(r"SPAM|SMTP|Proxy|HTTP-Ad|Click|Attempt|ICMP|DDoS|Scan", re.IGNORECASE)
_DNS = re.compile(r"(?:^|-)(?:UDP-)?(?:Attempt-)?DNS(?:-|$)")

#: How a ``From-Botnet`` flow is labelled.
#: ``behaviour`` (default): by what the flow does, as the labelbook defines the
#:   classes. C&C -> ``c2``; spam, click-fraud ads, connection attempts, ICMP
#:   floods -> ``botnet``; DNS lookups and plain sessions from the infected host
#:   -> ``unknown`` (from a bot, not shown to be malicious; never benign).
#: ``host``: every flow from an infected host is ``botnet`` (or ``c2``), the
#:   mapping the G1 and NDR v1/v2 research used. CTU-13 labels hosts, so this
#:   calls a bot's ordinary DNS lookup "botnet traffic".
FLOW_LABEL_SEMANTICS = ("behaviour", "host")


def map_label(raw: str, semantics: str = "behaviour") -> tuple[str, str]:
    """``(class, confidence)`` for a CTU-13 flow label."""
    if semantics not in FLOW_LABEL_SEMANTICS:
        raise ValueError(f"semantics must be one of {FLOW_LABEL_SEMANTICS}, not {semantics!r}")
    label = raw.strip().removeprefix("flow=")
    if "From-Botnet" in label:
        if _CC.search(label):
            return "c2", "high"
        if semantics == "host":
            return "botnet", "high"
        if _DNS.search(label):
            return "unknown", "low"
        if _BOT_BEHAVIOUR.search(label):
            return "botnet", "high"
        return "unknown", "low"
    if "To-Botnet" in label or "To-Normal" in label:
        # The destination was verified; the source was not. Seen on the
        # published data as e.g. To-Normal-V45-UDP-NTP-server.
        return "unknown", "low"
    if "Normal" in label:
        return "benign", "medium"
    return "unknown", "low"


@dataclass(frozen=True)
class Flow:
    scenario: str
    start: float
    duration: float
    proto: str
    src: str
    sport: str
    dst: str
    dport: str
    state: str
    packets: int
    bytes: int
    label: str
    #: Where the flow came from: the archive member and its 1-based data row,
    #: so a case built from it can be traced back to the exact source line.
    source_file: str = ""
    row: int = 0


def _time(value: str) -> float | None:
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt).timestamp()
        except ValueError:
            continue
    return None


def _num(value: str, cast: type = float) -> Any:
    try:
        return cast(value)
    except (TypeError, ValueError):
        return cast(0)


def parse_binetflow(text: Iterable[str], scenario: str, source_file: str = "") -> Iterator[Flow]:
    reader = csv.DictReader(text)
    for n, row in enumerate(reader, start=1):
        start = _time(row.get("StartTime", ""))
        if start is None:
            continue
        yield Flow(
            scenario=scenario,
            start=start,
            duration=_num(row.get("Dur", "0")),
            proto=(row.get("Proto") or "").lower(),
            src=row.get("SrcAddr") or "",
            sport=row.get("Sport") or "",
            dst=row.get("DstAddr") or "",
            dport=row.get("Dport") or "",
            state=row.get("State") or "",
            packets=_num(row.get("TotPkts", "0"), int),
            bytes=_num(row.get("TotBytes", "0"), int),
            label=row.get("Label") or "",
            source_file=source_file,
            row=n,
        )


def _scenario_of(path: str) -> str:
    """The archive's per-scenario directory (``1`` .. ``13``), else the file stem."""
    parts = [p for p in path.split("/") if p]
    for part in reversed(parts[:-1]):
        if part.isdigit():
            return part
    return parts[-1].split(".")[0]


def iter_flows(path: Path) -> Iterator[Flow]:
    """Flows from a ``.binetflow`` file, a directory of them, or the CTU-13 tarball."""
    if path.is_dir():
        for f in sorted(path.rglob("*.binetflow")):
            with f.open(encoding="utf-8", errors="replace", newline="") as fh:
                rel = str(f.relative_to(path))
                yield from parse_binetflow(fh, _scenario_of(rel), rel)
        return
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            for member in tf:
                if not (member.isfile() and member.name.endswith(".binetflow")):
                    continue
                raw = tf.extractfile(member)
                if raw is None:
                    continue
                text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
                yield from parse_binetflow(text, _scenario_of(member.name), member.name)
        return
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        yield from parse_binetflow(fh, path.stem.split(".")[0], path.name)


@dataclass
class _Conversation:
    key: tuple[str, str, str, str, str]
    flows: int = 0
    packets: int = 0
    bytes: int = 0
    first: float = math.inf
    last: float = -math.inf
    starts: list[float] = field(default_factory=list)
    states: Counter[str] = field(default_factory=Counter)
    classes: Counter[str] = field(default_factory=Counter)
    confidence: dict[str, str] = field(default_factory=dict)
    sample: list[Flow] = field(default_factory=list)
    #: Any flow labelled From-Botnet, whatever it was mapped to. A conversation
    #: with one is never benign: under behaviour labels a bot's DNS lookup is
    #: ``unknown``, and it must not leave a mixed conversation looking clean.
    from_bot: bool = False


def _keep(key: tuple[str, ...], rate: float) -> bool:
    h = int(hashlib.sha1("\x00".join(key).encode(), usedforsecurity=False).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < rate


def build_cases(
    flows: Iterable[Flow],
    *,
    source_hash: str,
    background_rate: float = 0.001,
    max_cases_per_class: int = 300,
    max_evidence: int = 10,
    flow_labels: str = "behaviour",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Aggregate flows into conversations and attach flow-label gold."""
    convs: dict[tuple[str, str, str, str, str], _Conversation] = {}
    seen_flows: Counter[str] = Counter()
    for f in flows:
        cls, conf = map_label(f.label, flow_labels)
        seen_flows[cls] += 1
        key = (f.scenario, f.src, f.dst, f.dport, f.proto)
        conv = convs.get(key)
        if conv is None:
            if cls == "unknown" and not _keep(key, background_rate):
                continue
            conv = convs[key] = _Conversation(key)
        conv.flows += 1
        conv.packets += f.packets
        conv.bytes += f.bytes
        conv.first = min(conv.first, f.start)
        conv.last = max(conv.last, f.start + f.duration)
        if len(conv.starts) < 5000:
            conv.starts.append(f.start)
        conv.states[f.state] += 1
        conv.from_bot = conv.from_bot or "From-Botnet" in f.label
        if cls != "unknown":
            conv.classes[cls] += 1
            conv.confidence[cls] = conf
        if len(conv.sample) < max_evidence:
            conv.sample.append(f)

    def verdict(c: _Conversation) -> tuple[str, str]:
        labelled = [k for k in c.classes if c.classes[k]]
        if labelled == ["benign"] and c.from_bot:
            return "unknown", "low"
        if len(labelled) == 1:
            return labelled[0], c.confidence[labelled[0]]
        if {"c2", "botnet"} >= set(labelled) and labelled:
            return "c2", "high"  # a bot conversation that includes its C&C
        return "unknown", "low"

    # Capped per (scenario, class), not per class: splits are by scenario, and
    # on the published data a global cap gave one scenario over half of all
    # benign cases and another most of the C&C.
    by_class: dict[tuple[str, str], list[_Conversation]] = {}
    for c in convs.values():
        by_class.setdefault((c.key[0], verdict(c)[0]), []).append(c)
    kept: list[_Conversation] = []
    for _key, items in sorted(by_class.items()):
        # Deterministic cap: smallest key hashes win, so a rebuild keeps the same set.
        items.sort(
            key=lambda c: hashlib.sha1(
                "\x00".join(c.key).encode(), usedforsecurity=False
            ).hexdigest()
        )
        kept.extend(items[:max_cases_per_class])
    kept.sort(key=lambda c: c.key)

    cases: list[dict[str, Any]] = []
    golds: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for n, c in enumerate(kept, start=1):
        scenario, src, dst, dport, proto = c.key
        cid = case_id("ctu13", n)
        starts = sorted(c.starts)
        gaps = [b - a for a, b in pairwise(starts) if b > a]
        mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
        cv = (
            (math.sqrt(sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)) / mean_gap)
            if gaps and mean_gap
            else 0.0
        )
        evidence = [
            {
                "id": f"e{i}",
                "field": "flow",
                "value": f"{f.proto} {f.src}:{f.sport} -> {f.dst}:{f.dport} {f.state} "
                f"{f.packets} pkts {f.bytes} B {f.duration:.3f}s",
            }
            for i, f in enumerate(c.sample, start=1)
        ]
        cases.append(
            {
                "case_id": cid,
                "family_id": f"ctu13-scenario-{scenario}",
                "source_dataset": SOURCE_ID,
                "source_hash": source_hash,
                "telemetry_origin": "lab-reproduction",
                "language": "n/a",
                "task_id": TASK_ID,
                "state": {
                    "proto": proto,
                    "src": src,
                    "dst": dst,
                    "dport": dport,
                    "flows": c.flows,
                    "total_packets": c.packets,
                    "total_bytes": c.bytes,
                    "duration_s": round(max(0.0, c.last - c.first), 3),
                    "mean_interval_s": round(mean_gap, 3),
                    "interval_cv": round(cv, 3),
                    "states": ", ".join(f"{k}:{v}" for k, v in c.states.most_common(5)),
                },
                "evidence": evidence,
                "split_group": f"ctu13-scenario-{scenario}",
                "license_reviewed": False,
                "pii_reviewed": False,
            }
        )
        cls, conf = verdict(c)
        counts[cls] += 1
        golds.append(
            {
                "case_id": cid,
                "task_id": TASK_ID,
                "label_origin": "flow-label",
                "gold": {
                    "verdict": cls,
                    "confidence_of_gold": conf,
                    "allowed_alternatives": ["botnet"]
                    if cls == "c2"
                    else (["c2"] if cls == "botnet" else []),
                    "attack_stage": None,
                },
                "adjudication_status": "not-applicable",
                "adjudication_note": (
                    "flow labels: " + ", ".join(f"{k}={v}" for k, v in sorted(c.classes.items()))
                    if c.classes
                    else "no labelled flow in this conversation (Background or To-Botnet); unknown, not benign"
                ),
            }
        )

    known = sum(v for k, v in counts.items() if k != "unknown")
    summary = {
        "source": SOURCE_ID,
        "task_id": TASK_ID,
        "label_origin": "flow-label",
        "trust_tier": "C",
        "cases": len(cases),
        "by_verdict": dict(counts),
        "known_label_cases": known,
        "unknown_label_cases": counts.get("unknown", 0),
        "flows_seen_by_class": dict(seen_flows),
        "families": sorted({c["family_id"] for c in cases}),
        "background_rate": background_rate,
        "max_cases_per_class": max_cases_per_class,
        "flow_labels": flow_labels,
        "caveat": (
            "unknown is Background or To-Botnet traffic: unverified, never benign. Report "
            "known-label metrics and actions taken on unknown traffic separately. 2011 traffic."
        ),
    }
    return cases, golds, summary
