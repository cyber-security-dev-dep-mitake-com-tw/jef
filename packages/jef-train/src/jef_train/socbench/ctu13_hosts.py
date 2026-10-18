"""CTU-13 -> host-window NDR cases: one source host over one time window.

Why not one conversation per case (``ctu13.build_cases``)? CTU-13's ground
truth is **host-level**. The authors labelled every flow from an infected
host ``From-Botnet``, including its ordinary DNS lookups and web requests, and
every flow from a verified normal host ``From-Normal``. A single flow summary
with its addresses masked cannot carry that truth: the NDR v2 root-cause
analysis found 34 of 36 confident mistakes were exactly such flows
(docs/validation/research/ndr-head-v2). The unit the labels describe is the
host, so the case is a host in a window, with the context a single flow lacks:
fan-out, port mix, DNS and SMTP volume, unanswered connections, periodicity.

Label semantics, stated per class:

=========  ==========================================================  ======
verdict    meaning                                                     level
=========  ==========================================================  ======
``c2``     the host is infected and, in this window, sends at least    flow
           one flow the authors tagged as command and control
``botnet`` the host is infected (its flows are ``From-Botnet``); no   host
           C&C-tagged flow in this window
``benign`` the host is one of the authors' verified normal hosts      host
``unknown`` any other host (Background): unverified, never benign     host
=========  ==========================================================  ======

``botnet`` therefore says "this host is a bot", not "every flow in this window
is malicious". Flow-level maliciousness is not in the source and is not
invented here; it needs reviewer labels.

Addresses are masked in what the scorer sees: the host is ``<host>``, every
destination is ``<int>`` (inside the capture's university network,
147.32.0.0/16) or ``<ext>``. CTU-13 reuses the same infected and normal IPs in
every scenario, so an unmasked address would be the label. Traceability lives
in the gold record, which no scorer reads: the host's real address, the window,
and the archive member and data row of every flow shown as evidence.

**Fusion** (``fusion=True``, ``gold="behaviour"``; ``jef-bench build --unit
host-fusion``) keeps the host window and adds what the review asked for:

* gold by *behaviour* inside the window, as the labelbook defines the classes:
  ``c2`` when a C&C-tagged flow is in the window, ``botnet`` when spam,
  click-fraud ads, connection attempts or floods are, ``unknown`` for an
  infected host whose window shows neither (its DNS and plain sessions), and
  ``benign`` for a verified-normal host. Label origin ``host-behaviour-label``.
  The behaviour tags decide the gold only; they never reach the state or the
  evidence, which would hand the scorer its label;
* conversation summaries as evidence (protocol/port, destination role, flow
  count, bytes, unanswered share, timing regularity) instead of single flows;
* a per-minute time series of flows and distinct destinations.

Provenance stays in the gold note: host address, window, host role, behaviour
counts, and the archive row of the first flow of every conversation shown.

Hosts are sampled by host, not by flow: every infected and normal host is
kept; Background hosts are kept when a hash of (scenario, host) falls under
``background_host_rate``. Windows are then capped per (scenario, verdict) by a
hash, so a rebuild picks the same cases.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from .ctu13 import _CC, Flow, map_label
from .records import case_id

__all__ = ["HOST_TASK_ID", "build_host_windows", "ctu13_classify", "host_label"]

SOURCE_ID = "ctu-13"
HOST_TASK_ID = "soc.ndr.host.v1"
LABEL_ORIGIN = "host-label"
_INTERNAL = "147.32."
_IRC = {str(p) for p in range(6660, 6670)} | {"6697"}
_WEB = {"80", "443", "8080"}


def host_label(raw: str) -> str | None:
    """``botnet``/``c2``/``benign`` for a flow the host *sent*, else ``None``."""
    label = raw.strip().removeprefix("flow=")
    if label.startswith("From-Botnet"):
        return "c2" if _CC.search(label) else "botnet"
    if label.startswith("From-Normal"):
        return "benign"
    return None


def _hash(*parts: str) -> int:
    return int(hashlib.sha1("\x00".join(parts).encode(), usedforsecurity=False).hexdigest()[:8], 16)


@dataclass
class _Conv:
    first: Flow
    flows: int = 0
    bytes: int = 0
    packets: int = 0
    short: int = 0
    starts: list[float] = field(default_factory=list)


def _cv(starts: list[float]) -> float:
    s = sorted(starts)
    gaps = [b - a for a, b in pairwise(s) if b > a]
    mean = sum(gaps) / len(gaps) if gaps else 0.0
    return (statistics.pstdev(gaps) / mean) if len(gaps) > 1 and mean else 0.0


def ctu13_classify(f: Flow) -> tuple[str | None, str]:
    """``(host label, behaviour)`` for a CTU-13 flow: host-level and behaviour-level."""
    return host_label(f.label), map_label(f.label, "behaviour")[0]


@dataclass
class _Window:
    scenario: str
    host: str
    index: int
    flows: int = 0
    packets: int = 0
    bytes: int = 0
    short: int = 0
    durations: list[float] = field(default_factory=list)
    starts: list[float] = field(default_factory=list)
    dsts: Counter[str] = field(default_factory=Counter)
    ports: Counter[tuple[str, str]] = field(default_factory=Counter)
    protos: Counter[str] = field(default_factory=Counter)
    states: Counter[str] = field(default_factory=Counter)
    labels: Counter[str] = field(default_factory=Counter)
    examples: dict[tuple[str, str], Flow] = field(default_factory=dict)
    # Fusion only (cheap to keep; not shown unless fusion=True).
    behaviour: Counter[str] = field(default_factory=Counter)
    minute_flows: Counter[int] = field(default_factory=Counter)
    minute_dsts: dict[int, set[str]] = field(default_factory=dict)
    convs: dict[tuple[str, str, str], _Conv] = field(default_factory=dict)

    def add(
        self,
        f: Flow,
        host_cls: str | None = None,
        behaviour: str | None = None,
        window_s: int = 300,
    ) -> None:
        self.flows += 1
        self.packets += f.packets
        self.bytes += f.bytes
        self.short += f.packets <= 2
        if len(self.starts) < 20000:
            self.starts.append(f.start)
            self.durations.append(f.duration)
        if len(self.dsts) < 20000 or f.dst in self.dsts:
            self.dsts[f.dst] += 1
        self.ports[(f.proto, f.dport)] += 1
        self.protos[f.proto] += 1
        self.states[f.state] += 1
        cls = host_label(f.label) if host_cls is None else host_cls
        if cls:
            self.labels[cls] += 1
        self.examples.setdefault((f.proto, f.dport), f)
        if behaviour is not None:
            self.behaviour[behaviour] += 1
        minute = int((f.start - self.index * window_s) // 60)
        self.minute_flows[minute] += 1
        bucket = self.minute_dsts.setdefault(minute, set())
        if len(bucket) < 5000:
            bucket.add(f.dst)
        ck = (f.proto, f.dport, f.dst)
        conv = self.convs.get(ck)
        if conv is None:
            if len(self.convs) >= 5000:
                return
            conv = self.convs[ck] = _Conv(first=f)
        conv.flows += 1
        conv.bytes += f.bytes
        conv.packets += f.packets
        conv.short += f.packets <= 2
        if len(conv.starts) < 500:
            conv.starts.append(f.start)

    def behaviour_verdict(self, role: str | None) -> str:
        if self.behaviour.get("c2"):
            return "c2"
        if self.behaviour.get("botnet"):
            return "botnet"
        if role == "infected":
            return "unknown"  # an infected host whose window shows no attack behaviour
        if role == "normal":
            return "benign"
        return "unknown"

    def verdict(self) -> str:
        if self.labels.get("c2"):
            return "c2"
        if self.labels.get("botnet"):
            return "botnet"
        if self.labels.get("benign"):
            return "benign"
        return "unknown"


def _state(
    w: _Window,
    window_s: int,
    *,
    fusion: bool = False,
    internal: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    starts = sorted(w.starts)
    gaps = [b - a for a, b in pairwise(starts) if b > a]
    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
    cv = (statistics.pstdev(gaps) / mean_gap) if len(gaps) > 1 and mean_gap else 0.0
    is_int = internal or (lambda d: d.startswith(_INTERNAL))
    ext = sum(n for d, n in w.dsts.items() if not is_int(d))
    by_port = Counter({f"{p}/{d}": n for (p, d), n in w.ports.items()})
    port_n = sum(by_port.values())
    entropy = (
        -sum((n / port_n) * math.log2(n / port_n) for n in by_port.values()) if port_n else 0.0
    )
    return {
        "window_s": window_s,
        "flows": w.flows,
        "flows_per_min": round(w.flows / (window_s / 60), 2),
        "distinct_dsts": len(w.dsts),
        "external_dst_share": round(ext / w.flows, 3) if w.flows else 0.0,
        "max_flows_to_one_dst": max(w.dsts.values(), default=0),
        "distinct_dst_ports": len(w.ports),
        "dst_port_entropy_bits": round(abs(entropy), 3),  # abs: a one-port window is 0.0, not -0.0
        "top_dst_ports": ", ".join(f"{k}:{v}" for k, v in by_port.most_common(6)),
        "protocols": ", ".join(f"{k}:{v}" for k, v in w.protos.most_common(4)),
        "dns_flows": sum(n for (p, d), n in w.ports.items() if d == "53"),
        "smtp_flows": sum(n for (p, d), n in w.ports.items() if d in {"25", "587", "465"}),
        "web_flows": sum(n for (p, d), n in w.ports.items() if d in _WEB),
        "irc_flows": sum(n for (p, d), n in w.ports.items() if d in _IRC),
        "unanswered_share": round(w.short / w.flows, 3) if w.flows else 0.0,
        "total_bytes": w.bytes,
        "total_packets": w.packets,
        "median_flow_duration_s": round(statistics.median(w.durations), 3) if w.durations else 0.0,
        "interval_cv": round(cv, 3),
        "top_states": ", ".join(f"{k}:{v}" for k, v in w.states.most_common(4)),
    } | (
        {
            "flows_by_minute": ",".join(
                str(w.minute_flows.get(m, 0)) for m in range(window_s // 60)
            ),
            "distinct_dsts_by_minute": ",".join(
                str(len(w.minute_dsts.get(m, ()))) for m in range(window_s // 60)
            ),
            "conversations": len(w.convs),
            "busiest_conversation_share": round(
                max((c.flows for c in w.convs.values()), default=0) / w.flows, 3
            )
            if w.flows
            else 0.0,
        }
        if fusion
        else {}
    )


def build_host_windows(
    flows: Iterable[Flow],
    *,
    source_hash: str,
    window_s: int = 300,
    background_host_rate: float = 0.002,
    max_cases_per_class: int = 200,
    max_evidence: int = 8,
    min_flows: int = 3,
    gold: str = "host",
    fusion: bool = False,
    classify: Callable[[Flow], tuple[str | None, str]] | None = None,
    host_roles: Mapping[tuple[str, str], str] | None = None,
    internal: Callable[[str], bool] | None = None,
    source_id: str = SOURCE_ID,
    case_tag: str | None = None,
    family_prefix: str = "ctu13-scenario-",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Aggregate each source host's flows per window.

    ``gold="host"`` (default): host-level labels, ``label_origin`` ``host-label``.
    ``gold="behaviour"``: behaviour inside the window (see the module docstring),
    ``label_origin`` ``host-behaviour-label``. ``fusion=True`` adds conversation
    evidence and per-minute series. ``classify`` returns ``(host label,
    behaviour)`` per flow (default: CTU-13 labels); ``host_roles`` overrides the
    host's role (``infected``/``normal``) when it is known per capture, as for
    IoT-23; ``internal`` decides which destinations are inside the network.
    """
    if gold not in {"host", "behaviour"}:
        raise ValueError(f"gold must be host or behaviour, not {gold!r}")
    classify_fn = classify or ctu13_classify
    is_int = internal or (lambda d: d.startswith(_INTERNAL))
    tag = case_tag or ("ctu13f" if fusion else "ctu13h")
    origin = LABEL_ORIGIN if gold == "host" else "host-behaviour-label"
    labelled_hosts: dict[tuple[str, str], str] = {}
    windows: dict[tuple[str, str, int], _Window] = {}
    seen = Counter[str]()
    for f in flows:
        cls, behaviour = classify_fn(f)
        key = (f.scenario, f.src)
        known_role = host_roles is not None and key in host_roles
        if cls:
            labelled_hosts.setdefault(key, cls)
        elif (
            not known_role
            and key not in labelled_hosts
            and _hash(*key) / 0xFFFFFFFF >= background_host_rate
        ):
            continue
        idx = int(f.start // window_s)
        w = windows.get((f.scenario, f.src, idx))
        if w is None:
            w = windows[(f.scenario, f.src, idx)] = _Window(f.scenario, f.src, idx)
        w.add(f, cls, behaviour, window_s)
        seen[cls or "unlabelled"] += 1

    def role_of(w: _Window) -> str | None:
        key = (w.scenario, w.host)
        if host_roles is not None and key in host_roles:
            return host_roles[key]
        lab = labelled_hosts.get(key)
        return "infected" if lab in {"botnet", "c2"} else "normal" if lab == "benign" else None

    def verdict_of(w: _Window) -> str:
        return w.verdict() if gold == "host" else w.behaviour_verdict(role_of(w))

    # A host sampled as Background before its first labelled flow arrived is
    # still that labelled host; its windows keep their own flow labels.
    by_class: dict[tuple[str, str], list[_Window]] = {}
    for w in windows.values():
        if w.flows < min_flows:
            continue
        by_class.setdefault((w.scenario, verdict_of(w)), []).append(w)
    kept: list[_Window] = []
    for _key, items in sorted(by_class.items()):
        items.sort(key=lambda w: _hash(w.scenario, w.host, str(w.index)))
        kept.extend(items[:max_cases_per_class])
    kept.sort(
        key=lambda w: (int(w.scenario) if w.scenario.isdigit() else 0, w.scenario, w.host, w.index)
    )

    def mask(addr: str) -> str:
        return "<int>" if is_int(addr) else "<ext>"

    cases: list[dict[str, Any]] = []
    golds: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    hosts_by_class: dict[str, set[tuple[str, str]]] = {}
    for n, w in enumerate(kept, start=1):
        cid = case_id(tag, n)
        if fusion:
            convs = sorted(
                w.convs.values(),
                key=lambda c: (-c.flows, c.first.proto, c.first.dport, c.first.dst),
            )
            shown_convs = convs[:max_evidence]
            evidence = [
                {
                    "id": f"e{i}",
                    "field": "conversation",
                    "value": f"{c.first.proto}/{c.first.dport} -> {mask(c.first.dst)} x{c.flows} flows, "
                    f"{c.bytes} B, {c.packets} pkts, unanswered {c.short / c.flows:.2f}, "
                    f"interval cv {_cv(c.starts):.2f}",
                }
                for i, c in enumerate(shown_convs, start=1)
            ]
            shown = [c.first for c in shown_convs]
        else:
            shown = [w.examples[k] for k, _ in w.ports.most_common(max_evidence)]
            evidence = [
                {
                    "id": f"e{i}",
                    "field": "flow",
                    "value": f"{f.proto} <host>:{f.sport} -> {mask(f.dst)}:{f.dport} {f.state} "
                    f"{f.packets} pkts {f.bytes} B {f.duration:.3f}s (x{w.ports[(f.proto, f.dport)]} to this port)",
                }
                for i, f in enumerate(shown, start=1)
            ]
        verdict = verdict_of(w)
        counts[verdict] += 1
        hosts_by_class.setdefault(verdict, set()).add((w.scenario, w.host))
        cases.append(
            {
                "case_id": cid,
                "family_id": f"{family_prefix}{w.scenario}",
                "source_dataset": source_id,
                "source_hash": source_hash,
                "telemetry_origin": "lab-reproduction",
                "language": "n/a",
                "task_id": HOST_TASK_ID,
                "state": _state(w, window_s, fusion=fusion, internal=internal),
                "evidence": evidence,
                "split_group": f"{family_prefix}{w.scenario}",
                "license_reviewed": False,
                "pii_reviewed": False,
            }
        )
        refs = "; ".join(f"e{i}={f.source_file}#{f.row}" for i, f in enumerate(shown, start=1))
        extra = (
            f"host role: {role_of(w) or 'unverified'}; flow behaviour in window: {dict(w.behaviour)}; "
            if gold == "behaviour"
            else ""
        )
        golds.append(
            {
                "case_id": cid,
                "task_id": HOST_TASK_ID,
                "label_origin": origin,
                "gold": {
                    "verdict": verdict,
                    "confidence_of_gold": {"c2": "high", "botnet": "high", "benign": "medium"}.get(
                        verdict, "low"
                    ),
                    "allowed_alternatives": ["botnet"]
                    if verdict == "c2"
                    else (["c2"] if verdict == "botnet" else []),
                    "attack_stage": None,
                },
                "adjudication_status": "not-applicable",
                "adjudication_note": (
                    f"host {w.host}, scenario {w.scenario}, window {w.index} ({window_s}s); {extra}"
                    f"host labels in window: {dict(w.labels) or 'none (Background)'}; "
                    f"{'conversations' if fusion else 'flows'}: {refs}"
                ),
            }
        )
    summary = {
        "source": source_id,
        "task_id": HOST_TASK_ID,
        "label_origin": origin,
        "trust_tier": "C",
        "unit": f"one source host over {window_s}s" + (" (fusion)" if fusion else ""),
        "cases": len(cases),
        "by_verdict": dict(counts),
        "distinct_hosts_by_verdict": {k: len(v) for k, v in sorted(hosts_by_class.items())},
        "flows_seen_by_host_label": dict(seen),
        "families": sorted({c["family_id"] for c in cases}),
        "window_s": window_s,
        "background_host_rate": background_host_rate,
        "max_cases_per_class": max_cases_per_class,
        "gold": gold,
        "fusion": fusion,
        "caveat": (
            "botnet means the host is infected, not that every flow is malicious; c2 means a "
            "C&C-tagged flow in the window. Normal hosts are few and repeat across scenarios. "
            "Addresses are masked in the state and evidence; the gold note keeps them."
        )
        if gold == "host"
        else (
            "behaviour gold: c2/botnet need C&C or attack-tagged flows in the window; an infected "
            "host's quiet window is unknown. Tags decide the gold only and never reach the state or "
            "evidence. Hosts repeat across scenarios (see the leakage audit)."
        ),
    }
    return cases, golds, summary
