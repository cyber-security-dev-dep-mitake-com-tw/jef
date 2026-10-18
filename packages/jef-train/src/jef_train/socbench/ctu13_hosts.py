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
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from .ctu13 import _CC, Flow
from .records import case_id

__all__ = ["HOST_TASK_ID", "build_host_windows", "host_label"]

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


def _mask(addr: str) -> str:
    return "<int>" if addr.startswith(_INTERNAL) else "<ext>"


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

    def add(self, f: Flow) -> None:
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
        cls = host_label(f.label)
        if cls:
            self.labels[cls] += 1
        self.examples.setdefault((f.proto, f.dport), f)

    def verdict(self) -> str:
        if self.labels.get("c2"):
            return "c2"
        if self.labels.get("botnet"):
            return "botnet"
        if self.labels.get("benign"):
            return "benign"
        return "unknown"


def _state(w: _Window, window_s: int) -> dict[str, Any]:
    starts = sorted(w.starts)
    gaps = [b - a for a, b in pairwise(starts) if b > a]
    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
    cv = (statistics.pstdev(gaps) / mean_gap) if len(gaps) > 1 and mean_gap else 0.0
    ext = sum(n for d, n in w.dsts.items() if not d.startswith(_INTERNAL))
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
    }


def build_host_windows(
    flows: Iterable[Flow],
    *,
    source_hash: str,
    window_s: int = 300,
    background_host_rate: float = 0.002,
    max_cases_per_class: int = 200,
    max_evidence: int = 8,
    min_flows: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Aggregate each source host's flows per window; gold from host-level labels."""
    labelled_hosts: dict[tuple[str, str], str] = {}
    windows: dict[tuple[str, str, int], _Window] = {}
    seen = Counter[str]()
    for f in flows:
        cls = host_label(f.label)
        key = (f.scenario, f.src)
        if cls:
            labelled_hosts.setdefault(key, cls)
        elif key not in labelled_hosts and _hash(*key) / 0xFFFFFFFF >= background_host_rate:
            continue
        idx = int(f.start // window_s)
        w = windows.get((f.scenario, f.src, idx))
        if w is None:
            w = windows[(f.scenario, f.src, idx)] = _Window(f.scenario, f.src, idx)
        w.add(f)
        seen[cls or "unlabelled"] += 1

    # A host sampled as Background before its first labelled flow arrived is
    # still that labelled host; its windows keep their own flow labels.
    by_class: dict[tuple[str, str], list[_Window]] = {}
    for w in windows.values():
        if w.flows < min_flows:
            continue
        by_class.setdefault((w.scenario, w.verdict()), []).append(w)
    kept: list[_Window] = []
    for _key, items in sorted(by_class.items()):
        items.sort(key=lambda w: _hash(w.scenario, w.host, str(w.index)))
        kept.extend(items[:max_cases_per_class])
    kept.sort(key=lambda w: (int(w.scenario) if w.scenario.isdigit() else 0, w.host, w.index))

    cases: list[dict[str, Any]] = []
    golds: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    hosts_by_class: dict[str, set[tuple[str, str]]] = {}
    for n, w in enumerate(kept, start=1):
        cid = case_id("ctu13h", n)
        shown = [w.examples[k] for k, _ in w.ports.most_common(max_evidence)]
        evidence = [
            {
                "id": f"e{i}",
                "field": "flow",
                "value": f"{f.proto} <host>:{f.sport} -> {_mask(f.dst)}:{f.dport} {f.state} "
                f"{f.packets} pkts {f.bytes} B {f.duration:.3f}s (x{w.ports[(f.proto, f.dport)]} to this port)",
            }
            for i, f in enumerate(shown, start=1)
        ]
        verdict = w.verdict()
        counts[verdict] += 1
        hosts_by_class.setdefault(verdict, set()).add((w.scenario, w.host))
        cases.append(
            {
                "case_id": cid,
                "family_id": f"ctu13-scenario-{w.scenario}",
                "source_dataset": SOURCE_ID,
                "source_hash": source_hash,
                "telemetry_origin": "lab-reproduction",
                "language": "n/a",
                "task_id": HOST_TASK_ID,
                "state": _state(w, window_s),
                "evidence": evidence,
                "split_group": f"ctu13-scenario-{w.scenario}",
                "license_reviewed": False,
                "pii_reviewed": False,
            }
        )
        refs = "; ".join(f"e{i}={f.source_file}#{f.row}" for i, f in enumerate(shown, start=1))
        golds.append(
            {
                "case_id": cid,
                "task_id": HOST_TASK_ID,
                "label_origin": LABEL_ORIGIN,
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
                    f"host {w.host}, scenario {w.scenario}, window {w.index} ({window_s}s); "
                    f"host labels in window: {dict(w.labels) or 'none (Background)'}; flows: {refs}"
                ),
            }
        )
    summary = {
        "source": SOURCE_ID,
        "task_id": HOST_TASK_ID,
        "label_origin": LABEL_ORIGIN,
        "trust_tier": "C",
        "unit": f"one source host over {window_s}s",
        "cases": len(cases),
        "by_verdict": dict(counts),
        "distinct_hosts_by_verdict": {k: len(v) for k, v in sorted(hosts_by_class.items())},
        "flows_seen_by_host_label": dict(seen),
        "families": sorted({c["family_id"] for c in cases}),
        "window_s": window_s,
        "background_host_rate": background_host_rate,
        "max_cases_per_class": max_cases_per_class,
        "caveat": (
            "botnet means the host is infected, not that every flow is malicious; c2 means a "
            "C&C-tagged flow in the window. Normal hosts are few and repeat across scenarios. "
            "Addresses are masked in the state and evidence; the gold note keeps them."
        ),
    }
    return cases, golds, summary
