"""Normalised Cloudflare requests -> a private, unlabelled incident-triage pack.

Requests from one pseudonymous actor with no gap longer than ``gap_minutes``
are one incident and one case, whichever routes they touch: a scanner sweeping
forty paths is one incident, not forty. The pack uses the
existing incident-triage format (``cases.jsonl``, ``expectations.jsonl``,
``manifest.json``) so it goes through ``load_pack``, the lock and, once a
person has labelled it, ``candidate-export`` -- with three differences that
keep it honest:

* every expectation is ``label_status: unlabelled`` with ``expected: null``.
  Scoring and export count such cases and never use them. Labels come only
  from the reviewer queue.
* the WAF's own outputs (action, source, rule) are written to ``weak.jsonl``
  (Tier D), which no scorer or exporter reads. By default the state a model or
  reviewer sees does not show them either, so a model cannot be credited for
  repeating Cloudflare's verdict; ``with_vendor_signals`` adds a second,
  marked variant of each case for the ablation that measures that gain.
* families are joined over the pseudonymous actor (across the whole pack) and
  over route family per ISO week, so one scanner or one campaign never lands
  in two splits. Approval twins and vendor variants share their case's family.

Approval evidence (change tickets, approved scanners, maintenance windows,
uptime monitors) comes from an optional context file. A case it matches is
written twice: with the evidence, and as a twin without it. The twin is a
separate unlabelled case; nothing copies a label from one to the other.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .cloudflare_normalize import Pseudonymizer, Request, route_family
from .cloudflare_source import _append_private, _write_private, check_alias, check_private_dir

__all__ = [
    "SOURCE_ID",
    "Incident",
    "build_cases",
    "group_incidents",
    "public_report",
    "write_private_pack",
]

SOURCE_ID = "cloudflare-own-zone"
CONTEXT_KINDS = frozenset(
    {"change_ticket", "approved_scanner", "maintenance_window", "uptime_monitor"}
)


def _dt(minute: str) -> datetime:
    return datetime.strptime(minute, "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)


@dataclass
class Incident:
    actor: str
    requests: list[Request] = field(default_factory=list)
    approvals: list[Mapping[str, Any]] = field(default_factory=list)

    @property
    def start(self) -> str:
        return self.requests[0].minute

    @property
    def end(self) -> str:
        return self.requests[-1].minute

    @property
    def key(self) -> str:
        return f"{self.actor}|{self.start}"

    @property
    def route_families(self) -> list[str]:
        return sorted({route_family(r.route) for r in self.requests})

    @property
    def week(self) -> str:
        y, w, _ = _dt(self.start).isocalendar()
        return f"{y}-W{w:02d}"


def group_incidents(requests: Iterable[Request], gap_minutes: int = 15) -> list[Incident]:
    """Sessionise each actor's requests with a maximum gap."""
    by_actor: dict[str, list[Request]] = defaultdict(list)
    for r in requests:
        by_actor[r.actor].append(r)
    out: list[Incident] = []
    gap = timedelta(minutes=gap_minutes)
    for actor, reqs in by_actor.items():
        reqs.sort(key=lambda q: (q.minute, q.request_id))
        current = Incident(actor, [reqs[0]])
        for r in reqs[1:]:
            if _dt(r.minute) - _dt(current.end) > gap:
                out.append(current)
                current = Incident(actor, [r])
            else:
                current.requests.append(r)
        out.append(current)
    out.sort(key=lambda i: (i.start, i.key))
    return out


def _families(incidents: Sequence[Incident], zone_alias: str) -> list[str]:
    """Union-find over shared actor (global) and shared route family per week."""
    parent = list(range(len(incidents)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    first: dict[tuple[str, str], int] = {}
    for i, inc in enumerate(incidents):
        keys = [("actor", inc.actor)] + [
            ("route-week", f"{fam}|{inc.week}") for fam in inc.route_families
        ]
        for k in keys:
            if k in first:
                union(first[k], i)
            else:
                first[k] = i
    members: dict[int, list[str]] = defaultdict(list)
    for i, inc in enumerate(incidents):
        members[find(i)].append(inc.key)
    names = {
        root: f"cf|{zone_alias}|"
        + hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()[:12]
        for root, keys in members.items()
    }
    return [names[find(i)] for i in range(len(incidents))]


def attach_context(
    incidents: Sequence[Incident], context: Iterable[Mapping[str, Any]], pseud: Pseudonymizer
) -> None:
    """Attach each approval record to the incidents it covers (actor, route, time)."""
    for rec in context:
        if rec.get("kind") not in CONTEXT_KINDS:
            raise ValueError(
                f"context kind {rec.get('kind')!r} is not one of {sorted(CONTEXT_KINDS)}"
            )
        match = rec.get("match") or {}
        actor = pseud("actor", str(match["actor_ip"])) if match.get("actor_ip") else None
        prefix = str(match.get("route_prefix") or "/")
        lo, hi = _parse(str(rec["start"])), _parse(str(rec["end"]))
        for inc in incidents:
            if actor and inc.actor != actor:
                continue
            if not all(r.route.startswith(prefix) for r in inc.requests):
                continue
            if _dt(inc.end) < lo or _dt(inc.start) > hi:
                continue
            inc.approvals.append(rec)


def _counts(values: Iterable[Any]) -> str:
    c = Counter(str(v) for v in values if v is not None)
    return ", ".join(f"{k}×{n}" if n > 1 else k for k, n in sorted(c.items())) or "none recorded"


def _sampling(inc: Incident) -> tuple[bool | str, float | None]:
    intervals = [r.sample_interval for r in inc.requests]
    known = [i for i in intervals if i is not None]
    if not known:
        return "unknown", None
    top = max(known)
    return (top > 1), top


def _render(
    inc: Incident, zone_alias: str, *, approvals: bool, vendor: bool
) -> tuple[str, list[dict[str, str]]]:
    reqs = inc.requests
    first = reqs[0]
    routes = sorted({r.route for r in reqs})
    keys = sorted({k for r in reqs for k in r.query_keys})
    events = sum(len(r.detections) for r in reqs)
    sampled, _ = _sampling(inc)
    evidence = [
        ("zone", zone_alias),
        ("actor", f"{inc.actor} (AS{first.asn}, {first.country})"),
        ("routes", ", ".join(routes)),
        ("window_utc", f"{inc.start} to {inc.end}"),
        ("requests", str(len(reqs))),
        ("methods", _counts(r.method for r in reqs)),
        ("origin_status", _counts(r.status for r in reqs)),
        ("client_type", _counts(r.ua_family for r in reqs)),
        ("query_parameters", ", ".join(keys) if keys else "none"),
        ("security_events", str(events)),
    ]
    lines = [
        f"Edge security telemetry for zone {zone_alias}.",
        f"{len(reqs)} request(s) from actor {inc.actor} (AS{first.asn}, {first.country}) "
        f"to {', '.join(routes)} between {inc.start} and {inc.end} UTC.",
        f"Methods: {_counts(r.method for r in reqs)}. "
        f"Origin responses: {_counts(r.status for r in reqs)}. "
        f"Client type: {_counts(r.ua_family for r in reqs)}.",
        "Query parameters seen (values removed): " + (", ".join(keys) if keys else "none") + ".",
        f"Security events recorded: {events}.",
    ]
    if sampled is True:
        lines.append("Events in this window were sampled; counts are not complete.")
    elif sampled == "unknown":
        lines.append("Whether these events were sampled is unknown.")
    if approvals and inc.approvals:
        refs = "; ".join(
            f"{a['kind']} {a.get('ref', '')}: {a.get('note', '')} ({a['start']} to {a['end']})".strip()
            for a in inc.approvals
        )
        lines.append(f"Approval or change evidence: {refs}.")
        evidence.append(("approval", refs))
    else:
        lines.append("Approval or change evidence: none provided.")
    if vendor:
        dets = [d for r in reqs for d in r.detections]
        signal = (
            f"actions {_counts(d['action'] for d in dets)}; "
            f"sources {_counts(d['source'] for d in dets)}; "
            f"rules {_counts(d['rule'] for d in dets)}"
        )
        lines.append(f"Detector output (Cloudflare): {signal}.")
        evidence.append(("detector_output", signal))
    return " ".join(lines), [
        {"id": f"e{n}", "field": f, "value": v} for n, (f, v) in enumerate(evidence, 1)
    ]


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_cases(
    requests: Sequence[Request],
    *,
    zone_alias: str,
    pseud: Pseudonymizer,
    context: Iterable[Mapping[str, Any]] = (),
    gap_minutes: int = 15,
    with_vendor_signals: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """``(cases, expectations, weak)`` for one zone; every expectation unlabelled."""
    check_alias(zone_alias)
    incidents = group_incidents(requests, gap_minutes)
    attach_context(incidents, context, pseud)
    families = _families(incidents, zone_alias)
    cases: list[dict[str, Any]] = []
    weak: list[dict[str, Any]] = []

    def emit(
        prefix: str,
        n: int,
        inc: Incident,
        family: str,
        *,
        approvals: bool,
        vendor: bool,
        **links: str,
    ) -> str:
        state, evidence = _render(inc, zone_alias, approvals=approvals, vendor=vendor)
        sampled, interval = _sampling(inc)
        cid = f"itc-{prefix}-{n:03d}"
        cases.append(
            {
                "case_id": cid,
                "case_family": family,
                "origin": "own-telemetry",
                "language": "en",
                "domain": "web",
                "state": state,
                "evidence": evidence,
                "licence_or_permission": (
                    "own zone telemetry, pseudonymised; private: data-owner approval needed "
                    "before any use outside this pack, never sent to external models"
                ),
                "derived_from": (
                    f"{SOURCE_ID}:firewallEventsAdaptive, zone alias {zone_alias}; "
                    "rendered from pseudonymised requests, not the original records"
                ),
                "state_sha256": _sha(state),
                "observed_at": inc.start,
                "observation": {
                    "source": SOURCE_ID,
                    "sampled": sampled,
                    "sample_interval": interval,
                    "requests": len(inc.requests),
                    "events": max(1, sum(len(r.detections) for r in inc.requests)),
                    "window": [inc.start, inc.end],
                },
                "vendor_signals": vendor,
                **links,
            }
        )
        dets = [d for r in inc.requests for d in r.detections]
        weak.append(
            {
                "case_id": cid,
                "source": SOURCE_ID,
                "trust_tier": "D",
                "note": "detector output, not a label",
                "actions": dict(Counter(str(d["action"]) for d in dets)),
                "sources": dict(Counter(str(d["source"]) for d in dets)),
                "rules": dict(Counter(str(d["rule"]) for d in dets if d["rule"])),
            }
        )
        return cid

    for n, (inc, family) in enumerate(zip(incidents, families, strict=True), 1):
        base = emit("cfwaf", n, inc, family, approvals=True, vendor=False)
        if inc.approvals:
            emit("cfwaft", n, inc, family, approvals=False, vendor=False, twin_of=base)
        if with_vendor_signals:
            emit("cfwafv", n, inc, family, approvals=True, vendor=True, variant_of=base)

    expectations = [
        {
            "case_id": c["case_id"],
            "label_origin": "none",
            "label_status": "unlabelled",
            "reviewers": [],
            "truth_of_event": "unknown",
            "truth_of_alert": "unknown",
            "expected": None,
            "prohibited_actions": [],
            "required_context": [],
        }
        for c in cases
    ]
    return cases, expectations, weak


def write_private_pack(
    out: Path,
    cases: Sequence[Mapping[str, Any]],
    expectations: Sequence[Mapping[str, Any]],
    weak: Sequence[Mapping[str, Any]],
) -> Path:
    """Write the pack owner-only, outside any git checkout, with its manifest."""
    from .triage import pack_manifest

    out = check_private_dir(out)
    for name, rows in (
        ("cases.jsonl", cases),
        ("expectations.jsonl", expectations),
        ("weak.jsonl", weak),
    ):
        _write_private(
            out / name,
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
        )
    _write_private(
        out / "manifest.json",
        json.dumps(pack_manifest(out), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return out


def write_requests(out: Path, requests: Sequence[Request]) -> Path:
    """Normalised requests (``requests.jsonl``), owner-only, outside any checkout."""
    out = check_private_dir(out)
    path = out / "requests.jsonl"
    _write_private(path, "")
    for r in requests:
        _append_private(path, json.dumps(r.to_json(), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def public_report(
    cases: Sequence[Mapping[str, Any]],
    weak: Sequence[Mapping[str, Any]],
    expectations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregates safe to publish: counts only, and no rate over sampled data.

    Detector counts come from the primary cases only (not twins or vendor
    variants, which repeat an incident). If any incident was sampled or its
    sampling is unknown, the denominator is unknown and no rate is given.
    """
    primary = {c["case_id"] for c in cases if not c.get("twin_of") and not c.get("variant_of")}
    sampled = Counter(str(c["observation"]["sampled"]) for c in cases if c["case_id"] in primary)
    actions: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    rules: Counter[str] = Counter()
    for w in weak:
        if w["case_id"] in primary:
            actions.update(w["actions"])
            sources.update(w["sources"])
            rules.update(w["rules"])
    complete = set(sampled) == {"False"}
    return {
        "evidence_class": "REAL TELEMETRY — UNREVIEWED AGGREGATES (detector outputs are not labels)",
        "incidents": len(primary),
        "cases_including_twins_and_variants": len(cases),
        "requests": sum(c["observation"]["requests"] for c in cases if c["case_id"] in primary),
        "security_events": sum(
            c["observation"]["events"] for c in cases if c["case_id"] in primary
        ),
        "sampling": dict(sampled),
        "denominator": "complete" if complete else "denominator_unknown",
        "rates": None,
        "rates_note": (
            "no detector rate is reported: detector outputs are not labels"
            + ("" if complete else ", and the events are sampled or of unknown sampling")
        ),
        "detector_actions": dict(actions),
        "detector_sources": dict(sources),
        "rules_seen": len(rules),
        "label_status": dict(Counter(str(e["label_status"]) for e in expectations)),
        "families": len({c["case_family"] for c in cases}),
    }
