"""IoT-23 (Stratosphere, 2018-2019) -> host-window fusion cases.

A second NDR source on a different network, years and device class from
CTU-13, so a result here says something CTU-13 alone cannot: whether a model
trained on one network's hosts judges *unseen* hosts elsewhere. (CTU-13 reuses
every infected and verified-normal host across its scenarios; the leakage audit
measures that.)

Input: Zeek ``conn.log.labeled`` files, one per capture. The last columns are
``tunnel_parents``, ``label`` and ``detailed-label``; most files pack them into
one tab field separated by three spaces, some use three tab fields, and the
honeypot captures write ``benign`` in lower case. All three are read.

Labels are per flow and behavioural: ``Benign``, or ``Malicious`` with a
behaviour (``C&C``, ``C&C-Torii``, ``C&C-FileDownload``, ``DDoS``, ``Attack``,
``PartOfAHorizontalPortScan``, ``FileDownload``, ...). Mapped like CTU-13's
behaviour labels: any ``C&C`` -> ``c2``; other Malicious -> ``botnet``;
Benign -> ``benign`` at flow level.

Host roles are decided per capture, before windows are built: a host that
sends any Malicious flow is ``infected``; an internal host in a honeypot
(benign-only) capture is ``normal``; every other host is ``unverified``. So
an infected device's quiet window is ``unknown`` and an unverified host is
never ``benign``.
"""

from __future__ import annotations

import ipaddress
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from .ctu13 import Flow

__all__ = ["SOURCE_ID", "classify", "host_roles", "internal", "iter_flows", "parse_conn_log"]

SOURCE_ID = "iot-23"


#: Inside the lab network: RFC 1918, link-local and IPv6 unique-local/link-local.
#: Not ``ipaddress.is_private``, which also counts documentation and other
#: reserved ranges as private.
_INTERNAL_NETS = tuple(
    ipaddress.ip_network(n)
    for n in (
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "169.254.0.0/16",
        "fc00::/7",
        "fe80::/10",
    )
)


def internal(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return any(ip.version == n.version and ip in n for n in _INTERNAL_NETS)


def _scenario(name: str) -> str:
    """``CTU-IoT-Malware-Capture-34-1.conn.log.labeled`` -> ``malware-34-1``."""
    stem = name.split(".")[0]
    kind = "honeypot" if "Honeypot" in stem else "malware"
    return f"{kind}-" + "-".join(stem.split("-")[-2:])


def _float(v: str) -> float:
    try:
        return float(v)
    except ValueError:
        return 0.0


def _int(v: str) -> int:
    try:
        return int(v)
    except ValueError:
        return 0


def parse_conn_log(path: Path) -> Iterator[Flow]:
    scenario = _scenario(path.name)
    fields: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        row = 0
        for line in fh:
            if line.startswith("#fields"):
                fields = line.rstrip("\n").split("\t")[1:]
                continue
            if line.startswith("#") or not line.strip():
                continue
            row += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 23:
                label, detail = parts[-2], parts[-1]
            else:
                tail = parts[-1].split()
                label, detail = (tail[1], tail[2]) if len(tail) >= 3 else ("", "")
            rec = dict(zip(fields[:20], parts[:20], strict=False))
            try:
                start = float(rec.get("ts", ""))
            except ValueError:
                continue
            yield Flow(
                scenario=scenario,
                start=start,
                duration=_float(rec.get("duration", "0")),
                proto=rec.get("proto", "").lower(),
                src=rec.get("id.orig_h", ""),
                sport=rec.get("id.orig_p", ""),
                dst=rec.get("id.resp_h", ""),
                dport=rec.get("id.resp_p", ""),
                state=rec.get("conn_state", ""),
                packets=_int(rec.get("orig_pkts", "0")) + _int(rec.get("resp_pkts", "0")),
                bytes=_int(rec.get("orig_ip_bytes", "0")) + _int(rec.get("resp_ip_bytes", "0")),
                label=f"{label.strip().capitalize()}|{detail.strip()}",
                source_file=path.name,
                row=row,
            )


def iter_flows(path: Path) -> Iterator[Flow]:
    """Every capture under ``path`` (a directory) or one file, in name order."""
    files = sorted(path.glob("*.conn.log.labeled")) if path.is_dir() else [path]
    for f in files:
        yield from parse_conn_log(f)


def classify(f: Flow) -> tuple[str | None, str]:
    """``(host label, behaviour)`` for an IoT-23 flow."""
    label, _, detail = f.label.partition("|")
    if label == "Malicious":
        cls = "c2" if "C&C" in detail else "botnet"
        return cls, cls
    if label == "Benign":
        return "benign", "benign"
    return None, "unknown"


def host_roles(path: Path) -> dict[tuple[str, str], str]:
    """Per (capture, host): ``infected`` / ``normal`` / ``unverified`` (see the module docstring)."""
    malicious: dict[str, set[str]] = defaultdict(set)
    seen: dict[str, set[str]] = defaultdict(set)
    any_malicious: dict[str, bool] = defaultdict(bool)
    for f in iter_flows(path):
        seen[f.scenario].add(f.src)
        if f.label.startswith("Malicious"):
            malicious[f.scenario].add(f.src)
            any_malicious[f.scenario] = True
    roles: dict[tuple[str, str], str] = {}
    for scenario, hosts in seen.items():
        honeypot = scenario.startswith("honeypot") and not any_malicious[scenario]
        for h in hosts:
            if h in malicious[scenario]:
                roles[(scenario, h)] = "infected"
            elif honeypot and internal(h):
                roles[(scenario, h)] = "normal"
            else:
                roles[(scenario, h)] = "unverified"
    return roles
