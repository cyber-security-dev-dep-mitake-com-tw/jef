"""Leakage audit for scenario-held-out benchmarks.

Splitting by scenario stops a case from sitting on both sides of a split. It
does not stop three quieter leaks, and this module measures them:

* **Host reuse.** CTU-13 replays the same infected and verified-normal hosts
  in many scenarios. With addresses masked a scorer cannot read the host, but
  it can still learn that host's habits. ``host_reuse`` reports, per class,
  how many hosts occur in more than one scenario, and per held-out scenario
  the share of its cases whose host also occurs in another scenario.
* **Near-duplicate states.** Exact duplicates are removed before locking
  (``dedupe_states``); coarse ones are not. ``near_duplicates`` buckets every
  numeric state value on a log2 scale and compares the resulting keys across
  scenarios: a held-out case whose key appears elsewhere *with the same
  label* is memorisation risk, *with only other labels* is label ambiguity.
* **How a label was decided.** For host windows, ``window_label_basis``
  reads the gold notes: how many C&C-tagged flows made a window ``c2``, and
  how many windows were labelled from a single flow.

Window overlap is not audited because it cannot leak here: a flow belongs to
exactly one window (by start time), and every window of a scenario is in the
same fold.
"""

from __future__ import annotations

import ast
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["audit", "host_of", "host_reuse", "near_duplicates", "window_label_basis"]

_HOST = re.compile(r"^host (\S+?),")
_LABELS = re.compile(r"host labels in window: (\{[^}]*\}|none)")


def host_of(case: Mapping[str, Any], gold: Mapping[str, Any]) -> str | None:
    """The source host of a case: from a host-window gold note, else the state's ``src``."""
    m = _HOST.match(str(gold.get("adjudication_note", "")))
    if m:
        return m.group(1)
    state = case.get("state")
    if isinstance(state, Mapping) and state.get("src"):
        return str(state["src"])
    return None


def host_reuse(
    cases: Sequence[Mapping[str, Any]], golds: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    gold_by = {g["case_id"]: g for g in golds}
    scen_of_host: dict[str, set[str]] = defaultdict(set)
    hosts_by_class: dict[str, set[str]] = defaultdict(set)
    rows = []
    for c in cases:
        g = gold_by[c["case_id"]]
        host = host_of(c, g)
        if host is None:
            continue
        verdict = g["gold"]["verdict"]
        scen_of_host[host].add(c["family_id"])
        hosts_by_class[verdict].add(host)
        rows.append((c["family_id"], host, verdict))
    per_class = {
        v: {
            "hosts": len(hs),
            "in_more_than_one_scenario": sum(len(scen_of_host[h]) > 1 for h in hs),
            "max_scenarios_per_host": max((len(scen_of_host[h]) for h in hs), default=0),
        }
        for v, hs in sorted(hosts_by_class.items())
    }
    per_fold: dict[str, Any] = {}
    for fam in sorted({r[0] for r in rows}):
        test = [r for r in rows if r[0] == fam]
        seen_elsewhere = [r for r in test if scen_of_host[r[1]] - {fam}]
        by_class = Counter(r[2] for r in seen_elsewhere)
        per_fold[fam] = {
            "cases": len(test),
            "host_also_in_training": len(seen_elsewhere),
            "share": round(len(seen_elsewhere) / len(test), 4) if test else None,
            "by_class": dict(by_class),
        }
    shares = [f["share"] for f in per_fold.values() if f["share"] is not None]
    return {
        "per_class": per_class,
        "per_held_out_scenario": per_fold,
        "median_share_of_test_cases_whose_host_is_in_training": round(
            sorted(shares)[len(shares) // 2], 4
        )
        if shares
        else None,
    }


def _coarse(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        if value <= 0:
            return 0
        return math.floor(math.log2(value + 1))
    return value


def near_duplicates(
    cases: Sequence[Mapping[str, Any]], golds: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    gold_by = {g["case_id"]: g["gold"]["verdict"] for g in golds}
    key_of: dict[str, tuple[Any, ...]] = {}
    where: dict[tuple[Any, ...], list[tuple[str, str]]] = defaultdict(list)
    for c in cases:
        state = c["state"] if isinstance(c["state"], Mapping) else {"state": c["state"]}
        key = (c["task_id"], tuple(sorted((k, _coarse(v)) for k, v in state.items())))
        key_of[c["case_id"]] = key
        where[key].append((c["family_id"], gold_by[c["case_id"]]))
    per_fold: dict[str, Any] = {}
    for fam in sorted({c["family_id"] for c in cases}):
        test = [c for c in cases if c["family_id"] == fam]
        same = ambiguous = 0
        for c in test:
            others = [lab for f, lab in where[key_of[c["case_id"]]] if f != fam]
            if not others:
                continue
            if gold_by[c["case_id"]] in others:
                same += 1
            else:
                ambiguous += 1
        per_fold[fam] = {
            "cases": len(test),
            "near_duplicate_same_label_elsewhere": same,
            "near_duplicate_other_label_only": ambiguous,
        }
    tot = sum(f["cases"] for f in per_fold.values())
    return {
        "bucketing": "log2 of every numeric state value; strings exact; evidence ignored",
        "per_held_out_scenario": per_fold,
        "share_same_label_elsewhere": round(
            sum(f["near_duplicate_same_label_elsewhere"] for f in per_fold.values()) / tot, 4
        )
        if tot
        else None,
        "share_other_label_only": round(
            sum(f["near_duplicate_other_label_only"] for f in per_fold.values()) / tot, 4
        )
        if tot
        else None,
    }


def window_label_basis(golds: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """For host windows: how many tagged flows decided each attack window's label."""
    c2_counts: list[int] = []
    attack_only_one = 0
    attack = 0
    for g in golds:
        m = _LABELS.search(str(g.get("adjudication_note", "")))
        if not m:
            continue
        labels = {} if m.group(1) == "none" else ast.literal_eval(m.group(1))
        verdict = g["gold"]["verdict"]
        if verdict == "c2":
            c2_counts.append(int(labels.get("c2", 0)))
        if verdict in {"c2", "botnet"}:
            attack += 1
            deciding = labels.get("c2", 0) if verdict == "c2" else labels.get("botnet", 0)
            attack_only_one += deciding == 1
    if not attack:
        return None
    c2_counts.sort()
    return {
        "attack_windows": attack,
        "decided_by_a_single_flow": attack_only_one,
        "c2_windows": len(c2_counts),
        "c2_flows_per_c2_window": {
            "median": c2_counts[len(c2_counts) // 2] if c2_counts else None,
            "exactly_one": sum(n == 1 for n in c2_counts),
        },
    }


def audit(cases: Sequence[Mapping[str, Any]], golds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(cases),
        "host_reuse": host_reuse(cases, golds),
        "near_duplicates": near_duplicates(cases, golds),
        "window_label_basis": window_label_basis(golds),
        "window_overlap": "not a leak by construction: one window per flow, one fold per scenario",
    }
