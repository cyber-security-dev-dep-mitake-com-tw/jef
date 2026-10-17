"""Family-grouped splits and the gold lock.

A split is assigned per ``family_id``, never per case: a paired hard negative,
a translated twin and every alert of one attack chain share a family, and a
family on both sides of a split is leakage (DATA-003). The lock records the
assignment and the SHA-256 of the exact case and gold files it was made from,
and refuses to be written while any check fails, so "the test set" always
means one set of bytes.

Checks before locking:

* every case has exactly one gold record, and every gold record a case;
* no family appears in two splits;
* no ``paired_with`` partner sits in a different split;
* no two cases in different splits have the same state (normalised JSON).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from .annotate import NEGATIVE_LABELS
from .records import trust_tier

__all__ = [
    "LockError",
    "assign_splits",
    "check_leakage",
    "digest_records",
    "make_lock",
    "verify_lock",
]

LOCK_VERSION = 1
DEFAULT_RATIOS: Mapping[str, float] = {"train": 0.6, "calibration": 0.2, "test": 0.2}


class LockError(ValueError):
    pass


def digest_records(records: Iterable[Mapping[str, Any]]) -> str:
    """Order-independent digest: sorted canonical JSON lines."""
    lines = sorted(
        json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for r in records
    )
    return "sha256:" + hashlib.sha256("\n".join(lines).encode()).hexdigest()


def assign_splits(
    cases: Iterable[Mapping[str, Any]],
    *,
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
    seed: str = "socbench",
    pinned: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """``family_id -> split`` by a seeded hash; ``pinned`` overrides per family."""
    total = sum(ratios.values())
    if total <= 0:
        raise LockError("split ratios must sum to something positive")
    bounds: list[tuple[float, str]] = []
    acc = 0.0
    for name, r in ratios.items():
        acc += r / total
        bounds.append((acc, name))
    out: dict[str, str] = {}
    for fam in sorted({c["family_id"] for c in cases}):
        if pinned and fam in pinned:
            out[fam] = pinned[fam]
            continue
        x = int(hashlib.sha256(f"{seed}\x00{fam}".encode()).hexdigest()[:12], 16) / float(16**12)
        out[fam] = next(name for edge, name in bounds if x < edge or edge == bounds[-1][0])
    return out


def _state_key(case: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(case["state"], sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def check_leakage(cases: Sequence[Mapping[str, Any]], family_split: Mapping[str, str]) -> list[str]:
    problems: list[str] = []
    split_of = {c["case_id"]: family_split[c["family_id"]] for c in cases}
    fam_splits: dict[str, set[str]] = defaultdict(set)
    for c in cases:
        fam_splits[c["family_id"]].add(split_of[c["case_id"]])
    for fam, splits in sorted(fam_splits.items()):
        if len(splits) > 1:
            problems.append(f"family {fam} is in {sorted(splits)}")
    for c in cases:
        partner = c.get("paired_with")
        if partner and partner in split_of and split_of[partner] != split_of[c["case_id"]]:
            problems.append(
                f"{c['case_id']} is in {split_of[c['case_id']]} but its pair {partner} is in {split_of[partner]}"
            )
    states: dict[str, set[str]] = defaultdict(set)
    first: dict[str, str] = {}
    for c in cases:
        key = _state_key(c)
        states[key].add(split_of[c["case_id"]])
        first.setdefault(key, c["case_id"])
    for key, splits in states.items():
        if len(splits) > 1:
            problems.append(f"identical state (e.g. {first[key]}) appears in {sorted(splits)}")
    return problems


def make_lock(
    cases: Sequence[Mapping[str, Any]],
    golds: Sequence[Mapping[str, Any]],
    *,
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
    seed: str = "socbench",
    pinned: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    case_ids = [c["case_id"] for c in cases]
    gold_ids = [g["case_id"] for g in golds]
    problems: list[str] = []
    for name, ids in (("case", case_ids), ("gold", gold_ids)):
        dupes = [k for k, v in Counter(ids).items() if v > 1]
        if dupes:
            problems.append(f"duplicate {name} ids: {dupes[:5]}")
    missing_gold = sorted(set(case_ids) - set(gold_ids))
    orphan_gold = sorted(set(gold_ids) - set(case_ids))
    if missing_gold:
        problems.append(f"{len(missing_gold)} case(s) have no gold, e.g. {missing_gold[:3]}")
    if orphan_gold:
        problems.append(f"{len(orphan_gold)} gold record(s) have no case, e.g. {orphan_gold[:3]}")
    family_split = assign_splits(cases, ratios=ratios, seed=seed, pinned=pinned)
    problems += check_leakage(cases, family_split)
    if problems:
        raise LockError("refusing to lock:\n  " + "\n  ".join(problems))

    gold_by = {g["case_id"]: g for g in golds}
    splits: dict[str, list[str]] = defaultdict(list)
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    negatives: dict[str, list[int]] = defaultdict(
        lambda: [0, 0]
    )  # split -> [negatives, hard negatives]
    for c in sorted(cases, key=lambda c: c["case_id"]):
        split = family_split[c["family_id"]]
        splits[split].append(c["case_id"])
        g = gold_by[c["case_id"]]
        verdict = g["gold"]["verdict"]
        counts[split]["verdict"][verdict] += 1
        counts[split]["tier"][trust_tier(g["label_origin"])] += 1
        counts[split]["task"][g["task_id"]] += 1
        counts[split]["language"][str(c.get("language"))] += 1
        if verdict in NEGATIVE_LABELS:
            negatives[split][0] += 1
            if c.get("hard_negative"):
                negatives[split][1] += 1
    return {
        "lock_version": LOCK_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "ratios": dict(ratios),
        "cases_digest": digest_records(cases),
        "gold_digest": digest_records(golds),
        "families": {fam: family_split[fam] for fam in sorted(family_split)},
        "splits": {k: sorted(v) for k, v in sorted(splits.items())},
        "counts": {s: {k: dict(v) for k, v in d.items()} for s, d in sorted(counts.items())},
        "hard_negative_share_of_negatives": {
            s: (round(h / n, 4) if n else None) for s, (n, h) in sorted(negatives.items())
        },
        "targets": {
            "hard_negative_share_of_negatives": [0.35, 0.45],
            "double_reviewed_test_cases_min": 300,
        },
    }


def verify_lock(
    lock: Mapping[str, Any], cases: Sequence[Mapping[str, Any]], golds: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Problems with these files against a lock; empty means they are the locked bytes."""
    problems: list[str] = []
    if lock.get("lock_version") != LOCK_VERSION:
        problems.append(f"lock_version {lock.get('lock_version')!r} is not {LOCK_VERSION}")
    if digest_records(cases) != lock.get("cases_digest"):
        problems.append("cases differ from the locked cases")
    if digest_records(golds) != lock.get("gold_digest"):
        problems.append("gold differs from the locked gold")
    return problems
