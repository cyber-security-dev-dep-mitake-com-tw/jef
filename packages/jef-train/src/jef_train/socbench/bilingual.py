"""Paired cases: hard-negative twins and translations (review 10.1, 10.2).

Two kinds of pair, both carried by ``paired_with`` + ``pair_kind`` and both
sharing a ``family_id`` so a family split keeps them together (DATA-003):

* ``hard-negative``: a legitimate twin that differs from the positive only in
  the decisive detail (sender domain, approval, schedule, signature);
* ``translation``: the same evidence in another language. It measures
  language robustness (does the verdict, calibration or abstention move when
  only the language changes), not local-context understanding, which needs
  native cases from that locale.

``zh-TW`` and ``zh-CN`` are different languages here and are reported apart.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["check_pairs", "pair_translation"]


def pair_translation(
    case: Mapping[str, Any],
    gold: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    new_case_id: str,
    language: str,
    translator: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """A translated twin of ``case`` and its gold. The verdict carries over; the origin does too.

    ``state`` must have the same keys as the original: a translation that adds
    or drops a field is a different case, not the same evidence.
    """
    if language == case.get("language"):
        raise ValueError(f"{case['case_id']} is already {language}")
    if set(state) != set(case["state"]):
        raise ValueError(
            f"translation of {case['case_id']} must keep the same fields: "
            f"{sorted(case['state'])} vs {sorted(state)}"
        )
    evidence = [
        {
            "id": e["id"],
            "field": e["field"],
            "value": state[e["field"]] if e["field"] in state else e["value"],
        }
        for e in case["evidence"]
    ]
    twin = {
        **{
            k: v
            for k, v in case.items()
            if k
            not in {
                "case_id",
                "state",
                "evidence",
                "language",
                "paired_with",
                "pair_kind",
                "hard_negative",
            }
        },
        "case_id": new_case_id,
        "language": language,
        "state": dict(state),
        "evidence": evidence,
        "paired_with": case["case_id"],
        "pair_kind": "translation",
        "hard_negative": bool(case.get("hard_negative", False)),
    }
    twin_gold = {
        **gold,
        "case_id": new_case_id,
        "adjudication_note": f"translation of {case['case_id']} by {translator}; same evidence, same verdict",
    }
    return twin, twin_gold


def check_pairs(
    cases: Sequence[Mapping[str, Any]], golds: Sequence[Mapping[str, Any]] = ()
) -> list[str]:
    """Problems with paired cases; empty when every pair is consistent."""
    by_id = {c["case_id"]: c for c in cases}
    gold_by = {g["case_id"]: g for g in golds}
    problems: list[str] = []
    for c in cases:
        partner_id = c.get("paired_with")
        if not partner_id:
            continue
        partner = by_id.get(partner_id)
        cid = c["case_id"]
        if partner is None:
            problems.append(f"{cid} is paired with {partner_id}, which is not in the set")
            continue
        if partner["family_id"] != c["family_id"]:
            problems.append(f"{cid} and its pair {partner_id} are in different families")
        if partner["task_id"] != c["task_id"]:
            problems.append(f"{cid} and its pair {partner_id} ask different tasks")
        kind = c.get("pair_kind")
        if kind == "translation":
            if partner.get("language") == c.get("language"):
                problems.append(f"{cid} is a translation in the same language as {partner_id}")
            if (
                cid in gold_by
                and partner_id in gold_by
                and gold_by[cid]["gold"]["verdict"] != gold_by[partner_id]["gold"]["verdict"]
            ):
                problems.append(f"translation {cid} does not carry {partner_id}'s verdict")
        elif kind == "hard-negative":
            if bool(c.get("hard_negative")) == bool(partner.get("hard_negative")):
                problems.append(
                    f"{cid} and {partner_id} are a hard-negative pair but not one positive and one negative"
                )
            if partner.get("paired_with") != cid:
                problems.append(f"hard-negative pair {cid} -> {partner_id} is not mutual")
        else:
            problems.append(f"{cid} has paired_with but no pair_kind")
    return problems
