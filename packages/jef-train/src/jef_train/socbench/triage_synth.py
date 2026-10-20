"""Render the synthetic incident-triage families and lock them by family.

The families live in ``benchmark/incident-triage/synthetic/families-v1.yaml``,
where a reviewer can read every scenario next to the label LABELBOOK.md gives
it. This module only renders slots, links the variants of a family, and
refuses to lock a set whose families would leak across splits or whose text
is a near copy of the training corpus.

Labels are ``by_construction`` (Tier E). They test the engineering contract
and cover the label space; they are not gold.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .splits import DEFAULT_RATIOS, assign_splits, check_leakage, digest_records
from .triage import _sha256

__all__ = [
    "CLOSE_ACTIONS",
    "LockError",
    "build_families",
    "char_ngrams",
    "jaccard",
    "near_duplicates",
    "training_templates",
    "triage_lock",
    "verify_triage_lock",
    "write_pack",
]

TRIAGE_LOCK_VERSION = 1
CLOSE_ACTIONS = ["close", "close_pending_review", "suppress"]
#: Two cases of different families this similar are one scenario twice.
NEAR_DUP_ACROSS_FAMILIES = 0.80
#: An evaluation case this similar to a training template measures recall.
NEAR_DUP_TO_TRAINING = 0.60

_HOSTS = [f"{p}.corp.example" for p in ("app-07", "fs-03", "db-11", "web-02", "batch-04", "ws-19")]
_LAPTOPS = [f"nb-{n:04d}.corp.example" for n in (118, 274, 391, 552)]
_IPS = [
    "192.0.2.14",
    "192.0.2.201",
    "198.51.100.36",
    "198.51.100.140",
    "203.0.113.58",
    "203.0.113.99",
]
_TIMES = ["00:42", "02:09", "04:26", "10:15", "14:37", "19:03", "23:51"]
_USERS = ["u-1042", "u-2391", "u-3305", "u-4176", "u-5820"]


class LockError(ValueError):
    pass


def _fill(rng: random.Random) -> dict[str, str]:
    return {
        "host": rng.choice(_HOSTS),
        "host2": rng.choice(_LAPTOPS),
        "ip": rng.choice(_IPS),
        "time": rng.choice(_TIMES),
        "n": str(rng.randint(2, 9)),
        "user": rng.choice(_USERS),
        "ticket": f"CHG-2026-{rng.randint(1000, 9999)}",
    }


def build_families(spec_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Cases and expectations for every variant, in zh-TW and en."""
    spec = yaml.safe_load(Path(spec_path).read_text("utf-8"))
    version, seed = spec["version"], spec["seed"]
    cases: list[dict[str, Any]] = []
    exps: list[dict[str, Any]] = []
    n = 0
    for fam in spec["families"]:
        ids: dict[tuple[str, str], str] = {}  # (kind, language) -> case_id
        for vi, v in enumerate(fam["variants"]):
            rng = random.Random(f"{seed}:{fam['family']}:{vi}")  # noqa: S311 -- reproducible fixtures
            slots = _fill(rng)
            for lang, key in (("zh-TW", "zh"), ("en", "en")):
                n += 1
                cid = f"itc-synth-v{version}-{n:03d}"
                ids[(v["kind"], lang)] = cid
                state = v[key].format(**slots)
                case: dict[str, Any] = {
                    "case_id": cid,
                    "case_family": fam["family"],
                    "domain": fam["domain"],
                    "origin": "synthetic",
                    "language": lang,
                    "state": state,
                    "evidence": [],
                    "licence_or_permission": "authored for JEF; no third-party alert content",
                    "derived_from": f"synthetic/families-v{version}.yaml#{fam['family']}/{v['kind']}",
                    "twin_of": None,
                    "translation_of": ids[(v["kind"], "zh-TW")] if lang == "en" else None,
                    "hard_negative_of": None,
                    "state_sha256": _sha256(state.encode("utf-8")),
                }
                if v["kind"] == "hard_negative":
                    case["hard_negative_of"] = ids[("base", lang)]
                if v["kind"] == "twin":
                    case["twin_of"] = ids[(v["of"], lang)]
                cases.append(case)
                labels = v["labels"]
                exps.append(
                    {
                        "case_id": cid,
                        "label_origin": "synthetic",
                        "label_status": "by_construction",
                        "reviewers": [],
                        "truth_of_event": v["event"],
                        "truth_of_alert": v["alert"],
                        "expected": {
                            "false_positive": labels["fp"],
                            "evidence_sufficiency": labels["ev"],
                            "owner": labels["owner"],
                            "severity": labels["severity"],
                            "urgency": labels["urgent"],
                        },
                        "prohibited_actions": (
                            list(CLOSE_ACTIONS) if v["event"] != "benign" else []
                        ),
                        "required_context": list(v.get("required_context", [])),
                        "note": f"by construction: LABELBOOK rules for {fam['family']}/{v['kind']}",
                    }
                )
    return cases, exps


def write_pack(
    out: Path, cases: Sequence[Mapping[str, Any]], exps: Sequence[Mapping[str, Any]]
) -> None:
    from .triage import pack_manifest

    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("cases.jsonl", cases), ("expectations.jsonl", exps)):
        (out / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
            "utf-8",
        )
    (out / "manifest.json").write_text(
        json.dumps(pack_manifest(out), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        "utf-8",
    )


# --------------------------------------------------------------------------- #
# near-duplicates
# --------------------------------------------------------------------------- #

_SLOT_NOISE = re.compile(
    r"\d+|[a-z0-9-]+\.corp\.example|nb-\d+|u-\d+|chg-\d+-\d+|\{[a-z0-9]+\}", re.IGNORECASE
)


def char_ngrams(text: str, n: int = 3) -> set[str]:
    """Character n-grams with slot values (numbers, hosts, ids) masked out."""
    t = re.sub(r"\s+", " ", _SLOT_NOISE.sub("#", text.lower()))
    return {t[i : i + n] for i in range(max(0, len(t) - n + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def training_templates() -> list[tuple[str, str]]:
    """``(scenario, template)`` of the synthetic training corpus for soar_zh."""
    from ..corpus.soar_zh import SCENARIOS

    return [(s.key, t) for s in SCENARIOS for t in s.templates]


def near_duplicates(
    cases: Sequence[Mapping[str, Any]], training: Iterable[tuple[str, str]] = ()
) -> dict[str, Any]:
    grams = {c["case_id"]: char_ngrams(c["state"]) for c in cases}
    across: list[dict[str, Any]] = []
    for i, a in enumerate(cases):
        for b in cases[i + 1 :]:
            if a["case_family"] == b["case_family"] or a["language"] != b["language"]:
                continue
            j = jaccard(grams[a["case_id"]], grams[b["case_id"]])
            if j >= NEAR_DUP_ACROSS_FAMILIES:
                across.append({"a": a["case_id"], "b": b["case_id"], "jaccard": round(j, 3)})
    train = [(k, char_ngrams(t)) for k, t in training]
    to_training: list[dict[str, Any]] = []
    worst = 0.0
    for c in cases:
        for key, g in train:
            j = jaccard(grams[c["case_id"]], g)
            worst = max(worst, j)
            if j >= NEAR_DUP_TO_TRAINING:
                to_training.append({"case": c["case_id"], "training": key, "jaccard": round(j, 3)})
    return {
        "thresholds": {
            "across_families": NEAR_DUP_ACROSS_FAMILIES,
            "to_training": NEAR_DUP_TO_TRAINING,
        },
        "across_families": across,
        "to_training": to_training,
        "max_jaccard_to_training": round(worst, 3),
    }


# --------------------------------------------------------------------------- #
# lock
# --------------------------------------------------------------------------- #


def triage_lock(
    cases: Sequence[Mapping[str, Any]],
    exps: Sequence[Mapping[str, Any]],
    *,
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
    seed: str = "incident-triage",
    training: Iterable[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Family-grouped splits, refused on any leak or near copy."""
    as_socbench = [
        {
            **c,
            "family_id": c["case_family"],
            "paired_with": c.get("translation_of") or c.get("hard_negative_of") or c.get("twin_of"),
        }
        for c in cases
    ]
    family_split = assign_splits(as_socbench, ratios=ratios, seed=seed)
    problems = check_leakage(as_socbench, family_split)
    ids = [c["case_id"] for c in cases]
    if sorted(ids) != sorted(e["case_id"] for e in exps):
        problems.append("expectations do not cover exactly the cases")
    dups = near_duplicates(cases, training_templates() if training is None else training)
    problems += [
        f"{d['a']} and {d['b']} are near-duplicates across families (jaccard {d['jaccard']})"
        for d in dups["across_families"]
    ]
    problems += [
        f"{d['case']} is a near copy of training scenario {d['training']} (jaccard {d['jaccard']})"
        for d in dups["to_training"]
    ]
    if problems:
        raise LockError("refusing to lock:\n  " + "\n  ".join(problems))

    splits: dict[str, list[str]] = defaultdict(list)
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    exp_by = {e["case_id"]: e for e in exps}
    for c in sorted(cases, key=lambda c: c["case_id"]):
        split = family_split[c["case_family"]]
        splits[split].append(c["case_id"])
        counts[split]["language"][c["language"]] += 1
        counts[split]["domain"][str(c.get("domain"))] += 1
        counts[split]["label_status"][exp_by[c["case_id"]]["label_status"]] += 1
        counts[split]["hard_negative"][str(bool(c.get("hard_negative_of")))] += 1
    return {
        "lock_version": TRIAGE_LOCK_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "ratios": dict(ratios),
        "cases_digest": digest_records(cases),
        "expectations_digest": digest_records(exps),
        "families": {fam: family_split[fam] for fam in sorted(family_split)},
        "splits": {k: sorted(v) for k, v in sorted(splits.items())},
        "counts": {s: {k: dict(v) for k, v in d.items()} for s, d in sorted(counts.items())},
        "near_duplicates": dups,
    }


def verify_triage_lock(
    lock: Mapping[str, Any], cases: Sequence[Mapping[str, Any]], exps: Sequence[Mapping[str, Any]]
) -> list[str]:
    problems = []
    if lock.get("lock_version") != TRIAGE_LOCK_VERSION:
        problems.append(f"lock_version {lock.get('lock_version')!r} is not {TRIAGE_LOCK_VERSION}")
    if digest_records(cases) != lock.get("cases_digest"):
        problems.append("cases differ from the locked cases")
    if digest_records(exps) != lock.get("expectations_digest"):
        problems.append("expectations differ from the locked expectations")
    # The splits are trusted downstream (baselines, candidate export), so they
    # must be the ones the seed and ratios produce, not a hand-edited list.
    families = assign_splits(
        [{"family_id": c["case_family"]} for c in cases],
        ratios=lock.get("ratios", DEFAULT_RATIOS),
        seed=str(lock.get("seed")),
    )
    if families != lock.get("families"):
        problems.append("families are not assigned as the lock's seed and ratios assign them")
    expected: dict[str, list[str]] = defaultdict(list)
    for c in cases:
        expected[families[c["case_family"]]].append(c["case_id"])
    if {k: sorted(v) for k, v in expected.items()} != lock.get("splits"):
        problems.append("splits are not the cases' family splits")
    return problems
