"""SOCBench gold -> a ``jef_train.fit`` corpus, split exactly as the lock says.

``jef_train.fit`` reads its own corpus format (``Sample`` records in
``train/calibration/test.jsonl`` plus ``all.jsonl`` and a manifest). SOCBench
produces cases, gold records and a lock. This module is the bridge, and it is
deliberately narrow:

* It refuses to run unless the files are the locked bytes (``verify_lock``), so
  a corpus can always be traced to one lock digest.
* Splits come from the lock's family-grouped assignment, never recomputed, and
  ``assert_no_group_leakage`` is run on the result anyway.
* Only gold at or above the chosen trust tier is exported (``A`` by default:
  two independent reviewers). Weak labels are counted and reported, not used.
* The test split is written for ``fit`` to *report* on; the fitter never fits
  temperature, isotonic maps or conformal quantiles on it. The manifest says so.

Every sample carries ``task=<task_id>`` so the calibrator fits task-scoped
buckets (schema 2): ``soc.alert.verdict.v1|choice:5``, not ``choice:5``.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..build_corpus import summarise
from ..schema import Sample, assert_no_group_leakage, write_samples
from .annotate import TASK_LABELS
from .baselines import LABEL_TEXT, TASK_QUESTIONS, case_text
from .records import trust_tier
from .splits import LockError, verify_lock

__all__ = ["TASK_QUESTIONS", "ExportError", "export_corpus", "samples_from_gold"]

TIER_ORDER = "ABCDE"


_LANG = {"n/a": "en", "zh": "zh-TW"}


class ExportError(ValueError):
    """The inputs cannot become a trustworthy corpus."""


def _tier_ok(tier: str, minimum: str) -> bool:
    return TIER_ORDER.index(tier) <= TIER_ORDER.index(minimum)


def samples_from_gold(
    cases: Sequence[Mapping[str, Any]],
    golds: Sequence[Mapping[str, Any]],
    split_of: Mapping[str, str],
    *,
    min_tier: str = "A",
    question_lang: str = "en",
) -> tuple[list[Sample], Counter[str]]:
    """``(samples, excluded_by_tier)``. One choice sample per gold record."""
    if min_tier not in TIER_ORDER:
        raise ExportError(f"min_tier must be one of {TIER_ORDER}, not {min_tier!r}")
    by_id = {c["case_id"]: c for c in cases}
    out: list[Sample] = []
    excluded: Counter[str] = Counter()
    for g in golds:
        cid = g["case_id"]
        case = by_id.get(cid)
        if case is None:
            raise ExportError(f"gold {cid!r} has no case")
        tier = trust_tier(g["label_origin"])
        if not _tier_ok(tier, min_tier):
            excluded[tier] += 1
            continue
        task = g["task_id"]
        labels = TASK_LABELS.get(task)
        if labels is None:
            raise ExportError(f"{cid}: unknown task {task!r}")
        verdict = g["gold"]["verdict"]
        if verdict not in labels:
            raise ExportError(f"{cid}: verdict {verdict!r} is not a {task} label")
        if cid not in split_of:
            raise ExportError(f"{cid}: not assigned to any split by the lock")
        lang = _LANG.get(str(case.get("language")), str(case.get("language")))
        qlang = question_lang if question_lang in TASK_QUESTIONS[task] else "en"
        out.append(
            Sample(
                state=case_text(case),
                kind="choice",
                instructions=TASK_QUESTIONS[task][qlang],
                option_keys=list(labels),
                option_labels=[LABEL_TEXT.get(lab, lab.replace("_", " ")) for lab in labels],
                label=labels.index(verdict),
                source=f"socbench.{case['source_dataset']}",
                group=case["family_id"],
                lang=lang,
                task=task,
                meta={
                    "case_id": cid,
                    "label_origin": g["label_origin"],
                    "trust_tier": tier,
                    "split": split_of[cid],
                    "hard_negative": bool(case.get("hard_negative", False)),
                    "state_lang": lang,
                    "confidence_of_gold": g["gold"].get("confidence_of_gold"),
                },
            )
        )
    return out, excluded


def export_corpus(
    cases: Sequence[Mapping[str, Any]],
    golds: Sequence[Mapping[str, Any]],
    lock: Mapping[str, Any],
    out: Path,
    *,
    min_tier: str = "A",
    question_lang: str = "en",
) -> dict[str, Any]:
    """Write the corpus and return its manifest. Raises before writing anything."""
    problems = verify_lock(lock, cases, golds)
    if problems:
        raise LockError("inputs are not the locked bytes:\n  " + "\n  ".join(problems))
    split_of = {cid: name for name, ids in lock["splits"].items() for cid in ids}
    samples, excluded = samples_from_gold(
        cases, golds, split_of, min_tier=min_tier, question_lang=question_lang
    )
    if not samples:
        raise ExportError(f"no gold at trust tier {min_tier} or better; nothing to export")
    by_split: dict[str, list[Sample]] = {name: [] for name in ("train", "calibration", "test")}
    for s in samples:
        by_split.setdefault(str(s.meta["split"]), []).append(s)
    for required in ("train", "calibration"):
        if not by_split[required]:
            raise ExportError(f"the {required} split is empty at tier {min_tier}; cannot fit")
    assert_no_group_leakage(*(by_split[k] for k in ("train", "calibration", "test")))

    out.mkdir(parents=True, exist_ok=True)
    counts = {name: write_samples(rows, out / f"{name}.jsonl") for name, rows in by_split.items()}
    counts["all"] = write_samples(samples, out / "all.jsonl")
    tiers = {
        name: dict(Counter(str(s.meta["trust_tier"]) for s in rows))
        for name, rows in by_split.items()
    }
    manifest: dict[str, Any] = {
        "builder": "jef_train.socbench.corpus_export",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "lock": {
            "lock_version": lock.get("lock_version"),
            "seed": lock.get("seed"),
            "ratios": lock.get("ratios"),
            "cases_digest": lock.get("cases_digest"),
            "gold_digest": lock.get("gold_digest"),
        },
        "min_trust_tier": min_tier,
        "question_lang": question_lang,
        "excluded_by_tier": dict(excluded),
        "splits": counts,
        "trust_tiers_by_split": tiers,
        "label_origins": dict(Counter(str(s.meta["label_origin"]) for s in samples)),
        "summary": summarise(samples),
        "test_split": (
            "written for evaluation only: jef_train.fit reports metrics on it and never "
            "fits the head, temperature, isotonic maps or conformal quantiles on it"
        ),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
