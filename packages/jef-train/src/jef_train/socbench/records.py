"""Case and gold records: ids, trust tiers, validation, JSONL I/O.

Every adapter writes two files per source: ``cases.jsonl`` (what a model
sees; no verdict) and ``gold.jsonl`` (the answer key, keyed by case_id). Both
are validated against ``schema/case.schema.json`` and ``schema/gold.schema.json``
before anything is written, so a malformed record fails the build, not a
benchmark run three steps later.

Derived files land under the ignored data directory
(``.cache/socbench/derived/<source>/``). They contain third-party data and are
never committed; only the code that derives them is.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .manifest import SocbenchRepo

__all__ = [
    "TRUST_TIERS",
    "case_id",
    "read_jsonl",
    "trust_tier",
    "validate_records",
    "write_jsonl",
]

#: Review section 9.4. A: two analysts adjudicated. B: one analyst. C: a
#: high-quality source ground truth. D: a weak label (attack-phase window,
#: emulation, partial positives). E: synthetic or LLM-authored. Public numbers
#: are reported per tier, never pooled.
TRUST_TIERS: Mapping[str, str] = {
    "double-human-reviewed": "A",
    "single-human-reviewed": "B",
    "flow-label": "C",
    "ctf-answer-key": "C",
    "spam-ham": "C",
    "phishing-vs-benign-website": "C",
    "policy-fixture": "C",
    "scam-vs-hard-negative-prompts": "C",
    "attack-phase-window": "D",
    "partial-red-team-positives": "D",
    "adversary-emulation-semi-labelled": "D",
    "llm-authored": "E",
    "synthetic": "E",
}


def trust_tier(label_origin: str) -> str:
    """The trust tier of a label origin; unknown origins are the weakest."""
    return TRUST_TIERS.get(label_origin, "E")


def case_id(tag: str, n: int) -> str:
    """``socb-<tag>-<6 digits>``: unique across sources because the tag is the source."""
    return f"socb-{tag}-{n:06d}"


def validate_records(repo: SocbenchRepo, kind: str, records: Iterable[Mapping[str, Any]]) -> int:
    """Validate every record against ``<kind>.schema.json``; raise on the first failure."""
    n = 0
    for n, record in enumerate(records, start=1):
        repo.validate(kind, record, where=f"{kind} record {record.get('case_id', n)}")
    return n


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Write JSON lines atomically: a killed build never leaves a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    n = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
