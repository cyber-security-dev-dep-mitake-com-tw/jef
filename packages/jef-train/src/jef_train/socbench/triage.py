"""Incident-triage cases: the five ``soar_zh`` tasks the deployed scene asks.

Kept apart from the socbench case and gold files on purpose. A socbench case
carries one task; an incident-triage case is one alert that five questions
are asked about, and the scene may stop before asking most of them. Scoring
"five tasks" and scoring "what the scene proposed" are different
measurements, and this module keeps the inputs for both without letting one
stand in for the other.

A pack is a directory with ``cases.jsonl`` (inputs only), ``expectations.jsonl``
(what a reviewer expects, scored blind) and, optionally, ``replay.jsonl``
(answers a past run actually produced, replayed to make scene behaviour
deterministic). ``manifest.json`` pins every file by sha256.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jef_core import Engine
from jef_core.calibration import Calibrator
from jef_core.engine import SharedState
from jef_core.types import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    Question,
    RawQuestion,
    ScoreAnswer,
    normalize,
)
from jsonschema import Draft202012Validator

__all__ = [
    "PACK_FILES",
    "TASK_OF_QUESTION",
    "Pack",
    "PackError",
    "ReplayEngine",
    "find_root",
    "load_pack",
    "pack_manifest",
    "privacy_findings",
]

#: Question id in ``scenes/incident-triage.zh-tw.yaml`` -> expectation key.
TASK_OF_QUESTION = {
    "is_false_positive": "false_positive",
    "has_evidence": "evidence_sufficiency",
    "owner": "owner",
    "severity": "severity",
    "urgent": "urgency",
}

PACK_FILES = ("cases.jsonl", "expectations.jsonl", "replay.jsonl")


class PackError(ValueError):
    """A pack that does not match its schema, its manifest or its own hashes."""


def find_root(start: Path | None = None) -> Path:
    """``benchmark/incident-triage`` in this checkout."""
    for base in [start or Path.cwd(), Path(__file__).resolve()]:
        for p in [base, *base.parents]:
            candidate = p / "benchmark" / "incident-triage"
            if (candidate / "schema" / "case.schema.json").is_file():
                return candidate
    raise FileNotFoundError("benchmark/incident-triage not found; run from a checkout")


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for n, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise PackError(f"{path.name}:{n}: {exc}") from exc
    return rows


def _validator(root: Path, name: str) -> Draft202012Validator:
    schema = json.loads((root / "schema" / f"{name}.schema.json").read_text("utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@dataclass
class Pack:
    path: Path
    cases: list[dict[str, Any]]
    expectations: dict[str, dict[str, Any]]
    replay: dict[str, dict[str, Any]] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)

    def case(self, case_id: str) -> dict[str, Any]:
        return next(c for c in self.cases if c["case_id"] == case_id)


def pack_manifest(path: Path) -> dict[str, Any]:
    """The manifest a pack directory should carry, computed from its files."""
    files = {
        name: _sha256((path / name).read_bytes()) for name in PACK_FILES if (path / name).is_file()
    }
    cases = _read_jsonl(path / "cases.jsonl")
    expectations = _read_jsonl(path / "expectations.jsonl")
    return {
        "pack": path.name,
        "evidence_class": "CANARY — NOT AN ACCURACY ESTIMATE",
        "files": files,
        "cases": len(cases),
        "families": len({c["case_family"] for c in cases}),
        "label_status": sorted({str(e["label_status"]) for e in expectations}),
    }


def load_pack(path: Path, *, root: Path | None = None) -> Pack:
    """Load and check a pack: schemas, state hashes, manifest, cross-references."""
    path = Path(path)
    root = root or find_root(path)
    case_v, exp_v = _validator(root, "case"), _validator(root, "expectation")
    problems: list[str] = []

    cases = _read_jsonl(path / "cases.jsonl")
    for c in cases:
        for e in case_v.iter_errors(c):
            problems.append(f"cases {c.get('case_id')}: {'/'.join(map(str, e.path))} {e.message}")
        if "state" in c and c.get("state_sha256") != _sha256(c["state"].encode("utf-8")):
            problems.append(f"cases {c.get('case_id')}: state_sha256 does not match the state")
    ids = [str(c.get("case_id")) for c in cases]
    if len(set(ids)) != len(ids):
        problems.append("cases: duplicate case_id")

    expectations: dict[str, dict[str, Any]] = {}
    for e in _read_jsonl(path / "expectations.jsonl"):
        for err in exp_v.iter_errors(e):
            problems.append(
                f"expectations {e.get('case_id')}: {'/'.join(map(str, err.path))} {err.message}"
            )
        expectations[e.get("case_id", "")] = e
    if set(expectations) != set(ids):
        problems.append(
            "expectations do not cover exactly the cases: "
            f"missing {sorted(set(ids) - set(expectations))}, "
            f"extra {sorted(set(expectations) - set(ids))}"
        )
    for c in cases:
        twin = c.get("twin_of")
        if twin is not None and twin not in ids:
            problems.append(f"cases {c['case_id']}: twin_of {twin} is not in the pack")

    replay: dict[str, dict[str, Any]] = {}
    if (path / "replay.jsonl").is_file():
        for r in _read_jsonl(path / "replay.jsonl"):
            if r.get("case_id") not in ids:
                problems.append(f"replay {r.get('case_id')}: not a case in the pack")
            unknown = set(r.get("observed", {})) - set(TASK_OF_QUESTION)
            if unknown:
                problems.append(f"replay {r.get('case_id')}: unknown questions {sorted(unknown)}")
            replay[r["case_id"]] = r

    manifest: dict[str, Any] = {}
    if (path / "manifest.json").is_file():
        manifest = json.loads((path / "manifest.json").read_text("utf-8"))
        if manifest != pack_manifest(path):
            problems.append("manifest.json is stale; regenerate it with pack_manifest()")
    else:
        problems.append("manifest.json is missing")

    if problems:
        raise PackError(f"{path}:\n  " + "\n  ".join(problems))
    return Pack(path, cases, expectations, replay, manifest)


# --------------------------------------------------------------------------- #
# privacy lint
# --------------------------------------------------------------------------- #

_PRIVATE_IP = re.compile(
    r"\b(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b"
)
_ANY_IP = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
#: RFC 5737 documentation ranges: the only addresses a fixture may carry.
_DOC_IP = re.compile(r"^(?:192\.0\.2|198\.51\.100|203\.0\.113)\.\d{1,3}$")
_SECRET = re.compile(
    r"(?:AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|xox[abp]-[A-Za-z0-9-]{10,}|"
    r"sk-[A-Za-z0-9]{20,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY)"
)
_EMAIL = re.compile(r"[\w.+-]+@([\w-]+\.)+[\w-]+")
_FQDN = re.compile(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", re.IGNORECASE)
_ALLOWED_DOMAIN = re.compile(r"\.(?:example|invalid|test|local)$|^[\w.-]+\.(?:exe|dll|txt)$", re.I)


def privacy_findings(text: str) -> list[str]:
    """Anything in ``text`` that should not be in a public fixture."""
    found = [f"private IP {m}" for m in _PRIVATE_IP.findall(text)]
    found += [
        f"non-documentation IP {m}"
        for m in _ANY_IP.findall(text)
        if not _DOC_IP.match(m) and not _PRIVATE_IP.match(m)
    ]
    found += [f"secret-like token {m[:8]}…" for m in _SECRET.findall(text)]
    found += [f"email {m.group(0)}" for m in _EMAIL.finditer(text)]
    found += [
        f"real-looking domain {m.group(0)}"
        for m in _FQDN.finditer(text)
        if not _ALLOWED_DOMAIN.search(m.group(0)) and not _ANY_IP.fullmatch(m.group(0))
    ]
    return found


# --------------------------------------------------------------------------- #
# replay
# --------------------------------------------------------------------------- #


def _filler(question: Question | RawQuestion, qid: str) -> Answer:
    """A deliberately uninformative answer for a question the source did not record."""
    q = normalize(qid, question)
    if q.kind == "noul":
        return NoulAnswer(noul=0.5, confidence=0.0)
    keys = list(q.option_keys)
    probs = {k: round(1.0 / len(keys), 6) for k in keys}
    if q.kind == "choice":
        return ChoiceAnswer(choice=keys[0], probabilities=probs, confidence=0.0)
    return ScoreAnswer(score=(len(keys) - 1) / 2, legend=keys, probabilities=probs, confidence=0.0)


def _observed(question: Question | RawQuestion, qid: str, value: Any) -> Answer:
    """An answer carrying what the source recorded.

    noul values are P(true). For choice and score only the top answer was
    recorded, so it carries 0.6 and the rest share 0.4 -- a stated filler,
    which no gate under legacy calibration reads (``p_correct`` is None and
    the prediction set is the full set).
    """
    q = normalize(qid, question)
    if q.kind == "noul":
        p = float(value)
        return NoulAnswer(noul=p, confidence=abs(2 * p - 1))
    keys = list(q.option_keys)
    if value not in keys:
        raise PackError(f"replay value {value!r} is not an option of {qid} ({keys})")
    rest = round(0.4 / (len(keys) - 1), 6)
    probs = {k: (0.6 if k == value else rest) for k in keys}
    if q.kind == "choice":
        return ChoiceAnswer(choice=value, probabilities=probs, confidence=0.6)
    return ScoreAnswer(
        score=float(keys.index(value)), legend=keys, probabilities=probs, confidence=0.6
    )


class ReplayEngine(Engine):
    """Answers each question with what a past run recorded, so a scene replays exactly.

    Questions the record does not cover get :func:`_filler`. Which questions
    were filled is reported in ``filled`` so a report can say so.
    """

    def __init__(self, record: Mapping[str, Any], calibrator: Calibrator | None = None) -> None:
        super().__init__(calibrator=calibrator)
        self.record = record
        self.filled: list[str] = []

    def answer(
        self, shared: SharedState, questions: Mapping[str, Question | RawQuestion]
    ) -> dict[str, Answer]:
        observed = self.record.get("observed", {})
        out: dict[str, Answer] = {}
        for qid, question in questions.items():
            if qid in observed:
                out[qid] = _observed(question, qid, observed[qid])
            else:
                self.filled.append(qid)
                out[qid] = _filler(question, qid)
        return out
