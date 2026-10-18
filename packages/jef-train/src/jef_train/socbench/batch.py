"""Annotation batches managed in git: labels in the repository, data never.

A batch is a directory under ``benchmark/socbench/annotation/batches/``::

    <batch-id>/
      batch.json                    what is being labelled and by whom
      labels-analyst-anon-01.jsonl  one file per reviewer, written only by them
      labels-analyst-anon-02.jsonl  (gold batches: a second, independent reviewer)
      decisions.jsonl               the adjudicator's calls on disagreements

``batch.json`` pins the cases by digest and lists their ids and tasks, with
the command that regenerates them from the pinned public sources. Case text
never enters the repository (the source manifest's rule); reviewers export
their blind tasks locally with ``jef-bench annotate export``.

:func:`check_batch` is what CI runs on a pull request that touches a batch.
It needs no case data for the structural checks, and with ``--cases`` it also
runs the full :func:`annotate.parse_annotations` validation. Independence is
checked from git: every commit touching a reviewer's file must come from one
author, the two reviewers' authors must not overlap, no single commit may
touch both reviewers' files, and the adjudicator must be neither reviewer.
A git author is an email the committer chose, so this catches honest
mistakes and lazy shortcuts (one person committing both files), not a
determined impostor; the labelbook's rule that reviewer B is a different
qualified person, never a second handle and never an LLM, still stands.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .annotate import TASK_LABELS, AnnotationError, cohen_kappa, parse_annotations
from .records import read_jsonl
from .splits import digest_records

__all__ = ["LABEL_FIELDS", "BatchError", "check_batch", "git_authors", "new_batch"]

_HANDLE = re.compile(r"^analyst-anon-[0-9]{2}$")
_BATCH_ID = re.compile(r"^[a-z0-9][a-z0-9.-]{2,63}$")

#: The only fields a label row may carry. Anything else -- the state, the
#: evidence text, a model's answer -- has no business in the repository.
LABEL_FIELDS = frozenset(
    {
        "case_id",
        "annotator",
        "verdict",
        "confidence",
        "evidence_ids",
        "prohibited_actions",
        "required_followup",
        "note",
        "seconds_spent",
    }
)
_DECISION_FIELDS = frozenset({"case_id", "adjudicator", "verdict", "note"})
PURPOSES = {"pilot": 1, "gold": 2}  # purpose -> number of independent reviewers


class BatchError(ValueError):
    """batch.json itself is unusable; nothing else can be checked."""


def _load_batch(path: Path) -> dict[str, Any]:
    try:
        meta = json.loads((path / "batch.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchError(f"{path}/batch.json: {exc}") from exc
    problems = []
    if not isinstance(meta.get("batch_id"), str) or not _BATCH_ID.match(meta["batch_id"]):
        problems.append("batch_id must be 3-64 of [a-z0-9.-]")
    elif meta["batch_id"] != path.name:
        problems.append(f"batch_id {meta['batch_id']!r} does not match the directory {path.name!r}")
    if meta.get("purpose") not in PURPOSES:
        problems.append(f"purpose must be one of {sorted(PURPOSES)}")
    cases = meta.get("cases")
    if not isinstance(cases, dict) or not cases:
        problems.append("cases must map every case_id to its task_id")
    else:
        bad = sorted(t for t in set(cases.values()) if t not in TASK_LABELS)
        if bad:
            problems.append(f"unknown task ids {bad}")
    if not str(meta.get("cases_digest", "")).startswith("sha256:"):
        problems.append("cases_digest must be the sha256 digest of the case file")
    reviewers = meta.get("reviewers")
    want = PURPOSES.get(str(meta.get("purpose")), 0)
    if not isinstance(reviewers, list) or len(reviewers) != want:
        problems.append(f"a {meta.get('purpose')} batch names exactly {want} reviewer(s)")
    elif len(set(reviewers)) != len(reviewers) or not all(
        isinstance(r, str) and _HANDLE.match(r) for r in reviewers
    ):
        problems.append("reviewers must be distinct analyst-anon-NN handles")
    adj = meta.get("adjudicator")
    if adj is not None and (not isinstance(adj, str) or not _HANDLE.match(adj)):
        problems.append("adjudicator must be an analyst-anon-NN handle or null")
    if isinstance(reviewers, list) and adj in reviewers:
        problems.append("the adjudicator must be a third person, not a reviewer")
    if not meta.get("recipe"):
        problems.append("recipe must say how to regenerate the cases from pinned sources")
    if problems:
        raise BatchError(f"{path.name}: " + "; ".join(problems))
    return meta


def git_authors(batch_dir: Path) -> dict[str, Any]:
    """``{"files": {name: [emails]}, "commits": [[sha, email, [names]]]}`` for the batch."""
    out = subprocess.run(  # noqa: S603 - fixed git invocation over a repository path
        ["git", "log", "--format=@%H%x09%ae", "--name-only", "--", str(batch_dir)],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
        cwd=batch_dir,
    ).stdout
    commits: list[list[Any]] = []
    for line in out.splitlines():
        if line.startswith("@"):
            sha, email = line[1:].split("\t", 1)
            commits.append([sha, email, []])
        elif line.strip() and commits:
            commits[-1][2].append(Path(line.strip()).name)
    files: dict[str, set[str]] = defaultdict(set)
    for _, email, names in commits:
        for n in names:
            files[n].add(email)
    return {"files": {k: sorted(v) for k, v in files.items()}, "commits": commits}


def _rows(path: Path, allowed: frozenset[str], problems: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = read_jsonl(path)
    for i, row in enumerate(rows, 1):
        extra = sorted(set(row) - allowed)
        if extra:
            problems.append(f"{path.name}:{i}: fields {extra} are not allowed in the repository")
    return rows


def check_batch(
    batch_dir: Path,
    *,
    cases: Sequence[Mapping[str, Any]] | None = None,
    authors: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything CI needs to say about one batch. ``problems`` empty means it may merge."""
    meta = _load_batch(batch_dir)
    case_task: dict[str, str] = meta["cases"]
    reviewers: list[str] = meta["reviewers"]
    problems: list[str] = []
    notes: list[str] = []

    # -- what is in the directory
    known = {"batch.json", "decisions.jsonl", *(f"labels-{r}.jsonl" for r in reviewers)}
    for f in sorted(p.name for p in batch_dir.iterdir()):
        if f not in known and f != "README.md":
            problems.append(
                f"unexpected file {f}: only batch.json, labels-<reviewer>.jsonl, decisions.jsonl, README.md"
            )

    # -- labels: shape, membership, one per (case, reviewer)
    label_rows: dict[str, list[dict[str, Any]]] = {}
    for r in reviewers:
        rows = _rows(batch_dir / f"labels-{r}.jsonl", LABEL_FIELDS, problems)
        label_rows[r] = rows
        seen: Counter[str] = Counter()
        for row in rows:
            cid = row.get("case_id")
            if row.get("annotator") != r:
                problems.append(f"labels-{r}.jsonl: a row is signed {row.get('annotator')!r}")
            if cid not in case_task:
                problems.append(f"labels-{r}.jsonl: {cid!r} is not in this batch")
                continue
            seen[cid] += 1
            if row.get("verdict") not in TASK_LABELS[case_task[cid]]:
                problems.append(
                    f"labels-{r}.jsonl: {cid}: {row.get('verdict')!r} is not a {case_task[cid]} label"
                )
        for cid, n in seen.items():
            if n > 1:
                problems.append(f"labels-{r}.jsonl: {cid} labelled {n} times")

    # -- full validation against the cases, when they are available
    if cases is not None:
        digest = digest_records(cases)
        if digest != meta["cases_digest"]:
            problems.append(f"cases digest {digest} is not the pinned {meta['cases_digest']}")
        by_id = {c["case_id"]: c for c in cases}
        missing = sorted(set(case_task) - set(by_id))
        if missing:
            problems.append(
                f"{len(missing)} batch case(s) absent from the case file, e.g. {missing[0]}"
            )
        try:
            parse_annotations([row for rows in label_rows.values() for row in rows], by_id)
        except AnnotationError as exc:
            problems.append(f"annotation check: {exc}")
    else:
        notes.append("structural checks only: no case file was supplied (CI has no data)")

    completeness = {
        r: {"labelled": len({row.get("case_id") for row in rows}), "of": len(case_task)}
        for r, rows in label_rows.items()
    }
    complete = all(c["labelled"] == c["of"] for c in completeness.values())

    # -- agreement and the disagreement queue (gold batches)
    queue: list[str] = []
    kappa: float | None = None
    if len(reviewers) == 2 and all(label_rows.values()):
        a = {row["case_id"]: row["verdict"] for row in label_rows[reviewers[0]]}
        b = {row["case_id"]: row["verdict"] for row in label_rows[reviewers[1]]}
        both = sorted(set(a) & set(b))
        kappa = cohen_kappa([a[c] for c in both], [b[c] for c in both]) if both else None
        queue = [c for c in both if a[c] != b[c]]

    # -- decisions: only on the queue, only by the adjudicator
    decisions = _rows(batch_dir / "decisions.jsonl", _DECISION_FIELDS, problems)
    decided = set()
    for d in decisions:
        cid = d.get("case_id")
        if meta["purpose"] != "gold":
            problems.append("decisions.jsonl belongs to gold batches only")
            break
        if d.get("adjudicator") != meta.get("adjudicator"):
            problems.append(
                f"decisions.jsonl: {cid} decided by {d.get('adjudicator')!r}, not the batch's adjudicator"
            )
        if cid not in queue:
            problems.append(
                f"decisions.jsonl: {cid} is not a disagreement; agreed cases are not re-decided"
            )
        elif d.get("verdict") not in TASK_LABELS[case_task[cid]]:
            problems.append(
                f"decisions.jsonl: {cid}: {d.get('verdict')!r} is not a {case_task[cid]} label"
            )
        decided.add(cid)

    # -- independence, from git
    independence: dict[str, Any] = {}
    if authors is not None:
        files: Mapping[str, Sequence[str]] = authors.get("files", {})
        by_reviewer = {r: set(files.get(f"labels-{r}.jsonl", ())) for r in reviewers}
        for r, emails in by_reviewer.items():
            if len(emails) > 1:
                problems.append(
                    f"labels-{r}.jsonl was committed by {len(emails)} different authors"
                )
        if len(reviewers) == 2:
            shared = by_reviewer[reviewers[0]] & by_reviewer[reviewers[1]]
            if shared:
                problems.append("both reviewers' labels were committed by the same author")
        label_files = {f"labels-{r}.jsonl" for r in reviewers}
        for sha, _, names in authors.get("commits", []):
            if len(label_files & set(names)) > 1:
                problems.append(f"commit {sha[:12]} touches more than one reviewer's labels")
        adj_emails = set(files.get("decisions.jsonl", ()))
        if adj_emails & set().union(*by_reviewer.values()):
            problems.append(
                "decisions were committed by a reviewer's author; the adjudicator must be a third person"
            )
        independence = {
            "reviewer_authors": {r: len(e) for r, e in by_reviewer.items()},
            "distinct_reviewer_authors": len(set().union(*by_reviewer.values())),
        }
    else:
        notes.append("independence not checked: no git history was supplied")

    open_queue = sorted(set(queue) - decided)
    if meta["purpose"] == "pilot":
        tier = "B" if complete and not problems else None
    else:
        tier = "A" if complete and not open_queue and not problems and authors is not None else None
    return {
        "batch_id": meta["batch_id"],
        "purpose": meta["purpose"],
        "cases": len(case_task),
        "tasks": dict(Counter(case_task.values())),
        "completeness": completeness,
        "complete": complete,
        "kappa": None if kappa is None else round(kappa, 4),
        "disagreements": len(queue),
        "open_disagreements": len(open_queue),
        "independence": independence,
        "tier_if_merged": tier,
        "problems": problems,
        "notes": notes,
    }


def new_batch(
    cases: Sequence[Mapping[str, Any]],
    *,
    batch_id: str,
    purpose: str,
    reviewers: Sequence[str],
    recipe: str,
    adjudicator: str | None = None,
    note: str = "",
) -> dict[str, Any]:
    """batch.json for these cases: ids, tasks and digest, never the case text."""
    meta = {
        "batch_id": batch_id,
        "purpose": purpose,
        "cases_digest": digest_records(cases),
        "cases": {c["case_id"]: c["task_id"] for c in cases},
        "families": dict(Counter(str(c.get("family_id")) for c in cases)),
        "sources": dict(Counter(str(c.get("source_dataset")) for c in cases)),
        "reviewers": list(reviewers),
        "adjudicator": adjudicator,
        "recipe": recipe,
        "note": note,
    }
    return meta
