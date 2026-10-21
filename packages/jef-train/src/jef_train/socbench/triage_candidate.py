"""A task-scoped (schema 2) candidate for the five incident-triage tasks.

Plan 2026-10-20, PR-06. The goal is not a calibration file whose ``schema``
says 2; it is calibration fitted on independent, task-matched data, for the
exact buckets the scene asks (``soar_zh.routing.v1|choice:5`` and the rest),
checked on a test split nothing was fitted on. Until such data exists this
module does its job by refusing:

``export_triage_corpus``
    an incident-triage pack and its family lock -> a ``jef_train.fit`` corpus,
    one sample per case and task, questions taken from the scene. Only labels
    at or above ``min_tier`` (default B: a named reviewer) are exported;
    synthetic labels (Tier E) only with ``dry_run=True``, and the manifest then
    says the result can never be deployed.

``check_candidate``
    a fitted candidate directory -> a report whose ``status`` is
    ``insufficient`` unless every required bucket is ``validated`` (at least
    ``MIN_CORRECTNESS_SAMPLES`` calibration records, a correctness map and a
    conformal quantile at the scene's alpha), its artifacts hash to what the
    calibration says, the corpus splits share no family, and conformal coverage
    holds on the test split. Even then the status is ``research``: promoting a
    candidate needs human-labelled evidence (Tier A) and a decision recorded in
    the validation ledger, not a flag here. The shipped ``models/jef-v0`` is
    never written to.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from jef_core import BilinearHead, Engine
from jef_core.calibration import MIN_CORRECTNESS_SAMPLES, Calibrator
from jef_core.types import ChoiceAnswer, NoulAnswer, ScoreAnswer, normalize
from jef_scene.schema import Scene

from ..schema import Sample, assert_no_group_leakage, read_samples, write_samples
from .records import trust_tier
from .triage import TASK_OF_QUESTION, Pack
from .triage_eval import scene_questions
from .triage_rca import label_prior
from .triage_synth import verify_triage_lock

__all__ = ["CandidateError", "check_candidate", "export_triage_corpus", "required_buckets"]

TIER_ORDER = "ABCDE"
SPLITS = ("train", "calibration", "test")
#: Never a candidate's output directory: the shipped artifact.
SHIPPED = "models/jef-v0"


class CandidateError(ValueError):
    pass


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def required_buckets(scene: Scene) -> dict[str, dict[str, Any]]:
    """``task|kind:n`` for every question the scene asks."""
    out = {}
    for qid, raw in scene_questions(scene).items():
        q = normalize(qid, raw)
        out[f"{q.task_id}|{q.kind}:{q.n_options}"] = {
            "question": qid,
            "task_id": q.task_id,
            "kind": q.kind,
            "n_options": q.n_options,
        }
    return out


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #


def _label_index(kind: str, keys: list[str], gold: Any) -> int:
    if kind == "noul":
        return 1 if gold is True else 0
    return keys.index(gold[0])  # the primary acceptable answer


def export_triage_corpus(
    pack: Pack,
    lock: Mapping[str, Any],
    scene: Scene,
    out: Path,
    *,
    min_tier: str = "B",
    dry_run: bool = False,
    with_vendor_signals: bool = False,
) -> dict[str, Any]:
    """Write a ``fit`` corpus from the locked pack. Raises before writing anything.

    Cases whose state shows the detector's own output (``vendor_signals``)
    are left out unless ``with_vendor_signals`` asks for them.
    """
    if min_tier not in TIER_ORDER:
        raise CandidateError(f"min_tier must be one of {TIER_ORDER}")
    if dry_run:
        min_tier = "E"
    elif min_tier in "CDE":
        raise CandidateError(
            f"tier {min_tier} labels cannot train a deployable candidate; "
            "use dry_run=True to exercise the pipeline on them"
        )
    exps = [pack.expectations[c["case_id"]] for c in pack.cases]
    problems = verify_triage_lock(lock, pack.cases, exps)
    if problems:
        raise CandidateError("inputs are not the locked bytes: " + "; ".join(problems))
    split_of = {cid: name for name, ids in lock["splits"].items() for cid in ids}
    questions = scene_questions(scene)

    samples: list[Sample] = []
    excluded: Counter[str] = Counter()
    for case in pack.cases:
        e = pack.expectations[case["case_id"]]
        if case.get("vendor_signals") and not with_vendor_signals:
            # The detector's own verdict in the state: training on it would
            # credit the model for repeating Cloudflare (the S5 arm opts in).
            excluded["vendor_signals"] += 1
            continue
        tier = trust_tier(e["label_origin"])
        if TIER_ORDER.index(tier) > TIER_ORDER.index(min_tier):
            excluded[f"tier_{tier}"] += 1
            continue
        if e["label_status"] not in ("confirmed", "by_construction"):
            excluded[f"label_{e['label_status']}"] += 1
            continue
        for qid, task in TASK_OF_QUESTION.items():
            gold = e["expected"][task]
            if gold == "unknown":
                excluded[f"{task}_unknown"] += 1
                continue
            q = normalize(qid, questions[qid])
            keys = list(q.option_keys)
            samples.append(
                Sample(
                    state=case["state"],
                    kind=q.kind,
                    instructions=q.instructions,
                    option_keys=keys,
                    option_labels=[str(x) if x is not None else "" for x in q.option_labels],
                    label=_label_index(q.kind, keys, gold),
                    source=f"incident_triage.{task}",
                    group=case["case_family"],
                    lang=case["language"],
                    task=q.task_id,
                    meta={
                        "case_id": case["case_id"],
                        "label_origin": e["label_origin"],
                        "trust_tier": tier,
                        "split": split_of[case["case_id"]],
                    },
                )
            )
    if not samples:
        raise CandidateError(f"no labels at tier {min_tier} or better; nothing to export")
    by_split: dict[str, list[Sample]] = {name: [] for name in SPLITS}
    for s in samples:
        by_split[str(s.meta["split"])].append(s)
    for required in ("train", "calibration"):
        if not by_split[required]:
            raise CandidateError(f"the {required} split is empty; cannot fit")
    assert_no_group_leakage(*(by_split[k] for k in SPLITS))

    out.mkdir(parents=True, exist_ok=True)
    counts = {name: write_samples(rows, out / f"{name}.jsonl") for name, rows in by_split.items()}
    counts["all"] = write_samples(samples, out / "all.jsonl")
    manifest = {
        "builder": "jef_train.socbench.triage_candidate",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "deployable": False if dry_run else None,
        "min_trust_tier": min_tier,
        "lock": {"cases_digest": lock["cases_digest"], "seed": lock["seed"]},
        "pack": pack.manifest,
        "splits": counts,
        "buckets": {name: dict(Counter(s.bucket for s in rows)) for name, rows in by_split.items()},
        "excluded": dict(excluded),
        "note": (
            "DRY RUN on synthetic labels: exercises the pipeline only; never deployable"
            if dry_run
            else "test split is reported on, never fitted on"
        ),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    return manifest


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #


def _distribution(answer: Any, keys: list[str]) -> np.ndarray:
    if isinstance(answer, NoulAnswer):
        return np.array([1.0 - answer.value, answer.value])
    if isinstance(answer, (ChoiceAnswer, ScoreAnswer)):
        return np.array([answer.probabilities[k] for k in keys])
    raise CandidateError(f"unsupported answer {type(answer).__name__}")


def check_candidate(
    candidate: Path,
    corpus: Path,
    scene: Scene,
    *,
    backbone: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Everything a reviewer needs before calling a candidate more than ``insufficient``."""
    candidate, corpus = Path(candidate), Path(corpus)
    if repo_root is not None and candidate.resolve() == (repo_root / SHIPPED).resolve():
        raise CandidateError(f"{SHIPPED} is the shipped artifact, never a candidate")
    head_path, cal_path = candidate / "head.npz", candidate / "calibration.json"
    calibrator = Calibrator.load(str(cal_path))
    manifest = json.loads((corpus / "manifest.json").read_text("utf-8"))
    problems: list[str] = []

    # Artifact identity: the calibration names the head and corpus it was fitted for.
    art = calibrator.artifact
    if art is None or art.model_sha256 != _sha(head_path):
        problems.append("calibration.json was not fitted for this head.npz")
    if art is None or art.dataset_digest != _sha(corpus / "manifest.json"):
        problems.append("calibration.json was not fitted on this corpus")
    if calibrator.legacy:
        problems.append("calibration is schema 1 (legacy)")

    # Leakage: no family on two sides.
    splits = {name: list(read_samples(corpus / f"{name}.jsonl")) for name in SPLITS}
    try:
        assert_no_group_leakage(*(splits[k] for k in SPLITS))
    except ValueError as exc:
        problems.append(f"split leakage: {exc}")

    # Every bucket the scene asks, at the scene's alpha.
    buckets: dict[str, Any] = {}
    engine = Engine(backbone, BilinearHead.load(str(head_path)), calibrator)
    encoded: dict[str, Any] = {}
    test_by_bucket: dict[str, list[Sample]] = {}
    for s in splits["test"]:
        test_by_bucket.setdefault(s.bucket, []).append(s)
    for key, info in required_buckets(scene).items():
        b = calibrator.buckets.get(key)
        status = calibrator.status(info["kind"], info["n_options"], scene.alpha, info["task_id"])
        entry: dict[str, Any] = {
            "status": status,
            "calibration_records": b.n_samples if b else 0,
            "min_records": MIN_CORRECTNESS_SAMPLES,
        }
        test = test_by_bucket.get(key, [])
        if test:
            probs = []
            for s in test:
                # One backbone encode per state, shared by its five tasks.
                if s.state not in encoded:
                    encoded[s.state] = engine.prepare(s.state)
                ans = engine.answer(encoded[s.state], {"q": s.to_question() | {"task_id": s.task}})
                probs.append(_distribution(ans["q"], list(s.option_keys)))
            entry["coverage"] = calibrator.coverage_check(
                probs, [s.label for s in test], info["kind"], info["n_options"],
                scene.alpha, info["task_id"],
            )  # fmt: skip
        else:
            entry["coverage"] = {"status": "no_test_cases", "n": 0}
        buckets[key] = entry

    # One rule: research only when nothing is missing. Coverage that is
    # violated, too thin to judge, or untested is as much a gap as data.
    all_validated = all(b["status"] == "validated" for b in buckets.values())
    coverage_ok = all(b["coverage"].get("status") == "holds" for b in buckets.values())
    dry = bool(manifest.get("dry_run"))
    verdict = (
        "research" if all_validated and coverage_ok and not problems and not dry else "insufficient"
    )
    return {
        "candidate": str(candidate),
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": verdict,
        "status_ceiling": "research",
        "why_not_higher": (
            "promotion needs Tier A human labels and a validation-ledger decision; "
            "this check never grants it"
        ),
        "dry_run": bool(manifest.get("dry_run")),
        "min_trust_tier": manifest.get("min_trust_tier"),
        "problems": problems,
        "buckets": buckets,
        "artifacts": {
            "head_sha256": _sha(head_path),
            "calibration_sha256": _sha(cal_path),
            "corpus_manifest_sha256": _sha(corpus / "manifest.json"),
        },
        "splits": {k: len(v) for k, v in splits.items()},
        # R5 of the triage RCA on every candidate: what it answers with no content.
        "label_prior_R5": label_prior(engine, scene),
    }
