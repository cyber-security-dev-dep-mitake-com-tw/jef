"""S1-S6: one locked pack, six training corpora that differ in one thing each.

The v5 plan's ablation matrix (§6.4), built so the arms can be compared
honestly:

=====  =============================================================  ========
arm    what changes                                                    fit
=====  =============================================================  ========
S1     human labels only; state as written; no hard negatives in        no class
       train; no own telemetry                                          weights
S2     S1 + the structured evidence rendered into the state             no class
                                                                        weights
S3     S2 + hard negatives back in train, class weighting               default
S4     the S3 corpus, judged by schema-2 calibration and coverage       default
       (``candidate-check``) rather than raw accuracy
S5     S3 + own telemetry, including vendor-signal variants             default
S6     S3 + own telemetry without vendor-signal variants                default
=====  =============================================================  ========

Every arm is cut from a single export of the locked pack, so the families
on each side of every split are the same in all arms: an arm changes what is
*trained on* or how a state is *rendered*, never which cases are tested. The
calibration and test case sets of S1-S4 are asserted identical; S5 and S6
add the own-telemetry cases to every split and are asserted identical to
each other apart from the vendor-signal variants.

Selection is on the calibration split. The test split is read once, after
the choice, and ``ablation.json`` says so; nothing here fits a model or reads
a score. On synthetic labels (``dry_run``) every arm is marked not deployable,
and ``candidate-check`` reports ``insufficient`` for each.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from jef_scene.schema import Scene

from ..schema import Sample, read_samples, write_samples
from .triage import Pack
from .triage_candidate import SPLITS, export_triage_corpus

__all__ = ["ARMS", "AblationError", "Arm", "build_ablation"]

#: Evidence a state never shows in the primary arms: the detector's own verdict.
HIDDEN_EVIDENCE = frozenset({"detector_output"})


class AblationError(ValueError):
    pass


@dataclass(frozen=True)
class Arm:
    name: str
    description: str
    evidence: bool
    hard_negatives: bool
    class_weighting: bool
    telemetry: str  # "none" | "with_vendor" | "without_vendor"
    judged_by: str = "decision metrics on the calibration split"


ARMS: dict[str, Arm] = {
    a.name: a
    for a in (
        Arm("S1", "human labels only, state as written", False, False, False, "none"),
        Arm("S2", "S1 + structured evidence in the state", True, False, False, "none"),
        Arm("S3", "S2 + hard negatives in train + class weighting", True, True, True, "none"),
        Arm(
            "S4",
            "the S3 corpus, judged by schema-2 calibration status and coverage",
            True,
            True,
            True,
            "none",
            judged_by="candidate-check: validated buckets and coverage_check on test",
        ),
        Arm(
            "S5", "S3 + own telemetry with vendor-signal variants", True, True, True, "with_vendor"
        ),
        Arm("S6", "S3 + own telemetry without vendor signals", True, True, True, "without_vendor"),
    )
}


def _with_evidence(case: Mapping[str, Any]) -> str:
    items = [e for e in case.get("evidence", []) if e["field"] not in HIDDEN_EVIDENCE]
    if not items:
        return str(case["state"])
    rendered = "; ".join(f"{e['field']}={e['value']}" for e in items)
    return f"{case['state']}\nEvidence: {rendered}"


def _keep(arm: Arm, case: Mapping[str, Any], split: str) -> str | None:
    """None to keep the sample, else the reason it is left out of this arm."""
    telemetry = case.get("origin") == "own-telemetry"
    if telemetry and arm.telemetry == "none":
        return "own_telemetry"
    if case.get("vendor_signals") and arm.telemetry != "with_vendor":
        return "vendor_signals"
    if split == "train" and case.get("hard_negative_of") and not arm.hard_negatives:
        return "hard_negative_in_train"
    return None


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def build_ablation(
    pack: Pack,
    lock: Mapping[str, Any],
    scene: Scene,
    out: Path,
    *,
    arms: Sequence[str] = tuple(ARMS),
    min_tier: str = "B",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Write ``out/<arm>/`` corpora and ``out/ablation.json``; fits nothing."""
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise AblationError(f"unknown arms {unknown}; choose from {list(ARMS)}")
    out = Path(out)
    full_dir = out / "_full"
    full = export_triage_corpus(
        pack, lock, scene, full_dir, min_tier=min_tier, dry_run=dry_run, with_vendor_signals=True
    )
    cases = {c["case_id"]: c for c in pack.cases}
    samples = list(read_samples(full_dir / "all.jsonl"))

    summary: dict[str, Any] = {}
    held: dict[str, set[tuple[str, str]]] = {}
    for name in arms:
        arm = ARMS[name]
        by_split: dict[str, list[Sample]] = {s: [] for s in SPLITS}
        left_out: Counter[str] = Counter()
        for s in samples:
            case = cases[str(s.meta["case_id"])]
            split = str(s.meta["split"])
            reason = _keep(arm, case, split)
            if reason:
                left_out[reason] += 1
                continue
            if arm.evidence and case.get("origin") != "own-telemetry":
                s = replace(s, state=_with_evidence(case))
            by_split[split].append(s)
        arm_dir = out / name
        arm_dir.mkdir(parents=True, exist_ok=True)
        counts = {k: write_samples(v, arm_dir / f"{k}.jsonl") for k, v in by_split.items()}
        counts["all"] = write_samples(
            [x for k in SPLITS for x in by_split[k]], arm_dir / "all.jsonl"
        )
        held[name] = {
            (str(x.meta["case_id"]), x.task or "")
            for k in ("calibration", "test")
            for x in by_split[k]
        }
        manifest = {
            "builder": "jef_train.socbench.triage_ablation",
            "arm": asdict(arm),
            "dry_run": dry_run,
            "deployable": False if dry_run else None,
            "min_trust_tier": full["min_trust_tier"],
            "lock": full["lock"],
            "pack": full["pack"],
            "splits": counts,
            "left_out": dict(left_out),
            "fit_args": [] if arm.class_weighting else ["--no-class-weighting"],
            "note": (
                "DRY RUN on synthetic labels: exercises the pipeline only; never deployable"
                if dry_run
                else "choose on the calibration split; the test split is read once, after"
            ),
        }
        (arm_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", "utf-8"
        )
        summary[name] = {
            "description": arm.description,
            "judged_by": arm.judged_by,
            "splits": counts,
            "left_out": dict(left_out),
            "corpus_sha256": _digest(arm_dir / "all.jsonl"),
            "fit": f"uv run python -m jef_train.fit --corpus {arm_dir} --out {out / 'models' / name} "
            + " ".join(manifest["fit_args"]),
            "check": (
                f"uv run jef-bench triage candidate-check --candidate {out / 'models' / name} "
                f"--corpus {arm_dir} --backbone <the backbone fit used> --out {out / 'checks' / name}.json"
            ),
        }

    # The arms differ in what is trained on, never in what is judged.
    base = [a for a in ("S1", "S2", "S3", "S4") if a in held]
    if any(held[a] != held[base[0]] for a in base[1:]):
        raise AblationError("S1-S4 do not hold out the same calibration and test cases")
    if "S5" in held and "S6" in held:
        vendor = {
            cid
            for cid, c in cases.items()
            if c.get("vendor_signals") and c.get("origin") == "own-telemetry"
        }
        if {x for x in held["S5"] if x[0] not in vendor} != held["S6"]:
            raise AblationError("S5 and S6 differ in more than the vendor-signal variants")

    report = {
        "builder": "jef_train.socbench.triage_ablation",
        "dry_run": dry_run,
        "selection": {
            "select_on": "calibration",
            "test": "read once, after the arm is chosen; never used to choose",
            "preregistration": "docs/validation/plans/2026-10-21-soc-head-to-head-preregistration.md",
        },
        "status_ceiling": "research",
        "arms": summary,
    }
    (out / "ablation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    return report
