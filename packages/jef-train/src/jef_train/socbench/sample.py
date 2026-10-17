"""Family-aware, stratified, reproducible sampling of cases for annotation.

The annotation pipeline (``annotate export`` → two reviewers → ``agree`` →
``adjudicate`` → ``lock``) needs a ``sampled-cases.jsonl``: an authorised
subset small enough for two analysts to label. Picking it by hand invites the
two mistakes that make a benchmark lie -- choosing cases the model handles
well, and splitting an incident family across train and test. This module
picks it by rule instead:

* **Within families, never across them.** Cases are sampled per family, and
  the lock later assigns whole families to splits, so a sampled family never
  straddles the train/test line.
* **Stratified on the weak label, blind to it in the output.** Draws are
  water-filled across weak-label strata, rarest first, so a rare class (a C2
  conversation among thousands of benign flows) is taken in full before common
  ones grow, and ``per_family`` is a hard cap. The weak verdict is used only to
  choose; the written cases carry no label, so reviewers stay blind.
* **Weights to undo the oversampling.** Every sampled case carries its inverse
  inclusion probability within its family and stratum, recorded in the
  manifest, so metrics can be re-weighted back to the source's base rate.
* **Reproducible and additive.** A seed fixes the draw; ``exclude`` lets a
  second batch (after a 50-case pilot) skip what was already sampled.

The manifest also says plainly when the source has too few independent
families for a family-grouped split to produce meaningful confidence
intervals. AIT-ADS has 8 scenarios and CTU-13 has 13: a 60/20/20 split leaves
one to three families in test, and a cluster bootstrap over that few clusters
is not a confidence interval anyone should quote.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from .splits import digest_records

__all__ = ["MIN_FAMILIES_FOR_CI", "SampleError", "sample_cases"]

#: Below this many families a family-grouped test split holds too few clusters
#: for the bootstrap intervals in the report to mean much. A warning, not a stop.
MIN_FAMILIES_FOR_CI = 10
UNLABELLED = "(no weak label)"


class SampleError(ValueError):
    """The request cannot be satisfied honestly."""


def _rng(seed: str, family: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}\x00{family}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))  # noqa: S311 - sampling, not security


def _allocate(sizes: Mapping[str, int], target: int) -> dict[str, int]:
    """Water-fill ``target`` draws across strata, rarest first, never past a stratum's size.

    Every stratum gets one draw before any gets a second, so a rare class is
    taken in full before common ones grow, and the total is exactly
    ``min(target, available)``: ``per_family`` is a hard cap.
    """
    take = dict.fromkeys(sizes, 0)
    remaining = min(target, sum(sizes.values()))
    order = sorted(sizes, key=lambda s: (sizes[s], s))
    while remaining > 0:
        progressed = False
        for stratum in order:
            if remaining == 0:
                break
            if take[stratum] < sizes[stratum]:
                take[stratum] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return take


def sample_cases(
    cases: Sequence[Mapping[str, Any]],
    weak_golds: Iterable[Mapping[str, Any]] = (),
    *,
    task: str | None = None,
    per_family: int = 60,
    seed: str = "socbench-sample",
    families: Iterable[str] | None = None,
    exclude: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """``(sampled cases, manifest)``. The cases carry no label; the manifest does the accounting."""
    if per_family < 1:
        raise SampleError("per_family must be at least 1")
    weak = {g["case_id"]: g["gold"]["verdict"] for g in weak_golds}
    skip = set(exclude)
    wanted_families = set(families) if families is not None else None

    pool: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for c in cases:
        if task and c["task_id"] != task:
            continue
        if c["case_id"] in skip:
            continue
        if wanted_families is not None and c["family_id"] not in wanted_families:
            continue
        pool[c["family_id"]][weak.get(c["case_id"], UNLABELLED)].append(c)
    if not pool:
        raise SampleError("no cases match the task, family and exclusion filters")

    chosen: list[dict[str, Any]] = []
    weights: dict[str, float] = {}
    by_family: dict[str, Any] = {}
    for fam in sorted(pool):
        strata = pool[fam]
        sizes = {s: len(v) for s, v in strata.items()}
        take = _allocate(sizes, per_family)
        rng = _rng(seed, fam)
        fam_rows: dict[str, list[int]] = {}
        for stratum in sorted(strata):
            rows = sorted(strata[stratum], key=lambda c: c["case_id"])
            picked = rng.sample(rows, take[stratum]) if take[stratum] else []
            for c in sorted(picked, key=lambda c: c["case_id"]):
                chosen.append(dict(c))
                weights[c["case_id"]] = round(sizes[stratum] / take[stratum], 6)
            fam_rows[stratum] = [sizes[stratum], take[stratum]]
        by_family[fam] = {
            "available": sum(sizes.values()),
            "sampled": sum(take.values()),
            "strata": {s: {"available": a, "sampled": t} for s, (a, t) in fam_rows.items()},
        }

    n_fam = len(by_family)
    warnings: list[str] = []
    if n_fam < MIN_FAMILIES_FOR_CI:
        warnings.append(
            f"only {n_fam} independent families: a family-grouped 60/20/20 split leaves about "
            f"{max(1, round(n_fam * 0.2))} in test, too few clusters for the report's bootstrap "
            "intervals to be meaningful. Report the family count beside every metric, or add "
            "sources with more independent incidents."
        )
    strata_total: Counter[str] = Counter()
    for f in by_family.values():
        for s, v in f["strata"].items():
            strata_total[s] += v["sampled"]
    if UNLABELLED in strata_total:
        warnings.append("some cases had no weak label and were sampled as one stratum")

    manifest: dict[str, Any] = {
        "sampler": "jef_train.socbench.sample",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "task": task,
        "per_family": per_family,
        "allocation": "water-filled across weak-label strata, rarest first; per_family is a cap",
        "input_cases_digest": digest_records(cases),
        "weak_labels_used_for_stratification_only": bool(weak),
        "excluded": len(skip),
        "families": n_fam,
        "sampled": len(chosen),
        "sampled_by_weak_label": dict(strata_total),
        "by_family": by_family,
        "inverse_inclusion_weights": weights,
        "warnings": warnings,
        "note": (
            "Weak labels chose the strata and are not in the sampled cases, so reviewers stay "
            "blind. Use inverse_inclusion_weights to re-weight metrics to the source base rate."
        ),
    }
    return chosen, manifest
