#!/usr/bin/env python3
"""SOURCE-LABEL RESEARCH: CTU-13, soc.ndr.technique.v1, leave-one-scenario-out.

Reproduce (from the repository root, after `uv sync --all-packages`):

    uv run jef-bench fetch --source ctu-13 --pin && uv run jef-bench check --source ctu-13
    uv run jef-bench build --source ctu-13 --background-rate 0.001 --flow-labels host
    uv run jef-bench dedupe --cases .cache/socbench/derived/ctu-13/cases.jsonl \\
        --gold .cache/socbench/derived/ctu-13/gold.jsonl \\
        --out-cases .cache/socbench/research/ctu13-v1/cases.jsonl \\
        --out-gold .cache/socbench/research/ctu13-v1/gold.jsonl
    uv run python docs/validation/research/ctu13-source-label-v1/run.py

Design, fixed before any score was seen:

* Labels are CTU-13's own flow labels (Tier C). Background is ``unknown`` and
  never scored as benign; it is excluded from the binary metrics.
* CTU-13 has 13 scenario families. A seeded 60/20/20 family split put no family
  in test, so instead every scenario is held out once (13 folds): the scorers
  train on the other 12 and are scored on the held-out one. Rules do not train;
  logistic regression is refitted per fold on that fold's training families.
* Per-fold results are reported as they are. Folds without both positives and
  negatives have no PR-AUC; that is shown, not imputed.
* Two variants. ``as_built`` scores the cases as the adapter wrote them.
  ``ips_masked`` removes ``src`` and ``dst`` from the state and replaces every
  IPv4 address in evidence text with ``<ip>``. CTU-13 reuses the same hosts in
  all 13 scenarios (147.32.84.165 is always botnet/C2; .164 and .170 are always
  benign), so ``as_built`` largely measures whether a scorer memorised
  addresses. ``ips_masked`` is the number that says anything about behaviour.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from jef_train.socbench.baselines import LogisticScorer, RuleScorer
from jef_train.socbench.records import read_jsonl
from jef_train.socbench.report import evaluate
from jef_train.socbench.splits import dedupe_states, digest_records, make_lock

TASK = "soc.ndr.technique.v1"
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def mask_ips(case: dict) -> dict:
    state = {k: v for k, v in case["state"].items() if k not in {"src", "dst"}}
    evidence = [{**e, "value": IPV4.sub("<ip>", str(e["value"]))} for e in case["evidence"]]
    return {**case, "state": state, "evidence": evidence}


HERE = Path(__file__).resolve().parent
DATA = Path(".cache/socbench/research/ctu13-v1")


def run(cases: list[dict], golds: list[dict], label: str) -> dict:
    gold_by = {g["case_id"]: g for g in golds}
    families = sorted({c["family_id"] for c in cases}, key=lambda f: int(f.rsplit("-", 1)[1]))
    folds = []
    confusion: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for held in families:
        pins = {f: ("test" if f == held else "train") for f in families}
        lock = make_lock(cases, golds, seed=f"lofo-{held}", pinned=pins)
        train = [c for c in cases if c["case_id"] in set(lock["splits"]["train"])]
        scorers = [RuleScorer(), LogisticScorer().fit(train, gold_by)]
        report = evaluate(cases, golds, lock, scorers, split="test")
        row = {"held_out": held, "n": report["lock"]["n"]}
        for name, s in report["scorers"].items():
            block = s["tasks"].get(TASK, {})
            binary = block.get("binary", {})
            row[name] = {
                "macro_f1": block.get("macro_f1"),
                "accuracy": block.get("accuracy"),
                "ece": block.get("ece"),
                "pr_auc": binary.get("pr_auc"),
                "positives": binary.get("positives"),
                "n_known": binary.get("n_known"),
                "fnr_at_fpr_1pct": (binary.get("fnr_at_fpr_1pct") or {}).get("fnr"),
                "mistaken_closures": block.get("unknown_handling", {}).get("mistaken_closures"),
                "gold": block.get("gold_distribution"),
            }
            for g, preds in (block.get("confusion") or {}).items():
                confusion[name][g].update(preds)
        folds.append(row)
        print(
            f"[{label}] {held:18s} n={row['n']:5d}  rules F1={row['rules']['macro_f1']}  "
            f"logreg F1={row['logreg']['macro_f1']}",
            file=sys.stderr,
        )

    def med(name: str, key: str) -> float | None:
        vals = [f[name][key] for f in folds if f[name][key] is not None]
        return round(statistics.median(vals), 4) if vals else None

    return {
        "cases_digest": digest_records(cases),
        "median_over_folds": {
            name: {
                k: med(name, k)
                for k in ("macro_f1", "accuracy", "ece", "pr_auc", "fnr_at_fpr_1pct")
            }
            for name in ("rules", "logreg")
        },
        "folds_with_pr_auc": {
            n: sum(f[n]["pr_auc"] is not None for f in folds) for n in ("rules", "logreg")
        },
        "mistaken_closures_total": {
            n: sum(f[n]["mistaken_closures"] or 0 for f in folds) for n in ("rules", "logreg")
        },
        "pooled_confusion": {
            n: {g: dict(p) for g, p in sorted(c.items())} for n, c in confusion.items()
        },
        "folds": folds,
    }


def main() -> int:
    cases, golds = read_jsonl(DATA / "cases.jsonl"), read_jsonl(DATA / "gold.jsonl")
    summary = {
        "evidence_class": "SOURCE-LABEL RESEARCH",
        "task": TASK,
        "source": "ctu-13 (CC-BY), archive sha256 1f8daeca146131a368b432b2d8625de5e4429bce833f3e4cf4bea8312344ab7f",
        "label_origin": "flow-label (Tier C); Background = unknown",
        "design": "leave-one-scenario-out, 13 folds; rules untrained, logreg refitted per fold",
        "cases": len(cases),
        "gold_digest": digest_records(golds),
        "gold_distribution": dict(Counter(g["gold"]["verdict"] for g in golds)),
        "families": len({c["family_id"] for c in cases}),
        "variants": {
            "as_built": run(cases, golds, "as_built"),
        },
    }
    # Removing addresses makes flows that differed only by address identical;
    # the same dedupe rule applies (conflicting labels drop every copy).
    masked, masked_golds, dd = dedupe_states([mask_ips(c) for c in cases], golds)
    summary["variants"]["ips_masked"] = {
        "dedupe_after_masking": {
            "kept": dd["kept_cases"],
            "dropped_same_label": len(dd["dropped_same_label"]),
            "dropped_conflicting_labels": len(dd["dropped_conflicting_labels"]),
            "gold_distribution_kept": dict(Counter(g["gold"]["verdict"] for g in masked_golds)),
        },
        **run(masked, masked_golds, "ips_masked"),
    }
    (HERE / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    for v, r in summary["variants"].items():
        print(v, json.dumps(r["median_over_folds"]), "closures", r["mistaken_closures_total"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
