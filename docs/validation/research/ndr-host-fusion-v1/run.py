#!/usr/bin/env python3
"""SOURCE-LABEL RESEARCH: Behaviour + host-context fusion, and transfer to IoT-23.

Fixed before any score on these cases was seen (2026-10-18).

**Cases.** ``jef-bench build --source ctu-13 --unit host-fusion`` (task
``soc.ndr.host.v1``, label origin ``host-behaviour-label``, Tier C): one
source host over 300 s; gold by behaviour inside the window (``c2`` for a C&C
flow, ``botnet`` for spam, ads, attempts or floods, ``unknown`` for an
infected host's quiet window, ``benign`` for a verified-normal host);
conversation summaries as evidence; a per-minute series of flows and
destinations; addresses masked; the same sampling and caps as host-window v1.
Byte-identical states are removed before the folds. ``jef-bench build
--source iot-23`` builds the same kind of case from IoT-23's pinned subset.

**Folds and scorers.** As in ``../ndr-host-window-v1``: hold out one CTU-13
scenario, calibrate on the next two, train on ten; ``rules``, ``logreg``,
``jef`` (plain) and ``jef_cost`` (cost weight chosen per fold on the
calibration scenarios only). ``jef_cost`` is the primary JEF result.

**Transfer.** One more ``jef_cost`` head and ``logreg``, trained on CTU-13
scenarios 1-11 with scenarios 12 and 13 as calibration (weight chosen there),
scored on every IoT-23 case. IoT-23 shares no host with CTU-13: it is the only
unseen-host test available, because every infected and verified-normal CTU-13
host occurs in several scenarios (``audit-ctu-13-fusion.json``).

**Criteria, fixed in advance (CTU-13 folds, ``jef_cost``).**

C1 decision quality: median macro-F1 >= 0.70 and >= logreg's minus 0.03.
C2 miss rate: median miss rate at 1% FPR (infected vs benign) <= 0.10.
C3 confident errors: <= 2.0 mistaken closures per 1,000 attack windows.
C4 coverage: ``coverage_check`` holds in >= 75% of the folds that have enough
   cases for a verdict.
C5 gate safety: the calibration-fitted close gate closes no attack window on
   any fold.
C6 prioritisation (secondary): median recall of attack windows within the
   top 10% of each held-out scenario's windows, ranked by P(c2) + P(botnet),
   is at least logreg's.

**Transfer criteria (IoT-23, ``jef_cost`` trained on CTU-13).** T1 PR-AUC for
attack vs benign windows >= 0.80; T2 miss rate at 5% FPR <= 0.30; T3 the gate
fitted on CTU-13 calibration closes no IoT-23 attack window. IoT-23's benign
windows come from two honeypot captures only (134 windows), so T1 and T2 are
read with that in mind.

Run from the repository root after both builds:

    uv run python docs/validation/research/ndr-host-fusion-v1/run.py
"""

from __future__ import annotations

import importlib.util
import json
import logging
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from jef_core.backends import load_backbone
from jef_core.backends.pins import resolve_backbone_spec
from jef_train.cache import FeatureCache, build_feature_cache
from jef_train.fit import main as fit_main
from jef_train.schema import write_samples
from jef_train.socbench.annotate import NEGATIVE_LABELS, TASK_LABELS
from jef_train.socbench.baselines import LogisticScorer, RuleScorer
from jef_train.socbench.corpus_export import samples_from_gold
from jef_train.socbench.gate import fit_close_gate
from jef_train.socbench.leakage import audit
from jef_train.socbench.metrics import average_precision, fnr_at_fpr
from jef_train.socbench.records import read_jsonl
from jef_train.socbench.report import POSITIVE_LABELS, evaluate
from jef_train.socbench.splits import dedupe_states, digest_records, make_lock


def _json(o: Any) -> Any:
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)


HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "ndr_v1", HERE.parent / "ndr-head-and-transfer-v1" / "run.py"
)
if _spec is None or _spec.loader is None:
    raise SystemExit("cannot load the v1 study helpers")
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)

TASK = "soc.ndr.host.v1"
LABELS = list(TASK_LABELS[TASK])
ATTACK = sorted(POSITIVE_LABELS[TASK])
CLOSING = sorted(set(LABELS) & set(NEGATIVE_LABELS))
ATTACK_IDX = [LABELS.index(x) for x in ATTACK]
CLOSING_IDX = [LABELS.index(x) for x in CLOSING]
AUTO, ALPHA, LAMBDAS = 0.9, 0.10, (1.0, 2.0, 4.0)
CTU = Path(".cache/socbench/derived/ctu-13-fusion")
IOT = Path(".cache/socbench/derived/iot-23-fusion")
WORK = Path(".cache/socbench/research/ndr-fusion-v1")
FEATURES = WORK / "features"
BACKBONE = "jhu-clsp/mmBERT-base"
COST_FLAGS = ["--attack-labels", ",".join(ATTACK), "--closing-labels", ",".join(CLOSING)]
log = logging.getLogger("research")


def features(samples: list) -> FeatureCache:  # type: ignore[type-arg]
    model_id, revision = resolve_backbone_spec(BACKBONE)
    path = FEATURES / f"all.{model_id.replace('/', '__')}@{revision}.npz"
    if path.exists():
        cache = FeatureCache.load(path)
        if set(cache.uids) >= {s.uid for s in samples}:
            return cache
    started = time.perf_counter()
    cache = build_feature_cache(samples, load_backbone(BACKBONE))
    log.warning("encoded %d cases in %.0fs", len(samples), time.perf_counter() - started)
    FEATURES.mkdir(parents=True, exist_ok=True)
    cache.save(path)
    return cache


def train(name: str, splits: Mapping[str, list], extra: list[str]) -> Path:  # type: ignore[type-arg]
    corpus, out = WORK / "folds" / name / "corpus", WORK / "folds" / name / "model"
    for split, items in splits.items():
        write_samples(items, corpus / f"{split}.jsonl")
    write_samples([s for items in splits.values() for s in items], corpus / "all.jsonl")
    argv = [
        "--corpus",
        str(corpus),
        "--out",
        str(out),
        "--cache-dir",
        str(FEATURES),
        "--backbone",
        BACKBONE,
        *extra,
    ]
    if fit_main(argv) not in (0, None):
        raise SystemExit(f"fit failed for {name}")
    return out


def mistaken(p: np.ndarray, gold: str) -> bool:
    k = int(np.argmax(p))
    return LABELS[k] in NEGATIVE_LABELS and p[k] >= AUTO and gold in ATTACK


def attack_score(p: np.ndarray) -> float:
    return float(p[ATTACK_IDX].sum())


def top10_recall(scores: Sequence[float], is_attack: Sequence[bool]) -> float | None:
    """Share of attack windows within the top 10% of windows by attack score."""
    n_att = sum(is_attack)
    if not n_att or not scores:
        return None
    k = max(1, round(0.10 * len(scores)))
    order = np.argsort(-np.asarray(scores), kind="mergesort")[:k]
    return round(sum(is_attack[i] for i in order) / n_att, 4)


def choose_cost(
    held: str,
    splits: Mapping[str, list],
    ids: Mapping[str, list],
    cache: FeatureCache,
    uid_of: Mapping[str, str],
    gold: Mapping[str, str],  # type: ignore[type-arg]
) -> tuple[Any, float]:
    choice = []
    for lam in LAMBDAS:
        jef = v1.CachedJef(
            train(f"{held}-cost{lam:g}", splits, ["--fn-cost", f"{lam:g}", *COST_FLAGS]),
            cache,
            uid_of,
            TASK,
        )
        cal = [(jef.probs(c), c) for c in ids["calibration"]]
        mc = sum(mistaken(p, gold[c]) for p, c in cal)
        known = [(p, c) for p, c in cal if gold[c] in ATTACK or gold[c] in CLOSING]
        fnr = fnr_at_fpr(
            np.array([attack_score(p) for p, _ in known]),
            np.array([gold[c] in ATTACK for _, c in known]),
            0.01,
        ).get("fnr")
        choice.append((mc, 1.0 if fnr is None else float(fnr), lam, jef))
    choice.sort(key=lambda t: (t[0], t[1], t[2]))
    return choice[0][3], choice[0][2]


def scorer_block(
    name: str,
    fn: Callable[[str], np.ndarray],
    test_ids: list[str],
    cal_ids: list[str],
    gold: Mapping[str, str],
    calibrator: Any = None,
    fpr: float = 0.01,
) -> dict[str, Any]:
    tp = [fn(c) for c in test_ids]
    cp = [fn(c) for c in cal_ids]
    t_att = [gold[c] in ATTACK for c in test_ids]
    t_clo = [gold[c] in CLOSING for c in test_ids]
    known = [i for i, c in enumerate(test_ids) if gold[c] in ATTACK or gold[c] in CLOSING]
    scores = [attack_score(tp[i]) for i in known]
    pos = [t_att[i] for i in known]
    out: dict[str, Any] = {
        "mistaken_closure_ids": [
            c for p, c in zip(tp, test_ids, strict=True) if mistaken(p, gold[c])
        ],
        "pr_auc_attack_vs_benign": average_precision(np.array(scores), np.array(pos))
        if 0 < sum(pos) < len(pos)
        else None,
        f"fnr_at_fpr_{int(fpr * 100)}pct": fnr_at_fpr(np.array(scores), np.array(pos), fpr).get(
            "fnr"
        )
        if 0 < sum(pos) < len(pos)
        else None,
        "top10_recall": top10_recall([attack_score(p) for p in tp], t_att),
    }
    out["mistaken_closures"] = len(out["mistaken_closure_ids"])
    gate = fit_close_gate(
        [float(p[CLOSING_IDX].sum()) for p in cp], [gold[c] in ATTACK for c in cal_ids]
    )
    out["gate"] = {
        "threshold": gate.threshold,
        **gate.evaluate([float(p[CLOSING_IDX].sum()) for p in tp], t_att, t_clo),
    }
    if calibrator is not None:
        check = calibrator.coverage_check(
            tp, [LABELS.index(gold[c]) for c in test_ids], "choice", len(LABELS), ALPHA, TASK
        )
        out["coverage"] = {k: check.get(k) for k in ("status", "empirical", "wilson_95")}
    return out


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ctu_raw = (read_jsonl(CTU / "cases.jsonl"), read_jsonl(CTU / "gold.jsonl"))
    cases, golds, dd = dedupe_states(*ctu_raw)
    iot_cases, iot_golds, iot_dd = dedupe_states(
        read_jsonl(IOT / "cases.jsonl"), read_jsonl(IOT / "gold.jsonl")
    )
    all_cases, all_golds = cases + iot_cases, golds + iot_golds
    gold = {g["case_id"]: g["gold"]["verdict"] for g in all_golds}
    gold_by = {g["case_id"]: g for g in all_golds}
    by_id = {c["case_id"]: c for c in all_cases}
    samples, _ = samples_from_gold(
        all_cases, all_golds, {c["case_id"]: "all" for c in all_cases}, min_tier="C"
    )
    sample_of = {s.meta["case_id"]: s for s in samples}
    uid_of = {cid: s.uid for cid, s in sample_of.items()}
    cache = features(samples)
    (HERE / "audit-ctu-13-fusion.json").write_text(json.dumps(audit(cases, golds), indent=2) + "\n")
    fams = v1.family_order([c["family_id"] for c in cases])
    folds: list[dict[str, Any]] = []

    for k, held in enumerate(fams):
        calib = {fams[(k + 1) % len(fams)], fams[(k + 2) % len(fams)]}
        pins = {f: "test" if f == held else "calibration" if f in calib else "train" for f in fams}
        lock = make_lock(cases, golds, seed=f"ndr-fusion-v1-{held}", pinned=pins)
        ids = {sp: list(v) for sp, v in lock["splits"].items()}
        splits = {sp: [sample_of[c] for c in v] for sp, v in ids.items()}
        lr = LogisticScorer().fit([by_id[c] for c in ids["train"]], gold_by)

        def lr_probs(cid: str, _lr: LogisticScorer = lr) -> np.ndarray:
            d = _lr.predict(by_id[cid])
            return np.array([d[x] for x in LABELS])

        plain = v1.CachedJef(train(f"{held}-plain", splits, []), cache, uid_of, TASK)
        cost, lam = choose_cost(held, splits, ids, cache, uid_of, gold)
        plain.name, cost.name = "jef", "jef_cost"
        report = evaluate(cases, golds, lock, [RuleScorer(), lr, plain, cost], split="test")
        row: dict[str, Any] = {
            "held_out": held,
            "n_test": len(ids["test"]),
            "test_gold": dict(Counter(gold[c] for c in ids["test"])),
            "lambda": lam,
        }
        for name in ("rules", "logreg", "jef", "jef_cost"):
            block = report["scorers"][name]["tasks"][TASK]
            entry: dict[str, Any] = {"macro_f1": block.get("macro_f1"), "ece": block.get("ece")}
            if name != "rules":
                fn = {"logreg": lr_probs, "jef": plain.probs, "jef_cost": cost.probs}[name]
                cal = None if name == "logreg" else (plain if name == "jef" else cost).calibrator
                entry |= scorer_block(name, fn, ids["test"], ids["calibration"], gold, cal)
            else:
                entry["fnr_at_fpr_1pct"] = (
                    block.get("binary", {}).get("fnr_at_fpr_1pct") or {}
                ).get("fnr")
                entry["mistaken_closures"] = block.get("unknown_handling", {}).get(
                    "mistaken_closures"
                )
            row[name] = entry
        folds.append(row)
        print(
            f"[fusion] {held:18s} n={row['n_test']:4d} lam={lam:g}  "
            + "  ".join(
                f"{n} F1 {row[n]['macro_f1']} FNR {row[n].get('fnr_at_fpr_1pct')} MC {row[n]['mistaken_closures']}"
                for n in ("logreg", "jef", "jef_cost")
            ),
            file=sys.stderr,
        )

    # -- transfer: train on CTU-13 1-11, calibrate on 12-13, score IoT-23
    calib = {fams[-2], fams[-1]}
    pins = {f: "calibration" if f in calib else "train" for f in fams}
    lock = make_lock(cases, golds, seed="ndr-fusion-v1-transfer", pinned=pins)
    ids = {sp: list(v) for sp, v in lock["splits"].items()} | {
        "test": [c["case_id"] for c in iot_cases]
    }
    splits = {sp: [sample_of[c] for c in v] for sp, v in ids.items()}
    lr = LogisticScorer().fit([by_id[c] for c in ids["train"]], gold_by)
    cost, lam = choose_cost("transfer", splits, ids, cache, uid_of, gold)

    def lr_t(cid: str) -> np.ndarray:
        d = lr.predict(by_id[cid])
        return np.array([d[x] for x in LABELS])

    transfer = {
        "train": "CTU-13 scenarios 1-11 (fusion)",
        "calibration": sorted(calib),
        "test": "IoT-23 subset (fusion)",
        "lambda": lam,
        "test_gold": dict(Counter(gold[c] for c in ids["test"])),
        "logreg": scorer_block("logreg", lr_t, ids["test"], ids["calibration"], gold, fpr=0.05),
        "jef_cost": scorer_block(
            "jef_cost", cost.probs, ids["test"], ids["calibration"], gold, cost.calibrator, fpr=0.05
        ),
    }
    print(
        f"[transfer] jef_cost {transfer['jef_cost']['pr_auc_attack_vs_benign']} logreg {transfer['logreg']['pr_auc_attack_vs_benign']}",
        file=sys.stderr,
    )

    # -- summary and verdicts
    attack_total = sum(sum(v for k2, v in f["test_gold"].items() if k2 in ATTACK) for f in folds)

    def med(n: str, key: str) -> float | None:
        vals = [f[n][key] for f in folds if f[n].get(key) is not None]
        return round(statistics.median(vals), 4) if vals else None

    summary: dict[str, Any] = {}
    for n in ("rules", "logreg", "jef", "jef_cost"):
        mc = sum(f[n]["mistaken_closures"] or 0 for f in folds)
        s: dict[str, Any] = {
            "median_macro_f1": med(n, "macro_f1"),
            "median_fnr_at_fpr_1pct": med(n, "fnr_at_fpr_1pct"),
            "median_pr_auc": med(n, "pr_auc_attack_vs_benign"),
            "median_top10_recall": med(n, "top10_recall"),
            "median_ece": med(n, "ece"),
            "mistaken_closures": mc,
            "mistaken_closures_per_1000_attacks": round(1000 * mc / attack_total, 2)
            if attack_total
            else None,
        }
        if "gate" in folds[0][n]:
            s["gate_closed_attacks"] = sum(f[n]["gate"]["closed_attacks"] for f in folds)
            s["gate_closed_benign"] = sum(f[n]["gate"]["closed_benign"] for f in folds)
        if "coverage" in folds[0][n]:
            s["coverage_check"] = dict(Counter(f[n]["coverage"]["status"] for f in folds))
        summary[n] = s
    p, lg = summary["jef_cost"], summary["logreg"]
    cov = p["coverage_check"]
    judged = cov.get("holds", 0) + cov.get("violated", 0)
    verdict = {
        "C1_macro_f1": (p["median_macro_f1"] or 0) >= 0.70
        and (p["median_macro_f1"] or 0) >= (lg["median_macro_f1"] or 0) - 0.03,
        "C2_fnr": (p["median_fnr_at_fpr_1pct"] if p["median_fnr_at_fpr_1pct"] is not None else 1)
        <= 0.10,
        "C3_confident_errors": (p["mistaken_closures_per_1000_attacks"] or 0) <= 2.0,
        "C4_coverage": judged > 0 and cov.get("holds", 0) / judged >= 0.75,
        "C5_gate_safe": p["gate_closed_attacks"] == 0,
        "C6_prioritisation": (p["median_top10_recall"] or 0) >= (lg["median_top10_recall"] or 0),
        "T1_transfer_pr_auc": (transfer["jef_cost"]["pr_auc_attack_vs_benign"] or 0) >= 0.80,
        "T2_transfer_fnr_at_5pct": (
            transfer["jef_cost"]["fnr_at_fpr_5pct"]
            if transfer["jef_cost"]["fnr_at_fpr_5pct"] is not None
            else 1
        )
        <= 0.30,
        "T3_transfer_gate_safe": transfer["jef_cost"]["gate"]["closed_attacks"] == 0,
    }
    results = {
        "evidence_class": "SOURCE-LABEL RESEARCH",
        "task": TASK,
        "backbone": {"id": BACKBONE, "revision": resolve_backbone_spec(BACKBONE)[1]},
        "inputs": {
            "ctu": {
                "before_dedupe": len(ctu_raw[0]),
                "cases": len(cases),
                "dropped_same_label": len(dd["dropped_same_label"]),
                "dropped_conflicting": len(dd["dropped_conflicting_labels"]),
                "cases_digest": digest_records(cases),
                "gold_digest": digest_records(golds),
                "gold_distribution": dict(Counter(g["gold"]["verdict"] for g in golds)),
                "build": json.loads((CTU / "summary.json").read_text()),
            },
            "iot23": {
                "cases": len(iot_cases),
                "dropped": len(iot_dd["dropped_same_label"])
                + len(iot_dd["dropped_conflicting_labels"]),
                "cases_digest": digest_records(iot_cases),
                "gold_digest": digest_records(iot_golds),
                "gold_distribution": dict(Counter(g["gold"]["verdict"] for g in iot_golds)),
                "build": json.loads((IOT / "summary.json").read_text()),
            },
            "attack_windows_ctu": attack_total,
        },
        "summary": summary,
        "transfer": transfer,
        "verdict": verdict,
        "folds": folds,
    }
    (HERE / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=_json) + "\n"
    )
    print(
        json.dumps(
            {
                "summary": summary,
                "verdict": verdict,
                "transfer": {k: transfer[k] for k in ("logreg", "jef_cost")},
            },
            indent=1,
            default=_json,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
