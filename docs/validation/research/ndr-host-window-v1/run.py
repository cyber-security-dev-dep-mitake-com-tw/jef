#!/usr/bin/env python3
"""SOURCE-LABEL RESEARCH: does host and time-window context fix the NDR head?

Fixed before any score on this data was seen (2026-10-18).

**Question.** v1 and v2 scored single CTU-13 conversations. Their confident
mistakes were flows from infected hosts that look benign on their own,
because CTU-13 labels hosts, not behaviours (``../ndr-head-v2``). Does a case
that *is* the labelled unit -- one source host over 300 seconds, with fan-out,
port mix, DNS/SMTP volume, unanswered share and periodicity, addresses masked
-- let JEF judge unseen scenarios better?

**Data.** ``jef-bench build --source ctu-13 --unit host-window`` (task
``soc.ndr.host.v1``, label origin ``host-label``, Tier C): every infected and
normal host, Background hosts sampled by host at 0.2%, at most 200 windows
per (scenario, verdict), windows with fewer than 3 flows dropped, then
byte-identical states removed (conflicting labels drop every copy).

**Folds.** The v1 rule: hold out one scenario, calibrate on the next two in
numeric order (cyclic), train on the other ten. 13 folds.

**Scorers.** ``rules`` (the host-window rule in ``baselines.RuleScorer``),
``logreg`` (refitted per fold on the training scenarios), ``jef`` (plain
``jef_train.fit``) and ``jef_cost`` (``--fn-cost`` with the weight chosen per
fold from {1, 2, 4} on the calibration scenarios only: fewest mistaken
closures, then lowest miss rate at 1% FPR, then the smaller weight).
``jef_cost`` is the primary JEF result, because the product's cost of a missed
infection is the reason for the study.

**Measures.** Macro-F1; PR-AUC and miss rate at 1% FPR for infected vs
benign; mistaken closures (an infected window called benign with p >= 0.9);
a close gate fitted on calibration and frozen (``socbench.gate``); conformal
coverage and ``Calibrator.coverage_check`` per held-out scenario.

**Success criteria, fixed in advance.**

1. *Better than the single-flow head (v1):* ``jef_cost`` median miss rate at
   1% FPR below v1's 0.374, mistaken closures per 1,000 attack cases below
   v1's rate, and median macro-F1 at least v1's 0.768. (A different unit and
   label set from v1, so this compares decision quality, not identical
   tasks.)
2. *Close to logreg on the same host windows:* median macro-F1 within 0.03,
   median miss rate at 1% FPR within 0.05, and no more mistaken closures.
3. *Gate safe:* the frozen gate closes no infected window on any test fold.

**Second arm: the same conversations, labelled by behaviour** (``run.py
flow-behaviour``). The v1 cases rebuilt with ``--flow-labels behaviour``: a
bot's DNS lookups and plain sessions become ``unknown`` instead of
``botnet``; C&C, spam, click-fraud ads, connection attempts and ICMP floods
keep their attack labels. Same masking (src/dst dropped, IPv4 in evidence
replaced), the same dedupe, folds, scorers and measures, task
``soc.ndr.technique.v1``. It asks whether the v1 errors were the labels or the
unit. The same three criteria apply against v1, with "attack" meaning a
behaviour-labelled attack. Results go to ``results-flow-behaviour.json``.

Run from the repository root (needs the torch extra and the cached
mmBERT-base weights):

    uv run python docs/validation/research/ndr-host-window-v1/run.py
    uv run jef-bench build --source ctu-13 --background-rate 0.001 --flow-labels behaviour \
        --out .cache/socbench/derived/ctu-13-behaviour
    uv run python docs/validation/research/ndr-host-window-v1/run.py flow-behaviour
"""

from __future__ import annotations

import importlib.util
import json
import logging
import statistics
import sys
import time
from collections import Counter
from collections.abc import Mapping
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
from jef_train.socbench.metrics import fnr_at_fpr
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

ARM = sys.argv[1] if len(sys.argv) > 1 else "host"
if ARM not in {"host", "flow-behaviour"}:
    raise SystemExit("usage: run.py [host|flow-behaviour]")
TASK = "soc.ndr.host.v1" if ARM == "host" else "soc.ndr.technique.v1"
LABELS = list(TASK_LABELS[TASK])
ATTACK = sorted(POSITIVE_LABELS[TASK])
CLOSING = sorted(set(LABELS) & set(NEGATIVE_LABELS))
ATTACK_IDX = [LABELS.index(x) for x in ATTACK]
CLOSING_IDX = [LABELS.index(x) for x in CLOSING]
AUTO, ALPHA, LAMBDAS = 0.9, 0.10, (1.0, 2.0, 4.0)
DATA = Path(".cache/socbench/derived/" + ("ctu-13-hosts" if ARM == "host" else "ctu-13-behaviour"))
WORK = Path(
    ".cache/socbench/research/" + ("ndr-host-v1" if ARM == "host" else "ndr-flow-behaviour-v1")
)
FEATURES = WORK / "features"
OUT = HERE / ("results.json" if ARM == "host" else "results-flow-behaviour.json")
BACKBONE = "jhu-clsp/mmBERT-base"
COST_FLAGS = ["--attack-labels", ",".join(ATTACK), "--closing-labels", ",".join(CLOSING)]
V1_REFERENCE = {"median_macro_f1": 0.768, "median_fnr_at_fpr_1pct": 0.374, "mistaken_closures": 36}

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


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    raw_cases, raw_golds = read_jsonl(DATA / "cases.jsonl"), read_jsonl(DATA / "gold.jsonl")
    if ARM == "flow-behaviour":  # v1's masking: the host is the label otherwise
        raw_cases = [v1.mask_ips(c) for c in raw_cases]
    cases, golds, dd = dedupe_states(raw_cases, raw_golds)
    gold = {g["case_id"]: g["gold"]["verdict"] for g in golds}
    gold_by = {g["case_id"]: g for g in golds}
    by_id = {c["case_id"]: c for c in cases}
    samples, _ = samples_from_gold(cases, golds, {c["case_id"]: "all" for c in cases}, min_tier="C")
    sample_of = {s.meta["case_id"]: s for s in samples}
    uid_of = {cid: s.uid for cid, s in sample_of.items()}
    cache = features(samples)
    fams = v1.family_order([c["family_id"] for c in cases])
    folds: list[dict[str, Any]] = []

    for k, held in enumerate(fams):
        calib = {fams[(k + 1) % len(fams)], fams[(k + 2) % len(fams)]}
        pins = {f: "test" if f == held else "calibration" if f in calib else "train" for f in fams}
        lock = make_lock(cases, golds, seed=f"ndr-{ARM}-v1-{held}", pinned=pins)
        ids = {sp: list(v) for sp, v in lock["splits"].items()}
        splits = {sp: [sample_of[c] for c in v] for sp, v in ids.items()}
        lr = LogisticScorer().fit([by_id[c] for c in ids["train"]], gold_by)

        def lr_probs(cid: str, _lr: LogisticScorer = lr) -> np.ndarray:
            d = _lr.predict(by_id[cid])
            return np.array([d[x] for x in LABELS])

        plain = v1.CachedJef(train(f"{held}-plain", splits, []), cache, uid_of, TASK)
        choice = []
        for lam in LAMBDAS:
            model = train(f"{held}-cost{lam:g}", splits, ["--fn-cost", f"{lam:g}", *COST_FLAGS])
            jef = v1.CachedJef(model, cache, uid_of, TASK)
            cal = [(jef.probs(c), c) for c in ids["calibration"]]
            mc = sum(mistaken(p, gold[c]) for p, c in cal)
            known = [(p, c) for p, c in cal if gold[c] != "unknown"]
            fnr = fnr_at_fpr(
                np.array([float(p[ATTACK_IDX].sum()) for p, _ in known]),
                np.array([gold[c] in ATTACK for _, c in known]),
                0.01,
            ).get("fnr")
            choice.append((mc, 1.0 if fnr is None else float(fnr), lam, jef))
        choice.sort(key=lambda t: (t[0], t[1], t[2]))
        cost = choice[0][3]
        plain.name, cost.name = "jef", "jef_cost"
        report = evaluate(cases, golds, lock, [RuleScorer(), lr, plain, cost], split="test")

        row: dict[str, Any] = {
            "held_out": held,
            "n_test": len(ids["test"]),
            "test_gold": dict(Counter(gold[c] for c in ids["test"])),
            "lambda": choice[0][2],
        }
        test_y = [LABELS.index(gold[c]) for c in ids["test"]]
        t_att = [gold[c] in ATTACK for c in ids["test"]]
        t_clo = [gold[c] in CLOSING for c in ids["test"]]
        c_att = [gold[c] in ATTACK for c in ids["calibration"]]
        for name, fn in {
            "rules": None,
            "logreg": lr_probs,
            "jef": plain.probs,
            "jef_cost": cost.probs,
        }.items():
            block = report["scorers"][name]["tasks"][TASK]
            entry: dict[str, Any] = {
                "macro_f1": block.get("macro_f1"),
                "pr_auc": block.get("binary", {}).get("pr_auc"),
                "fnr_at_fpr_1pct": (block.get("binary", {}).get("fnr_at_fpr_1pct") or {}).get(
                    "fnr"
                ),
                "ece": block.get("ece"),
                "mistaken_closures": block.get("unknown_handling", {}).get("mistaken_closures"),
            }
            if fn is not None:
                tp = [fn(c) for c in ids["test"]]
                cp = [fn(c) for c in ids["calibration"]]
                entry["mistaken_closure_ids"] = [
                    c for p, c in zip(tp, ids["test"], strict=True) if mistaken(p, gold[c])
                ]
                entry["mistaken_closures"] = len(entry["mistaken_closure_ids"])
                gate = fit_close_gate([float(p[CLOSING_IDX].sum()) for p in cp], c_att)
                entry["gate"] = {
                    "threshold": gate.threshold,
                    **gate.evaluate([float(p[CLOSING_IDX].sum()) for p in tp], t_att, t_clo),
                }
                if name != "logreg":
                    calib_obj = (plain if name == "jef" else cost).calibrator
                    check = calib_obj.coverage_check(tp, test_y, "choice", len(LABELS), ALPHA, TASK)
                    entry["coverage"] = {
                        k2: check.get(k2) for k2 in ("status", "empirical", "wilson_95")
                    }
                    entry["calibration_status"] = calib_obj.status(
                        "choice", len(LABELS), ALPHA, TASK
                    )
            row[name] = entry
        folds.append(row)
        print(
            f"[{ARM}] {held:18s} n={row['n_test']:4d} lam={row['lambda']:g}  "
            + "  ".join(
                f"{n} F1 {row[n]['macro_f1']} FNR {row[n]['fnr_at_fpr_1pct']} MC {row[n]['mistaken_closures']}"
                for n in ("logreg", "jef", "jef_cost")
            ),
            file=sys.stderr,
        )

    names = ("rules", "logreg", "jef", "jef_cost")
    attack_total = sum(sum(v for k2, v in f["test_gold"].items() if k2 in ATTACK) for f in folds)

    def med(n: str, key: str) -> float | None:
        vals = [f[n][key] for f in folds if f[n].get(key) is not None]
        return round(statistics.median(vals), 4) if vals else None

    summary: dict[str, Any] = {}
    for n in names:
        mc = sum(f[n]["mistaken_closures"] or 0 for f in folds)
        s: dict[str, Any] = {
            "median_macro_f1": med(n, "macro_f1"),
            "median_pr_auc": med(n, "pr_auc"),
            "median_fnr_at_fpr_1pct": med(n, "fnr_at_fpr_1pct"),
            "median_ece": med(n, "ece"),
            "mistaken_closures": mc,
            "mistaken_closures_per_1000_attacks": round(1000 * mc / attack_total, 2)
            if attack_total
            else None,
            "worst_fold_macro_f1": min(
                (f[n]["macro_f1"] for f in folds if f[n]["macro_f1"] is not None), default=None
            ),
        }
        if "gate" in folds[0][n]:
            s["gate"] = {
                "closed": sum(f[n]["gate"]["closed"] for f in folds),
                "closed_attacks": sum(f[n]["gate"]["closed_attacks"] for f in folds),
                "closed_benign": sum(f[n]["gate"]["closed_benign"] for f in folds),
                "folds_with_an_attack_closed": sum(
                    f[n]["gate"]["closed_attacks"] > 0 for f in folds
                ),
            }
        if "coverage" in folds[0][n]:
            s["coverage_check"] = dict(Counter(f[n]["coverage"]["status"] for f in folds))
            s["median_coverage"] = statistics.median(
                f[n]["coverage"]["empirical"]
                for f in folds
                if f[n]["coverage"]["empirical"] is not None
            )
        summary[n] = s

    v1_ctu = json.loads((HERE.parent / "ndr-head-and-transfer-v1" / "results.json").read_text())
    v1_attack_n = sum(
        v
        for k2, v in v1_ctu["inputs"]["ctu-13"]["gold_distribution"].items()
        if k2 in {"c2", "botnet", "dns_tunnel"}
    )
    v1_rate = round(1000 * V1_REFERENCE["mistaken_closures"] / v1_attack_n, 2)
    p, lg = summary["jef_cost"], summary["logreg"]
    verdict = {
        "better_than_single_flow_v1": {
            "fnr_below_0.374": (
                p["median_fnr_at_fpr_1pct"] if p["median_fnr_at_fpr_1pct"] is not None else 1
            )
            < V1_REFERENCE["median_fnr_at_fpr_1pct"],
            "mistaken_closure_rate_below_v1": (p["mistaken_closures_per_1000_attacks"] or 0)
            < v1_rate,
            "macro_f1_at_least_0.768": (p["median_macro_f1"] or 0)
            >= V1_REFERENCE["median_macro_f1"],
            "v1_mistaken_closures_per_1000_attacks": v1_rate,
        },
        "close_to_logreg": {
            "macro_f1_within_0.03": (p["median_macro_f1"] or 0)
            >= (lg["median_macro_f1"] or 0) - 0.03,
            "fnr_within_0.05": (
                p["median_fnr_at_fpr_1pct"] if p["median_fnr_at_fpr_1pct"] is not None else 1
            )
            <= (lg["median_fnr_at_fpr_1pct"] or 0) + 0.05,
            "no_more_mistaken_closures": p["mistaken_closures"] <= lg["mistaken_closures"],
        },
        "gate_safe": {
            n: summary[n]["gate"]["closed_attacks"] == 0 for n in ("logreg", "jef", "jef_cost")
        },
    }
    for key in ("better_than_single_flow_v1", "close_to_logreg"):
        verdict[key]["met"] = all(v for k2, v in verdict[key].items() if isinstance(v, bool))
    results = {
        "evidence_class": "SOURCE-LABEL RESEARCH",
        "arm": ARM,
        "task": TASK,
        "backbone": {"id": BACKBONE, "revision": resolve_backbone_spec(BACKBONE)[1]},
        "inputs": {
            "cases_before_dedupe": len(raw_cases),
            "cases": len(cases),
            "dedupe_dropped_same_label": len(dd["dropped_same_label"]),
            "dedupe_dropped_conflicting": len(dd["dropped_conflicting_labels"]),
            "cases_digest": digest_records(cases),
            "gold_digest": digest_records(golds),
            "gold_distribution": dict(Counter(gold.values())),
            "attack_windows": attack_total,
            "build_summary": json.loads((DATA / "summary.json").read_text()),
        },
        "summary": summary,
        "verdict": verdict,
        "folds": folds,
    }
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=_json) + "\n")
    print(
        json.dumps({"summary": summary, "verdict": verdict}, indent=1, default=_json),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
