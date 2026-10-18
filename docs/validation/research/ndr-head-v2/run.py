#!/usr/bin/env python3
"""SOURCE-LABEL RESEARCH: root cause of v1's mistaken closures, and a v2 NDR head.

Fixed before any v2 score was seen (2026-10-18). v1 is
``../ndr-head-and-transfer-v1``; its folds, data, features and fit settings
are reused unchanged, so v1 and v2 differ only in what is listed here.

**Stage 1: root cause analysis (v1 heads, no retraining).** For every v1
mistaken closure -- a case labelled botnet, C2 or DNS tunnel that the v1 head
called benign or normal automation with probability >= 0.9 -- record the fold,
case id, gold label, both scorers' answers, the flow summary, how the case's
(protocol, destination port) slice was labelled in that fold's training
scenarios, and the labels of its ten nearest training cases in the frozen
feature space. The stored v1 heads must reproduce v1's count of 36 exactly.

**Stage 2: three heads per fold, same folds as v1.**

``v1``         the stored v1 head (cross-entropy, inverse-frequency weights).
``cost``       adds the expected false-negative cost to the loss
               (``jef_train.fit --fn-cost``): attacks are c2, botnet,
               dns_tunnel; closing labels are benign, normal_automation. The
               weight is chosen per fold from {1, 2, 4} on that fold's
               *calibration* scenarios only: fewest mistaken closures there,
               then lowest miss rate at 1% false positives, then the smaller
               weight. The test scenario plays no part.
``cost_hard``  the chosen weight plus one hard-positive retraining pass:
               training attacks the first head put >= 0.5 on closing labels
               are weighted 3x (``--hard-positive-weight 3``).

``logreg`` (refitted on the fold's training scenarios, as in v1) is scored
beside them.

**Stage 3: abstention and a measured risk gate.** For each scorer, a close
gate (``jef_train.socbench.gate``) is fitted on the fold's calibration
scenarios and frozen: close only when P(closing labels) clears the lowest
threshold whose calibration closures contain attacks at a one-sided 95%
upper bound of at most 1%. JEF heads are gated twice, on the score alone and
also requiring an all-closing conformal set (alpha 0.10). Reported on test:
cases closed, attacks closed, the share of benign cases spared review.

Coverage is reported three ways: marginal split-conformal coverage, the
attack classes' coverage under that marginal quantile, and attack-class
coverage under class-conditional (Mondrian) quantiles fitted on the same
calibration cases. ``Calibrator.coverage_check`` gives the verdict a
deployment would see on each test scenario.

**Success criteria, fixed in advance.** A v2 head counts as an improvement
over v1 only if, over the 13 folds, (1) total mistaken closures fall, (2) the
median miss rate at 1% false positives falls, and (3) median macro-F1 drops by
no more than 0.02. The gate counts as safe only if it closes no attack on any
test fold. "Close to logreg" means within 0.03 median macro-F1 and no more
mistaken closures than logreg.

Run from the repository root after the v1 study (it reuses v1's features):

    uv run python docs/validation/research/ndr-head-v2/run.py
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from jef_core.calibration import Calibrator
from jef_train.cache import FeatureCache
from jef_train.fit import main as fit_main
from jef_train.schema import write_samples
from jef_train.socbench.annotate import NEGATIVE_LABELS, TASK_LABELS
from jef_train.socbench.baselines import LogisticScorer, RuleScorer
from jef_train.socbench.corpus_export import samples_from_gold
from jef_train.socbench.gate import fit_close_gate
from jef_train.socbench.metrics import fnr_at_fpr, risk_coverage
from jef_train.socbench.report import POSITIVE_LABELS, evaluate
from jef_train.socbench.splits import make_lock


def _json(o: Any) -> Any:
    """numpy scalars as the JSON types they are (str() would turn True into "True")."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)


HERE = Path(__file__).resolve().parent
V1_PATH = HERE.parent / "ndr-head-and-transfer-v1" / "run.py"
_spec = importlib.util.spec_from_file_location("ndr_v1", V1_PATH)
if _spec is None or _spec.loader is None:
    raise SystemExit(f"cannot load {V1_PATH}")
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)

NDR = v1.NDR
LABELS = list(TASK_LABELS[NDR])
ATTACK = sorted(POSITIVE_LABELS[NDR])
CLOSING = sorted(set(LABELS) & set(NEGATIVE_LABELS))
ATTACK_IDX = [LABELS.index(x) for x in ATTACK]
CLOSING_IDX = [LABELS.index(x) for x in CLOSING]
AUTO = 0.9
ALPHA = 0.10
LAMBDAS = (1.0, 2.0, 4.0)
HARD_WEIGHT = 3.0
V1_WORK = v1.WORK
#: SMOKE=1 (inherited from v1) runs on v1's smoke features and a few cases.
WORK = v1.ROOT / "research" / ("ndr-v2-smoke" if v1.SMOKE else "ndr-v2")
COST_FLAGS = ["--attack-labels", ",".join(ATTACK), "--closing-labels", ",".join(CLOSING)]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def mistaken(probs: np.ndarray, gold: str) -> bool:
    k = int(np.argmax(probs))
    return LABELS[k] in NEGATIVE_LABELS and probs[k] >= AUTO and gold in POSITIVE_LABELS[NDR]


def binary_fnr(scores: Sequence[float], positive: Sequence[bool]) -> float | None:
    v = fnr_at_fpr(np.asarray(scores), np.asarray(positive), 0.01).get("fnr")
    return None if v is None else float(v)


def train(name: str, splits: Mapping[str, list], extra: list[str]) -> Path:
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
        str(v1.FEATURES),
        "--backbone",
        v1.BACKBONE,
        *extra,
    ]
    if fit_main(argv) not in (0, None):
        raise SystemExit(f"fit failed for {name}")
    return out


class Probs:
    """Calibrated class probabilities for a scorer over named cases."""

    def __init__(self, fn) -> None:  # type: ignore[no-untyped-def]
        self.fn = fn

    def __call__(self, case_id: str) -> np.ndarray:
        return np.asarray(self.fn(case_id), dtype=np.float64)


def mondrian_quantiles(
    cal_probs: list[np.ndarray], cal_y: list[int], alpha: float
) -> dict[int, float]:
    """Per-class split-conformal quantile of 1 - p(true class)."""
    by: dict[int, list[float]] = defaultdict(list)
    for p, y in zip(cal_probs, cal_y, strict=True):
        by[y].append(1.0 - float(p[y]))
    out = {}
    for y, scores in by.items():
        n = len(scores)
        k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
        out[y] = sorted(scores)[k - 1] if n else 1.0
    return out


def coverage_block(
    jef_cal: Calibrator,
    cal_probs: list[np.ndarray],
    cal_y: list[int],
    test_probs: list[np.ndarray],
    test_y: list[int],
) -> dict[str, Any]:
    n = len(LABELS)
    sets = [jef_cal.prediction_set(p, "choice", n, ALPHA, NDR) for p in test_probs]
    attack_rows = [i for i, y in enumerate(test_y) if y in ATTACK_IDX]
    q = mondrian_quantiles(cal_probs, cal_y, ALPHA)
    msets = [
        [j for j in range(n) if p[j] >= 1.0 - q.get(j, 1.0)] or list(range(n)) for p in test_probs
    ]
    check = jef_cal.coverage_check(test_probs, test_y, "choice", n, ALPHA, NDR)

    def cov(ss: list[list[int]], rows: list[int]) -> float | None:
        return round(float(np.mean([test_y[i] in ss[i] for i in rows])), 4) if rows else None

    return {
        "marginal": cov(sets, list(range(len(test_y)))),
        "attack_marginal": cov(sets, attack_rows),
        "attack_mondrian": cov(msets, attack_rows),
        "mondrian_mean_set_size": round(float(np.mean([len(s) for s in msets])), 3),
        "coverage_check": {k: check[k] for k in ("status", "empirical", "wilson_95") if k in check},
        "closing_only_sets": [all(j in CLOSING_IDX for j in s) for s in sets],
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cases, golds, _ = v1.ctu13()
    if v1.SMOKE:  # the same thinning v1's smoke run used: 40 cases per family
        seen: Counter[str] = Counter()
        cases = [c for c in cases if (seen.update([c["family_id"]]) or seen[c["family_id"]] <= 40)]
        keep = {c["case_id"] for c in cases}
        golds = [g for g in golds if g["case_id"] in keep]
    gold = {g["case_id"]: g["gold"]["verdict"] for g in golds}
    gold_by = {g["case_id"]: g for g in golds}
    by_id = {c["case_id"]: c for c in cases}
    samples, _ = samples_from_gold(cases, golds, {c["case_id"]: "all" for c in cases}, min_tier="C")
    sample_of = {s.meta["case_id"]: s for s in samples}
    uid_of = {cid: s.uid for cid, s in sample_of.items()}
    cache = FeatureCache.load(next(v1.FEATURES.glob("all.*.npz")))
    index = {u: i for i, u in enumerate(cache.uids)}

    def embed(cid: str) -> np.ndarray:
        ctx, _, _ = cache.sample(index[uid_of[cid]])
        e = ctx.mean(axis=0)
        return e / max(float(np.linalg.norm(e)), 1e-9)

    fams = v1.family_order([c["family_id"] for c in cases])
    v1_results = json.loads((V1_PATH.parent / "results.json").read_text())
    rca: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []

    for k, held in enumerate(fams):
        calib = {fams[(k + 1) % len(fams)], fams[(k + 2) % len(fams)]}
        pins = {f: "test" if f == held else "calibration" if f in calib else "train" for f in fams}
        lock = make_lock(cases, golds, seed=f"{v1.SEED}-{held}", pinned=pins)
        splits = {sp: [sample_of[cid] for cid in ids] for sp, ids in lock["splits"].items()}
        ids = {sp: list(v) for sp, v in lock["splits"].items()}
        train_cases = [by_id[c] for c in ids["train"]]
        lr = LogisticScorer().fit(train_cases, gold_by)

        def lr_probs(cid: str, _lr: LogisticScorer = lr) -> np.ndarray:
            d = _lr.predict(by_id[cid])
            return np.array([d[x] for x in LABELS])

        heads: dict[str, Path] = {"v1": V1_WORK / "folds" / f"ndr-{held}" / "model"}

        # -- stage 1: root cause of v1's mistaken closures in this fold
        v1_jef = v1.CachedJef(heads["v1"], cache, uid_of, NDR)
        train_emb = np.stack([embed(c) for c in ids["train"]])
        train_gold = [gold[c] for c in ids["train"]]
        slice_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        for c in ids["train"]:
            st = by_id[c]["state"]
            slice_counts[(str(st.get("proto")), str(st.get("dport")))][gold[c]] += 1
        for cid in ids["test"]:
            p = v1_jef.probs(cid)
            if not mistaken(p, gold[cid]):
                continue
            st = by_id[cid]["state"]
            sims = train_emb @ embed(cid)
            nn = np.argsort(-sims)[:10]
            lp = lr_probs(cid)
            rca.append(
                {
                    "fold": held,
                    "case_id": cid,
                    "gold": gold[cid],
                    "jef_v1": {"pred": LABELS[int(p.argmax())], "p": round(float(p.max()), 4)},
                    "logreg": {"pred": LABELS[int(lp.argmax())], "p": round(float(lp.max()), 4)},
                    "flow": {
                        x: st.get(x)
                        for x in (
                            "proto",
                            "dport",
                            "flows",
                            "total_bytes",
                            "total_packets",
                            "duration_s",
                            "states",
                            "interval_cv",
                        )
                    },
                    "training_slice_labels": dict(
                        slice_counts[(str(st.get("proto")), str(st.get("dport")))]
                    ),
                    "nearest_training_labels": dict(Counter(train_gold[i] for i in nn)),
                    "nearest_similarity": round(float(sims[nn[0]]), 4),
                }
            )

        # -- stage 2: choose the cost weight on calibration, then train cost_hard
        choice = []
        for lam in LAMBDAS:
            model = train(f"{held}-cost{lam:g}", splits, ["--fn-cost", f"{lam:g}", *COST_FLAGS])
            jef = v1.CachedJef(model, cache, uid_of, NDR)
            cal_p = [jef.probs(c) for c in ids["calibration"]]
            mc = sum(mistaken(p, gold[c]) for p, c in zip(cal_p, ids["calibration"], strict=True))
            known = [
                (p, c)
                for p, c in zip(cal_p, ids["calibration"], strict=True)
                if gold[c] != "unknown"
            ]
            fnr = binary_fnr(
                [float(p[ATTACK_IDX].sum()) for p, _ in known],
                [gold[c] in ATTACK for _, c in known],
            )
            choice.append((mc, fnr if fnr is not None else 1.0, lam, model))
        choice.sort(key=lambda t: (t[0], t[1], t[2]))
        lam = choice[0][2]
        heads["cost"] = choice[0][3]
        heads["cost_hard"] = train(
            f"{held}-cost{lam:g}-hard",
            splits,
            ["--fn-cost", f"{lam:g}", "--hard-positive-weight", f"{HARD_WEIGHT:g}", *COST_FLAGS],
        )

        # -- stage 3: score everything the same way
        scorers: dict[str, Any] = {
            name: v1.CachedJef(path, cache, uid_of, NDR) for name, path in heads.items()
        }
        renamed = []
        for name, s in scorers.items():
            s.name = name
            renamed.append(s)
        report = evaluate(cases, golds, lock, [RuleScorer(), lr, *renamed], split="test")
        row: dict[str, Any] = {
            "held_out": held,
            "n_test": len(ids["test"]),
            "lambda_choice": [
                {"lambda": c[2], "cal_mistaken_closures": c[0], "cal_fnr_at_fpr_1pct": c[1]}
                for c in choice
            ],
            "lambda": lam,
        }
        test_y = [LABELS.index(gold[c]) for c in ids["test"]]
        cal_y = [LABELS.index(gold[c]) for c in ids["calibration"]]
        test_attack = [gold[c] in ATTACK for c in ids["test"]]
        test_closing = [gold[c] in CLOSING for c in ids["test"]]
        cal_attack = [gold[c] in ATTACK for c in ids["calibration"]]
        prob_fns: dict[str, Any] = {"logreg": lr_probs, **{n: s.probs for n, s in scorers.items()}}
        for name, fn in prob_fns.items():
            block = report["scorers"][name]["tasks"][NDR]
            test_p = [fn(c) for c in ids["test"]]
            cal_p = [fn(c) for c in ids["calibration"]]
            entry: dict[str, Any] = {
                "macro_f1": block.get("macro_f1"),
                "pr_auc": block.get("binary", {}).get("pr_auc"),
                "fnr_at_fpr_1pct": (block.get("binary", {}).get("fnr_at_fpr_1pct") or {}).get(
                    "fnr"
                ),
                "ece": block.get("ece"),
                "mistaken_closures": sum(
                    mistaken(p, gold[c]) for p, c in zip(test_p, ids["test"], strict=True)
                ),
                "mistaken_closure_ids": [
                    c for p, c in zip(test_p, ids["test"], strict=True) if mistaken(p, gold[c])
                ],
                "aurc": risk_coverage(
                    [float(p.max()) for p in test_p],
                    [int(p.argmax()) == y for p, y in zip(test_p, test_y, strict=True)],
                )["aurc"],
            }
            score = [float(p[CLOSING_IDX].sum()) for p in test_p]
            cal_score = [float(p[CLOSING_IDX].sum()) for p in cal_p]
            gate = fit_close_gate(cal_score, cal_attack)
            entry["gate_score_only"] = {
                "threshold": gate.threshold,
                **gate.evaluate(score, test_attack, test_closing),
            }
            if name != "logreg":
                cov = coverage_block(scorers[name].calibrator, cal_p, cal_y, test_p, test_y)
                cal_cov = coverage_block(scorers[name].calibrator, cal_p, cal_y, cal_p, cal_y)
                gate_set = fit_close_gate(
                    cal_score, cal_attack, set_closing_only=cal_cov["closing_only_sets"]
                )
                entry["gate_with_set"] = {
                    "threshold": gate_set.threshold,
                    **gate_set.evaluate(
                        score, test_attack, test_closing, cov.pop("closing_only_sets")
                    ),
                }
                entry["coverage"] = cov
            row[name] = entry
        folds.append(row)
        print(
            f"[v2] {held:18s} lam={lam:g}  "
            + "  ".join(
                f"{n} F1 {row[n]['macro_f1']} MC {row[n]['mistaken_closures']}"
                for n in ("logreg", "v1", "cost", "cost_hard")
            ),
            file=sys.stderr,
        )

    # -- reproduction check and summary
    v1_total = v1_results["A_B_ndr_head"]["mistaken_closures_total"]["jef"]
    reproduced = sum(f["v1"]["mistaken_closures"] for f in folds)
    names = ("logreg", "v1", "cost", "cost_hard")

    def med(name: str, key: str) -> float | None:
        vals = [f[name][key] for f in folds if f[name][key] is not None]
        return round(statistics.median(vals), 4) if vals else None

    def gate_sum(name: str, key: str) -> dict[str, Any] | None:
        if key not in folds[0][name]:
            return None
        rows = [f[name][key] for f in folds]
        benign = sum(r["closed_benign"] for r in rows)
        return {
            "closed": sum(r["closed"] for r in rows),
            "closed_attacks": sum(r["closed_attacks"] for r in rows),
            "folds_with_an_attack_closed": sum(r["closed_attacks"] > 0 for r in rows),
            "closed_benign": benign,
            "folds_closing_anything": sum(r["closed"] > 0 for r in rows),
        }

    summary = {
        name: {
            "median_macro_f1": med(name, "macro_f1"),
            "median_pr_auc": med(name, "pr_auc"),
            "median_fnr_at_fpr_1pct": med(name, "fnr_at_fpr_1pct"),
            "median_ece": med(name, "ece"),
            "median_aurc": med(name, "aurc"),
            "mistaken_closures_total": sum(f[name]["mistaken_closures"] for f in folds),
            "gate_score_only": gate_sum(name, "gate_score_only"),
            "gate_with_set": gate_sum(name, "gate_with_set"),
            **(
                {
                    "coverage": {
                        "median_marginal": statistics.median(
                            f[name]["coverage"]["marginal"] for f in folds
                        ),
                        "median_attack_marginal": statistics.median(
                            f[name]["coverage"]["attack_marginal"]
                            for f in folds
                            if f[name]["coverage"]["attack_marginal"] is not None
                        ),
                        "median_attack_mondrian": statistics.median(
                            f[name]["coverage"]["attack_mondrian"]
                            for f in folds
                            if f[name]["coverage"]["attack_mondrian"] is not None
                        ),
                        "coverage_check": dict(
                            Counter(f[name]["coverage"]["coverage_check"]["status"] for f in folds)
                        ),
                    }
                }
                if name != "logreg"
                else {}
            ),
        }
        for name in names
    }

    def verdict(name: str) -> dict[str, Any]:
        s, b = summary[name], summary["v1"]
        crit = {
            "fewer_mistaken_closures": s["mistaken_closures_total"] < b["mistaken_closures_total"],
            "lower_median_fnr": (s["median_fnr_at_fpr_1pct"] or 1)
            < (b["median_fnr_at_fpr_1pct"] or 1),
            "macro_f1_within_0.02": (s["median_macro_f1"] or 0)
            >= (b["median_macro_f1"] or 0) - 0.02,
        }
        lg = summary["logreg"]
        return {
            **crit,
            "improvement": all(crit.values()),
            "close_to_logreg": (s["median_macro_f1"] or 0) >= (lg["median_macro_f1"] or 0) - 0.03
            and s["mistaken_closures_total"] <= lg["mistaken_closures_total"],
        }

    results = {
        "evidence_class": "SOURCE-LABEL RESEARCH",
        "design": "see run.py docstring; folds, data and features identical to ndr-head-and-transfer-v1",
        "v1_reproduction": {
            "v1_reported": v1_total,
            "recomputed": reproduced,
            "ok": v1_total == reproduced,
        },
        "rca_cases": len(rca),
        "summary": summary,
        "verdict": {n: verdict(n) for n in ("cost", "cost_hard")},
        "gate_safe": {
            n: {
                g: (summary[n][g] or {}).get("closed_attacks") == 0
                for g in ("gate_score_only", "gate_with_set")
                if summary[n][g]
            }
            for n in names
        },
        "folds": folds,
    }
    if not v1.SMOKE:
        (HERE / "results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False, default=_json) + "\n"
        )
        (HERE / "rca.json").write_text(
            json.dumps(rca, indent=2, ensure_ascii=False, default=_json) + "\n"
        )
    print(
        json.dumps(
            {k: results[k] for k in ("v1_reproduction", "verdict", "gate_safe")},
            indent=1,
            default=_json,
        ),
        file=sys.stderr,
    )
    print(json.dumps(summary, indent=1, default=_json), file=sys.stderr)
    return 0 if results["v1_reproduction"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
