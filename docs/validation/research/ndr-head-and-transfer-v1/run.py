#!/usr/bin/env python3
"""SOURCE-LABEL RESEARCH: a JEF NDR head, its task-scoped calibration, and transfer.

Three questions, one frozen backbone (jhu-clsp/mmBERT-base at the pinned
revision), every number from source labels (CTU-13 Tier C, AIT-ADS Tier D):

A. **NDR head.** Does a JEF head trained on CTU-13 flows beat the baselines on
   scenarios it never saw? Leave-one-scenario-out over the 13 scenarios of the
   IP-masked, deduplicated CTU-13 set (the G1 variant that measures behaviour,
   not addresses). Per fold: the held-out scenario is ``test``; the next two
   scenarios in numeric order (cyclic) are ``calibration``; the other ten are
   ``train``. ``rules``, ``logreg`` (refitted on the fold's train scenarios) and
   ``jef`` (head trained on the same scenarios) are scored by the same SOCBench
   ``evaluate`` call.
B. **Schema-2 calibration.** Each fold's head is calibrated by ``jef_train.fit``
   on its two calibration scenarios only (task-scoped buckets, schema 2). We
   report the bucket's status, its sample count, test ECE before and after,
   and empirical coverage of the alpha=0.10 conformal sets on the held-out
   scenario. Source labels: this can say whether the machinery calibrates on
   unseen families; it cannot make a model ``automation_ready``.
C. **Transfer.** One shared binary question for both sources, *"Does this
   evidence show attack activity?"*: CTU-13 botnet/C2 and AIT-ADS
   attack-window cases are ``attack``; everything else is ``no_attack``
   (benign, Background and unlabelled windows: unlabelled is not benign, so
   ``no_attack`` here means "not labelled as attack"). Train on one source,
   test on all of the other, both directions; in-domain leave-one-family-out
   on each source is the reference. Scorers: ``logreg`` and a ``jef`` head on
   the shared question.

Fixed before any score was seen: the fold rule, the calibration rule, the
binary mapping, and the AIT-ADS sample. AIT-ADS has 19,726 windows (497 in an
attack phase); after state deduplication every attack window is kept and at
most 400 unlabelled windows per scenario are drawn, ordered by
sha256("ndr-head-and-transfer-v1" + case_id). Training settings are
``jef_train.fit`` defaults (rank 64, 40 epochs, lr 3e-3, class weighting,
seed 1337).

Reproduce (repository root, ``uv sync --all-packages``, the ``torch`` extra):

    uv run jef-bench fetch --source ctu-13 --source ait-ads
    uv run jef-bench build --source ctu-13 --background-rate 0.001
    uv run jef-bench build --source ait-ads
    uv run jef-bench dedupe ...   # as in ../ctu13-source-label-v1/run.py
    uv run python docs/validation/research/ndr-head-and-transfer-v1/run.py

The first run encodes every distinct case text once (about 20 minutes on an
Apple M3 with MPS) into ``.cache/socbench/research/ndr-v1/features``; later
runs reuse it. Heads, calibrations and fold corpora stay under ``.cache``;
only ``results.json`` is committed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from jef_core.backends import load_backbone
from jef_core.backends.pins import resolve_backbone_spec
from jef_core.calibration import Calibrator
from jef_core.engine import Engine
from jef_core.head import BilinearHead
from jef_core.mathx import softmax
from jef_train.cache import FeatureCache, build_feature_cache
from jef_train.fit import main as fit_main
from jef_train.schema import Sample, write_samples
from jef_train.socbench.annotate import TASK_LABELS
from jef_train.socbench.baselines import LogisticScorer, RuleScorer, case_text
from jef_train.socbench.corpus_export import samples_from_gold
from jef_train.socbench.metrics import average_precision, cluster_bootstrap, ece, fnr_at_fpr
from jef_train.socbench.records import read_jsonl
from jef_train.socbench.report import evaluate
from jef_train.socbench.splits import dedupe_states, digest_records, make_lock

HERE = Path(__file__).resolve().parent
ROOT = Path(".cache/socbench")
#: SMOKE=1 runs the whole pipeline on the hashing backbone and a few hundred
#: cases, to check the code in minutes. Its numbers mean nothing and it never
#: writes results.json.
SMOKE = os.environ.get("SMOKE") == "1"
WORK = ROOT / "research" / ("ndr-v1-smoke" if SMOKE else "ndr-v1")
FEATURES = WORK / "features"
BACKBONE = "hashing" if SMOKE else "jhu-clsp/mmBERT-base"
SEED = "ndr-head-and-transfer-v1"
NDR = "soc.ndr.technique.v1"
ALERT = "soc.alert.verdict.v1"
BINARY = "research.attack_activity.v1"
BINARY_Q = "Does this evidence show attack activity?"
BINARY_KEYS = ["attack", "no_attack"]
BINARY_LABELS = [
    "attack activity: an intrusion, a botnet or command and control is in progress",
    "no attack activity is evident in this evidence",
]
ATTACK = {
    NDR: {"botnet", "c2", "dns_tunnel", "exfiltration", "scanning"},
    ALERT: {"suspicious", "confirmed_malicious"},
}
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
AIT_UNKNOWN_PER_FAMILY = 400
ALPHA = 0.10

log = logging.getLogger("research")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


def mask_ips(case: dict) -> dict:
    state = {k: v for k, v in case["state"].items() if k not in {"src", "dst"}}
    evidence = [{**e, "value": IPV4.sub("<ip>", str(e["value"]))} for e in case["evidence"]]
    return {**case, "state": state, "evidence": evidence}


def ctu13() -> tuple[list[dict], list[dict], dict]:
    cases = read_jsonl(ROOT / "research" / "ctu13-v1" / "cases.jsonl")
    golds = read_jsonl(ROOT / "research" / "ctu13-v1" / "gold.jsonl")
    masked, mgolds, dd = dedupe_states([mask_ips(c) for c in cases], golds)
    return masked, mgolds, {"kept": dd["kept_cases"], "input": len(cases)}


def ait_ads() -> tuple[list[dict], list[dict], dict]:
    cases = read_jsonl(ROOT / "derived" / "ait-ads" / "cases.jsonl")
    golds = read_jsonl(ROOT / "derived" / "ait-ads" / "gold.jsonl")
    kept, kgolds, dd = dedupe_states(cases, golds)
    verdict = {g["case_id"]: g["gold"]["verdict"] for g in kgolds}
    by_family: dict[str, list[dict]] = defaultdict(list)
    for c in kept:
        by_family[c["family_id"]].append(c)
    chosen: list[dict] = []
    for fam in sorted(by_family):
        attack = [c for c in by_family[fam] if verdict[c["case_id"]] in ATTACK[ALERT]]
        rest = sorted(
            (c for c in by_family[fam] if verdict[c["case_id"]] not in ATTACK[ALERT]),
            key=lambda c: hashlib.sha256((SEED + c["case_id"]).encode()).hexdigest(),
        )
        chosen += attack + rest[:AIT_UNKNOWN_PER_FAMILY]
    ids = {c["case_id"] for c in chosen}
    chosen.sort(key=lambda c: c["case_id"])
    sgolds = [g for g in kgolds if g["case_id"] in ids]
    return (
        chosen,
        sgolds,
        {
            "input": len(cases),
            "after_dedupe": dd["kept_cases"],
            "sampled": len(chosen),
            "rule": f"all attack windows + first {AIT_UNKNOWN_PER_FAMILY} unlabelled per scenario by sha256(seed+case_id)",
        },
    )


def binary_sample(case: Mapping[str, Any], attack: bool) -> Sample:
    return Sample(
        state=case_text(case),
        kind="choice",
        instructions=BINARY_Q,
        option_keys=list(BINARY_KEYS),
        option_labels=list(BINARY_LABELS),
        label=0 if attack else 1,
        source=f"research.{case['source_dataset']}",
        group=case["family_id"],
        lang="en",
        task=BINARY,
        meta={"case_id": case["case_id"]},
    )


def family_order(fams: Sequence[str]) -> list[str]:
    def key(f: str) -> tuple[int, str]:
        tail = f.rsplit("-", 1)[-1]
        return (int(tail), f) if tail.isdigit() else (0, f)

    return sorted(set(fams), key=key)


# --------------------------------------------------------------------------- #
# features: every distinct case text encoded once
# --------------------------------------------------------------------------- #


def universe_cache(samples: list[Sample]) -> FeatureCache:
    model_id, revision = (
        ("hashing", None) if BACKBONE == "hashing" else resolve_backbone_spec(BACKBONE)
    )
    if not revision and BACKBONE != "hashing":
        raise SystemExit("the backbone must be pinned to reuse features")
    # The name jef_train.fit looks for, so every fold reuses these features.
    path = FEATURES / (
        f"all.{model_id.replace('/', '__')}" + (f"@{revision}" if revision else "") + ".npz"
    )
    if path.exists():
        cache = FeatureCache.load(path)
        if set(cache.uids) >= {s.uid for s in samples}:
            log.info("reusing %s (%d samples)", path, len(cache))
            return cache
        log.info("cache at %s lacks samples; re-encoding", path)
    backbone = load_backbone(BACKBONE)
    started = time.perf_counter()
    cache = build_feature_cache(samples, backbone)
    log.info("encoded %d samples in %.0fs", len(samples), time.perf_counter() - started)
    FEATURES.mkdir(parents=True, exist_ok=True)
    cache.save(path)
    return cache


def train_fold(name: str, splits: Mapping[str, list[Sample]]) -> Path:
    """Write a fold corpus and run ``jef_train.fit`` on the shared feature cache."""
    corpus, out = WORK / "folds" / name / "corpus", WORK / "folds" / name / "model"
    for split, items in splits.items():
        write_samples(items, corpus / f"{split}.jsonl")
    write_samples([s for items in splits.values() for s in items], corpus / "all.jsonl")
    code = fit_main(
        [
            "--corpus",
            str(corpus),
            "--out",
            str(out),
            "--cache-dir",
            str(FEATURES),
            "--backbone",
            BACKBONE,
        ]
    )
    if code not in (0, None):
        raise SystemExit(f"fit failed for {name}")
    return out


class CachedJef:
    """``jef`` from cached features: the same logits and temperature the engine uses."""

    name = "jef"

    def __init__(
        self, model: Path, cache: FeatureCache, uid_of: Mapping[str, str], task: str
    ) -> None:
        self.head = BilinearHead.load(model / "head.npz")
        self.calibrator = Calibrator.load(model / "calibration.json")
        self.cache, self.uid_of, self.task = cache, uid_of, task
        self.index = {u: i for i, u in enumerate(cache.uids)}
        self.n = len(TASK_LABELS.get(task, BINARY_KEYS))

    def logits(self, case_id: str) -> np.ndarray:
        ctx, queries, _ = self.cache.sample(self.index[self.uid_of[case_id]])
        a = ctx @ self.head.u
        a = a / np.linalg.norm(a, axis=-1, keepdims=True).clip(min=1e-9)
        b = queries @ self.head.v
        b = b / np.linalg.norm(b, axis=-1, keepdims=True).clip(min=1e-9)
        return np.einsum("nr,nr->n", a, b).astype(np.float64) * self.head.scale + self.head.bias

    def probs(self, case_id: str, *, calibrated: bool = True) -> np.ndarray:
        t = self.calibrator.temperature("choice", self.n, self.task) if calibrated else 1.0
        return softmax(self.logits(case_id), temperature=t)

    def describe(self) -> dict[str, Any]:
        return {"type": "jef", "head": self.head.name, "egress": "none", "features": "cached"}

    def predict(self, case: Mapping[str, Any]) -> dict[str, float]:
        labels = TASK_LABELS[case["task_id"]]
        return dict(zip(labels, map(float, self.probs(case["case_id"])), strict=True))


def engine_parity(
    model: Path, cases: Sequence[Mapping[str, Any]], scorer: CachedJef, question: Mapping[str, Any]
) -> float:
    """Largest gap between cached and real-engine probabilities over ``cases``."""
    engine = Engine(
        BACKBONE, BilinearHead.load(model / "head.npz"), Calibrator.load(model / "calibration.json")
    )
    worst = 0.0
    for c in cases:
        answer = engine.evaluate(case_text(c), {"q": dict(question)}).answers["q"]
        real = np.array([answer.probabilities[k] for k in question["criteria"]])
        worst = max(worst, float(np.max(np.abs(real - scorer.probs(c["case_id"])))))
    return worst


# --------------------------------------------------------------------------- #
# A + B: the NDR head, per held-out scenario
# --------------------------------------------------------------------------- #


def study_ndr(
    cases: list[dict], golds: list[dict], cache: FeatureCache, ndr_samples: list[Sample]
) -> dict:
    gold_by = {g["case_id"]: g for g in golds}
    uid_of = {s.meta["case_id"]: s.uid for s in ndr_samples}
    sample_of = {s.meta["case_id"]: s for s in ndr_samples}
    fams = family_order([c["family_id"] for c in cases])
    labels = TASK_LABELS[NDR]
    folds = []
    from jef_train.socbench.baselines import verdict_question

    question = verdict_question(NDR)
    for k, held in enumerate(fams):
        calib = {fams[(k + 1) % len(fams)], fams[(k + 2) % len(fams)]}
        pins = {f: "test" if f == held else "calibration" if f in calib else "train" for f in fams}
        lock = make_lock(cases, golds, seed=f"{SEED}-{held}", pinned=pins)
        split_of = {cid: sp for sp, ids in lock["splits"].items() for cid in ids}
        splits = {sp: [sample_of[cid] for cid in ids] for sp, ids in lock["splits"].items()}
        model = train_fold(f"ndr-{held}", splits)
        jef = CachedJef(model, cache, uid_of, NDR)
        train = [c for c in cases if split_of[c["case_id"]] == "train"]
        report = evaluate(
            cases,
            golds,
            lock,
            [RuleScorer(), LogisticScorer().fit(train, gold_by), jef],
            split="test",
        )
        test = [c for c in cases if split_of[c["case_id"]] == "test"]
        y = np.array([labels.index(gold_by[c["case_id"]]["gold"]["verdict"]) for c in test])
        raw = np.array([jef.probs(c["case_id"], calibrated=False) for c in test])
        cal = np.array([jef.probs(c["case_id"]) for c in test])
        sets = [jef.calibrator.prediction_set(p, "choice", len(labels), ALPHA, NDR) for p in cal]
        status = jef.calibrator.status("choice", len(labels), ALPHA, NDR)
        row: dict[str, Any] = {
            "held_out": held,
            "calibration_families": sorted(calib),
            "n_test": len(test),
            "calibration": {
                "status": status,
                "schema": jef.calibrator.schema,
                "bucket": f"{NDR}|choice:{len(labels)}",
                "samples": jef.calibrator.artifact.sample_count,
                "temperature": round(jef.calibrator.temperature("choice", len(labels), NDR), 4),
                "test_ece_raw": ece(raw.max(axis=1), raw.argmax(axis=1) == y),
                "test_ece_calibrated": ece(cal.max(axis=1), cal.argmax(axis=1) == y),
                "conformal_coverage": round(
                    float(np.mean([int(t) in s for t, s in zip(y, sets, strict=True)])), 4
                ),
                "conformal_mean_set_size": round(float(np.mean([len(s) for s in sets])), 3),
            },
        }
        if k == 0:
            row["engine_parity_max_abs_diff"] = engine_parity(model, test[:12], jef, question)
        for name, s in report["scorers"].items():
            block = s["tasks"].get(NDR, {})
            binary = block.get("binary", {})
            row[name] = {
                "macro_f1": block.get("macro_f1"),
                "accuracy": block.get("accuracy"),
                "ece": block.get("ece"),
                "pr_auc": binary.get("pr_auc"),
                "fnr_at_fpr_1pct": (binary.get("fnr_at_fpr_1pct") or {}).get("fnr"),
                "mistaken_closures": block.get("unknown_handling", {}).get("mistaken_closures"),
            }
        folds.append(row)
        print(
            f"[ndr] {held:18s} n={len(test):5d}  rules {row['rules']['macro_f1']}  "
            f"logreg {row['logreg']['macro_f1']}  jef {row['jef']['macro_f1']}  "
            f"cal={status} cov={row['calibration']['conformal_coverage']}",
            file=sys.stderr,
        )

    def med(name: str, key: str) -> float | None:
        vals = [f[name][key] for f in folds if f[name][key] is not None]
        return round(statistics.median(vals), 4) if vals else None

    cal_rows = [f["calibration"] for f in folds]
    return {
        "median_over_folds": {
            n: {k: med(n, k) for k in ("macro_f1", "accuracy", "ece", "pr_auc", "fnr_at_fpr_1pct")}
            for n in ("rules", "logreg", "jef")
        },
        "worst_fold_macro_f1": {
            n: min((f[n]["macro_f1"] for f in folds if f[n]["macro_f1"] is not None), default=None)
            for n in ("rules", "logreg", "jef")
        },
        "jef_beats_logreg_folds": sum(
            (f["jef"]["macro_f1"] or 0) > (f["logreg"]["macro_f1"] or 0) for f in folds
        ),
        "mistaken_closures_total": {
            n: sum(f[n]["mistaken_closures"] or 0 for f in folds)
            for n in ("rules", "logreg", "jef")
        },
        "calibration_summary": {
            "statuses": dict(Counter(r["status"] for r in cal_rows)),
            "median_calibration_samples": statistics.median(r["samples"] for r in cal_rows),
            "median_test_ece_raw": round(
                statistics.median(
                    r["test_ece_raw"] for r in cal_rows if r["test_ece_raw"] is not None
                ),
                4,
            ),
            "median_test_ece_calibrated": round(
                statistics.median(
                    r["test_ece_calibrated"]
                    for r in cal_rows
                    if r["test_ece_calibrated"] is not None
                ),
                4,
            ),
            "median_conformal_coverage": statistics.median(
                r["conformal_coverage"] for r in cal_rows
            ),
            "target_coverage": 1 - ALPHA,
            "folds_below_target_coverage": sum(
                r["conformal_coverage"] < 1 - ALPHA for r in cal_rows
            ),
        },
        "folds": folds,
    }


# --------------------------------------------------------------------------- #
# C: transfer on the shared binary question
# --------------------------------------------------------------------------- #


def binary_metrics(scores: np.ndarray, y: np.ndarray, groups: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n": len(y),
        "positives": int(y.sum()),
        "prevalence": round(float(y.mean()), 4),
    }
    if 0 < y.sum() < len(y):
        out["pr_auc"] = average_precision(scores, y)
        from sklearn.metrics import roc_auc_score

        out["roc_auc"] = round(float(roc_auc_score(y, scores)), 4)
        for f in (0.01, 0.05):
            fnr = fnr_at_fpr(scores, y, f).get("fnr")
            out[f"recall_at_fpr_{int(f * 100)}pct"] = (
                None if fnr is None else round(1 - float(fnr), 4)
            )
        out["pr_auc_ci95_by_family"] = cluster_bootstrap(
            list(groups), lambda idx: average_precision(scores[idx], y[idx]), n_boot=300, seed=1337
        )
    return out


def logreg_binary(train_texts: list[str], train_y: list[int]):  # type: ignore[no-untyped-def]
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import LogisticRegression

    if len(set(train_y)) < 2:
        return None  # one class in training: nothing to learn, reported as such
    vec = HashingVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), n_features=2**16, alternate_sign=False
    )
    model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        vec.transform(train_texts), train_y
    )
    return lambda texts: model.predict_proba(vec.transform(texts))[:, list(model.classes_).index(1)]


def study_transfer(
    sources: Mapping[str, tuple[list[dict], dict[str, bool]]],
    cache: FeatureCache,
    bin_samples: dict[str, Sample],
) -> dict:
    uid_of = {cid: s.uid for cid, s in bin_samples.items()}

    def run(
        train_src: str, test_cases: list[dict], test_name: str, exclude: set[str] | None = None
    ) -> dict:
        tr_cases, attack = sources[train_src]
        tr = [c for c in tr_cases if not exclude or c["family_id"] not in exclude]
        fams = family_order([c["family_id"] for c in tr])
        calib = set(fams[-2:])
        splits = {
            "train": [bin_samples[c["case_id"]] for c in tr if c["family_id"] not in calib],
            "calibration": [bin_samples[c["case_id"]] for c in tr if c["family_id"] in calib],
            "test": [bin_samples[c["case_id"]] for c in test_cases],
        }
        model = train_fold(f"bin-{train_src}-to-{test_name}", splits)
        jef = CachedJef(model, cache, uid_of, BINARY)
        y_all = {**sources["ctu-13"][1], **sources["ait-ads"][1]}
        y = np.array([int(y_all[c["case_id"]]) for c in test_cases])
        groups = [c["family_id"] for c in test_cases]
        train_cases = [c for c in tr if c["family_id"] not in calib]
        lr = logreg_binary(
            [case_text(c) for c in train_cases], [int(attack[c["case_id"]]) for c in train_cases]
        )
        return {
            "train": train_src,
            "train_families": len(fams) - len(calib),
            "test": test_name,
            "logreg": binary_metrics(np.asarray(lr([case_text(c) for c in test_cases])), y, groups)
            if lr is not None
            else {"n": len(test_cases), "not_trained": "one class in the training families"},
            "jef": binary_metrics(
                np.array([jef.probs(c["case_id"])[0] for c in test_cases]), y, groups
            ),
            "jef_calibration_status": jef.calibrator.status("choice", 2, ALPHA, BINARY),
        }

    out: dict[str, Any] = {"cross": [], "in_domain": {}}
    for a, b in (("ctu-13", "ait-ads"), ("ait-ads", "ctu-13")):
        out["cross"].append(run(a, sources[b][0], b))
        print(f"[transfer] {a} -> {b} done", file=sys.stderr)
    for src in ("ctu-13", "ait-ads"):
        cases = sources[src][0]
        rows = []
        for held in family_order([c["family_id"] for c in cases]):
            test = [c for c in cases if c["family_id"] == held]
            r = run(src, test, f"{src}-{held}", exclude={held})
            rows.append({"held_out": held, **{k: r[k] for k in ("logreg", "jef")}})
        out["in_domain"][src] = {
            "folds": rows,
            "median_pr_auc": {
                n: (
                    round(statistics.median(vals), 4)
                    if (vals := [r[n]["pr_auc"] for r in rows if r[n].get("pr_auc") is not None])
                    else None
                )
                for n in ("logreg", "jef")
            },
            "folds_with_both_classes": sum(r["jef"].get("pr_auc") is not None for r in rows),
        }
        print(f"[transfer] in-domain {src} done", file=sys.stderr)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("jef.train", "jef_train", "httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ctu, ctu_g, ctu_info = ctu13()
    ait, ait_g, ait_info = ait_ads()
    if SMOKE:  # a few cases per family, both sources

        def thin(cs: list[dict], gs: list[dict], n: int) -> tuple[list[dict], list[dict]]:
            seen: Counter[str] = Counter()
            keep = []
            for c in cs:
                if seen[c["family_id"]] < n:
                    seen[c["family_id"]] += 1
                    keep.append(c)
            ids = {c["case_id"] for c in keep}
            return keep, [g for g in gs if g["case_id"] in ids]

        ctu, ctu_g = thin(ctu, ctu_g, 40)
        ait, ait_g = thin(ait, ait_g, 40)
    ctu_v = {g["case_id"]: g["gold"]["verdict"] for g in ctu_g}
    ait_v = {g["case_id"]: g["gold"]["verdict"] for g in ait_g}
    ndr_samples, _ = samples_from_gold(ctu, ctu_g, {c["case_id"]: "all" for c in ctu}, min_tier="C")
    bin_samples = {c["case_id"]: binary_sample(c, ctu_v[c["case_id"]] in ATTACK[NDR]) for c in ctu}
    bin_samples |= {
        c["case_id"]: binary_sample(c, ait_v[c["case_id"]] in ATTACK[ALERT]) for c in ait
    }
    # Adjacent samples share one state encode (build_feature_cache's memo).
    universe: list[Sample] = []
    for s in ndr_samples:
        universe += [s, bin_samples[s.meta["case_id"]]]
    universe += [bin_samples[c["case_id"]] for c in ait]
    cache = universe_cache(universe)

    results = {
        "evidence_class": "SOURCE-LABEL RESEARCH",
        "backbone": {"id": BACKBONE, "revision": resolve_backbone_spec(BACKBONE)[1]},
        "inputs": {
            "ctu-13": {
                **ctu_info,
                "label_origin": "flow-label (Tier C)",
                "cases_digest": digest_records(ctu),
                "gold_digest": digest_records(ctu_g),
                "gold_distribution": dict(Counter(ctu_v.values())),
            },
            "ait-ads": {
                **ait_info,
                "label_origin": "attack-phase-window (Tier D)",
                "cases_digest": digest_records(ait),
                "gold_digest": digest_records(ait_g),
                "gold_distribution": dict(Counter(ait_v.values())),
            },
        },
        "binary_mapping": {
            "attack": {NDR: sorted(ATTACK[NDR]), ALERT: sorted(ATTACK[ALERT])},
            "no_attack": "every other verdict, including unknown",
        },
        "fit_settings": "jef_train.fit defaults: rank 64, 40 epochs, lr 3e-3, batch 64, class weighting, seed 1337",
    }
    results["A_B_ndr_head"] = study_ndr(ctu, ctu_g, cache, ndr_samples)
    sources = {
        "ctu-13": (ctu, {cid: v in ATTACK[NDR] for cid, v in ctu_v.items()}),
        "ait-ads": (ait, {cid: v in ATTACK[ALERT] for cid, v in ait_v.items()}),
    }
    results["C_transfer"] = study_transfer(sources, cache, bin_samples)
    if not SMOKE:
        (HERE / "results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False, default=str) + "\n"
        )
    a = results["A_B_ndr_head"]
    print(
        json.dumps(
            {"ndr_median": a["median_over_folds"], "calibration": a["calibration_summary"]},
            indent=2,
        ),
        file=sys.stderr,
    )
    for row in results["C_transfer"]["cross"]:
        print(
            f"{row['train']} -> {row['test']}: logreg PR-AUC {row['logreg'].get('pr_auc')}  jef PR-AUC {row['jef'].get('pr_auc')}  prevalence {row['jef']['prevalence']}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
