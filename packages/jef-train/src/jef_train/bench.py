"""Evaluate JEF on benchmarks this project did not construct.

Every number in docs/RESULTS.md is self-evaluated, which is the ecosystem's
credibility problem restated rather than solved. These are third-party sets:

* **TMMLU+** (``ikala/tmmluplus``) -- 66 subjects of Traditional Chinese
  multiple choice. Entirely out of domain, so it measures whether the head
  learned anything transferable or only learned security prose.
* **CTI-Bench MCQ** (``AI4Sec/cti-bench``) -- threat-intelligence multiple
  choice, largely ATT&CK-derived. Near-domain but not the same task: JEF trained
  on technique-to-tactic classification, not on these questions.
* **CTI-Bench VSP** -- CVE descriptions with their full CVSS v3.1 vector, which
  yields *exactly* the four tasks JEF trained on: severity band, attack vector,
  privileges required, user interaction.

That last one carries a real hazard. CTI-Bench VSP is built from NVD and so is
JEF's corpus, so the two can share CVEs -- and evaluating on a CVE the head
trained on is the same contamination that made SOAR routing score 1.000, just
across datasets instead of within one. Overlapping CVE ids are therefore
detected and excluded, and the count of exclusions is reported rather than
quietly handled.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from jef_core import Engine
from jef_core.calibration import Calibrator
from jef_core.mathx import brier_score, confidence, expected_calibration_error

from .evaluate import macro_f1, reliability_curve
from .schema import read_samples

log = logging.getLogger("jef.train.bench")

__all__ = ["BenchCase", "main", "run_benchmark"]

_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)

#: CVSS v3.1 severity bands, matching jef_train.corpus.cve.SEVERITY_ZH.
_SEVERITY_LEVELS = ["無", "低", "中", "高", "危急"]
_AV_KEYS = ["network", "adjacent", "local", "physical"]
_AV_LABELS = [
    "可經由網路遠端觸發，攻擊者不需位於同一網段",
    "需與目標位於同一邏輯或實體鄰接網段",
    "需在本機上以既有存取權執行",
    "需實體接觸目標裝置",
]
_AV_FROM_VECTOR = {"N": 0, "A": 1, "L": 2, "P": 3}


@dataclass
class BenchCase:
    """One benchmark item, already shaped as a JEF question."""

    state: str
    question: dict[str, Any]
    option_keys: list[str]
    label: int
    task: str


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #


def _tmmluplus(limit_per_subject: int, subjects: list[str] | None) -> Iterator[BenchCase]:
    from datasets import get_dataset_config_names, load_dataset

    names = subjects or get_dataset_config_names("ikala/tmmluplus")
    letters = ["A", "B", "C", "D"]
    for subject in names:
        try:
            rows = load_dataset("ikala/tmmluplus", subject, split="test")
        except Exception as exc:
            log.warning("tmmlu+ subject %s unavailable: %s", subject, type(exc).__name__)
            continue
        for row in list(rows)[:limit_per_subject]:
            answer = str(row.get("answer", "")).strip().upper()
            if answer not in letters:
                continue
            options = {letter: str(row[letter]) for letter in letters}
            yield BenchCase(
                state=str(row["question"]),
                question={
                    "type": "choice",
                    "instructions": "根據題目內容，選出正確的答案。",
                    "criteria": options,
                },
                option_keys=letters,
                label=letters.index(answer),
                task=f"tmmluplus/{subject}",
            )


def _cti_mcq(limit: int) -> Iterator[BenchCase]:
    from datasets import load_dataset

    letters = ["A", "B", "C", "D"]
    rows = load_dataset("AI4Sec/cti-bench", "cti-mcq", split="test")
    for row in list(rows)[:limit]:
        answer = str(row.get("GT", "")).strip().upper()
        if answer not in letters:
            continue
        options = {letter: str(row[f"Option {letter}"]) for letter in letters}
        yield BenchCase(
            state=str(row["Question"]),
            question={
                "type": "choice",
                "instructions": "根據威脅情報知識，選出正確的答案。",
                "criteria": options,
            },
            option_keys=letters,
            label=letters.index(answer),
            task="cti-mcq",
        )


def _parse_cvss(vector: str) -> dict[str, str]:
    return dict(
        part.split(":", 1)
        for part in vector.split("/")
        if ":" in part and not part.startswith("CVSS")
    )


def _severity_band(metrics: dict[str, str]) -> int | None:
    """Derive the severity band from the vector's own base-score components.

    CTI-Bench VSP ships the vector, not the score, so the band comes from
    recomputing the CVSS v3.1 base score rather than from a lookup table.
    """
    weights_c = {"H": 0.56, "L": 0.22, "N": 0.0}
    try:
        conf = weights_c[metrics["C"]]
        integ = weights_c[metrics["I"]]
        avail = weights_c[metrics["A"]]
        scope_changed = metrics["S"] == "C"
        impact_base = 1 - (1 - conf) * (1 - integ) * (1 - avail)
        if scope_changed:
            impact = 7.52 * (impact_base - 0.029) - 3.25 * (impact_base - 0.02) ** 15
        else:
            impact = 6.42 * impact_base
        if impact <= 0:
            return 0

        av = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}[metrics["AV"]]
        ac = {"L": 0.77, "H": 0.44}[metrics["AC"]]
        pr_table = (
            {"N": 0.85, "L": 0.68, "H": 0.50}
            if scope_changed
            else {"N": 0.85, "L": 0.62, "H": 0.27}
        )
        pr = pr_table[metrics["PR"]]
        ui = {"N": 0.85, "R": 0.62}[metrics["UI"]]
        exploitability = 8.22 * av * ac * pr * ui

        raw = min((1.08 if scope_changed else 1.0) * (impact + exploitability), 10.0)
        score = float(np.ceil(raw * 10) / 10)
    except KeyError:
        return None

    for index, lower in enumerate([0.0, 0.1, 4.0, 7.0, 9.0]):
        if score >= lower:
            band = index
    return band


def _cti_vsp(limit: int, excluded: set[str]) -> Iterator[BenchCase]:
    from datasets import load_dataset

    rows = load_dataset("AI4Sec/cti-bench", "cti-vsp", split="test")
    skipped = 0
    emitted = 0
    for row in rows:
        if emitted >= limit:
            break
        url = str(row.get("URL", ""))
        found = _CVE_RE.search(url) or _CVE_RE.search(str(row.get("Description", "")))
        cve_id = found.group(0).upper() if found else ""
        if cve_id and cve_id in excluded:
            skipped += 1
            continue

        metrics = _parse_cvss(str(row.get("GT", "")))
        description = str(row.get("Description", "")).strip()
        if len(description) < 80:
            continue
        state = f"漏洞編號：{cve_id}\n\n漏洞描述：\n{description}"
        emitted += 1

        band = _severity_band(metrics)
        if band is not None:
            yield BenchCase(
                state=state,
                question={
                    "type": "score",
                    "instructions": "依據漏洞描述評估其嚴重程度。",
                    "criteria": _SEVERITY_LEVELS,
                },
                option_keys=_SEVERITY_LEVELS,
                label=band,
                task="cti-vsp/severity",
            )
        if metrics.get("AV") in _AV_FROM_VECTOR:
            yield BenchCase(
                state=state,
                question={
                    "type": "choice",
                    "instructions": "攻擊者要利用此漏洞，需要什麼樣的存取位置？",
                    "criteria": dict(zip(_AV_KEYS, _AV_LABELS, strict=True)),
                },
                option_keys=_AV_KEYS,
                label=_AV_FROM_VECTOR[metrics["AV"]],
                task="cti-vsp/attack_vector",
            )
        if metrics.get("PR") in ("N", "L", "H"):
            yield BenchCase(
                state=state,
                question={
                    "type": "noul",
                    "instructions": "利用此漏洞是否需要事先取得任何權限？",
                    "criteria": {"true": "需要既有帳號或權限", "false": "不需任何權限即可利用"},
                },
                option_keys=["false", "true"],
                label=0 if metrics["PR"] == "N" else 1,
                task="cti-vsp/privileges",
            )
        if metrics.get("UI") in ("N", "R"):
            yield BenchCase(
                state=state,
                question={
                    "type": "noul",
                    "instructions": "利用此漏洞是否需要使用者的操作配合？",
                    "criteria": {"true": "需使用者點擊或開啟等操作", "false": "無需使用者互動"},
                },
                option_keys=["false", "true"],
                label=1 if metrics["UI"] == "R" else 0,
                task="cti-vsp/user_interaction",
            )

    log.info("cti-vsp: %d CVE(s) excluded for appearing in the training corpus", skipped)


def training_cve_ids(corpus: Path) -> set[str]:
    """CVE ids the head may have trained on.

    Read from train *and* calibration: a CVE used to fit temperature has still
    informed the model a client would receive.
    """
    ids: set[str] = set()
    for split in ("train", "calibration"):
        path = corpus / f"{split}.jsonl"
        if not path.exists():
            continue
        for sample in read_samples(path):
            cve = sample.meta.get("cve_id")
            if cve:
                ids.add(str(cve).upper())
    return ids


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def run_benchmark(engine: Engine, cases: list[BenchCase], alpha: float = 0.10) -> dict[str, Any]:
    """Evaluate the engine over benchmark cases, grouped by task."""
    by_task: dict[str, list[BenchCase]] = {}
    for case in cases:
        by_task.setdefault(case.task, []).append(case)

    calibrator: Calibrator = engine.calibrator
    results: dict[str, Any] = {}

    for task, items in sorted(by_task.items()):
        probs_list: list[np.ndarray] = []
        labels: list[int] = []
        preds: list[int] = []
        confs: list[float] = []
        p_corrects: list[float] = []
        briers: list[float] = []
        covered = 0
        set_sizes: list[int] = []

        for case in items:
            answer = engine.evaluate(case.state, {"q": case.question}).answers["q"]
            dump = answer.model_dump(exclude_none=True)
            if "probabilities" in dump:
                probs = np.array([dump["probabilities"][k] for k in case.option_keys])
            else:
                value = dump.get("noul", dump.get("probability", 0.5))
                probs = np.array([1.0 - value, value])
            probs = probs / probs.sum()

            kind = "noul" if case.question["type"] in ("noul", "boolean") else case.question["type"]
            width = len(case.option_keys)
            conf = confidence(probs)

            probs_list.append(probs)
            labels.append(case.label)
            preds.append(int(np.argmax(probs)))
            confs.append(conf)
            briers.append(brier_score(probs, case.label))

            mapped = calibrator.p_correct(conf, kind, width)
            if mapped is not None:
                p_corrects.append(mapped)

            if calibrator.is_fitted():
                pset = calibrator.prediction_set(probs, kind, width, alpha)
                set_sizes.append(len(pset))
                if case.label in pset:
                    covered += 1

        n = len(items)
        width = len(items[0].option_keys)
        hits = np.array([p == y for p, y in zip(preds, labels, strict=True)])
        tops = np.array([float(np.max(p)) for p in probs_list])

        entry: dict[str, Any] = {
            "n": n,
            "n_options": width,
            "random_baseline": 1.0 / width,
            "accuracy": float(hits.mean()),
            "macro_f1": macro_f1(preds, labels, width),
            "ece": expected_calibration_error(tops, hits),
            "ece_confidence": expected_calibration_error(np.array(confs), hits),
            "brier": float(np.mean(briers)),
            "mean_confidence": float(np.mean(confs)),
            "reliability": reliability_curve(list(tops), list(hits)),
        }
        if len(p_corrects) == n:
            entry["ece_p_correct"] = expected_calibration_error(np.array(p_corrects), hits)
        else:
            entry["ece_p_correct"] = None
        if calibrator.is_fitted() and set_sizes:
            empirical = covered / n
            entry["conformal_coverage"] = empirical
            entry["mean_set_size"] = float(np.mean(set_sizes))
            slack = 2.0 * float(np.sqrt(alpha * (1 - alpha) / max(n, 1)))
            entry["coverage_violation"] = bool(empirical < (1 - alpha) - slack)

        results[task] = entry
        log.info(
            "%-28s n=%-5d acc %.3f (random %.3f)  macroF1 %.3f  ECE %.3f",
            task,
            n,
            entry["accuracy"],
            entry["random_baseline"],
            entry["macro_f1"],
            entry["ece"],
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate JEF on third-party benchmarks")
    parser.add_argument("--model-dir", default="models/jef-v0")
    parser.add_argument("--backbone", default="jhu-clsp/mmBERT-base")
    parser.add_argument("--corpus", default="data/corpus", help="used to exclude seen CVEs")
    parser.add_argument("--out", default="docs/benchmarks.json")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tmmlu-per-subject", type=int, default=12)
    parser.add_argument("--tmmlu-subjects", nargs="*", default=None)
    parser.add_argument("--cti-mcq-limit", type=int, default=400)
    parser.add_argument("--cti-vsp-limit", type=int, default=250)
    parser.add_argument("--uncalibrated", action="store_true", help="skip the calibrator")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from jef_core import BilinearHead
    from jef_core.backends import load_backbone

    model_dir = Path(args.model_dir)
    head = BilinearHead.load(model_dir / "head.npz")
    calibrator = (
        Calibrator() if args.uncalibrated else Calibrator.load(model_dir / "calibration.json")
    )
    engine = Engine(load_backbone(args.backbone, threads=args.threads), head, calibrator)

    excluded = training_cve_ids(Path(args.corpus))
    log.info("training corpus contributed %d CVE id(s) to the exclusion set", len(excluded))

    cases: list[BenchCase] = []
    cases += list(_tmmluplus(args.tmmlu_per_subject, args.tmmlu_subjects))
    cases += list(_cti_mcq(args.cti_mcq_limit))
    cases += list(_cti_vsp(args.cti_vsp_limit, excluded))
    log.info("evaluating %d benchmark cases", len(cases))

    results = run_benchmark(engine, cases)

    payload = {
        "model_dir": str(model_dir),
        "backbone": args.backbone,
        "calibrated": calibrator.is_fitted(),
        "excluded_training_cves": len(excluded),
        "note": (
            "Third-party benchmarks. CTI-Bench VSP is NVD-derived and so is JEF's "
            "corpus, so CVEs seen in training or calibration are excluded; "
            "evaluating on those would repeat the contamination that made SOAR "
            "routing score 1.000."
        ),
        "tasks": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
