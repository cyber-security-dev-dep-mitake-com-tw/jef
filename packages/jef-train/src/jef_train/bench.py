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
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from jef_core import Engine
from jef_core.calibration import Calibrator
from jef_core.mathx import brier_score, confidence, expected_calibration_error

from .evaluate import macro_f1, reliability_curve
from .schema import read_samples

log = logging.getLogger("jef.train.bench")

__all__ = ["BenchCase", "EngineScorer", "Scorer", "main", "run_benchmark"]

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
    # A limit of zero means "skip this source". Checked before touching the Hub,
    # because resolving 66 subject configs and then slicing them to nothing
    # takes minutes and downloads everything.
    if limit_per_subject <= 0:
        return

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
    if limit <= 0:
        return

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
    if limit <= 0:
        return

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
# Long-context sweep
# --------------------------------------------------------------------------- #

#: Filler that looks like what actually surrounds an alert in a SIEM: routine,
#: plausible, and carrying no bearing on the question. Random text would be an
#: easier test than reality.
_FILLER = [
    "{ts} kernel: [{n}.{m}] TCP: request_sock_TCP: Possible SYN flooding on port 443. Sending cookies.",
    "{ts} sshd[{n}]: Accepted publickey for deploy from 10.0.{m}.{n} port 5{n} ssh2: ED25519 SHA256:redacted",
    '{ts} nginx: 10.0.{m}.{n} - - [req] "GET /api/v2/health HTTP/1.1" 200 17 "-" "kube-probe/1.29"',
    "{ts} systemd[1]: Started Session {n} of user svc_batch.",
    "{ts} kubelet: I{n} pod/web-front-{m} Container image already present on machine",
    "{ts} auditd: type=SYSCALL msg=audit({n}.{m}:{n}): arch=c000003e syscall=59 success=yes exit=0",
    "{ts} postgres[{n}]: LOG:  checkpoint complete: wrote {n} buffers ({m}%); sync files={m}",
    "{ts} haproxy[{n}]: 10.0.{m}.{n}:5{n} [req] api api/srv1 0/0/1/12/13 200 512 - - ---- 8/8/0/0/0",
]


def _filler_lines(count: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    lines = []
    for i in range(count):
        template = _FILLER[i % len(_FILLER)]
        lines.append(
            template.format(
                ts=f"2026-09-{1 + i % 28:02d}T{i % 24:02d}:{i % 60:02d}:{(i * 7) % 60:02d}Z",
                n=int(rng.integers(1000, 99999)),
                m=int(rng.integers(1, 250)),
            )
        )
    return lines


def _pad_to_tokens(evidence: str, target: int, position: str, backbone: Any, seed: int) -> str:
    """Bury the evidence in routine log noise until the state hits `target` tokens.

    `position` decides where the evidence sits. Always putting it first would
    test only whether the model reads the opening, which is not what long
    context means -- a real alert arrives somewhere in the middle of a log
    excerpt.
    """
    if backbone.count_tokens(evidence) >= target:
        return evidence

    pool = _filler_lines(4000, seed)

    # Binary search on the number of filler lines. Growing in fixed blocks
    # overshot every small target -- twenty-five log lines already exceed 512
    # tokens, so the loop kept an empty prefix and padded nothing below 2k --
    # and correcting line by line meant re-tokenising an 8k string once per
    # line, which is minutes rather than seconds.
    def tokens_with(count: int) -> int:
        return backbone.count_tokens("\n".join([evidence, *pool[:count]]))

    low, high = 0, len(pool)
    if tokens_with(high) <= target:
        low = high
    else:
        while low < high:
            middle = (low + high + 1) // 2
            if tokens_with(middle) <= target:
                low = middle
            else:
                high = middle - 1
    lines = pool[:low]

    if position == "start":
        parts = [evidence, *lines]
    elif position == "end":
        parts = [*lines, evidence]
    else:
        middle = len(lines) // 2
        parts = [*lines[:middle], evidence, *lines[middle:]]
    return "\n".join(parts)


def context_sweep(
    engine: Engine,
    cases: list[BenchCase],
    lengths: Sequence[int],
    *,
    positions: Sequence[str] = ("start", "middle", "end"),
    seed: int = 1337,
) -> dict[str, Any]:
    """Accuracy as the state grows, with the evidence placed at varying depths.

    The point of this measurement is comparative. Laya's English checkpoint tops
    out at 512 tokens and its multilingual one at 1,024, with a stated state
    budget around 320 -- so from the 1,024 row onward this is a benchmark the
    alternatives cannot run at all. A truncated state does not error; it answers
    from whatever survived the cut.
    """
    backbone = engine.backbone
    results: dict[str, Any] = {}

    for target in lengths:
        for position in positions:
            hits: list[bool] = []
            observed: list[int] = []
            for i, case in enumerate(cases):
                state = _pad_to_tokens(case.state, target, position, backbone, seed + i)
                observed.append(backbone.count_tokens(state))
                answer = engine.evaluate(state, {"q": case.question}).answers["q"]
                dump = answer.model_dump(exclude_none=True)
                if "probabilities" in dump:
                    probs = np.array([dump["probabilities"][k] for k in case.option_keys])
                else:
                    value = dump.get("noul", dump.get("probability", 0.5))
                    probs = np.array([1.0 - value, value])
                hits.append(int(np.argmax(probs)) == case.label)

            key = f"{target}:{position}"
            results[key] = {
                "target_tokens": target,
                "position": position,
                "median_tokens": int(np.median(observed)),
                "n": len(hits),
                "accuracy": float(np.mean(hits)),
                # Laya truncates past these; the row is still reported, with a
                # note, because "we can run this and they cannot" is the claim.
                "beyond_laya_english": target > 512,
                "beyond_laya_multilingual": target > 1024,
            }
            log.info(
                "%5d tokens (%-6s) median=%-5d n=%-4d acc %.3f%s",
                target,
                position,
                results[key]["median_tokens"],
                len(hits),
                results[key]["accuracy"],
                "   [beyond Laya's limit]" if target > 1024 else "",
            )

    return results


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


class Scorer(Protocol):
    """Anything that can answer a `BenchCase` with a probability vector.

    The benchmark used to take an `Engine` directly, which quietly made "the
    number" and "JEF's number" the same thing -- there was nowhere to put a
    comparator. P1.5d needs the opposite: several systems over *identical*
    cases, because a comparison against someone else's reported figure on
    someone else's subset is not a comparison. So the runner depends on this
    much and no more.

    `probs` returns one probability per entry of `case.option_keys`, in that
    order, already normalised. `calibrator` may be unfitted -- a baseline
    generally has nothing fitted, and the runner simply reports fewer columns
    for it rather than pretending otherwise.
    """

    name: str

    # A property, not a plain attribute: a scorer that wraps something else
    # generally forwards its calibrator rather than owning one, and a settable
    # attribute here would reject exactly that.
    @property
    def calibrator(self) -> Calibrator: ...

    def probs(self, case: BenchCase) -> np.ndarray: ...


@dataclass
class EngineScorer:
    """A JEF `Engine` as a scorer. This is the path every existing number took."""

    engine: Engine
    name: str = "jef"

    @property
    def calibrator(self) -> Calibrator:
        return self.engine.calibrator

    def probs(self, case: BenchCase) -> np.ndarray:
        answer = self.engine.evaluate(case.state, {"q": case.question}).answers["q"]
        dump = answer.model_dump(exclude_none=True)
        if "probabilities" in dump:
            probs = np.array([dump["probabilities"][k] for k in case.option_keys])
        else:
            value = dump.get("noul", dump.get("probability", 0.5))
            probs = np.array([1.0 - value, value])
        return probs / probs.sum()


def run_benchmark(scorer: Scorer, cases: list[BenchCase], alpha: float = 0.10) -> dict[str, Any]:
    """Evaluate a scorer over benchmark cases, grouped by task."""
    by_task: dict[str, list[BenchCase]] = {}
    for case in cases:
        by_task.setdefault(case.task, []).append(case)

    calibrator: Calibrator = scorer.calibrator
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
            probs = scorer.probs(case)

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
    parser.add_argument(
        "--scorer",
        choices=["jef", "zero-shot"],
        default="jef",
        help="which system answers the cases. 'zero-shot' is the same backbone "
        "with no trained head and nothing calibrated -- the floor the head has "
        "to beat, measured on exactly the cases JEF is measured on.",
    )
    # Defaults per scorer, because docs/benchmarks.json is what the badge script
    # reads: a baseline run must not be able to overwrite JEF's published
    # numbers just by leaving a flag off.
    parser.add_argument("--out", default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tmmlu-per-subject", type=int, default=12)
    parser.add_argument("--tmmlu-subjects", nargs="*", default=None)
    parser.add_argument("--cti-mcq-limit", type=int, default=400)
    parser.add_argument("--cti-vsp-limit", type=int, default=250)
    parser.add_argument("--uncalibrated", action="store_true", help="skip the calibrator")
    parser.add_argument(
        "--context-sweep",
        action="store_true",
        help="measure accuracy as the state grows, with the evidence buried at "
        "varying depths. Past 1,024 tokens this is a benchmark Laya cannot run.",
    )
    parser.add_argument(
        "--sweep-lengths",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048, 4096, 8192],
    )
    parser.add_argument("--sweep-cases", type=int, default=40)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from jef_core import BilinearHead
    from jef_core.backends import load_backbone
    from jef_core.head import DecisionHead, ZeroShotHead

    if args.out is None:
        args.out = (
            "docs/benchmarks.json"
            if args.scorer == "jef"
            else f"docs/benchmarks-{args.scorer}.json"
        )

    model_dir = Path(args.model_dir)
    head: DecisionHead
    if args.scorer == "zero-shot":
        # The baseline is the untrained system, so it gets neither the head nor
        # the calibrator -- loading either would be measuring JEF again. Same
        # backbone, same cases, same exclusions: the only difference is the
        # thing P1.3 trained.
        head = ZeroShotHead()
        calibrator = Calibrator()
        log.info("scorer: zero-shot (%s, no trained head, uncalibrated)", args.backbone)
    else:
        head = BilinearHead.load(model_dir / "head.npz")
        calibrator = (
            Calibrator() if args.uncalibrated else Calibrator.load(model_dir / "calibration.json")
        )
    engine = Engine(load_backbone(args.backbone, threads=args.threads), head, calibrator)
    scorer = EngineScorer(engine, name=args.scorer)

    excluded = training_cve_ids(Path(args.corpus))
    log.info("training corpus contributed %d CVE id(s) to the exclusion set", len(excluded))

    cases: list[BenchCase] = []
    cases += list(_tmmluplus(args.tmmlu_per_subject, args.tmmlu_subjects))
    cases += list(_cti_mcq(args.cti_mcq_limit))
    cases += list(_cti_vsp(args.cti_vsp_limit, excluded))
    log.info("evaluating %d benchmark cases", len(cases))

    if args.context_sweep:
        # One task only: attack vector is where the model is strongest, so a
        # decline with length is attributable to length rather than to the task.
        sweep_cases = [c for c in cases if c.task == "cti-vsp/attack_vector"][: args.sweep_cases]
        if not sweep_cases:
            log.error("no cti-vsp/attack_vector cases available for the sweep")
            return 1
        log.info("context sweep over %d cases", len(sweep_cases))
        sweep = context_sweep(engine, sweep_cases, args.sweep_lengths)
        suffix = "" if args.scorer == "jef" else f"-{args.scorer}"
        out = Path(args.out).with_name(f"context-sweep{suffix}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "backbone": args.backbone,
                    "note": (
                        "Accuracy as the state grows, with the evidence buried at "
                        "varying depths in routine log noise. Laya tops out at 512 "
                        "tokens (English) and 1,024 (multilingual), so rows beyond "
                        "those cannot be run on it -- a truncated state does not "
                        "error, it answers from whatever survived the cut."
                    ),
                    "results": sweep,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        log.info("wrote %s", out)
        return 0

    results = run_benchmark(scorer, cases)

    payload = {
        "scorer": scorer.name,
        "model_dir": str(model_dir) if args.scorer != "zero-shot" else None,
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
