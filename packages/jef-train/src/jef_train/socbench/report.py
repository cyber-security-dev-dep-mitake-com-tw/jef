"""Score a locked split with several scorers and write one comparable report.

Comparability is enforced, not hoped for: the lock is verified against the
case and gold files first (any edit refuses to run), every scorer sees the
same cases in the same order, the bag-of-words baseline is fitted on the
lock's train split only, and the report records the lock digests, split and
scorer configuration beside the numbers.

Per scorer, per task, the report gives (review 11.3, 11.5):

* multiclass: accuracy and macro-F1 with family-clustered 95% CIs, the
  confusion matrix, ECE / Brier / NLL, risk-coverage;
* binary (attack vs not, over known labels): PR-AUC with CI, FNR at 1% and
  5% FPR, precision at analyst budgets, and the PPV that the TPR and FPR at
  0.5 imply at 1, 3 and 5 percent prevalence;
* unknown handling, separately: how often a scorer would have auto-closed
  a case whose gold is ``unknown``, and how often it would have auto-closed
  a positive (mistaken closures);
* the same headline numbers broken down by language, trust tier and source;
* p50 / p95 scoring latency.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import metrics as m
from .annotate import NEGATIVE_LABELS, TASK_LABELS
from .baselines import Scorer
from .records import trust_tier
from .splits import verify_lock

__all__ = ["POSITIVE_LABELS", "evaluate"]

#: What counts as "an attack" for the binary metrics, per task.
POSITIVE_LABELS: Mapping[str, frozenset[str]] = {
    "soc.alert.verdict.v1": frozenset({"confirmed_malicious", "suspicious"}),
    "soc.ndr.technique.v1": frozenset({"c2", "botnet", "dns_tunnel"}),
    "soc.edr.execution.v1": frozenset({"malicious"}),
    "soc.identity.abuse.v1": frozenset({"compromised", "suspicious"}),
    "soc.phishing.verdict.v1": frozenset({"phishing", "suspicious"}),
}
#: Gold verdicts excluded from the binary metrics (they are neither attack nor not).
UNDETERMINED = frozenset({"unknown", "insufficient_evidence"})
AUTO_THRESHOLD = 0.9
PREVALENCES = (0.01, 0.03, 0.05)
BUDGETS = (10, 50, 100)


def _macro_f1(y: Sequence[str], pred: Sequence[str]) -> float | None:
    labels = sorted(set(y))
    if not labels:
        return None
    f1s = []
    for lab in labels:
        tp = sum(a == lab and b == lab for a, b in zip(y, pred, strict=True))
        fp = sum(a != lab and b == lab for a, b in zip(y, pred, strict=True))
        fn = sum(a == lab and b != lab for a, b in zip(y, pred, strict=True))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def _round(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


def _block(task: str, rows: list[dict[str, Any]], *, with_ci: bool) -> dict[str, Any]:
    labels = list(TASK_LABELS[task])
    y = [r["gold"] for r in rows]
    pred = [r["pred"] for r in rows]
    probs = np.array([[r["probs"][lab] for lab in labels] for r in rows])
    idx = [labels.index(v) if v in labels else labels.index(labels[-1]) for v in y]
    top = probs.max(axis=1)
    correct = [a == b for a, b in zip(y, pred, strict=True)]
    families = [r["family"] for r in rows]

    out: dict[str, Any] = {
        "n": len(rows),
        "accuracy": float(np.mean(correct)) if rows else None,
        "macro_f1": _macro_f1(y, pred),
        "ece": m.ece(top, correct),
        "brier": m.brier(probs, idx),
        "nll": m.nll(probs, idx),
        "risk_coverage": m.risk_coverage(top, correct),
        "gold_distribution": dict(Counter(y)),
    }
    if with_ci and rows:
        arr_y, arr_p = np.array(y), np.array(pred)
        out["accuracy_ci"] = m.cluster_bootstrap(
            families, lambda i: float(np.mean(arr_y[i] == arr_p[i]))
        )
        out["macro_f1_ci"] = m.cluster_bootstrap(
            families, lambda i: _macro_f1(list(arr_y[i]), list(arr_p[i]))
        )
        out["confusion"] = {
            g: dict(Counter(p for gg, p in zip(y, pred, strict=True) if gg == g))
            for g in sorted(set(y))
        }

    positives = POSITIVE_LABELS.get(task)
    if positives:
        known = [i for i, v in enumerate(y) if v not in UNDETERMINED]
        score = [sum(rows[i]["probs"].get(lab, 0.0) for lab in positives) for i in known]
        truth = [y[i] in positives for i in known]
        binary: dict[str, Any] = {"n_known": len(known), "positives": int(sum(truth))}
        if known and any(truth) and not all(truth):
            binary["pr_auc"] = m.average_precision(score, truth)
            if with_ci:
                s_arr, t_arr, fam = np.array(score), np.array(truth), [families[i] for i in known]
                binary["pr_auc_ci"] = m.cluster_bootstrap(
                    fam, lambda i: m.average_precision(s_arr[i], t_arr[i])
                )
            binary["fnr_at_fpr_1pct"] = m.fnr_at_fpr(score, truth, 0.01)
            binary["fnr_at_fpr_5pct"] = m.fnr_at_fpr(score, truth, 0.05)
            binary["precision_at_budget"] = {
                str(b): m.precision_at_budget(score, truth, b) for b in BUDGETS if b <= len(known)
            }
            tpr, fpr = m.tpr_fpr_at(score, truth, 0.5)
            binary["at_threshold_0.5"] = {"tpr": tpr, "fpr": fpr}
            if tpr is not None and fpr is not None:
                binary["ppv_at_prevalence"] = {
                    str(p): m.ppv_at_prevalence(tpr, fpr, p) for p in PREVALENCES
                }
        out["binary"] = binary

        undetermined = [r for r in rows if r["gold"] in UNDETERMINED]
        auto_close = [
            r
            for r in rows
            if r["pred"] in NEGATIVE_LABELS and r["probs"][r["pred"]] >= AUTO_THRESHOLD
        ]
        out["unknown_handling"] = {
            "undetermined_gold": len(undetermined),
            "would_auto_close_undetermined": sum(r in auto_close for r in undetermined),
            "mistaken_closures": sum(r["gold"] in positives for r in auto_close),
            "auto_close_threshold": AUTO_THRESHOLD,
        }
    return out


def evaluation_manifest(
    scorers: Sequence[Scorer], lock: Mapping[str, Any], split: str
) -> dict[str, Any]:
    """What produced this report, pinned: versions, artifacts, prompts, egress.

    The plan's rule for a fair comparison is that everything shared is the
    same and everything that differs is disclosed. The lock pins the shared
    part; this pins the rest, per scorer, from the scorer's own ``describe()``.
    """
    import platform
    import sys
    from datetime import UTC, datetime

    try:
        from jef_core import __version__ as core_version
    except Exception:  # pragma: no cover - version lookup is best effort
        core_version = "unknown"
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jef_core": core_version,
        "lock": {
            "cases_digest": lock.get("cases_digest"),
            "gold_digest": lock.get("gold_digest"),
            "seed": lock.get("seed"),
            "split": split,
        },
        "scorers": {
            s.name: (s.describe() if hasattr(s, "describe") else {"type": type(s).__name__})
            for s in scorers
        },
        "egress": sorted(
            {
                str(d.get("egress", "unknown"))
                for d in (s.describe() if hasattr(s, "describe") else {} for s in scorers)
            }
        ),
    }


def evaluate(
    cases: Sequence[Mapping[str, Any]],
    golds: Sequence[Mapping[str, Any]],
    lock: Mapping[str, Any],
    scorers: Sequence[Scorer],
    *,
    split: str = "test",
) -> dict[str, Any]:
    problems = verify_lock(lock, cases, golds)
    if problems:
        raise ValueError("the files are not the locked ones: " + "; ".join(problems))
    wanted = set(lock["splits"].get(split, []))
    if not wanted:
        raise ValueError(f"the lock has no {split!r} split")
    gold_by = {g["case_id"]: g for g in golds}
    items = sorted((c for c in cases if c["case_id"] in wanted), key=lambda c: c["case_id"])

    report: dict[str, Any] = {
        "lock": {
            "cases_digest": lock["cases_digest"],
            "gold_digest": lock["gold_digest"],
            "split": split,
            "n": len(items),
        },
        "scorers": {},
    }
    for scorer in scorers:
        rows: list[dict[str, Any]] = []
        latencies: list[float] = []
        failures: list[dict[str, str]] = []
        for c in items:
            started = time.perf_counter()
            try:
                probs = scorer.predict(c)
            except Exception as exc:  # a timeout, refusal or unparseable reply is a result too
                latencies.append(time.perf_counter() - started)
                failures.append(
                    {"case_id": c["case_id"], "task": c["task_id"], "error": type(exc).__name__}
                )
                continue
            latencies.append(time.perf_counter() - started)
            g = gold_by[c["case_id"]]
            rows.append(
                {
                    "task": c["task_id"],
                    "gold": g["gold"]["verdict"],
                    "pred": max(probs, key=lambda k: probs[k]),
                    "probs": probs,
                    "family": c["family_id"],
                    "language": str(c.get("language")),
                    "tier": trust_tier(g["label_origin"]),
                    "source": c["source_dataset"],
                    "track": "hard_negative" if c.get("hard_negative") else "natural",
                }
            )
        by_task: dict[str, Any] = {}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            grouped[r["task"]].append(r)
        for task, task_rows in sorted(grouped.items()):
            block = _block(task, task_rows, with_ci=True)
            for facet in ("language", "tier", "source", "track"):
                parts: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for r in task_rows:
                    parts[r[facet]].append(r)
                block[f"by_{facet}"] = {
                    k: _block(task, v, with_ci=False) for k, v in sorted(parts.items())
                }
            by_task[task] = block
        lat = np.array(latencies) if latencies else np.zeros(1)
        report["scorers"][scorer.name] = {
            "tasks": by_task,
            "latency_s": {
                "p50": float(np.quantile(lat, 0.5)),
                "p95": float(np.quantile(lat, 0.95)),
                "p99": float(np.quantile(lat, 0.99)),
            },
            # Every case the scorer could not answer -- timeout, refusal, bad JSON.
            # Metrics above cover the answered cases only; read them with this.
            "failures": {
                "count": len(failures),
                "rate": (len(failures) / len(items)) if items else 0.0,
                "by_error": dict(Counter(f["error"] for f in failures)),
                "cases": failures[:50],
            },
        }

    report["manifest"] = evaluation_manifest(scorers, lock, split)
    report["comparison"] = [
        {
            "scorer": name,
            "task": task,
            "macro_f1": b["macro_f1"],
            "pr_auc": b.get("binary", {}).get("pr_auc"),
            "ece": b["ece"],
            "mistaken_closures": b.get("unknown_handling", {}).get("mistaken_closures"),
            "failure_rate": s["failures"]["rate"],
        }
        for name, s in report["scorers"].items()
        for task, b in s["tasks"].items()
    ]
    return _round(report)
