"""Metrics, reported the way the ecosystem does not.

Accuracy leads nowhere on this corpus: CVE attack vectors are ~74% `network`,
and only a fifth of SOAR alerts are known-benign, so a model that reads nothing
and answers the majority class scores well. Macro-F1 is reported alongside, and
the calibration metrics are the point:

  - **ECE** -- does a stated probability of 0.9 correspond to being right 90% of
    the time? Nearly every open System One project declines to answer this.

    Two are reported, and the distinction is not pedantic. ``ece`` uses the
    top probability, which is the standard definition, the thing temperature
    scaling actually optimises, and the number comparable to what other projects
    publish. ``ece_confidence`` uses Jev's confidence statistic
    ``(n*peak - 1)/(n - 1)``, which is a *sharpness* measure on a different
    scale -- it is what scene gates threshold on, so its reliability curve is
    what a gate threshold should be read off. Conflating the two makes
    calibration look like it is failing when it is working: temperature scaling
    improves the first while leaving the second on its own scale.
  - **Brier** -- proper scoring rule over the full distribution, not just argmax.
  - **Conformal coverage** -- does the prediction set actually contain the truth
    at the promised rate? This is what turns "low confidence" into a guarantee
    rather than a feeling.

  - **ECE of P(correct)** -- the headline for scene gates. If the calibrator's
    confidence-to-correctness map is honest, thresholding a gate at 0.9 means
    the answer is right about 90% of the time. This is the number that decides
    whether the ecosystem's standing disclaimer still applies to JEF.

A reliability curve is emitted too, because a single ECE number hides whether a
model is uniformly overconfident or only wrong at the extremes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from itertools import pairwise

import numpy as np
from jef_core.calibration import Calibrator
from jef_core.mathx import brier_score, confidence, expected_calibration_error, softmax

from .cache import FeatureCache

__all__ = ["BucketReport", "Report", "evaluate_cache", "macro_f1", "reliability_curve"]


def macro_f1(predictions: Sequence[int], labels: Sequence[int], n_classes: int) -> float:
    """Unweighted mean of per-class F1.

    Unweighted on purpose: it is the metric that refuses to be flattered by a
    skewed prior, which is exactly what this corpus has.
    """
    scores = []
    for c in range(n_classes):
        tp = sum(1 for p, y in zip(predictions, labels, strict=True) if p == c and y == c)
        fp = sum(1 for p, y in zip(predictions, labels, strict=True) if p == c and y != c)
        fn = sum(1 for p, y in zip(predictions, labels, strict=True) if p != c and y == c)
        if tp == 0 and (fp or fn):
            scores.append(0.0)
        elif tp == 0:
            continue  # class absent from both truth and predictions
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def reliability_curve(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> list[dict[str, float]]:
    """Per-bin mean confidence vs observed accuracy."""
    conf = np.asarray(confidences, dtype=np.float64)
    hit = np.asarray(correct, dtype=bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict[str, float]] = []
    for lo, hi in pairwise(edges):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        out.append(
            {
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "count": int(mask.sum()),
                "mean_confidence": float(conf[mask].mean()),
                "accuracy": float(hit[mask].mean()),
            }
        )
    return out


@dataclass
class BucketReport:
    bucket: str
    n: int
    accuracy: float
    macro_f1: float
    #: Standard ECE over the top probability -- comparable across projects.
    ece: float
    #: ECE over Jev's raw confidence statistic, before the correctness map.
    ece_confidence: float
    #: ECE over the calibrated P(correct). None when no map was fitted.
    ece_p_correct: float | None
    brier: float
    mean_confidence: float
    mean_top_probability: float
    temperature: float
    conformal_coverage: dict[str, float] = field(default_factory=dict)
    mean_set_size: dict[str, float] = field(default_factory=dict)
    #: Top-probability reliability: bin mean probability against observed accuracy.
    reliability: list[dict[str, float]] = field(default_factory=list)
    #: Confidence-statistic reliability. Read gate thresholds off this one.
    reliability_confidence: list[dict[str, float]] = field(default_factory=list)


@dataclass
class Report:
    overall_accuracy: float
    overall_macro_f1: float
    overall_ece: float
    overall_ece_confidence: float
    overall_ece_p_correct: float | None
    overall_brier: float
    n: int
    buckets: list[BucketReport]
    by_source: dict[str, dict[str, float]]
    calibrated: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_cache(
    logits: list[np.ndarray],
    cache: FeatureCache,
    calibrator: Calibrator | None = None,
    *,
    alphas: Sequence[float] = (0.05, 0.10, 0.20),
) -> Report:
    """Score a head's raw logits against a labelled cache."""
    calib = calibrator or Calibrator()
    buckets = cache.buckets()

    all_conf: list[float] = []
    all_pcorrect: list[float] = []
    all_top: list[float] = []
    all_hit: list[bool] = []
    all_brier: list[float] = []
    per_source: dict[str, list[bool]] = defaultdict(list)
    reports: list[BucketReport] = []
    total_correct = 0

    for bucket, idx in sorted(buckets.items()):
        kind, n_options = bucket.split(":")
        width = int(n_options)
        temperature = calib.temperature(kind, width)

        probs = [softmax(logits[i], temperature=temperature) for i in idx]
        preds = [int(np.argmax(p)) for p in probs]
        labels = [int(cache.labels[i]) for i in idx]
        confs = [confidence(p) for p in probs]
        tops = [float(np.max(p)) for p in probs]
        hits = [p == y for p, y in zip(preds, labels, strict=True)]
        briers = [brier_score(p, y) for p, y in zip(probs, labels, strict=True)]

        for i, hit in zip(idx, hits, strict=True):
            per_source[cache.sources[i]].append(hit)

        p_corrects = [calib.p_correct(c, kind, width) for c in confs]
        have_map = all(p is not None for p in p_corrects) and bool(p_corrects)
        bucket_ece_pc = (
            expected_calibration_error(
                np.array([p for p in p_corrects if p is not None]), np.array(hits)
            )
            if have_map
            else None
        )
        if have_map:
            all_pcorrect.extend([p for p in p_corrects if p is not None])

        coverage: dict[str, float] = {}
        set_size: dict[str, float] = {}
        if calib.is_fitted():
            for alpha in alphas:
                sets = [calib.prediction_set(p, kind, width, alpha) for p in probs]
                covered = sum(1 for s, y in zip(sets, labels, strict=True) if y in s)
                coverage[f"{alpha:.2f}"] = covered / len(labels)
                set_size[f"{alpha:.2f}"] = float(np.mean([len(s) for s in sets]))

        reports.append(
            BucketReport(
                bucket=bucket,
                n=len(idx),
                accuracy=float(np.mean(hits)),
                macro_f1=macro_f1(preds, labels, width),
                ece=expected_calibration_error(np.array(tops), np.array(hits)),
                ece_confidence=expected_calibration_error(np.array(confs), np.array(hits)),
                ece_p_correct=bucket_ece_pc,
                brier=float(np.mean(briers)),
                mean_confidence=float(np.mean(confs)),
                mean_top_probability=float(np.mean(tops)),
                temperature=temperature,
                reliability_confidence=reliability_curve(confs, hits),
                conformal_coverage=coverage,
                mean_set_size=set_size,
                reliability=reliability_curve(tops, hits),
            )
        )

        all_conf.extend(confs)
        all_top.extend(tops)
        all_hit.extend(hits)
        all_brier.extend(briers)
        total_correct += sum(hits)

    return Report(
        overall_accuracy=total_correct / max(len(cache), 1),
        # Weighted by bucket size so a tiny bucket cannot dominate the headline.
        overall_macro_f1=float(
            np.average([r.macro_f1 for r in reports], weights=[r.n for r in reports])
        )
        if reports
        else 0.0,
        overall_ece=expected_calibration_error(np.array(all_top), np.array(all_hit)),
        overall_ece_confidence=expected_calibration_error(np.array(all_conf), np.array(all_hit)),
        overall_ece_p_correct=(
            expected_calibration_error(np.array(all_pcorrect), np.array(all_hit))
            if all_pcorrect and len(all_pcorrect) == len(all_hit)
            else None
        ),
        overall_brier=float(np.mean(all_brier)) if all_brier else 0.0,
        n=len(cache),
        buckets=reports,
        by_source={
            source: {"n": len(hits), "accuracy": float(np.mean(hits))}
            for source, hits in sorted(per_source.items())
        },
        calibrated=calib.is_fitted(),
    )


def majority_baseline(cache: FeatureCache) -> float:
    """Accuracy of always answering each bucket's most common label.

    Reported next to the model so a skewed prior cannot be mistaken for skill.
    """
    correct = 0
    for _bucket, idx in cache.buckets().items():
        counts = Counter(int(cache.labels[i]) for i in idx)
        correct += counts.most_common(1)[0][1]
    return correct / max(len(cache), 1)
