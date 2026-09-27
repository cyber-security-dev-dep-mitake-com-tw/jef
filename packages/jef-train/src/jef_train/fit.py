"""End-to-end: corpus -> features -> head -> calibrator -> report.

    python -m jef_train.fit --corpus data/corpus --out models/jef-v0

Writes ``head.npz``, ``calibration.json`` and ``report.json``. The report
contains the before/after calibration comparison, because "we calibrated it" is
a claim, and the delta is the evidence.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from jef_core.backends import load_backbone
from jef_core.calibration import Calibrator

from .cache import FeatureCache, build_feature_cache
from .evaluate import evaluate_cache, majority_baseline
from .schema import read_samples
from .train import TrainConfig, logits_for, train_head

log = logging.getLogger("jef.train.fit")

__all__ = ["main"]

_SPLITS = ("train", "calibration", "test")


def _cache_for(
    split: str, corpus: Path, cache_dir: Path, backbone_spec: str, threads: int | None
) -> FeatureCache:
    """Load a cached feature set, or build it once and keep it.

    The cache is keyed by backbone because features from a different encoder are
    not interchangeable -- silently reusing them would train a head against
    vectors it will never see at inference.
    """
    safe = backbone_spec.replace("/", "__")
    path = cache_dir / f"{split}.{safe}.npz"
    if path.exists():
        cached = FeatureCache.load(path)
        log.info("loaded cached features: %s (%d samples)", path, len(cached))
        return cached

    samples = list(read_samples(corpus / f"{split}.jsonl"))
    kwargs: dict[str, object] = {}
    if threads and backbone_spec != "hashing":
        kwargs["threads"] = threads
    backbone = load_backbone(backbone_spec, **kwargs)

    log.info("encoding %d %s samples with %s ...", len(samples), split, backbone_spec)
    started = time.perf_counter()
    cache = build_feature_cache(samples, backbone)
    log.info("encoded %s in %.1fs", split, time.perf_counter() - started)
    cache.save(path)
    return cache


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and calibrate a JEF head")
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--out", default="models/jef-v0")
    parser.add_argument("--cache-dir", default=".cache/features")
    parser.add_argument("--backbone", default="jhu-clsp/mmBERT-base")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--no-class-weighting",
        action="store_true",
        help="train on the raw prior; expect better accuracy and worse macro-F1",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    corpus = Path(args.corpus)
    cache_dir = Path(args.cache_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    caches = {
        split: _cache_for(split, corpus, cache_dir, args.backbone, args.threads)
        for split in _SPLITS
    }

    head = train_head(
        caches["train"],
        caches["calibration"],
        TrainConfig(
            rank=args.rank,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            class_weighting=not args.no_class_weighting,
            seed=args.seed,
            threads=args.threads,
        ),
    )
    head.save(out / "head.npz")

    # Calibrate on the split the head never trained on. Fitting temperature on
    # training data produces a calibrator that looks excellent and generalises
    # not at all -- the exact failure this project claims to fix.
    calib_logits = logits_for(head, caches["calibration"])
    calibrator = Calibrator(backbone=args.backbone, head=head.name)
    for bucket, idx in caches["calibration"].buckets().items():
        kind, width = bucket.split(":")
        calibrator.fit_bucket(
            kind,
            int(width),
            [calib_logits[i] for i in idx],
            [int(caches["calibration"].labels[i]) for i in idx],
        )
    calibrator.notes = (
        "Temperature and conformal quantiles fitted on a held-out calibration "
        "split, never on training data."
    )
    calibrator.save(out / "calibration.json")

    test_logits = logits_for(head, caches["test"])
    before = evaluate_cache(test_logits, caches["test"], None)
    after = evaluate_cache(test_logits, caches["test"], calibrator)
    baseline = majority_baseline(caches["test"])

    report = {
        "backbone": args.backbone,
        "head": head.name,
        "rank": args.rank,
        "class_weighting": not args.no_class_weighting,
        "splits": {k: len(v) for k, v in caches.items()},
        "majority_baseline_accuracy": baseline,
        "uncalibrated": before.to_dict(),
        "calibrated": after.to_dict(),
        "temperatures": {k: v.temperature for k, v in calibrator.buckets.items()},
    }
    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("--- test set (%d samples) ---", after.n)
    log.info("majority baseline accuracy  %.4f", baseline)
    log.info("accuracy                    %.4f", after.overall_accuracy)
    log.info("macro-F1                    %.4f", after.overall_macro_f1)
    log.info(
        "ECE (top prob)   uncalibrated %.4f -> calibrated %.4f",
        before.overall_ece,
        after.overall_ece,
    )
    log.info(
        "ECE (confidence) uncalibrated %.4f -> calibrated %.4f  [raw sharpness]",
        before.overall_ece_confidence,
        after.overall_ece_confidence,
    )
    if after.overall_ece_p_correct is not None:
        log.info(
            "ECE (P(correct)) %.4f  <- what a scene gate threshold actually means",
            after.overall_ece_p_correct,
        )
    log.info(
        "Brier  uncalibrated %.4f -> calibrated %.4f", before.overall_brier, after.overall_brier
    )
    for bucket in after.buckets:
        log.info(
            "  %-12s n=%-5d acc %.3f  macroF1 %.3f  ECE %.3f  T=%.2f  coverage@0.10 %s",
            bucket.bucket,
            bucket.n,
            bucket.accuracy,
            bucket.macro_f1,
            bucket.ece,
            bucket.temperature,
            bucket.conformal_coverage.get("0.10", float("nan")),
        )
        if bucket.ece_p_correct is None:
            log.info(
                "    %-12s no P(correct) map: too few calibration samples to fit one honestly",
                bucket.bucket,
            )
        else:
            log.info("    %-12s ECE of P(correct) %.3f", bucket.bucket, bucket.ece_p_correct)
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
