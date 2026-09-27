"""Build the JEF training corpus.

    python -m jef_train.build_corpus --out data/corpus

Writes ``train.jsonl``, ``calibration.jsonl``, ``test.jsonl`` plus a
``manifest.json`` recording exactly what went in. The manifest matters: every
claim this project makes about calibration is only as credible as the ability to
say where the numbers came from.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from .corpus.attack import build_attack_samples
from .corpus.cve import build_cve_samples
from .corpus.soar_zh import build_soar_zh_samples
from .schema import Sample, assert_no_group_leakage, split_samples, write_samples

log = logging.getLogger("jef.train.build")

__all__ = ["build_all", "main"]


def build_all(
    *,
    attack: bool = True,
    cve: bool = True,
    soar: bool = True,
    cve_records: int = 2000,
    soar_per_scenario: int = 40,
    refresh: bool = False,
) -> list[Sample]:
    samples: list[Sample] = []
    if attack:
        samples += build_attack_samples(refresh=refresh)
    if cve:
        samples += build_cve_samples(max_records=cve_records, refresh=refresh)
    if soar:
        samples += build_soar_zh_samples(per_scenario=soar_per_scenario)
    return samples


def summarise(samples: list[Sample]) -> dict[str, object]:
    """Per-source and per-bucket counts, plus label distributions.

    Label distributions are reported rather than hidden because several are
    genuinely skewed -- CVE attack vectors are overwhelmingly `network`, and most
    alerts are not known-benign. The skew is real, but it makes plain accuracy a
    misleading metric, so training weights classes and reporting leads with
    macro-F1.

    ``by_state_lang`` is reported for the same reason. Every *question* in this
    corpus is zh-TW, but two thirds of the *states* are English prose from ATT&CK
    and NVD. That is a deliberate mix, not an oversight, and any zh-TW claim has
    to be read against it.
    """
    counts: Counter[str] = Counter()
    kinds: dict[str, str] = {}
    n_options: dict[str, int] = {}
    labels: dict[str, Counter[str]] = {}

    for s in samples:
        counts[s.source] += 1
        kinds[s.source] = s.kind
        n_options[s.source] = s.n_options
        labels.setdefault(s.source, Counter())[s.option_keys[s.label]] += 1

    return {
        "total": len(samples),
        "by_bucket": dict(Counter(s.bucket for s in samples)),
        "by_state_lang": dict(Counter(str(s.meta.get("state_lang", "unknown")) for s in samples)),
        # Groups are the split unit. Far fewer groups than samples is expected
        # and is the point -- several questions share one piece of evidence.
        "groups": len({s.group_key for s in samples}),
        "by_source": {
            source: {
                "count": counts[source],
                "kind": kinds[source],
                "n_options": n_options[source],
                "labels": dict(labels[source]),
            }
            for source in sorted(counts)
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the JEF training corpus")
    parser.add_argument("--out", default="data/corpus", help="output directory")
    parser.add_argument("--cve-records", type=int, default=2000)
    parser.add_argument("--soar-per-scenario", type=int, default=40)
    parser.add_argument("--skip-attack", action="store_true")
    parser.add_argument("--skip-cve", action="store_true")
    parser.add_argument("--skip-soar", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="bypass the download cache")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    samples = build_all(
        attack=not args.skip_attack,
        cve=not args.skip_cve,
        soar=not args.skip_soar,
        cve_records=args.cve_records,
        soar_per_scenario=args.soar_per_scenario,
        refresh=args.refresh,
    )
    if not samples:
        log.error("no samples built -- every source was skipped or empty")
        return 1

    train, calibration, test = split_samples(samples, seed=args.seed)

    # Contamination fails the build rather than quietly inflating a published
    # number. This check is the reason the numbers in report.json can be trusted
    # at all: the first run of this pipeline scored 1.000 on SOAR routing
    # because the same alert text sat on both sides of the split.
    assert_no_group_leakage(train, calibration, test)

    out = Path(args.out)
    counts = {
        # all.jsonl lets the feature cache be built once over the whole corpus,
        # so re-splitting never costs another pass over the backbone.
        "all": write_samples(samples, out / "all.jsonl"),
        "train": write_samples(train, out / "train.jsonl"),
        "calibration": write_samples(calibration, out / "calibration.jsonl"),
        "test": write_samples(test, out / "test.jsonl"),
    }

    manifest = {
        "splits": counts,
        "seed": args.seed,
        "summary": summarise(samples),
        "note": (
            "soar_zh.* is synthetic and for training only. Headline numbers come "
            "from human-verified jef-bench-zh-tw and from independent third-party "
            "benchmarks; evaluating on a generator's own templates measures the "
            "generator."
        ),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("wrote %s to %s", counts, out)
    for bucket, n in sorted(manifest["summary"]["by_bucket"].items()):
        log.info("  bucket %-12s %d", bucket, n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
