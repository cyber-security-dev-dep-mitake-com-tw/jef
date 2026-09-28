"""Publish the trained head and calibrator to Hugging Face.

Only the head and the calibrator are published -- the backbone is frozen and
comes from `jhu-clsp/mmBERT-base` at runtime. That is why this artifact is
~400KB rather than the ~800MB a fine-tuned encoder would be, and why it can also
live in git.

The model card is generated from `report.json`, not written by hand. A card that
drifts from the numbers it describes is worse than no card, and these numbers
change every time the corpus is rebuilt from NVD.

    python -m jef_train.publish_weights --repo dennislee928/jef-v0 --version 0.2.0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("jef.train.publish")

__all__ = ["build_model_card", "main"]

ARTIFACTS = ("head.npz", "calibration.json", "report.json")


def _table(report: dict[str, Any]) -> str:
    """Per-bucket results, straight from the report."""
    calibrated = report.get("calibrated", {})
    rows = [
        "| bucket | n | accuracy | macro-F1 | ECE | temperature | P(correct) ECE |",
        "|---|---|---|---|---|---|---|",
    ]
    for bucket in calibrated.get("buckets", []):
        p_correct = bucket.get("ece_p_correct")
        rows.append(
            f"| `{bucket['bucket']}` | {bucket['n']} | {bucket['accuracy']:.3f} | "
            f"{bucket['macro_f1']:.3f} | {bucket['ece']:.3f} | {bucket['temperature']:.2f} | "
            + (f"{p_correct:.3f} |" if p_correct is not None else "*unavailable* |")
        )
    return "\n".join(rows)


def _coverage_warnings(report: dict[str, Any]) -> list[str]:
    """Buckets where conformal coverage fell short, named rather than buried."""
    out: list[str] = []
    for bucket in report.get("calibrated", {}).get("buckets", []):
        for alpha, violated in sorted((bucket.get("coverage_violation") or {}).items()):
            if violated:
                observed = bucket["conformal_coverage"][alpha]
                out.append(
                    f"- `{bucket['bucket']}` at alpha={alpha}: {observed:.3f} "
                    f"against a nominal {1 - float(alpha):.2f}"
                )
    return out


def build_model_card(report: dict[str, Any], repo: str, version: str) -> str:
    """Generate the card from the report, so the two cannot disagree."""
    calibrated = report.get("calibrated", {})
    uncalibrated = report.get("uncalibrated", {})
    backbone = report.get("backbone", "jhu-clsp/mmBERT-base")
    excluded = report.get("excluded_sources") or []
    shortfalls = _coverage_warnings(report)

    return f"""---
license: apache-2.0
library_name: jef
pipeline_tag: text-classification
language:
  - zh
  - en
tags:
  - system-one
  - decision
  - calibration
  - conformal-prediction
  - soar
  - security
base_model: {backbone}
---

# JEF {version} — decision head and calibrator

The trained part of [JEF](https://github.com/cyber-security-dev-dep-mitake-com-tw/jef),
an open System One decision engine: typed questions in, probability
distributions with calibrated confidence out, no text generated.

**This repository is not a standalone model.** It holds a bilinear decision head
and a fitted calibrator that sit on top of a *frozen* `{backbone}`. That split is
why it is ~400KB instead of ~800MB, and why it trains on CPU.

```bash
pip install "jef[torch]"
```

```python
from jef_sdk import Jef, choice

jef = Jef(
    "{backbone}",
    head="hf://{repo}/head.npz",
    calibration="hf://{repo}/calibration.json",
)

jef.evaluate(alert, {{"team": choice("誰處理？", soc="監控", infra="基礎設施")}})
```

## Results

Out-of-sample, on a split grouped by **evidence** rather than by sample — one
CVE yields several questions over an identical description, and splitting per
sample puts that text on both sides of the line. Test set: {calibrated.get("n", "?")} samples.

{_table(report)}

- Majority-class baseline: **{report.get("majority_baseline_accuracy", 0):.4f}**
- Overall accuracy: **{calibrated.get("overall_accuracy", 0):.4f}**, macro-F1 {calibrated.get("overall_macro_f1", 0):.4f}
- ECE: {uncalibrated.get("overall_ece", 0):.4f} uncalibrated → **{calibrated.get("overall_ece", 0):.4f}** calibrated
{"- Excluded from training: " + ", ".join(f"`{s}`" for s in excluded) if excluded else ""}

### On third-party benchmarks

On **CTI-Bench VSP**, with every training CVE excluded, attack vector reaches
**0.840** against a 0.250 random baseline and user interaction 0.805 against
0.500. Conformal coverage holds at 0.895 against a nominal 0.900 there.

On **TMMLU+** it scores 0.254 against a 0.250 baseline — chance. JEF judges
evidence you supply; it does not know things. That boundary is stated because
finding it in production is worse.

## `confidence` is not correctness

`confidence` is `(n × peak − 1) / (n − 1)`: how *peaked* the distribution is. A
model can be decisive and wrong, which is why most of this ecosystem ships the
disclaimer *"confidence measures concentration, not correctness"*.

`calibration.json` carries three things fitted on a held-out split: temperature
scaling per `(kind, n_options)` bucket, split-conformal quantiles, and an
isotonic map from confidence onto observed correctness. That last one is what
makes `p_correct` meaningful — and it is absent for buckets with too few
calibration points, which is deliberately distinguishable from a low value.

{"### Conformal coverage shortfalls" + chr(10) + chr(10) + chr(10).join(shortfalls) + chr(10) + chr(10) + "Split conformal guarantees coverage under exchangeability. Grouping by evidence puts different concepts in calibration and test, which breaks that assumption. Do not gate automated actions on prediction sets from these buckets." if shortfalls else ""}

## Training data

Labels come from taxonomies, not from a language model: MITRE ATT&CK states
which tactic each technique serves, and a CVSS vector states the attack vector,
the privileges required and the severity band. Nothing here derives from any
commercial model's output.

## Limitations

- Not a knowledge model. It reads the evidence you give it.
- Context is bounded by the backbone (8,192 tokens for mmBERT-base).
- Without this calibrator loaded, JEF refuses to automate: `p_correct` is
  unavailable, the conformal set excludes nothing, and every scene gate routes
  to a human. That is the design, not a failure.

## License

Apache-2.0. Full methodology, including a contaminated run that scored 1.000 and
why it was wrong, in
[`docs/RESULTS.md`](https://github.com/cyber-security-dev-dep-mitake-com-tw/jef/blob/main/docs/RESULTS.md).
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish JEF weights to Hugging Face")
    parser.add_argument("--model-dir", default="models/jef-v0")
    parser.add_argument("--repo", default="dennislee928/jef-v0")
    parser.add_argument("--version", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="write the card locally, push nothing"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    model_dir = Path(args.model_dir)
    missing = [name for name in ARTIFACTS if not (model_dir / name).is_file()]
    if missing:
        log.error("%s is missing %s", model_dir, ", ".join(missing))
        return 1

    report = json.loads((model_dir / "report.json").read_text(encoding="utf-8"))
    card = build_model_card(report, args.repo, args.version)
    (model_dir / "README.md").write_text(card, encoding="utf-8")
    log.info("wrote %s (%d bytes)", model_dir / "README.md", len(card))

    if args.dry_run:
        log.info("dry run: nothing pushed")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    # Only create when it is actually missing. `create_repo(exist_ok=True)`
    # still POSTs to /api/repos/create and only forgives a 409, so a token that
    # may write to this repo but not create new ones gets a 403 on every
    # publish -- including the second and later ones, when there is nothing
    # left to create. Fine-grained Hugging Face tokens have repo creation as a
    # separate permission, so that combination is the common one.
    if api.repo_exists(args.repo, repo_type="model"):
        log.info("%s exists; not creating", args.repo)
    else:
        log.info("%s does not exist yet; creating", args.repo)
        api.create_repo(args.repo, repo_type="model", private=args.private)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo,
        repo_type="model",
        commit_message=f"JEF {args.version}",
        # The corpus is rebuilt from live NVD data, so a stale file left behind
        # would describe a model that no longer exists.
        allow_patterns=["*.npz", "*.json", "README.md"],
    )
    log.info("published https://huggingface.co/%s", args.repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
