#!/usr/bin/env python3
"""Summarise a triage-quality run for the job summary; flag, never gate, quality drift.

    python scripts/triage-quality-summary.py OUT_DIR

Reads what ``.github/workflows/triage-quality.yml`` wrote to OUT_DIR and the
last published RCA (``docs/validation/research/triage-rca-v1/results.json``).
Pre-registered in docs/validation/plans/2026-10-20-incident-triage-validation-first.md:
a change above 0.10 in owner macro-F1 or in the R5 probe is *flagged for a
person*; only the safety checks (P1-P3) fail the workflow, and they do that in
their own steps.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PUBLISHED = Path("docs/validation/research/triage-rca-v1/results.json")
FLAG = 0.10


def load(path: Path) -> dict:
    return json.loads(path.read_text("utf-8")) if path.is_file() else {}


def main(out: Path) -> int:
    published = load(PUBLISHED)
    rca = load(out / "rca" / "results.json")
    baselines = load(out / "baselines" / "results.json")
    scene = load(out / "canary-scene" / "results.json")
    candidate = load(out / "candidate" / "check.json")

    lines = ["# incident-triage quality run", ""]
    lines.append(
        "Quality is **reported, not gated** (fewer than 100 human-labelled cases per task). "
        "Safety checks P1-P3 fail the workflow in their own steps."
    )
    lines.append("")
    if scene:
        m = scene["metrics"]
        lines.append(
            f"- P1 canary scene mode: proposed {m['proposed_actions']}, dispatched {m['dispatched']}, "
            f"failures {scene['failures'] or 'none'}"
        )
    if candidate:
        lines.append(f"- P2 candidate dry run: **{candidate['status']}** (must be insufficient)")
    if baselines:
        lines.append(f"- Routing decision: {baselines['routing_decision'].get('routing')}")
        lines += [
            "",
            "| baseline | owner macro-F1 | attack→FP | severity MAE | missed urgent |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, res in baselines["baselines"].items():
            t = res["tasks"]
            lines.append(
                f"| {name} | {t['owner']['macro_f1']} | {t['false_positive'].get('attack_called_false_positive')} "
                f"| {t['severity']['mae']} | {t['urgency'].get('urgent_missed')} |"
            )
    flags = []
    if rca and published:
        now, then = rca["R5"]["mean_p_fraud"], published["R5"]["mean_p_fraud"]
        if abs(now - then) > FLAG:
            flags.append(f"R5 P(fraud) on content-free states moved {then} -> {now}")
        now_f1 = rca["B_baselines"]["tasks"]["jef"]["owner"]["macro_f1"]
        then_f1 = published["B_baselines"]["tasks"]["jef"]["owner"]["macro_f1"]
        if now_f1 is not None and then_f1 is not None and abs(now_f1 - then_f1) > FLAG:
            flags.append(f"jef owner macro-F1 moved {then_f1} -> {now_f1}")
        lines += ["", f"R5 P(fraud) on content-free states: {now} (published {then})"]
    lines += ["", "## Flags for a person", *([f"- {f}" for f in flags] or ["- none"])]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
