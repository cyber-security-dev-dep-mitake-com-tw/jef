# Validation ledger

One row per gate from the v3 plan. A result may be claimed only when its row names an artifact, the commit it came from, who produced it, and what it showed. Everything else is **Pending**. Review meetings report only rows with an artifact.

Frozen candidate: [`freeze-2026-10-17.md`](freeze-2026-10-17.md).

| Gate | What it needs | Artifact | Commit | By | Result | Known gap |
|---|---|---|---|---|---|---|
| **G0** engineering baseline | same-SHA CI, release, code-level safety pack | CI run `39034461974`; release run `39034480834`; artifact `safety-evidence-4f947838…` | `4f94783` | GitHub Actions | CI and release **pass**; safety pack S01–S13, S15, S16 and the production profile **pass**, 0 failed | S14 egress is a lab property and has no evidence yet |
| G0 lab | S14 egress log, mock ledger, restart persistence on the lab VM | **Pending** | | | | needs the no-uplink ESXi port group |
| **G1** research benchmark | source-labelled, reproducible, with families and failure rates | [`research/ctu13-source-label-v1/`](research/ctu13-source-label-v1/README.md): `results.json`, `run.py`, `dedupe.json` | code on this branch, data from the pinned CTU-13 archive | Claude Code session for Dennis Lee | **SOURCE-LABEL RESEARCH.** Leave-one-scenario-out, 13 folds, rules and logreg only. With IPs masked, logreg median macro-F1 0.854 (worst 0.337); rules 0.155. As built, logreg 0.950 is mostly IP memorisation | no JEF NDR model yet; one dataset; 13 families; C2 concentrated in one scenario |
| **G2** human evidence | 50–65 case single-reviewer pilot (Tier B), then independent double review | **Pending** | | | | no reviewer yet; never a second handle or an LLM as reviewer B |
| **G3** calibrated SOC task | schema 2, ≥150 calibration records per bucket, unseen test split | **Pending** | | | | shipped model is schema 1 |
| **G4** shadow SOC | side-effect-free replay with cost and risk measurement | **Pending** | | | | needs G0 lab, G1, G2 |

## How to add a row

1. Produce the artifact from a frozen commit (`scripts/safety-evidence.py`, `jef-bench evaluate`, `jef-bench annotate pilot-report`, a benchmark artifact verified with `scripts/benchmark-manifest.py verify`).
2. Record the artifact's name or path and digest, the commit, the operator, and the headline result in the table, plus the gap that remains.
3. A report's `evidence_class` decides how it may be described: `TIER-A GOLD`, `SINGLE-REVIEW RESEARCH` or `SOURCE-LABEL RESEARCH`.

## Stop conditions (plan v3 §10.3)

Stop the candidate at its current gate if any T2+ action dispatches without a grant, a legacy or incompatible calibration reports `automation_ready`, a dataset or test-set digest cannot be traced, a test split leaks into training, data handling breaks a licence or confidentiality, or the model cannot keep `unknown` / abstain.
