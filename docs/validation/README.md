# Validation ledger

One row per gate from the v3 plan. A result may be claimed only when its row names an artifact, the commit it came from, who produced it, and what it showed. Everything else is **Pending**. Review meetings report only rows with an artifact.

Frozen candidate: [`freeze-2026-10-17.md`](freeze-2026-10-17.md).

| Gate | What it needs | Artifact | Commit | By | Result | Known gap |
|---|---|---|---|---|---|---|
| **G0** engineering baseline | same-SHA CI, release, code-level safety pack | CI run `39034461974`; release run `39034480834`; artifact `safety-evidence-4f947838…` | `4f94783` | GitHub Actions | CI and release **pass**; safety pack S01–S13, S15, S16 and the production profile **pass**, 0 failed | S14 egress is a lab property and has no evidence yet |
| G0 lab | S14 egress log, mock ledger, restart persistence on the lab VM | **Pending** | | | | needs the no-uplink ESXi port group |
| **G1** research benchmark | source-labelled, reproducible, with families and failure rates | [`research/ctu13-source-label-v1/`](research/ctu13-source-label-v1/README.md): `results.json`, `run.py`, `dedupe.json` | code on this branch, data from the pinned CTU-13 archive | Claude Code session for Dennis Lee | **SOURCE-LABEL RESEARCH.** Leave-one-scenario-out, 13 folds, rules and logreg only. With IPs masked, logreg median macro-F1 0.854 (worst 0.337); rules 0.155. As built, logreg 0.950 is mostly IP memorisation | no JEF NDR model yet; one dataset; 13 families; C2 concentrated in one scenario |
| G1 extension: JEF NDR head and transfer | a JEF scorer on held-out scenarios; a second dataset | [`research/ndr-head-and-transfer-v1/`](research/ndr-head-and-transfer-v1/README.md): `results.json` (sha256 `5b75839c…`), `run.py` | this branch; data from the pinned CTU-13 and AIT-ADS archives | Claude Code session for Dennis Lee | **SOURCE-LABEL RESEARCH.** JEF head median macro-F1 0.768 against logreg 0.849 (wins 6 of 13 folds); **36 mistaken closures** against logreg's 0; lowest ECE (0.052). Transfer between flows and alerts is near chance (best ROC-AUC 0.65) | head not yet usable for closing; Tier C and D labels only |
| G1 v2: mistaken closures and cost-aware heads | root cause per case; pre-registered v2 against v1 on the same folds | [`research/ndr-head-v2/`](research/ndr-head-v2/README.md): `results.json` (`29d9b14d…`), `rca.json`, `run.py` | this branch | Claude Code session for Dennis Lee | **SOURCE-LABEL RESEARCH.** Root cause: CTU-13 labels flows by infected host, so DNS and short web flows from those hosts look benign once IPs are masked (logreg also called 34 of the 36 benign). Cost-aware head: mistaken closures 36 → 12, miss rate at 1% FPR 0.374 → 0.266, but macro-F1 0.768 → 0.713, so **not an improvement** under the fixed criteria. No calibration-fitted close gate was safe on test (logreg closed 33 attacks) | needs host and time context, and flow-level reviewer labels |
| **G2** human evidence | 50–65 case single-reviewer pilot (Tier B), then independent double review | **Pending** | | | | no reviewer yet; never a second handle or an LLM as reviewer B |
| **G3** calibrated SOC task | schema 2, ≥150 calibration records per bucket, unseen test split | **Pending** (machinery shown on source labels: [`ndr-head-and-transfer-v1`](research/ndr-head-and-transfer-v1/README.md) §B, 13 of 13 buckets `validated` with a median of 1,252 records; conformal coverage below target in 7 of 13 shifted folds) | | | | shipped model is schema 1; needs Tier A labels |
| **G4** shadow SOC | side-effect-free replay with cost and risk measurement | **Pending** | | | | needs G0 lab, G1, G2 |

## Manual actions

Everything that needs a person, an account, a contract or a lab is in [`manually_action_need_todos.md`](../../manually_action_need_todos.md).

## Cost of the JEF / Laya / official Jev comparison

Estimated, not run: [`cost-jef-laya-jev-2026-10-18.md`](cost-jef-laya-jev-2026-10-18.md). About US$5 of official Jev API usage per full comparison round; access, egress approval and reviewer time are the real constraints.

## How to add a row

1. Produce the artifact from a frozen commit (`scripts/safety-evidence.py`, `jef-bench evaluate`, `jef-bench annotate pilot-report`, a benchmark artifact verified with `scripts/benchmark-manifest.py verify`).
2. Record the artifact's name or path and digest, the commit, the operator, and the headline result in the table, plus the gap that remains.
3. A report's `evidence_class` decides how it may be described: `TIER-A GOLD`, `SINGLE-REVIEW RESEARCH` or `SOURCE-LABEL RESEARCH`.

## Stop conditions (plan v3 §10.3)

Stop the candidate at its current gate if any T2+ action dispatches without a grant, a legacy or incompatible calibration reports `automation_ready`, a dataset or test-set digest cannot be traced, a test split leaks into training, data handling breaks a licence or confidentiality, or the model cannot keep `unknown` / abstain.
