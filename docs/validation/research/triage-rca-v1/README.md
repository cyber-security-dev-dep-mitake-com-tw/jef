# Incident-triage RCA on jef-v0: fraud bias, severity, baselines (v1)

**Evidence class: SYNTHETIC LABELS RESEARCH (Tier E) on the deployed `jef-v0`.**

- [`run.py`](run.py) fixed the criteria for R1–R5 on 2026-10-20, before they
  were measured.
- One thing had already been seen when `run.py` was written: the baseline
  comparison B on the synthetic test split. Its decision rule was
  pre-registered earlier, in
  [`../../plans/2026-10-20-incident-triage-validation-first.md`](../../plans/2026-10-20-incident-triage-validation-first.md).
- Results: [`results.json`](results.json) (sha256 `54cfea3f…`).

**Cases**
- [`synthetic/dev-v1`](../../../../benchmark/incident-triage/synthetic/): 78
  cases, labels by construction.
- The six K8S Lab canary reconstructions and their three twins; their labels
  are pending.

**Model:** `jhu-clsp/mmBERT-base` at the pinned revision, with the shipped
head and schema-1 calibration. Head sha256 `202b9857…` matches
`models/jef-v0/manifest.json`.

## What this says, first

`jef-v0` was never trained on `soar_zh`; it was excluded because routing
scored at chance when it was tried. Its incident-triage answers are the
head's zero-shot reading of the scene's own label texts, and on these cases
they are worse than a page of keyword rules:

- It reads a real incident as a false positive in 10 of 22 cases.
- Its routing is at or below chance.
- Its severity is close to inverted: ransomware reads 資訊, benign activity
  reads 危急.

**The fraud bias is a label prior, not the alert.** Content-free states
route to fraud: 「告警。」 0.59 and 「系統通知。」 0.61. Option order changes
nothing. This is what the Lab saw.

## Verdicts against the pre-registered criteria

| | Question | Criterion | Result | Verdict |
|---|---|---|---|---|
| R1 | Fraud bias on non-fraud cases (72) | mean P(fraud) > 0.30 and fraud top > 40% | 0.365; top for 48.6% | **present** |
| R2 | Option order (5 rotations, 87 cases) | spread > 0.10 or top changes > 25% | spread 0.000; changes 0% | not order-sensitive |
| R3 | zh-TW/en pairs (39) | mean \|ΔP(fraud)\| > 0.10 or top disagrees > 25% | 0.118; disagrees 46% | **language-dependent** |
| R4 | State length | \|Spearman ρ\| > 0.30 | ρ = −0.088 | not length-driven |
| R5 | Content-free states (4) | mean P(fraud) > 0.30 | 0.472 (告警。0.59, 系統通知。0.61, Notification. 0.48, Alert. 0.22) | **prior towards fraud** |
| B | Routing decision (plan rule) | B3 macro-F1 < B1 → keep a person | 0.089 vs 0.867 | **routing stays human or rule-assisted** |

**R2 says the head scores each option independently**, so no
debiasing-by-shuffling will help. **R5 locates the bias** in how the
backbone and head read 「金融詐欺與異常交易：盜用帳戶、異常下單、洗錢態樣」
against an alert. With nothing to read, that description wins, and it wins
more in zh-TW than in English.

## B: baselines on the locked synthetic test split (24 cases)

| Baseline | Attack → FP | Evidence recall (insufficient) | Owner macro-F1 | Severity MAE / critical→low | Missed urgent |
|---|---:|---:|---:|---:|---:|
| B1 rules | 0 / 22 | 1.00 | 0.867 | 0.05 / 0 | 0 |
| B2 logreg (44 training cases) | 1 / 22 | 0.00 | 0.478 | 0.32 / 0 | 0 |
| **B3 jef-v0** | **10 / 22** | 0.00 | **0.089** | **1.36 / 2** | **6** |

B0 fail-closed sends 24 of 24 to a person and automates nothing.

**Caveat on B1.** The B1 rules were written by someone who had read the
synthetic families, so B1 is optimistic here. That only strengthens the
decision, which already keeps routing with people. Read B1 as "a page of
rules can encode the labelbook", not as rule accuracy in production.

## S1: severity errors, by event (14)

- **Critical read as 資訊.** 2 cases: the vssadmin ransomware pair, zh-TW and
  en. Also 1 MFA-fatigue takeover.
- **Benign read as 危急.** 11 cases, all hard negatives with approval
  evidence: SCCM baseline, monitoring heartbeat, approved Nessus, jump host,
  service account, MFA re-enrolment, DAST scan, and a registered trip abroad.

Each case, with its state and distribution, is in `results.json`. The pattern
is the opposite of what triage needs. This is not a threshold problem.

## L: the canary reconstructions, live, against what the Lab recorded

| Case | Lab P(FP) | Live P(FP) on reconstruction | Lab owner → live | Lab severity → live |
|---|---:|---:|---|---|
| c2-beacon | 0.49 | 0.40 | fraud → fraud | 危急 → 中 |
| ransomware-vss | 0.85 | 0.72 | fraud → infra | 低 → 資訊 |
| ssh-brute-success | 0.59 | 0.45 | fraud → infra | — → 危急 |
| nightly-backup | 0.36 | 0.44 | fraud → fraud | 高 → 危急 |
| win-patch | 0.92 | 0.77 | not asked → infra | not asked → 資訊 |
| nessus-approved | 0.29 | 0.37 | soc → fraud | 危急 → 中 |

The reconstructions are not the original text, so this is a direction check
only. The same ordering appears: ransomware and the patch alert read as the
most "benign".

## Next

- **Routing stays with people.** No routing gate may be calibrated or enabled
  (decision rule B).
- **Training.** A `soar_zh` head needs task-specific, human-labelled data.
  PR-06 builds the pipeline and its refusal checks. On the synthetic data
  alone the expected outcome is `insufficient`, and that outcome is the
  point.
- **Fraud prior.** Any future candidate must report R5 on the same
  content-free states. A prior above 0.30 on content-free states disqualifies
  it from routing.
- **Severity.** Severity needs the labelbook matrix inputs (asset criticality,
  containment) in the state or as features. These are not in today's alerts;
  that is a data and connector question before it is a model question.
