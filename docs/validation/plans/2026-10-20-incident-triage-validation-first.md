# Plan: Incident-triage validation-first (2026-10-20)

Source: the "K8S Lab → Scene Quality, Validation-First v4" proposal, checked
against `main@9f765af`. Everything below the "Pre-registered" heading was
fixed on 2026-10-20, before any incident-triage score existed.

## Assessment of the proposal

The proposal is right that G0 Lab E2E passing says nothing about G2/G3
decision quality, and that the next work is the deployed `incident-triage`
scene rather than more CTU-13 research. Three corrections from the code:

- **The policy already blocked dispatch.** `policy.authorize` returned
  `review_required` for every incident-triage action. The defect was the
  *recommendation*: `proposed_action`, and the sealed proposal built from it,
  could be `close_pending_review` on a raw probability. PR-01 fixes the
  recommendation, not the dispatch path.
- **With the shipped `models/jef-v0` only two outcomes were reachable.** Its
  calibration is schema 1 (legacy). Tasked questions get `p_correct = None`,
  there is no `choice:5` bucket, so `owner.set_size = 5`. The scene ended
  either at L1 gate 0 (`close_pending_review`, on `P(FP) > 0.90`) or at L2
  gate 2 (`escalate_human`). L3 never ran, so urgency was never asked in
  scene mode.
- **Routing and severity were never trained.** `soar_zh` was excluded from
  jef-v0 (`docs/RESULTS.md`: routing 0.193 against 0.200 random). The Lab's
  owner and severity answers are zero-shot head output. The "Fraud bias" is a
  property of an untrained task, not a regression.

## Work (one PR each)

| # | Branch | Output |
|---|---|---|
| 1 | `fix/triage-legacy-advisory` | Engine advisory guard, scene v2 without the close gate, this plan |
| 2 | `test/triage-six-canaries` | `benchmark/incident-triage/canary/six-alerts-v1/`, replayed Lab answers, safety tests |
| 3 | `feat/triage-eval-harness` | `jef-bench triage evaluate --mode questions\|scene` |
| 4 | `feat/triage-dataset-schema` | Labelbook, synthetic dev set, hard negatives, zh-TW/en pairs, family lock |
| 5 | `bench/triage-baselines` | B0–B3 baselines, Fraud/Severity RCA |
| 6 | `train/soar-task-schema2-candidate` | Candidate pipeline and checks; status `Research` until human gold |
| 7 | `ci/triage-quality-gate` | Per-PR canary job, periodic real-model run, artifact-first |
| 8 | `ops/k8s-shadow-evidence` | Runtime evidence script for the existing lab |

## Advisory guard (PR-01)

When a `when` gate fires and any question it reads has a
`calibration_status` other than `validated`, the engine checks whether the
selected action is in the scene's advisory allowlist. If it is not:

- the trace records the selection as `withheld_action`;
- the scene's `advisory.unvalidated_fallback` (or a safe `fallthrough`) is
  proposed instead;
- `advisory_only` is set, and `advisory_reasons` names each question and its
  status.

An `else` gate or the fallthrough is reached by elimination, so it rests on
every condition that was evaluated and did not fire. Inverting a condition
(`p < 0.10 → escalate`, `else → close`) is therefore withheld too. The withheld
selection is also recorded in `reason_codes` (`ADVISORY_WITHHELD:<action>`,
`UNVALIDATED_CALIBRATION:<question>`), which the sealed proposal keeps.
Questions listed in `advisory.observe_only` may not be read by any gate. The
incident-triage allowlist is `escalate_human`, `queue_for_review` and
`page_oncall`: escalating to a person is not a state change.

The allowlist is by action key, so renaming an action does not get past it.
Allowlisted actions must have `human_review: true` and tier 1 or below. A
scene with no safe fallback proposes no action.

## Conditions for a close recommendation to return

All five must hold, in a later PR with its own study:

1. `is_false_positive.calibration_status == "validated"` against a
   task-scoped schema-2 artifact for `soar_zh.false_positive.v1`.
2. `is_false_positive.p_correct is not None`.
3. `P(FP = true)` above a threshold measured on an independent holdout and
   registered before that holdout is scored.
4. Approved change or routine-work evidence in `state`. Without it the
   expected label is `unknown`, never benign.
5. The action stays `human_review: true`, T2. A person confirms every close.

## Pre-registered

### Canary pack (PR-02)

The six canaries are regression and diagnosis cases. They are never an
accuracy estimate, never training data and never used for model selection.
Their labels carry `label_status: pending_owner_confirmation` until the owner
signs them off, and until then the canary tests assert safety only:

- S1: 6/6 `policy_verdict == review_required`.
- S2: 6/6 `execution_status == not_dispatched` and `executed_action == null`.
- S3: 0 cases with a proposed action outside the scene's
  `allowed_when_unvalidated` (for incident-triage v2: `escalate_human`,
  `queue_for_review`, `page_oncall`).
- S4: the replayed Lab answers (patch 0.92, ransomware 0.85 and the rest)
  reproduce byte-identical traces given the same scene and code SHA.

### Harness (PR-03)

- Each task is scored on its own denominator. `not_asked`, `unknown`,
  `insufficient_evidence`, `not_applicable` and `pending` labels are excluded
  from the denominator and reported as counts. None is scored as wrong, and
  `unknown` is never scored as benign.
- noul answers are scored on `P(true)` with a 0.5 threshold. `confidence` is
  never read as `P(true)`.
- Questions mode and scene mode are separate reports and are never merged.
- Metrics:

  | Task | Metrics |
  |---|---|
  | false_positive | precision and recall on `true`; FNR (real attack called FP); count of malicious cases with a close recommendation |
  | evidence_sufficiency | precision, recall, unknown rate |
  | routing | macro-F1, confusion matrix, per-owner base rate, mean `P(fraud)` |
  | severity | MAE on the 0–4 scale, quadratic-weighted kappa, under-estimation rate (predicted < gold), count of critical→low or info |
  | urgency | FNR and recall on `true` |

- Below 30 scored cases per task, the report lists per-case results only.
  There is no ECE, no reliability diagram and no confidence interval.
- Bootstrap intervals, when shown, resample independent incident families,
  not cases.
- Any run on the hashing backbone is stamped
  `CONTRACT SMOKE — NOT MODEL QUALITY`.

### Baselines (PR-05)

- B0 is fail-closed: every case goes to review.
- B1 is rules: approval or change evidence is required for benign, and
  routing is by keyword.
- B2 is TF-IDF logistic regression per task, trained on the synthetic train
  split only.
- B3 is jef-v0 at a pinned SHA.
- Decision rule: if B3's routing macro-F1 is below B1's on the same synthetic
  dev split, routing stays human or rule-assisted. No routing gate may be
  calibrated or enabled until a candidate beats B1 on independent
  human-labelled data.
- Synthetic results are evidence class Tier E and are described as
  engineering-contract coverage, never as accuracy.

### CI gate (PR-07)

The per-PR job gates on S1–S3 only. Model-quality numbers are reported, not
gated, until at least 100 independently human-labelled cases exist per task.
Thresholds for the periodic real-model run are added to this file, in a
separate commit, before that run's first comparison.

### Periodic real-model run (PR-07), fixed 2026-10-20 before its first run

`triage-quality.yml` runs on a schedule and on demand, against the pinned
backbone, head and calibration. It checks every artifact SHA against the
manifest, and a mismatch fails the run.

**Fails the run.** These are safety properties, not quality:
- **P1:** scene mode on the canaries breaks S1–S3, dispatches anything, or
  proposes a state-changing action on a malicious case.
- **P2:** the candidate dry run on synthetic labels reports anything other
  than `insufficient`. A pipeline that promotes Tier E data is broken.
- **P3:** a model, head, calibration or scene SHA does not match what the run
  was asked to test.

**Reported, never gated.** Model quality is reported, never gated, while
fewer than 100 independently human-labelled cases exist per task. The report
covers:
- the questions-mode metrics on the canaries and the synthetic test split;
- the baseline table;
- the routing decision;
- the RCA's R5 probe (P(fraud) on content-free states).

A change of more than 0.10 in owner macro-F1 or in R5 against the last
published run is flagged in the summary for a person to read. It is not a
failure: on 24 synthetic cases that size of change is within noise.

**When gating may start.** Gating on quality starts only after a new commit
to this file. That commit must name the human-labelled set, its size per
task, and the thresholds, and it must land before the first run that uses
them.

## What stays manual

- **Canary labels:** the owner signs off the expected labels for the six
  canaries.
- **Scene-specific pilot:** 50–100 cases labelled for the five tasks (Tier B).
  The existing 63-case SOCBench pilot does not count: its tasks are
  `soc.ndr.technique` and `soc.alert.verdict`.
- **Tier A:** two independent reviewers plus an adjudicator, at least 300
  cases.
- **Delivery:** proof that a queue actually received a case needs an
  external receipt from a test SOAR or ticketing tenant. `review_required` is
  a policy verdict, not a delivery.
- **Runtime:** the K8S runtime evidence and the 1/2/4/8 concurrency runs on
  the existing lab.
