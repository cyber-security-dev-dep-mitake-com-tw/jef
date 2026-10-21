# Pre-registration: incident-triage head-to-head (2026-10-21)

Fixed on 2026-10-21, before any human-labelled incident-triage test set or any comparison score existed. It applies to every comparison of a JEF specialist head with baselines (B0 fail-closed, B1 rules, B2 logistic regression, B3 shipped `jef-v0`) and with external systems (the official TypeSafe Jev, Laya, a fixed LLM). Changing anything below after a number is seen makes the comparison exploratory, and it must be reported as such.

Implementation: `jef-bench triage compare` (`jef_train.socbench.triage_compare`). Source: §7 of the SOC Specialist v5 plan.

## What is compared

- **Bare decision quality** (this pre-registration): `jef-bench triage evaluate --mode questions` reports, one per system. All systems get the same pack bytes, the same scene (the same questions, options and option order) and the same locked split.
- **End-to-end scene outcomes** (separate leaderboard): `--mode scene` reports. The scene's conservative policy is never counted as model accuracy.
- **Pairing:** each task is compared on the cases every system answered. Contract failures (a refusal, a missing question, a partial distribution) are listed per system and are counted as failures; they are never filled in.

## Primary metric per task

| Task | Primary metric | Direction | Also reported |
|---|---|---|---|
| `false_positive` | recall of real events (gold `false`) = 1 − attack→benign rate | higher | every attack called benign, by case id |
| `evidence_sufficiency` | recall of "insufficient" (gold `false`) | higher | |
| `owner` | macro-F1 (the harness's definition: an acceptable owner counts as correct) | higher | confusion, mean P(fraud) |
| `severity` | MAE to the nearest acceptable level | lower | critical→low/info, by case id |
| `urgency` | recall of urgent (gold `true`) | higher | |

Priority ranking: cases are sorted by 1 − P(false positive = true). Recall of `truth_of_event = malicious` cases is measured in the first 10 % and the first 20 %.

## Statistics

- 95 % percentile intervals by **paired cluster bootstrap**: families are resampled, 1000 draws, seed 1337. Differences are oriented so that positive means better than the reference system.
- **Minimum support:** at least **30 families** with gold for a task in the compared split. Below that, the task is descriptive only.

## Decision rules

1. **Safety no-go (scene leaderboard).** Any dispatch, close or suppress, or any T2+ proposal while `automation_ready` is false, fails the system outright, whatever its quality.
2. **Non-inferiority to baselines.** A new head is non-inferior on a task when the lower bound of (head − strongest of B1, B2) is above **−0.05** on that task's primary metric. B1 and B2 are both run; the stronger one on the *calibration* split is the reference.
3. **Superiority.** Superiority over a system on a task needs the lower bound of the paired difference to be **above 0**. It is claimed per task, never as "better overall".
4. **Critical misses.** The head's critical→low/info count and its attack→benign count must not exceed the reference's. Every case id is listed. With fewer than 10 gold-critical cases, report "insufficient support" and make no claim about rare misses.
5. **Calibration.** A head's probabilities are compared on Brier/ECE only when its schema-2 buckets are `validated` and `coverage_check` holds on this split. Otherwise only decisions are compared.
6. **Workflow vs quality.** If a system wins only on cost or workload and not on rules 2–3, it may be described as a workflow or cost advantage, never as better decision quality.

## When a report is a claim

The report says `PRE-REGISTERED COMPARISON` only when all of these hold. Otherwise it says `NOT A CLAIM` and lists the reasons.

- Every scored label is `double-human-reviewed` and `confirmed` (Tier A).
- The pack is the locked bytes.
- Every compared task meets the minimum support.

These do not block the report, but they are recorded with it:

- **Arm selection.** S1–S6 arms are chosen on the calibration split. A test-split result is read once, after the choice, and never used to choose.
- **Unreachable systems.** A system that could not be reached is `NOT_RUN`. A stand-in (e.g. `research/jev-local`) is never reported as the official Jev.
- **Private data.** Private telemetry and reviewer-sourced cases are not sent to external systems without a separate data-processing approval.
