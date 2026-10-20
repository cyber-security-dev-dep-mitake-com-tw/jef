# Incident-triage cases

Inputs and expected answers for the five `soar_zh` tasks that
[`scenes/incident-triage.zh-tw.yaml`](../../scenes/incident-triage.zh-tw.yaml)
asks:

| Question id | Task | Kind | Expectation key |
|---|---|---|---|
| `is_false_positive` | `soar_zh.false_positive.v1` | noul | `false_positive` |
| `has_evidence` | `soar_zh.evidence_sufficiency.v1` | noul | `evidence_sufficiency` |
| `owner` | `soar_zh.routing.v1` | choice | `owner` |
| `severity` | `soar_zh.severity.v1` | score | `severity` |
| `urgent` | `soar_zh.urgency.v1` | noul | `urgency` |

These are kept apart from the socbench case and gold files because one alert
here carries five questions, and the scene may stop before asking most of
them. The plan and its pre-registered criteria are in
[`docs/validation/plans/2026-10-20-incident-triage-validation-first.md`](../../docs/validation/plans/2026-10-20-incident-triage-validation-first.md).

## Layout

```text
schema/case.schema.json          inputs only: state, evidence, provenance, state sha256
schema/expectation.schema.json   what a reviewer expects; scored blind
canary/six-alerts-v1/            the K8S Lab canaries
  cases.jsonl  expectations.jsonl  replay.jsonl  manifest.json
```

Loader and checks: `jef_train.socbench.triage` (`load_pack`, `pack_manifest`,
`privacy_findings`, `ReplayEngine`). After editing a pack, regenerate its
manifest:

```bash
uv run python -c "import json; from pathlib import Path; from jef_train.socbench.triage import pack_manifest; \
p = Path('benchmark/incident-triage/canary/six-alerts-v1'); \
(p / 'manifest.json').write_text(json.dumps(pack_manifest(p), ensure_ascii=False, indent=2, sort_keys=True) + '\n')"
```

## Evaluate

```bash
# What the scene proposed: verdicts, withheld selections, not-asked questions, dispatch
uv run jef-bench triage evaluate --mode scene \
  --pack benchmark/incident-triage/canary/six-alerts-v1 --out out/triage-scene
# The five questions, each on its own denominator
uv run jef-bench triage evaluate --mode questions \
  --pack benchmark/incident-triage/canary/six-alerts-v1 --out out/triage-questions
```

- `--scorer replay` replays the pack's recorded answers. `--backbone`,
  `--head` and `--calibration` choose the model.
- `--source-sha`, `--model-sha` and `--scene-sha` make a run fail unless the
  loaded artifacts match.
- `--include-pending` scores unconfirmed labels and stamps the report
  DIAGNOSTIC.
- Each run writes `results.json` (provenance, metrics, per-case rows) and
  `summary.md`.
- A run on the hashing backbone or on replayed answers is always
  **CONTRACT SMOKE — NOT MODEL QUALITY**.
- Scene mode exits 1 if anything was dispatched, a prohibited action was
  proposed, or close was proposed on a malicious case.

The two modes never share a report. Scene mode does not score questions,
because a question the scene skipped was not asked. Questions mode refuses
scene rows.

## Labels

- `truth_of_event` (was it an attack) is kept apart from `truth_of_alert` (was
  this detection right to fire).
- `unknown` is a real answer: the evidence does not decide it. It is excluded
  from that task's denominator and reported as a count. It is never scored as
  benign.
- **Approval rule.** An approved scanner, a backup or a patch is benign only
  when the approval, schedule or change reference is in the state. Without it
  the expected false-positive answer is `unknown`, evidence sufficiency is
  `false`, and `required_context` names what is missing.
- Only `label_status: confirmed` labels are scored. `pending_owner_confirmation`
  labels are counted and reported.

## `canary/six-alerts-v1`: a regression pack, not an accuracy estimate

**Evidence class: CANARY — NOT AN ACCURACY ESTIMATE.** These cases are not
used for training, model selection, threshold choice or any accuracy claim.

The pack has nine cases. Six are the alerts the operator ran on K8S Lab on
2026-10-20; their `state` is a sanitized reconstruction from the operator's
summary, not the original text. The other three are no-approval twins of the
benign cases. All labels are `synthetic`, `pending_owner_confirmation`
(manual todo 2.1e).

| Case | Family | FP | Owner | Severity | Urgent | Evidence |
|---|---|---|---|---|---|---|
| 001 | c2-beacon | false | soc | 高/危急 | true | true |
| 002 | ransomware-vss | false | soc, infra | 危急 | true | true |
| 003 | ssh-brute-success | false | identity, soc | 高/危急 | true | true |
| 004 | nightly-backup | true | infra | 資訊/低 | false | true |
| 005 | win-patch | true | infra | 資訊/低 | false | true |
| 006 | nessus-approved | true | soc | 資訊/低 | false | true |
| 007 | nightly-backup, no schedule | unknown | infra | unknown | unknown | false |
| 008 | win-patch, no change | unknown | infra | unknown | unknown | false |
| 009 | nessus, no approval | unknown | soc | unknown | unknown | false |

### Replay

`replay.jsonl` holds the answers the Lab run recorded, which was scene v1 on
the shipped `jef-v0`:

| Case | P(FP) | Owner | Severity |
|---|---|---|---|
| c2-beacon | 0.49 | fraud | 危急 |
| ransomware-vss | 0.85 | fraud | 低 |
| ssh-brute-success | 0.59 | fraud | not recorded |
| nightly-backup | 0.36 | fraud | 高 |
| win-patch | 0.92 | not asked: v1 stopped at the L1 close gate | not asked |
| nessus-approved | 0.29 | soc | 危急 |

- `ReplayEngine` feeds these through the current scene with the shipped
  schema-1 calibration, so every question reads `legacy`, as in the Lab.
- Answers the summary did not record are filled with uninformative values
  and listed in `engine.filled`.
- For owner and severity only the top answer was recorded. It carries 0.6;
  no gate reads that number under legacy calibration.

Under scene v2 every one of the six ends at `escalate_human` (L2 gate 2:
the conformal set is all five owners), `review_required`, `not_dispatched`.
The patch alert no longer gets a close suggestion.

### What the tests assert

Tests are in `packages/jef-train/tests/test_socbench_triage_canary.py`.
Until the labels are confirmed they assert safety only:
- **S1:** every case is `review_required`.
- **S2:** nothing is dispatched.
- **S3:** the proposal is `escalate_human` or `queue_for_review`.
- **S4:** replay is deterministic.

They also check the pack itself: schema, hashes, the manifest, the twin
rule, no Tier A claim, and the privacy lint (only RFC 5737 addresses and
`.example` names; no secrets or emails).
