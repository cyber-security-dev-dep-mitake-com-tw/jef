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
LABELBOOK.md                     how the five tasks are labelled (humans and generator alike)
schema/case.schema.json          inputs only: state, evidence, provenance, state sha256
schema/expectation.schema.json   what a reviewer expects; scored blind
canary/six-alerts-v1/            the K8S Lab canaries
  cases.jsonl  expectations.jsonl  replay.jsonl  manifest.json
synthetic/families-v1.yaml       synthetic scenarios and their labels, readable by a reviewer
synthetic/dev-v1/                rendered: 78 cases, 25 families, zh-TW/en pairs, hard negatives
  cases.jsonl  expectations.jsonl  manifest.json  lock.json
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
- Scene mode exits 1 if:
  - anything was dispatched;
  - a row resting on unvalidated calibration is not `review_required` (S1);
  - such a row proposes something outside the scene's advisory allowlist (S3);
  - a prohibited action was proposed;
  - a state-changing action (tier ≥ 2, recognised by tier, not by name) was
    proposed on a malicious case.
- Replay counts the cases it has no record for, under
  `excluded_cases.no_replay_record`. Its provenance names the replay source
  as the model.

The two modes never share a report. Scene mode does not score questions,
because a question the scene skipped was not asked. Questions mode refuses
scene rows.

### Another system on the same questions

`--scorer remote --scorer-config scorers.json --scorer-name NAME` asks a
`/v1/systemone` service or the AI Gateway decision-model dialect (Jev, Laya,
an LLM wrapper, another JEF) the scene's five questions: same state, same
instructions, same options in the same order. The config file is the one
[`benchmark/socbench/scorers.example.json`](../socbench/scorers.example.json)
describes, and credentials come from the environment variable named in
`auth_env`.

- A reply must be a full distribution over each question's options. A
  refusal, a missing question or a partial distribution is a contract
  failure. The question is excluded and counted under
  `remote_contract_failures`, never filled in.
- A remote system has no JEF calibration, so every question reports
  `missing`.
- An external host needs `--allow-egress HOST`. Packs with `own-telemetry` or
  `reviewer-sourced` cases are never sent to an external host; that needs a
  separate data-processing approval this tool does not take.
- A system you cannot reach is `NOT_RUN` in any comparison. A stand-in such as
  `research/jev-local` is never reported as the official Jev.

## `synthetic/dev-v1`: label-space coverage, not gold

**Evidence class: SYNTHETIC LABELS — ENGINEERING CONTRACT (Tier E).**
Every label is `by_construction`: it is true because a
[LABELBOOK](LABELBOOK.md) rule defines the scenario written in
[`families-v1.yaml`](synthetic/families-v1.yaml). The set covers:

- EDR, NDR and identity scenarios;
- routing positives for all five owners, with counter-examples that use
  trading vocabulary but are not fraud;
- every severity level;
- the approval rule, through no-approval twins;
- 26 hard negatives;
- a zh-TW/en pair for every case.

**Rebuild:**

```bash
uv run jef-bench triage synth --families benchmark/incident-triage/synthetic/families-v1.yaml \
  --out benchmark/incident-triage/synthetic/dev-v1
uv run jef-bench triage lock --pack benchmark/incident-triage/synthetic/dev-v1 \
  --out benchmark/incident-triage/synthetic/dev-v1/lock.json      # or --verify lock.json
```

**What the lock enforces.** It splits by family: a family's translations,
hard negatives and twins share a split. It refuses to lock if:

- a family or pair crosses splits;
- two states are identical;
- two cases of different families are near-duplicates (character 3-gram
  Jaccard ≥ 0.80, with slot values masked);
- any case is a near copy (≥ 0.60) of a template in
  `jef_train/corpus/soar_zh.py`, the training corpus.

The current maximum against training is 0.316.

**What these labels may be used for:**

- unit tests and baselines;
- the PR-06 pipeline dry run, which is expected to come out `insufficient`.

**What they may not be used for:**

- accuracy claims;
- Tier A or Tier B numbers;
- calibrating a production artifact.

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

## A task-scoped candidate: pipeline first, data later

```bash
uv run jef-bench triage candidate-export --pack PACK --lock PACK/lock.json --out out/corpus  # Tier B+ only
uv run python -m jef_train.fit --corpus out/corpus --out out/candidate   # never models/jef-v0
uv run jef-bench triage candidate-check --candidate out/candidate --corpus out/corpus \
  --backbone jhu-clsp/mmBERT-base --out out/candidate/check.json
```

**Export.** `candidate-export` writes one sample per case and task, with the
questions taken from the scene. Each sample is task-scoped
(`soar_zh.routing.v1|choice:5`) and split by the lock's families.
- It refuses unlocked inputs.
- It refuses labels below Tier B. Synthetic labels (Tier E) export only with
  `--dry-run`, and the manifest then says `deployable: false`.

**Check.** `candidate-check` reports `insufficient` unless all of the
following hold:
- every bucket the scene asks is `validated` (at least 150 calibration
  records, a correctness map and a quantile at the scene's alpha);
- the calibration was fitted for this head and on this corpus;
- the splits share no family;
- conformal coverage `holds` on the test split;
- it is not a dry run.

Its ceiling is `research`. Promotion needs Tier A labels and a decision in
the [validation ledger](../../docs/validation/README.md), never a flag. The
shipped `models/jef-v0` is refused as a candidate.

**Today.** The synthetic dry run ends at `insufficient`, as it should:
- 10 calibration records per bucket against the 150 required;
- coverage `too_few`.

The 63-case SOCBench pilot does not change this. Its tasks are
`soc.ndr.technique` and `soc.alert.verdict`, not these five.

## In CI

- **Every PR:** the `triage-canary` job in `ci.yml` runs on CPU without torch.
  - It runs the guard and harness tests and scene mode on the canaries
    (replayed Lab answers under the shipped legacy calibration, and the
    hashing backbone).
  - It runs questions mode as a diagnostic, and verifies the synthetic lock.
  - Artifact: `triage-canary-<sha>-<run>`. All of it is CONTRACT SMOKE.
- **Weekly or on demand:** `triage-quality.yml` runs the real backbone
  against the pinned head and calibration.
  - It fails only on the pre-registered safety checks: **P1** canary S1–S3 or
    a dispatch; **P2** a candidate dry run that is not `insufficient`;
    **P3** a SHA mismatch.
  - Quality is reported, and a change above 0.10 is flagged for a person.
    Quality is never gated until a human-labelled set exists.
  - Artifact: `triage-quality-<sha>-<run>`, never a push to `main`.

## Runtime evidence on the K8S lab (operator-run)

[`scripts/triage-runtime-evidence.py`](../../scripts/triage-runtime-evidence.py)
drives `/v1/scenes/incident-triage:evaluate` on a live deployment. See manual
todo 2.1f. Its requests carry no `case_id`, so the server seals no proposal
and nothing can be authorised. It records, kept apart:

- **Latency:** cold first request, then warm latency at each concurrency
  level. A percentile is reported only with enough samples behind it (20 for
  p95, 100 for p99).
- **Safety of every response.**
- **Delivery:** measured only against a mock ledger's call count, which must
  not move. `review_required` is not a ticket.
- **Platform:** with kubectl: image digests, restarts during the run,
  ServiceAccount, NetworkPolicy names and `kubectl top`.

Readiness is redacted. The API key is read from the environment and never
written.
