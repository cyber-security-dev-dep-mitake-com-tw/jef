# SOCBench labelbook

How two analysts label a SOCBench case so that the result can be called **Tier A gold**. Read this once before the pilot and keep it open while labelling. Every rule here is either enforced by `jef-bench` or is the reviewer's responsibility, and each says which.

## 1. What you are and are not doing

You decide what the evidence in one case supports. You are **not** grading a model, guessing what a weak label said, or deciding what action to take on a live system.

- You see the case's `state`, its `evidence` items, the `allowed_labels`, and the `language`. You do **not** see model output, the source's weak label, the other reviewer's answer, or which split the case will land in. `jef-bench annotate export` produces exactly that view; do not look anything up in the source dataset by `case_id`.
- Label from the evidence in the case. If the decisive fact is not in the case, the honest answer is `insufficient_evidence` or `unknown` (see §3), not your best guess.
- Work alone. Discussing a case with the other reviewer before both labels are in turns two reviews into one and the gold into Tier B.

## 2. The record you fill in

`annotate export` writes one task per case. Fill these fields and leave the rest untouched:

| Field | Required | Rule (enforced = checked by `annotate agree`) |
|---|---|---|
| `verdict` | yes | One of `allowed_labels` for the case's task (enforced). |
| `confidence` | yes | `high`, `medium` or `low` (enforced). How sure **you** are, not how bad the event is. |
| `evidence_ids` | when the verdict rests on evidence | The `e<n>` ids that decide it. Must exist in the case (enforced). Leave empty only for `unknown` / `insufficient_evidence`. |
| `prohibited_actions` | when any apply | Actions that would be wrong on this case even if a model proposed them (vocabulary in §5). |
| `required_followup` | for `insufficient_evidence`, `suspicious` | What would decide it: `fetch_dns_logs`, `confirm_change_ticket`, `check_sender_auth`, ... |
| `note` | for any `low` confidence or non-obvious call | One or two sentences a third reviewer can check. |
| `seconds_spent` | please | Wall-clock seconds on this case. Optional but the pilot uses it to schedule the rest (validated: non-negative number). |

## 3. `unknown` versus `insufficient_evidence`

Both are legitimate answers. They are not "missing data" and they are never counted as negatives.

- **`insufficient_evidence`**: a decisive field is absent from *this case*, and you can name it. A process alert with no parent process; an e-mail with no authentication headers. Fill `required_followup` with what would decide it.
- **`unknown`**: the evidence that would decide it does not exist in the source, or you cannot decide even with every field present. CTU-13 `Background` conversations are the textbook case: the capture never labelled them.

Tasks that only offer `unknown` (`soc.ndr.technique.v1`, `soc.edr.execution.v1`, `soc.identity.abuse.v1`) use it for both situations; say which in `note`.

Never resolve doubt toward benign. A benign verdict can later close an alert automatically; an `unknown` cannot.

## 4. Labels, per task

The label sets are fixed in code (`annotate.TASK_LABELS`); the definitions below are how to apply them.

### `soc.ndr.technique.v1` (start here: the pilot task)

| Label | Use when the evidence shows | Not when |
|---|---|---|
| `c2` | A command-and-control channel: periodic check-ins to an external host, beacon timing, known C2 infrastructure, tasking/response pattern. | Merely talking to a rare destination. |
| `botnet` | Other botnet traffic: spam relay, scanning, click fraud, DDoS participation. | A single port scan with no other signal (`unknown` or `benign` with a scanner ticket). |
| `dns_tunnel` | Data or control carried in DNS: long, high-entropy labels, TXT volume far above baseline to one domain. | High TXT volume to DMARC/DKIM/SPF or service-discovery names. |
| `normal_automation` | A legitimate scheduled or automated job: backup, monitoring heartbeat, approved scanner, update service. Evidence names the job. | You assume it is automation because it is periodic. |
| `benign` | Ordinary legitimate traffic with nothing pointing elsewhere. | You simply saw nothing malicious in a capture that was never labelled. |
| `unknown` | See §3. Includes all unlabelled background. | |

### `soc.alert.verdict.v1`

| Label | Use when |
|---|---|
| `confirmed_malicious` | The evidence itself demonstrates attacker activity (execution, C2, credential use by an attacker). |
| `suspicious` | Indicators point toward an attack but do not demonstrate it; an analyst should investigate. Fill `required_followup`. |
| `confirmed_benign` | The evidence shows a legitimate cause: approved change, known admin tool with ticket, update service. A high-severity rule firing is **not** evidence of malice and a low one is not evidence of innocence. |
| `insufficient_evidence` / `unknown` | See §3. An AIT-ADS alert outside an attack-phase window is not thereby benign. |

### `soc.phishing.verdict.v1`

`phishing` (credential or payment lure, impersonation, malicious link or attachment), `suspicious`, `spam_other` (unwanted but not phishing), `legitimate` (sender authenticated and request consistent with a known process), `insufficient_evidence`, `unknown`. Urgent wording alone is not phishing: real IT and finance notices are urgent too. Those pairs are exactly the hard negatives the set is built around.

### `soc.edr.execution.v1`, `soc.identity.abuse.v1`

`malicious` / `benign` / `unknown`; `compromised` / `suspicious` / `legitimate` / `unknown`. A signed admin tool deployed by a change ticket is `benign` even when its command line looks like an attack; an unsigned binary with no parent context is not thereby `malicious`.

### `soc.response.authority.v1`

`allowed`, `review_required`, `prohibited`. This task labels **policy**, not threat: given the case, the asset and the change calendar, may the named action proceed. Label it from the written policy, never from how likely the threat is. It is reported separately and never folded into detection accuracy.

## 5. `prohibited_actions` vocabulary

Use these names so they aggregate: `close_alert`, `suppress_rule`, `isolate_host`, `disable_user`, `revoke_tokens`, `block_ip`, `block_domain`, `quarantine_email`, `delete_email`. Add a new name only if none fits, and say so in `note`. A benign verdict on a production server often still carries `isolate_host`: being wrong there costs more than being slow.

## 6. Agreement and adjudication (enforced by the tools)

1. Both reviewers label every case. `annotate agree` computes percent agreement and Cohen's kappa per task and per class, queues every disagreement, and reports measured throughput from `seconds_spent`.
2. **Agreed** verdicts become `double-human-reviewed` with the lower of the two confidences, and with only the evidence ids both cited.
3. **Disagreements** go to a third reviewer who did not review the case (enforced). The adjudicator records a verdict from the allowed labels **and a written reason** (both enforced). The losing verdicts are kept as `allowed_alternatives`.
4. A disagreement with no adjudication becomes **`disputed`** with verdict `unknown` and low confidence. That is the correct outcome, not a failure; never settle it by majority vote or by asking the model.
5. A case with one review is `single-human-reviewed` (Tier B) and never counts toward the Tier A total (enforced: only `double-human-reviewed` is Tier A).

Kappa is unstable when one class dominates. Always read it next to the per-class counts the report prints; a kappa of 0.4 on 290 benign and 10 c2 says little about the c2 calls.

## 7. Locking, and what changes after

- `jef-bench lock` assigns whole families to splits and records SHA-256 digests of the exact cases and gold. `lock --verify` fails on any byte change.
- After a lock, the **test** split is never relabelled in place. A labelling error found later goes into a new dataset version with a new lock; the old one stays as it was, so a published number can always be reproduced.
- Model and tuning people do not see test-split gold. Only the evaluation owner runs `jef-bench evaluate` against it.

## 8. The pilot (first 50–65 cases)

Batches live in git as labels only: see [`batches/`](batches/README.md). The prepared pilot is [`batches/pilot-2026-10-v1`](batches/pilot-2026-10-v1/README.md) (63 cases from CTU-13 and AIT-ADS); its README has every command. A pull request that adds labels runs the `annotation batch` check, which refuses case text in the repository, foreign case ids, invalid verdicts, and non-independent reviewer commits.

### With one reviewer (the usual start)

A single security engineer can run the pilot. The result is **Tier B** (`single-human-reviewed`) and is reported as that, never as agreement.

1. `jef-bench sample --task soc.ndr.technique.v1 --per-family 5 --seed pilot-01 ...`: at most 65 cases (13 CTU-13 scenarios × 5); read the real count from the manifest.
2. `jef-bench annotate export --annotator analyst-anon-01 ...` and fill every task, including `note` and `seconds_spent`.
3. `jef-bench annotate pilot-report --cases ... --labels review-a.jsonl --weak-gold gold.jsonl --hourly-rate <rate> --out pilot.json` reports completeness field by field, the undetermined share, median and p90 seconds, the share of cases traceable to a pinned source artifact, a **source-label check** (your verdict against the dataset's own label, explicitly not inter-annotator agreement, no kappa), and reviewer cost kept apart from inference cost.
4. `jef-bench annotate adjudicate --labels review-a.jsonl --out gold.jsonl` with the one file writes `single-human-reviewed` gold.

Do not run a second "reviewer" yourself, with a second handle, or with an LLM. That is the one shortcut that turns the whole set into a false Tier A.

### With two independent reviewers

1. Export the same sample to both, blind to each other, the weak labels and any model output.
2. Both label independently, filling `seconds_spent`.
3. `jef-bench annotate agree`: read kappa per class, the disagreement queue, and `timing.reviewer_hours_per_100_cases_double_reviewed`.
4. Meet once with the adjudicator: for each disagreement, decide whether the **labelbook** was unclear (fix this file) or a reviewer erred (adjudicate). Do not relabel agreed cases.
5. Schedule the rest of the gold set from the measured hours, not from an estimate. Sample the next batch with `--exclude sampled-pilot.jsonl`.
6. For the 300+ case Tier A set, open a **Gold annotation batch** issue per batch, create it with `jef-bench annotate new-batch --purpose gold --reviewer <A> --reviewer <B> --adjudicator <C>`, and have each person commit only their own file. `check-batch` reports `tier_if_merged: A` only when both reviewers are complete, every disagreement has the adjudicator's decision, and git shows three separate authors.

## 9. Things that are never Tier A

An AIT-ADS attack-phase window (Tier D), a CTU-13 flow label (Tier C), an LLM-authored seed (Tier E), a single review (Tier B), and anything a script wrote into `label_origin`. `export-corpus --min-tier A` refuses to train on them, and the report scores each tier separately.
