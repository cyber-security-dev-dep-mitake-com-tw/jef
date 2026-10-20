# Incident-triage labelbook

How to label the five `soar_zh` tasks of one alert. It applies to human
reviewers and to the synthetic generator alike, whose rules in
[`synthetic/families-v1.yaml`](synthetic/families-v1.yaml) must follow this
book. Record fields are defined in
[`schema/expectation.schema.json`](schema/expectation.schema.json).

Label from what the alert **shows**, not from what you later learned
happened. If the state does not contain it, it is not evidence. A label
that uses later knowledge leaks into every score built on it.

## 1. Two truths

- **`truth_of_event`**: was the underlying activity an attack (`malicious`),
  legitimate (`benign`), or undecidable from the state (`unknown`)?
- **`truth_of_alert`**: was this detection right to fire?
  - A certificate that really is about to expire is a `true_positive` alert
    about a `benign` event.
  - An approved scan is a `false_positive` alert about a `benign` event.

The false-positive task (§2) follows `truth_of_alert`.

## 2. `false_positive` (`soar_zh.false_positive.v1`, noul)

| Answer | When |
|---|---|
| `true` | The detection fired on known benign or approved activity, **and the approval, schedule, change reference or service role is in the state.** |
| `false` | The activity is, or plausibly is, an attack; or the alert is right to fire even though nothing malicious happened (expiry, capacity). |
| `unknown` | The state cannot decide it. In particular, an approved-looking activity *without* its approval evidence. |

**Approval rule.** An approved scanner, a backup, a patch, a deployment or
an admin script is benign only when the evidence of approval is in the
state. Without that evidence:
- false positive is `unknown` and evidence sufficiency is `false`;
- `required_context` names what is missing (`change_ticket`,
  `scanner_approval`, `backup_schedule`, `maintenance_window`, …).

The same traffic without its ticket is also what an attacker's version looks
like.

## 3. `evidence_sufficiency` (`soar_zh.evidence_sufficiency.v1`, noul)

- **`true`**: the state has the concrete detail a reviewer needs to answer
  §2 and §4: host, account, time, behaviour, and the approval reference
  where §2 depends on one.
- **`false`**: it is a conclusion without verifiable detail, or the
  deciding context is missing (an unknown parent process, no approval
  reference, no asset record when severity hinges on it).

`unknown` here (`insufficient_evidence` in the socbench labelbook §3) is
different from `unknown` in §2: one says the alert lacks detail; the other
says even full detail would not decide the question.

## 4. `owner` (`soar_zh.routing.v1`, choice)

List every team a competent SOC lead would accept. The first entry is the
primary owner.

| Team | Owns | Commonly confused with |
|---|---|---|
| `soc` | Malware, ransomware, C2, endpoint and network intrusion, reconnaissance, phishing payloads | Ransomware on a file server is `soc` (`infra` acceptable), not `infra` alone |
| `appsec` | Web and API attacks, application vulnerabilities, dependency risk, DAST and SAST findings | A WAF-blocked SQL injection is `appsec`; a SQL injection that dropped a web shell is `appsec` + `soc` |
| `infra` | Availability, capacity, certificates, backups, patching, configuration baselines | A patch alert is `infra` even when the EDR raised it |
| `identity` | Employee and service accounts, privileges, SSO, MFA, authentication attacks on servers | SSH brute force on a server is `identity` (`soc` acceptable) |
| `fraud` | **Customer** money and accounts: account takeover of customers, abnormal orders, AML patterns | An *employee* or service account on a trading host is `identity` or `soc`. A host name containing 交易 or "trade" does not make an alert fraud. |

The last row is the confusion behind the Lab's fraud bias. Every routing
family in the synthetic set has a counter-example that uses trading
vocabulary but belongs to another team.

## 5. `severity` (`soar_zh.severity.v1`, score: 資訊 低 中 高 危急)

Severity combines how likely the activity is harmful with how critical the
asset is:

| Likelihood ＼ asset criticality | high | medium or unknown | low |
|---|---|---|---|
| confirmed harmful, ongoing | 危急 | 高 | 中 |
| likely harmful | 高 | 中 | 低 |
| possible | 中 | 低 | 低 |
| approved or benign | 資訊 | 資訊 | 資訊 |

**Modifiers**, each applied once:
- **Blocked** before effect (the proxy or WAF stopped it, no execution):
  one level down.
- **Contained or recovered**: one level down.
- **Unknown criticality** (no asset record): use the "medium or unknown"
  column, and allow the adjacent level.

**Boundary rule.** When a case sits on a boundary, list both levels; a
prediction of either is correct.

Severity is scored on its error distance. A critical→low error is reported
on its own, never averaged away.

## 6. `urgency` (`soar_zh.urgency.v1`, noul)

- **`true`**: damage is happening or imminent. Examples: active encryption,
  exfiltration in progress, an attacker with a session, active C2, an
  account takeover with a pending withdrawal, a disk that fills within hours.
- **`false`**: scheduled handling is enough. Examples: a vulnerability
  without exploitation, a certificate expiring in weeks, a blocked attempt,
  benign activity.
- **`unknown`**: whether anything is happening cannot be told from the state.

## 7. Variants and families

- **Family.** A family is one scenario. Its translation (zh-TW ↔ en, via
  `translation_of`), its hard negative (`hard_negative_of`) and its
  no-approval twin (`twin_of`) stay in the family and in the same split.
  A translation is never an independent case for statistics.
- **Hard negative.** A hard negative looks like its parent and has a
  different answer. Examples: a legitimate heartbeat against a C2 beacon;
  an approved scan against reconnaissance.
- **Translation.** A translation carries the same labels as its source. If
  a translation would change a label, it is a different case.

## 8. Status of a label

| `label_status` | Meaning | Scored by default |
|---|---|---|
| `by_construction` | Synthetic. The label is true because a rule in this book defines the scenario. Tier E. | yes; the report says **SYNTHETIC LABELS** |
| `pending_owner_confirmation` | Written down, not yet confirmed | no (`--include-pending` scores it as DIAGNOSTIC) |
| `confirmed` | Signed off by the named reviewer(s) | yes |
| `disputed` | Reviewers disagree and adjudication is open | no |

**Never counted as human gold.** Synthetic labels are not counted as human
gold, are not mixed into Tier A or Tier B numbers, and are not used to
calibrate a production artifact.
