---
name: jef
description: >-
  Use when a decision has known options and you need it to be cheap, consistent
  and measurably reliable rather than eloquent: routing a ticket or alert to a
  team, rating severity or priority against defined levels, judging whether a
  claim holds against evidence in front of you, or running a triage playbook.
  Triggers on "which team", "how severe", "is this a false positive",
  "prioritise these", "classify", "triage", "route", and on any question you are
  about to answer by reasoning over options someone already listed.
---

# JEF — typed decisions

JEF answers bounded questions about evidence and returns a probability
distribution over the options *you* name. It generates no text. It cannot answer
a question whose options you have not listed, and it does not know anything that
is not in the evidence you pass it.

Reach for it when you would otherwise reason your way to a label. Your reasoning
will be fluent and unrepeatable; JEF's answer is a distribution you can threshold
and audit, and it costs a fraction as much.

## When to use it, and when not to

| Use JEF | Do not use JEF |
|---|---|
| "Which of these five teams owns this alert?" | "What does CVE-2024-3400 affect?" (knowledge) |
| "Rate this incident: info / low / medium / high / critical" | "Write the incident summary" (generation) |
| "Does this transcript show a refund was issued?" | "What should our refund policy be?" (judgement without evidence) |
| "Is this alert a known-benign pattern?" | "Find the root cause" (investigation) |

The dividing line is whether the answer is one of a set you can write down, and
whether the evidence to decide it is in the text you have.

## The three question types

**choice** — pick one of the options you supply. Write real descriptions; they
are what the model reads, and `soc: "monitoring alerts, malicious connections,
endpoint detections"` decides far better than `soc: "SOC"`.

**score** — rate against ordered levels, **lowest first**. The answer is the
expectation over your ordering, so it can land between levels (2.7 on a
five-level scale) and reversing the list inverts the scale silently.

**noul** (also spelled `boolean`) — the probability a statement holds. Near 0.5
means the evidence does not say, which is different from "no".

## Ask everything at once

Every question in one call is evaluated in parallel against the same evidence,
and the evidence is read once. Ten questions cost barely more than one.

So do not call back three times. Ask the speculative questions up front and
branch on what returns — `jef_ask_many` exists for exactly this.

## Reading the answer, which is where people go wrong

```json
{"type": "choice", "choice": "soc",
 "probabilities": {"soc": 0.71, "infra": 0.22, "appsec": 0.07},
 "confidence": 0.57}
```

`confidence` is **how peaked the distribution is**: `(n × peak − 1) / (n − 1)`.
It is not the chance of being right. A model can be decisive and wrong, and
`confidence: 0.94` says only that it was decisive.

`p_correct` is the calibrated probability the answer is correct, fitted on
held-out data. **Threshold on this one.**

`p_correct: null` means *unknown*, not *low*. The server had too little
calibration data for that question shape to answer honestly. Treat it as "I
cannot tell you how reliable this is" — which is a reason to involve a person,
not a reason to assume the answer is bad.

`prediction_set` is a conformal set: the options that cannot be excluded at the
promised coverage. More than one entry means the model genuinely cannot separate
them; that is information, not failure.

## Before you act on any of it

Call `jef_health` first, or read the `caveats` field that every tool returns.

- `calibrated: false` — `p_correct` is unavailable everywhere and nothing should
  be automated on these numbers.
- `test_backbone: true` — the answers are well-formed and carry **no meaning**.
  Do not act on them at all.

Both appear in `caveats` when they apply. If you see a caveat, say so in your
answer rather than passing the number along as if it were sound.

## Scenes

A scene is a playbook: layers of typed questions with calibrated gates between
them, encoded once for the whole run. `jef_run_scene` returns the verdict plus a
full decision trace — the evidence, every question, the permitted answers, where
the mass fell, and which gate fired and why.

Check `human_review`. When it is true the scene declined to decide, usually
because `p_correct` was unavailable or a conformal set was too wide. Report that
as the outcome; do not override it with your own opinion of the evidence.

## Worked example

> Triage this: "CrowdStrike detected mass file encryption on db-core-02 at
> 02:14, extensions renamed to .lockbit, shadow copies deleted."

```
jef_ask_many(
  state = "<the alert>",
  questions = {
    "false_positive": {"type": "noul",
      "instructions": "Is this a known-benign pattern?"},
    "team": {"type": "choice", "instructions": "Which team owns this?",
      "criteria": {"soc": "monitoring, malicious connections, endpoint detection",
                   "infra": "hosts, networking, certificates, availability",
                   "appsec": "application vulnerabilities and code"}},
    "severity": {"type": "score", "instructions": "How severe is this?",
      "criteria": ["info", "low", "medium", "high", "critical"]},
    "urgent": {"type": "noul", "instructions": "Does this need immediate action?"}
  }
)
```

One call, four answers, one read of the evidence. Then report the distribution
and the reliability — not just the labels — and say plainly if `p_correct` was
unavailable or a caveat was returned.
