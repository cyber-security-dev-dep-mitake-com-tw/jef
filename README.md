# JEF

**An open System One decision engine.** Evaluate typed questions against a shared
state and get back probability distributions with *calibrated* confidence. No text
generation, no parsing, no prompt wrangling.

```python
from jef_core import Engine

engine = Engine("jhu-clsp/mmBERT-base")

result = engine.evaluate(
    state="客戶回報：連續三天付款失敗，已影響營收。錯誤碼 502。",
    questions={
        "urgent": {"type": "noul", "instructions": "這則訊息是否表達時間緊迫？"},
        "team": {
            "type": "choice",
            "instructions": "應由哪個團隊處理？",
            "criteria": {
                "billing": "付款、發票、退款",
                "infra": "基礎設施、網路、主機層",
                "appsec": "應用程式漏洞與程式碼相關",
            },
        },
        "severity": {
            "type": "score",
            "instructions": "評估此事件的嚴重度",
            "criteria": ["資訊", "低", "中", "高", "危急"],
        },
    },
)

result.answers["team"].choice        # 'billing'
result.answers["team"].confidence    # 0.91
result.usage.outputTokens            # 0 -- nothing is ever generated
```

## Why another one?

There are already 65+ open reimplementations of TypeSafe's Jev. Almost all of
them stop at the same two places, and JEF exists because of that:

**1. They disclaim calibration.** The line *"confidence measures concentration,
not correctness"* appears in project after project. It is honest, and it makes
the number useless: a threshold built on an uncalibrated distribution is a coin
flip with extra steps. JEF fits temperature scaling **and** split-conformal
prediction per question type, and publishes ECE, Brier and coverage on
**independent third-party benchmarks** -- not on its own.

**2. They stop at the primitive.** Jev deliberately leaves decision composition
to the caller (*"Your application defines the possible answers and uses the
results in its logic"*), so every open project does too. But a real SOAR
playbook is layers of gated decisions, and today that is hand-written if/else in
Splunk SOAR, Shuffle or TheHive. JEF ships that layer.

And one thing nobody does at all: **zh-TW is a first-class language here**, not
an afterthought behind an English-only encoder.

## Scene gates

A scene is layers of typed questions with confidence thresholds between them.
The state is encoded **once** for the entire scene, so a twenty-gate playbook
costs one state read, not twenty.

```yaml
scene: incident-triage
layers:
  - id: L1-noise-filter
    questions:
      is_false_positive:
        type: noul
        instructions: 這則告警是否為已知誤報樣態？
    gates:
      - when: "is_false_positive.probability > 0.85"
        then: close
      - else: next

  - id: L2-routing
    questions:
      owner:
        type: choice
        instructions: 應由哪個團隊處理？
        criteria:
          soc: 一般資安監控事件
          appsec: 應用程式漏洞與程式碼相關
          infra: 基礎設施、網路、主機層
    gates:
      - when: "owner.confidence >= 0.9"
        then: assign
      - else: escalate_human          # honest uncertainty, routed to a person
```

Every run emits a **decision trace** -- evidence, question, allowed answers,
probabilities, confidence, which gate fired and why -- which is what an audit
actually needs.

## Compatibility

JEF serves `POST /v1/systemone`, so existing clients work unchanged. Both
dialects of the yes/no primitive are accepted and echoed back: TypeSafe calls it
`noul`, Vercel's AI SDK calls it `boolean`.

## Status

Under active development. See `docs/` for design notes.

## License

Apache-2.0
