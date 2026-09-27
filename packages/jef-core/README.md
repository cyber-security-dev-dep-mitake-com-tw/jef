# jef-core

The engine behind [JEF](https://github.com/cyber-security-dev-dep-mitake-com-tw/jef),
an open System One decision engine. Evaluate typed questions against a shared
state and get back probability distributions with calibrated confidence. No text
generation, no parsing.

```bash
pip install jef-core          # engine only, no model
pip install jef-core[torch]   # + the real backbone
```

Most users want [`jef`](https://pypi.org/project/jef/) instead, which bundles
this with the scene layer, the SDK and a CLI.

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
            "criteria": {"billing": "付款、發票、退款", "infra": "基礎設施"},
        },
        "severity": {
            "type": "score",
            "instructions": "評估嚴重度",
            "criteria": ["資訊", "低", "中", "高", "危急"],
        },
    },
)

result.answers["team"].choice  # 'billing'
result.usage.outputTokens  # 0 — nothing is ever generated
engine.encode_count  # 1 — the state is read once, for all questions
```

## The mechanism

The state is encoded **once** by a frozen backbone, and every question attends
over that same encoding independently. Questions cannot contaminate each other,
and one more question costs a short query encode plus a small matmul rather than
another pass over the state. `encode_count` and `query_encode_count` are exposed
so the property is observable rather than assumed.

All three primitives reduce to "score these N options", which is what lets a
single head serve them: `choice` scores its options, `score` scores its ordered
levels and takes the expectation so results can land between them, and `noul`
scores `(false, true)` so index 1 is always P(true).

## Calibration is not optional here

`confidence` is `(n × peak − 1) / (n − 1)`: how *peaked* the distribution is, not
how likely the answer is to be correct. That distinction is why most of this
ecosystem ships the disclaimer *"confidence measures concentration, not
correctness"*.

`jef_core.calibration` fits three things on a held-out split: temperature
scaling, split-conformal quantiles, and an isotonic map from confidence onto
observed correctness. That last one is what makes `p_correct` meaningful, and it
refuses to fit below 150 calibration points rather than produce a confident
curve from memorised noise.

## License

Apache-2.0
