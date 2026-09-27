# jef-sdk

Python client for [JEF](https://github.com/cyber-security-dev-dep-mitake-com-tw/jef).
Embedded and remote share one shape, so moving a workload between them is a
change of constructor rather than a change of code.

```bash
pip install jef-sdk
```

```python
from jef_sdk import Jef, JefClient, choice, noul, score

jef = Jef("jhu-clsp/mmBERT-base", calibration="models/jef-v0/calibration.json")
# ...or, identically:
jef = JefClient("http://jef.internal:8080")

result = jef.evaluate(
    alert,
    {
        "urgent": noul("這則訊息是否表達時間緊迫？"),
        "team": choice("應由哪個團隊處理？", soc="監控事件", infra="基礎設施"),
        "severity": score("評估嚴重度", "低", "中", "高"),
    },
)

trace = jef.run_scene("incident-triage", alert)
```

For a SOAR worker that already has the alert in memory, the HTTP hop is pure
overhead — the same engine object serves both, so embedding skips a
serialisation round trip per decision without changing an answer.

## Why builders rather than dict literals

`choice()` fixes option order to keyword order, because that order is the index
each option occupies in the returned distribution. `score()` takes levels
positionally, lowest first, because the answer is the expectation over that
ordering — reversing it inverts the scale silently instead of erroring.
`noul()` and `boolean()` are the same primitive in the two spellings, so code
ported from `experimental_evaluate` reads unchanged.

## Errors carry the server's code

`JefHTTPError` keeps the server's own error code, so you can branch on
`invalid_question` versus `too_many_questions` without parsing prose.

## License

Apache-2.0
