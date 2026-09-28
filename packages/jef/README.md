# jef

**An open System One decision engine.** Evaluate typed questions against a
shared state and get back probability distributions with *calibrated*
confidence. No text generation, no parsing, no prompt wrangling.

```bash
pip install jef                # engine, scenes, SDK and CLI
pip install "jef[torch]"       # + the real backbone
pip install "jef[server]"      # + the HTTP server
pip install "jef[mcp]"         # + an MCP server for agent clients
```

## Command line

```bash
# one question
jef ask "客戶回報：連續三天付款失敗，已影響營收。錯誤碼 502。" \
    --choice "應由哪個團隊處理？" billing=付款、發票、退款 infra=基礎設施 appsec=應用程式漏洞

# a whole playbook, piped from wherever the alert came from
siem-query --last 5m | jef scene incident-triage - --fail-on-human-review

# lint scenes in CI; needs no model
jef validate scenes/

# is this server worth trusting?
jef models
```

`-` reads stdin, `@file` reads a file, and JSON is sent as structure rather than
flattened text. `--json` gives machine-readable output; exit codes are meant to
be branched on.

## Python

```python
from jef_sdk import Jef, choice, noul, score

jef = Jef("jhu-clsp/mmBERT-base", calibration="models/jef-v0/calibration.json")

result = jef.evaluate(
    alert,
    {
        "urgent": noul("這則訊息是否表達時間緊迫？"),
        "team": choice("應由哪個團隊處理？", soc="監控事件", infra="基礎設施"),
        "severity": score("評估嚴重度", "資訊", "低", "中", "高", "危急"),
    },
)

result.answers["team"].choice  # 'soc' | 'infra'
result.usage.outputTokens  # 0 — nothing is ever generated
```

Every question in a request is evaluated independently against the same state,
which is read **once**. Adding a question costs a short encode, not another pass
over the evidence — at 8,192 tokens of state, ten extra questions add 0.4%.

## Scenes: the layer nobody else ships

A scene is layers of typed questions with confidence gates between them, and the
state is encoded once for the whole playbook.

```yaml
layers:
  - id: L2-routing
    questions:
      owner:
        type: choice
        instructions: 應由哪一個團隊處理？
        criteria: { soc: 一般資安監控事件, infra: 基礎設施、網路、主機層 }
    gates:
      - when: 'owner.p_correct is not None and owner.p_correct >= 0.95'
        then: assign
      - else: true
        then: escalate_human
```

Every run emits a decision trace: the evidence, each question, the answers that
were permitted, where the probability mass fell, and which gate fired and why.

## `confidence` is not correctness

`confidence` is `(n × peak − 1) / (n − 1)` — how *peaked* the distribution is. A
model can be decisive and wrong, which is why most of this ecosystem ships the
disclaimer *"confidence measures concentration, not correctness"*.

JEF fits an isotonic map from confidence onto observed correctness on a held-out
split and exposes `p_correct`, which is what scene gates threshold on. It is
`null` when there is not enough calibration data to answer honestly — and that
is deliberately distinguishable from a low probability, because the two lead to
different decisions. `jef validate` will warn you when a gate automates on
`confidence` where it probably meant `p_correct`.

**An uncalibrated deployment cannot automate anything.** `p_correct` is
unavailable, the conformal set excludes nothing, and every path ends at a human.
That is the design working.

## Results

On **CTI-Bench VSP** — a third-party benchmark of the task JEF trains on, with
every training CVE excluded — attack vector reaches **0.840** against a 0.250
random baseline. Conformal coverage holds at **0.895** against a nominal 0.900
on that third-party data, and collapses to 0.44 on a different task, exactly as
the exchangeability assumption predicts.

On **TMMLU+** it scores 0.254 against a 0.250 baseline: chance. JEF judges
evidence you give it; it does not know things.

Full numbers, including a contaminated run that scored 1.000 and why that was
wrong, are in [`docs/RESULTS.md`](https://github.com/cyber-security-dev-dep-mitake-com-tw/jef/blob/main/docs/RESULTS.md).

## Module reference

`pip install jef` gives you four modules in one distribution:

| Module | What it is |
|---|---|
| [`jef_core`](https://github.com/cyber-security-dev-dep-mitake-com-tw/jef/blob/main/docs/modules/jef_core.md) | shared-state encoding, decision head, calibration |
| [`jef_scene`](https://github.com/cyber-security-dev-dep-mitake-com-tw/jef/blob/main/docs/modules/jef_scene.md) | YAML DAG gates, expression sandbox, decision trace |
| [`jef_sdk`](https://github.com/cyber-security-dev-dep-mitake-com-tw/jef/blob/main/docs/modules/jef_sdk.md) | embedded engine and HTTP client |
| `jef_cli` | the `jef` command |

## Also available

`@jef-ai/sdk` on npm · `n8n-nodes-jef` · Shuffle app · Cortex analyzers ·
a Go server that answers byte-identically · Helm chart · Terraform, Pulumi,
Ansible, Chef and Puppet for Proxmox.

## License

Apache-2.0
