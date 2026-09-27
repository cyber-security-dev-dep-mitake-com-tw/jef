# n8n-nodes-jef

An [n8n](https://n8n.io) node for JEF — typed decisions with calibrated
confidence. Route, score and judge without generating text to parse.

```bash
npm install n8n-nodes-jef
```

Or install it from **Settings → Community Nodes** as `n8n-nodes-jef`.

## Credentials

One credential, `JEF API`: the base URL of your server, and an optional bearer
token for when JEF sits behind an authenticating proxy. The credential test hits
`/healthz`, so a wrong URL fails when you save it rather than when a workflow
runs at 3am.

## Operations

| Operation | Use it for |
|---|---|
| **Run Scene** | The usual case. Runs a whole playbook and returns its decision trace. |
| **Ask Choice** | One routing or classification decision. |
| **Ask Score** | One rating against levels you define. |
| **Ask Yes/No** | One boolean judgement. |
| **Ask Many** | Several questions in one call. |
| **Health** | Gate the workflow on whether this server can be trusted. |

## Ask everything at once

Every question in a request is evaluated in parallel against the same state, and
the state is read only once. Five questions cost barely more than one, so **Ask
Many** followed by an IF/Switch beats one HTTP round trip per branch:

```json
{
  "false_positive": { "type": "noul", "instructions": "這則告警是否為已知的良性樣態？" },
  "urgent":         { "type": "noul", "instructions": "是否需要立即處理？" },
  "team": {
    "type": "choice",
    "instructions": "應由哪一個團隊處理？",
    "criteria": { "soc": "一般資安監控事件", "appsec": "應用程式漏洞", "infra": "基礎設施" }
  }
}
```

Then branch on `{{ $json.answers.team.choice }}` or
`{{ $json.answers.urgent.noul }}`.

## Run Scene output

```
{{ $json.action }}          // 'assign' | 'escalate_human' | 'close' | ...
{{ $json.human_review }}    // whether a person must see this
{{ $json.state_encodes }}   // 1, for the whole playbook
{{ $json.layers_skipped }}  // never reached — not the same as inconclusive
```

## Before you automate on it

`confidence` is how *peaked* the distribution is, not the chance of being right.
A model can be decisive and wrong. Scene gates read `p_correct` — the calibrated
mapping of confidence onto observed correctness — and refuse to fire when the
server cannot supply it, which routes the case to a human instead.

So check **Health** first. `calibrated: false` means no threshold on that server
means what it looks like, and `test_backbone: true` means the answers are
well-formed and carry no semantics at all.

## License

Apache-2.0
