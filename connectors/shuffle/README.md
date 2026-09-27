# Shuffle app for JEF

```bash
cp -r connectors/shuffle/jef /path/to/shuffle-apps/
# then reload apps from the Shuffle admin UI
```

Configure `url` (your JEF server) once as app authentication.

## Actions

| Action | Use it for |
|---|---|
| `run_scene` | The usual case: run a whole playbook and branch on `$jef.action`. |
| `ask_choice` | One routing or classification decision. |
| `ask_score` | One severity or impact rating, on levels you define. |
| `ask_yes_no` | One boolean judgement. |
| `ask_many` | Several questions in one call. |
| `health` | Gate the workflow on whether this server can be trusted. |

## Ask everything at once

JEF evaluates every question in a request in parallel against the same state,
and the state is read only once. Five questions cost barely more than one, so
`ask_many` and then branching beats one round trip per branch:

```json
{
  "false_positive": {"type": "noul", "instructions": "這則告警是否為已知的良性樣態？"},
  "urgent":         {"type": "noul", "instructions": "是否需要立即處理？"},
  "team": {
    "type": "choice",
    "instructions": "應由哪一個團隊處理？",
    "criteria": {"soc": "一般資安監控事件", "appsec": "應用程式漏洞", "infra": "基礎設施"}
  },
  "severity": {
    "type": "score",
    "instructions": "評估嚴重程度",
    "criteria": ["資訊", "低", "中", "高", "危急"]
  }
}
```

Then branch on `$jef.answers.team.choice`, `$jef.answers.severity.score`, or
`$jef.answers.urgent.noul`.

## Before you automate on it

Check `health` first. `calibrated: false` means no threshold on this server
means what you think it means, and `test_backbone: true` means the answers have
no semantics whatsoever — they are well-formed and meaningless.

`confidence` is how peaked the distribution is, not the chance of being right.
For the latter, use `run_scene`: scene gates read `p_correct`, the calibrated
mapping of confidence onto observed correctness, and refuse to fire when the
server cannot supply it.
