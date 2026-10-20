# `jef_core` — engine

Shipped inside the `jef` distribution (`pip install jef`); it is not a
separate package on PyPI.

The engine behind [JEF](https://github.com/cyber-security-dev-dep-mitake-com-tw/jef),
an open System One decision engine. Evaluate typed questions against a shared
state and get back probability distributions with calibrated confidence. No text
generation, no parsing.

```bash
pip install jef              # this module and the other three
pip install 'jef[torch]'     # + the real backbone
```

Most users want [`jef`](https://pypi.org/project/jef/) instead, which bundles
this with the scene layer, the SDK and a CLI.

```python
from jef_core import Engine

engine = Engine("jhu-clsp/mmBERT-base")

result = engine.evaluate(
    state="EDR 於 02:14 偵測主機 fs-core-02 執行 vssadmin delete shadows /all，五分鐘內 3,200 個檔案被改為 .lockbit。",
    questions={
        "urgent": {"type": "noul", "instructions": "這則告警是否需要立即處理？"},
        "team": {
            "type": "choice",
            "instructions": "應由哪個團隊處理？",
            "criteria": {"soc": "資安監控與惡意程式", "infra": "主機、備份與可用性"},
        },
        "severity": {
            "type": "score",
            "instructions": "評估嚴重度",
            "criteria": ["資訊", "低", "中", "高", "危急"],
        },
    },
)

result.answers["team"].choice  # 'soc' or 'infra'
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


## Calibration status and task identity

Calibration is looked up by **task**, not by answer shape. `bucket_key(kind,
n_options, task_id)` spells it `cve.user_interaction.v1|noul:2`; a question
that declares `task_id` is answered only by a bucket fitted for that task, and
a question that declares none is answered only by a shape-scoped bucket. The
two never alias (POL-002): "CVE user interaction" and "phishing true positive"
are both `noul:2` and share nothing that matters to calibration.

`Calibrator.status(kind, n_options, alpha, task_id)` reports one bucket's
state:

| status | meaning | unlocks automation |
|---|---|---|
| `validated` | correctness map and a conformal quantile at `alpha`, fitted on enough data, for this task | yes |
| `insufficient` | the bucket exists but one of those is missing | no |
| `missing` | no bucket for this task and shape | no |
| `legacy` | the artifact is schema 1 (shape keys, no provenance); numbers are shown for analysis only | no |
| `incompatible` | the artifact's `model_sha256` differs from the head that is running; `p_correct` is `None` and prediction sets are full | no |

Artifacts are **schema 2**: `calibration_schema: 2`, an `artifact` block
(`model_sha256`, `label_schema_version`, `language`, `dataset_digest`,
`calibration_window`, `split_policy`, `sample_count`,
`min_validated_samples`), and task-scoped bucket keys. `jef-train` writes it
with the head's digest and the corpus manifest's digest; loaders bind the
running head's digest with `bind_model()` so a mismatch is seen.

`models/jef-v0/calibration.json` is still schema 1 and loads as `legacy`:
every question reports `calibration_status: legacy` and no scene gate
automates until the artifact is refitted (task-scoped buckets come out of
`jef-train` automatically now) or migrated on purpose:

```bash
uv run jef calibration show models/jef-v0/calibration.json
uv run jef calibration migrate old.json new.json --task-id cve.severity.v1 --model-sha256 sha256:...
```

`migrate` relabels every bucket to one task and refuses an artifact that
already carries tasks. It is a human asserting what the file was fitted for;
nothing in the loader does it silently.
