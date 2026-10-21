# SOCBench sources

`sources.yaml` is the provenance manifest for the datasets SOCBench may draw on. It records, per source, what the data actually labels (`ground_truth_level`), how it can be obtained (`access`), and the caveat that must survive into every derived case. No dataset is stored in this repository; M1 adapters download into an ignored directory and record SHA-256 digests in `manifests/`.

Rules that apply to every source:

1. The ground-truth level is never upgraded. An attack-phase window, a flow label, a CTF answer key or a partial red-team list is recorded as exactly that; an analyst verdict exists only after the two-reviewer workflow in the roadmap, section 7.3.
2. `Background`, unlabelled, and non-red-team traffic is `unknown`, not benign.
3. Licences and redistribution terms are checked before any derived case leaves the ignored data directory. Several sources require registration or an access request.
4. `zh-TW` and `zh-CN` are reported separately, and a translated case is a paired sample of its original, never an independent ground truth.

See [`docs/UPGRADE-PLAN-2026-10.md`](../../../docs/UPGRADE-PLAN-2026-10.md) section 3 for the priority order and [`docs/JEF-SECURITY-ROADMAP-AND-SOC-BENCHMARK.md`](../../../docs/JEF-SECURITY-ROADMAP-AND-SOC-BENCHMARK.md) sections 5 to 7 for the task definitions.

## Registry v2 (2026-10-21)

Every row now also says:

| field | meaning |
|---|---|
| `kind` | `dataset`, `sample-logs`, `rules`, `tool`, `schema`, `feed`, `statistics`, `reference` or `own-telemetry`. Rules, tools and schemas are not labelled data. |
| `label_granularity` | The unit a label attaches to: `event`, `flow`, `window`, `scenario`, `ctf-answer`, `message`, `page`, `url`, `aggregate`, `none`, `unknown`. A window or scenario label never becomes a per-alert label. |
| `trust_tier` | What the source's labels are worth: C source-labelled, D weak or derived (detector actions, attack windows, feeds), E synthetic or by construction, U unlabelled, `none` for non-data. Tiers A and B exist only after human review and the schema rejects them here. |
| `five_task_gold` | Always `false`. No source supplies gold for the five incident-triage tasks; that comes from the reviewer workflow. |
| `adapter` | `implemented`: an entry in `jef_train.socbench.adapters.ADAPTERS` and a pin manifest; `fetch`, `check` and `pin` work. `catalogue`: a planning row, refused by `fetch`. |
| `access_verified` | The date a person confirmed url, access and licence, or `false` when the row only repeats a planning document. |

`jef-bench sources` lists the wired rows; `jef-bench sources --catalogue` lists everything with its tier and label granularity.

Rows with `kind: own-telemetry` (your own Cloudflare zone, origin logs, change records, lab sensors) are private. Raw records never enter this repository, a GitHub artifact or an external model. A WAF `block` or `challenge` is a detector output (Tier D) and an `allow` is unlabelled (U), never benign.
