# SOC Specialist v5: data registry, Cloudflare pipeline, reviewer queue

**Status:** plan, checked against `main@9e03ab8` on 2026-10-21. Nothing here is a result.
**Source:** `JEF_SOC_Specialist_Dataset_Cloudflare_Training_Pipeline_v5_2026-10-20.md` (not in the repository).

## What v5 adds to the incident-triage work (#48–#59)

#48–#59 made incident-triage safe (no unvalidated close recommendation) and measurable (canaries, harness, baselines, candidate pipeline). v5 is about the missing input: traceable, tiered ground truth for the five `soar_zh` tasks. The code-only part is:

| PR | What | Needs a human or a permission for |
|---|---|---|
| P0-1 `data/socbench-source-registry-v2` | `sources.yaml` v2: trust tier, label granularity, `five_task_gold: false`, wired vs catalogue rows | checking each catalogue row's url, access and licence |
| P0-2 `feat/cloudflare-waf-ingest` | read-only GraphQL reader, mock-tested | a zone, an Analytics-read token, data-owner approval |
| P0-3 `feat/cloudflare-triage-cases` | normalise, pseudonymise, dedupe, build a private unlabelled pack | the pseudonym key's custody, the real run |
| P0-4 `feat/triage-reviewer-queue` | blind five-task worksheets, import, Tier B/A report | the reviewers |
| P1-5 … P2-8 | external baselines, training and ablation tooling, head-to-head, drift | Tier A labels, Jev/Laya access, a private runner |

## Where the repository differs from the v5 document

1. The registry exists already (`benchmark/socbench/sources/sources.yaml`); v2 extends it rather than adding a second one.
2. An incident-triage pack could not hold unlabelled telemetry: `load_pack` required one labelled expectation per case. P0-3 adds `label_status: unlabelled`, counted and never scored or exported, and keeps WAF actions in a separate `weak.jsonl` that evaluation and export never read.
3. Annotation and remote scorers know only the single-task socbench ids; the five `soar_zh` tasks need their own reviewer queue (P0-4) and scorer (P1-5).
4. The R1–R5 probes are a one-off script; P1-6 makes them a library call that every candidate check runs.
5. The lock has no temporal holdout; P0-3 adds one for own-telemetry cases.

## Fixed rules

- WAF `block`/`challenge`/score is Tier D, `allow` is U. Neither is gold, and a label derived from a WAF action is never also a model input in the primary arm.
- Raw Cloudflare data, IPs, queries, tokens and the zone id never enter the repository, a GitHub artifact or an external model.
- 100 pilot cases cannot reach schema 2 (150 calibration records per task bucket); the candidate status ceiling stays `research`.
- `models/jef-v0` is not written. No automation is enabled.
