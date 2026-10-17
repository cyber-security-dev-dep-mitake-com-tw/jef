# Capability matrix

What JEF does today, what is in review, and what it deliberately does not do.
`scripts/check-docs.py` (run in CI) fails when a **shipped** row cites a path
that does not exist, so this table cannot quietly outlive the code it
describes. Status is as of the commit it is read at; an "in review" row names
its pull request.

## Deployment tiers

From the 2026-10-17 engineering review, section 1.2. JEF's current state
supports **L0** and a controlled **L1** proof of concept. It is not a
production automation system.

| Tier | What it may do | Status |
|---|---|---|
| L0 Research | jev-local compatibility, replay, offline evaluation, no sensitive data | supported |
| L1 Decision support | annotate, rank and suggest; never change the source system's state | supported for a controlled PoC: needs a calibrated task artifact, SOCBench gold and app-level auth |
| L2 Low-impact automation | enrichment, tickets, queue routing (T0/T1) | gated: the policy allows T0/T1 only on `validated` calibration, and the shipped calibration is `legacy`, so nothing automates yet |
| L3 Disruptive response | isolate, disable, block, close-and-suppress (T2-T4) | **no-go**: needs a signed per-case grant, an approval service and receipts; never automatic |

## Decision engine

| Capability | Status | Evidence | Notes |
|---|---|---|---|
| Jev-compatible `/v1/systemone` | shipped | `packages/jef-server/src/jef_server/app.py`, `test/acceptance/01_systemone_contract.robot` | contract-compatible; not a claim of Jev's accuracy |
| Native frozen encoder + bilinear head | shipped | `packages/jef/src/jef_core/head.py`, `models/jef-v0/head.npz` | head is tied to its exact backbone |
| Temperature, conformal and isotonic calibration | shipped | `packages/jef/src/jef_core/calibration.py` | |
| Conformal set fails closed (full set without a quantile) | shipped | `packages/jef/src/jef_core/calibration.py`, `packages/jef-go/internal/calib/calib.go` | Python and Go agree |
| Per-question calibration status | shipped | `packages/jef/src/jef_scene/trace.py`, `docs/modules/jef_core.md` | `validated / insufficient / missing / legacy / incompatible` |
| Task-scoped calibration (schema 2) | shipped | `packages/jef/src/jef_core/calibration.py` | **the shipped `models/jef-v0/calibration.json` is schema 1, so it loads as `legacy` and nothing automates until it is refit** |
| Scenes, gates and decision traces | shipped | `packages/jef/src/jef_scene/engine.py`, `scenes/incident-triage.zh-tw.yaml` | |
| Runtime policy verdict and action tiers | shipped | `packages/jef/src/jef_scene/policy.py` | no uncalibrated auto-close; T2+ never automates |
| Proposal, grant, receipt contract | shipped | `packages/jef/src/jef_scene/authz.py`, `packages/jef/src/jef_scene/executor.py` | sealed proposals, two-phase authorize / complete |
| HTTP auth, input budgets, rate limits | shipped | `packages/jef-server/src/jef_server/security.py` | auth is off unless `JEF_API_KEYS` is set; `/healthz` says so |
| Readiness in three strengths | shipped | `packages/jef-server/src/jef_server/readiness.py` | `/healthz` is liveness only; `/readyz` is model loaded + manifest verified (503 on mismatch); `/v1/readiness` reports `decision_support_ready`, `automation_ready`, `validated_tasks[]` and `blocking_reasons[]`, decided per task in `JEF_REQUIRED_TASK_IDS` and fail-closed: legacy, missing, insufficient or incompatible calibration, the hashing backbone, no auth (unless `JEF_TRUSTED_INGRESS`), a memory-only policy store or disabled decisions each block automation. `production_safe` on `/healthz` now equals `automation_ready` and is deprecated. Even `automation_ready: true` authorises no T2+ action. |
| `/v1/decisions` authorize and receipt endpoints | shipped | `packages/jef-server/src/jef_server/decisions.py`, `packages/jef-server/tests/test_decisions.py` | off by default; needs API keys; receipts are in memory, one replica |
| Connector authorize / record outcome (Shuffle, n8n, Cortex) | shipped | `connectors/shuffle/jef/1.0.0/src/app.py`, `connectors/n8n/nodes/Jef/Jef.node.ts`, `connectors/cortex/JEF/jef_responder.py` | connectors never execute; platforms do |
| Model manifest, attestations, SBOMs | shipped | `scripts/model-manifest.py`, `models/jef-v0/manifest.json`, `.github/workflows/release.yml` | mismatch at load drops calibration |
| One version source | shipped | `scripts/set-version.py`, `packages/jef-sdk-ts/src/version.ts` | |
| jev-local research reproduction | shipped | `research/jev-local/backends/index.mjs`, `research/jev-local/test/isolation.test.mjs` | research only; the Claude backend sends state to Anthropic and is opt-in |

## SOCBench

| Capability | Status | Evidence | Notes |
|---|---|---|---|
| Source manifest, fetch, SHA-256 pins | shipped | `benchmark/socbench/sources/sources.yaml`, `packages/jef-train/src/jef_train/socbench/pipeline.py` | restricted sources are never downloaded |
| Case and gold schemas | shipped | `benchmark/socbench/schema/case.schema.json`, `benchmark/socbench/schema/gold.schema.json` | |
| AIT-ADS and CTU-13 adapters | shipped | `packages/jef-train/src/jef_train/socbench/ait_ads.py`, `packages/jef-train/src/jef_train/socbench/ctu13.py` | weak gold, tiers D and C; validated on the published archives (2,655,821 alerts; 19,976,700 flows) |
| Two-reviewer annotation and the gold lock | shipped | `packages/jef-train/src/jef_train/socbench/annotate.py`, `packages/jef-train/src/jef_train/socbench/splits.py` | |
| Family-aware sampler and analyst labelbook | shipped | `packages/jef-train/src/jef_train/socbench/sample.py`, `benchmark/socbench/annotation/LABELBOOK.md` | `jef-bench sample` draws within families, water-fills weak-label strata rarest first under a hard per-family cap, writes cases without the weak label so reviewers stay blind, and records inverse-inclusion weights; it warns when a source has too few families for meaningful intervals. The labelbook defines every task's labels, `unknown` versus `insufficient_evidence`, and the adjudication rules as the tools enforce them. Reviews may carry `seconds_spent`; `annotate agree` reports measured throughput. |
| Baselines and SOC metrics report | shipped | `packages/jef-train/src/jef_train/socbench/report.py`, `packages/jef-train/src/jef_train/socbench/metrics.py`, `packages/jef-train/src/jef_train/socbench/baselines.py` | B0, B1, B2/B3 and B4; B5 needs a schema-2 calibration |
| zh-TW seed cases and bilingual track | shipped | `benchmark/socbench/seeds/zh-TW/cases.jsonl`, `packages/jef-train/src/jef_train/socbench/bilingual.py` | LLM-authored seeds, tier E, awaiting review |
| BOTS v3 investigation cases | planned | | the Q&A is not in the public repository |
| LANL, CICIDS2017, CAC datasets | planned | | access needs a human request or registration |
| Analyst-reviewed test set (300+ double-reviewed) | not started | | needs analysts; no number from SOCBench is a SOC claim until it exists |

## Not supported, on purpose

| | Why |
|---|---|
| Automatic isolation, account disablement, firewall blocks, close-and-suppress | a probability is not an authorization; T2+ needs a person's signed grant |
| General knowledge questions | chance level on TMMLU+; JEF judges evidence you supply |
| Text generation | a System One engine returns distributions, never prose |
