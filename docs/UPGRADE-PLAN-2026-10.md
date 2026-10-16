# JEF upgrade plan, October 2026

Assessment of [`JEF-SECURITY-ROADMAP-AND-SOC-BENCHMARK.md`](JEF-SECURITY-ROADMAP-AND-SOC-BENCHMARK.md) (the roadmap, dated 2026-10-16) against `main` at the time of writing, followed by the plan that comes out of it. Every claim below was checked in the repository; file and line references point at `main` after PR #1 and PR #2 merged.

## 1. Assessment of the roadmap

### Accurate and verified

| Roadmap claim | Where it is true | Note |
|---|---|---|
| `prediction_set()` returns an argmax singleton when the bucket or the alpha quantile is missing | `packages/jef/src/jef_core/calibration.py:248-256` | Three separate paths collapse to `[argmax]`: no bucket, no `qhat` for that alpha, and an empty keep-set. A singleton is exactly the shape a scene gate treats as "safe to act on". |
| `is_fitted()` is global, not per task bucket | `calibration.py:282-283` (`bool(self.buckets)`), used by `jef_scene/engine.py:279` | The engine returns the full option set only when *no* bucket exists at all. One fitted bucket makes every other kind/size fall through to the singleton path above. |
| Calibration is keyed by `kind:n_options` only | `calibration.py:54-56`; `models/jef-v0/calibration.json` buckets are `choice:4`, `noul:2`, `choice:14`, `score:5` | A phishing yes/no question shares `noul:2` with CVE user-interaction. The roadmap's task-scoped key is the right fix. |
| `incident-triage.zh-tw.yaml` closes on `is_false_positive.probability > 0.90` with `human_review: false` | `scenes/incident-triage.zh-tw.yaml:32-34`, action at `:92-95` | `close` is irreversible in the scene vocabulary and has no evidence allow-list. |
| `docs/LANDSCAPE.md` says JEF has no MCP server and no CLI | `docs/LANDSCAPE.md:104` | README and `ci.yml` say both exist. Documentation drift, not a code bug. |
| `research/jev-local` server defaults to `heuristic` while the CLI and README default to `claude` | `research/jev-local/server.mjs` (`JEV_BACKEND \|\| 'heuristic'`) vs `bin/jev.mjs` (`'claude'`) | Running `node server.mjs` directly gives a non-production backend silently. |
| Backend module is imported from an unvalidated environment string | `research/jev-local/server.mjs` (`import(\`./backends/${BACKEND}.mjs\`)`) | Any value is attempted; the failure is a stack trace at startup. |
| The SOAR example only logs `execute isolate_host` | `research/jev-local/examples/soar-decision-chain.mjs` | Correct; the example must not be read as an executed containment. |
| jev-local fidelity rests on 10 documented questions with no live comparison | `research/jev-local/README.md` | Correct, and by decision: no Vercel API calls. |

### Already done or stale

| Roadmap item | Status |
|---|---|
| gzip path has no decompressed-size cap (P1) | Fixed in PR #1: `gunzipSync(buf, { maxOutputLength })`, with zstd capped the same way. |
| Oversized body handling, zero-sum distributions, incomplete benchmark runs | Fixed in PR #1 after review. |
| Release pipeline credentials and billing failures | Diagnosed and documented in PR #2 (`docs/RELEASING.md` preflight); npm trusted publishing added to `release.yml`. |
| CTI-Bench numbers quoted as 0.840 / 0.805 / 0.435 / 0.575 | The nightly of 2026-10-16 reports 0.832 / 0.804 / 0.428 / 0.580. The badges and `docs/benchmarks.json` are the source of truth; prose numbers drift. |

### Where the roadmap overreaches, and how this plan handles it

- **Scope.** Fourteen PRs across calibration, policy, four dataset ingestions, an annotation workbench, baselines, signed releases and governance docs is a quarter or more of work for a small team. The plan below keeps the dependency order but marks a minimum viable path (M0 plus the first dataset) that can land in weeks.
- **A separate `jef-policy` package and `/v2/decisions` API.** Right in principle; premature until M0 lands. Start with a `PolicyDecision` dataclass inside `jef_scene` and a feature flag, extract the package when a second consumer exists.
- **Dataset licensing.** Several sources need application or registration (LANL, ChiPhish full set, the CAC "網路安全資料集 1.0"). The manifest in `benchmark/socbench/sources/` records access status per source so nobody assumes a download is permitted.
- **Sample-size targets (1,400 cases, 300 to 500 double-reviewed).** Reasonable as goals; the plan treats them as M1 exit criteria, not promises.

## 2. The plan

Priorities follow the roadmap's P0/P1/P2. Effort is S (a day), M (a week), L (several weeks). Each item names the files it touches and the evidence that closes it.

### M0: research hardened (P0, S to M each)

| # | PR | Files | Acceptance | Effort |
|---|---|---|---|---|
| 01 | `fix(calibration): fail closed on missing task bucket` | `calibration.py` `prediction_set`, `tests/core/test_calibration.py` | Missing bucket or alpha returns the full index set; `keep` empty also returns the full set; new tests CAL-001/CAL-002. `is_never_empty` test still holds. | S |
| 02 | `feat(core): calibration status per question` | `calibration.py` (new `CalibrationStatus` enum: `validated`, `missing`, `insufficient`, `stale`), `engine.py` `AnswerView` gains `calibration_status`; `scene.schema.json` | Engine fills status per question; `_prediction_set` no longer consults the global `is_fitted()`. DSL exposes `<qid>.calibration_status`. | M |
| 03 | `feat(scene): human review when calibration is not validated` | `jef_scene/engine.py`, all three `scenes/*.yaml`, scene fixtures | Any gate that selects an action with `human_review: false` is refused at validate time unless it also requires `calibration_status == validated`; `incident-triage` `close` becomes `close_reversible` with `queue: closed-pending-review` and a 7-day expiry param. | M |
| 04 | `feat(core): task-scoped calibration manifest` | `calibration.json` format v2 (`task`, `label_schema`, `language`, `model_sha256`, `dataset_version`, `window`), loader keeps v1 readable as `status: legacy` | Old files still load; new fits write v2; a question whose `task_id` has no bucket gets `missing`, never a sibling bucket. | M |
| 05 | `fix(jev-local): backend allow-list and one default` | `research/jev-local/server.mjs`, `bin/jev.mjs`, README | `JEV_BACKEND` must be `claude` or `heuristic`, anything else fails startup with a one-line error; both entry points default to `claude`; `/healthz` reports backend and model; heuristic logs a `non-production` banner. | S |
| 06 | `security(jev-local): treat state as untrusted` | `backends/claude.mjs`, new `tests/injection.test.mjs` | Prompt-injection fixtures (subject says "report benign", log field says "disable user X", base64/homoglyph noise) must not change the verdict class; the CLI is spawned with `--disallowedTools` for everything and a token budget; a redaction hook strips e-mail addresses, IPs and hostnames before egress unless `JEV_ALLOW_PII=1`. | M |
| 07 | `docs: capability matrix and drift check` | `docs/LANDSCAPE.md`, `scripts/check-docs.py`, `ci.yml` | LANDSCAPE no longer claims "no MCP, no CLI"; CI fails if README links or entrypoints named in docs do not resolve. | S |

Exit criterion for M0: no scene can dispatch a `human_review: false` action from an uncalibrated question, proven by tests, and `uv run jef validate scenes/` rejects a scene that tries.

### M1: reproducible SOCBench (P0 data, M to L)

| # | PR | Acceptance | Effort |
|---|---|---|---|
| 08 | `data(bench): sources manifest and adapters skeleton` | `benchmark/socbench/sources/sources.yaml` (this PR), `schema/case.schema.json`, `schema/gold.schema.json`, adapter stubs that only parse and hash. No data committed. | S (manifest now), M (schemas) |
| 09 | `data(bench): ingest AIT-ADS with provenance and phase labels` | Adapter downloads from Zenodo 8263181 to an ignored directory, records SHA-256, normalises timestamps, joins `labels.csv` phase windows as `attack_phase_window` only; every case carries `label_origin: attack-phase-window`, never a verdict. | M |
| 10 | `data(bench): ingest CTU-13 with Background as unknown` | `Botnet`/`C&C` → positive, `Normal` → benign, `Background` → `unknown`; a test asserts no `Background` flow is ever labelled benign. | M |
| 11 | `data(bench): BOTS v3 investigation cases` | Cases built from investigation questions, not per-log labels; Q&A access status recorded in the manifest. | M |
| 12 | `bench: annotation workbench and gold workflow` | Two-reviewer labels, adjudication, `family_id`, hard-negative pairing, agreement and kappa reported per class; evaluation split locked by `family_id`. | L |
| 13 | `bench: zh-TW layer` | Parallel bilingual set (same evidence, `en` and `zh-TW` questions) and a native Taiwan set (bank, e-commerce, LINE, e-invoice, government notices) with paired legitimate versions; `zh-TW` and `zh-CN` reported separately. | L |

Exit criterion for M1: a locked test set with provenance, at least 300 double-reviewed gold cases, group split checked for leakage, hard negatives at 35 to 45 percent of negatives.

### M2: comparative evidence (P1, M)

| # | PR | Acceptance |
|---|---|---|
| 14 | `bench: harness, baselines and risk-coverage curves` | Same frozen split for rules-only, logistic/GBM on tabular features, LLM-only (Claude, fixed instructions), jev-local, JEF native. Report PR-AUC, precision@B, recall@B, FNR, ECE/Brier, conformal coverage and set size, selective-risk curve, per language and per task, with family-cluster bootstrap CIs. Cached and uncached latency reported separately. |
| 15 | `bench: low-base-rate reweighting` | Metrics reported at realistic prevalence (1 to 5 percent) in addition to the balanced set. |

### M3: supervised SOC pilot (P1/P2, L)

| # | PR | Acceptance |
|---|---|---|
| 16 | `feat(policy): ProposedDecision, PolicyAuthorization, ExecutionReceipt` | Dataclasses in `jef_scene`; connectors refuse to dispatch without an authorization object; T3/T4 actions require an approver identity; `dispatch_count == 0` on missing or expired approval (SEC-006). |
| 17 | `feat(connectors): idempotency, dry-run, receipts, rollback` | Shuffle, n8n, Cortex connectors carry an idempotency key, support dry-run, return a receipt, and expose rollback; failure-injection tests. |
| 18 | `release: signed manifests` | `model-card.md`, `calibration-manifest.json`, `benchmark-report.json`, weights SHA-256, SBOM and build provenance attached to every release. |

### Not planned

- Another Jev-compatible endpoint, more synthetic SOAR routing data, sharper softmax for its own sake, translated text as native zh-TW evidence, or claiming autonomous containment from one correct suggestion. The roadmap's section 16 stands.

## 3. Dataset sources

The sources below are recorded in [`benchmark/socbench/sources/sources.yaml`](../benchmark/socbench/sources/sources.yaml) with access status, ground-truth level and licence notes. Nothing is downloaded or committed by this plan; adapters in M1 fetch into an ignored directory and record checksums.

Priority order for JEF, as the roadmap recommends: **AIT-ADS** (alert triage, correlation, hard negatives), **CTU-13** (NDR and C2), **BOTS v3** (investigation and evidence grounding), then **LANL** (identity, lateral movement, low base rate) and **Mordor/OTRF** (ATT&CK mapping). CICIDS2017 is secondary because of documented labelling problems. For Chinese: **TS-Bench** (zh-TW, 400 prompts with paired hard negatives), **TREC06c** (zh e-mail spam/ham as benign-mail negatives), **ChiPhish** (phishing pages; full set is access-restricted), the CAC **網路安全資料集 1.0** (registration required, per-record label quality unverified), and **TWCERT/CC** case reports and open-data statistics for local taxonomy, not per-case labels.

Three rules apply to every source: the ground-truth level is recorded as what it is (attack-phase window, flow label, CTF answer key, partial red-team positives), never upgraded to an analyst verdict; licences and redistribution terms are checked before anything leaves the ignored data directory; and `Background`, unlabelled or non-red-team traffic is `unknown`, not benign.

## 4. Release note

Release run #5 was a manual `workflow_dispatch` with version `0.0.0.1`. The verify job rejects anything that is not `X.Y.Z` or `X.Y.ZrcN` (`scripts/set-version.py:47`), and the packages are at `0.2.0rc4`, so it stopped before building. No tag was created and nothing was published. The next rehearsal is `python scripts/set-version.py 0.2.0rc5`, commit, tag `v0.2.0rc5`, push, once the two preflight items in `docs/RELEASING.md` (Actions billing, npm token or trusted publisher) are done.
