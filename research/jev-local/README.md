# jev-local: Jev reproduced on this Mac

A local reproduction of TypeSafe AI's **Jev** "System One" decision model that runs on this machine with **no Vercel traffic**. It speaks Jev's two public protocols, so the official TypeSafe SDK, the Vercel AI SDK's `experimental_decide`, plain curl, and the bundled `jev` CLI all work against it unchanged. Judgment comes from Claude through your local `claude` CLI (Anthropic only); an offline heuristic backend exists for contract tests.

What is reproduced: the API contract, limits, aliases, confidence maths, answer types, zstd/gzip request handling, both wire dialects, and Jev's decision behaviour as measured against every answer TypeSafe has published. What is not: Jev's weights. Those are private, so the judgment engine is a stand-in model prompted to behave as a calibrated decision model.

## Quick start

```bash
npm install
npm start                                   # local Jev on http://127.0.0.1:4141, Claude backend
```

Ask it something (security triage is the intended workload; see `examples/`):

```bash
node bin/jev.mjs ask "From: IT Service Desk <helpdesk@m1crosoft-support.co>. URGENT: re-validate your password within 24h at hxxp://login-m1crosoft-support.co or lose your mailbox." \
  --noul is_phishing="Is this email a phishing attempt?" \
  --choice action="What should the SOAR playbook do?" quarantine_and_alert="Pull and alert the user" flag_for_analyst="Hold for an analyst" deliver="Deliver normally" \
  --score severity="How severe if a user acts on it?" "Harmless" "Nuisance" "Credential theft risk" "Credential theft plus lateral movement"
```

Examples in `examples/`:

- `phishing-triage.sh`: anti-phishing triage of one email in a single request (lure type, playbook action, severity).
- `soar-decision-chain.mjs`: a SOAR playbook as a chain of Jev decisions with confidence gates (triage → containment → escalation), state passed as structured JSON.
- `ndr-alert-triage.mjs`: batch NDR alert triage, ranked by true-positive probability × severity.

Use it from the official TypeSafe SDK (same code you would write against api.typesafe.ai, only `baseURL` differs):

```js
import { TypeSafeClient, choice, noul, score } from '@typesafe-ai/sdk';
const client = new TypeSafeClient({ apiKey: 'local', baseURL: 'http://127.0.0.1:4141', defaultModel: 'jev-latest' });
const r = await client.systemOne({ state: ticket, questions: { department: choice('Which team?', { billing: '…', technical: '…' }) } });
```

Use it from the Vercel AI SDK without touching Vercel (the shim sets the default provider to the local server):

```bash
node --import ./register.mjs snippet.mjs     # the vercel.com model-page snippet, verbatim
```

## Options

| Env / flag | Default | Meaning |
|---|---|---|
| `--port`, `PORT` | 4141 | listen port (loopback only) |
| `--backend`, `JEV_BACKEND` | `claude` | `claude` (Anthropic via `claude` CLI) or `heuristic` (offline) |
| `--model`, `JEV_CLAUDE_MODEL` | `claude-sonnet-5-5` | Claude model used for judgment; `claude-haiku-5-5` is about 3x cheaper and slightly less calibrated |
| `--samples`, `JEV_SAMPLES` | 1 | self-consistency samples averaged per request (×cost, better calibration) |
| `JEV_CONCURRENCY` | 2 | parallel `claude` processes |
| `JEV_CACHE_DIR` | `.jev-cache/` | content-hash cache; identical requests are free and instant |
| `TypeSafe-Danger-Zone-Long-Context: true` header | off | raises both context limits to 262k tokens, as the real API does |

Cost: Jev's own price does not apply. Each uncached request is one `claude` CLI call, roughly $0.01 to $0.02 with Sonnet or $0.003 to $0.007 with Haiku, billed to your Anthropic account. Identical requests are served from the cache for free.

## Endpoints

| Route | Dialect | Client |
|---|---|---|
| `POST /v1/systemone` | TypeSafe native (`state`, `model`, `questions` with `noul` / `choice` / `score`) | `@typesafe-ai/sdk`, curl, `jev ask` |
| `POST /v1/decision-model` | Vercel AI Gateway decision-model spec v4 (`stateParts`, `questions` with `boolean` / `choice` / `score`) | `ai` `experimental_decide` via `@ai-sdk/gateway` with `baseURL` set to the local server |
| `GET /v1/models` | TypeSafe shape (`?gateway=1` for the gateway shape) | both |
| `GET /healthz` | | |

## Fidelity vs Jev's published answers

`fixtures-docs.json` holds every request/response pair in TypeSafe's docs where both the full request and the real `jev-1.13.0` response are printed: 5 states, 10 questions across noul, choice and score. `npm run bench` replays them against a running local Jev. Decision agreement = same yes/no side, same chosen option, or same peak score level. Probability error = mean |Δp| for noul, or half the L1 distance between distributions for choice and score (0 = identical distribution).

| Configuration | Decision agreement | Mean probability error |
|---|---|---|
| heuristic (offline) | 80% | 0.313 |
| claude-haiku-5-5, prompt v1, 1 sample (first run, 2026-10-15) | 100% | 0.105 |
| claude-haiku-5-5, prompt v1, 1 sample (re-run) | 90% | 0.154 |
| claude-haiku-5-5, prompt v2, 1 sample | 90% | 0.145 |
| claude-haiku-5-5, prompt v2, 3 samples | 80% | 0.160 |
| claude-sonnet-5-5, prompt v2, 1 sample (default) | 90% | 0.084 |

Measured 2026-10-15/16. Claude's sampling makes single runs vary by about one decision on this small set; the two Haiku prompt-v1 rows are the same configuration run twice. Sonnet is the default because it has the lowest probability error, and its one miss (quickstart `frustration`, 0.43 vs Jev's 0.67) is a case Jev itself split 0.33/0.67. The remaining documented responses (5-question returns triage, 3-score bug triage, `return_topic`, structured-criteria `bug_severity`) are excluded because the docs do not print their input state. No live comparison against the real model was run: the user chose not to call Vercel's API.

## How it was reverse engineered

| Contract item | Source |
|---|---|
| Native HTTP API, question/answer types, limits, errors, compression | docs.typesafe.ai/api.md (public) |
| Confidence formulas | docs.typesafe.ai/confidence.md |
| Models, aliases, context limits, long-context header, pricing | docs.typesafe.ai/models.md |
| Gateway dialect (`/decision-model`, spec headers, body and answer schemas) | REA `analyze-javascript-application` on `@ai-sdk/gateway@4.0.114`, see `rea-evidence.md` |
| Client-side answer validation the server must satisfy | REA on `ai@7.0.141`, see `rea-evidence.md` |
| Official SDK behaviour (zstd by default, `baseURL`, retry headers) | `@typesafe-ai/sdk@0.7.0` exercised by `test-typesafe-sdk.mjs` |
| Documented Jev answers used as ground truth | `fixtures-docs.json`, harvested from the docs |

REA needed a shadow Command Line Tools toolchain on this host (`scripts/make-shadow-clt.sh`) because the installed CLT ships a duplicate `SwiftBridging` module map that breaks the Swift helper REA compiles.

## Layout

```
bin/jev.mjs             launcher and CLI: serve | ask | models
server.mjs              HTTP server, both dialects, zstd/gzip bodies, limits, errors
jev-core.mjs            contract: validation, limits, aliases, confidence, answer builders
backends/claude.mjs     judgment via `claude` CLI: Jev system prompt, validation, retries, sampling, cache, concurrency gate
backends/heuristic.mjs  offline deterministic stand-in for contract tests
register.mjs            default-provider shim so `decide({ model: 'typesafe-ai/jev' })` stays local
examples/               cybersecurity decision examples: phishing triage, SOAR chain, NDR triage
test-typesafe-sdk.mjs   official SDK against local Jev (zstd and plain)
test-decide.mjs         AI SDK decide() + native endpoint + error contract
bench-fidelity.mjs      replays fixtures-docs.json; writes fidelity.json
fixtures-docs.json      documented jev-1.13.0 request/response pairs
rea-evidence.md         REA evidence ids and source ranges
scripts/make-shadow-clt.sh  toolchain workaround for REA on this Mac
```

## Not reproduced

- Jev's weights and RLCD training. Behaviour is imitated by Claude under a decision-model prompt; agreement is measured, not guaranteed, and latency is seconds per uncached request instead of Jev's sub-second.
- `box` (image) questions and image state; Jev itself is text-only.
- Rate limiting, the developer console, and TypeSafe's exact token counter (usage is estimated at 4 characters per token).
