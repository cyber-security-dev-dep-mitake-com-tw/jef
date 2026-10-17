# jef-server

HTTP server for [JEF](https://github.com/cyber-security-dev-dep-mitake-com-tw/jef).
Serves the `/v1/systemone` contract, so existing TypeSafe and Vercel AI SDK
clients work against it unchanged.

```bash
pip install jef-server
JEF_BACKBONE=jhu-clsp/mmBERT-base python -m jef_server
```

Or `pip install "jef[server]"` to get it with the CLI.

## Endpoints

| Method | Path | |
|---|---|---|
| POST | `/v1/systemone` | The compatible contract. Both `noul` and `boolean` spellings accepted and echoed. |
| POST | `/v1/scenes/{id}:evaluate` | Run a scene, return its full decision trace. |
| GET | `/v1/scenes` `/v1/models` `/v1/limits` | Discovery. |
| GET | `/healthz` | Includes `calibrated` and `test_backbone`, both worth gating on. |
| GET | `/metrics` | Prometheus. |
| GET | `/ui` | Decision viewer, when `JEF_UI_DIR` is set. |

## Configuration

| Variable | |
|---|---|
| `JEF_BACKBONE` | Model id, or `hashing` for the deterministic test stub |
| `JEF_HEAD_PATH` `JEF_CALIBRATION_PATH` | Trained artifacts |
| `JEF_MODEL_NAME` | The model id reported in responses and `/v1/models` (default: derived from backbone and head) |
| `JEF_ALPHA` | Stored on the engine (default 0.10) but **currently unused**: scene prediction sets use each scene's own `alpha`, and `/v1/systemone` returns no prediction set |
| `JEF_MODEL_MANIFEST` | `manifest.json` to verify the head and calibration against; defaults to the one beside the head. On a mismatch the server runs without calibration and `/healthz` says `model_manifest: mismatch`. |
| `JEF_SCENES_DIR` | Scenes to load and compile at startup |
| `JEF_THREADS` | Intra-op threads. Set below the CPU limit — oversubscribing worsens p99. |
| `JEF_UI_DIR` | Enables `/ui` |
| `JEF_API_KEYS` | Comma-separated keys, optionally labelled (`soc-prod:<secret>`). When set, every endpoint but `/healthz` requires `Authorization: Bearer <key>` or `X-API-Key`. Empty means no app-level auth: only acceptable behind an ingress that authenticates, and `/healthz` says `auth: none`. |
| `JEF_REQUIRE_AUTH` | Refuse to start when `JEF_API_KEYS` is empty. |
| `JEF_MAX_BODY_BYTES` | Request body limit by `Content-Length` (default 8 000 000). Bodies without `Content-Length` are 411; compressed bodies are 415. |
| `JEF_MAX_STATE_CHARS` `JEF_MAX_JSON_DEPTH` `JEF_MAX_QUESTIONS` | Per-request budgets checked after parsing and before inference (413 / 422 / 413). |
| `JEF_RATE_LIMIT_PER_MINUTE` | Per-caller fixed window; 429 with `Retry-After`. 0 disables. |
| `JEF_METRICS_PUBLIC` | Leave `/metrics` reachable without a key. |
| `JEF_DECISIONS_ENABLED` | Serve `/v1/decisions` (authorize, complete, receipt). Off by default; requires `JEF_API_KEYS`. JEF authorizes and records; it executes nothing. |
| `JEF_GRANT_SECRET` | Verifies `AuthorizationGrant`s, shared with your approval service. Empty: every grant is refused. |
| `JEF_PROPOSAL_SECRET` `JEF_PROPOSAL_TTL_SECONDS` | Seals `proposal_token`s (default: random per process) and how long they stay redeemable (3600). |
| `JEF_REQUIRED_TASK_IDS` | Comma-separated tasks this deployment must be calibrated for before `/v1/readiness` reports `automation_ready`: `task_id` (every bucket fitted for the task must be `validated`) or `task_id:kind:n` (one exact bucket). Empty is itself a blocking reason. |
| `JEF_TRUSTED_INGRESS` | The operator asserts that an ingress (mTLS, OIDC gateway) authenticates every caller, so empty `JEF_API_KEYS` does not block readiness. Reported as `auth: trusted-ingress`, never silently. |

### Readiness

Three endpoints, three strengths, so no probe is mistaken for permission:

| Endpoint | Answers | Auth |
|---|---|---|
| `GET /healthz` | The process answers. Descriptive fields stay for dashboards; `production_safe` is deprecated and now equals `automation_ready` below. | open |
| `GET /readyz` | The model is loaded and its files match the manifest. **503** on a mismatch, with the manifest's problems. | open |
| `GET /v1/readiness` | `decision_support_ready` (a real backbone, no manifest mismatch, authenticated), `automation_ready` (all of it), `validated_tasks[]`, `blocking_reasons[]`, `checks{}`. | as other `/v1` routes |

`automation_ready` is decided per task in `JEF_REQUIRED_TASK_IDS` from that task's own calibration buckets and fails closed. Each of these blocks it and is named in `blocking_reasons`: `backbone:test`, `model_manifest:<status>`, `auth:none`, `calibration:legacy|incompatible|absent`, `<task>:missing|insufficient|legacy|incompatible_calibration`, `required_tasks:none_configured`, `decisions:disabled`, `policy_store:not_persistent`. A validated bucket for another task, or for the same shape without a task id, never counts. Even `automation_ready: true` authorises no T2+ action; every response says so.

## The metric worth alerting on

`jef_state_encodes_total / jef_requests_total` must stay at **1.0** regardless of
how many questions a request carries. A rising ratio means shared-state encoding
has regressed into per-question encoding, which is the one failure that would
leave the project without a reason to exist.

## License

Apache-2.0
