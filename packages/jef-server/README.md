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
| `JEF_SCENES_DIR` | Scenes to load and compile at startup |
| `JEF_THREADS` | Intra-op threads. Set below the CPU limit — oversubscribing worsens p99. |
| `JEF_UI_DIR` | Enables `/ui` |
| `JEF_API_KEYS` | Comma-separated keys, optionally labelled (`soc-prod:<secret>`). When set, every endpoint but `/healthz` requires `Authorization: Bearer <key>` or `X-API-Key`. Empty means no app-level auth: only acceptable behind an ingress that authenticates, and `/healthz` says `auth: none`. |
| `JEF_REQUIRE_AUTH` | Refuse to start when `JEF_API_KEYS` is empty. |
| `JEF_MAX_BODY_BYTES` | Request body limit by `Content-Length` (default 8 000 000). Bodies without `Content-Length` are 411; compressed bodies are 415. |
| `JEF_MAX_STATE_CHARS` `JEF_MAX_JSON_DEPTH` `JEF_MAX_QUESTIONS` | Per-request budgets checked after parsing and before inference (413 / 422 / 413). |
| `JEF_RATE_LIMIT_PER_MINUTE` | Per-caller fixed window; 429 with `Retry-After`. 0 disables. |
| `JEF_METRICS_PUBLIC` | Leave `/metrics` reachable without a key. |

### Readiness

`GET /healthz` always answers `status: ok` when the process is up, and adds what a deployer needs to decide whether it is *fit to serve*: `test_backbone`, `calibrated`, `calibration_scope`, `auth`, and `production_safe`, which is true only with a real backbone, a fitted calibrator and app-level authentication. Gate a production rollout on `production_safe`, not on `status`.

## The metric worth alerting on

`jef_state_encodes_total / jef_requests_total` must stay at **1.0** regardless of
how many questions a request carries. A rising ratio means shared-state encoding
has regressed into per-question encoding, which is the one failure that would
leave the project without a reason to exist.

## License

Apache-2.0
