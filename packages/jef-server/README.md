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

## The metric worth alerting on

`jef_state_encodes_total / jef_requests_total` must stay at **1.0** regardless of
how many questions a request carries. A rising ratio means shared-state encoding
has regressed into per-question encoding, which is the one failure that would
leave the project without a reason to exist.

## License

Apache-2.0
