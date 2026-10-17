# Shadow-only lab

The replay and safety lab from the next-phase plan (§8): offline case replay, JEF behind auth with a persistent decision store, a SOAR connector in dry-run, and **mock** EDR / IAM / firewall / SIEM targets. Nothing in it reaches a production SIEM, directory, EDR console or firewall, and the whole point of the exercise is to produce evidence that it cannot.

## What is here

| File | Purpose |
|---|---|
| `docker-compose.yml` | JEF (production-shaped env, SQLite decision store) plus the mock targets on an **internal** network with no egress. |
| `mock_targets.py` | Records every `isolate_host` / `disable_user` / `block_ip` / `close_alert` call with its idempotency key; `X-Mock-Behaviour: timeout` makes the action land without answering, `error` fails it. `GET /calls` is the evidence; `GET /calls/<key>` is the reconciliation question. |
| `egress_check.sh` | Probes the real Jev, the gateway, external LLMs, Hugging Face, PyPI, GitHub and a public resolver from inside the segment. Every probe must fail. |

## Isolation (do this before anything else)

1. On ESXi/vSphere: a vSwitch **with no physical uplink** and a port group for `JEF-LAB-CORE`. VM-level firewalls deny all egress; a second port group `JEF-LAB-MGMT` admits only the admin IPs over SSH. Do not rely on `/etc/hosts` or a container flag as the isolation boundary.
2. Inside the VM: `sudo ufw default deny outgoing && sudo ufw default deny incoming && sudo ufw allow from <MGMT_IP> to any port 22 proto tcp && sudo ufw enable`. Confirm out-of-band console access first; a wrong rule locks you out.
3. Run `deploy/lab/egress_check.sh` and keep its output. It must end with `PASS`.
4. Stage what the lab needs by hand: wheels or the image, scenes, and (for the real backbone) the `mmBERT-base` weights, each with a recorded SHA-256.

Compose's `internal: true` network is a convenience for a developer laptop, not the lab boundary.

## Bring it up

```bash
docker compose -f deploy/lab/docker-compose.yml up --build -d
curl -s http://127.0.0.1:8080/readyz
curl -s -H 'X-API-Key: lab-only-not-a-secret' http://127.0.0.1:8080/v1/readiness | python3 -m json.tool
curl -s http://127.0.0.1:9009/healthz
```

With the hashing backbone `readiness` reports `backbone:test` and `decision_support_ready: false`; that is correct for a smoke run. Switch `JEF_BACKBONE` and stage weights for the real thing.

## The safety matrix (§8.4), as runbook steps

| Test | Do | Evidence to keep |
|---|---|---|
| Authentication | call `/v1/systemone` without a key | 401, and `jef_requests_total` unchanged for inference |
| Request budget | 3 MB body; 20-deep JSON; 30 questions | 413 / 422 / 413 before inference |
| Rate limit | 61 requests in a minute from one caller | 429 with `Retry-After` |
| Unauthorized action | run a scene with `case_id`, redeem the proposal **without** a grant | `not_dispatched`, `NO_GRANT`; `GET /calls` on the mock stays empty |
| Forged proposal / grant | edit the token or the grant payload | `invalid_proposal` / `INVALID_GRANT`; mock ledger empty |
| Cross-asset replay | grant for host A, proposal for host B | `GRANT_ASSET_MISMATCH` |
| Double spend | one grant, two idempotency keys | second is `GRANT_ALREADY_USED` |
| Unknown timeout | connector calls the mock with `X-Mock-Behaviour: timeout`, then reports `timeout` | receipt `unknown`; `GET /calls/<key>` shows it **landed**; reconcile, do not retry |
| Restart | `docker compose restart jef` after `authorized` | `GET /v1/decisions/<key>` returns the same receipt |
| Store down | stop the volume / make the DB path unwritable | `503 store_unavailable`; nothing authorized |
| Egress | `egress_check.sh` | `PASS` |

Collect `GET /calls` from the mock, the receipts, the readiness report and the egress output into the Safety Evidence Pack. Every T2+ action in that pack must show a grant nonce or a refusal; a dispatch without one is a No-Go.

## What this lab is not

It is not a benchmark of detection quality (that is `jef-bench evaluate` on a locked gold set), not a load test, and not an authorisation to automate: `JEF_DECISIONS_ENABLED=1` here only lets the authorize/complete contract be exercised against mocks.
