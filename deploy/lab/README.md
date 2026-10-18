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
docker compose -f deploy/lab/docker-compose.yml exec jef python -c \
  "import urllib.request as u; r = u.Request('http://127.0.0.1:8080/v1/readiness', headers={'X-API-Key': 'lab-only-not-a-secret'}); print(u.urlopen(r).read().decode())"
docker compose -f deploy/lab/docker-compose.yml exec mock-targets python -c \
  "import urllib.request as u; print(u.urlopen('http://127.0.0.1:9009/healthz').read().decode())"
```

Nothing is published to the host: Docker does not publish ports for containers whose only network is `internal`, which is the point. Reach the services with `docker compose exec`, or run the matrix below with the `mock-soar` service.

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

### The same matrix, automated

`mock_soar.py` runs the rows above except Egress from inside the lab network, as the SOAR connector and the approval stub, and writes one JSON report per phase:

```bash
mkdir -p deploy/lab/soar-out
docker compose -f deploy/lab/docker-compose.yml up --build -d jef mock-targets
docker compose -f deploy/lab/docker-compose.yml --profile soar run --rm mock-soar matrix
docker compose -f deploy/lab/docker-compose.yml restart jef
docker compose -f deploy/lab/docker-compose.yml --profile soar run --rm mock-soar after-restart
```

CI runs the same steps, plus the store outage, in the `lab-compose` job and archives `lab-compose-<sha>`. It proves the contract on a CI runner; it is **not** evidence that an ESXi segment has no egress (S14), which only `egress_check.sh` on the lab VM can give.

Collect `GET /calls` from the mock, the receipts, the readiness report and the egress output into the Safety Evidence Pack, alongside the automated half. Run it inside the lab VM so the pack is bound to where it ran:

```bash
bash deploy/lab/egress_check.sh | tee /tmp/egress.log
uv run python scripts/safety-evidence.py --operator <name> --lab-host <vm> \
  --port-group JEF-LAB-CORE --egress-log /tmp/egress.log
```

That maps every code-level case to the tests that prove it for the exact SHA, records the operator, host, port group and compose file digest, copies the egress log in with its SHA-256, and passes S14 only if the log ends in `PASS` with no `LEAK` line. Without `--egress-log` the pack says it is a code-level run and S14 stays manual. CI archives the code-level pack per commit. Every T2+ action in that pack must show a grant nonce or a refusal; a dispatch without one is a No-Go.

## What this lab is not

It is not a benchmark of detection quality (that is `jef-bench evaluate` on a locked gold set), not a load test, and not an authorisation to automate: `JEF_DECISIONS_ENABLED=1` here only lets the authorize/complete contract be exercised against mocks.
