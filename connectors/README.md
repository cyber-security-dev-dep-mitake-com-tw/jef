# SOAR connectors

JEF's scene layer exists because playbook decisions are today hand-written
if/else inside SOAR platforms. These connectors put it where those playbooks
already live.

| Platform | Path | Form |
|---|---|---|
| [Shuffle](https://shuffler.io) | `shuffle/jef/` | App (`api.yaml` + Python) |
| [n8n](https://n8n.io) | `n8n/` | Community node (`n8n-nodes-jef`) |
| [TheHive / Cortex](https://thehive-project.org) | `cortex/JEF/` | Two analyzers and a responder |

Each has its own README with install steps.

## What they all do the same way

**Ask everything at once.** JEF evaluates every question in a request in
parallel against the same state, which it reads only once. Five questions cost
barely more than one, so every connector offers a "many questions" action:
asking the speculative ones up front and branching on the results beats one
round trip per branch.

**Preserve option order.** The order of your options is the index each one gets
in the returned distribution, so every connector keeps insertion order from the
form field through to the response. There is an acceptance test that asserts
this survives the whole round trip, because it is the kind of thing that breaks
silently and produces answers that look fine.

**Distinguish "unknown" from "low".** `confidence` is how peaked the
distribution is, not the chance of being right. `p_correct` is the calibrated
mapping onto observed correctness, and it is `null` when the server has no data
to support one. The connectors surface that as *unknown*, never as a number and
never as zero — an analyst deciding whether to trust an automated verdict is
making a different decision in each case.

**Let a workflow refuse to automate.** Every connector exposes the server's
health, and two flags there are worth gating on: `calibrated: false` means no
threshold means what it looks like, and `test_backbone: true` means the answers
are well-formed and carry no semantics whatsoever.

## Testing

```bash
# Request shaping, with the platform SDKs stubbed out
pytest connectors/tests

# End to end against a live server, including that option order survives
./scripts/run-acceptance.sh --include connectors
```

The split is deliberate. Installing `cortexutils` and `shuffle_sdk` to test
request shaping would test those platforms, not the connector; the Robot suite
covers the other half by driving the real connectors against a real server.


## A scene result is a proposal, not an authorization

Every connector returns the decision trace, and the trace now says two
things separately: `action` (what the scene selected) and `policy_verdict`
(`allow` or `review_required`, with `reason_codes`). A workflow may act on
`allow` for T0/T1 actions (tagging, routing, enrichment, tickets). For
anything that changes state (`close`, `suppress`, `isolate_host`,
`disable_user`, `block_ip`) the trace will say `review_required`, and the
only thing that turns it into an action is a signed `AuthorizationGrant` for
that case, that action and that asset, verified by `jef_scene.Executor`
before it calls the platform. `approved: true` in a payload is not a grant.

The connectors themselves never execute. The Cortex responder tags the case
`JEF:proposed=<action>` and `JEF:policy=<verdict>`; the Shuffle app and the
n8n node return the trace for the workflow to branch on. When you do wire a
platform operation to a result, wrap it in a `Transport` and go through
`Executor.dispatch` with an idempotency key: it executes once per key,
refuses without a valid grant, and returns an `ExecutionReceipt` whose
`status` is `success` only when the platform confirmed it. A timeout is
`unknown`, never `success`. See `docs/modules/jef_scene.md`.


## Authorize, act, record

When a workflow does act on a scene result, it goes through the server, not
around it. The server never performs the action and never holds EDR, IAM or
firewall credentials; the platform does, with its own.

1. **Run the scene with a case.** Pass `case_id` (and the `asset_id` the
   action would touch). With `JEF_DECISIONS_ENABLED=1` on the server, the
   trace carries a `proposal_token`: the proposal sealed with a server-only
   secret, so the action, tier, verdict, case and asset cannot be edited on
   the way back.
2. **Authorize.** `POST /v1/decisions:authorize` with the token, an
   `idempotency_key` (stable per intended action, e.g.
   `<case>:<action>:<asset>`), and for anything beyond an allowed T0/T1
   action a signed `AuthorizationGrant` from your approval service.
   `dry_run: true` runs every check and records nothing. Branch on
   `status == "authorized"`; anything else is `not_dispatched` with
   `reason_codes`. A retried step with the same key gets the same answer.
3. **Act** with the platform's own app.
4. **Record.** `POST /v1/decisions/{key}:complete` with `outcome`:
   `success` or `failed` only on the platform's own answer, `timeout` (stored
   as `unknown`) when it did not answer. A finished key cannot change its
   outcome. `GET /v1/decisions/{key}` returns the receipt.

| Connector | How |
|---|---|
| Shuffle | `run_scene` takes `case_id`/`asset_id`; actions `authorize_action`, `record_outcome`, `get_decision` |
| n8n | Run Scene takes Case ID / Asset ID; operations Authorize Action, Record Outcome, Get Decision |
| Cortex | the responder sends the TheHive case id and puts `proposal_token` in its report (not in a tag) |

Receipts live in the server's memory: run one replica (`JEF_WORKERS=1`)
until a shared store exists.
