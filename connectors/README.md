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
