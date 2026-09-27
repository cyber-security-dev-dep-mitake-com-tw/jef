# Acceptance tests

Robot Framework suites that drive JEF's **real HTTP surface**, not its Python
objects. That distinction is the point: these prove what an external TypeSafe or
Vercel AI SDK client would actually experience against a running deployment.

```bash
./scripts/run-acceptance.sh                      # starts a server, runs everything
./scripts/run-acceptance.sh --include d3         # one tag
```

Reports land in `test/results/` (gitignored): `report.html`, `log.html`.

## Layout

| Path | Covers |
|---|---|
| `resources/jef.resource` | Shared keywords. `Ask JEF` posts to `/v1/systemone`; `Metric Value` reads one Prometheus counter. |
| `acceptance/01_systemone_contract.robot` | The compatible contract: all three primitives, both yes/no dialects, accepted state shapes, rejections. |
| `acceptance/02_shared_state.robot` | The D3 guarantee, asserted over HTTP via exported metrics. |

## Tags

`p0` `contract` `d3` `performance` — and one per later phase as they land.

## Why `Ask JEF` and not `Evaluate`

Robot's BuiltIn library already owns the keyword `Evaluate`, which takes a single
Python expression. Naming ours the same shadows it and produces
"expected 2 arguments, got 1" from unrelated lines.
