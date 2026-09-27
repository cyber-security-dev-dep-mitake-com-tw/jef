# How Jev, Laya and Von reach their users

Written while deciding how JEF should be delivered. All three competitors
converge on the same wire contract, which turns out to be the most important
fact here.

## The short version

| | Jev | Laya | Von | **JEF** |
|---|---|---|---|---|
| Weights | **closed** | Apache-2.0 on HF | Apache-2.0 on HF | Apache-2.0 on HF |
| Access | hosted API only | self-host only | self-host only | self-host |
| Cost | $0.042 / M input tokens, output free | free | free | free |
| Endpoint | `POST /v1/systemone` | `POST /v1/systemone` | `POST /v1/systemone` | `POST /v1/systemone` |
| Backbone | undisclosed | ModernBERT-large 421M / mmBERT-base 322M | 395M, 1.5 GB | mmBERT-base ~307M |
| Context | — | **512** (en) / **1,024** (multi) | — | **8,192** |
| Scene layer | no (explicitly caller's job) | no | no | **yes** |

## Jev — hosted API, no weights

Access is through TypeSafe's console: sign up, create an API key, top up, call
`/v1/systemone`. The waitlist came off on 20 September 2026. Pricing is
**$0.042 per million input tokens with output free**, which is coherent with the
architecture — a System One model generates nothing, so there are no output
tokens to bill. Rate limits were 250,000 tokens/second and 1,200 requests/minute
as of September 2026, described as dynamic during early access.

It is also resold through several gateways at the same price: **Vercel AI
Gateway** (no markup, no platform fee, and every team gets $5/month of credit on
the free tier), **Cloudflare Workers AI** (`env.AI.run('typesafe/jev', …)`),
OpenRouter and Requesty.

Client surface: a Python SDK, a JavaScript SDK, the Vercel AI SDK's
`experimental_evaluate`, and an agent skill for Claude Code and Codex.

There is no self-hosted option and no downloadable weights. That is the whole
reason this repository exists.

## Laya — Apache-2.0 weights, self-host only

```bash
pip install "laya[serve]"
LAYA_DEVICE=cuda LAYA_PRELOAD=1 laya-serve      # binds 0.0.0.0:8000
```

Three checkpoints on Hugging Face under `convaiinnovations/`: English (~808 MB,
ModernBERT-large 421M), multilingual (~647 MB, mmBERT-base 322M), and one
fine-tuned on the typed-decisions benchmark. Configured by environment:
`LAYA_HOST`, `LAYA_PORT`, `LAYA_DEVICE`, `LAYA_PRELOAD`, `LAYA_API_KEY`,
`LAYA_THREADS`.

Delivery is unusually thorough: a Docker Compose setup with ARM64 and DGX Spark
builds, a NixOS module, an MCP server (`pip install "laya[mcp]"`,
`laya-mcp-server`) for Claude Desktop and Cursor, a CLI, and a third-party
Node/TypeScript runtime via ONNX Runtime. Reported latency is 32.8 ms on a T4
and 193–464 ms on CPU.

Jev-wire-compatible: **clients change only the base URL.**

## Von — Apache-2.0 weights, self-host only

```bash
pip install von-sdk          # or: bun add von-sdk
von serve --host 0.0.0.0 --port 8000
```

395M parameters, 1.5 GB on Hugging Face. Python and TypeScript bindings, and
acceleration across CUDA, ROCm, Apple MPS, Intel OpenVINO and multithreaded CPU.
Post-trained with RLCD to optimise accuracy and calibration jointly. States
"fully compatible with the TypeSafe `/v1/systemone` specification". No hosted
offering.

## What this means for JEF

**1. `/v1/systemone` is now a de-facto standard, not a compatibility gesture.**
Three independent projects implement it, and Laya's pitch is literally "change
the base URL". JEF already serves it, and the Robot conformance suite is
therefore testing against a real ecosystem contract rather than one vendor's
API. Worth keeping strict.

**2. Competing on "open weights" alone is competing on a solved problem.**
Laya and Von both ship Apache-2.0 weights, self-host cleanly, and are free. That
axis is occupied. JEF's differentiators have to be the two things none of them
do: the **scene layer** (Jev explicitly leaves decision composition to the
caller, and both open models follow it there) and **published calibration on
third-party data**.

**3. Context length is an underrated gap.** Laya is 512 tokens for English and
1,024 multilingual, with a stated state budget of ~320 tokens. A SOAR alert with
log excerpts does not fit in 320 tokens. JEF's mmBERT-base carries 8,192, and
`docs/RESULTS.md` measures the latency curve out to that limit — where the
marginal cost of an extra question is 0.4%. Long state is exactly where the
shared-state architecture pays off, and it is where the alternatives cannot go.

**4. Delivery ergonomics are table stakes, and Laya sets the bar.** Docker, a
CLI, an MCP server, a NixOS module, Node bindings. JEF has a container, a Helm
chart, five IaC stacks, Python and TypeScript SDKs, a Go binary and three SOAR
connectors — but **no MCP server and no CLI**, which are the two cheapest
remaining gaps.

**5. Their numbers are self-reported too.** Von claims 72.0% macro on a 49-task
suite and Laya reports ECE 0.081, both on their own benchmarks. That is the
problem `docs/RESULTS.md` is trying not to repeat, and it is also why a
head-to-head on TMMLU+ and CTI-Bench would say more than either project's
README.

## Sources

- <https://vercel.com/i/what-is-jev>, <https://docs.typesafe.ai/introduction>
- <https://developers.cloudflare.com/ai/models/typesafe/jev/>
- <https://openrouter.ai/typesafe/jev-1.13>, <https://apimodels.app/access/jev-api>
- <https://laya-ai.com/guides/self-host-laya>, <https://systemonemodels.org/models/laya/>
- <https://github.com/receptron/laya>
- <https://github.com/PraiseSinkamba/von>

Retrieved 2026-09-27. Pricing and limits change; re-check before quoting.
