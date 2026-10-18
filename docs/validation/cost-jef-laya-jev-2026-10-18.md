# Cost estimate: JEF vs Laya vs official Jev (2026-10-18)

Status: **estimate only, nothing was run.** No call was made to TypeSafe or the Vercel AI Gateway, and no Laya checkpoint was downloaded. NEXT-V3-08 stays pending until someone with an account, a pinned checkpoint and approved egress runs it.

## Prices and limits as published

| | JEF | Laya | Official Jev |
|---|---|---|---|
| Where it runs | this Mac or the lab VM | self-hosted | TypeSafe API, or AI Gateway (`typesafe-ai/jev`) |
| API price | none | none | **$0.042 per 1M input tokens**; output tokens free |
| Gateway markup | n/a | n/a | none ("no markup and no platform fee on tokens") |
| Context | 8,192 tokens per state (backbone setting) | 512 tokens (English checkpoint), 1,024 (multilingual), per this repository's notes | 64k per request, 32k per question (262k with the long-context header, same price); the Gateway page lists 32,000 |
| Versions | pinned head + backbone revision | must be pinned by checkpoint digest | `jev-1.14.0` (preview), `jev-1.13.0` (`jev-latest`) |
| Access | open | Hugging Face repo returned **401** on 2026-10-18: private, gated or gone; must be checked by hand | **early access**, waitlist; rate limits per organisation, numbers not published |
| Data handling | local | local | DPA offered; zero data retention for enterprise only; "not trained on customer requests" |

Sources, read 2026-10-18: `docs.typesafe.ai/models.md`, `docs.typesafe.ai/legal.md`, `typesafe.ai`, the TypeSafe launch post of 2026-09-15, `vercel.com/ai-gateway/models/jev`, `vercel.com/docs/ai-gateway/pricing`. Laya's limits come from [`docs/RESULTS.md`](../RESULTS.md); its model card could not be read.

## What the benchmark sets would cost on official Jev

Input tokens counted with the mmBERT tokenizer on the exact case text every scorer receives (state, evidence, and the task question). Jev's tokenizer is not published; a characters ÷ 4 count is 2.5 to 3.5 times lower on these log-heavy cases, so the figures below are the conservative end.

| Set | Cases | Input tokens | One pass |
|---|---|---|---|
| Pilot batch `pilot-2026-10-v1` | 63 | 25 k | $0.001 |
| Tier A gold set (300, at the pilot's mean of 404 tokens) | 300 | 121 k | $0.005 |
| CTU-13 research set (deduplicated) | 9,911 | 3.35 M | $0.14 |
| AIT-ADS, every window | 19,726 | 16.9 M | $0.71 |
| **Full comparison round**: both sources, both Jev versions, three repeats to check determinism | | 122 M | **$5.11** |

For scale: a shadow run of 10,000 alerts a day at the pilot's mean size is about 121 M tokens a month, about **$5 a month**.

## Laya and JEF: compute, not money

Neither has an API bill. JEF's measured cost on an Apple M3 (MPS) is about 10 distinct case texts a second for the backbone encode (11,871 in 1,195 s during the NDR study); the 30,000 cases above take under an hour of local compute. Laya is assumed to be in the same class until its checkpoint can be inspected.

Laya's context limit decides how much of the benchmark it can see at all:

| Set | Cases over 512 tokens | Over 1,024 tokens |
|---|---|---|
| CTU-13 research set | 13.2 % | 0 % |
| AIT-ADS, every window | 32.5 % | 22.8 % |

A truncated state does not error; it answers from whatever survived the cut. Report Laya only on cases within its limit, or report truncation as its own row.

## The costs that actually decide it

The API bill is not the constraint. These are:

1. **Access to official Jev.** Early access with a waitlist, and rate limits that are only visible after sign-up.
2. **Approval to send data out.** Even public CTU-13 and AIT-ADS cases need `--allow-egress` and a recorded decision. Company SIEM or EDR data needs the DPA, and zero data retention needs an enterprise agreement.
3. **A Laya checkpoint you can pin.** The public repository was not readable today.
4. **Engineer time.** About one day to stand up Laya, run three scorers with `jef-bench evaluate --scorer-config`, and record the result in the ledger.
5. **Labels that make the comparison mean something.** Source-label sets (Tier C and D) can rank systems only as research. The comparison worth reporting runs on the Tier A set, whose cost is reviewer hours (measure them with the pilot), not tokens.

**Recommendation:** budget **US$10** for official Jev API usage. That covers about two full comparison rounds with margin. Treat access, egress approval and reviewer time as the real schedule items.
