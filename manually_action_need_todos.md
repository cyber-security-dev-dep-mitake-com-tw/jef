# Manual actions needed

Everything here needs a person, an account, a contract or a lab. Code and CI cannot do it. Each item says who acts, what to do, what it unblocks, and what to record afterwards. Record results in the [validation ledger](docs/validation/README.md); a claim without an artifact there stays **Pending**.

Last updated 2026-10-18.

## 1. Cost of the JEF vs Laya vs official Jev comparison

Official Jev costs **$0.042 per million input tokens**, and output is free. One full comparison round costs about **$5.11**: both datasets, both Jev versions, three repeats. **Budget US$10.** JEF and Laya have no API cost. Laya's 512 and 1,024 token limits would cut off 13% to 33% of cases. Details and sources: [cost estimate](docs/validation/cost-jef-laya-jev-2026-10-18.md).

The real blockers are not money:

- **Official Jev is in early access with a waitlist.**
- **Sending data out needs an egress approval.**
- **Laya's Hugging Face page returned 401,** so there is no checkpoint to pin yet.
- **Meaningful results need reviewer-labelled data,** which costs reviewer hours.

## 2. Still needs people or setup

| # | Action | Who | Unblocks | Record afterwards |
|---|---|---|---|---|
| 2.1 | **Label the 63-case pilot batch.** Follow [`pilot-2026-10-v1/README.md`](benchmark/socbench/annotation/batches/pilot-2026-10-v1/README.md): regenerate the cases, export blind tasks, label them, and commit only `labels-analyst-anon-01.jsonl` from your own account. | one security engineer | gate G2 (Tier B); measured reviewer hours for the Tier A schedule | `pilot-report.json` and the hours per 100 cases, as SINGLE-REVIEW RESEARCH |
| 2.1b | **Priority: run the pilot now.** It is the cheapest step that produces non-source-label evidence, and it measures reviewer hours for everything below. It needs one qualified engineer and no lab. | one security engineer | G2, scheduling of 2.2 | as 2.1 |
| 2.1c | **Label which flows of an infected host are malicious.** CTU-13 labels hosts. The new `--flow-labels behaviour` mapping uses the authors' behaviour tags (spam, click fraud, connection attempts, ICMP floods, C&C), but plain sessions from a bot stay `unknown` until a person looks. A reviewer batch on `soc.ndr.technique.v1` or `soc.ndr.host.v1` windows gives flow-level or window-level truth. | qualified reviewer(s), via a batch | an NDR result that is not capped by host-level labels | `check-batch` report and the batch's gold |
| 2.2 | **Label the Tier A set (300+ cases)** with two independent reviewers and a third-person adjudicator. Open a *Gold annotation batch* issue per batch; each person commits only their own file. | two qualified reviewers plus an adjudicator; never a second handle, never an LLM | G2 (Tier A), G3, a meaningful JEF/Laya/Jev comparison | `check-batch --git` showing `tier_if_merged: A`; kappa per class |
| 2.3 | **Get official Jev access.** Join the TypeSafe early-access waitlist or ask sales; set `TYPESAFE_API_KEY` (or `AI_GATEWAY_API_KEY`) in your shell only, never in a file. | account owner | the Jev column of the comparison | model version used (`jev-1.13.0` / `jev-1.14.0`), date, organisation rate limits |
| 2.4 | **Approve egress for the comparison.** Allow `api.typesafe.ai` or `ai-gateway.vercel.sh` for **public** cases only (CTU-13, AIT-ADS, the pilot), and record the decision. Company SIEM or EDR data additionally needs the TypeSafe DPA; zero data retention needs an enterprise agreement. | security owner; legal for the DPA | running `jef-bench evaluate --allow-egress` | who approved, which hosts, which datasets |
| 2.5 | **Obtain and pin a Laya checkpoint.** The public repository returned 401 on 2026-10-18. Get access or a copy, record its SHA-256, serve it on loopback, and fill `laya-pinned` in [`scorers.example.json`](benchmark/socbench/scorers.example.json). | ML engineer | the Laya column | checkpoint digest, release tag, hardware |
| 2.6 | **Run the comparison** once 2.3 to 2.5 are done: `jef-bench evaluate --scorer-config scorers.json --allow-egress <host>` on the same locked split for every system. Report Laya only on cases within its context limit, or report truncation as its own row. | ML engineer, about one day | NEXT-V3-08 | the report with `evidence_class`, cost actually spent |
| 2.7 | **Isolated lab, S14:** build the ESXi port group with no physical uplink, run `deploy/lab/egress_check.sh` inside it, and pass the log to `scripts/safety-evidence.py --egress-log`. | infrastructure | gate G0 lab | the safety pack with S14 passing, operator, host, port group |
| 2.8 | **Real EDR / IAM / firewall integration and failure drills** against non-production tenants. The Compose mock SOAR run covers the contract on mocks only. | SOC platform owner | anything beyond shadow mode | receipts, platform references, the drill log |
| 2.9 | **Shadow SOC pilot** on real alerts with human decisions alongside, after 2.1, 2.2 and 2.7. | SOC lead | gate G4 | side-by-side decisions, cost and risk measured |
| 2.10 | **A cross-network dataset with comparable labels.** The CTU-13 to AIT-ADS transfer is near chance, and its binary mapping counts unlabelled events as "no attack", which is a research convenience, not a fact. Request access to a dataset whose labels mean the same thing across networks (CICIDS2017 is behind a request form in [`sources.yaml`](benchmark/socbench/sources/sources.yaml)), or label one with 2.2. | data owner | a transfer result that means something | source, licence, label definition, digests |
| 2.11 | **Let GitHub Actions open pull requests** (Settings → Actions → General → "Allow GitHub Actions to create and approve pull requests"). It is off today, so the benchmark publisher prints a compare link instead. | repository admin | `benchmark-publish` opening its PR itself | none |

## 3. Decision quality before scale

Current positioning: **an engineering-complete research platform that has not yet passed production SOC decision-quality validation.** The research question worth the next investment is whether JEF can use several flows, host context, time and more precise ground truth to judge unseen network scenarios clearly better. That work is mostly data processing, training and the GitHub pipeline. It does not need the physical SOC lab first. The host-window study is in [`ndr-host-window-v1`](docs/validation/research/ndr-host-window-v1/README.md).


The NDR study shows the order: improve JEF's decisions first, and expand SOAR features, the lab or enterprise rollout later. Auto-close and auto-suppress stay prohibited in the product. A scene cannot declare them below T2, and every close goes to a person. Revisit a larger shadow SOC only when JEF is at least close to logistic regression on both safety and performance on the same 13-fold benchmark. Its current standing is in [`ndr-head-v2`](docs/validation/research/ndr-head-v2/README.md).
