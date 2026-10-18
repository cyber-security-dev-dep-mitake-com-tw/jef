# NDR fusion: behaviour gold + host context, and transfer to IoT-23 (v1)

**Evidence class: SOURCE-LABEL RESEARCH** (CTU-13 and IoT-23, Tier C). [`run.py`](run.py) recorded the cases, folds, scorers and all nine criteria before any score on these cases existed. Results: [`results.json`](results.json) (sha256 `f6475f43…`). Leakage audit: [`audit-ctu-13-fusion.json`](audit-ctu-13-fusion.json) (`88cd8147…`). The plan is in [`../../plans/2026-10-18-fusion-and-second-source.md`](../../plans/2026-10-18-fusion-and-second-source.md).

## What was built

- **Fusion cases** (`jef-bench build --unit host-fusion`): one host over 300 s, with addresses masked.
  - Gold follows behaviour inside the window: `c2` for a C&C flow, `botnet` for spam, ads, attempts or floods, `unknown` for an infected host's quiet window, `benign` for a verified-normal host. The label origin is `host-behaviour-label`.
  - Evidence is conversation summaries plus a per-minute series of flows and destinations.
  - The behaviour tags decide the gold only and never reach the model; a test checks this.
  - 2,902 CTU-13 windows remain after deduplication.
- **IoT-23** (`jef-bench build --source iot-23`): 10 pinned captures, chosen by a size rule fixed before any score (2 benign honeypots, 8 malware). The cases are built the same way. 1,601 windows remain after deduplication.
- **Leakage audit** (`jef-bench audit`), reproduced in the table below.

## Leakage audit (CTU-13 fusion)

| Check | Result |
|---|---|
| Held-out cases whose host also appears in a training scenario (median over scenarios) | **0.91** |
| Infected and verified-normal hosts that occur in more than one scenario | all but one normal host |
| Coarse near-duplicate of a case elsewhere with the same label / only other labels | 0.045 / 0.003 |
| `c2` windows decided by a single C&C flow | 58 of 712 (median 5 C&C flows per window) |
| Window overlap | not a leak: one window per flow, one fold per scenario |

CTU-13 cannot hold out unseen attack or benign hosts, because the same hosts recur in every scenario. Every CTU-13 number below may include a host-habit effect. IoT-23 shares no host with CTU-13, which makes it the only unseen-host test here.

## CTU-13, 13 held-out scenarios (median)

| | logreg | JEF | **JEF cost-aware** |
|---|---|---|---|
| Macro-F1 | 0.569 | 0.509 | **0.585** |
| Miss rate at 1% FPR (infected vs benign) | 0.524 | 0.500 | **0.238** |
| PR-AUC | 0.927 | 0.827 | **0.942** |
| Attack windows in each scenario's top 10% | 0.255 | 0.236 | **0.400** |
| Mistaken closures per 1,000 attack windows | **0** | 2.9 | 4.8 |
| Attacks closed by the calibration-fitted gate | 22 | **1** | 11 |
| `coverage_check` holds / violated / too few | — | 4 / 6 / 3 | 4 / 6 / 3 |

## IoT-23, trained on CTU-13 only

| | logreg | JEF cost-aware |
|---|---|---|
| PR-AUC, attack vs benign windows | 0.994 | 0.991 |
| Miss rate at 5% FPR | **0.222** | 0.366 |
| Mistaken closures | 0 | 0 |
| Gate (fitted on CTU-13 calibration) | closes nothing | closes nothing |
| Conformal coverage at alpha 0.10 | — | **0.32**, violated |

## Verdicts against the pre-registered criteria

| Criterion | Result |
|---|---|
| C1 macro-F1 ≥ 0.70 and within 0.03 of logreg | **not met** (0.585; it is above logreg's 0.569, but below the bar) |
| C2 miss rate at 1% FPR ≤ 0.10 | **not met** (0.238) |
| C3 ≤ 2.0 mistaken closures per 1,000 attacks | **not met** (4.8) |
| C4 coverage holds in ≥ 75% of judged scenarios | **not met** (4 of 10) |
| C5 gate closes no attack | **not met** (11) |
| C6 prioritisation at least logreg's | **met** (0.40 against 0.25) |
| T1 IoT-23 PR-AUC ≥ 0.80 | **met** (0.991) |
| T2 IoT-23 miss rate at 5% FPR ≤ 0.30 | **not met** (0.366) |
| T3 IoT-23 gate closes no attack | met, trivially: the gate closed nothing at all |

## What this says

- **Host context is a prioritisation signal, not a verdict.** The cost-aware head ranks infected windows far better than logreg: 40% of them land in each scenario's top 10%, against 25%, and its miss rate is less than half of logreg's. It does not reach the decision-quality, coverage or gate-safety bars. This matches the review's reading. The first use of host windows is host compromise prioritisation, not four-way adjudication or auto-close.
- **Behaviour gold makes the host task harder, and more honest.** Against host-window v1, which used host labels, the miss rate rose from 0.069 to 0.238. Infected hosts' quiet windows are now `unknown`, and `botnet` needs an attack in the window. The earlier figure partly measured "this host is a bot" through host habits that recur across scenarios, as the audit shows.
- **Unseen hosts on another network can be ranked, not trusted.** IoT-23 PR-AUC is 0.99 for both scorers. Its benign side is only two honeypot captures, 130 windows, so that separation is probably easier than a real network's would be. The miss rate at 5% FPR (0.37) and the collapsed coverage (0.32) say calibration fitted on CTU-13 means nothing on IoT-23. The calibration must come from the environment being served.
- **Results vary widely by scenario.** The cost-aware head collapsed on scenario 9, missing 99.5%, where the plain head did well. One fold's choice of cost weight can swing a scenario. More scenarios, or more networks, would steady the medians; tuning on these 13 would not.

## Next

1. **Human labels before more tuning.** Run the 63-case pilot (manual action 2.1b), then label host windows on scenarios 7, 8, 10 and 12, where every scorer fails.
2. **Prioritisation as its own product question.** Pre-register a host-ranking evaluation, with recall at a fixed analyst budget per scenario and per network, as the primary metric for host windows.
3. **More networks.** Add the rest of IoT-23 (the captures above 200 MB) and a non-Stratosphere source. Fit and check calibration per network with `coverage_check`.
4. Auto-close stays prohibited.
