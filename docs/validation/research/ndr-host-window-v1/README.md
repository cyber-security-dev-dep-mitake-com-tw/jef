# NDR: host windows and behaviour labels (v1)

**Evidence class: SOURCE-LABEL RESEARCH** (CTU-13, Tier C). [`run.py`](run.py) recorded both arms, every scorer and the success criteria before any score on this data existed. Results: [`results.json`](results.json) for host windows (sha256 `24d7540c…`) and [`results-flow-behaviour.json`](results-flow-behaviour.json) for behaviour-labelled conversations (`e4867df5…`). The folds, cost-weight selection, gate and coverage check are those of [v2](../ndr-head-v2/README.md).

The question: can JEF judge unseen network scenarios clearly better when the case carries host context over time, and when the ground truth means what it says?

## The two changes

1. **Behaviour labels for conversations** (`jef-bench build --flow-labels behaviour`, now the default). CTU-13 tags a bot's flows with behaviour: C&C, spam, click-fraud ads, connection attempts, ICMP floods, DNS, plain sessions. The old mapping called all of them `botnet`, while the labelbook defines `botnet` as spam, scanning, click fraud and floods. Under the new mapping a bot's DNS lookups and plain sessions are `unknown`, and a conversation with any bot flow is never `benign`.
2. **Host windows** (`--unit host-window`, task `soc.ndr.host.v1`). One source host over 300 seconds, with addresses masked. Its features are fan-out, port mix and entropy, DNS/SMTP/IRC volume, unanswered share and periodicity. The gold is host-level (`host-label`): `botnet` means the host is infected, and `c2` means a C&C-tagged flow appears in the window. The gold note keeps the real address and the archive row of every evidence flow.

## Results, median over 13 held-out scenarios

| | v1: conversations, host labels | **Conversations, behaviour labels** | **Host windows, host labels** |
|---|---|---|---|
| Cases (after dedupe) | 8,174 | 7,976 | 2,898 |
| JEF cost-aware macro-F1 | 0.713 (v2) | **0.781** | 0.559 |
| JEF cost-aware miss rate at 1% FPR | 0.266 (v2) | 0.098 | **0.069** |
| JEF cost-aware PR-AUC | 0.971 (v2) | 0.991 | **0.997** |
| JEF mistaken closures per 1,000 attacks | 13.6 (v1) | 4.9 | **1.9** |
| logreg macro-F1 | 0.849 | **0.870** | 0.641 |
| logreg miss rate at 1% FPR | 0.087 | **0.029** | 0.500 |
| Calibration-fitted gate: attacks closed, JEF cost / logreg | 4 / 33 (v2) | **0 / 43** | 3 / 30 |
| `coverage_check` holds, JEF cost | 8 of 13 (v2) | **10 of 13** | 5 of 13 (3 too few) |

## Verdicts against the pre-registered criteria

| Criterion | Conversations, behaviour labels | Host windows |
|---|---|---|
| 1. Better than the single-flow head v1 (lower miss rate, lower mistaken-closure rate, macro-F1 ≥ 0.768) | **met** | not met (macro-F1 0.559) |
| 2. Close to logreg on the same cases (macro-F1 within 0.03, miss rate within 0.05, no more mistaken closures) | not met | not met |
| 3. Gate closes no attack on any test fold | **met for both JEF heads** (logreg's gate closed 43) | not met |

## What this says

- **v1's confident errors were mostly the labels.** Labelling a bot's conversations by behaviour, as the labelbook does, takes the same JEF architecture past v1 on every pre-registered measure. Mistaken closures per 1,000 attacks fall from 13.6 to 4.9. Its probabilities are well calibrated: ECE 0.050, and coverage holds in 10 of 13 scenarios. A gate fitted on calibration scenarios closed no attack on any held-out scenario, which v2 could not achieve.
- **Host context makes infection detection much sharper, but not the four-way verdict.** On host windows the cost-aware head misses 6.9% of infected windows at 1% false positives, where logreg misses 50% and v1 37.4%. Pooled over folds, infected-versus-benign recall is 0.79 with 3.8% of benign windows flagged; this diagnostic was computed after the run and is not a criterion. Macro-F1 stays low mostly because of 235 swaps between `c2` and `botnet` (a C&C flow present in this window or not) and 167 attack windows answered `unknown`. Neither closes anything.
- **JEF is still not close to logistic regression.** On behaviour-labelled conversations logreg is better on macro-F1 and miss rate and makes no confident benign errors. JEF still makes 12, concentrated in scenarios 12, 2, 4 and 5. Logreg's probabilities, however, cannot be gated safely (43 closed attacks), and JEF's can.
- Some scenarios are hard for every scorer (12 for conversations; 8, 10 and 12 for host windows, where every scorer misses most infected windows at 1% FPR). These are where the next labels should come from.

## Next, in order

1. Keep `--flow-labels behaviour` as the default for every new NDR result.
2. Combine the two: conversation evidence inside a host window, so the case keeps the behaviour tags and gains the host context. Pre-register it, with the infected-versus-benign miss rate as a primary criterion beside macro-F1.
3. Reviewer labels on the scenarios every scorer fails, through an annotation batch (manual action 2.1c).
4. Auto-close stays prohibited. A gate that closed no attack on 13 scenarios of one 2011 dataset is a reason to measure more, not to automate.
