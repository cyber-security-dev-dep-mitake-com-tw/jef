# NDR head v2: root cause of the mistaken closures, cost-aware training, a measured gate

**Evidence class: SOURCE-LABEL RESEARCH.** CTU-13 flow labels are Tier C. The folds, data, frozen features and fit settings are identical to [v1](../ndr-head-and-transfer-v1/README.md). [`run.py`](run.py) recorded every variant, the gate and the success criteria before any v2 score existed. Results are in [`results.json`](results.json) (sha256 `29d9b14d…`), and the per-case analysis is in [`rca.json`](rca.json) (`bb105635…`). The stored v1 heads reproduce v1's 36 mistaken closures exactly.

## Verdict against the pre-registered criteria

| Criterion (fixed in advance) | `cost` | `cost_hard` |
|---|---|---|
| Fewer mistaken closures than v1 | yes: 36 → **12** | yes: 36 → 12 |
| Lower median miss rate at 1% FPR | yes: 0.374 → **0.266** | yes: 0.374 → 0.281 |
| Median macro-F1 within 0.02 of v1 | **no**: 0.768 → 0.713 | **no**: 0.768 → 0.704 |
| **Improvement** | **not met** | **not met** |
| Close to logreg (within 0.03 macro-F1, no more mistaken closures) | no | no |
| Gate closes no attack on any test fold | no | no |

Cost-aware training does what it was designed to do. Confident benign answers on attacks fall by two thirds, and so does the miss rate at a fixed false-positive budget. The cost is macro-F1, which is lower on 6 of 13 folds than v1. The pre-registered rule says this is not an improvement, so the v2 head is not adopted.

## Root cause of v1's 36 mistaken closures

| Group | Cases | What they are | Training slice, same protocol and port | Logreg on the same cases |
|---|---|---|---|---|
| A | 22 | scenario 1 botnet **DNS** (UDP 53) conversations | 25 benign, 11 botnet, 3 unknown | benign on 21 of 22, below 0.9 |
| B | 12 | single or short **HTTP/HTTPS** botnet flows (TCP 80/443) in scenarios 2, 4, 5, 13 | about 1,400 benign against about 100 botnet | benign on 11 of 12 |
| C | 2 | C2 on TCP 5296 (port never seen in training), and one C2 flow on 443 | none / 471 benign | benign on both |

The cause is the labels more than the model. CTU-13 labels a flow by the host it came from. Every flow from an infected host is "Botnet", including its ordinary DNS lookups and web requests. Once addresses are masked, which G1 requires so a scorer cannot memorise hosts, groups A and B look like benign traffic. Their nearest training neighbours are mostly benign or a benign/botnet mix. Logistic regression called 34 of the 36 benign too. It escapes the "mistaken closure" count only because its probabilities are less peaked: 0.43 to 0.81, under the 0.9 auto-close line.

The cost-sensitive head still makes mistaken closures on 11 of the 36 v1 cases, plus one new case. The hard-positive variant still makes them on 10 of the 36, plus two new cases. Telling these flows apart needs context a single flow summary does not carry: the host's other traffic over a time window, or labels that say which flows are malicious rather than which host sent them.

## A gate fitted on calibration does not survive a new scenario

Each scorer's close gate was fitted on its fold's two calibration scenarios and frozen. It closes only above the lowest threshold whose calibration closures contain attacks at a 95% upper bound of 1% or less. Totals over the 13 test scenarios:

| Scorer | Cases closed | Attacks closed | Folds where an attack was closed | Benign cases spared review |
|---|---|---|---|---|
| logreg | 1,369 | **33** | 4 | 1,272 |
| JEF v1 | 341 | 1 | 1 | 330 |
| JEF cost | 427 | 4 | 1 | 401 |
| JEF cost, requiring an all-benign conformal set | 417 | 3 | 1 | 395 |
| JEF cost_hard | 337 | 1 | 1 | 330 |

No gate is safe. Logistic regression's "zero mistaken closures" turns into 33 closed attacks as soon as a threshold is fitted honestly on other scenarios. Its probabilities are badly calibrated (ECE 0.12), so a threshold learned on two scenarios means something else on a third. The JEF gates abstain on most folds and close little, which is why they close fewer attacks. That is a property of abstention, not of separation. **Auto-close stays prohibited.** A scene now cannot declare a close or suppress action below T2.

## Coverage under scenario shift

| | v1 | cost | cost_hard |
|---|---|---|---|
| Median marginal coverage (target 0.90) | 0.896 | 0.913 | 0.898 |
| Median coverage of attack cases | 0.909 | 0.934 | 0.934 |
| Same, class-conditional (Mondrian) quantiles | 0.869 | 0.857 | 0.842 |
| `coverage_check` verdict on the 13 test scenarios | 5 violated | 5 violated | 4 violated |

`validated` describes the artifact, not the environment. `Calibrator.coverage_check` is the new measurement for the environment, and it flags 4 or 5 of 13 held-out scenarios as clearly below target. Class-conditional quantiles made attack coverage worse, because each class's calibration set is smaller and the shift is the same.

## What follows

1. **Do not tune further on CTU-13 flow summaries.** The remaining errors are mostly cases the labels and features cannot separate. The next round needs host and time-window context in the case, which is an adapter change, and flow-level reviewer labels (manual action 2.2).
2. Keep cost-aware training available: it is off by default and tested. It is the right tool once the labels can say which flows are malicious.
3. Measure coverage with `coverage_check` on labelled data from the environment being served before any prediction set gates anything.
4. Compare against logistic regression with a fitted gate, not with a fixed 0.9 line. On that measure neither scorer is safe to auto-close today.
