# JEF NDR head, schema-2 calibration, and cross-dataset transfer (v1)

**Evidence class: SOURCE-LABEL RESEARCH.** CTU-13 flow labels are Tier C and AIT-ADS attack-phase windows are Tier D. Nothing here is a reviewer label, and none of it may unlock automation. [`run.py`](run.py) fixes the design, written before any score was seen. [`results.json`](results.json) holds every number below. Its sha256 is `5b75839c…0b926a`, and it was produced on an Apple M3 from the frozen `jhu-clsp/mmBERT-base` at revision `c5955035…`.

## A. A JEF head for NDR, scenario held out

The test set is the IP-masked, deduplicated CTU-13 set from G1: 8,174 cases in 13 scenarios. Each fold holds out one scenario for test, calibrates on the next two, and trains on the other ten. All three scorers are scored by the same `jef-bench evaluate` call.

| Median over 13 folds | rules | logreg | JEF head |
|---|---|---|---|
| macro-F1 | 0.155 | **0.849** | 0.768 |
| PR-AUC (attack vs not) | 0.385 | **0.990** | 0.968 |
| Miss rate at 1 % false positives | 1.00 | **0.087** | 0.374 |
| ECE | 0.266 | 0.122 | **0.052** |
| Worst fold, macro-F1 | 0.059 | 0.333 | 0.357 |
| **Mistaken closures, all folds** | 0 | **0** | **36** |

- **The head does not beat character n-grams.** It wins 6 of 13 folds and loses the median by 0.08 macro-F1. Its worst folds are scenarios 10 and 7, where it scores 0.357 and 0.517 against logistic regression's 0.895 and 0.827.
- **It closes attacks.** 36 cases labelled botnet or C2 received a confident benign verdict, at or above the auto-close threshold; 23 of them were in scenario 1. Logistic regression made none. This alone rules the head out of any automated closing.
- **Its probabilities are honest.** It has the lowest ECE of the three. A confident answer is right about as often as it claims, and the failures above sit in the minority it is unsure about.

## B. Task-scoped calibration (schema 2)

Every fold's calibration was fitted by `jef_train.fit` on its two calibration scenarios only, in the bucket `soc.ndr.technique.v1|choice:6`.

| | Value |
|---|---|
| Bucket status, all 13 folds | `validated` |
| Median calibration records per fold | 1,252 (G3 asks for at least 150) |
| Test ECE, raw / after temperature | 0.049 / 0.052 |
| Conformal coverage at alpha 0.10, median | 0.896 (target 0.90) |
| Folds below target coverage | 7 of 13 (worst 0.577 and 0.610) |

The machinery works end to end. Fitted temperatures sit near 1 because the head is already calibrated. Coverage drops wherever a held-out scenario looks unlike the calibration scenarios. The conformal guarantee assumes exchangeability, and family shift breaks it. A deployment must calibrate on data from the environment it serves, and must re-check coverage when that environment changes.

The cached-score path matches the real engine on the first fold's test cases to within **0.0016** probability.

## C. Transfer between datasets

Both datasets get one question: *"Does this evidence show attack activity?"* CTU-13 botnet and C2 cases, and AIT-ADS windows inside an attack phase, count as `attack`. Every other case counts as `no_attack`, including unlabelled cases, which are not thereby benign. The AIT-ADS set is all 497 attack windows plus 400 unlabelled windows per scenario, 3,697 cases in total.

| Train → test | Prevalence | logreg PR-AUC / ROC-AUC | JEF PR-AUC / ROC-AUC |
|---|---|---|---|
| CTU-13 → AIT-ADS | 0.134 | 0.156 / 0.541 | 0.143 / 0.504 |
| AIT-ADS → CTU-13 | 0.324 | 0.299 / 0.456 | **0.473 / 0.649** |
| CTU-13 in domain (leave one scenario out) | | 0.970 | 0.894 |
| AIT-ADS in domain (leave one scenario out) | | 0.285 | 0.222 |

- **Nothing transfers from flows to alerts.** CTU-13 → AIT-ADS is at chance for both scorers.
- **The frozen backbone carries a little signal the other way.** JEF reaches ROC-AUC 0.65 from alerts to flows, where n-grams fall below chance. That is a hint, not a result.
- **AIT-ADS window labels are weak even in domain.** A time window marks many routine alerts as attack-phase, so neither scorer gets past PR-AUC 0.29 at a base rate of 0.13. Tier D labels can rank systems roughly; they cannot validate one.

## What this changes

- G1 now has a second dataset and a JEF scorer. The head is **not** better than a simple baseline on held-out scenarios, and it makes confident benign mistakes the baseline does not.
- G3's calibration machinery is shown on unseen families. G3 itself stays **Pending**: it needs the same run on Tier A labels.
- Before training on any further data, train with the cost of a missed attack in the loss, or keep `benign` from ever auto-closing. Then rerun this study unchanged and compare.

## Reproduce

See the docstring in [`run.py`](run.py). The first run encodes 11,871 distinct case texts once, which took 20 minutes on an M3 (1,195 s for 20,045 samples). `SMOKE=1` runs the whole pipeline on the hashing backbone in about two minutes.
