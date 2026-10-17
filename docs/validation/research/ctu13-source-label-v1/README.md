# SOURCE-LABEL RESEARCH: CTU-13, `soc.ndr.technique.v1`

> **Evidence class: SOURCE-LABEL RESEARCH.** Labels are CTU-13's own flow labels (Tier C). No person reviewed any case. This is a research result on one public dataset, not SOC validation, and says nothing about JEF: no JEF model trained for this task exists yet.

## What was run

| | |
|---|---|
| Source | CTU-13 (CC-BY), archive `sha256:1f8daeca146131a368b432b2d8625de5e4429bce833f3e4cf4bea8312344ab7f`, verified against the pin |
| Build | `jef-bench build --source ctu-13 --background-rate 0.001`: 9,932 cases, 13 scenario families |
| Dedupe | `jef-bench dedupe`: 21 same-label duplicate states dropped, 0 conflicts → **9,911 cases** ([`dedupe.json`](dedupe.json)) |
| Labels | unknown 2,738 · benign 3,326 · botnet 3,498 · c2 349. Background is `unknown`, excluded from binary metrics, never benign |
| Design | **leave-one-scenario-out**, 13 folds. Rules do not train; logistic regression refitted per fold on the other 12 scenarios |
| Scorers | B0 `rules`, B1 `logreg` (hashed bag-of-words). Zero egress |
| Code | [`run.py`](run.py); results in [`results.json`](results.json) |

### Why not the default split

The seed registered before any scoring (`ctu13-research-v1`) assigned the 13 families 9 / 4 / **0** to train / calibration / test. Re-seeding until a test split appeared would have been choosing the split, so the design switched, before any score was seen, to holding out each scenario once.

## Results (median over 13 held-out scenarios)

| Variant | Scorer | Macro-F1 | Accuracy | ECE | PR-AUC (attack vs benign) | FNR @ 1 % FPR | Worst / best fold macro-F1 |
|---|---|---:|---:|---:|---:|---:|---|
| as built | rules | 0.127 | 0.318 | 0.279 | 0.448 | 1.000 | 0.062 / 0.214 |
| as built | logreg | 0.950 | 0.959 | 0.088 | 1.000 | 0.000 | 0.479 / 0.989 |
| IPs masked | rules | 0.155 | 0.331 | 0.266 | 0.385 | 1.000 | 0.059 / 0.254 |
| IPs masked | logreg | **0.854** | 0.888 | 0.115 | 0.991 | 0.077 | **0.337** / 0.924 |

Mistaken auto-closures (an attack predicted benign at ≥ 0.9): 0 for both scorers in both variants.

## What the numbers mean, and what they do not

1. **The as-built score is mostly address memorisation.** CTU-13 reuses the same hosts in all 13 scenarios: `147.32.84.165` is always botnet or C2, `.164` and `.170` are always benign. A bag-of-words model learns the addresses. Removing `src` and `dst` (and IPv4s in evidence text) drops logreg's median macro-F1 from 0.95 to 0.85 and its worst scenario from 0.48 to 0.34. Any CTU-13 number reported without masking should be read as a leakage measurement.
2. **The masked score may still be optimistic.** Ports, byte counts and timing can also be scenario-specific. Only an evaluation on different networks (another dataset or lab runs) would show generalisation.
3. **Masking exposes label ambiguity.** 1,635 flows were identical apart from their addresses, and **102** had byte-identical behaviour features but different labels; those were dropped, because no rule can say which label is right. A flow summary alone cannot separate them.
4. **C2 is too concentrated for a C2 claim.** 221 of 349 C2 cases come from scenario 9; four scenarios have one or two. Per-fold C2 recall would rest on a handful of cases.
5. **The rules baseline is not tuned for flow summaries** (it keys on words, and these states are numbers and addresses). Its low score is a floor, not a competitor.
6. **13 families** is the unit of independence here, not 9,911 cases. The fold-to-fold spread (0.34 to 0.92 masked) is the honest uncertainty.

## Next steps this enables

- Train and evaluate a JEF candidate under the same leave-one-scenario-out design, IPs masked, labelled the same way.
- Add a second NDR source with different networks before any generalisation claim.
