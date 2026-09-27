# Results

Every number here is out-of-sample, on a split grouped by evidence, produced by
`python -m jef_train.fit`. The raw artifact is `models/jef-v0/report.json`.

Two things are stated up front because they are what makes the rest worth
reading:

1. **These are self-evaluated numbers.** The ecosystem's credibility problem is
   that everyone reports their own benchmark. JEF's headline numbers are not
   final until the independent third-party evaluations land (TMMLU+, CTI-Bench,
   MMTEB zh) — see *Pending* below. Until then, read this as "what the pipeline
   measures", not "how good the model is".
2. **The first version of these numbers was wrong**, and the way it was wrong is
   more instructive than the numbers themselves.

## The contamination

The first training run reported:

| bucket | accuracy | macro-F1 | ECE |
|---|---|---|---|
| SOAR team routing (`choice:5`) | **1.000** | **1.000** | **0.000** |

A model does not do that. The split did.

Splitting was per sample, and samples are not independent: one CVE produces a
severity question *and* an attack-vector question over an identical description,
and one SOAR scenario produces alerts differing only in hostname and timestamp.
The corpus holds 11,996 samples over **3,268 distinct pieces of evidence**, so
under a per-sample split most test states had already been read during training.

| metric | per-sample split (contaminated) | grouped by evidence |
|---|---|---|
| accuracy | 0.8204 | 0.6013 |
| macro-F1 | 0.7593 | 0.5455 |
| ECE (top probability, calibrated) | 0.0130 | 0.0838 |
| SOAR routing accuracy | 1.000 | 0.193 |

Splitting is now by evidence group — ATT&CK technique id, CVE id, SOAR scenario
template — the corpus build fails outright on contamination, and `fit` re-checks
the property against what it actually loaded.

## The synthetic data was hurting

With a clean split, per-source accuracy separated sharply:

| source | questions | accuracy | random |
|---|---|---|---|
| `cve.user_interaction` | 234 | 0.897 | 0.500 |
| `cve.attack_vector` | 423 | 0.766 | 0.250 |
| `cve.privileges` | 189 | 0.720 | 0.500 |
| `cve.severity` | 423 | 0.598 | 0.200 |
| `attack.tactic` | 64 | 0.594 | **0.071** |
| `soar_zh.routing` | 270 | **0.193** | 0.200 |
| `soar_zh.severity` | 270 | 0.352 | 0.200 |

`soar_zh.routing` at 0.193 against a 0.200 random baseline is not weak
performance; it is no performance. The generator has fifteen scenarios, so
holding a scenario out means the model has never seen that description-to-team
mapping at all. **That data can be memorised, not learned from.**

The ablation (`--exclude-source soar_zh`) settled whether it was merely useless
or actively harmful:

| metric | with synthetic SOAR | without |
|---|---|---|
| accuracy | 0.6013 | **0.7258** |
| ECE (top probability, calibrated) | 0.0838 | **0.0359** |
| `attack.tactic` (14-way) | 0.594 | **0.688** |
| `attack.tactic_noul` | 0.578 | **0.734** |
| `cve.attack_vector` | 0.766 | **0.787** |
| P(correct) ECE on `noul:2` | 0.121 | **0.041** |
| conformal coverage violations | 7 | **1** |

It was harmful. The shipped model excludes it. The generator stays in the
repository because it is genuinely useful for authoring scenes and for
integration fixtures — it is simply not training data.

## Shipped model: `models/jef-v0`

`jhu-clsp/mmBERT-base` (frozen) + bilinear head, rank 64, trained on 6,041
samples, calibrated on 1,858, tested on 1,397. Synthetic sources excluded.

| bucket | n | accuracy | macro-F1 | ECE | T | P(correct) ECE |
|---|---|---|---|---|---|---|
| `choice:14` (ATT&CK tactic) | 64 | 0.688 | 0.592 | 0.153 | 1.16 | *unavailable* |
| `choice:4` (CVE attack vector) | 423 | 0.787 | 0.520 | 0.072 | 1.13 | 0.052 |
| `noul:2` | 487 | 0.813 | 0.808 | 0.035 | 1.15 | 0.041 |
| `score:5` (CVE severity) | 423 | 0.570 | 0.460 | 0.094 | 1.01 | 0.099 |
| **overall** | **1397** | **0.7258** | 0.6055 | **0.0359** | | |

Majority-class baseline: 0.5870. Calibration reduces ECE from 0.0420 to 0.0359.

`choice:14` has no P(correct) map: only 418 ATT&CK samples exist in total, so the
calibration split holds ~84 — below the 150 the isotonic fit requires to be
honest. A bucket with 62 points produced a map whose out-of-sample ECE was
0.235, against 0.046 for one with 290, so the threshold is measured rather than
guessed. **Scene gates on 14-option questions therefore cannot use `p_correct`
and will route to a human.** That is the intended behaviour.

## Where conformal coverage fails, and why

Split-conformal guarantees marginal coverage **under exchangeability**. Grouping
by evidence puts *different concepts* in calibration and test, which breaks that
assumption, and the guarantee goes with it:

| bucket | nominal | empirical @ alpha=0.10 |
|---|---|---|
| `choice:14` | 0.900 | 0.906 ✓ |
| `noul:2` | 0.900 | 0.916 ✓ |
| `score:5` | 0.900 | 0.872 ✗ |
| `choice:4` | 0.900 | 0.861 ✗ |

`fit` flags every shortfall as a warning and `report.json` records
`coverage_violation` per bucket per alpha. This is reported rather than buried
because a coverage claim nobody checks is exactly the failure this project
exists to stop shipping. **Do not gate automated actions on prediction sets from
a bucket flagged here.**

## Two ECEs, and why

`ece` is computed over the top probability. That is the standard definition, the
quantity temperature scaling optimises, and the number comparable to what other
projects publish.

`ece_confidence` is computed over Jev's confidence statistic
`(n·peak − 1)/(n − 1)`, which is a *sharpness* measure on a different scale.
Reporting only that one made calibration look like it was failing — ECE
apparently "rose" from 0.076 to 0.132 — when on the standard definition it more
than halved.

Which exposed the real gap. `confidence >= 0.9` means "the distribution is
peaked", not "right 90% of the time". So the calibrator also fits an isotonic
map from confidence onto observed correctness and exposes `p_correct`, and
scene gates threshold on that. **The ecosystem's standing disclaimer —
*"confidence measures concentration, not correctness"* — is true of the raw
statistic and is the reason this map exists.**

## Reproducing

```bash
python -m jef_train.build_corpus --out data/corpus --cve-records 3000
python -m jef_train.fit --corpus data/corpus --out models/jef-v0 \
    --backbone jhu-clsp/mmBERT-base --exclude-source soar_zh
```

The corpus is rebuilt from public sources (MITRE ATT&CK, NVD) whose contents
change over time, so exact numbers will drift. `data/corpus/manifest.json`
records what went in.

## Latency: what "adding questions is nearly free" actually costs

The plan's criterion was *"ten extra questions add under 20% latency"*. Measured
honestly, that depends entirely on how long the state is, so a single number
would be misleading. `jhu-clsp/mmBERT-base`, 8 threads, median of five runs:

| state chars | tokens | 1 question | 11 questions | growth for +10 | batching speedup |
|---|---|---|---|---|---|
| 116 | 91 | 33 ms | 58 ms | 73.2% | 6.35× |
| 464 | 355 | 71 ms | 93 ms | 29.9% | 8.47× |
| 1,740 | 1,323 | 247 ms | 289 ms | **16.8%** | 9.42× |
| 4,640 | 3,523 | 951 ms | 1,010 ms | **6.1%** | 10.36× |
| 11,600 | 8,192 | 4,264 ms | 4,280 ms | **0.4%** | 10.96× |

The criterion is met from roughly 1,300 tokens of state onward, and the speedup
approaches its 11× ceiling as the state grows. That is the regime JEF is for --
an alert with logs, a CVE description, a case with its observables. For a
one-sentence state the fixed cost is small enough that questions dominate, and
saying so is better than quoting the one row that flatters.

### The measurement found a real bug

The first run of this table showed **189%** growth, not 46%. The state was being
encoded once, as designed, but the option texts were being encoded *per
question*: eleven questions meant eleven separate forward passes over two short
strings each. `encode_count` was 1 the whole time, which is exactly why it did
not catch it -- a shared state encoding followed by per-question work still
reports one state read while costing linearly.

Option encodes are now batched across every question in a request, the same trick
Jev applies across the option batch applied across the question batch as well.
Marginal cost per question fell from 12.3 ms to 2.6 ms. `query_encode_count` is
now exposed alongside `encode_count`, and tests assert twenty questions cost one
batched option encode.

## Independent third-party benchmarks

Everything above is self-evaluated. These are not: the data, the labels and the
task framing all come from published datasets this project did not construct.
Reproduce with `python -m jef_train.bench`; the artifact is `docs/benchmarks.json`.

### CTI-Bench VSP — the task JEF was trained for

`AI4Sec/cti-bench` ships CVE descriptions with their full CVSS v3.1 vector,
which yields exactly the four questions JEF trains on. **Zero CVE overlap**: all
2,397 CVE ids from the training and calibration splits were put in an exclusion
set, and none appeared — the corpus draws from recent publication windows while
CTI-Bench VSP is built from 2024 entries. The check runs every time regardless,
because "there was no overlap last time" is not a property.

| task | n | accuracy | random | vs random | ECE | P(correct) ECE | conformal @ 0.10 |
|---|---|---|---|---|---|---|---|
| attack vector | 200 | **0.840** | 0.250 | 3.4× | 0.079 | 0.075 | **0.895** |
| user interaction | 200 | **0.805** | 0.500 | 1.6× | 0.070 | 0.072 | 0.875 |
| severity band | 200 | 0.435 | 0.200 | 2.2× | 0.111 | 0.231 | 0.810 |
| privileges required | 200 | 0.575 | 0.500 | 1.15× | 0.163 | 0.153 | 0.780 |

Attack vector and user interaction transfer well. Privileges-required barely
beats a coin flip, which is consistent with its in-corpus result and suggests the
signal is genuinely weak in prose rather than that the head failed to find it.

### Where JEF is at chance, and why that is the correct result

| benchmark | n | accuracy | random |
|---|---|---|---|
| TMMLU+ (66 zh-TW subjects) | 528 | 0.254 | 0.250 |
| CTI-Bench MCQ | 300 | 0.317 | 0.250 |

TMMLU+ asks things like *"which of these CIDR statements is wrong?"*, and
CTI-Bench MCQ asks which ATT&CK mitigation covers a behaviour. Both are
**knowledge** questions. JEF is a small trained head over a frozen encoder: it
judges evidence you supply, it does not know things. Reported here because a
capability boundary stated plainly is worth more than one discovered in
production.

### Conformal coverage behaves exactly as the theory says

| data | relationship to calibration | coverage @ alpha=0.10 |
|---|---|---|
| CTI-Bench VSP, attack vector | same task, third-party data | **0.895** (nominal 0.900) ✓ |
| CTI-Bench VSP, user interaction | same task, third-party data | 0.875 |
| CTI-Bench MCQ | different task entirely | **0.440** ✗ |

Split conformal guarantees coverage **under exchangeability**. On third-party
data drawn from the same task, coverage lands within a point of nominal —
on an independently constructed dataset, which is the strongest evidence in this
document. On a different task it collapses to 0.44, because the guarantee never
applied there.

This is the practical rule the scene layer encodes: prediction sets are
trustworthy for the question type the calibrator was fitted on, and `fit` and
`bench` both flag every bucket where empirical coverage falls short.

## Pending

- **`jef-bench-zh-tw`** — a human-verified zh-TW evaluation set. Two thirds of
  the current corpus's *states* are English prose from ATT&CK and NVD; every
  *question* is zh-TW. TMMLU+ shows JEF is not a zh-TW knowledge model, which
  was never the claim; what is still unmeasured is zh-TW **evidence** judgement,
  and that needs a human-labelled set.
- Comparison against Laya and Von on the same independent sets.
