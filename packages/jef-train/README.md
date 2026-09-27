# jef-train

Corpus building, CPU training, calibration and benchmarking for
[JEF](https://github.com/cyber-security-dev-dep-mitake-com-tw/jef).

```bash
pip install jef-train        # pulls torch and transformers
```

```bash
python -m jef_train.build_corpus --out data/corpus --cve-records 3000
python -m jef_train.fit --corpus data/corpus --out models/jef-v0 \
    --backbone jhu-clsp/mmBERT-base --exclude-source soar_zh
python -m jef_train.bench --model-dir models/jef-v0
```

## Training runs on CPU, and here is why that works

The backbone is frozen, and the head's attention pooling uses the raw query
vectors — the trainable factors act only on the pooled result. So the pooled
context and query vectors are constants, and `jef_train.cache` computes them
once for the whole corpus. An epoch becomes a handful of small matmuls rather
than thousands of transformer forward passes. The tradeoff is explicit: pooling
is fixed, not learned.

## Labels come from taxonomies, not from a language model

MITRE ATT&CK states which tactic each technique serves. A CVSS vector states the
attack vector, the privileges required and the severity band. Those are ground
truth, they are cheaper than generating labels, and they keep the resulting
weights clean to publish.

## Splitting is by evidence, and the build fails if it is not

One CVE yields a severity question and an attack-vector question over an
identical description. Splitting per sample puts that text in train and test at
once. The corpus holds ~12,000 samples over ~3,300 pieces of evidence, so under
a per-sample split most test states have already been read during training — the
first run of this pipeline scored 1.000 on one bucket for exactly that reason.
`assert_no_group_leakage` now fails the build instead.

## What it measures

Accuracy and macro-F1, ECE over the top probability **and** over the calibrated
`p_correct`, Brier, reliability curves, and conformal coverage — with an
explicit warning on any bucket where empirical coverage falls below nominal. A
coverage claim nobody checks is the failure this project exists to stop
shipping.

See [`docs/RESULTS.md`](https://github.com/cyber-security-dev-dep-mitake-com-tw/jef/blob/main/docs/RESULTS.md)
for the numbers, including the runs that were wrong and why.

## License

Apache-2.0
