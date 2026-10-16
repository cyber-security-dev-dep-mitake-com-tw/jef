# SOCBench sources

`sources.yaml` is the provenance manifest for the datasets SOCBench may draw on. It records, per source, what the data actually labels (`ground_truth_level`), how it can be obtained (`access`), and the caveat that must survive into every derived case. No dataset is stored in this repository; M1 adapters download into an ignored directory and record SHA-256 digests in `manifests/`.

Rules that apply to every source:

1. The ground-truth level is never upgraded. An attack-phase window, a flow label, a CTF answer key or a partial red-team list is recorded as exactly that; an analyst verdict exists only after the two-reviewer workflow in the roadmap, section 7.3.
2. `Background`, unlabelled, and non-red-team traffic is `unknown`, not benign.
3. Licences and redistribution terms are checked before any derived case leaves the ignored data directory. Several sources require registration or an access request.
4. `zh-TW` and `zh-CN` are reported separately, and a translated case is a paired sample of its original, never an independent ground truth.

See [`docs/UPGRADE-PLAN-2026-10.md`](../../../docs/UPGRADE-PLAN-2026-10.md) section 3 for the priority order and [`docs/JEF-SECURITY-ROADMAP-AND-SOC-BENCHMARK.md`](../../../docs/JEF-SECURITY-ROADMAP-AND-SOC-BENCHMARK.md) sections 5 to 7 for the task definitions.
