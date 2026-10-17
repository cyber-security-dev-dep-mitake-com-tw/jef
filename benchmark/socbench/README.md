# SOCBench

The benchmark data layer for JEF's SOC tasks. This directory holds **provenance and contracts, never data**: which datasets exist, what each one actually labels, how it may be obtained, the SHA-256 of what was last fetched, and the JSON schemas every derived case and gold record must satisfy.

```
benchmark/socbench/
├── sources/sources.yaml   # provenance manifest (upgrade plan §3)
├── manifests/<id>.json    # artifact pins: url, size, upstream checksum, our sha256
├── schema/
│   ├── sources.schema.json    # shape of sources.yaml
│   ├── artifacts.schema.json  # shape of manifests/*.json
│   ├── case.schema.json       # a benchmark input (no verdict)
│   └── gold.schema.json       # the answer key, with label_origin
└── README.md
```

The code lives in `packages/jef-train/src/jef_train/socbench/` and is exposed as `jef-bench`.

## Commands

```bash
uv run jef-bench sources                      # every source, access status, priority
uv run jef-bench fetch --source ts-bench --pin   # download, hash, record the digest
uv run jef-bench fetch --all-public --pin     # every fetchable source
uv run jef-bench check                        # offline: cache vs pins, no network
uv run jef-bench pin --source lanl-cyber1 --local   # digests for files a human placed
```

Downloads go to `JEF_SOCBENCH_DATA` (default `.cache/socbench/<id>/`), which is gitignored. `check` exits 1 on a `MISMATCH`; `fetch --source X` exits 2 when `X` is a restricted source.

## From files to cases

```bash
uv run jef-bench fetch --source ait-ads --pin      # ~96 MB, CC-BY-4.0
uv run jef-bench build --source ait-ads            # -> .cache/socbench/derived/ait-ads/
uv run jef-bench fetch --source ctu-13 --pin       # ~1.9 GB, CC-BY
uv run jef-bench build --source ctu-13 --background-rate 0.001
```

`build` writes `cases.jsonl` (what a model sees, no verdict), `gold.jsonl`
(the answer key) and `summary.json`, all validated against the schemas, into
the ignored data directory. Case ids carry the source (`socb-ait-000001`,
`socb-ctu13-000001`), and `family_id` is the scenario or capture, so a split
grouped by family keeps an attack chain or a botnet capture on one side.

| Source | Case | Gold | Tier |
|---|---|---|---|
| AIT-ADS | all alerts on one host in one 10-minute bucket of one scenario | `suspicious` + `attack_stage` when an alert falls inside a `labels.csv` phase window, else `unknown`; never `confirmed_malicious` (DATA-002) | D |
| CTU-13 | one conversation: same capture, source, destination, port, protocol | `c2` (From-Botnet with CC), `botnet` (other From-Botnet), `benign` (Normal), `unknown` (Background, To-Botnet, or disagreeing labels); never benign for Background (DATA-001) | C |

AIT-ADS event times follow the dataset authors' `analyze.py`: Suricata's
`data.timestamp`, then Wazuh's `predecoder.timestamp`, then the timestamp in
`full_log` for the log file named in `location`; AMiner's last
`DetectionTimestamp`. Wazuh's top-level `timestamp` is the replay time and is
never used. Alerts with no recoverable event time are counted in
`summary.parse.untimed`, not guessed. The host is `agent.ip` (every Wazuh
record names the same replaying agent, `wazuh-client`), which is also
AMiner's `AMiner.ID`; the display name is the most common
`predecoder.hostname` for that IP. Unlabelled cases with an alarming alert
(Wazuh level 10 or more that is not a counting rule such as "Multiple IDS
alerts for same id", or Suricata severity 1 to 2) are listed as
`hard_negative_candidates` for review, at most 20 per signature per
scenario, with every signature's total in `hard_negative_signatures`.
Nothing is marked a hard negative without review.

On the published archive (sha256 `9f0595c4…`, checked 2026-10-17) the
build reads all 2,655,821 alerts with none untimed, and makes 19,726 cases
over 94 hosts: 497 `suspicious` (2.5%) and 19,229 `unknown`, with 309
hard-negative candidates. It takes about 70 seconds and 1.4 GB of memory.
Read a few in-window cases before trusting the label: inside a `dirb`
window the alerts can be nothing but DNS lookups, and inside a `dnsteal`
window nothing but successful mail logins.

On the published archive (sha256 `1f8daeca…`, checked 2026-10-17) CTU-13
reads 19,976,700 flows in all 13 scenarios (19.18 M unknown, 438,678
botnet, 355,165 benign, 6,021 C&C) and, at `--background-rate 0.001` and
the default cap, makes 9,932 cases: 349 `c2`, 3,500 `botnet`, 3,345
`benign`, 2,738 `unknown`. It takes about 7 minutes (bzip2 is read end to
end) and under 300 MB of memory. Real C&C is not a tidy beacon: the median
interval coefficient of variation of C&C conversations with five or more
flows is 0.92.

CTU-13 keeps every conversation that has a Botnet or Normal flow and samples
Background-only conversations by a hash of their key (`--background-rate`),
so rebuilds pick the same ones, and caps each verdict per scenario
(`--max-cases-per-class`, 300) so no single capture dominates a class. `summary.json` reports known-label and
unknown-label case counts separately; report them separately too.

Trust tiers (review section 9.4): A two analysts adjudicated, B one analyst,
C a source's own ground truth, D a weak label, E synthetic or LLM-authored.
Publish numbers per tier.

## From weak labels to gold

Weak labels (tier C or D) are where a case starts, never where it ends. The
two-reviewer workflow (review section 10.3) turns them into tier A or B gold:

```bash
uv run jef-bench annotate export --cases cases.jsonl --annotator analyst-anon-01 --out a-tasks.jsonl
uv run jef-bench annotate export --cases cases.jsonl --annotator analyst-anon-02 --out b-tasks.jsonl
# each reviewer fills in verdict, confidence, evidence_ids, prohibited_actions, note
uv run jef-bench annotate agree --cases cases.jsonl --labels a.jsonl --labels b.jsonl \
    --out agreement.json --queue queue.jsonl
# a third reviewer decides each queued case, with a written reason
uv run jef-bench annotate adjudicate --cases cases.jsonl --labels a.jsonl --labels b.jsonl \
    --decisions decisions.jsonl --out gold.jsonl
uv run jef-bench lock --cases cases.jsonl --gold gold.jsonl --out lock.json --pin ait-fox=test
uv run jef-bench lock --verify lock.json --cases cases.jsonl --gold gold.jsonl
```

- **Blind.** An exported task has the state, the evidence and the allowed
  labels; never the source's weak gold, the other reviewer's label or a
  model's answer. Each reviewer gets their own order.
- **Agreement** is percent agreement and Cohen's kappa per task and per class,
  with the class counts beside it, because kappa is unstable under imbalance.
- **Origin follows what happened.** Two reviewers who agree, or a third who
  adjudicates with a written reason, give `double-human-reviewed` (tier A);
  one reviewer gives `single-human-reviewed` (B); an undecided disagreement is
  `disputed` with verdict `unknown`. An adjudicator who reviewed the case is
  refused. The schema refuses `double-human-reviewed` with fewer than two
  annotators (DATA-005).
- **The lock** assigns splits per `family_id` (seeded hash, `--pin` to place a
  family by hand: with eight AIT scenarios a hash alone may not reach every
  split), and refuses to lock when a case lacks gold, a family spans splits,
  a `paired_with` partner sits in another split (DATA-003), or two splits
  share an identical state. It records the SHA-256 of the case and gold files,
  so `--verify` detects any edit, and reports counts per verdict, tier, task
  and language, and the hard-negative share of negatives (target 35 to 45
  percent).

## Rules the code enforces

1. **Restricted sources are never downloaded.** `request-form` and `registration-or-contact` sources (LANL, CICIDS2017, the CAC dataset) raise before any network call and print how a human applies. The pipeline does not fill forms or scrape around gates: the gate is the owner's condition of use.
2. **https only, atomic writes.** The download loop is `jef_train.corpus.sources.fetch_to`, shared with the training-corpus builder.
3. **Everything is hashed.** A pinned SHA-256 that disagrees with what arrived is a `MISMATCH`; the file is kept for inspection and the exit code is non-zero. An unpinned artifact is only recorded with `--pin`, so a first fetch is a deliberate act.
4. **No label leaves this layer.** Adapters in item 08 inspect shape only (zip members, CSV rows). Label parsing arrives with items 09 to 11, each with tests on provenance: AIT-ADS windows stay `attack-phase-window`, CTU-13 `Background` becomes `unknown`, BOTS v3 cases come from investigation questions.

## Sources that need a human (as of 2026-10-16)

| Source | Why | What to do |
|---|---|---|
| `lanl-cyber1` | request form (email, intended use) | fill it at csr.lanl.gov, place the `.txt.gz` files, `pin --local` |
| `cicids2017` | download page is a form | fill it at cicresearch.ca, place the zips, `pin --local` |
| `cac-cyber-dataset-1.0` | registration or contact; terms unknown | apply via the portal; keep files out of shared artifacts until licence review |
| `chiphish` (full set) | Zenodo record is restricted | the 20-sample public subset fetches; request the rest through the record's form |
| `trec06c` | upstream URL 404s | obtain from the track organisers, place, `pin --local` |

## Scheduled pulls

Optional, and only for `public-download` sources. A cron that re-fetches without pins would silently accept a changed upstream, so a scheduled job should run `jef-bench check` after `fetch` and treat `MISMATCH` as a page, not a refresh. Two sources change legitimately: `twcert-cc` is a monthly statistics release, and GitHub archive bytes (`otrf-security-datasets`, `chiphish`) can change while the pinned commit does not. The manifest notes say what to verify before re-pinning. Nothing in CI downloads a dataset.

See `docs/UPGRADE-PLAN-2026-10.md` §2 (M1) and §3, and `docs/JEF-SECURITY-ROADMAP-AND-SOC-BENCHMARK.md` §7 for the task definitions the schemas encode.
