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
