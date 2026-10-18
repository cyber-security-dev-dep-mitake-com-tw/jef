# Pilot batch `pilot-2026-10-v1` (single reviewer, Tier B)

63 cases for the 50–65 case single-reviewer pilot in [LABELBOOK §8](../../LABELBOOK.md): 39 CTU-13 flow conversations (13 scenarios × 3, task `soc.ndr.technique.v1`) and 24 AIT-ADS alert windows (8 scenarios × 3, task `soc.alert.verdict.v1`). `batch.json` pins them by digest. The case text is not in this repository; regenerate it locally from the pinned public archives.

## 1. Regenerate the cases (once, about 15 minutes)

```bash
uv run jef-bench fetch --source ctu-13 --source ait-ads
uv run jef-bench build --source ctu-13 --background-rate 0.001 --flow-labels host
uv run jef-bench build --source ait-ads
uv run jef-bench dedupe \
  --cases .cache/socbench/derived/ctu-13/cases.jsonl --gold .cache/socbench/derived/ctu-13/gold.jsonl \
  --out-cases .cache/socbench/research/ctu13-v1/cases.jsonl --out-gold .cache/socbench/research/ctu13-v1/gold.jsonl
P=.cache/socbench/pilot/pilot-2026-10-v1
uv run jef-bench sample --cases .cache/socbench/research/ctu13-v1/cases.jsonl \
  --weak-gold .cache/socbench/research/ctu13-v1/gold.jsonl --per-family 3 --seed pilot-2026-10-v1 --out $P/ctu13.jsonl
uv run jef-bench sample --cases .cache/socbench/derived/ait-ads/cases.jsonl \
  --weak-gold .cache/socbench/derived/ait-ads/gold.jsonl --per-family 3 --seed pilot-2026-10-v1 --out $P/ait-ads.jsonl
cat $P/ctu13.jsonl $P/ait-ads.jsonl > $P/cases.jsonl
uv run jef-bench annotate check-batch --batch benchmark/socbench/annotation/batches/pilot-2026-10-v1 --cases $P/cases.jsonl
```

The last command must print `ok`: it proves your cases are the pinned ones. The sampler never writes the source's weak label into `cases.jsonl`, so you label blind.

## 2. Label

```bash
uv run jef-bench annotate export --cases $P/cases.jsonl --annotator analyst-anon-01 --out $P/tasks-analyst-anon-01.jsonl
```

Fill in each task's `verdict`, `confidence`, `evidence_ids` and `seconds_spent` as the labelbook says. Keep `unknown` and `insufficient_evidence` when that is the honest answer. Then keep **only the label fields** and save them as `labels-analyst-anon-01.jsonl` in this directory:

```bash
python3 -c "import json,sys; keep={'case_id','annotator','verdict','confidence','evidence_ids','prohibited_actions','required_followup','note','seconds_spent'}; [print(json.dumps({k:v for k,v in json.loads(l).items() if k in keep}, ensure_ascii=False)) for l in open(sys.argv[1])]" \
  $P/tasks-analyst-anon-01.jsonl > benchmark/socbench/annotation/batches/pilot-2026-10-v1/labels-analyst-anon-01.jsonl
```

Commit that one file from your own account and open a pull request. The `annotation batch` check validates it and posts the result.

## 3. Report

```bash
uv run jef-bench annotate pilot-report --cases $P/cases.jsonl \
  --labels benchmark/socbench/annotation/batches/pilot-2026-10-v1/labels-analyst-anon-01.jsonl \
  --weak-gold <(cat .cache/socbench/research/ctu13-v1/gold.jsonl .cache/socbench/derived/ait-ads/gold.jsonl) \
  --hourly-rate <your rate> --out pilot-report.json
```

Record the report in the validation ledger as **SINGLE-REVIEW RESEARCH** (gate G2). The source-label check in it is not inter-annotator agreement.

The handle `analyst-anon-01` is a placeholder. Change it in `batch.json` before labelling if your team uses another numbering.
