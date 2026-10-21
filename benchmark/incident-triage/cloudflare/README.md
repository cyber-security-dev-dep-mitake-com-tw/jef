# Your own Cloudflare zone as an incident-triage source

This pipeline turns your own zone's WAF / security events into **private, unlabelled** incident-triage cases that a person can review. It is read-only towards Cloudflare and never puts raw data in this repository.

## What it is and is not

- A Cloudflare `block`, `managed_challenge` or attack score is a **detector output** (Tier D). An `allow` that raised no event is **unlabelled** (U), not benign. Neither is ever a label.
- `firewallEventsAdaptive` may be **sampled**, and one request can raise several events. Event counts are not request counts, and no rate over all traffic can be computed from them.
- One zone is one organisation. Results on it are not a cross-organisation claim.

## Steps

```bash
# Once: a private directory outside any git checkout, and a key only you hold.
export PRIVATE_JEF_DATA=~/jef-private            # never inside a repository
export JEF_PSEUDONYM_KEY="$(openssl rand -hex 32)"  # keep it in a password manager, not a file here
export CF_ANALYTICS_TOKEN=...                    # Analytics Read, this zone only, with an expiry
export CF_ZONE_ID=...

# 1. Pull (read-only). Resumable: a rerun skips complete windows.
uv run jef-bench triage ingest-cloudflare --token-env CF_ANALYTICS_TOKEN --zone-id-env CF_ZONE_ID \
  --zone-alias site-a --start 2026-10-19T00:00:00Z --end 2026-10-20T00:00:00Z \
  --out-private "$PRIVATE_JEF_DATA/cloudflare/raw"

# 2. Pseudonymise and fold events into requests.
uv run jef-bench triage normalize-cloudflare --input "$PRIVATE_JEF_DATA/cloudflare/raw" \
  --output "$PRIVATE_JEF_DATA/cloudflare/normalized"

# 3. Incidents -> an unlabelled pack (optionally with approval evidence and vendor-signal variants).
uv run jef-bench triage build-cloudflare-cases --input "$PRIVATE_JEF_DATA/cloudflare/normalized" \
  --out "$PRIVATE_JEF_DATA/cloudflare/pack" --zone-alias site-a \
  --context "$PRIVATE_JEF_DATA/cloudflare/context.jsonl"

# 4. Counts that are safe to share (refused if the privacy lint finds anything).
uv run jef-bench triage cloudflare-report --pack "$PRIVATE_JEF_DATA/cloudflare/pack" \
  --out out/cloudflare-public-aggregate.json
```

## What each step keeps

| Data | Kept as |
|---|---|
| Client IP, Ray ID | keyed HMAC pseudonym (`JEF_PSEUDONYM_KEY`); the same IP is the same actor within one key, unrelated under another |
| Path | route template: numeric ids, UUIDs, long hex and token-like segments become `{id}` |
| Query string | parameter **names** only; every value is dropped |
| User agent | a coarse family (`scanner:sqlmap`, `monitor:uptime`, `browser:chrome` …) |
| Time | the minute |
| Action, source, rule | `weak.jsonl` only, which no scorer or exporter reads; shown in the state only in the marked `--with-vendor-signals` variant |
| Token, zone id | nowhere. The zone is recorded by the alias you choose |

Every case is `label_status: unlabelled`. Evaluation counts such cases and never scores them, and `candidate-export` refuses them. Labels come only from the reviewer queue. Approval evidence (`--context`, one JSON object per line with `kind` = `change_ticket`, `approved_scanner`, `maintenance_window` or `uptime_monitor`, plus `ref`, `start`, `end`, `match.actor_ip`, `match.route_prefix`, `note`) is attached to the incidents it covers. Each such case also gets a twin without it, so a reviewer labels "approved scan" and "the same scan, no approval" separately.

Families join every incident of one actor and every incident on one route family in the same ISO week, so a scanner or campaign never spans splits. For a temporal holdout, lock with `jef-bench triage lock --pack … --out … --holdout-after 2026-11-01`.

## Never

- Commit anything from `$PRIVATE_JEF_DATA`, upload it as a GitHub artifact, or send it to an external model or API (Jev, Laya, an LLM) without a separate data-processing and egress approval.
- Run the real ingest in CI. `triage-cloudflare-contract.yml` uses a recorded fake and synthetic fixtures only.
- Generate attack traffic against a production zone. Test traffic belongs in a separate lab zone, and cases from it are `test_generated`, not unseen real attacks.

## 人工標註 Labelling a batch

One reviewer gives Tier B (`single-human-reviewed`). Tier A needs a second, independent reviewer and a third person to adjudicate (`review import --answers a.jsonl --answers b.jsonl --adjudication c.jsonl`). Everything below stays in the private directory; worksheets and answers of private cases are written owner-only and refused inside a git checkout.

```bash
P="$PRIVATE_JEF_DATA/cloudflare"; R="$PRIVATE_JEF_DATA/review"
# 1. pick 100 incidents, one per family (vendor-signal variants are never picked: same incident)
uv run jef-bench triage review sample --pack "$P/site-a-pack" --batch-id cf-pilot-1 \
  --stratum web=100 --max-per-family 1 --out "$R/cf-pilot-1.batch.json"
# 2. a blind worksheet for you: evidence only, no WAF action, no score
uv run jef-bench triage review export --pack "$P/site-a-pack" --batch "$R/cf-pilot-1.batch.json" \
  --reviewer <handle> --out "$R/cf-pilot-1.<handle>.worksheet.jsonl"
# 3. label in the terminal (zh-TW). Quit with q any time; the same command resumes.
uv run jef-bench triage review label --worksheet "$R/cf-pilot-1.<handle>.worksheet.jsonl"
# 4. answers -> a reviewed private pack, then the report
uv run jef-bench triage review import --pack "$P/site-a-pack" --batch "$R/cf-pilot-1.batch.json" \
  --answers "$R/cf-pilot-1-<handle>.answers.jsonl" --out "$P/site-a-reviewed-1"
uv run jef-bench triage review check --pack "$P/site-a-reviewed-1" \
  --answers "$R/cf-pilot-1-<handle>.answers.jsonl"
```

Keys in the labeller: numbers choose, `u` = 不確定 unknown, `b` = back one question, `?` = the labelbook rule for this question, `r` = the raw state, `s` = skip this case for now, `q` = save and quit. For owner and severity, several numbers separated by commas list every acceptable answer, the first being the primary. `--case <id>` re-labels one case. Answer from what the screen shows; the rules are in [`../LABELBOOK.md`](../LABELBOOK.md).
