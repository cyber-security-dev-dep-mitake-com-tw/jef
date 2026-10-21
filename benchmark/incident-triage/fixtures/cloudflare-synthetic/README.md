# Synthetic Cloudflare security events (fixture)

**SYNTHETIC — NOT TELEMETRY.** Every row was written by hand for the contract tests of `jef_train.socbench.cloudflare_*`. The addresses are RFC 5737 documentation ranges, the ASNs are RFC 5398 documentation ASNs, the Ray IDs are counters and the query strings are invented. Nothing here came from a real zone.

- `events.jsonl`: rows shaped like `firewallEventsAdaptive` results: a scanner whose first request hits three rules (one Ray ID, three events), a login brute force under rate limiting, an uptime monitor being challenged, an SQL-injection probe with secret-looking query values, an approved DAST run, and sampled log-only events (`sampleInterval: 10`).
- `context.jsonl`: approval evidence a reviewer could cite: a maintenance window for the DAST run, and the uptime monitor's registration.

These rows are for testing the pipeline only. They are not labels and not examples of real attacks.
