# Plan: Behaviour + host-context fusion, leakage audit, a second NDR source (2026-10-18)

## Assessment of the review

The review is accurate.
- The behaviour-label gain is real, but it came with a change of label semantics and case set. It is not the same test set scoring higher.
- Host windows detect infected hosts far better than they assign the four-way verdict. Prioritisation, not adjudication or auto-close, is their plausible first use.
- Every gate above G1 is still Pending.

Two corrections:
- **Window overlap is not a leakage path in the current design.** A flow belongs to exactly one window (by start time), and every window of a scenario is in the same fold. Host reuse across scenarios is a real path, though. CTU-13 uses the same verified-normal hosts in every scenario, so a model can learn those hosts' habits even with addresses masked. The audit must measure it.
- **Behaviour tags are labels, not features.** "Keep flow behaviour in the host window" must mean behaviour-defined *gold*, plus the conversation structure the tags describe. It must never mean putting the tag text in front of the model, which would leak the label.

## What can be done in code and GitHub (this plan)

| # | Work | Output |
|---|---|---|
| 1 | **Leakage audit** for host windows and conversations: hosts shared across scenarios per class, the share of each held-out scenario whose host also appears in training, coarse near-duplicate states across scenarios (same label: memorisation risk; different label: ambiguity), and how `c2` and `botnet` windows were decided (C&C flows per window). | `socbench/leakage.py`, `jef-bench audit`, audit JSON in the study directory |
| 2 | **Fusion cases** (`--unit host-fusion`): host windows whose gold follows behaviour semantics (`c2` for a C&C flow; `botnet` for spam, ads, attempts or floods; an infected host showing neither is `unknown`; verified-normal host `benign`), with conversation summaries as evidence and a per-minute time series. Provenance stays in the gold note. | `ctu13_hosts.py`, label origin `host-behaviour-label` (Tier C) |
| 3 | **Pre-registered fusion study** on the same 13 folds, with criteria for macro-F1, miss rate, coverage and gate safety fixed in `run.py` before any run, plus host-compromise prioritisation as a secondary measure (infected-window recall in the top 10% of each scenario). | `docs/validation/research/ndr-host-fusion-v1/` |
| 4 | **Second NDR source: IoT-23** (Stratosphere, 2018–2019, IoT devices, a different network, flow-level behaviour labels). Subset rule fixed now: every capture whose `conn.log.labeled` was at most 200 MB on 2026-10-18 (2 benign honeypot, 8 malware captures, about 180 MB). Adapter, pins, and a pre-registered transfer test (train on CTU-13 fusion, test on IoT-23 fusion). | `socbench/iot23.py`, `sources.yaml` entry, transfer results |
| 5 | Ledger, README positioning (host windows framed as prioritisation), manual todo list. | docs |

## What stays manual

These stay in [`manually_action_need_todos.md`](../../../manually_action_need_todos.md):
- the 63-case pilot (P0, needs one engineer);
- ESXi S14 and fault injection (needs the lab);
- the Laya and official Jev comparison (needs a checkpoint and access);
- Tier A labels.

None of them waits on this plan.
